# Copyright 2025 Eli Lilly and Company
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Impact model."""

import logging
from collections.abc import Callable
from inspect import signature
from os import cpu_count
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from typing import Self, cast
from warnings import warn

import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import (
    Array,
    default_backend,
    device_get,
    device_put,
    jit,
    local_device_count,
    make_mesh,
    random,
)
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike
from numpyro.handlers import do, seed, substitute, trace
from numpyro.infer import MCMC, SVI
from numpyro.infer.svi import SVIRunResult, SVIState
from tqdm.auto import tqdm
from xarray import open_zarr
from zarr import open_group

from aimz.model._core import BaseModel
from aimz.sampling._forward import _sample_forward
from aimz.utils._format import _dict_to_datatree, _make_attrs
from aimz.utils._kwargs import _group_kwargs
from aimz.utils._output import (
    _create_output_subdir,
    _shutdown_writer_threads,
    _start_writer_threads,
    _writer,
)
from aimz.utils._validation import (
    _check_is_fitted,
    _validate_group,
    _validate_kernel_body,
    _validate_X_y_to_jax,
)
from aimz.utils.data import ArrayLoader
from aimz.utils.data._input_setup import _setup_inputs
from aimz.utils.data._sharding import (
    _create_sharded_log_likelihood,
    _create_sharded_sampler,
)

logger = logging.getLogger(__name__)


class ImpactModel(BaseModel):
    """A class for impact modeling."""

    def __init__(
        self,
        kernel: Callable,
        rng_key: ArrayLike,
        inference: SVI | MCMC,
        *,
        param_input: str = "X",
        param_output: str = "y",
    ) -> None:
        """Initialize an ImpactModel instance.

        Args:
            kernel (Callable): A probabilistic model with Pyro primitives.
            rng_key (ArrayLike): A pseudo-random number generator key.
            inference (SVI | MCMC): An inference method supported by NumPyro, such as an
                instance of :external:py:class:`numpyro.infer.svi.SVI` or
                :external:py:class:`numpyro.infer.mcmc.MCMC`.
            param_input (str, optional): The name of the parameter in the ``kernel`` for
                the main input data.
            param_output (str, optional): The name of the parameter in the ``kernel``
                for the output data.

        Warning:
            The ``rng_key`` parameter should be provided as a **typed key array**
            created with :external:py:func:`jax.random.key`, rather than a legacy
            ``uint32`` key created with :external:py:func:`jax.random.PRNGKey`.
        """
        super().__init__(kernel, param_input, param_output)
        if isinstance(rng_key, Array) and rng_key.dtype == jnp.uint32:
            msg = "Legacy `uint32` PRNGKey detected; converting to a typed key array."
            warn(msg, category=UserWarning, stacklevel=2)
            rng_key = random.wrap_key_data(rng_key)
        self._rng_key = rng_key
        if not isinstance(inference, (SVI, MCMC)):
            msg = (
                f"Unsupported inference object: `{type(inference).__name__}`. "
                "Expected `SVI` or `MCMC` from `numpyro.infer`."
            )
            raise TypeError(msg)
        self.inference = inference
        self._vi_state = None
        self._posterior: dict[str, Array] | None = None
        self._init_runtime_attrs()

    def _init_runtime_attrs(self) -> None:
        """Initialize runtime attributes."""
        self._fn_vi_update: Callable | None = None
        self._fn_sample_prior_predictive: Callable | None = None
        self._fn_sample_posterior_predictive: Callable | None = None
        self._fn_log_likelihood: Callable | None = None
        self._mesh: Mesh | None
        self._device: NamedSharding | None
        num_devices = local_device_count()
        if num_devices > 1:
            self._mesh = make_mesh((num_devices,), ("obs",))
            self._device = NamedSharding(self._mesh, PartitionSpec("obs"))
        else:
            self._mesh = None
            self._device = None
        logger.info(
            "Backend: %s, Devices: %d",
            default_backend(),
            num_devices,
        )

    def __del__(self) -> None:
        """Clean up the temporary directory when the instance is deleted."""
        self.cleanup()
        # Call the parent's __del__ method only if it exists and is callable
        super_del = getattr(super(), "__del__", None)
        if callable(super_del):
            super_del()

    def __getstate__(self) -> dict:
        """Return the state of the object excluding runtime attributes.

        Returns:
            The state of the object, excluding runtime attributes.
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if not (
                k.startswith("_fn")
                or k in {"_device", "_mesh", "_num_devices", "temp_dir"}
            )
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore the state and reinitialize runtime attributes.

        Args:
            state (dict): The state to restore, excluding the runtime attributes.
        """
        self.__dict__.update(state)
        self._init_runtime_attrs()

    @property
    def rng_key(self) -> ArrayLike:
        """Pseudo-random number generator key."""
        return self._rng_key

    @property
    def vi_result(self) -> SVIRunResult:
        """Variational inference result.

        :setter: This sets :external:py:data:`numpyro.infer.svi.SVIRunResult` and marks
            the model as fitted. It does not perform posterior sampling — use
            :py:meth:`~aimz.ImpactModel.sample` separately to obtain samples.
        """
        return self._vi_result

    @vi_result.setter
    def vi_result(self, vi_result: SVIRunResult) -> None:
        """Set the variational inference result manually.

        Args:
            vi_result (SVIRunResult): The result from a prior variational inference run.
                It must be a NamedTuple or similar object with the following fields:
                - params (dict): Learned parameters from inference.
                - state (SVIState): Internal SVI state object.
                - losses (ArrayLike): Loss values recorded during optimization.
        """
        if np.any(np.isnan(vi_result.losses)):
            msg = "Loss contains NaN or Inf, indicating numerical instability."
            warn(msg, category=RuntimeWarning, stacklevel=2)

        self._is_fitted = True

        self._vi_result = vi_result

    @property
    def posterior(self) -> dict[str, Array] | None:
        """Posterior samples by variable name."""
        return self._posterior

    def sample_prior_predictive_on_batch(
        self,
        X: ArrayLike,
        *,
        num_samples: int = 1000,
        rng_key: ArrayLike | None = None,
        return_sites: tuple[str] | None = None,
        **kwargs: object,
    ) -> xr.DataTree:
        """Draw samples from the prior predictive distribution.

        Args:
            X (ArrayLike): Input data with shape ``(n_samples_X, n_features)``.
            num_samples (int, optional): The number of samples to draw.
            rng_key (ArrayLike | None, optional): A pseudo-random number generator key.
                By default, an internal key is used and split as needed.
            return_sites (tuple[str] | None, optional): Names of variables (sites) to
                return. If ``None``, samples all latent, observed, and deterministic
                sites.
            **kwargs (object): Additional arguments passed to the model. All array-like
                values are expected to be JAX arrays.

        Returns:
            Prior predictive samples.

        Raises:
            TypeError: If ``self.param_output`` is passed as an argument.
        """
        X = cast("Array", _validate_X_y_to_jax(X))

        # Validate the provided parameters against the kernel's signature
        args_bound = (
            signature(self.kernel).bind(**{self.param_input: X, **kwargs}).arguments
        )
        if self.param_output in args_bound:
            msg = f"Specifying {self.param_output!r} is not allowed."
            raise TypeError(msg)

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        prior_predictive_samples = device_get(
            _sample_forward(
                self.kernel,
                rng_key=rng_key,
                num_samples=num_samples,
                return_sites=return_sites,
                samples=None,
                model_kwargs=args_bound,
            ),
        )
        out = xr.DataTree(name="root")
        out["prior_predictive"] = _dict_to_datatree(prior_predictive_samples)

        return out

    def sample_prior_predictive(
        self,
        X: ArrayLike,
        *,
        num_samples: int = 1000,
        rng_key: ArrayLike | None = None,
        return_sites: tuple[str] | None = None,
        batch_size: int | None = None,
        output_dir: str | Path | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> xr.DataTree:
        """Draw samples from the prior predictive distribution.

        Results are written to disk in the Zarr format, with computing and file writing
        decoupled and executed concurrently.

        Args:
            X (ArrayLike): Input data with shape ``(n_samples_X, n_features)``.
            num_samples (int, optional): The number of samples to draw.
            rng_key (ArrayLike | None, optional): A pseudo-random number generator key.
                By default, an internal key is used and split as needed.
            return_sites (tuple[str] | None, optional): Names of variables (sites) to
                return. If ``None``, samples ``self.param_output`` and deterministic
                sites.
            batch_size (int | None, optional): The size of batches for data loading
                during prior predictive sampling. By default, it is set to the total
                number of samples (``n_samples_X``). This value also determines the
                chunk size for storing the prior predictive samples.
            output_dir (str | Path | None, optional): The directory where the outputs
                will be saved. If the specified directory does not exist, it will be
                created automatically. If ``None``, a default temporary directory will
                be created. A timestamped subdirectory will be generated within this
                directory to store the outputs. Outputs are saved in the Zarr format.
            progress (bool, optional): Whether to display a progress bar.
            **kwargs (object): Additional arguments passed to the model. All array-like
                values are expected to be JAX arrays.

        Returns:
            Prior predictive samples.

        Raises:
            TypeError: If ``self.param_output`` is passed as an argument.

        See Also:
            :py:meth:`~aimz.ImpactModel.cleanup` to remove the temporary directory if
            created.
        """
        # Validate the provided parameters against the kernel's signature
        args_bound = (
            signature(self.kernel).bind(**{self.param_input: X, **kwargs}).arguments
        )
        if self.param_output in args_bound:
            msg = f"Specifying {self.param_output!r} is not allowed."
            raise TypeError(msg)

        # Prior sampling
        dataloader, _ = _setup_inputs(
            X=X,
            y=None,
            rng_key=self.rng_key,
            batch_size=self._device.num_devices if self._device is not None else 1,
            shuffle=False,
            device=self._device,
            **kwargs,
        )
        batch, _ = next(iter(dataloader))
        model_trace = trace(seed(self.kernel, rng_seed=self.rng_key)).get_trace(**batch)
        return_sites = return_sites or tuple(
            k
            for k, site in model_trace.items()
            if site["type"] == "deterministic" or k == self.param_output
        )
        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)
        rng_key, rng_subkey = random.split(rng_key)
        prior_samples = _sample_forward(
            self.kernel,
            rng_key=rng_subkey,
            num_samples=num_samples,
            return_sites=None,
            samples=None,
            model_kwargs=batch,
        )
        prior_samples = {
            k: v for k, v in prior_samples.items() if k not in return_sites
        }

        kwargs_array, kwargs_extra = _group_kwargs(kwargs)
        if self._fn_sample_prior_predictive is None:
            self._fn_sample_prior_predictive = _create_sharded_sampler(
                self._mesh,
                len(kwargs_array),
                len(kwargs_extra),
            )

        if output_dir is None:
            if not hasattr(self, "temp_dir"):
                self.temp_dir = TemporaryDirectory()
                logger.info(
                    "Temporary directory created at: %s",
                    self.temp_dir.name,
                )
            output_dir = self.temp_dir.name
            logger.info(
                "No output directory provided. Using the model's temporary directory "
                "for storing output.",
            )
        output_subdir = _create_output_subdir(output_dir)

        dataloader, _ = _setup_inputs(
            X=X,
            y=None,
            rng_key=self.rng_key,
            batch_size=batch_size,
            shuffle=False,
            device=self._device,
            **kwargs,
        )

        pbar = tqdm(
            desc=(f"Prior predictive sampling [{', '.join(return_sites)}]"),
            total=len(dataloader),
            disable=not progress,
        )

        self._sample_and_write(
            num_samples,
            rng_key=rng_key,
            return_sites=return_sites,
            output_dir=output_subdir,
            kernel=self.kernel,
            sampler=self._fn_sample_prior_predictive,
            samples=prior_samples,
            dataloader=dataloader,
            pbar=pbar,
            **kwargs,
        )

        ds = open_zarr(output_subdir, consolidated=False).expand_dims(
            dim="chain",
            axis=0,
        )
        ds = ds.assign_coords(
            {k: np.arange(ds.sizes[k]) for k in ds.sizes},
        ).assign_attrs(_make_attrs())
        out = xr.DataTree(name="root")
        out["prior_predictive"] = xr.DataTree(ds)

        return out

    def sample(
        self,
        num_samples: int = 1000,
        rng_key: ArrayLike | None = None,
        return_sites: tuple[str] | None = None,
        **kwargs: object,
    ) -> xr.DataTree:
        """Draw posterior samples from a fitted model.

        Args:
            num_samples (int, optional): The number of posterior samples to draw.
            rng_key (ArrayLike | None, optional): A pseudo-random number generator key.
                By default, an internal key is used and split as needed. Has no effect
                if the inference method is MCMC, where the ``post_warmup_state``
                property will be used to continue sampling.
            return_sites (tuple[str] | None, optional): Names of variables (sites) to
                return. If ``None``, samples all latent sites. Has no effect if the
                inference method is MCMC.
            **kwargs (object): Additional arguments passed to the model. All array-like
                values are expected to be JAX arrays. Only relevant when the inference
                method is MCMC.

        Returns:
            Posterior samples.

        Raises:
            TypeError: If ``self.param_output`` is not passed as an argument if the
                inference method is MCMC.
        """
        _check_is_fitted(self)

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        if isinstance(self.inference, MCMC):
            # Validate the provided parameters against the kernel's signature
            args_bound = signature(self.kernel).bind(**kwargs).arguments
            if self.param_output not in args_bound:
                msg = f"{self.param_output!r} must be provided in `.sample()`."
                raise TypeError(msg)
            self.inference.post_warmup_state = self.inference.last_state
            self.inference.num_samples = num_samples
            self.inference.run(self.inference.post_warmup_state.rng_key, **args_bound)
            posterior_samples = device_get(self.inference.get_samples())
        else:
            posterior_samples = _sample_forward(
                substitute(self.inference.guide, data=self.vi_result.params),
                rng_key=rng_key,
                num_samples=num_samples,
                return_sites=return_sites,
                samples=None,
                model_kwargs=None,
            )

        out = xr.DataTree(name="root")
        out["posterior"] = _dict_to_datatree(posterior_samples)

        return out

    def sample_posterior_predictive_on_batch(
        self,
        X: ArrayLike,
        *,
        intervention: dict | None = None,
        rng_key: ArrayLike | None = None,
        in_sample: bool = True,
        return_sites: tuple[str] | None = None,
        **kwargs: object,
    ) -> xr.DataTree:
        """Draw samples from the posterior predictive distribution.

        This method is a convenience alias for
        :py:meth:`~aimz.ImpactModel.predict_on_batch`.

        Args:
            X (ArrayLike): Input data with shape ``(n_samples_X, n_features)``.
            intervention (dict | None, optional): A dictionary mapping sample sites to
                their corresponding intervention values. Interventions enable
                counterfactual analysis by modifying the specified sample sites during
                prediction (posterior predictive sampling).
            rng_key (ArrayLike | None, optional): A pseudo-random number generator key.
                By default, an internal key is used and split as needed.
            in_sample (bool, optional): Specifies the group where posterior predictive
                samples are stored in the returned output. If ``True``, samples are
                stored in the ``posterior_predictive`` group, indicating they were
                generated based on data used during model fitting. If ``False``, samples
                are stored in the ``predictions`` group, indicating they were generated
                based on out-of-sample data.
            return_sites (tuple[str] | None, optional): Names of variables (sites) to
                return. If ``None``, samples ``self.param_output`` and deterministic
                sites.
            **kwargs (object): Additional arguments passed to the model. All array-like
                values are expected to be JAX arrays.

        Returns:
            Posterior predictive samples. Posterior samples are included if available.

        Raises:
            TypeError: If ``self.param_output`` is passed as an argument.

        See Also:
            :py:meth:`~aimz.ImpactModel.predict_on_batch`.
        """
        return self.predict_on_batch(
            X,
            intervention=intervention,
            rng_key=rng_key,
            in_sample=in_sample,
            return_sites=return_sites,
            **kwargs,
        )

    def sample_posterior_predictive(
        self,
        X: ArrayLike | ArrayLoader,
        *,
        intervention: dict | None = None,
        rng_key: ArrayLike | None = None,
        in_sample: bool = True,
        return_sites: tuple[str] | None = None,
        batch_size: int | None = None,
        output_dir: str | Path | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> xr.DataTree:
        """Draw samples from the posterior predictive distribution.

        This method is a convenience alias for :py:meth:`~aimz.ImpactModel.predict`.

        Args:
            X (ArrayLike | ArrayLoader): Input data, either an array-like of shape
                ``(n_samples, n_features)`` or a data loader that holds all array-like
                objects and handles batching internally; if a data loader is passed,
                ``batch_size`` is ignored.
            intervention (dict | None, optional): A dictionary mapping sample sites to
                their corresponding intervention values. Interventions enable
                counterfactual analysis by modifying the specified sample sites during
                prediction (posterior predictive sampling).
            rng_key (ArrayLike | None, optional): A pseudo-random number generator key.
                By default, an internal key is used and split as needed.
            in_sample (bool, optional): Specifies the group where posterior predictive
                samples are stored in the returned output. If ``True``, samples are
                stored in the ``posterior_predictive`` group, indicating they were
                generated based on data used during model fitting. If ``False``, samples
                are stored in the ``predictions`` group, indicating they were generated
                based on out-of-sample data.
            return_sites (tuple[str] | None, optional): Names of variables (sites) to
                return. If ``None``, samples ``self.param_output`` and deterministic
                sites.
            batch_size (int | None, optional): The size of batches for data loading
                during posterior predictive sampling. By default, it is set to the
                total number of samples (``n_samples_X``). This value also determines
                the chunk size for storing the posterior predictive samples.
            output_dir (str | Path | None, optional): The directory where the outputs
                will be saved. If the specified directory does not exist, it will be
                created automatically. If ``None``, a default temporary directory will
                be created. A timestamped subdirectory will be generated within this
                directory to store the outputs. Outputs are saved in the Zarr format.
            progress (bool, optional): Whether to display a progress bar.
            **kwargs (object): Additional arguments passed to the model. All array-like
                values are expected to be JAX arrays.

        Returns:
            Posterior predictive samples. Posterior samples are included if available.

        Raises:
            TypeError: If ``self.param_output`` is passed as an argument.

        See Also:
            :py:meth:`~aimz.ImpactModel.predict()`.
        """
        return self.predict(
            X,
            intervention=intervention,
            rng_key=rng_key,
            in_sample=in_sample,
            return_sites=return_sites,
            batch_size=batch_size,
            output_dir=output_dir,
            progress=progress,
            **kwargs,
        )

    def train_on_batch(
        self,
        X: ArrayLike,
        y: ArrayLike,
        rng_key: ArrayLike | None = None,
        **kwargs: object,
    ) -> tuple[SVIState, Array]:
        """Run a single VI step on the given batch of data.

        Args:
            X (ArrayLike): Input data with shape ``(n_samples_X, n_features)``.
            y (ArrayLike): Output data with shape ``(n_samples_Y,)``.
            rng_key (ArrayLike | None, optional): A pseudo-random number generator key.
                By default, an internal key is used and split as needed. The key is only
                used for initialization if the internal SVI state is not yet set.
            **kwargs (object): Additional arguments passed to the model. All array-like
                values are expected to be JAX arrays.

        Returns:
            - Updated SVI state after the training step.

            - Loss value as a scalar array.

        Note:
            This method updates the internal SVI state on every call, so it is not
            necessary to capture the returned state externally unless explicitly needed.
            However, the returned loss value can be used for monitoring or logging.
        """
        batch = {self.param_input: X, self.param_output: y, **kwargs}

        if self._vi_state is None:
            # Validate the provided parameters against the kernel's signature
            model_trace = trace(seed(self.kernel, rng_seed=self.rng_key)).get_trace(
                **signature(self.kernel).bind(**batch).arguments,
            )
            # Validate the kernel body for output sample site and naming conflicts
            _validate_kernel_body(
                self.kernel,
                self.param_output,
                model_trace,
            )
            self._return_sites = (
                *(
                    k
                    for k, site in model_trace.items()
                    if site["type"] == "deterministic"
                ),
                self.param_output,
            )
            if rng_key is None:
                self._rng_key, rng_key = random.split(self._rng_key)
            self._vi_state = self.inference.init(rng_key, **batch)
        if self._fn_vi_update is None:
            _, kwargs_extra = _group_kwargs(kwargs)
            self._fn_vi_update = jit(
                self.inference.update,
                static_argnames=tuple(kwargs_extra._fields),
            )

        self._vi_state, loss = self._fn_vi_update(self._vi_state, **batch)

        return self._vi_state, loss

    def fit_on_batch(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        num_steps: int = 10000,
        num_samples: int = 1000,
        rng_key: ArrayLike | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> Self:
        """Fit the impact model to the provided batch of data.

        This method behaves differently depending on the inference method specified at
        initialization of the ImpactModel instance:

        - SVI
            Runs variational inference on the provided batch by invoking the
            :external:py:meth:`~numpyro.infer.svi.SVI.run` method of the
            :external:py:class:`~numpyro.infer.svi.SVI` instance from NumPyro to
            estimate the posterior distribution, then draws samples from it.

        - MCMC
            Runs posterior sampling by invoking the
            :external:py:meth:`~numpyro.infer.mcmc.MCMC.run` method of the
            :external:py:class:`~numpyro.infer.mcmc.MCMC` instance from NumPyro.

        Args:
            X (ArrayLike): Input data with shape ``(n_samples_X, n_features)``.
            y (ArrayLike): Output data with shape ``(n_samples_Y,)``.
            num_steps (int, optional): Number of steps for variational inference
                optimization. Has no effect if the inference method is MCMC.
            num_samples (int | None, optional): The number of posterior samples to draw.
                Has no effect if the inference method is MCMC.
            rng_key (ArrayLike | None, optional): A pseudo-random number generator key.
                By default, an internal key is used and split as needed.
            progress (bool, optional): Whether to display a progress bar. Has no effect
                if the inference method is MCMC.
            **kwargs (object): Additional arguments passed to the model. All array-like
                values are expected to be JAX arrays.

        Returns:
            The fitted model instance, enabling method chaining.

        Note:
            This method continues training from the existing SVI state if available. To
            start training from scratch, create a new model instance.
        """
        X, y = _validate_X_y_to_jax(X, y)

        # Validate the provided parameters against the kernel's signature
        args_bound = (
            signature(self.kernel)
            .bind(**{self.param_input: X, self.param_output: y, **kwargs})
            .arguments
        )
        model_trace = trace(seed(self.kernel, rng_seed=self.rng_key)).get_trace(
            **args_bound,
        )
        # Validate the kernel body for output sample site and naming conflicts
        _validate_kernel_body(
            self.kernel,
            self.param_output,
            model_trace,
        )
        self._return_sites = (
            *(k for k, site in model_trace.items() if site["type"] == "deterministic"),
            self.param_output,
        )

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)
        rng_key, rng_subkey = random.split(rng_key)
        if isinstance(self.inference, SVI):
            self._num_samples = num_samples
            logger.info("Performing variational inference optimization...")
            self.vi_result = self.inference.run(
                rng_subkey,
                num_steps=num_steps,
                progress_bar=progress,
                init_state=self._vi_state,
                **args_bound,
            )
            self._vi_state = self.vi_result.state
            if np.any(np.isnan(self.vi_result.losses)):
                warn(
                    "Loss contains NaN or Inf, indicating numerical instability.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            logger.info("Posterior sampling...")
            rng_key, rng_subkey = random.split(rng_key)
            self._posterior = _sample_forward(
                substitute(self.inference.guide, data=self.vi_result.params),
                rng_key=rng_subkey,
                num_samples=self._num_samples,
                return_sites=None,
                samples=None,
                model_kwargs=None,
            )
        elif isinstance(self.inference, MCMC):
            logger.info("Posterior sampling...")
            self.inference.run(rng_subkey, **args_bound)
            self._posterior = device_get(self.inference.get_samples())
            self._num_samples = (
                next(iter(self._posterior.values())).shape[0] if self._posterior else 0
            )

        self._is_fitted = True
        self._dt_posterior = (
            _dict_to_datatree(self._posterior) if self._posterior else None
        )

        return self

    def fit(
        self,
        X: ArrayLike | ArrayLoader,
        y: ArrayLike | None = None,
        *,
        num_samples: int = 1000,
        rng_key: ArrayLike | None = None,
        progress: bool = True,
        batch_size: int | None = None,
        epochs: int = 1,
        shuffle: bool = True,
        **kwargs: object,
    ) -> Self:
        """Fit the impact model to the provided data using epoch-based training.

        This method implements an epoch-based training loop, where the data is iterated
        over in minibatches for a specified number of epochs. Variational inference is
        performed by repeatedly updating the model parameters on each minibatch, and
        then posterior samples are drawn from the fitted model.

        Args:
            X (ArrayLike | ArrayLoader): Input data, either an array-like of shape
                ``(n_samples, n_features)`` or a data loader that holds all array-like
                objects and handles batching internally; if a data loader is passed,
                ``batch_size`` is ignored.
            y (ArrayLike | None): Output data with shape ``(n_samples_Y,)``. Must be
                ``None`` if ``X`` is a data loader.
            num_samples (int | None, optional): The number of posterior samples to draw.
            rng_key (ArrayLike | None, optional): A pseudo-random number generator key.
                By default, an internal key is used and split as needed.
            progress (bool, optional): Whether to display a progress bar.
            batch_size (int | None, optional): The number of data points processed at
                each step of variational inference. If ``None``, the entire dataset is
                used as a single batch in each epoch.
            epochs (int, optional): The number of epochs for variational inference
                optimization.
            shuffle (bool, optional): Whether to shuffle the data at each epoch.
            **kwargs (object): Additional arguments passed to the model. All array-like
                values are expected to be JAX arrays.

        Returns:
            The fitted model instance, enabling method chaining.

        Raises:
            TypeError: If the inference method is MCMC.

        Note:
            This method continues training from the existing SVI state if available.
            To start training from scratch, create a new model instance. It does not
            check whether the model or guide is written to support subsampling semantics
            (e.g., using NumPyro's :external:py:func:`~numpyro.primitives.subsample` or
            similar constructs).
        """
        if isinstance(self.inference, MCMC):
            msg = (
                "`.fit()` is not supported for MCMC inference. Use `.fit_on_batch()` "
                "instead."
            )
            raise TypeError(msg)

        self._num_samples = num_samples

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        rng_key, rng_subkey = random.split(rng_key)
        dataloader, kwargs_extra = _setup_inputs(
            X=X,
            y=y,
            rng_key=rng_subkey,
            batch_size=batch_size,
            shuffle=shuffle,
            device=None,
            **kwargs,
        )

        logger.info("Performing variational inference optimization...")
        losses: list[float] = []
        rng_key, rng_subkey = random.split(rng_key)
        for epoch in range(epochs):
            losses_epoch: list[float] = []
            pbar = tqdm(
                dataloader,
                total=len(dataloader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                disable=not progress,
            )
            for batch, _ in pbar:
                self._vi_state, loss = self.train_on_batch(
                    **batch,
                    **kwargs_extra._asdict(),
                    rng_key=rng_subkey,
                )
                loss_batch = device_get(loss)
                losses_epoch.append(loss_batch)
                pbar.set_postfix({"loss": f"{float(loss_batch):.4f}"})
            losses_epoch_arr = jnp.stack(losses_epoch)
            losses.extend(losses_epoch_arr)
            tqdm.write(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Average loss: {float(jnp.mean(losses_epoch_arr)):.4f}",
            )
        self.vi_result = SVIRunResult(
            params=self.inference.get_params(self._vi_state),
            state=self._vi_state,
            losses=jnp.asarray(losses),
        )
        if np.any(np.isnan(self.vi_result.losses)):
            msg = "Loss contains NaN or Inf, indicating numerical instability."
            warn(msg, category=RuntimeWarning, stacklevel=2)

        self._is_fitted = True

        logger.info("Posterior sampling...")
        rng_key, rng_subkey = random.split(rng_key)
        self._posterior = _sample_forward(
            substitute(self.inference.guide, data=self.vi_result.params),
            rng_key=rng_subkey,
            num_samples=self._num_samples,
            return_sites=None,
            samples=None,
            model_kwargs=None,
        )
        self._dt_posterior = (
            _dict_to_datatree(self._posterior) if self._posterior else None
        )

        return self

    def is_fitted(self) -> bool:
        """Check fitted status.

        Returns:
            `True` if the model is fitted, `False` otherwise.

        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def set_posterior_sample(
        self,
        posterior_sample: dict[str, Array],
        return_sites: tuple[str] | None = None,
    ) -> Self:
        """Set posterior samples for the model.

        This method sets externally obtained posterior samples on the model instance,
        enabling downstream analysis without requiring a call to
        :py:meth:`~aimz.ImpactModel.fit` or :py:meth:`~aimz.ImpactModel.fit_on_batch`.

        It is primarily intended for workflows where posterior sampling is performed
        manually—for example, using NumPyro's
        :external:py:class:`~numpyro.infer.svi.SVI` (or
        :external:py:class:`~numpyro.infer.mcmc.MCMC`) with the
        :external:py:class:`~numpyro.infer.util.Predictive` API—and the resulting
        posterior samples are injected into the model for further use.

        Internally, ``batch_ndims`` is set to ``1`` by default to correctly handle the
        batch dimensions of the posterior samples. For more information, refer to the
        `NumPyro documentation <https://num.pyro.ai/en/stable/utilities.html#predictive>`__.

        Args:
            posterior_sample (dict[str, Array]): Posterior samples to set for the model.
            return_sites (tuple[str] | None, optional): Names of variable (sites) to
                return in :py:meth:`~aimz.ImpactModel.predict`. By default, it is set to
                ``self.param_output``.

        Returns:
            The model instance, treated as fitted with posterior samples set, enabling
            method chaining.

        Raises:
            ValueError: If the batch shapes in ``posterior_sample`` are inconsistent.
        """
        batch_ndims = 1
        batch_shapes = {
            sample.shape[:batch_ndims] for sample in posterior_sample.values()
        }
        if len(batch_shapes) > 1:
            msg = f"Inconsistent batch shapes found in posterior_sample: {batch_shapes}"
            raise ValueError(msg)
        (self._num_samples,) = batch_shapes.pop()
        self._posterior = posterior_sample
        self._return_sites = return_sites or (self.param_output,)
        self._is_fitted = True
        self._dt_posterior = _dict_to_datatree(self._posterior)

        return self

    def predict_on_batch(
        self,
        X: ArrayLike,
        *,
        intervention: dict | None = None,
        rng_key: ArrayLike | None = None,
        in_sample: bool = True,
        return_sites: tuple[str] | None = None,
        **kwargs: object,
    ) -> xr.DataTree:
        """Predict the output based on the fitted model.

        This method returns predictions for a single batch of input data and is better
        suited for:

            1) Models incompatible with :py:meth:`~aimz.ImpactModel.predict` due to
            their posterior sample shapes.

            2) Scenarios where writing results to to files (e.g., disk, cloud storage)
            is not desired.

            3) Smaller datasets, as this method may be slower due to limited
            parallelism.

        Args:
            X (ArrayLike): Input data with shape ``(n_samples_X, n_features)``.
            intervention (dict | None, optional): A dictionary mapping sample sites to
                their corresponding intervention values. Interventions enable
                counterfactual analysis by modifying the specified sample sites during
                prediction (posterior predictive sampling).
            rng_key (ArrayLike | None, optional): A pseudo-random number generator key.
                By default, an internal key is used and split as needed.
            in_sample (bool, optional): Specifies the group where posterior predictive
                samples are stored in the returned output. If ``True``, samples are
                stored in the ``posterior_predictive`` group, indicating they were
                generated based on data used during model fitting. If ``False``, samples
                are stored in the ``predictions`` group, indicating they were generated
                based on out-of-sample data.
            return_sites (tuple[str] | None, optional): Names of variables (sites) to
                return. If ``None``, samples ``self.param_output`` and deterministic
                sites.
            **kwargs (object): Additional arguments passed to the model. All array-like
                values are expected to be JAX arrays.

        Returns:
            Posterior predictive samples. Posterior samples are included if available.

        Raises:
            TypeError: If ``self.param_output`` is passed as an argument.
        """
        _check_is_fitted(self)

        X = cast("Array", _validate_X_y_to_jax(X))

        # Validate the provided parameters against the kernel's signature
        args_bound = (
            signature(self.kernel).bind(**{self.param_input: X, **kwargs}).arguments
        )
        if self.param_output in args_bound:
            msg = f"Specifying {self.param_output!r} is not allowed."
            raise TypeError(msg)

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        if intervention is None:
            kernel = self.kernel
        else:
            rng_key, rng_subkey = random.split(rng_key)
            kernel = seed(do(self.kernel, data=intervention), rng_seed=rng_subkey)

        out = xr.DataTree(name="root")
        out["posterior"] = self._dt_posterior
        out["posterior_predictive" if in_sample else "predictions"] = _dict_to_datatree(
            _sample_forward(
                kernel,
                rng_key=rng_key,
                num_samples=self._num_samples,
                return_sites=return_sites or self._return_sites,
                samples=self._posterior,
                model_kwargs=args_bound,
            ),
        )

        return out

    def predict(
        self,
        X: ArrayLike | ArrayLoader,
        *,
        intervention: dict | None = None,
        rng_key: ArrayLike | None = None,
        in_sample: bool = True,
        return_sites: tuple[str] | None = None,
        batch_size: int | None = None,
        output_dir: str | Path | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> xr.DataTree:
        """Predict the output based on the fitted model.

        This method performs posterior predictive sampling to generate model-based
        predictions. It is optimized for batch processing of large input data and is not
        recommended for use in loops that process only a few inputs at a time. Results
        are written to disk in the Zarr format, with sampling and file writing decoupled
        and executed concurrently.

        Args:
            X (ArrayLike | ArrayLoader): Input data, either an array-like of shape
                ``(n_samples, n_features)`` or a data loader that holds all array-like
                objects and handles batching internally; if a data loader is passed,
                ``batch_size`` is ignored.
            intervention (dict | None, optional): A dictionary mapping sample sites to
                their corresponding intervention values. Interventions enable
                counterfactual analysis by modifying the specified sample sites during
                prediction (posterior predictive sampling).
            rng_key (ArrayLike | None, optional): A pseudo-random number generator key.
                By default, an internal key is used and split as needed.
            in_sample (bool, optional): Specifies the group where posterior predictive
                samples are stored in the returned output. If ``True``, samples are
                stored in the ``posterior_predictive`` group, indicating they were
                generated based on data used during model fitting. If ``False``, samples
                are stored in the ``predictions`` group, indicating they were generated
                based on out-of-sample data.
            return_sites (tuple[str] | None, optional): Names of variables (sites) to
                return. If ``None``, samples ``self.param_output`` and deterministic
                sites.
            batch_size (int | None, optional): The size of batches for data loading
                during posterior predictive sampling. By default, it is set to the
                total number of samples (``n_samples_X``). This value also determines
                the chunk size for storing the posterior predictive samples.
            output_dir (str | Path | None, optional): The directory where the outputs
                will be saved. If the specified directory does not exist, it will be
                created automatically. If ``None``, a default temporary directory will
                be created. A timestamped subdirectory will be generated within this
                directory to store the outputs. Outputs are saved in the Zarr format.
            progress (bool, optional): Whether to display a progress bar.
            **kwargs (object): Additional arguments passed to the model. All array-like
                values are expected to be JAX arrays.

        Returns:
            Posterior predictive samples. Posterior samples are included if available.

        Raises:
            TypeError: If ``self.param_output`` is passed as an argument.

        See Also:
            :py:meth:`~aimz.ImpactModel.cleanup` to remove the temporary directory if
            created.
        """
        _check_is_fitted(self)

        if isinstance(X, ArrayLike):
            ndim_posterior_sample = 2
            if self._posterior and any(
                v.ndim == ndim_posterior_sample and v.shape[1] == len(X)
                for v in self._posterior.values()
            ):
                msg = (
                    "One or more posterior sample shapes are not compatible with "
                    "`.predict()` under sharded parallelism; falling back to "
                    "`.predict_on_batch()`."
                )
                warn(msg, category=UserWarning, stacklevel=2)

                return self.predict_on_batch(
                    cast("ArrayLike", X),
                    intervention=intervention,
                    rng_key=rng_key,
                    in_sample=in_sample,
                    return_sites=return_sites or self._return_sites,
                    **kwargs,
                )
            # Validate the provided parameters against the kernel's signature
            args_bound = (
                signature(self.kernel).bind(**{self.param_input: X, **kwargs}).arguments
            )
            if self.param_output in args_bound:
                msg = f"Specifying {self.param_output!r} is not allowed."
                raise TypeError(msg)

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        if intervention is None:
            kernel = self.kernel
        else:
            rng_key, rng_subkey = random.split(rng_key)
            kernel = seed(do(self.kernel, data=intervention), rng_seed=rng_subkey)

        return_sites = return_sites or self._return_sites

        kwargs_array, kwargs_extra = _group_kwargs(kwargs)
        if self._fn_sample_posterior_predictive is None:
            self._fn_sample_posterior_predictive = _create_sharded_sampler(
                self._mesh,
                len(kwargs_array),
                len(kwargs_extra),
            )

        if output_dir is None:
            if not hasattr(self, "temp_dir"):
                self.temp_dir = TemporaryDirectory()
                logger.info(
                    "Temporary directory created at: %s",
                    self.temp_dir.name,
                )
            output_dir = self.temp_dir.name
            logger.info(
                "No output directory provided. Using the model's temporary directory "
                "for storing output.",
            )
        output_subdir = _create_output_subdir(output_dir)

        dataloader, _ = _setup_inputs(
            X=X,
            y=None,
            rng_key=self.rng_key,
            batch_size=batch_size,
            shuffle=False,
            device=self._device,
            **kwargs,
        )

        pbar = tqdm(
            desc=(f"Posterior predictive sampling [{', '.join(return_sites)}]"),
            total=len(dataloader),
            disable=not progress,
        )

        self._sample_and_write(
            num_samples=self._num_samples,
            rng_key=rng_key,
            return_sites=return_sites,
            output_dir=output_subdir,
            kernel=kernel,
            sampler=self._fn_sample_posterior_predictive,
            samples=cast("dict[str, Array]", self._posterior),
            dataloader=dataloader,
            pbar=pbar,
            **kwargs,
        )

        ds = open_zarr(output_subdir, consolidated=False).expand_dims(
            dim="chain",
            axis=0,
        )
        ds = ds.assign_coords(
            {k: np.arange(ds.sizes[k]) for k in ds.sizes},
        ).assign_attrs(_make_attrs())
        out = xr.DataTree(name="root")
        out["posterior"] = self._dt_posterior
        out["posterior_predictive" if in_sample else "predictions"] = xr.DataTree(ds)

        return out

    def estimate_effect(
        self,
        output_baseline: xr.DataTree | None = None,
        output_intervention: xr.DataTree | None = None,
        args_baseline: dict | None = None,
        args_intervention: dict | None = None,
    ) -> xr.DataTree:
        """Estimate the effect of an intervention.

        Args:
            output_baseline (xr.DataTree | None, optional): Precomputed output for the
                baseline scenario.
            output_intervention (xr.DataTree | None, optional): Precomputed output for
                the intervention scenario.
            args_baseline (dict | None, optional): Input arguments for the baseline
                scenario. Passed to the :py:meth:`~aimz.ImpactModel.predict` to compute
                predictions if ``output_baseline`` is not provided. Ignored if
                ``output_baseline`` is already given.
            args_intervention (dict | None, optional): Input arguments for the
                intervention scenario. Passed to the
                :py:meth:`~aimz.ImpactModel.predict` to compute predictions if
                ``output_intervention`` is not provided. Ignored if
                ``output_intervention`` is already given.

        Returns:
            The estimated impact of an intervention.

        Raises:
            ValueError: If neither ``output_baseline`` nor ``args_baseline`` is
                provided, or if neither ``output_intervention`` nor
                ``args_intervention`` is provided.

        See Also:
            :py:meth:`~aimz.ImpactModel.cleanup` to remove the temporary directory if
            created.
        """
        _check_is_fitted(self)

        if output_baseline:
            dt_baseline = output_baseline
        elif args_baseline:
            dt_baseline = self.predict(**args_baseline)
        else:
            msg = "Either `output_baseline` or `args_baseline` must be provided."
            raise ValueError(msg)

        if output_intervention:
            dt_intervention = output_intervention
        elif args_intervention:
            dt_intervention = self.predict(**args_intervention)
        else:
            msg = (
                "Either `output_intervention` or `args_intervention` must be provided."
            )
            raise ValueError(msg)

        group = _validate_group(dt_baseline, dt_intervention)

        out = xr.DataTree(name="root")
        out[group] = dt_intervention[group] - dt_baseline[group]

        return out

    def log_likelihood(
        self,
        X: ArrayLike | ArrayLoader,
        y: ArrayLike | None = None,
        *,
        batch_size: int | None = None,
        output_dir: str | Path | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> xr.DataTree:
        """Compute the log likelihood of the data under the given model.

        Results are written to disk in the Zarr format, with computing and file writing
        decoupled and executed concurrently.

        Args:
            X (ArrayLike | ArrayLoader): Input data, either an array-like of shape
                ``(n_samples, n_features)`` or a data loader that holds all array-like
                objects and handles batching internally; if a data loader is passed,
                ``batch_size`` is ignored.
            y (ArrayLike | None): Output data with shape ``(n_samples_Y,)``. Must be
                ``None`` if ``X`` is a data loader.
            batch_size (int | None, optional): The size of batches for data loading
                during posterior predictive sampling. By default, it is set to the total
                number of samples (``n_samples_X``). This value also determines the
                chunk size for storing the log-likelihood values.
            output_dir (str | Path | None, optional): The directory where the outputs
                will be saved. If the specified directory does not exist, it will be
                created automatically. If ``None``, a default temporary directory will
                be created. A timestamped subdirectory will be generated within this
                directory to store the outputs. Outputs are saved in the Zarr format.
            progress (bool, optional): Whether to display a progress bar.
            **kwargs (object): Additional arguments passed to the model. All array-like
                values are expected to be JAX arrays.

        Returns:
            Log-likelihood values. Posterior samples are included if available.

        See Also:
            :py:meth:`~aimz.ImpactModel.cleanup` to remove the temporary directory if
            created.
        """
        _check_is_fitted(self)

        kwargs_array, kwargs_extra = _group_kwargs(kwargs)
        if self._fn_log_likelihood is None:
            self._fn_log_likelihood = _create_sharded_log_likelihood(
                self._mesh,
                len(kwargs_array),
                len(kwargs_extra),
            )

        if output_dir is None:
            if not hasattr(self, "temp_dir"):
                self.temp_dir = TemporaryDirectory()
                logger.info(
                    "Temporary directory created at: %s",
                    self.temp_dir.name,
                )
            output_dir = self.temp_dir.name
            logger.info(
                "No output directory provided. Using the model's temporary directory "
                "for storing output.",
            )
        output_subdir = _create_output_subdir(output_dir)

        dataloader, _ = _setup_inputs(
            X=X,
            y=y,
            rng_key=self.rng_key,
            batch_size=batch_size,
            shuffle=False,
            device=self._device,
            **kwargs,
        )

        site = self.param_output
        pbar = tqdm(
            desc=(f"Computing log-likelihood of {site}..."),
            total=len(dataloader),
            disable=not progress,
        )

        zarr_group = open_group(output_subdir, mode="w")
        zarr_arr = {}
        threads, queues, error_queue = _start_writer_threads(
            (site,),
            group_path=output_subdir,
            writer=_writer,
            queue_size=min(cpu_count() or 1, 4),
        )
        try:
            for batch, n_pad in dataloader:
                kwargs_batch = [
                    v
                    for k, v in batch.items()
                    if k not in (self.param_input, self.param_output)
                ]
                arr = device_get(
                    self._fn_log_likelihood(
                        # Although computing the log-likelihood is deterministic, the
                        # model still needs to be seeded in order to trace its graph.
                        seed(self.kernel, rng_seed=self.rng_key),
                        self._posterior,
                        self.param_input,
                        site,
                        kwargs_array._fields + kwargs_extra._fields,
                        batch[self.param_input],
                        batch[self.param_output],
                        *(*kwargs_batch, *kwargs_extra),
                    ),
                )
                if site not in zarr_arr:
                    zarr_arr[site] = zarr_group.create_array(
                        name=site,
                        shape=(self._num_samples, 0, *arr.shape[2:]),
                        dtype="float32" if arr.dtype == "bfloat16" else arr.dtype,
                        chunks=(
                            self._num_samples,
                            dataloader.batch_size,
                            *arr.shape[2:],
                        ),
                        dimension_names=(
                            "draw",
                            *tuple(f"{site}_dim_{i}" for i in range(arr.ndim - 1)),
                        ),
                    )
                queues[site].put(arr[:, : -n_pad or None])
                if not error_queue.empty():
                    _, exc, tb = error_queue.get()
                    raise exc.with_traceback(tb)
                pbar.update()
            pbar.set_description("Computation complete, writing in progress...")
            _shutdown_writer_threads(threads, queues=queues)
        except:
            _shutdown_writer_threads(threads, queues=queues)
            logger.exception(
                "Exception encountered. Cleaning up output directory: %s",
                output_subdir,
            )
            rmtree(output_subdir, ignore_errors=True)
            raise
        finally:
            pbar.close()

        ds = open_zarr(output_subdir, consolidated=False).expand_dims(
            dim="chain",
            axis=0,
        )
        ds = ds.assign_coords(
            {k: np.arange(ds.sizes[k]) for k in ds.sizes},
        ).assign_attrs(_make_attrs())
        out = xr.DataTree(name="root")
        out["posterior"] = self._dt_posterior
        out["log_likelihood"] = xr.DataTree(ds)

        return out

    def cleanup(self) -> None:
        """Clean up the temporary directory created for storing outputs.

        If the temporary directory was never created or has already been cleaned up,
        this method does nothing. It does not delete any explicitly specified output
        directory. While the temporary directory is typically removed automatically
        during garbage collection, this behavior is not guaranteed. Therefore, calling
        this method explicitly is recommended to ensure timely resource release.
        """
        if hasattr(self, "temp_dir"):
            logger.info("Temporary directory cleaned up at: %s", self.temp_dir.name)
            self.temp_dir.cleanup()
            del self.temp_dir

    def _sample_and_write(
        self,
        num_samples: int,
        rng_key: ArrayLike,
        return_sites: tuple[str],
        output_dir: Path,
        kernel: Callable,
        sampler: Callable,
        samples: dict[str, Array],
        dataloader: ArrayLoader,
        pbar: tqdm,
        **kwargs: object,
    ) -> None:
        """Draw samples using a predictive function and write them concurrently to disk.

        This function iterates over batches from the provided data loader, calls the
        specified sampling function (``sampler``) to generate predictions conditioned on
        the provided ``samples``, and writes the resulting arrays to a Zarr group.

        Args:
            num_samples (int): Number of samples to draw.
            rng_key (ArrayLike): Pseudo-random number generator key.
            return_sites (tuple[str]): Names of variables (sites) to return.
            output_dir (Path): Directory where outputs will be saved.
            kernel (Callable): Probabilistic model with Pyro primitives.
            sampler (Callable): Function performing predictive sampling; must accept
                the same signature as this function.
            samples (dict[str, Array]): Arrays to condition predictions on.
            dataloader (ArrayLoader): Iterator over batches of input data. Each batch
                must be a tuple containing a dictionary of arrays and a padding value.
            pbar (tqdm): Progress bar instance to display sampling progress.
            **kwargs (object): Additional arguments passed to the model. All array-like
                values are expected to be JAX arrays.

        Raises:
            Exception: Any exception raised during sampling or writing is logged, the
                output directory is cleaned up, and the exception is re-raised.
        """
        kwargs_array, kwargs_extra = _group_kwargs(kwargs)

        rng_key, *subkeys = random.split(rng_key, num=len(dataloader) + 1)
        if self._device and self._mesh:
            subkeys = device_put(
                subkeys,
                NamedSharding(self._mesh, PartitionSpec()),
            )

        zarr_group = open_group(output_dir, mode="w")
        zarr_arr = {}
        threads, queues, error_queue = _start_writer_threads(
            return_sites,
            group_path=output_dir,
            writer=_writer,
            queue_size=min(cpu_count() or 1, 4),
        )
        try:
            for (batch, n_pad), subkey in zip(dataloader, subkeys, strict=True):
                kwargs_batch = [
                    v
                    for k, v in batch.items()
                    if k not in (self.param_input, self.param_output)
                ]
                dict_arr = device_get(
                    sampler(
                        kernel,
                        num_samples,
                        subkey,
                        return_sites,
                        samples,
                        self.param_input,
                        kwargs_array._fields + kwargs_extra._fields,
                        batch[self.param_input],
                        *(*kwargs_batch, *kwargs_extra),
                    ),
                )
                for site, arr in dict_arr.items():
                    if site not in zarr_arr:
                        zarr_arr[site] = zarr_group.create_array(
                            name=site,
                            shape=(num_samples, 0, *arr.shape[2:]),
                            dtype="float32" if arr.dtype == "bfloat16" else arr.dtype,
                            chunks=(
                                num_samples,
                                dataloader.batch_size,
                                *arr.shape[2:],
                            ),
                            dimension_names=(
                                "draw",
                                *tuple(f"{site}_dim_{i}" for i in range(arr.ndim - 1)),
                            ),
                        )
                    queues[site].put(arr[:, : -n_pad or None])
                if not error_queue.empty():
                    _, exc, tb = error_queue.get()
                    raise exc.with_traceback(tb)
                pbar.update()
            pbar.set_description("Sampling complete, writing in progress...")
            _shutdown_writer_threads(threads, queues)
        except:
            _shutdown_writer_threads(threads, queues)
            logger.exception(
                "Exception encountered. Cleaning up output directory: %s",
                output_dir,
            )
            rmtree(output_dir, ignore_errors=True)
            raise
        finally:
            pbar.close()
