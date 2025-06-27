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
from sys import modules
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, NamedTuple, Self
from warnings import warn

import arviz as az
import jax.numpy as jnp
import numpy as np
import xarray as xr
from arviz.data.base import make_attrs
from jax import (
    default_backend,
    device_put,
    local_device_count,
    make_mesh,
    random,
)
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax_dataloader import ArrayDataset
from numpyro.handlers import do, seed, substitute, trace
from numpyro.infer.svi import SVIRunResult
from sklearn.utils.validation import check_array, check_X_y
from tqdm.auto import tqdm
from xarray import open_zarr
from zarr import open_group

from aimz.data import (
    ArrayLoader,
)
from aimz.data._sharding import _create_sharded_log_likelihood, _create_sharded_sampler
from aimz.model._core import BaseModel
from aimz.sampling._forward import _sample_forward
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
)

if TYPE_CHECKING:
    import jax
    from numpyro.infer import SVI

logger = logging.getLogger(__name__)


class ImpactModel(BaseModel):
    """A class for impact modeling."""

    def __init__(
        self,
        kernel: Callable,
        rng_key: "jax.Array",
        vi: "SVI",
        *,
        param_input: str = "X",
        param_output: str = "y",
    ) -> None:
        """Initialize the ImpactModel.

        Args:
            kernel (Callable): A probabilistic model with Pyro primitives.
            rng_key (jax.Array): A pseudo-random number generator key.
            vi (SVI): A variational inference object supported by NumPyro, such as an
                instance of `numpyro.infer.svi.SVI` or any other object that implements
                variational inference.
            param_input (str, optional): The name of the parameter in the `kernel` for
                the main input data. Defaults to `"X"`.
            param_output (str, optional): The name of the parameter in the `kernel` for
                the output data. Defaults to `"y"`.
        """
        super().__init__(kernel, param_input, param_output)
        self.rng_key = rng_key
        self.vi = vi
        self._init_runtime_attrs()

    def _init_runtime_attrs(self) -> None:
        """Initialize runtime attributes."""
        self._fn_sample_posterior_predictive: Callable | None = None
        self._fn_log_likelihood: Callable | None = None
        self._num_devices: int = local_device_count()
        self._mesh: Mesh | None
        self._device: NamedSharding | None
        if self._num_devices > 1:
            self._mesh = make_mesh((self._num_devices,), ("obs",))
            self._device = NamedSharding(self._mesh, PartitionSpec("obs"))
        else:
            self._mesh = None
            self._device = None
        logger.info(
            "Backend: %s, Devices: %d",
            default_backend(),
            self._num_devices,
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
            dict: The state of the object, excluding runtime attributes.
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
    def guide(self) -> Callable:
        """Get the current variational guide function.

        Returns:
            Callable: The guide function used for variational inference.
        """
        return self._guide

    @guide.setter
    def guide(self, guide_fn: Callable) -> None:
        """Set the guide function used for variational inference.

        This allows manual assignment of a custom or pre-defined guide function. It is
        primarily intended for use with `.sample()` without calling `.fit()`. Note that
        calling `.fit()` will overwrite any existing guide.

        Args:
            guide_fn (Callable): A guide function, typically created using NumPyro's
                `AutoGuide` classes or a user-defined guide following the NumPyro
                convention.
        """
        self._guide = guide_fn

    @property
    def vi_result(self) -> SVIRunResult:
        """Get the current variational inference result.

        Returns:
            SVIRunResult: The stored result from variational inference.
        """
        return self._vi_result

    @vi_result.setter
    def vi_result(self, vi_result: SVIRunResult) -> None:
        """Set the variational inference result manually.

        This sets the result from a variational inference run and marks the model as
        fitted. It does not perform posterior sampling — use `.sample()` separately to
        obtain samples.

        Args:
            vi_result (SVIRunResult): The result from a prior variational inference run.
                It must be a NamedTuple or similar object with the following fields:
                - params (dict): Learned parameters from inference.
                - state (numpyro.infer.svi.SVIState): Internal SVI state object.
                - losses (list[float]): List of loss values recorded during
                    optimization.
        """
        if np.any(np.isnan(vi_result.losses)):
            msg = "Loss contains NaN or Inf, indicating numerical instability."
            warn(msg, category=RuntimeWarning, stacklevel=2)

        self._is_fitted = True

        self._vi_result = vi_result

    def sample_prior_predictive(
        self,
        X: "jax.Array",
        *,
        num_samples: int = 1000,
        rng_key: "jax.Array | None" = None,
        return_sites: tuple[str] | None = None,
        **kwargs: object,
    ) -> dict["str", "jax.Array"]:
        """Draw samples from the prior predictive distribution.

        Args:
            X (jax.Array): Input data with shape `(n_samples_X, n_features)`.
            num_samples (int, optional): The number of samples to draw. Defaults to
                `1000`.
            rng_key (jax.Array | None, optional): A pseudo-random number generator key.
                Defaults to `None`, then an internal key is used and split as needed.
            return_sites (tuple[str] | None, optional): Names of variables (sites) to
                return. If `None`, samples all latent, observed, and deterministic
                sites. Defaults to `None`.
            **kwargs (object): Additional arguments passed to the model, except for
                `X` and `y`. All array-like objects in `**kwargs` are expected to be
                JAX arrays.

        Returns:
            dict[str, jax.Array]: The prior predictive samples.

        Raises:
            TypeError: If `self.param_output` is passed as an argument.
        """
        if rng_key is None:
            self.rng_key, rng_key = random.split(self.rng_key)

        # Validate the provided parameters against the kernel's signature
        args_bound = (
            signature(self.kernel).bind(**{self.param_input: X, **kwargs}).arguments
        )
        if self.param_output in args_bound:
            sub = self.param_output
            msg = f"{sub!r} is not allowed in `.sample_prior_predictive()`."
            raise TypeError(msg)

        return _sample_forward(
            self.kernel,
            rng_key=rng_key,
            num_samples=num_samples,
            return_sites=return_sites,
            posterior_samples=None,
            model_kwargs=args_bound,
        )

    def sample(
        self,
        num_samples: int = 1000,
        rng_key: "jax.Array | None" = None,
        return_sites: tuple[str] | None = None,
    ) -> dict["str", "jax.Array"]:
        """Draw posterior samples from a fitted model.

        Args:
            num_samples (int | None, optional): The number of posterior samples.
                Defaults to `1000`.
            rng_key (jax.Array | None, optional): A pseudo-random number generator key.
                Defaults to `None`, then an internal key is used and split as needed.
            return_sites (tuple[str] | None, optional): Names of variables (sites) to
                return. If `None`, samples all latent sites. Defaults to `None`.

        Returns:
            dict[str, jax.Array]: The posterior samples.

        """
        _check_is_fitted(self)

        if rng_key is None:
            self.rng_key, rng_key = random.split(self.rng_key)

        return _sample_forward(
            substitute(self.guide, data=self.vi_result.params),
            rng_key=rng_key,
            num_samples=num_samples,
            return_sites=return_sites,
            posterior_samples=None,
            model_kwargs=None,
        )

    def sample_posterior_predictive(
        self,
        X: "jax.Array",
        *,
        rng_key: "jax.Array | None" = None,
        return_sites: tuple[str] | None = None,
        intervention: dict | None = None,
        **kwargs: object,
    ) -> dict["str", "jax.Array"]:
        """Draw samples from the posterior predictive distribution.

        Args:
            X (jax.Array): Input data with shape `(n_samples_X, n_features)`.
            rng_key (jax.Array | None, optional): A pseudo-random number generator key.
                Defaults to `None`, then an internal key is used and split as needed.
            return_sites (tuple[str] | None, optional): Names of variables (sites) to
                return. If `None`, samples all latent, observed, and deterministic
                sites. Defaults to `None`.
            intervention (dict | None, optional): A dictionary mapping sample sites to
                their corresponding intervention values. Interventions enable
                counterfactual analysis by modifying the specified sample sites during
                prediction (posterior predictive sampling). Defaults to `None`.
            **kwargs (object): Additional arguments passed to the model, except for
                `X` and `y`. All array-like objects in `**kwargs` are expected to be
                JAX arrays.

        Returns:
            dict[str, jax.Array]: The posterior predictive samples.

        Raises:
            TypeError: If `self.param_output` is passed as an argument.
        """
        _check_is_fitted(self)

        if rng_key is None:
            self.rng_key, rng_key = random.split(self.rng_key)

        X = jnp.asarray(check_array(X))

        # Validate the provided parameters against the kernel's signature
        args_bound = (
            signature(self.kernel).bind(**{self.param_input: X, **kwargs}).arguments
        )
        if self.param_output in args_bound:
            sub = self.param_output
            msg = f"{sub!r} is not allowed in `.sample_prior_predictive()`."
            raise TypeError(msg)

        if intervention is None:
            kernel = self.kernel
        else:
            rng_key, rng_subkey = random.split(rng_key)
            kernel = seed(do(self.kernel, data=intervention), rng_seed=rng_subkey)

        return _sample_forward(
            kernel,
            rng_key=rng_key,
            num_samples=self.num_samples,
            return_sites=return_sites or self._return_sites,
            posterior_samples=self.posterior_sample_,
            model_kwargs=args_bound,
        )

    def fit(
        self,
        X: "jax.Array",
        y: "jax.Array",
        *,
        num_steps: int = 10000,
        num_samples: int = 1000,
        rng_key: "jax.Array | None" = None,
        progress: bool = True,
        **kwargs: object,
    ) -> Self:
        """Fit the impact model to the provided data.

        This method runs variational inference to estimate the posterior distribution
        and then draws samples from it.

        Args:
            X (jax.Array): Input data with shape `(n_samples_X, n_features)`.
            y (jax.Array): Output data with shape `(n_samples_Y,)`.
            num_steps (int, optional): Number of steps for variational inference
                optimization. Defaults to `10000`.
            num_samples (int | None, optional): The number of posterior samples.
                Defaults to `1000`.
            rng_key (jax.Array | None, optional): A pseudo-random number generator key.
                Defaults to `None`, then an internal key is used and split as needed.
            progress (bool, optional): Whether to display a progress bar. Defaults to
                `True`.
            **kwargs (object): Additional arguments passed to the model, except for
                `X` and `y`. All array-like objects in `**kwargs` are expected to be
                JAX arrays.

        Returns:
            ImpactModel: The fitted model instance, enabling method chaining.
        """
        if rng_key is None:
            self.rng_key, rng_key = random.split(self.rng_key)

        X, y = map(jnp.asarray, check_X_y(X, y, y_numeric=True))

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

        self.num_samples = num_samples

        logger.info("Performing variational inference optimization...")
        rng_key, rng_subkey = random.split(rng_key)
        self.guide = self.vi.guide
        self.vi_result = self.vi.run(
            rng_subkey,
            num_steps=num_steps,
            progress_bar=progress,
            **args_bound,
        )
        if np.any(np.isnan(self.vi_result.losses)):
            msg = "Loss contains NaN or Inf, indicating numerical instability."
            warn(msg, category=RuntimeWarning, stacklevel=2)

        self._is_fitted = True

        logger.info("Posterior sampling...")
        rng_key, rng_subkey = random.split(rng_key)
        self.posterior_sample_ = self.sample(self.num_samples, rng_key=rng_subkey)

        return self

    def is_fitted(self) -> bool:
        """Check fitted status.

        Returns:
            bool: `True` if the model is fitted, `False` otherwise.

        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def set_posterior_sample(
        self,
        posterior_sample: dict[str, "jax.Array"],
        return_sites: tuple[str] | None = None,
    ) -> Self:
        """Set posterior samples for the model.

        This method sets externally obtained posterior samples on the model instance,
        enabling downstream analysis without requiring a call to `.fit()`.

        It is primarily intended for workflows where inference is performed manually—
        for example, using NumPyro's `SVI` with the `Predictive` API—and the resulting
        posterior samples are injected into the model for further use.

        Internally, `batch_ndims` is set to `1` by default to correctly handle the batch
        dimensions of the posterior samples. For more information, refer to the
        [NumPyro Predictive documentation]
        (https://num.pyro.ai/en/stable/utilities.html#predictive).

        Args:
            posterior_sample (dict[str, jax.Array]): Posterior samples to set for the
                model.
            return_sites (tuple[str] | None, optional): Names of variable (sites) to
                return in `.predict()`. Defaults to `None` and is set to `param_output`
                if not specified.

        Returns:
            ImpactModel: The model instance, treated as fitted with posterior samples
                set, enabling method chaining.

        Raises:
            ValueError: If the batch shapes in `posterior_sample` are inconsistent
                (i.e., have different shapes).
        """
        self.posterior_sample_ = posterior_sample

        self._return_sites = return_sites or (self.param_output,)

        batch_ndims = 1
        batch_shapes = {
            sample.shape[:batch_ndims] for sample in self.posterior_sample_.values()
        }
        if len(batch_shapes) > 1:
            msg = f"Inconsistent batch shapes found in posterior_sample: {batch_shapes}"
            raise ValueError(msg)

        (self.num_samples,) = batch_shapes.pop()

        self._is_fitted = True

        return self

    def __sample_posterior_predictive(
        self,
        *,
        fn_sample_posterior_predictive: Callable,
        kernel: Callable,
        X: "jax.Array",
        rng_key: "jax.Array",
        group: str,
        batch_size: int,
        output_dir: Path,
        progress: bool,
        kwargs_array: NamedTuple,
        kwargs_extra: NamedTuple,
    ) -> az.InferenceData:
        kwargs_key = kwargs_array._fields + kwargs_extra._fields

        dataloader: ArrayLoader = ArrayLoader(
            ArrayDataset(X, *kwargs_array),
            batch_size=batch_size,
            collate_fn=lambda batch: ArrayLoader.collate_without_output(
                batch,
                device=self._device,
            ),
        )

        pbar = tqdm(
            desc=(f"Posterior predictive sampling [{', '.join(self._return_sites)}]"),
            total=len(dataloader),
            disable=not progress,
        )

        rng_key, *subkeys = random.split(rng_key, num=len(dataloader) + 1)
        if self._device and self._mesh:
            subkeys = device_put(
                subkeys,
                NamedSharding(self._mesh, PartitionSpec()),
            )

        zarr_group = open_group(output_dir, mode="w")
        zarr_arr = {}
        threads, queues, error_queue = _start_writer_threads(
            self._return_sites,
            group_path=output_dir,
            writer=_writer,
            queue_size=min(cpu_count() or 1, 4),
        )
        try:
            for batch, subkey in zip(dataloader, subkeys, strict=True):
                n_pad, x_batch, *kwargs_batch = batch
                dict_arr = fn_sample_posterior_predictive(
                    kernel,
                    self.num_samples,
                    subkey,
                    self._return_sites,
                    self.posterior_sample_,
                    self.param_input,
                    kwargs_key,
                    x_batch,
                    *(*kwargs_batch, *kwargs_extra),
                )
                for site, arr in dict_arr.items():
                    if site not in zarr_arr:
                        zarr_arr[site] = zarr_group.create_array(
                            name=site,
                            shape=(self.num_samples, 0, *arr.shape[2:]),
                            dtype=arr.dtype,
                            chunks=(self.num_samples, batch_size, *arr.shape[2:]),
                            dimension_names=(
                                "draw",
                                *tuple(f"{site}_dim{j}" for j in range(arr.ndim - 1)),
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

        ds = open_zarr(output_dir, consolidated=False).expand_dims(dim="chain", axis=0)
        ds = ds.assign_coords(
            {k: np.arange(ds.sizes[k]) for k in ds.sizes},
        ).assign_attrs(make_attrs(library=modules["aimz"]))

        return az.convert_to_inference_data(ds, group=group)

    def predict(
        self,
        X: "jax.Array",
        *,
        intervention: dict | None = None,
        rng_key: "jax.Array | None" = None,
        in_sample: bool = True,
        batch_size: int | None = None,
        output_dir: str | Path | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> az.InferenceData:
        """Predict the output based on the fitted model.

        This method performs posterior predictive sampling to generate model-based
        predictions. It is optimized for batch processing of large input data and is not
        recommended for use in loops that process only a few inputs at a time. Results
        are written to disk in the Zarr format, with sampling and file writing decoupled
        and executed concurrently.

        Args:
            X (jax.Array): Input data with shape `(n_samples_X, n_features)`.
            intervention (dict | None, optional): A dictionary mapping sample sites to
                their corresponding intervention values. Interventions enable
                counterfactual analysis by modifying the specified sample sites during
                prediction (posterior predictive sampling). Defaults to `None`.
            rng_key (jax.Array | None, optional): A pseudo-random number generator key.
                Defaults to `None`, then an internal key is used and split as needed.
            in_sample (bool, optional): Specifies the group where posterior predictive
                samples are stored in the returned output. If `True`, samples are stored
                in the `posterior_predictive` group, indicating they were generated
                based on data used during model fitting. If `False`, samples are stored
                in the `predictions` group, indicating they were generated based on
                out-of-sample data.
            batch_size (int | None, optional): The size of batches for data loading
                during posterior predictive sampling. Defaults to `None`, which sets the
                batch size to the total number of samples (`n_samples_X`). This value
                also determines the chunk size for storing the posterior predictive
                samples. It should be carefully selected to optimize memory usage and
                computational efficiency, ensuring balanced performance for downstream
                tasks.
            output_dir (str | Path | None, optional): The directory where the outputs
                will be saved. If the specified directory does not exist, it will be
                created automatically. If `None`, a default temporary directory will be
                created. A timestamped subdirectory will be generated within this
                directory to store the outputs. Outputs are saved in the Zarr format.
            progress (bool, optional): Whether to display a progress bar. Defaults to
                `True`.
            **kwargs (object): Additional arguments passed to the model, except for `X`
                and `y`. All array-like objects in `**kwargs` are expected to be JAX
                arrays.

        Returns:
            az.InferenceData: An object containing posterior predictive samples.

        Raises:
            TypeError: If `self.param_output` is passed as an argument.
        """
        _check_is_fitted(self)

        # Check for compatibility with the `.predict()` method.
        #
        # If any array in the posterior has shape (num_samples, num_obs)—i.e.,
        # `ndim == 2` and the second dimension matches the number of observations in
        # `X`—it suggests that the model produces per-observation posterior samples.
        # This makes it incompatible with the current `.predict()` implementation, which
        # uses sharded parallelism. In such cases, fall back to `.predict_on_batch()`
        # and raise a warning.
        ndim_posterior_sample = 2
        if any(
            v.ndim == ndim_posterior_sample and v.shape[1] == len(X)
            for v in self.posterior_sample_.values()
        ):
            msg = (
                "One or more posterior sample shapes are not compatible with "
                "`.predict()` under sharded parallelism; falling back to "
                "`.predict_on_batch()`."
            )
            warn(msg, category=UserWarning, stacklevel=2)
            return self.predict_on_batch(
                X,
                intervention=intervention,
                rng_key=rng_key,
                in_sample=in_sample,
                **kwargs,
            )

        if rng_key is None:
            self.rng_key, rng_key = random.split(self.rng_key)

        X = jnp.asarray(check_array(X))

        # Validate the provided parameters against the kernel's signature
        args_bound = (
            signature(self.kernel).bind(**{self.param_input: X, **kwargs}).arguments
        )
        if self.param_output in args_bound:
            sub = self.param_output
            msg = f"{sub!r} is not allowed in `.predict()`."
            raise TypeError(msg)

        if intervention is None:
            kernel = self.kernel
        else:
            rng_key, rng_subkey = random.split(rng_key)
            kernel = seed(do(self.kernel, data=intervention), rng_seed=rng_subkey)

        kwargs_array, kwargs_extra = _group_kwargs(kwargs)
        if self._fn_sample_posterior_predictive is None:
            self._fn_sample_posterior_predictive = _create_sharded_sampler(
                self._mesh,
                len(kwargs_array),
                len(kwargs_extra),
            )

        if batch_size is None:
            batch_size = len(X)
            msg = (
                f"No `batch_size` specified. Using full dataset size ({batch_size}). "
                "Specify `batch_size` to prevent memory issues, or use "
                "`.predict_on_batch()` directly."
            )
            warn(msg, category=UserWarning, stacklevel=2)
        if batch_size % self._num_devices != 0:
            msg = (
                f"The `batch_size` ({batch_size}) is not divisible by the number of "
                f"devices ({self._num_devices}). Use a multiple of {self._num_devices} "
                "for optimal performance."
            )
            warn(msg, category=UserWarning, stacklevel=2)

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

        out = self.__sample_posterior_predictive(
            fn_sample_posterior_predictive=self._fn_sample_posterior_predictive,
            kernel=kernel,
            X=X,
            rng_key=rng_key,
            group="posterior_predictive" if in_sample else "predictions",
            batch_size=batch_size,
            output_dir=output_subdir,
            progress=progress,
            kwargs_array=kwargs_array,
            kwargs_extra=kwargs_extra,
        )
        out.add_groups(
            {
                "posterior": {
                    k: jnp.expand_dims(v, axis=0)
                    for k, v in self.posterior_sample_.items()
                },
            },
        )
        out["posterior"].attrs.update(make_attrs(library=modules["aimz"]))

        return out

    def predict_on_batch(
        self,
        X: "jax.Array",
        *,
        intervention: dict | None = None,
        rng_key: "jax.Array | None" = None,
        in_sample: bool = True,
        **kwargs: object,
    ) -> az.InferenceData:
        """Predict the output based on the fitted model.

        This method returns predictions for a single batch of input data and is better
        suited for:
            1) Models incompatible with `.predict()` due to their posterior sample
                shapes.
            2) Scenarios where writing results to to files (e.g., disk, cloud storage)
                is not desired.
            3) Smaller datasets, as this method may be slower due to limited
                parallelism.

        Args:
            X (jax.Array): Input data with shape `(n_samples_X, n_features)`.
            intervention (dict | None, optional): A dictionary mapping sample sites to
                their corresponding intervention values. Interventions enable
                counterfactual analysis by modifying the specified sample sites during
                prediction (posterior predictive sampling). Defaults to `None`.
            rng_key (jax.Array | None, optional): A pseudo-random number generator key.
                Defaults to `None`, then an internal key is used and split as needed.
            in_sample (bool, optional): Specifies the group where posterior predictive
                samples are stored in the returned output. If `True`, samples are stored
                in the `posterior_predictive` group, indicating they were generated
                based on data used during model fitting. If `False`, samples are stored
                in the `predictions` group, indicating they were generated based on
                out-of-sample data.
            **kwargs (object): Additional arguments passed to the model, except for `X`
                and `y`. All array-like objects in `**kwargs` are expected to be JAX
                arrays.

        Returns:
            az.InferenceData: An object containing posterior predictive samples.

        Raises:
            TypeError: If `self.param_output` is passed as an argument.
        """
        _check_is_fitted(self)

        if rng_key is None:
            self.rng_key, rng_key = random.split(self.rng_key)

        X = jnp.asarray(check_array(X))

        # Validate the provided parameters against the kernel's signature
        args_bound = (
            signature(self.kernel).bind(**{self.param_input: X, **kwargs}).arguments
        )
        if self.param_output in args_bound:
            sub = self.param_output
            msg = f"{sub!r} is not allowed in `.predict_on_batch()`."
            raise TypeError(msg)

        if intervention is None:
            kernel = self.kernel
        else:
            rng_key, rng_subkey = random.split(rng_key)
            kernel = seed(do(self.kernel, data=intervention), rng_seed=rng_subkey)

        posterior_predictive_sample = xr.Dataset(
            {
                site: xr.DataArray(
                    np.expand_dims(arr, axis=0),
                    coords={
                        "chain": np.arange(1),
                        "draw": np.arange(self.num_samples),
                        **{
                            f"{site}_dim{i}": np.arange(arr.shape[i + 1])
                            for i in range(arr.ndim - 1)
                        },
                    },
                    dims=(
                        # Adding the 'chain' dimension to support MCMC-style data
                        # structures.
                        "chain",
                        "draw",
                        # arr has shape (draw, dim0, dim1, ...), so arr.ndim includes
                        # 'draw' and we subtract 1
                        *[f"{site}_dim{i}" for i in range(arr.ndim - 1)],
                    ),
                    name=site,
                )
                for site, arr in _sample_forward(
                    kernel,
                    rng_key=rng_key,
                    num_samples=self.num_samples,
                    return_sites=self._return_sites,
                    posterior_samples=self.posterior_sample_,
                    model_kwargs=args_bound,
                ).items()
            },
        ).assign_attrs(make_attrs(library=modules["aimz"]))

        # Reorder the dimensions and add the return sites at the end
        dims_reordered = [
            "chain",
            "draw",
            *sorted(
                str(x)
                for x in list(posterior_predictive_sample.dims)
                if x not in {"chain", "draw"}
            ),
            *self._return_sites,
        ]

        out = az.convert_to_inference_data(
            posterior_predictive_sample[dims_reordered],
            group="posterior_predictive" if in_sample else "predictions",
        )
        out.add_groups(
            {
                "posterior": {
                    k: jnp.expand_dims(v, axis=0)
                    for k, v in self.posterior_sample_.items()
                },
            },
        )
        out["posterior"].attrs.update(make_attrs(library=modules["aimz"]))

        return out

    def estimate_effect(
        self,
        output_baseline: az.InferenceData | None = None,
        output_intervention: az.InferenceData | None = None,
        args_baseline: dict | None = None,
        args_intervention: dict | None = None,
    ) -> az.InferenceData:
        """Estimate the effect of an intervention.

        Args:
            output_baseline (az.InferenceData | None, optional): Precomputed output for
                the baseline scenario.
            output_intervention (az.InferenceData | None, optional): Precomputed output
                for the intervention scenario.
            args_baseline (dict | None, optional): Input arguments for the baseline
                scenario. Passed to the `.predict()` method to compute predictions if
                `output_baseline` is not provided. Ignored if `output_baseline` is
                already given.
            args_intervention (dict | None, optional): Input arguments for the
                intervention scenario. Passed to the `.predict()` method to compute
                predictions if `output_intervention` is not provided. Ignored if
                `output_intervention` is already given.

        Returns:
            az.InferenceData: The estimated impact of an intervention.

        Raises:
            ValueError: If neither `output_baseline` nor `args_baseline` is provided, or
                if neither `output_intervention` nor `args_intervention` is provided.
        """
        _check_is_fitted(self)

        if output_baseline:
            idata_baseline = output_baseline
        elif args_baseline:
            idata_baseline = self.predict(**args_baseline)
        else:
            msg = "Either `output_baseline` or `args_baseline` must be provided."
            raise ValueError(msg)

        if output_intervention:
            idata_intervention = output_intervention
        elif args_intervention:
            idata_intervention = self.predict(**args_intervention)
        else:
            msg = (
                "Either `output_intervention` or `args_intervention` must be provided."
            )
            raise ValueError(msg)

        group = _validate_group(idata_baseline, idata_intervention)

        return az.convert_to_inference_data(
            idata_intervention[group] - idata_baseline[group],
            group=group,
        )

    def log_likelihood(
        self,
        X: "jax.Array",
        y: "jax.Array",
        *,
        batch_size: int | None = None,
        output_dir: str | Path | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> az.InferenceData:
        """Compute the log likelihood of the data under the given model.

        Results are written to disk in the Zarr format, with computing and file writing
        decoupled and executed concurrently.

        Args:
            X (jax.Array): Input data with shape `(n_samples_X, n_features)`.
            y (jax.Array): Output data with shape `(n_samples_Y,)`.
            batch_size (int | None, optional): The size of batches for data loading
                during posterior predictive sampling. Defaults to `None`, which sets the
                batch size to the total number of samples (`n_samples_X`). This value
                also determines the chunk size for storing the log-likelihood values. It
                should be carefully selected to optimize memory usage and computational
                efficiency, ensuring balanced performance for downstream tasks.
            output_dir (str | Path | None, optional): The directory where the outputs
                will be saved. If the specified directory does not exist, it will be
                created automatically. If `None`, a default temporary directory will be
                created. A timestamped subdirectory will be generated within this
                directory to store the outputs. Outputs are saved in the Zarr format.
            progress (bool, optional): Whether to display a progress bar. Defaults to
                `True`.
            **kwargs (object): Additional arguments passed to the model, except for `X`
                and `y`. All array-like objects in `**kwargs` are expected to be JAX
                arrays.

        Returns:
            az.InferenceData: An object containing log-likelihood values.
        """
        _check_is_fitted(self)

        # Validate the provided parameters against the kernel's signature
        signature(self.kernel).bind(**{self.param_input: X, **kwargs})

        X, y = map(jnp.asarray, check_X_y(X, y, y_numeric=True))

        kwargs_array, kwargs_extra = _group_kwargs(kwargs)
        if self._fn_log_likelihood is None:
            self._fn_log_likelihood = _create_sharded_log_likelihood(
                self._mesh,
                len(kwargs_array),
                len(kwargs_extra),
            )

        if batch_size is None:
            batch_size = len(X)
            msg = (
                f"No `batch_size` specified. Using full dataset size ({batch_size}). "
                "Specify `batch_size` to prevent memory issues."
            )
            warn(msg, category=UserWarning, stacklevel=2)
        if batch_size % self._num_devices != 0:
            msg = (
                f"The `batch_size` ({batch_size}) is not divisible by the number of "
                f"devices ({self._num_devices}). Use a multiple of {self._num_devices} "
                "for optimal performance."
            )
            warn(msg, category=UserWarning, stacklevel=2)

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

        dataloader: ArrayLoader = ArrayLoader(
            ArrayDataset(X, y, *kwargs_array),
            batch_size=batch_size or len(X),
            collate_fn=lambda batch: ArrayLoader.collate_with_sharding(
                batch,
                device=self._device,
            ),
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
            for batch in dataloader:
                n_pad, x_batch, y_batch, *kwargs_batch = batch
                arr = self._fn_log_likelihood(
                    # Although computing the log-likelihood is deterministic, the model
                    # still needs to be seeded in order to trace its graph.
                    seed(self.kernel, rng_seed=self.rng_key),
                    self.posterior_sample_,
                    self.param_input,
                    site,
                    kwargs_array._fields + kwargs_extra._fields,
                    x_batch,
                    y_batch,
                    *(*kwargs_batch, *kwargs_extra),
                )
                if site not in zarr_arr:
                    zarr_arr[site] = zarr_group.create_array(
                        name=site,
                        shape=(self.num_samples, 0, *arr.shape[2:]),
                        dtype=arr.dtype,
                        chunks=(self.num_samples, batch_size, *arr.shape[2:]),
                        dimension_names=(
                            "draw",
                            *tuple(f"{site}_dim{j}" for j in range(arr.ndim - 1)),
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
        ).assign_attrs(make_attrs(library=modules["aimz"]))

        return az.convert_to_inference_data(ds, group="log_likelihood")

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
