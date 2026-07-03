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

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from inspect import signature, stack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, Self, cast
from warnings import warn
from weakref import WeakSet

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import xarray as xr
from jax import (
    Array,
    default_backend,
    device_get,
    jit,
    local_device_count,
    make_mesh,
    random,
)
from jax.sharding import AxisType, NamedSharding, PartitionSpec
from jax.typing import ArrayLike
from numpyro.handlers import do, seed, substitute, trace
from numpyro.infer import MCMC, SVI
from numpyro.infer.svi import SVIRunResult, SVIState
from tqdm.auto import tqdm

from aimz.model._core import BaseModel
from aimz.model._streaming import (
    _OutputStreamer,
    _RuntimeContext,
    _WriteRequest,
)
from aimz.model.kernel_spec import KernelSpec
from aimz.sampling._forward import _sample_forward
from aimz.utils._format import _build_datatree, _dict_to_datatree
from aimz.utils._kwargs import _group_kwargs
from aimz.utils._validation import (
    _check_is_fitted,
    _is_arraylike,
    _validate_aligned_inputs,
    _validate_batch_size,
    _validate_group,
    _validate_kernel_body,
    _validate_shard_axis,
    _validate_X_y_to_jax,
)
from aimz.utils.data._input_setup import (
    _resolve_batch_size,
    _setup_inputs,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sized

    from aimz.utils.data import ArrayLoader

logger = logging.getLogger(__name__)


class ImpactModel(BaseModel):
    """Impact modeling interface: fit, sample, predict, and estimate effects."""

    _models = WeakSet()

    def __init__(
        self,
        kernel: Callable,
        rng_key: Array,
        inference: SVI | MCMC,
        *,
        param_input: str = "X",
        param_output: str = "y",
    ) -> None:
        """Initialize an :class:`~aimz.ImpactModel` instance.

        Args:
            kernel: A probabilistic model with `NumPyro`_ primitives.
            rng_key: A pseudo-random number generator key.
            inference: An inference method supported by `NumPyro`_, such as an instance
                of :external:class:`~numpyro.infer.svi.SVI` or
                :external:class:`~numpyro.infer.mcmc.MCMC`.
            param_input: Name of the parameter in the ``kernel`` for the main input
                data.
            param_output: Name of the parameter in the ``kernel`` for the output data.

        Warning:
            The ``rng_key`` parameter should be provided as a **typed key array**
            created with :external:func:`jax.random.key`, rather than a legacy
            ``uint32`` key created with :external:func:`jax.random.PRNGKey`.
        """
        super().__init__(kernel, param_input, param_output)
        self._kernel_spec: KernelSpec | None = None
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
        self._inference = inference
        self._vi_result: SVIRunResult | None = None
        self._vi_state = None
        self._posterior: dict[str, Array] | None = None
        self._init_runtime_attrs()

        # Register this instance
        ImpactModel._models.add(self)

    def _init_runtime_attrs(self) -> None:
        """Initialize runtime attributes."""
        self._fn_vi_update: Callable | None = None
        self._num_devices = local_device_count()
        if self._num_devices > 1:
            mesh = make_mesh(
                (self._num_devices,),
                axis_names=("obs",),
                axis_types=(AxisType.Explicit,),
            )
            partitioned = NamedSharding(mesh, spec=PartitionSpec("obs"))
            replicated = NamedSharding(mesh, spec=PartitionSpec())
        else:
            mesh = partitioned = replicated = None
        # The streaming engine owns the sharded-callable and posterior placement caches
        self._streamer = _OutputStreamer(
            _RuntimeContext(
                self.param_input,
                param_output=self.param_output,
                mesh=mesh,
                num_devices=self._num_devices,
                replicated_sharding=replicated,
                partitioned_sharding=partitioned,
            ),
        )
        self._temp_dir: TemporaryDirectory | None = None
        logger.info(
            "Backend: %s, Devices: %d",
            default_backend(),
            self._num_devices,
        )

    def __str__(self) -> str:
        """Return a summary of the :class:`~aimz.ImpactModel` instance."""
        out = [
            "<ImpactModel>\n",
            f"Kernel: {getattr(self.kernel, '__name__', type(self.kernel).__name__)}",
            f"Inference method: {self.inference.__class__.__name__}",
            f"Input parameter: '{self.param_input}'",
            f"Output parameter: '{self.param_output}'",
            f"Fitted: {getattr(self, '_is_fitted', False)}",
        ]
        outdir = getattr(self, "temp_dir", None)
        if outdir:
            out.append(f"Output directory: {outdir}")

        return "\n".join(out)

    def __repr__(self) -> str:
        """Return a representation of the :class:`~aimz.ImpactModel` instance."""
        out = [
            "<ImpactModel",
            (
                f"kernel_name="
                f"{getattr(self.kernel, '__name__', type(self.kernel).__name__)};"
            ),
            f"rng_key_data={random.key_data(self._rng_key)};",
            f"inference_method={self.inference.__class__.__name__};",
            f"param_input={self.param_input!r};",
            f"param_output={self.param_output!r};",
            f"kernel_spec={self.kernel_spec!r};",
            f"fitted={getattr(self, '_is_fitted', False)};",
            f"temp_dir={getattr(self, 'temp_dir', None)!r}>",
        ]

        return " ".join(out)

    def __del__(self) -> None:
        """Clean up the temporary directory when the instance is garbage-collected."""
        with contextlib.suppress(AttributeError):
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
                or k
                in {
                    "_num_devices",
                    "_temp_dir",
                    "_streamer",
                }
            )
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore the state and reinitialize runtime attributes.

        Args:
            state: The state to restore, excluding the runtime attributes.
        """
        self.__dict__.update(state)
        self._init_runtime_attrs()

    @property
    def inference(self) -> SVI | MCMC:
        """The underlying `NumPyro`_ inference object."""
        return self._inference

    @property
    def kernel(self) -> Callable:
        """A probabilistic model with `NumPyro`_ primitives."""
        return self._kernel

    @property
    def kernel_spec(self) -> KernelSpec | None:
        """The cached :class:`~aimz.model.KernelSpec` or ``None`` if not yet built."""
        return self._kernel_spec

    @property
    def param_input(self) -> str:
        """Parameter name in :attr:`~aimz.ImpactModel.kernel` for the input data."""
        return self._param_input

    @property
    def param_output(self) -> str:
        """Parameter name in :attr:`~aimz.ImpactModel.kernel` for the output data."""
        return self._param_output

    @property
    def posterior(self) -> dict[str, Array] | None:
        """Posterior samples by variable name, or ``None`` if not set.

        Read-only by contract: mutating it in place is unsupported and desynchronizes
        the internal device-placement cache. Use
        :meth:`~aimz.ImpactModel.set_posterior_sample` or refit the model to change it.
        """
        return self._posterior

    @property
    def rng_key(self) -> Array:
        """Pseudo-random number generator key."""
        return self._rng_key

    @property
    def temp_dir(self) -> str | None:
        """Temporary directory path, or ``None`` if not set."""
        return self._temp_dir.name if self._temp_dir else None

    @property
    def vi_result(self) -> SVIRunResult | None:
        """Variational inference result, or ``None`` if not set.

        :setter: This sets :external:data:`~numpyro.infer.svi.SVIRunResult` and marks
            the model as fitted. It does not perform posterior sampling — use
            :meth:`~aimz.ImpactModel.sample` separately to obtain samples.
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

    def _bind_kernel_args(
        self,
        X: ArrayLike,
        kwargs: Mapping[str, object],
    ) -> dict[str, object]:
        """Bind the input and extra arguments to the kernel signature.

        Args:
            X: Input data, bound to :attr:`~aimz.ImpactModel.param_input`.
            kwargs: Additional arguments passed to the model.

        Returns:
            Mapping of fully bound keyword arguments to invoke the kernel.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_output` is passed as an
                argument.
        """
        args_bound = (
            signature(self.kernel).bind(**{self.param_input: X, **kwargs}).arguments
        )
        if self.param_output in args_bound:
            msg = f"Specifying {self.param_output!r} is not allowed."
            raise TypeError(msg)

        return args_bound

    def _build_kernel_spec(
        self,
        args_bound: Mapping[str, object],
        *,
        with_output: bool,
    ) -> None:
        """Trace the kernel (if needed) and cache default return sites.

        This method is idempotent: if a compatible spec already exists it is a no-op.
        A compatible spec means: ``with_output`` is ``False`` and we already traced
        once, or ``with_output`` is ``True`` and the existing spec was built with an
        observed output (``output_observed=True``).

        Args:
            args_bound: Mapping of fully bound keyword arguments to invoke the kernel
                (includes the input and, when ``with_output`` is ``True``, the observed
                output variable).
            with_output: If ``True`` the trace is validated expecting the output site to
                be observed. If ``False`` the trace may omit an observed output (e.g.,
                prior predictive).
        """
        if self._kernel_spec:
            if with_output:
                if self._kernel_spec.output_observed:
                    return
            elif self._kernel_spec.traced:
                return
        model_trace = trace(seed(self.kernel, rng_seed=self.rng_key)).get_trace(
            **args_bound,
        )
        _validate_kernel_body(
            self.kernel,
            param_output=self.param_output,
            model_trace=model_trace,
            with_output=with_output,
        )
        sample_sites = tuple(k for k, v in model_trace.items() if v["type"] == "sample")
        return_sites = (
            self.param_output,
            *tuple(k for k, v in model_trace.items() if v["type"] == "deterministic"),
        )
        output_observed = bool(with_output)
        self._kernel_spec = KernelSpec(
            traced=True,
            sample_sites=sample_sites,
            return_sites=return_sites,
            output_observed=output_observed,
        )

    def _coerce_return_sites(
        self,
        return_sites: str | Iterable[str] | None,
    ) -> tuple[str, ...]:
        """Return a normalized, validated tuple of site names.

        Args:
            return_sites: User-provided site name(s) or ``None``.

        Returns:
            A tuple of site names.

        Raises:
            ValueError: If a requested site is not present in the traced kernel.
        """
        spec = self._kernel_spec
        if return_sites is None:
            return cast("KernelSpec", spec).return_sites
        requested = (
            (return_sites,)
            if isinstance(return_sites, str)
            else tuple(str(s) for s in return_sites)
        )
        if spec is not None and spec.traced:
            known = set(spec.sample_sites) | set(spec.return_sites)
            unknown = [site for site in requested if site not in known]
            if unknown:
                msg = f"Unknown return site(s): {', '.join(map(repr, unknown))}."
                raise ValueError(msg)

        return requested

    def _create_output_subdir(
        self,
        output_dir: str | Path | None,
    ) -> tuple[Path, Path]:
        """Create a subdirectory for storing output.

        This function is called for its side effect: it creates a subdirectory within
        the specified output directory with a timestamp.

        Args:
            output_dir: Base directory where the output subdirectory will be created.

        Returns:
            The paths to the output directory and the created subdirectory.
        """
        if output_dir is None:
            if self._temp_dir is None:
                self._temp_dir = TemporaryDirectory()
                logger.info("Temporary directory created at: %s", self._temp_dir.name)
            output_dir = self._temp_dir.name
            logger.info(
                "No output directory provided; using the model's temporary directory "
                "for storing output",
            )
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        # Use the outermost method of this instance in the call stack as suffix
        caller = None
        for frame in stack():
            if frame.frame.f_locals.get("self") is not self:
                break
            caller = frame.function
        output_subdir = output_dir / f"{timestamp}_{caller}"
        output_subdir.mkdir(parents=False, exist_ok=False)

        return output_dir, output_subdir

    def _requires_whole_input(
        self,
        X: ArrayLike | ArrayLoader,
        batch_size: int | None,
    ) -> bool:
        """Return whether an observation-aligned posterior requires the whole input.

        Data-parallel replicates the posterior and splits the observation axis of
        ``X`` across devices or into batches. A posterior site shaped
        ``(num_samples, n_obs, ...)`` is aligned to that axis, so once it is split the
        site can no longer be matched to each slice. A single unsplit batch on one
        device sees every observation, so an aligned site still matches.

        Args:
            X: Input data. If array-like, the leading axis is the observation axis.
            batch_size: The requested batch size, or ``None`` to use the default.

        Returns:
            ``True`` if an observation-aligned posterior site would be split from the
            observations it indexes; ``False`` for a data loader, an empty posterior, a
            globally-shaped posterior, or a single unsplit batch.
        """
        if not isinstance(X, ArrayLike) or not self.posterior:
            return False

        n_obs = len(cast("Sized", X))
        min_aligned_ndim = 2
        aligned = any(
            v.ndim >= min_aligned_ndim and v.shape[1] == n_obs
            for v in self.posterior.values()
        )
        if not aligned:
            return False

        if self._num_devices > 1:
            return True

        batch_size = _resolve_batch_size(
            batch_size,
            axis_size=n_obs,
            other_size=self._num_samples,
            num_devices=self._num_devices,
        )

        return batch_size < n_obs

    def sample_prior_predictive_on_batch(
        self,
        X: ArrayLike,
        *,
        num_samples: int = 1000,
        rng_key: Array | None = None,
        return_sites: str | Iterable[str] | None = None,
        return_datatree: bool = True,
        **kwargs: object,
    ) -> xr.DataTree | dict[str, npt.NDArray]:
        """Draw samples from the prior predictive distribution.

        Args:
            X (ArrayLike): Input data. The leading axis is the observation axis.
            num_samples: The number of samples to draw.
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed.
            return_sites: Names of variables (sites) to return. If ``None``, samples
                :attr:`~aimz.ImpactModel.param_output` and deterministic sites.
            return_datatree: If ``True``, return a :external:class:`~xarray.DataTree`;
                otherwise return a :class:`dict`.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Prior predictive samples. Posterior samples are included if available.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_output` is passed as an
                argument.

        See Also:
            :meth:`~aimz.ImpactModel.sample_prior_predictive`
        """
        X = cast("Array", _validate_X_y_to_jax(X))

        args_bound = self._bind_kernel_args(X, kwargs=kwargs)
        self._build_kernel_spec(args_bound, with_output=False)

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        prior_predictive_samples = device_get(
            _sample_forward(
                self.kernel,
                rng_keys=random.split(rng_key, num=num_samples),
                return_sites=self._coerce_return_sites(return_sites),
                samples=None,
                model_kwargs=args_bound,
            ),
        )

        if not return_datatree:
            return prior_predictive_samples

        out = xr.DataTree(name="root")
        out["prior_predictive"] = _dict_to_datatree(prior_predictive_samples)
        if self.posterior:
            out["posterior"] = _dict_to_datatree(self.posterior)

        return out

    def sample_prior_predictive(
        self,
        X: ArrayLike,
        *,
        num_samples: int = 1000,
        rng_key: Array | None = None,
        return_sites: str | Iterable[str] | None = None,
        shard_axis: Literal["obs", "draw"] = "obs",
        batch_size: int | None = None,
        output_dir: str | Path | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> xr.DataTree:
        """Draw samples from the prior predictive distribution.

        Results are written to disk in the Zarr format, with computing and file writing
        decoupled and executed concurrently.

        Args:
            X (ArrayLike): Input data. The leading axis is the observation axis.
            num_samples: The number of samples to draw.
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed.
            return_sites: Names of variables (sites) to return. If ``None``, samples
                :attr:`~aimz.ImpactModel.param_output` and deterministic sites.
            shard_axis: Multi-device sharding strategy. ``"obs"`` (default) shards the
                input across devices and replicates the drawn samples. ``"draw"`` shards
                the drawn samples across devices and replicates the input.
            batch_size: Size of each batch, taken from the input under
                ``shard_axis="obs"`` and from the draws under ``shard_axis="draw"``.
                Also used as the chunk size when storing results. If ``None``, it is
                determined automatically from the input size and number of samples.
            output_dir: The directory where the outputs will be saved. If the specified
                directory does not exist, it will be created automatically. If ``None``,
                a model-owned temporary directory is used. A subdirectory is generated
                within this directory to store the outputs. The temporary directory is
                removed by :meth:`~aimz.ImpactModel.cleanup`,
                :meth:`~aimz.ImpactModel.cleanup_models`, or when the model is
                garbage-collected. Pass an explicit ``output_dir`` to keep results
                beyond the model's lifetime.
            progress: Whether to display a progress bar.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Prior predictive samples. Posterior samples are included if available.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_output` is passed as an
                argument.
            ValueError: If ``shard_axis`` is not ``"obs"`` or ``"draw"``.
            NotImplementedError: If a return site's axis-1 size does not match the
                input batch size (``shard_axis="obs"`` only).

        See Also:
            :meth:`~aimz.ImpactModel.sample_prior_predictive_on_batch` for an
            in-memory alternative.

            :meth:`~aimz.ImpactModel.cleanup` to remove the temporary directory if
            created.
        """
        _validate_shard_axis(shard_axis, X=X)
        _validate_batch_size(batch_size, X=X)
        _validate_aligned_inputs(X, y=None, kwargs=kwargs)

        args_bound = self._bind_kernel_args(X, kwargs=kwargs)

        # Build the kernel spec from a single-row slice so the trace runs on a tiny
        # input. Only when the spec is not already cached (fitted models keep theirs),
        # and before the return-site defaults resolve so they work even before fitting.
        if not (self._kernel_spec and self._kernel_spec.traced):
            probe = {
                k: (cast("Any", v)[:1] if _is_arraylike(v) else v)
                for k, v in args_bound.items()
            }
            self._build_kernel_spec(probe, with_output=False)

        return_sites = self._coerce_return_sites(return_sites)

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        output_dir, output_subdir = self._create_output_subdir(output_dir)

        self._streamer.write_predictive(
            _WriteRequest(
                shard_axis,
                X=X,
                return_sites=return_sites,
                num_samples=num_samples,
                batch_size=batch_size,
                output_dir=output_subdir,
                progress=progress,
                loader_rng_key=self.rng_key,
                kwargs=kwargs,
            ),
            kernel=self.kernel,
            rng_key=rng_key,
            group="prior_predictive",
            posterior=self.posterior,
        )

        return _build_datatree(
            output_dir,
            output_subdir=output_subdir,
            group="prior_predictive",
            posterior=self.posterior,
        )

    def sample(
        self,
        *,
        num_samples: int = 1000,
        rng_key: Array | None = None,
        return_sites: str | Iterable[str] | None = None,
        return_datatree: bool = True,
        **kwargs: object,
    ) -> xr.DataTree | dict[str, npt.NDArray]:
        """Draw posterior samples from a fitted model.

        Args:
            num_samples: The number of posterior samples to draw.
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed. Ignored if the inference method is MCMC,
                where the ``post_warmup_state`` property will be used to continue
                sampling.
            return_sites: Names of variables (sites) to return. If ``None``, samples
                all latent sites. Ignored if the inference method is MCMC.
            return_datatree: If ``True``, return a :external:class:`~xarray.DataTree`;
                otherwise return a :class:`dict`.
            **kwargs: Additional arguments passed to the model. Only relevant when the
                inference method is MCMC.

        Returns:
            Posterior samples.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_output` is not passed as an
                argument when the inference method is MCMC.
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
            posterior_samples = device_get(
                _sample_forward(
                    substitute(
                        self.inference.guide,
                        data=cast("SVIRunResult", self.vi_result).params,
                    ),
                    rng_keys=random.split(rng_key, num=num_samples),
                    return_sites=self._coerce_return_sites(return_sites)
                    if return_sites
                    else None,
                    samples=None,
                    model_kwargs=None,
                ),
            )

        if not return_datatree:
            return posterior_samples

        out = xr.DataTree(name="root")
        out["posterior"] = _dict_to_datatree(posterior_samples)

        return out

    def sample_posterior_predictive_on_batch(
        self,
        X: ArrayLike,
        *,
        intervention: dict | None = None,
        rng_key: Array | None = None,
        return_sites: str | Iterable[str] | None = None,
        return_datatree: bool = True,
        **kwargs: object,
    ) -> xr.DataTree | dict[str, npt.NDArray]:
        """Draw samples from the posterior predictive distribution.

        This method is a convenience alias for
        :meth:`~aimz.ImpactModel.predict_on_batch`, with ``in_sample`` automatically
        set to ``True``.

        Args:
            X (ArrayLike): Input data. The leading axis is the observation axis.
            intervention: A dictionary mapping sample sites to their corresponding
                intervention values. Interventions enable counterfactual analysis by
                modifying the specified sample sites during prediction (posterior
                predictive sampling).
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed.
            return_sites: Names of variables (sites) to return. If ``None``, samples
                :attr:`~aimz.ImpactModel.param_output` and deterministic sites.
            return_datatree: If ``True``, return a :external:class:`~xarray.DataTree`;
                otherwise return a :class:`dict`.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Posterior predictive samples. Posterior samples are included if available.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_output` is passed as an
                argument.

        See Also:
            :meth:`~aimz.ImpactModel.predict_on_batch`.
        """
        return self.predict_on_batch(
            X,
            intervention=intervention,
            rng_key=rng_key,
            in_sample=True,
            return_sites=return_sites,
            return_datatree=return_datatree,
            **kwargs,
        )

    def sample_posterior_predictive(
        self,
        X: ArrayLike | ArrayLoader,
        *,
        intervention: dict | None = None,
        rng_key: Array | None = None,
        return_sites: str | Iterable[str] | None = None,
        shard_axis: Literal["obs", "draw"] = "obs",
        batch_size: int | None = None,
        output_dir: str | Path | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> xr.DataTree:
        """Draw samples from the posterior predictive distribution.

        This method is a convenience alias for :meth:`~aimz.ImpactModel.predict`, with
        ``in_sample`` automatically set to ``True``.

        Args:
            X (ArrayLike | ArrayLoader): Input data. If array-like, the leading axis is
                the observation axis. Alternatively, a data loader that holds all
                array-like objects and handles batching internally.
            intervention: A dictionary mapping sample sites to their corresponding
                intervention values. Interventions enable counterfactual analysis by
                modifying the specified sample sites during prediction (posterior
                predictive sampling).
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed.
            return_sites: Names of variables (sites) to return. If ``None``, samples
                :attr:`~aimz.ImpactModel.param_output` and deterministic sites.
            shard_axis: Multi-device sharding strategy; no effect on a single device.
                ``"obs"`` (default) shards the input across devices and replicates the
                posterior. ``"draw"`` shards the posterior across devices and replicates
                the input, which must be an array, not a data loader. If the model has
                no posterior samples, the data path is used regardless of
                ``shard_axis``.
            batch_size: Size of each batch, taken from the input under
                ``shard_axis="obs"`` and from the draws under ``shard_axis="draw"``.
                Also used as the chunk size when storing results. If ``None``, it is
                determined automatically from the input size and number of samples.
                Ignored if ``X`` is a data loader, in which case the data loader is
                expected to handle batching internally.
            output_dir: The directory where the outputs will be saved. If the specified
                directory does not exist, it will be created automatically. If ``None``,
                a model-owned temporary directory is used. A subdirectory is generated
                within this directory to store the outputs. The temporary directory is
                removed by :meth:`~aimz.ImpactModel.cleanup`,
                :meth:`~aimz.ImpactModel.cleanup_models`, or when the model is
                garbage-collected. Pass an explicit ``output_dir`` to keep results
                beyond the model's lifetime.
            progress: Whether to display a progress bar.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Posterior predictive samples. Posterior samples are included if available.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_output` is passed as an
                argument, or ``shard_axis="draw"`` is used with a data loader ``X``.
            ValueError: If ``shard_axis`` is not ``"obs"`` or ``"draw"``.

        See Also:
            :meth:`~aimz.ImpactModel.predict()`.
        """
        return self.predict(
            X,
            intervention=intervention,
            rng_key=rng_key,
            in_sample=True,
            return_sites=return_sites,
            batch_size=batch_size,
            shard_axis=shard_axis,
            output_dir=output_dir,
            progress=progress,
            **kwargs,
        )

    def train_on_batch(
        self,
        X: ArrayLike,
        y: ArrayLike,
        rng_key: Array | None = None,
        **kwargs: object,
    ) -> tuple[SVIState, Array]:
        """Run a single VI step on the given batch of data.

        Args:
            X (ArrayLike): Input data. The leading axis is the observation axis.
            y (ArrayLike): Output data. The leading axis is the observation axis.
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed. The key is only used for initialization if
                the internal SVI state is not yet set.
            **kwargs: Additional arguments passed to the model.

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
            self._build_kernel_spec(
                signature(self.kernel).bind(**batch).arguments,
                with_output=True,
            )
            if rng_key is None:
                self._rng_key, rng_key = random.split(self._rng_key)
            self._vi_state = cast("SVI", self.inference).init(rng_key, **batch)
        if self._fn_vi_update is None:
            _, kwargs_extra = _group_kwargs(kwargs)
            self._fn_vi_update = jit(
                cast("SVI", self.inference).update,
                static_argnames=tuple(kwargs_extra),
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
        rng_key: Array | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> Self:
        """Fit the impact model to the provided batch of data.

        This method behaves differently depending on the inference method specified at
        th initialization:

        - SVI
            Runs variational inference on the provided batch by invoking the
            :external:meth:`~numpyro.infer.svi.SVI.run` method of the
            :external:class:`~numpyro.infer.svi.SVI` instance from `NumPyro`_ to
            estimate the posterior distribution, then draws samples from it.

        - MCMC
            Runs posterior sampling by invoking the
            :external:meth:`~numpyro.infer.mcmc.MCMC.run` method of the
            :external:class:`~numpyro.infer.mcmc.MCMC` instance from `NumPyro`_.

        Args:
            X (ArrayLike): Input data. The leading axis is the observation axis.
            y (ArrayLike): Output data. The leading axis is the observation axis.
            num_steps: Number of steps for variational inference optimization. Ignored
                if the inference method is MCMC.
            num_samples: The number of posterior samples to draw. Ignored if the
                inference method is MCMC.
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed.
            progress: Whether to display a progress bar. Ignored if the inference method
                is MCMC.
            **kwargs: Additional arguments passed to the model.

        Returns:
            The fitted model instance, enabling method chaining.

        Note:
            This method continues training from the existing SVI state if available. To
            start training from scratch, create a new model instance.
        """
        X, y = _validate_X_y_to_jax(X, y=y)

        # Validate the provided parameters against the kernel's signature
        args_bound = (
            signature(self.kernel)
            .bind(**{self.param_input: X, self.param_output: y, **kwargs})
            .arguments
        )
        self._build_kernel_spec(args_bound, with_output=True)
        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)
        rng_key, rng_subkey = random.split(rng_key)
        if isinstance(self.inference, SVI):
            self._num_samples = num_samples
            logger.info("Performing variational inference optimization")
            self.vi_result: SVIRunResult = self.inference.run(
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
            logger.info("Drawing posterior samples (num_samples=%d)", self._num_samples)
            rng_key, rng_subkey = random.split(rng_key)
            self._posterior = _sample_forward(
                substitute(self.inference.guide, data=self.vi_result.params),
                rng_keys=random.split(rng_subkey, num=self._num_samples),
                return_sites=None,
                samples=None,
                model_kwargs=None,
            )
        elif isinstance(self.inference, MCMC):
            logger.info(
                "Drawing posterior samples (num_samples=%d)",
                self.inference.num_samples,
            )
            self.inference.run(rng_subkey, **args_bound)
            self._posterior = device_get(self.inference.get_samples())
            self._num_samples = (
                next(iter(self.posterior.values())).shape[0] if self.posterior else 0
            )

        self._is_fitted = True

        return self

    def fit(
        self,
        X: ArrayLike | ArrayLoader,
        y: ArrayLike | None = None,
        *,
        num_samples: int = 1000,
        rng_key: Array | None = None,
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
            X (ArrayLike | ArrayLoader): Input data. If array-like, the leading axis is
                the observation axis. Alternatively, a data loader that holds all
                array-like objects and handles batching internally.
            y (ArrayLike | None): Output data. The leading axis is the observation axis.
                Must be ``None`` if ``X`` is a data loader.
            num_samples: The number of posterior samples to draw.
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed.
            progress: Whether to display a progress bar.
            batch_size: The number of data points processed at each step of variational
                inference. If ``None``, the entire dataset is used as a single batch in
                each epoch. Ignored if ``X`` is a data loader, in which case the data
                loader is expected to handle batching internally.
            epochs: The number of epochs for variational inference optimization.
            shuffle: Whether to shuffle the data at each epoch. Ignored if ``X`` is a
                data loader.
            **kwargs: Additional arguments passed to the model.

        Returns:
            The fitted model instance, enabling method chaining.

        Raises:
            TypeError: If the inference method is MCMC.

        Note:
            This method continues training from the existing SVI state if available.
            To start training from scratch, create a new model instance. It does not
            check whether the model or guide is written to support subsampling semantics
            (e.g., using `NumPyro`_'s :external:func:`~numpyro.primitives.subsample` or
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
            # The loader feeds `train_on_batch(**batch)`, whose parameters are `X`/`y`
            # (it maps them to the kernel's param_input/param_output itself), so the
            # dataset is keyed by those literal names, not the kernel's parameter names.
            param_input="X",
            param_output="y",
            rng_key=rng_subkey,
            batch_size=batch_size if batch_size is not None else len(cast("Sized", X)),
            num_samples=num_samples,
            shuffle=shuffle,
            device=None,
            **kwargs,
        )

        logger.info("Performing variational inference optimization")
        losses: list[float] = []
        rng_key, rng_subkey = random.split(rng_key)
        for epoch in range(epochs):
            losses_epoch: list[float] = []
            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                total=len(dataloader),
                disable=not progress,
                dynamic_ncols=True,
            )
            for batch, _ in pbar:
                self._vi_state, loss = self.train_on_batch(
                    **batch,
                    **kwargs_extra,
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
        if np.any(np.isnan(cast("SVIRunResult", self.vi_result).losses)):
            msg = "Loss contains NaN or Inf, indicating numerical instability."
            warn(msg, category=RuntimeWarning, stacklevel=2)

        self._is_fitted = True

        logger.info(
            "Drawing posterior samples (num_samples=%d)",
            self._num_samples,
        )
        rng_key, rng_subkey = random.split(rng_key)
        self._posterior = _sample_forward(
            substitute(
                self.inference.guide,
                data=cast("SVIRunResult", self.vi_result).params,
            ),
            rng_keys=random.split(rng_subkey, num=self._num_samples),
            return_sites=None,
            samples=None,
            model_kwargs=None,
        )

        return self

    def is_fitted(self) -> bool:
        """Check fitted status.

        Returns:
            `True` if the model is fitted, `False` otherwise.

        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def set_posterior_sample(self, posterior_sample: dict[str, Array]) -> Self:
        """Set posterior samples for the model.

        This method sets externally obtained posterior samples on the model instance,
        enabling downstream analysis without requiring a call to
        :meth:`~aimz.ImpactModel.fit` or :meth:`~aimz.ImpactModel.fit_on_batch`.

        It is primarily intended for workflows where posterior sampling is performed
        manually—for example, using `NumPyro`_'s
        :external:class:`~numpyro.infer.svi.SVI` (or
        :external:class:`~numpyro.infer.mcmc.MCMC`) with the
        :external:class:`~numpyro.infer.util.Predictive` API—and the resulting
        posterior samples are injected into the model for further use.

        Internally, ``batch_ndims`` is set to ``1`` by default to correctly handle the
        batch dimensions of the posterior samples. For more information, refer to the
        `NumPyro documentation <https://num.pyro.ai/en/stable/utilities.html#predictive>`__.

        Args:
            posterior_sample: Posterior samples to set for the model.

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
        if not batch_shapes:
            msg = "`posterior_sample` cannot be empty."
            raise ValueError(msg)
        if len(batch_shapes) > 1:
            msg = (
                f"Inconsistent batch shapes found in `posterior_sample`: {batch_shapes}"
            )
            raise ValueError(msg)
        (self._num_samples,) = batch_shapes.pop()
        self._posterior = posterior_sample
        if self._kernel_spec is None:
            self._kernel_spec = KernelSpec(
                traced=False,
                sample_sites=tuple(self._posterior.keys()),
                return_sites=(self.param_output,),
                output_observed=False,
            )
        self._is_fitted = True

        return self

    def predict_on_batch(
        self,
        X: ArrayLike,
        *,
        intervention: dict | None = None,
        rng_key: Array | None = None,
        in_sample: bool = True,
        return_sites: str | Iterable[str] | None = None,
        return_datatree: bool = True,
        **kwargs: object,
    ) -> xr.DataTree | dict[str, npt.NDArray]:
        """Predict the output based on the fitted model.

        This method returns predictions for a single batch of input data and is better
        suited for:

            1) Models incompatible with :meth:`~aimz.ImpactModel.predict` due to their
            posterior sample shapes.

            2) Scenarios where writing results to to files (e.g., disk, cloud storage)
            is not desired.

            3) Smaller datasets, as this method may be slower due to limited
            parallelism.

        Args:
            X (ArrayLike): Input data. The leading axis is the observation axis.
            intervention: A dictionary mapping sample sites to their corresponding
                intervention values. Interventions enable counterfactual analysis by
                modifying the specified sample sites during prediction (posterior
                predictive sampling).
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed.
            in_sample: Specifies the group where posterior predictive samples are stored
                in the returned output. If ``True``, samples are stored in the
                ``posterior_predictive`` group, indicating they were generated based on
                data used during model fitting. If ``False``, samples are stored in the
                ``predictions`` group, indicating they were generated based on
                out-of-sample data.
            return_sites: Names of variables (sites) to return. If ``None``, samples
                :attr:`~aimz.ImpactModel.param_output` and deterministic sites.
            return_datatree: If ``True``, return a :external:class:`~xarray.DataTree`;
                otherwise return a :class:`dict`.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Posterior predictive samples. Posterior samples are included if available.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_output` is passed as an
                argument.
        """
        _check_is_fitted(self)

        X = cast("Array", _validate_X_y_to_jax(X))

        args_bound = self._bind_kernel_args(X, kwargs=kwargs)

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        if intervention is None:
            kernel = self.kernel
        else:
            kernel = do(self.kernel, data=intervention)

        samples = device_get(
            _sample_forward(
                kernel,
                rng_keys=random.split(rng_key, num=self._num_samples),
                return_sites=self._coerce_return_sites(return_sites),
                samples=self.posterior,
                model_kwargs=args_bound,
            ),
        )

        if not return_datatree:
            return samples

        out = xr.DataTree(name="root")
        if self.posterior:
            out["posterior"] = _dict_to_datatree(self.posterior)
        out["posterior_predictive" if in_sample else "predictions"] = _dict_to_datatree(
            samples,
        )

        return out

    def predict(
        self,
        X: ArrayLike | ArrayLoader,
        *,
        intervention: dict | None = None,
        rng_key: Array | None = None,
        in_sample: bool = True,
        return_sites: str | Iterable[str] | None = None,
        shard_axis: Literal["obs", "draw"] = "obs",
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
            X (ArrayLike | ArrayLoader): Input data. If array-like, the leading axis is
            the observation axis. Alternatively, a data loader that holds all array-like
            objects and handles batching internally.
            intervention: A dictionary mapping sample sites to their corresponding
                intervention values. Interventions enable counterfactual analysis by
                modifying the specified sample sites during prediction (posterior
                predictive sampling).
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed.
            in_sample: Specifies the group where posterior predictive samples are stored
                in the returned output. If ``True``, samples are stored in the
                ``posterior_predictive`` group, indicating they were generated based on
                data used during model fitting. If ``False``, samples are stored in the
                ``predictions`` group, indicating they were generated based on
                out-of-sample data.
            return_sites: Names of variables (sites) to return. If ``None``, samples
                :attr:`~aimz.ImpactModel.param_output` and deterministic sites.
            shard_axis: Multi-device sharding strategy; no effect on a single device.
                ``"obs"`` (default) shards the input across devices and replicates the
                posterior. ``"draw"`` shards the posterior across devices and replicates
                the input, which must be an array, not a data loader. If the model has
                no posterior samples, the data path is used regardless of
                ``shard_axis``.
            batch_size: Size of each batch, taken from the input under
                ``shard_axis="obs"`` and from the draws under ``shard_axis="draw"``.
                Also used as the chunk size when storing results. If ``None``, it is
                determined automatically from the input size and number of samples.
                Ignored if ``X`` is a data loader, in which case the data loader is
                expected to handle batching internally.
            output_dir: The directory where the outputs will be saved. If the specified
                directory does not exist, it will be created automatically. If ``None``,
                a model-owned temporary directory is used. A subdirectory is generated
                within this directory to store the outputs. The temporary directory is
                removed by :meth:`~aimz.ImpactModel.cleanup`,
                :meth:`~aimz.ImpactModel.cleanup_models`, or when the model is
                garbage-collected. Pass an explicit ``output_dir`` to keep results
                beyond the model's lifetime.
            progress: Whether to display a progress bar.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Posterior predictive samples. Posterior samples are included if available.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_output` is passed as an
                argument, or ``shard_axis="draw"`` is used with a data loader ``X``.
            ValueError: If ``shard_axis`` is not ``"obs"`` or ``"draw"``.
            NotImplementedError: If a return site's axis-1 size does not match the
                input batch size (``shard_axis="obs"`` only).

        See Also:
            :meth:`~aimz.ImpactModel.cleanup` to remove the temporary directory if
            created.
        """
        _check_is_fitted(self)
        _validate_shard_axis(shard_axis, X=X)
        _validate_batch_size(batch_size, X=X)
        _validate_aligned_inputs(X, y=None, kwargs=kwargs)
        # No posterior to shard means draw-parallel has nothing to chunk, so it behaves
        # identically to the data-parallel path.
        if not self.posterior:
            shard_axis = "obs"

        if isinstance(X, ArrayLike):
            _ = self._bind_kernel_args(X, kwargs=kwargs)
            if shard_axis == "obs" and self._requires_whole_input(
                X,
                batch_size=batch_size,
            ):
                msg = (
                    "One or more posterior sample shapes are not compatible with "
                    "`.predict()` under `shard_axis='obs'`; rerunning with "
                    "`shard_axis='draw'`. Pass `shard_axis='draw'` to silence this "
                    "warning."
                )
                warn(msg, category=UserWarning, stacklevel=2)
                return self.predict(
                    X,
                    intervention=intervention,
                    rng_key=rng_key,
                    in_sample=in_sample,
                    return_sites=return_sites,
                    shard_axis="draw",
                    batch_size=batch_size,
                    output_dir=output_dir,
                    progress=progress,
                    **kwargs,
                )

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        if intervention is None:
            kernel = self.kernel
        else:
            kernel = do(self.kernel, data=intervention)

        return_sites = self._coerce_return_sites(return_sites)

        group = "posterior_predictive" if in_sample else "predictions"

        output_dir, output_subdir = self._create_output_subdir(output_dir)

        self._streamer.write_predictive(
            _WriteRequest(
                shard_axis,
                X=X,
                return_sites=return_sites,
                num_samples=self._num_samples,
                batch_size=batch_size,
                output_dir=output_subdir,
                progress=progress,
                loader_rng_key=self.rng_key,
                kwargs=kwargs,
            ),
            kernel=kernel,
            rng_key=rng_key,
            group=group,
            posterior=self.posterior,
        )

        return _build_datatree(
            output_dir,
            output_subdir=output_subdir,
            group=group,
            posterior=self.posterior,
        )

    def estimate_effect(
        self,
        output_baseline: xr.DataTree | None = None,
        output_intervention: xr.DataTree | None = None,
        args_baseline: dict | None = None,
        args_intervention: dict | None = None,
        *,
        on_batch: bool = False,
    ) -> xr.DataTree:
        """Estimate the effect of an intervention.

        This computes (intervention - baseline) for every variable in the shared
        predictive group, preserving sampling (chain/draw) dimensions. When
        interventions are used in prediction they are applied internally through
        `NumPyro`_'s :external:class:`~numpyro.handlers.do` effect handler (graph
        surgery) without requiring model rewrites.

        Args:
            output_baseline: Precomputed output for the baseline scenario.
            output_intervention: Precomputed output for the intervention scenario.
            args_baseline: Input arguments for the baseline scenario. Passed to the
                prediction method to compute predictions if ``output_baseline`` is not
                provided. Ignored if ``output_baseline`` is already given.
            args_intervention: Input arguments for the intervention scenario. Passed to
                the prediction method to compute predictions if
                ``output_intervention`` is not provided. Ignored if
                ``output_intervention`` is already given.
            on_batch: If ``True``, use
                :meth:`~aimz.ImpactModel.predict_on_batch` instead of
                :meth:`~aimz.ImpactModel.predict` when computing predictions from
                ``args_baseline`` or ``args_intervention``. Ignored when precomputed
                outputs are provided.

        Returns:
            The estimated impact of an intervention. Posterior samples are included if
            available.

        Raises:
            ValueError: If neither ``output_baseline`` nor ``args_baseline`` is
                provided, or if neither ``output_intervention`` nor
                ``args_intervention`` is provided.

        See Also:
            :meth:`~aimz.ImpactModel.cleanup` to remove the temporary directory if
            created.
        """
        _check_is_fitted(self)

        _predict = self.predict_on_batch if on_batch else self.predict

        if output_baseline:
            dt_baseline = output_baseline
        elif args_baseline:
            dt_baseline = _predict(**args_baseline)
        else:
            msg = "Either `output_baseline` or `args_baseline` must be provided."
            raise ValueError(msg)

        if output_intervention:
            dt_intervention = output_intervention
        elif args_intervention:
            dt_intervention = _predict(**args_intervention)
        else:
            msg = (
                "Either `output_intervention` or `args_intervention` must be provided."
            )
            raise ValueError(msg)

        if isinstance(dt_baseline, dict):
            in_sample = args_baseline.get("in_sample", True) if args_baseline else True
            group = "posterior_predictive" if in_sample else "predictions"
            wrapper = xr.DataTree(name="root")
            wrapper[group] = _dict_to_datatree(dt_baseline)
            dt_baseline = wrapper
        if isinstance(dt_intervention, dict):
            in_sample = (
                args_intervention.get("in_sample", True) if args_intervention else True
            )
            group = "posterior_predictive" if in_sample else "predictions"
            wrapper = xr.DataTree(name="root")
            wrapper[group] = _dict_to_datatree(dt_intervention)
            dt_intervention = wrapper

        group = _validate_group(dt_baseline, dt_intervention=dt_intervention)

        out = xr.DataTree(name="root")
        out[group] = dt_intervention[group] - dt_baseline[group]
        if self.posterior:
            out["posterior"] = _dict_to_datatree(self.posterior)
        # Propagate an output directory attribute to the effect result.
        # Precedence:
        #   1) intervention output_dir (if present)
        #   2) baseline output_dir (if present)
        #   3) model temporary directory (if it exists)
        output_dir_attr = (
            dt_intervention.attrs.get("output_dir")
            or dt_baseline.attrs.get("output_dir")
            or (self.temp_dir if self.temp_dir is not None else None)
        )
        if output_dir_attr is not None:
            out.attrs["output_dir"] = output_dir_attr

        return out

    def log_likelihood(
        self,
        X: ArrayLike | ArrayLoader,
        y: ArrayLike | None = None,
        *,
        shard_axis: Literal["obs", "draw"] = "obs",
        batch_size: int | None = None,
        output_dir: str | Path | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> xr.DataTree:
        """Compute the log-likelihood of the data under the given model.

        Results are written to disk in the Zarr format, with computing and file writing
        decoupled and executed concurrently.

        Args:
            X (ArrayLike | ArrayLoader): Input data. If array-like, the leading axis is
            the observation axis. Alternatively, a data loader that holds all array-like
            objects and handles batching internally.
            y (ArrayLike | None): Output data. The leading axis is the observation axis.
                Must be ``None`` if ``X`` is a data loader.
            shard_axis: Multi-device sharding strategy; no effect on a single device.
                ``"obs"`` (default) shards the input across devices and replicates the
                posterior. ``"draw"`` shards the posterior across devices and replicates
                the input, which must be an array, not a data loader. If the model has
                no posterior samples, the data path is used regardless of
                ``shard_axis``.
            batch_size: Size of each batch, taken from the input under
                ``shard_axis="obs"`` and from the draws under ``shard_axis="draw"``.
                Also used as the chunk size when storing results. If ``None``, it is
                determined automatically from the input size and number of samples.
                Ignored if ``X`` is a data loader, in which case the data loader is
                expected to handle batching internally.
            output_dir: The directory where the outputs will be saved. If the specified
                directory does not exist, it will be created automatically. If ``None``,
                a model-owned temporary directory is used. A subdirectory is generated
                within this directory to store the outputs. The temporary directory is
                removed by :meth:`~aimz.ImpactModel.cleanup`,
                :meth:`~aimz.ImpactModel.cleanup_models`, or when the model is
                garbage-collected. Pass an explicit ``output_dir`` to keep results
                beyond the model's lifetime.
            progress: Whether to display a progress bar.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Log-likelihood values. Posterior samples are included if available.

        Raises:
            TypeError: If ``shard_axis="draw"`` is used with a data loader ``X``.
            ValueError: If ``shard_axis`` is not ``"obs"`` or ``"draw"``.
            NotImplementedError: If a return site's axis-1 size does not match the
                input batch size (``shard_axis="obs"`` only).

        See Also:
            :meth:`~aimz.ImpactModel.cleanup` to remove the temporary directory if
            created.
        """
        _check_is_fitted(self)
        _validate_shard_axis(shard_axis, X=X)
        _validate_batch_size(batch_size, X=X)
        _validate_aligned_inputs(X, y=y, kwargs=kwargs)

        # No posterior to shard means draw-parallel has nothing to chunk, so it behaves
        # identically to the data-parallel path (a single-draw result).
        if not self.posterior:
            shard_axis = "obs"

        if shard_axis == "obs" and self._requires_whole_input(X, batch_size=batch_size):
            msg = (
                "One or more posterior sample shapes are not compatible with "
                "`.log_likelihood()` under `shard_axis='obs'`; rerunning with "
                "`shard_axis='draw'`. Pass `shard_axis='draw'` to silence this warning."
            )
            warn(msg, category=UserWarning, stacklevel=2)
            return self.log_likelihood(
                X,
                y,
                batch_size=batch_size,
                shard_axis="draw",
                output_dir=output_dir,
                progress=progress,
                **kwargs,
            )

        # Seed once so tracing succeeds even when posterior is empty or only partially
        # covers latent sites.
        kernel = seed(self.kernel, rng_seed=self.rng_key)

        output_dir, output_subdir = self._create_output_subdir(output_dir)

        self._streamer.write_log_likelihood(
            _WriteRequest(
                shard_axis,
                X=X,
                return_sites=(self.param_output,),
                num_samples=self._num_samples,
                batch_size=batch_size,
                output_dir=output_subdir,
                progress=progress,
                loader_rng_key=self.rng_key,
                kwargs=kwargs,
            ),
            kernel=kernel,
            posterior=self.posterior,
            y=y,
        )

        return _build_datatree(
            output_dir,
            output_subdir=output_subdir,
            group="log_likelihood",
            posterior=self.posterior,
        )

    def cleanup(self) -> None:
        """Clean up the temporary directory created for storing outputs.

        If the temporary directory was never created or has already been cleaned up,
        this method does nothing. It does not delete any explicitly specified output
        directory. While the temporary directory is typically removed automatically
        during garbage collection, this behavior is not guaranteed, so calling this
        method explicitly is recommended for timely resource release.

        See Also:
            :meth:`~aimz.ImpactModel.cleanup_models` — clean temporary directories for
            all tracked model instances.
        """
        if hasattr(self, "_temp_dir") and self._temp_dir is not None:
            logger.info("Temporary directory cleaned up at: %s", self._temp_dir.name)
            self._temp_dir.cleanup()
            self._temp_dir = None

    @classmethod
    def cleanup_models(cls) -> None:
        """Clean up temporary directories for all :class:`~aimz.ImpactModel` instances.

        See Also:
            :meth:`~aimz.ImpactModel.cleanup` — clean the temporary directory for a
            single instance.
        """
        for model in cls._models:
            model.cleanup()
