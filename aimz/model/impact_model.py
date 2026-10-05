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

import copyreg
import logging
import math
import pickle
from datetime import UTC, datetime
from functools import partial
from inspect import signature, stack
from pathlib import Path
from shutil import rmtree
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
    local_devices,
    make_mesh,
    random,
)
from jax.sharding import AxisType, NamedSharding, PartitionSpec
from jax.typing import ArrayLike
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import MCMC, SVI
from numpyro.infer.svi import SVIRunResult, SVIState
from tqdm.auto import tqdm

from aimz._exceptions import (
    _SKIP_FILE_PREFIXES,
    AimzWarning,
    FitWarning,
    NotFittedError,
    OutputWarning,
    PerformanceWarning,
)
from aimz.model._core import BaseModel
from aimz.model._streaming import (
    _OutputStreamer,
    _RuntimeContext,
    _WriteRequest,
)
from aimz.model.kernel_spec import KernelSpec
from aimz.sampling._forward import _sample_forward
from aimz.utils._format import (
    _build_datatree,
    _dict_to_datatree,
)
from aimz.utils._kwargs import _combine, _group_kwargs, _partition
from aimz.utils._validation import (
    _check_is_fitted,
    _validate_aligned_inputs,
    _validate_batch_size,
    _validate_group,
    _validate_intervention,
    _validate_kernel_body,
    _validate_shard_axis,
    _validate_store,
    _validate_X_y_to_jax,
)
from aimz.utils.data import ArrayLoader
from aimz.utils.data._input_setup import (
    _fits_single_batch,
    _resolve_batch_size,
    _setup_inputs,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sized

    from dask.array import Array as DaskArray

logger = logging.getLogger(__name__)


def _reduce_key(key: Array) -> str | tuple[Any, ...]:
    """Reduce a typed PRNG key for pickling.

    A typed key pickles a copy of its PRNG implementation, which some samplers reject
    in a new process, so a key of a registered implementation is rebuilt from its raw
    data and the implementation's name instead.

    Args:
        key: The typed PRNG key to pickle.

    Returns:
        The callable and arguments that rebuild the key.
    """
    impl = random.key_impl(key)
    if isinstance(impl, str):
        return partial(random.wrap_key_data, impl=impl), (random.key_data(key),)

    return key.__reduce_ex__(pickle.DEFAULT_PROTOCOL)


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

        Raises:
            KernelValidationError: If the kernel signature does not meet the required
                constraints.
            TypeError: If ``inference`` is neither SVI nor MCMC.

        Warning:
            The ``rng_key`` parameter should be provided as a **typed key array**
            created with :external:func:`jax.random.key`, rather than a legacy
            ``uint32`` key created with :external:func:`jax.random.PRNGKey`.
        """
        super().__init__(kernel, param_input, param_output)
        self._kernel_spec: KernelSpec | None = None
        self._dims: dict[str, tuple[str, ...]] = {}
        self._site_sizes: dict[str, tuple[int, int]] = {}
        if isinstance(rng_key, Array) and rng_key.dtype == jnp.uint32:
            msg = "Legacy `uint32` PRNGKey detected; converting to a typed key array."
            warn(msg, category=AimzWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
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
        self._is_fitted = False
        self._posterior: dict[str, Array] | None = None
        self._num_samples = 0
        self._num_chains = 1
        self._init_runtime_attrs()

    def _init_runtime_attrs(self) -> None:
        """Initialize runtime attributes."""
        self._fn_vi_update: Callable | None = None
        self._num_devices = local_device_count()
        if self._num_devices > 1:
            mesh = make_mesh(
                (self._num_devices,),
                axis_names=("obs",),
                axis_types=(AxisType.Explicit,),
                devices=local_devices(),
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
        ImpactModel._models.add(self)
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
            f"Fitted: {self._is_fitted}",
        ]
        temp_dir = getattr(self, "temp_dir", None)
        if temp_dir:
            out.append(f"Temporary directory: {temp_dir}")

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
            f"fitted={self._is_fitted};",
            f"temp_dir={getattr(self, 'temp_dir', None)!r}>",
        ]

        return " ".join(out)

    def __del__(self) -> None:
        """Clean up the temporary directory when the instance is garbage-collected."""
        # Module globals may already be torn down at interpreter shutdown
        try:
            self.cleanup()
        except AttributeError:
            return

    def __getstate__(self) -> dict:
        """Return the state of the object excluding runtime attributes.

        Returns:
            The state of the object, excluding runtime attributes.
        """
        # Typed keys anywhere in the state, including those nested in the inference
        # state, are pickled through `_reduce_key`.
        copyreg.pickle(type(random.key(0)), _reduce_key)
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
        # Models pickled before `_is_fitted` was initialized eagerly may lack it
        self.__dict__.setdefault("_is_fitted", False)
        # Models pickled before chains were kept stacked their draws as one chain
        self.__dict__.setdefault("_num_chains", 1)
        # Models pickled before dimension names were recorded keep the default names
        self.__dict__.setdefault("_dims", {})
        # Models pickled before site sizes were recorded count one value per observation
        self.__dict__.setdefault("_site_sizes", {})
        self._init_runtime_attrs()
        # A sampler pickled with parallel chains may be loaded on fewer devices
        inference = self._inference
        if (
            isinstance(inference, MCMC)
            and inference.chain_method == "parallel"
            and self._num_devices < inference.num_chains
        ):
            inference.chain_method = "sequential"
            msg = (
                "There are not enough devices to run parallel chains: expected "
                f"{inference.num_chains} but got {self._num_devices}. Chains will be "
                "drawn sequentially."
            )
            warn(
                msg, category=PerformanceWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES
            )

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
        """Path of the model's temporary directory, or ``None`` if none exists."""
        return self._temp_dir.name if self._temp_dir else None

    @property
    def vi_result(self) -> SVIRunResult | None:
        """Variational inference result, or ``None`` if not set.

        :setter: This sets :external:data:`~numpyro.infer.svi.SVIRunResult` without
            marking the model fitted. It does not perform posterior sampling; use
            :meth:`~aimz.ImpactModel.sample` separately to obtain samples.
        """
        return self._vi_result

    @vi_result.setter
    def vi_result(self, vi_result: SVIRunResult) -> None:
        """Set the variational inference result manually.

        Args:
            vi_result: The result from a prior variational inference run.
                It must be a NamedTuple or similar object with the following fields:
                - params: Learned parameters from inference.
                - state: Internal SVI state object.
                - losses: Loss values recorded during optimization.

        Note:
            This stores the result but does not mark the model fitted or draw
            posterior samples. Draw them with :meth:`~aimz.ImpactModel.sample` and
            register them via :meth:`~aimz.ImpactModel.set_posterior_sample`.
        """
        if not np.all(np.isfinite(vi_result.losses)):
            msg = "Loss contains NaN or Inf, indicating numerical instability."
            # A fixed stacklevel: skipping aimz frames would attribute this to MLflow's
            # autolog wrapper when it runs inside fit, and MLflow then hides it.
            warn(msg, category=FitWarning, stacklevel=2)
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
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument.
        """
        _group_kwargs(kwargs, forbid=(self.param_input, self.param_output))

        return signature(self.kernel).bind(**{self.param_input: X, **kwargs}).arguments

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
        observed output (``output_observed=True``). An upgrade re-trace merges the
        newly discovered sites with the existing spec, since a kernel may define
        different sites depending on whether data are supplied.

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
            *tuple(
                k
                for k, v in model_trace.items()
                if v["type"] == "deterministic" and k != self.param_output
            ),
        )
        # Name the dimensions of each site after its plates, outermost first, then the
        # event dimensions named in its `infer` dictionary. A site whose names are not
        # strings or outnumber its dimensions, or whose leading axes do not follow its
        # plates (a global quantity computed inside a plate, a site inside `scan`),
        # keeps the default names. The values a site holds per draw, per observation
        # when its leading axis follows the input's rows, size the streaming batches.
        rows = getattr(args_bound[self.param_input], "shape", ())[:1]
        for k, v in model_trace.items():
            if v["type"] not in {"sample", "deterministic"}:
                continue
            frames = sorted(v.get("cond_indep_stack") or (), key=lambda f: f.dim)
            event_dims = (v.get("infer") or {}).get("event_dims", ())
            names = (*(f.name for f in frames), *event_dims)
            shape = getattr(v["value"], "shape", ())
            if (
                0 < len(names) <= len(shape)
                and all(isinstance(name, str) for name in names)
                and shape[: len(frames)] == tuple(f.size for f in frames)
            ):
                self._dims[k] = names
            if hasattr(v["value"], "shape"):
                self._site_sizes[k] = (
                    (math.prod(shape[1:]), 0)
                    if shape[:1] == rows
                    else (0, math.prod(shape))
                )
        prev = self._kernel_spec
        if prev is not None and prev.traced:
            sample_sites = tuple(dict.fromkeys(prev.sample_sites + sample_sites))
            return_sites = tuple(dict.fromkeys(return_sites + prev.return_sites))
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
        """Return a normalized tuple of site names.

        Args:
            return_sites: User-provided site name(s) or ``None``.

        Returns:
            A tuple of site names.

        Warns:
            OutputWarning: If a requested site was not seen in any trace so far. The
                name is passed through, since a kernel may define it only at sampling
                time. A name absent from the forward trace is dropped from the output.
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
                msg = (
                    f"Return site(s) not seen in any trace so far: "
                    f"{', '.join(map(repr, unknown))}. They are passed through, but "
                    "will be missing from the output unless the kernel defines them "
                    "at sampling time."
                )
                warn(
                    msg, category=OutputWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES
                )

        return requested

    def _create_artifact_path(
        self,
        output_dir: str | Path | None,
    ) -> Path:
        """Create the artifact path for one disk-backed call.

        This function is called for its side effect: it creates a timestamped
        subdirectory within the specified output directory.

        Args:
            output_dir: Base directory where the output subdirectory will be created.

        Returns:
            The created call-specific artifact path (``<UTC-timestamp>_<caller>``
            under the resolved base directory); recorded on returned trees as the
            ``artifact_path`` attribute.
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
        artifact_path = output_dir / f"{timestamp}_{caller}"
        artifact_path.mkdir(parents=False, exist_ok=False)

        return artifact_path

    def _output_nbytes(self, return_sites: tuple[str, ...]) -> tuple[int, int]:
        """Return the bytes of output that the return sites hold in each draw.

        The value counts come from the kernel's trace, and each value counts at the
        width of the default float type, as the trace holds the observed data, whose
        type the predictive output need not share. A site the trace did not record
        counts as one value per observation.

        Args:
            return_sites: Names of the return sites.

        Returns:
            The bytes per observation, and those of the sites without an observation
            axis.
        """
        itemsize = jnp.result_type(float).itemsize
        sizes = [self._site_sizes.get(site, (1, 0)) for site in return_sites]

        return itemsize * sum(n for n, _ in sizes), itemsize * sum(n for _, n in sizes)

    def _plan_execution(
        self,
        X: ArrayLike | ArrayLoader | Iterable[Mapping[str, Array | np.ndarray]],
        *,
        shard_axis: Literal["obs", "draw"],
        batch_size: int | None,
        num_samples: int,
        nbytes: tuple[int, int],
        posterior: dict[str, Array] | None,
    ) -> tuple[Literal["obs", "draw"], int | None]:
        """Choose the sharding strategy and the batch size of a streamed call.

        A posterior site shaped ``(num_samples, n_obs, ...)`` indexes the observation
        axis of ``X``; data-parallel streaming splits that axis across devices or into
        batches, which would sever such a site from the observations it indexes. Only
        a single whole-input batch on a single device keeps it intact, so the call
        otherwise runs draw-parallel, with a warning. A requested observation batch that
        does not divide among the devices also warns. An automatic batch size keeps the
        output of each batch, counted over every return site, within the memory budget.

        Args:
            X: Input array with observations on the leading axis, or a data loader
                yielding batch mappings keyed by kernel parameter names.
            shard_axis: The requested sharding strategy.
            batch_size: The requested batch size, or ``None`` to choose it.
            num_samples: Number of draws the call produces.
            nbytes: Bytes of output in each draw, per observation and for the sites
                without an observation axis, as :meth:`_output_nbytes` returns them.
            posterior: The posterior samples the call conditions on, if any.

        Returns:
            The sharding strategy and the batch size, which is ``None`` for a data
            loader, as it batches itself.
        """
        if not isinstance(X, ArrayLike):
            return shard_axis, None

        n_obs = len(cast("Sized", X))
        row_nbytes, rest_nbytes = nbytes
        min_aligned_ndim = 2
        if shard_axis == "obs" and any(
            v.ndim >= min_aligned_ndim and v.shape[1] == n_obs
            for v in (posterior or {}).values()
        ):
            if (
                self._num_devices == 1
                and batch_size is None
                and _fits_single_batch(n_obs, item_nbytes=num_samples * row_nbytes)
            ):
                batch_size = n_obs
            elif self._num_devices > 1 or batch_size is None or batch_size < n_obs:
                msg = (
                    "One or more posterior sample shapes are not compatible with "
                    "`shard_axis='obs'`; rerunning with `shard_axis='draw'`. Pass "
                    "`shard_axis='draw'` to silence this warning."
                )
                warn(
                    msg,
                    category=PerformanceWarning,
                    skip_file_prefixes=_SKIP_FILE_PREFIXES,
                )
                shard_axis, batch_size = "draw", None
        if (
            shard_axis == "obs"
            and batch_size is not None
            and batch_size % self._num_devices
        ):
            msg = (
                f"The `batch_size` ({batch_size}) is not divisible by the number of "
                f"devices ({self._num_devices}). Use a multiple of {self._num_devices} "
                "for optimal performance."
            )
            warn(
                msg, category=PerformanceWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES
            )

        resolved = (
            _resolve_batch_size(
                batch_size,
                axis_size=n_obs,
                other_size=num_samples,
                num_devices=self._num_devices,
                item_nbytes=num_samples * row_nbytes,
            )
            if shard_axis == "obs"
            else _resolve_batch_size(
                batch_size,
                axis_size=num_samples,
                other_size=n_obs,
                num_devices=self._num_devices,
                item_nbytes=n_obs * row_nbytes + rest_nbytes,
            )
        )
        if batch_size is None:
            logger.debug("Resolved batch_size=%d automatically.", resolved)

        return shard_axis, resolved

    def _stream_to_datatree(
        self,
        write: Callable[[Path | None], dict[str, DaskArray] | None],
        *,
        store: str,
        output_dir: str | Path | None,
        group: str,
    ) -> xr.DataTree:
        """Run one streamed write and assemble its output tree.

        The single place where the result store forks: ``store="persistent"`` creates
        the call-specific artifact path, streams into it (removing it again if the
        stream fails), and reads it back lazily; ``store="memory"`` retains the
        streamed batches in host memory and assembles the tree over them directly.

        Args:
            write: Runs the streamed write against the given artifact path (``None``
                for in-memory accumulation) and returns the accumulated site arrays,
                if any.
            store: The result store, ``"persistent"`` or ``"memory"``.
            output_dir: Base directory for disk-backed outputs.
            group: Output group name for the resulting tree.

        Returns:
            The output tree: lazy in both modes, Dask-backed by the Zarr store for
            ``"persistent"`` and by the retained in-memory batches for ``"memory"``.
        """
        artifact_path = (
            self._create_artifact_path(output_dir) if store == "persistent" else None
        )
        try:
            result = write(artifact_path)
            dt = _build_datatree(
                (
                    artifact_path
                    if artifact_path is not None
                    else cast("dict[str, DaskArray]", result)
                ),
                group=group,
                posterior=self.posterior,
                num_chains=self._num_chains,
                dims=self._dims,
            )
        except BaseException:
            if artifact_path is not None:
                rmtree(artifact_path, ignore_errors=True)
            raise

        return dt

    def sample_prior_predictive_on_batch(
        self,
        X: ArrayLike,
        *,
        intervention: dict | None = None,
        num_samples: int = 1000,
        rng_key: Array | None = None,
        return_sites: str | Iterable[str] | None = None,
        return_datatree: bool = True,
        **kwargs: object,
    ) -> xr.DataTree | dict[str, npt.NDArray]:
        """Draw samples from the prior predictive distribution.

        Args:
            X: Input array with observations on the leading axis.
            intervention: A dictionary mapping sample site names to replacement values
                used during predictive sampling. No intervention is applied if ``None``.
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
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument.
            ValueError: If ``intervention`` names a site that is not a sample site of
                the kernel.

        See Also:
            :meth:`~aimz.ImpactModel.sample_prior_predictive`
        """
        X = cast("Array", _validate_X_y_to_jax(X))

        args_bound = self._bind_kernel_args(X, kwargs=kwargs)
        self._build_kernel_spec(args_bound, with_output=False)
        _validate_intervention(intervention, kernel_spec=self._kernel_spec)

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        prior_predictive_samples = device_get(
            _sample_forward(
                self.kernel,
                rng_keys=random.split(rng_key, num=num_samples),
                return_sites=self._coerce_return_sites(return_sites),
                samples=None,
                intervention=intervention,
                model_kwargs=args_bound,
            ),
        )

        if not return_datatree:
            return prior_predictive_samples

        return _build_datatree(
            prior_predictive_samples,
            group="prior_predictive",
            posterior=self.posterior,
            num_chains=self._num_chains,
            dims=self._dims,
        )

    def sample_prior_predictive(
        self,
        X: ArrayLike | ArrayLoader | Iterable[Mapping[str, Array | np.ndarray]],
        *,
        intervention: dict | None = None,
        num_samples: int = 1000,
        rng_key: Array | None = None,
        return_sites: str | Iterable[str] | None = None,
        shard_axis: Literal["obs", "draw"] = "obs",
        batch_size: int | None = None,
        store: Literal["persistent", "memory"] = "persistent",
        output_dir: str | Path | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> xr.DataTree:
        """Draw samples from the prior predictive distribution.

        Results are written to disk in the Zarr format, with computing and file writing
        decoupled and executed concurrently. Pass ``store="memory"`` to accumulate the
        results in host memory instead.

        Args:
            X: Input array with observations on the leading axis, or a data loader
                yielding batch mappings keyed by kernel parameter names.
            intervention: A dictionary mapping sample site names to replacement values
                used during predictive sampling. No intervention is applied if ``None``.
            num_samples: The number of samples to draw.
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed.
            return_sites: Names of variables (sites) to return. If ``None``, samples
                :attr:`~aimz.ImpactModel.param_output` and deterministic sites.
            shard_axis: Multi-device sharding strategy. ``"obs"`` (default) shards the
                input across devices and replicates the drawn samples. ``"draw"`` shards
                the drawn samples across devices and replicates the input, which must be
                an array, not a data loader.
            batch_size: Size of each batch, taken from the input under
                ``shard_axis="obs"`` and from the draws under ``shard_axis="draw"``.
                Also used as the chunk size when storing results. If ``None``, it is
                chosen automatically. Ignored for an existing data loader.
            store: Where results accumulate. ``"persistent"`` (default) streams
                batches to a
                Zarr store under ``output_dir`` and returns a lazy, Dask-backed tree
                recording its ``artifact_path`` attribute. ``"memory"`` retains the
                batches in host memory as the returned tree's chunks.
            output_dir: The directory where the outputs will be saved. If the specified
                directory does not exist, it will be created automatically. If ``None``,
                a model-owned temporary directory is used. A subdirectory is generated
                within this directory to store the outputs; its path is recorded in
                the returned tree's ``artifact_path`` attribute (on the root and the
                group node). The temporary directory is removed by
                :meth:`~aimz.ImpactModel.cleanup`,
                :meth:`~aimz.ImpactModel.cleanup_models`, or when the model is
                garbage-collected. Pass an explicit ``output_dir`` to keep results
                beyond the model's lifetime.
            progress: Whether to display a progress bar.
            **kwargs: Additional arguments passed to the model. Arrays aligned with
                ``X`` are batched with it; other values are passed whole, arrays
                traced and the rest static.

        Returns:
            Prior predictive samples. Posterior samples are included if available.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument, or
                ``shard_axis="draw"`` is used with a data loader ``X``.
            ValueError: If ``shard_axis`` is not ``"obs"`` or ``"draw"``, ``store`` is
                not ``"persistent"`` or ``"memory"``, ``output_dir`` is passed with
                ``store="memory"``, ``X`` is 0-D or empty, or ``intervention`` names
                a site that is not a sample site of the kernel.
            NotImplementedError: If a return site's axis-1 size does not match the
                input batch size (``shard_axis="obs"`` only).

        See Also:
            :meth:`~aimz.ImpactModel.sample_prior_predictive_on_batch` for a
            single-batch, in-memory alternative.

            :meth:`~aimz.ImpactModel.cleanup` to remove the temporary directory if
            created.
        """
        _validate_shard_axis(shard_axis, X=X)
        _validate_batch_size(batch_size, X=X)
        _validate_store(store, output_dir=output_dir)
        _validate_aligned_inputs(X, y=None)

        stream = None
        if isinstance(X, ArrayLike):
            args_bound = self._bind_kernel_args(X, kwargs=kwargs)
            obs_names = {
                self.param_input,
                *_group_kwargs(kwargs, n_obs=np.shape(X)[0])[0],
            }
        else:
            stream = self._streamer.setup_stream(
                _WriteRequest(
                    shard_axis,
                    X=X,
                    return_sites=(),
                    num_samples=num_samples,
                    batch_size=batch_size,
                    artifact_path=None,
                    progress=progress,
                    loader_rng_key=self.rng_key,
                    kwargs=kwargs,
                ),
                y=None,
            )
            batch = dict(stream[2])
            batch.pop(self.param_output, None)
            args_bound = self._bind_kernel_args(
                batch.pop(self.param_input),
                kwargs={**kwargs, **batch},
            )
            obs_names = {self.param_input, *batch}

        # Build the kernel spec from a single-row slice so the trace runs on a tiny
        # input. Only when the spec is not already cached (fitted models keep theirs),
        # and before the return-site defaults resolve so they work even before fitting.
        if not (self._kernel_spec and self._kernel_spec.traced):
            # Slice only the per-observation arguments; call constants stay whole
            probe = {
                k: (cast("Any", v)[:1] if k in obs_names else v)
                for k, v in args_bound.items()
            }
            self._build_kernel_spec(probe, with_output=False)
        _validate_intervention(intervention, kernel_spec=self._kernel_spec)

        return_sites = self._coerce_return_sites(return_sites)
        shard_axis, batch_size = self._plan_execution(
            X,
            shard_axis=shard_axis,
            batch_size=batch_size,
            num_samples=num_samples,
            nbytes=self._output_nbytes(return_sites),
            posterior=None,
        )

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        return self._stream_to_datatree(
            lambda artifact_path: self._streamer.write_predictive(
                _WriteRequest(
                    shard_axis,
                    X=X,
                    return_sites=return_sites,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    artifact_path=artifact_path,
                    progress=progress,
                    loader_rng_key=self.rng_key,
                    kwargs=kwargs,
                    dims=self._dims,
                ),
                kernel=self.kernel,
                rng_key=rng_key,
                group="prior_predictive",
                posterior=self.posterior,
                intervention=intervention,
                stream=stream,
            ),
            store=store,
            output_dir=output_dir,
            group="prior_predictive",
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
        """Draw posterior samples from the model's inference state.

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
            NotFittedError: If there is no inference state to sample from (no
                completed MCMC run, or no ``vi_result`` when the inference method
                is SVI).
            TypeError: If :attr:`~aimz.ImpactModel.param_output` is not passed as an
                argument when the inference method is MCMC.
        """
        if isinstance(self.inference, MCMC):
            if self.inference.last_state is None:
                msg = (
                    "This ImpactModel instance has no MCMC state to sample from. "
                    "Call `.fit_on_batch()`, or run the MCMC directly, before using "
                    "`.sample()`."
                )
                raise NotFittedError(msg)
            # Validate the provided parameters against the kernel's signature
            args_bound = signature(self.kernel).bind(**kwargs).arguments
            if self.param_output not in args_bound:
                msg = f"{self.param_output!r} must be provided in `.sample()`."
                raise TypeError(msg)
            # Continue the chain for this call only, so later fits keep their own warmup
            # and number of samples
            saved = self.inference.post_warmup_state, self.inference.num_samples
            self.inference.post_warmup_state = self.inference.last_state
            self.inference.num_samples = num_samples
            try:
                self.inference.run(
                    self.inference.post_warmup_state.rng_key,
                    **args_bound,
                )
            finally:
                self.inference.post_warmup_state, self.inference.num_samples = saved
                # Recompute the collection bounds from the restored number of samples,
                # as each run does at its end
                self.inference._set_collection_params()
            posterior_samples = device_get(self.inference.get_samples())
        else:
            if self.vi_result is None:
                msg = (
                    "This ImpactModel instance has no inference result to sample from. "
                    "Call `.fit()` or `.fit_on_batch()`, or set `vi_result`, before "
                    "using `.sample()`."
                )
                raise NotFittedError(msg)
            if rng_key is None:
                self._rng_key, rng_key = random.split(self._rng_key)
            posterior_samples = device_get(
                _sample_forward(
                    substitute(
                        self.inference.guide,
                        data=self.vi_result.params,
                    ),
                    rng_keys=random.split(rng_key, num=num_samples),
                    return_sites=self._coerce_return_sites(return_sites)
                    if return_sites is not None
                    else None,
                    samples=None,
                    intervention=None,
                    model_kwargs=None,
                ),
            )

        if not return_datatree:
            return posterior_samples

        return _build_datatree(
            posterior_samples,
            group="posterior",
            num_chains=(
                self.inference.num_chains if isinstance(self.inference, MCMC) else 1
            ),
            dims=self._dims,
        )

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
            X: Input array with observations on the leading axis.
            intervention: A dictionary mapping sample site names to replacement values
                used during predictive sampling. No intervention is applied if ``None``.
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed.
            return_sites: Names of variables (sites) to return. If ``None``, samples
                :attr:`~aimz.ImpactModel.param_output` and deterministic sites.
            return_datatree: If ``True``, return a :external:class:`~xarray.DataTree`;
                otherwise return a :class:`dict`.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Posterior predictive samples. Posterior samples are included if available
            when a :external:class:`~xarray.DataTree` is returned.

        Raises:
            NotFittedError: If the model is not fitted.
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument.
            ValueError: If ``intervention`` names a site that is not a sample site of
                the kernel.

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
        X: ArrayLike | ArrayLoader | Iterable[Mapping[str, Array | np.ndarray]],
        *,
        intervention: dict | None = None,
        rng_key: Array | None = None,
        return_sites: str | Iterable[str] | None = None,
        shard_axis: Literal["obs", "draw"] = "obs",
        batch_size: int | None = None,
        store: Literal["persistent", "memory"] = "persistent",
        output_dir: str | Path | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> xr.DataTree:
        """Draw samples from the posterior predictive distribution.

        This method is a convenience alias for :meth:`~aimz.ImpactModel.predict`, with
        ``in_sample`` automatically set to ``True``.

        Args:
            X: Input array with observations on the leading axis, or a data loader
                yielding batch mappings keyed by kernel parameter names.
            intervention: A dictionary mapping sample site names to replacement values
                used during predictive sampling. No intervention is applied if ``None``.
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed.
            return_sites: Names of variables (sites) to return. If ``None``, samples
                :attr:`~aimz.ImpactModel.param_output` and deterministic sites.
            shard_axis: Multi-device sharding strategy.
                ``"obs"`` (default) shards the input across devices and replicates the
                posterior. ``"draw"`` shards the posterior across devices and replicates
                the input, which must be an array, not a data loader. If the model has
                no posterior samples, the data path is used regardless of
                ``shard_axis``.
            batch_size: Size of each batch, taken from the input under
                ``shard_axis="obs"`` and from the draws under ``shard_axis="draw"``.
                Also used as the chunk size when storing results. If ``None``, it is
                chosen automatically.
                Ignored if ``X`` is a data loader, in which case the data loader is
                expected to handle batching internally.
            store: Where results accumulate. ``"persistent"`` (default) streams
                batches to a
                Zarr store under ``output_dir`` and returns a lazy, Dask-backed tree
                recording its ``artifact_path`` attribute. ``"memory"`` retains the
                batches in host memory as the returned tree's chunks.
            output_dir: The directory where the outputs will be saved. If the specified
                directory does not exist, it will be created automatically. If ``None``,
                a model-owned temporary directory is used. A subdirectory is generated
                within this directory to store the outputs; its path is recorded in
                the returned tree's ``artifact_path`` attribute (on the root and the
                group node). The temporary directory is removed by
                :meth:`~aimz.ImpactModel.cleanup`,
                :meth:`~aimz.ImpactModel.cleanup_models`, or when the model is
                garbage-collected. Pass an explicit ``output_dir`` to keep results
                beyond the model's lifetime.
            progress: Whether to display a progress bar.
            **kwargs: Additional arguments passed to the model. Arrays aligned with
                ``X`` are batched with it; other values are passed whole, arrays
                traced and the rest static.

        Returns:
            Posterior predictive samples. Posterior samples are included if available.

        Raises:
            NotFittedError: If the model is not fitted.
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument, or
                ``shard_axis="draw"`` is used with a data loader ``X``.
            ValueError: If ``shard_axis`` is not ``"obs"`` or ``"draw"``, ``store`` is
                not ``"persistent"`` or ``"memory"``, ``output_dir`` is passed with
                ``store="memory"``, ``X`` is 0-D or empty, or ``intervention`` names
                a site that is not a sample site of the kernel.
            NotImplementedError: If a return site's axis-1 size does not match the
                input batch size (``shard_axis="obs"`` only).

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
            store=store,
            output_dir=output_dir,
            progress=progress,
            **kwargs,
        )

    def train_on_batch(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        rng_key: Array | None = None,
        **kwargs: object,
    ) -> tuple[SVIState, Array]:
        """Run a single VI step on the given batch of data.

        Args:
            X: Input array with observations on the leading axis.
            y: Output array with observations on the leading axis.
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed. The key is only used for initialization if
                the internal SVI state is not yet set.
            **kwargs: Additional arguments passed to the model.

        Returns:
            - Updated SVI state after the training step.

            - Loss value as a scalar array.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument.

        Note:
            This method updates the internal SVI state on every call, so it is not
            necessary to capture the returned state externally unless explicitly needed.
            However, the returned loss value can be used for monitoring or logging.
        """
        _group_kwargs(kwargs, forbid=(self.param_input, self.param_output))
        batch = {self.param_input: X, self.param_output: y, **kwargs}

        if self._vi_state is None:
            self._build_kernel_spec(
                signature(self.kernel).bind(**batch).arguments,
                with_output=True,
            )
            if rng_key is None:
                self._rng_key, rng_key = random.split(self._rng_key)
            self._vi_state = cast("SVI", self.inference).init(rng_key, **batch)
        # Trace the array leaves and hold every other value static, so arguments such
        # as integers and strings can set shapes or drive control flow in the kernel.
        if self._fn_vi_update is None:
            svi = cast("SVI", self.inference)

            def update(
                state: SVIState,
                leaves: list,
                static: tuple,
            ) -> tuple[SVIState, Array]:
                return svi.update(state, **_combine(leaves, static))

            self._fn_vi_update = jit(update, static_argnums=2)
        leaves, static = _partition(batch)
        self._vi_state, loss = self._fn_vi_update(self._vi_state, leaves, static)

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
        initialization:

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
            X: Input array with observations on the leading axis.
            y: Output array with observations on the leading axis.
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

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument.
            ValueError: If ``y`` does not share ``X``'s leading-axis size.

        Note:
            This method continues training from the existing SVI state if available. To
            start training from scratch, create a new model instance.
        """
        X, y = _validate_X_y_to_jax(X, y=y)

        _group_kwargs(kwargs, forbid=(self.param_input, self.param_output))
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
            logger.info("Performing variational inference optimization")
            vi_result = self.inference.run(
                rng_subkey,
                num_steps=num_steps,
                progress_bar=progress,
                init_state=self._vi_state,
                **args_bound,
            )
            self.vi_result: SVIRunResult = SVIRunResult(
                params=vi_result.params,
                state=vi_result.state,
                losses=device_get(vi_result.losses),
            )
            self._vi_state = self.vi_result.state

            # The draw below rejects diverged guide parameters; clear the previous
            # fit first so a failed refit does not keep a stale posterior.
            self._is_fitted = False
            self._posterior = None
            logger.info("Drawing posterior samples (num_samples=%d)", num_samples)
            rng_key, rng_subkey = random.split(rng_key)
            self._posterior = _sample_forward(
                substitute(self.inference.guide, data=self.vi_result.params),
                rng_keys=random.split(rng_subkey, num=num_samples),
                return_sites=None,
                samples=None,
                intervention=None,
                model_kwargs=None,
            )
            self._num_samples = num_samples
        elif isinstance(self.inference, MCMC):
            logger.info(
                "Drawing posterior samples (num_samples=%d)",
                self.inference.num_samples,
            )
            self.inference.run(rng_subkey, **args_bound)
            self._posterior = device_get(self.inference.get_samples())
            self._num_chains = self.inference.num_chains
            self._num_samples = (
                next(iter(self.posterior.values())).shape[0] if self.posterior else 0
            )
        self._is_fitted = True

        return self

    def fit(
        self,
        X: ArrayLike | ArrayLoader | Iterable[Mapping[str, Array | np.ndarray]],
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
            X: Input array with observations on the leading axis, or a data loader
                yielding batch mappings keyed by kernel parameter names, including the
                observed output. A data loader is iterated once per epoch.
            y: Output array with observations on the leading axis. Must be ``None``
                if ``X`` is a data loader.
            num_samples: The number of posterior samples to draw.
            rng_key: A pseudo-random number generator key. By default, an internal key
                is used and split as needed.
            progress: Whether to display a progress bar and the average loss of each
                epoch.
            batch_size: The number of data points processed at each step of variational
                inference. If ``None``, the entire dataset is used as a single batch in
                each epoch. Ignored if ``X`` is a data loader, in which case the data
                loader is expected to handle batching internally.
            epochs: The number of epochs for variational inference optimization.
            shuffle: Whether to shuffle the data at each epoch. Ignored if ``X`` is a
                data loader.
            **kwargs: Additional arguments passed to the model. Arrays aligned with
                ``X`` are batched with it; other values are passed whole, arrays
                traced and the rest static.

        Returns:
            The fitted model instance, enabling method chaining.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument, the
                inference method is MCMC, or ``y`` is passed with a data loader ``X``.
            ValueError: If ``y`` is missing when ``X`` is array-like, ``y`` does not
                share ``X``'s leading-axis size, or a data loader yields no batches
                in an epoch.

        Note:
            This method continues training from the existing SVI state if available.
            To start training from scratch, create a new model instance. It does not
            check whether the model or guide is written to support subsampling semantics
            (e.g., using `NumPyro`_'s :external:func:`~numpyro.primitives.subsample` or
            similar constructs).
        """
        _validate_aligned_inputs(X, y=y)
        if y is None and isinstance(X, ArrayLike):
            msg = (
                "`y` is required for `fit()` when `X` is array-like. "
                "Provide `y`, or give `X` as a data loader that carries it."
            )
            raise ValueError(msg)
        if isinstance(self.inference, MCMC):
            msg = (
                "`.fit()` is not supported for MCMC inference. Use `.fit_on_batch()` "
                "instead."
            )
            raise TypeError(msg)

        rng_key_model = self._rng_key
        if rng_key is None:
            rng_key_model, rng_key = random.split(self._rng_key)

        rng_key, rng_subkey = random.split(rng_key)
        dataloader, kwargs_extra = _setup_inputs(
            X=X,
            y=y,
            param_input=self.param_input,
            param_output=self.param_output,
            rng_key=rng_subkey,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )
        # Validate the provided parameters against the kernel's signature; the fields
        # of a data loader may supply the remaining ones
        names = {self.param_input, self.param_output}
        if isinstance(X, ArrayLoader) and (missing := names - X.dataset.arrays.keys()):
            msg = f"The data loader has no field named {min(missing)!r}."
            raise ValueError(msg)
        if isinstance(X, ArrayLike):
            signature(self.kernel).bind(
                **{self.param_input: X, self.param_output: y, **kwargs},
            )
        else:
            signature(self.kernel).bind_partial(**kwargs)
        # Commit model state only once the inputs are accepted
        self._rng_key = rng_key_model

        logger.info("Performing variational inference optimization")
        losses: list[npt.NDArray] = []
        rng_key, rng_subkey = random.split(rng_key)
        for epoch in range(epochs):
            losses_epoch: list[npt.NDArray] = []
            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                # Other loaders need not define a length
                total=len(dataloader) if isinstance(dataloader, ArrayLoader) else None,
                disable=not progress,
                dynamic_ncols=True,
            )
            for batch in pbar:
                fields = dict(batch)
                _, loss = self.train_on_batch(
                    fields.pop(self.param_input),
                    fields.pop(self.param_output),
                    **fields,
                    **kwargs_extra,
                    rng_key=rng_subkey,
                )
                loss_batch = device_get(loss)
                losses_epoch.append(loss_batch)
                pbar.set_postfix({"loss": f"{float(loss_batch):.4f}"})
            if not losses_epoch:
                # An exhausted one-shot iterator would otherwise skip the epoch silently
                msg = (
                    f"The data loader yielded no batches in epoch {epoch + 1}; pass a "
                    "loader that can be iterated once per epoch."
                )
                raise ValueError(msg)
            # Host-side bookkeeping: the per-step losses are already on the host, so
            # stacking or accumulating them as device arrays would only add
            # host-to-device round trips and retain one device scalar per step.
            losses.extend(losses_epoch)
            if progress:
                tqdm.write(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Average loss: {float(np.mean(losses_epoch)):.4f}",
                )
        self.vi_result = SVIRunResult(
            params=self.inference.get_params(self._vi_state),
            state=self._vi_state,
            losses=np.asarray(losses),
        )

        # The draw below rejects diverged guide parameters; clear the previous
        # fit first so a failed refit does not keep a stale posterior.
        self._is_fitted = False
        self._posterior = None
        logger.info("Drawing posterior samples (num_samples=%d)", num_samples)
        rng_key, rng_subkey = random.split(rng_key)
        self._posterior = _sample_forward(
            substitute(
                self.inference.guide,
                data=cast("SVIRunResult", self.vi_result).params,
            ),
            rng_keys=random.split(rng_subkey, num=num_samples),
            return_sites=None,
            samples=None,
            intervention=None,
            model_kwargs=None,
        )
        self._num_samples = num_samples
        self._is_fitted = True

        return self

    def is_fitted(self) -> bool:
        """Check fitted status.

        Returns:
            ``True`` if the model is fitted, ``False`` otherwise.

        """
        return self._is_fitted

    def set_posterior_sample(
        self,
        posterior_sample: dict[str, Array],
        *,
        num_chains: int = 1,
    ) -> Self:
        """Set posterior samples for the model.

        This method sets externally obtained posterior samples on the model instance,
        enabling downstream analysis without requiring a call to
        :meth:`~aimz.ImpactModel.fit` or :meth:`~aimz.ImpactModel.fit_on_batch`.

        It is primarily intended for workflows where posterior sampling is performed
        manually, for example, using `NumPyro`_'s
        :external:class:`~numpyro.infer.svi.SVI` (or
        :external:class:`~numpyro.infer.mcmc.MCMC`) with the
        :external:class:`~numpyro.infer.util.Predictive` API, and the resulting
        posterior samples are injected into the model for further use.

        Internally, ``batch_ndims`` is set to ``1`` by default to correctly handle the
        batch dimensions of the posterior samples. For more information, refer to the
        `NumPyro documentation <https://num.pyro.ai/en/stable/utilities.html#predictive>`__.

        Args:
            posterior_sample: Posterior samples to set for the model, with the draws
                along the leading axis of each array.
            num_chains: Number of chains the draws are stacked from, chain by chain.
                Output trees then keep the chains along their ``chain`` dimension.

        Returns:
            The model instance, treated as fitted with posterior samples set, enabling
            method chaining.

        Raises:
            ValueError: If ``posterior_sample`` is empty, a value is 0-D, or the batch
                shapes in ``posterior_sample`` are inconsistent.

        Note:
            The kernel is not traced when samples are set this way, so ``deterministic``
            sites are not discovered; predictive methods return only the output site by
            default. Request ``deterministic`` (or other) sites explicitly via
            ``return_sites``.

            Include only latent sample sites. The output site is removed with a
            warning, but ``deterministic`` sites must be excluded by the caller:
            if present, they override the values recomputed by
            :meth:`~aimz.ImpactModel.log_likelihood`.
        """
        if self.param_output in posterior_sample:
            posterior_sample = {
                k: v for k, v in posterior_sample.items() if k != self.param_output
            }
            msg = f"The output site {self.param_output!r} is removed."
            warn(msg, category=AimzWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
        batch_ndims = 1
        posterior_sample = {
            name: sample
            if isinstance(sample, (Array, np.ndarray))
            else np.asarray(sample)
            for name, sample in posterior_sample.items()
        }
        for name, sample in posterior_sample.items():
            if sample.ndim < batch_ndims:
                msg = f"`posterior_sample[{name!r}]` must have at least 1 dimension."
                raise ValueError(msg)
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
        self._num_chains = num_chains
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

            2) Scenarios where writing results to files (e.g., disk, cloud storage)
            is not desired.

            3) Smaller datasets, as this method may be slower due to limited
            parallelism.

        Args:
            X: Input array with observations on the leading axis.
            intervention: A dictionary mapping sample site names to replacement values
                used during predictive sampling. No intervention is applied if ``None``.
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
            Posterior predictive samples. Posterior samples are included if available
            when a :external:class:`~xarray.DataTree` is returned; the dictionary
            contains only the requested sites.

        Raises:
            NotFittedError: If the model is not fitted.
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument.
            ValueError: If ``intervention`` names a site that is not a sample site of
                the kernel.
        """
        _check_is_fitted(self)
        _validate_intervention(intervention, kernel_spec=self._kernel_spec)

        X = cast("Array", _validate_X_y_to_jax(X))

        args_bound = self._bind_kernel_args(X, kwargs=kwargs)

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        samples = device_get(
            _sample_forward(
                self.kernel,
                rng_keys=random.split(rng_key, num=self._num_samples),
                return_sites=self._coerce_return_sites(return_sites),
                samples=self._streamer.place_posterior(self.posterior, sharding=None),
                intervention=intervention,
                model_kwargs=args_bound,
            ),
        )

        if not return_datatree:
            return samples

        return _build_datatree(
            samples,
            group="posterior_predictive" if in_sample else "predictions",
            posterior=self.posterior,
            num_chains=self._num_chains,
            dims=self._dims,
        )

    def predict(
        self,
        X: ArrayLike | ArrayLoader | Iterable[Mapping[str, Array | np.ndarray]],
        *,
        intervention: dict | None = None,
        rng_key: Array | None = None,
        in_sample: bool = True,
        return_sites: str | Iterable[str] | None = None,
        shard_axis: Literal["obs", "draw"] = "obs",
        batch_size: int | None = None,
        store: Literal["persistent", "memory"] = "persistent",
        output_dir: str | Path | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> xr.DataTree:
        """Predict the output based on the fitted model.

        This method performs posterior predictive sampling to generate model-based
        predictions. It is optimized for batch processing of large input data and is not
        recommended for use in loops that process only a few inputs at a time. Results
        are written to disk in the Zarr format, with sampling and file writing decoupled
        and executed concurrently. Pass ``store="memory"`` to accumulate the results in
        host memory instead.

        Args:
            X: Input array with observations on the leading axis, or a data loader
                yielding batch mappings keyed by kernel parameter names.
            intervention: A dictionary mapping sample site names to replacement values
                used during predictive sampling. No intervention is applied if ``None``.
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
            shard_axis: Multi-device sharding strategy.
                ``"obs"`` (default) shards the input across devices and replicates the
                posterior. ``"draw"`` shards the posterior across devices and replicates
                the input, which must be an array, not a data loader. If the model has
                no posterior samples, the data path is used regardless of
                ``shard_axis``.
            batch_size: Size of each batch, taken from the input under
                ``shard_axis="obs"`` and from the draws under ``shard_axis="draw"``.
                Also used as the chunk size when storing results. If ``None``, it is
                chosen automatically.
                Ignored if ``X`` is a data loader, in which case the data loader is
                expected to handle batching internally.
            store: Where results accumulate. ``"persistent"`` (default) streams
                batches to a
                Zarr store under ``output_dir`` and returns a lazy, Dask-backed tree
                recording its ``artifact_path`` attribute. ``"memory"`` retains the
                batches in host memory as the returned tree's chunks.
            output_dir: The directory where the outputs will be saved. If the specified
                directory does not exist, it will be created automatically. If ``None``,
                a model-owned temporary directory is used. A subdirectory is generated
                within this directory to store the outputs; its path is recorded in
                the returned tree's ``artifact_path`` attribute (on the root and the
                group node). The temporary directory is removed
                by :meth:`~aimz.ImpactModel.cleanup`,
                :meth:`~aimz.ImpactModel.cleanup_models`, or when the model is
                garbage-collected. Pass an explicit ``output_dir`` to keep results
                beyond the model's lifetime.
            progress: Whether to display a progress bar.
            **kwargs: Additional arguments passed to the model. Arrays aligned with
                ``X`` are batched with it; other values are passed whole, arrays
                traced and the rest static.

        Returns:
            Posterior predictive samples. Posterior samples are included if available.

        Raises:
            NotFittedError: If the model is not fitted.
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument, or
                ``shard_axis="draw"`` is used with a data loader ``X``.
            ValueError: If ``shard_axis`` is not ``"obs"`` or ``"draw"``, ``store`` is
                not ``"persistent"`` or ``"memory"``, ``output_dir`` is passed with
                ``store="memory"``, ``X`` is 0-D or empty, or ``intervention`` names
                a site that is not a sample site of the kernel.
            NotImplementedError: If a return site's axis-1 size does not match the
                input batch size (``shard_axis="obs"`` only).

        See Also:
            :meth:`~aimz.ImpactModel.cleanup` to remove the temporary directory if
            created.
        """
        _check_is_fitted(self)
        _validate_intervention(intervention, kernel_spec=self._kernel_spec)
        _validate_shard_axis(shard_axis, X=X)
        _validate_batch_size(batch_size, X=X)
        _validate_store(store, output_dir=output_dir)
        _validate_aligned_inputs(X, y=None)
        # No posterior to shard means draw-parallel has nothing to chunk, so it behaves
        # identically to the data-parallel path.
        if not self.posterior:
            shard_axis = "obs"

        if isinstance(X, ArrayLike):
            _ = self._bind_kernel_args(X, kwargs=kwargs)

        return_sites = self._coerce_return_sites(return_sites)
        shard_axis, batch_size = self._plan_execution(
            X,
            shard_axis=shard_axis,
            batch_size=batch_size,
            num_samples=self._num_samples,
            nbytes=self._output_nbytes(return_sites),
            posterior=self.posterior,
        )

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        group = "posterior_predictive" if in_sample else "predictions"

        return self._stream_to_datatree(
            lambda artifact_path: self._streamer.write_predictive(
                _WriteRequest(
                    shard_axis,
                    X=X,
                    return_sites=return_sites,
                    num_samples=self._num_samples,
                    batch_size=batch_size,
                    artifact_path=artifact_path,
                    progress=progress,
                    loader_rng_key=self.rng_key,
                    kwargs=kwargs,
                    dims=self._dims,
                ),
                kernel=self.kernel,
                rng_key=rng_key,
                group=group,
                posterior=self.posterior,
                intervention=intervention,
            ),
            store=store,
            output_dir=output_dir,
            group=group,
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
        predictive group, preserving sampling (chain/draw) dimensions. Precomputed
        outputs of :meth:`~aimz.ImpactModel.sample_prior_predictive` give the effect
        under the prior, also for an unfitted model; pass the same ``rng_key`` to both
        calls so the scenarios share their prior draws.

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
            available, except in an effect under the prior. When a scenario's output
            was streamed to disk, the effect tree records that scenario's
            call-specific artifact path in an ``artifact_path_baseline`` /
            ``artifact_path_intervention`` root attribute; in-memory results set
            neither.

        Raises:
            NotFittedError: If the model is not fitted and ``output_baseline`` or
                ``output_intervention`` is not provided.
            ValueError: If neither ``output_baseline`` nor ``args_baseline`` is
                provided, if neither ``output_intervention`` nor
                ``args_intervention`` is provided, if an ``intervention`` passed
                through ``args_baseline`` or ``args_intervention`` names a site that
                is not a sample site of the kernel, or if the scenarios do not share a
                predictive group.

        See Also:
            :meth:`~aimz.ImpactModel.cleanup` to remove the temporary directory if
            created.
        """
        if output_baseline is None or output_intervention is None:
            _check_is_fitted(self)
        for output, args in (
            (output_baseline, args_baseline),
            (output_intervention, args_intervention),
        ):
            if output is None and args is not None:
                _validate_intervention(
                    args.get("intervention"),
                    kernel_spec=self._kernel_spec,
                )

        if output_baseline is None and args_baseline is None:
            msg = "Either `output_baseline` or `args_baseline` must be provided."
            raise ValueError(msg)
        if output_intervention is None and args_intervention is None:
            msg = (
                "Either `output_intervention` or `args_intervention` must be provided."
            )
            raise ValueError(msg)

        # Ask predict_on_batch for a tree, which names its own predictive group
        _predict = cast(
            "Callable[..., xr.DataTree]",
            self.predict_on_batch if on_batch else self.predict,
        )
        overrides = {"return_datatree": True} if on_batch else {}

        # Lazily generated scenarios share one sampling key, so their contrast carries
        # only the intervention.
        if (
            output_baseline is None
            and output_intervention is None
            and args_baseline is not None
            and args_intervention is not None
            and args_baseline.get("rng_key") is None
            and args_intervention.get("rng_key") is None
        ):
            self._rng_key, rng_key = random.split(self._rng_key)
            args_baseline = {**args_baseline, "rng_key": rng_key}
            args_intervention = {**args_intervention, "rng_key": rng_key}

        dt_baseline = (
            output_baseline
            if output_baseline is not None
            else _predict(**{**cast("dict", args_baseline), **overrides})
        )
        dt_intervention = (
            output_intervention
            if output_intervention is not None
            else _predict(**{**cast("dict", args_intervention), **overrides})
        )

        group = _validate_group(dt_baseline, dt_intervention=dt_intervention)

        out = xr.DataTree(name="root")
        out[group] = dt_intervention[group] - dt_baseline[group]
        if self.posterior and group != "prior_predictive":
            out["posterior"] = _dict_to_datatree(
                self.posterior,
                num_chains=self._num_chains,
                dims=self._dims,
            )
        # Record each scenario's artifact path when the scenario was computed by a
        # disk-backed method. In-memory (on_batch / *_on_batch / store="memory")
        # results carry no artifact attrs.
        out.attrs.update(
            {
                f"artifact_path_{suffix}": path
                for suffix, tree in (
                    ("baseline", dt_baseline),
                    ("intervention", dt_intervention),
                )
                if (path := tree.attrs.get("artifact_path")) is not None
            },
        )

        return out

    def log_likelihood(
        self,
        X: ArrayLike | ArrayLoader | Iterable[Mapping[str, Array | np.ndarray]],
        y: ArrayLike | None = None,
        *,
        shard_axis: Literal["obs", "draw"] = "obs",
        batch_size: int | None = None,
        store: Literal["persistent", "memory"] = "persistent",
        output_dir: str | Path | None = None,
        progress: bool = True,
        **kwargs: object,
    ) -> xr.DataTree:
        """Compute the log-likelihood of the data under the given model.

        Results are written to disk in the Zarr format, with computing and file writing
        decoupled and executed concurrently. Pass ``store="memory"`` to accumulate the
        results in host memory instead.

        Args:
            X: Input array with observations on the leading axis, or a data loader
                yielding batch mappings keyed by kernel parameter names, including the
                observed output.
            y: Output array with observations on the leading axis. Must be ``None``
                if ``X`` is a data loader.
            shard_axis: Multi-device sharding strategy.
                ``"obs"`` (default) shards the input across devices and replicates the
                posterior. ``"draw"`` shards the posterior across devices and replicates
                the input, which must be an array, not a data loader. If the model has
                no posterior samples, the data path is used regardless of
                ``shard_axis``.
            batch_size: Size of each batch, taken from the input under
                ``shard_axis="obs"`` and from the draws under ``shard_axis="draw"``.
                Also used as the chunk size when storing results. If ``None``, it is
                chosen automatically.
                Ignored if ``X`` is a data loader, in which case the data loader is
                expected to handle batching internally.
            store: Where results accumulate. ``"persistent"`` (default) streams
                batches to a
                Zarr store under ``output_dir`` and returns a lazy, Dask-backed tree
                recording its ``artifact_path`` attribute. ``"memory"`` retains the
                batches in host memory as the returned tree's chunks.
            output_dir: The directory where the outputs will be saved. If the specified
                directory does not exist, it will be created automatically. If ``None``,
                a model-owned temporary directory is used. A subdirectory is generated
                within this directory to store the outputs; its path is recorded in
                the returned tree's ``artifact_path`` attribute (on the root and the
                group node). The temporary directory is removed
                by :meth:`~aimz.ImpactModel.cleanup`,
                :meth:`~aimz.ImpactModel.cleanup_models`, or when the model is
                garbage-collected. Pass an explicit ``output_dir`` to keep results
                beyond the model's lifetime.
            progress: Whether to display a progress bar.
            **kwargs: Additional arguments passed to the model. Arrays aligned with
                ``X`` are batched with it; other values are passed whole, arrays
                traced and the rest static.

        Returns:
            Log-likelihood values. Posterior samples are included if available.

        Raises:
            NotFittedError: If the model is not fitted.
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument,
                ``shard_axis="draw"`` is used with a data loader ``X``, or ``y`` is
                passed with a data loader ``X``.
            ValueError: If ``shard_axis`` is not ``"obs"`` or ``"draw"``, ``store`` is
                not ``"persistent"`` or ``"memory"``, ``output_dir`` is passed with
                ``store="memory"``, ``y`` is missing when ``X`` is array-like, or
                ``y`` does not share ``X``'s leading-axis size.
            NotImplementedError: If a return site's axis-1 size does not match the
                input batch size (``shard_axis="obs"`` only).

        See Also:
            :meth:`~aimz.ImpactModel.cleanup` to remove the temporary directory if
            created.
        """
        _check_is_fitted(self)
        _validate_shard_axis(shard_axis, X=X)
        _validate_batch_size(batch_size, X=X)
        _validate_store(store, output_dir=output_dir)
        _group_kwargs(kwargs, forbid=(self.param_input, self.param_output))
        _validate_aligned_inputs(X, y=y)
        if y is None and isinstance(X, ArrayLike):
            msg = (
                "`y` is required for `log_likelihood()` when `X` is array-like. "
                "Provide `y`, or give `X` as a data loader that carries it."
            )
            raise ValueError(msg)

        # No posterior to shard means draw-parallel has nothing to chunk, so it behaves
        # identically to the data-parallel path (a single-draw result).
        if not self.posterior:
            shard_axis = "obs"
        # One value of the default float type per observation, whatever the output's
        # own shape and dtype
        shard_axis, batch_size = self._plan_execution(
            X,
            shard_axis=shard_axis,
            batch_size=batch_size,
            num_samples=self._num_samples,
            nbytes=(jnp.result_type(float).itemsize, 0),
            posterior=self.posterior,
        )

        # With no posterior, the single-draw result samples every latent site from the
        # prior, which requires a seeded kernel. A posterior from fitting covers every
        # latent site, so no key is consumed and the bare kernel keeps a stable identity
        # for the compilation cache; a partial posterior raises an error.
        kernel = (
            self.kernel if self.posterior else seed(self.kernel, rng_seed=self.rng_key)
        )

        return self._stream_to_datatree(
            lambda artifact_path: self._streamer.write_log_likelihood(
                _WriteRequest(
                    shard_axis,
                    X=X,
                    return_sites=(self.param_output,),
                    num_samples=self._num_samples,
                    batch_size=batch_size,
                    artifact_path=artifact_path,
                    progress=progress,
                    loader_rng_key=self.rng_key,
                    kwargs=kwargs,
                    dims=self._dims,
                ),
                kernel=kernel,
                posterior=self.posterior,
                y=y,
            ),
            store=store,
            output_dir=output_dir,
            group="log_likelihood",
        )

    def cleanup(self) -> None:
        """Clean up the temporary directory created for storing outputs.

        If the temporary directory was never created or has already been cleaned up,
        this method does nothing. It does not delete any explicitly specified output
        directory. While the temporary directory is typically removed automatically
        during garbage collection, this behavior is not guaranteed, so calling this
        method explicitly is recommended for timely resource release.

        See Also:
            :meth:`~aimz.ImpactModel.cleanup_models`: clean temporary directories for
            all tracked model instances.
        """
        if hasattr(self, "_temp_dir") and self._temp_dir is not None:
            temp_dir = self._temp_dir.name
            self._temp_dir.cleanup()
            self._temp_dir = None
            logger.info("Temporary directory cleaned up at: %s", temp_dir)

    @classmethod
    def cleanup_models(cls) -> None:
        """Clean up temporary directories for all :class:`~aimz.ImpactModel` instances.

        See Also:
            :meth:`~aimz.ImpactModel.cleanup`: clean the temporary directory for a
            single instance.
        """
        for model in cls._models:
            try:
                model.cleanup()
            except OSError:
                logger.warning(
                    "Failed to clean up the temporary directory at: %s",
                    model.temp_dir,
                    exc_info=True,
                )
