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
import pickle
from copy import copy
from datetime import UTC, datetime
from functools import partial
from inspect import signature, stack
from itertools import count
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast
from warnings import warn
from weakref import WeakSet

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import xarray as xr
from jax import (
    Array,
    ShapeDtypeStruct,
    default_backend,
    device_get,
    eval_shape,
    jit,
    lax,
    local_device_count,
    local_devices,
    make_jaxpr,
    make_mesh,
    random,
)
from jax.sharding import AxisType, NamedSharding, PartitionSpec
from jax.typing import ArrayLike
from numpyro import validation_enabled
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import MCMC, SVI, init_to_sample
from numpyro.infer.svi import SVIRunResult, SVIState
from tqdm.auto import tqdm

from aimz._exceptions import (
    _SKIP_FILE_PREFIXES,
    AimzWarning,
    FitWarning,
    KernelValidationError,
    NotFittedError,
    OutputWarning,
    PerformanceWarning,
)
from aimz.model._core import BaseModel
from aimz.model._planning import _output_nbytes, _plan_execution
from aimz.model._scenario import _build_scenario, _check_scenario
from aimz.model._streaming import (
    _OutputStreamer,
    _RuntimeContext,
    _WriteRequest,
)
from aimz.model._tracing import _trace_kernel
from aimz.model.kernel_spec import KernelSpec
from aimz.sampling._forward import _sample_forward
from aimz.utils._format import (
    _build_datatree,
    _dict_to_datatree,
    _draw_attrs,
)
from aimz.utils._kwargs import (
    _combine,
    _group_kwargs,
    _partition,
    _to_jax,
)
from aimz.utils._validation import (
    _check_is_fitted,
    _coerce_return_sites,
    _validate_aligned_inputs,
    _validate_batch_size,
    _validate_condition,
    _validate_group,
    _validate_intervention,
    _validate_shard_axis,
    _validate_store,
    _validate_X_y_to_jax,
    _warn_missing_return_sites,
    _warn_unreachable_intervention,
)
from aimz.utils.data import ArrayLoader
from aimz.utils.data._input_setup import _setup_inputs
from aimz.utils.data.array_loader import _slice_rows

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sized

    from dask.array import Array as DaskArray

    from aimz.model.causal_card import CausalCard

logger = logging.getLogger(__name__)


def _reduce_key(key: Array) -> str | tuple[Any, ...]:
    """Reduce a typed PRNG key for pickling.

    A pickled typed key carries a copy of its PRNG implementation, which some samplers
    reject in a new process, so a key of a registered implementation is rebuilt from
    its raw data and the implementation's name.
    """
    impl = random.key_impl(key)
    if isinstance(impl, str):
        return partial(random.wrap_key_data, impl=impl), (random.key_data(key),)

    return key.__reduce_ex__(pickle.DEFAULT_PROTOCOL)


class ImpactModel(BaseModel):
    """Impact modeling interface: fit, sample, predict, and estimate effects."""

    _models: ClassVar[WeakSet[ImpactModel]] = WeakSet()

    def __init__(
        self,
        kernel: Callable,
        rng_key: Array,
        inference: SVI | MCMC,
        *,
        param_input: str = "X",
        param_output: str = "y",
        card: CausalCard | None = None,
    ) -> None:
        """Initialize an :class:`~aimz.ImpactModel` instance.

        Args:
            kernel: A probabilistic model with `NumPyro`_ primitives.
            rng_key: A typed key array from :external:func:`jax.random.key`.
            inference: An :external:class:`~numpyro.infer.svi.SVI` or
                :external:class:`~numpyro.infer.mcmc.MCMC` instance.
            param_input: Name of the ``kernel`` parameter for the input data.
            param_output: Name of the ``kernel`` parameter for the output data.
            card: A :class:`~aimz.model.CausalCard` declaring the causal scope the
                modeler asserts for the model, or ``None``.

        Raises:
            KernelValidationError: If the kernel takes variable arguments or
                positional-only parameters, lacks the input or output parameter, gives
                the input a default, or gives the output a default other than ``None``.
            TypeError: If ``inference`` is neither SVI nor MCMC.
            ValueError: If ``inference`` is SVI with static keyword arguments, which
                aimz does not pass to the guide or the kernel outside training.

        Warns:
            AimzWarning: If ``rng_key`` is a legacy ``uint32`` key, which is converted
                to a typed key array.
        """
        super().__init__(kernel, param_input, param_output)
        self._kernel_spec: KernelSpec | None = None
        self._dims: dict[str, tuple[str, ...]] = {}
        self._site_sizes: dict[str, tuple[int, int]] = {}
        self._observed_sites: set[str] = set()
        self._fitted_sites: set[str] = set()
        self._coupled_sites: set[str] = set()
        self._site_parents: dict[str, set[str]] = {}
        self._site_upstream: dict[str, set[str]] = {}
        self._site_arguments: dict[str, set[str]] = {}
        self._site_dists: dict[str, str] = {}
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
        if isinstance(inference, SVI) and inference.static_kwargs:
            msg = (
                "`SVI` static keyword arguments "
                f"({', '.join(inference.static_kwargs)}) are not supported, since aimz "
                "draws from the guide and runs the kernel without them. Pass them as "
                "keyword arguments of the kernel instead."
            )
            raise ValueError(msg)
        self._inference = inference
        self._vi_result: SVIRunResult | None = None
        self._vi_state = None
        self._is_fitted = False
        self._posterior: dict[str, Array] | None = None
        self._num_samples = 0
        self._num_chains = 1
        self._init_runtime_attrs()
        self._card: CausalCard | None = None
        self.card = card

    def _init_runtime_attrs(self) -> None:
        """Initialize runtime attributes."""
        self._fn_vi_update: Callable | None = None
        self._fn_vi_run: Callable | None = None
        # The return sites a streamed call cannot split, whether the kernel failed on
        # one row fewer, and the posterior sites whose size follows the rows, by the
        # call's shapes and constants, and the calls that passed `log_likelihood`'s
        # check of keyword arrays as long as each device's rows
        self._plan_cache: dict[
            tuple,
            tuple[frozenset[str], bool, dict[str, tuple[int, ...]]],
        ] = {}
        # The spec of each set of call arguments traced so far, by their static part
        self._specs: dict[tuple, KernelSpec | None] = {}
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
        """Return the state without the runtime attributes."""
        copyreg.pickle(type(random.key(0)), _reduce_key)
        return {
            k: v
            for k, v in self.__dict__.items()
            if not (
                k.startswith("_fn")
                or k
                in {
                    "_num_devices",
                    "_plan_cache",
                    "_specs",
                    "_temp_dir",
                    "_streamer",
                }
            )
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore the state and reinitialize the runtime attributes."""
        self.__dict__.update(state)
        # Attributes that models pickled by earlier versions lack
        self.__dict__.setdefault("_is_fitted", False)
        self.__dict__.setdefault("_num_chains", 1)
        self.__dict__.setdefault("_dims", {})
        self.__dict__.setdefault("_site_sizes", {})
        self.__dict__.setdefault("_observed_sites", set())
        self.__dict__.setdefault("_card", None)
        # A model pickled by an earlier version traced only the arguments it was
        # first called or fitted with
        spec = self.__dict__.get("_kernel_spec")
        self.__dict__.setdefault(
            "_fitted_sites",
            set(spec.sample_sites) if spec is not None and spec.traced else set(),
        )
        self.__dict__.setdefault("_coupled_sites", set())
        self.__dict__.setdefault("_site_parents", {})
        self.__dict__.setdefault("_site_upstream", {})
        self.__dict__.setdefault("_site_arguments", {})
        self.__dict__.setdefault("_site_dists", {})
        self._init_runtime_attrs()
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
    def card(self) -> CausalCard | None:
        """The :class:`~aimz.model.CausalCard` declared for the model, or ``None``.

        Setting it on a model whose kernel has been traced checks the names the
        card's treatments span and its unit name: each has to be a declared feature,
        the input ``X``, a keyword argument of the kernel, or one of its sample
        sites. A card given before the first trace is not checked.
        """
        return self._card

    @card.setter
    def card(self, card: CausalCard | None) -> None:
        """Attach a card, checking its names if the kernel has been traced.

        Raises:
            ValueError: If the kernel has been traced and a name a treatment spans, or
                the unit, is none of the declared features, the input ``X``, the
                kernel's keyword arguments, or its sample sites.
        """
        if card is not None and (spec := self._kernel_spec) is not None and spec.traced:
            known = {
                "X",
                *card.features,
                *(signature(self.kernel).parameters.keys() - {self.param_input}),
                *spec.sample_sites,
            }
            unknown = [
                name
                for name in (
                    *(name for names in card.treatments.values() for name in names),
                    *([card.unit] if card.unit else []),
                )
                if name not in known
            ]
            if unknown:
                msg = (
                    f"Card name(s) {', '.join(map(repr, unknown))} are none of the "
                    "declared features, the input `X`, the kernel's keyword "
                    "arguments, or its sample sites: "
                    f"{', '.join(map(repr, sorted(known)))}."
                )
                raise ValueError(msg)
        self._card = card

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
        """Posterior samples by site name, or ``None`` if not set.

        Replace them through :meth:`~aimz.ImpactModel.set_posterior_sample` or a refit;
        a change in place desynchronizes the device-placement cache.
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

        :setter: Sets the :external:data:`~numpyro.infer.svi.SVIRunResult` without
            marking the model fitted or drawing samples; :meth:`~aimz.ImpactModel.fit`
            and :meth:`~aimz.ImpactModel.fit_on_batch` continue from its state.
        """
        return self._vi_result

    @vi_result.setter
    def vi_result(self, vi_result: SVIRunResult) -> None:
        """Set the variational inference result.

        Args:
            vi_result: The result of an SVI run, with ``params``, ``state``, and
                ``losses``.

        Warns:
            FitWarning: If the losses contain NaN or Inf.
        """
        if not np.all(np.isfinite(vi_result.losses)):
            msg = "Loss contains NaN or Inf, indicating numerical instability."
            # Not `skip_file_prefixes`: skipping the aimz frames would blame MLflow's
            # autolog wrapper, which then hides the warning
            warn(msg, category=FitWarning, stacklevel=2)
        self._vi_result = vi_result
        self._vi_state = vi_result.state

    def _bind_kernel_args(
        self,
        X: ArrayLike,
        kwargs: Mapping[str, object],
    ) -> dict[str, object]:
        """Bind the input and the extra arguments to the kernel signature.

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
        rows_more: int | None = None,
        whole: Collection[str] = (),
    ) -> KernelSpec:
        """Trace the kernel with a call's arguments, as :func:`_trace_kernel` does.

        What the trace shows goes into the model's own containers, and the model's
        spec becomes the one with the trace's sites merged in.

        Returns:
            The spec of the trace with these arguments.
        """
        spec, self._kernel_spec = _trace_kernel(
            args_bound,
            with_output=with_output,
            rows_more=rows_more,
            whole=whole,
            kernel=self.kernel,
            rng_key=self.rng_key,
            param_input=self.param_input,
            param_output=self.param_output,
            kernel_spec=self._kernel_spec,
            specs=self._specs,
            dims=self._dims,
            site_sizes=self._site_sizes,
            observed_sites=self._observed_sites,
            fitted_sites=self._fitted_sites,
            coupled_sites=self._coupled_sites,
            site_parents=self._site_parents,
            site_upstream=self._site_upstream,
            site_arguments=self._site_arguments,
            site_dists=self._site_dists,
        )

        return spec

    def _call_spec(self, args_bound: Mapping[str, object]) -> KernelSpec | None:
        """Return the spec of a trace of the kernel with a call's arguments.

        The trace is abstract, so the arguments are copied into JAX arrays only when it
        fails, as for a kernel that reads a concrete value of an argument, which then
        runs once on them. A traced kernel that does not run on them at once, as one
        whose subsampling plate is smaller than the input, keeps the model's spec, for
        every later call with these arguments too.
        """
        static = _partition(args_bound)[1]
        if static not in self._specs:
            try:
                return self._build_kernel_spec(args_bound, with_output=False)
            except AimzWarning:
                raise
            except Exception:
                # A traced kernel keeps its spec for arguments it cannot be traced
                # with, as when it names the output otherwise without data; an
                # untraced one has none to fall back on
                if not (self._kernel_spec and self._kernel_spec.traced):
                    raise
                self._specs[static] = None

        return self._specs[static] or self._kernel_spec

    def _recomputed_sites(self) -> set[str]:
        """Return the posterior sites traced as deterministic with a value per row.

        A call keeps off the devices those that its own trace does not read, since the
        kernel recomputes them there.
        """
        spec = cast("KernelSpec", self._kernel_spec)

        return {
            k
            for k in set(spec.return_sites) - set(spec.sample_sites)
            if k in (self._posterior or {}) and self._site_sizes.get(k, (0, 0))[0]
        }

    def _create_artifact_path(
        self,
        output_dir: str | Path | None,
    ) -> Path:
        """Create a ``<UTC timestamp>_<caller>`` directory for one disk-backed call.

        Without an ``output_dir``, it goes under the model's temporary directory, which
        is created on first use.
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

    def _stream_to_datatree(
        self,
        write: Callable[[Path | None], dict[str, DaskArray] | None],
        *,
        store: str,
        output_dir: str | Path | None,
        group: str,
        attrs: Mapping[str, object] | None = None,
        requested: Iterable[str] = (),
    ) -> xr.DataTree:
        """Run one streamed write and assemble its output tree.

        ``write`` streams into the artifact path, or into host memory for ``None``, and
        returns the retained site arrays, if any. A failed persistent write removes its
        artifact path, also when a ``requested`` site missing from the output warns
        with a warning raised as an error.
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
                attrs=attrs,
            )
            _warn_missing_return_sites(requested, returned=dt[group].data_vars)
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

        The kernel's ``param`` sites keep their initial values, not those a fit learned.

        Args:
            X: Input array with observations on the leading axis.
            intervention: Replacement values by sample site name, applied while
                sampling.
            num_samples: The number of samples to draw.
            rng_key: A pseudo-random number generator key, split from an internal key by
                default.
            return_sites: Names of the sites to return; by default the output and the
                deterministic sites.
            return_datatree: Return an :external:class:`~xarray.DataTree`, or a
                :class:`dict` if ``False``.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Prior predictive samples. Posterior samples are included if available.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument.
            ValueError: If ``intervention`` names a site that is not a sample site of
                the kernel.
            NotImplementedError: If a site's prior cannot be sampled, as for an
                improper prior, also when the site is intervened on, since the site
                still draws its own value first; NumPyro's error names no site.

        Warns:
            OutputWarning: If the output site does not have one value per observation
                of the input, or a requested return site is intervened on, since it
                comes back with its own draws, or missing from the output.

        See Also:
            :meth:`~aimz.ImpactModel.sample_prior_predictive`
        """
        X = cast("Array", _validate_X_y_to_jax(X))

        args_bound = _to_jax(self._bind_kernel_args(X, kwargs=kwargs))
        spec = self._call_spec(args_bound)
        _validate_intervention(intervention, kernel_spec=spec)
        sites = _coerce_return_sites(
            return_sites,
            intervention=intervention,
            spec=spec or self._kernel_spec,
        )

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        prior_predictive_samples = device_get(
            _sample_forward(
                self.kernel,
                rng_keys=random.split(rng_key, num=num_samples),
                return_sites=sites,
                samples=None,
                params=None,
                intervention=_to_jax(intervention or {}),
                model_kwargs=args_bound,
            ),
        )
        if return_sites is not None:
            _warn_missing_return_sites(sites, returned=prior_predictive_samples)
        out = prior_predictive_samples.get(self.param_output)
        if out is not None and np.shape(out)[1:2] != (len(X),):
            msg = (
                f"The output site {self.param_output!r} does not have one value per "
                "observation of the input."
            )
            warn(msg, category=OutputWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)

        if not return_datatree:
            return prior_predictive_samples

        return _build_datatree(
            prior_predictive_samples,
            group="prior_predictive",
            posterior=self.posterior,
            num_chains=self._num_chains,
            dims=self._dims,
            attrs=_draw_attrs(rng_key),
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

        The batches are computed and written to a Zarr store concurrently, or kept in
        host memory with ``store="memory"``. The kernel's ``param`` sites keep their
        initial values, not those a fit learned.

        Args:
            X: Input array with observations on the leading axis, or a data loader
                yielding batch mappings keyed by kernel parameter names.
            intervention: Replacement values by sample site name, applied while
                sampling.
            num_samples: The number of samples to draw.
            rng_key: A pseudo-random number generator key, split from an internal key by
                default.
            return_sites: Names of the sites to return; by default the output and the
                deterministic sites.
            shard_axis: ``"obs"`` shards the input across devices and replicates the
                draws; ``"draw"`` shards the draws and replicates the input, which must
                then be an array.
            batch_size: Observations per batch under ``"obs"``, draws per batch under
                ``"draw"``, and the chunk size of the stored results. Chosen
                automatically if ``None``; ignored for a data loader.
            store: ``"persistent"`` streams the batches to a Zarr store under
                ``output_dir`` and returns a lazy tree recording its ``artifact_path``;
                ``"memory"`` keeps the batches in host memory.
            output_dir: Directory for the Zarr store, created if needed; by default a
                temporary directory of the model, removed by
                :meth:`~aimz.ImpactModel.cleanup`,
                :meth:`~aimz.ImpactModel.cleanup_models`, or garbage collection. Each
                call writes its own subdirectory, recorded as the tree's
                ``artifact_path`` attribute.
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
            NotImplementedError: If a site's prior cannot be sampled, as for an
                improper prior, also when the site is intervened on; or, with a data
                loader, if a return site has no observation axis, unless it is
                intervened on or the kernel does not run on one observation fewer than
                a device evaluates (two when it evaluates one), or if its axis-1 size
                does not match the number of observations a device evaluates from a
                batch.

        Warns:
            OutputWarning: If a requested return site is intervened on, since it comes
                back with its own draws, or missing from the output, or if the output
                site does not have one value per observation of an array input; under
                ``shard_axis="obs"`` the call then reruns under ``shard_axis="draw"``.
                Also if the batches of a data loader split a kernel that computes
                across observations.
            PerformanceWarning: If ``shard_axis="obs"`` cannot split an array input,
                as for a return site without an observation axis, so the call reruns
                under ``shard_axis="draw"``.

        See Also:
            :meth:`~aimz.ImpactModel.sample_prior_predictive_on_batch` for a
            single-batch, in-memory alternative.
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

        # Trace a one-row probe of the per-observation arguments, on JAX arrays as the
        # sampling step runs the kernel; a kernel that does not run on one row, such as
        # one whose plate needs several rows, is traced on all the rows
        probe = {
            k: (jnp.asarray(cast("Any", v)[:1]) if k in obs_names else v)
            for k, v in args_bound.items()
        }
        try:
            static = _partition(probe)[1]
            spec = (
                self._specs[static] or self._kernel_spec
                if static in self._specs
                else self._build_kernel_spec(_to_jax(probe), with_output=False)
            )
        except Exception:
            spec = self._call_spec(args_bound)
        _validate_intervention(intervention, kernel_spec=spec)

        requested = return_sites is not None
        return_sites = _coerce_return_sites(
            return_sites,
            intervention=intervention,
            spec=spec or self._kernel_spec,
        )
        shard_axis, batch_size = _plan_execution(
            X,
            shard_axis=shard_axis,
            batch_size=batch_size,
            num_samples=num_samples,
            nbytes=_output_nbytes(self._site_sizes, return_sites),
            posterior=None,
            return_sites=return_sites,
            kwargs=kwargs,
            intervention=intervention,
            kernel=self.kernel,
            rng_key=self.rng_key,
            param_input=self.param_input,
            param_output=self.param_output,
            kernel_spec=self._kernel_spec,
            coupled_sites=self._coupled_sites,
            observed_sites=self._observed_sites,
            num_devices=self._num_devices,
            plan_cache=self._plan_cache,
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
                params=None,
                intervention=intervention,
                stream=stream,
            ),
            store=store,
            output_dir=output_dir,
            group="prior_predictive",
            attrs=_draw_attrs(
                rng_key,
                X=X,
                shard_axis=shard_axis,
                batch_size=batch_size,
                num_devices=self._num_devices,
            ),
            requested=return_sites if requested else (),
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

        Under SVI, each draw is a call of the guide, so a guide whose call is a point
        estimate gives that point in every draw; set draws made another way with
        :meth:`~aimz.ImpactModel.set_posterior_sample`.

        Args:
            num_samples: The number of posterior samples to draw. Under MCMC, it counts
                per chain and before thinning, as the sampler's ``num_samples`` does.
            rng_key: A pseudo-random number generator key, split from an internal key by
                default. Ignored for MCMC, which continues its chain.
            return_sites: Names of the sites to return; by default the kernel's latent
                sites, or every site the guide draws if no trace of the kernel has
                observed the output yet. Ignored for MCMC.
            return_datatree: Return an :external:class:`~xarray.DataTree`, or a
                :class:`dict` if ``False``.
            **kwargs: Arguments passed to the model under MCMC, and to the guide under
                SVI.

        Returns:
            Posterior samples.

        Raises:
            NotFittedError: If there is no inference state to sample from (no
                completed MCMC run, or no ``vi_result`` when the inference method
                is SVI).
            TypeError: If an argument of the kernel without a default, or the output,
                is not passed when the inference method is MCMC, or SVI whose guide has
                not run yet, as with a ``vi_result`` from another SVI object; one
                missing argument is named at a time.

        Warns:
            OutputWarning: If a requested return site is missing from the output.
        """
        kwargs = _to_jax(kwargs)
        if isinstance(self.inference, MCMC):
            if self.inference.last_state is None:
                msg = (
                    "This ImpactModel instance has no MCMC state to sample from. "
                    "Call `.fit_on_batch()`, or run the MCMC directly, before using "
                    "`.sample()`."
                )
                raise NotFittedError(msg)
            args_bound = signature(self.kernel).bind(**kwargs).arguments
            if self.param_output not in args_bound:
                msg = f"{self.param_output!r} must be provided in `.sample()`."
                raise TypeError(msg)
            # Continue the chain for this call only; later fits keep their own settings
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
                # The collection bounds follow the restored number of samples
                self.inference._set_collection_params()
            self.inference.transfer_states_to_host()
            posterior_samples = dict(self.inference.get_samples())
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
            # A guide that has not run yet is set up before the draws, with the output
            # observed as in training
            if self.inference.constrain_fn is None:
                args_bound = signature(self.kernel).bind(**kwargs).arguments
                if self.param_output not in args_bound:
                    msg = f"{self.param_output!r} must be provided in `.sample()`."
                    raise TypeError(msg)
                rng_key, rng_init = random.split(rng_key)
                self.inference.init(rng_init, **args_bound)
            spec = self._kernel_spec
            requested = return_sites is not None
            if return_sites is None and spec is not None and spec.output_observed:
                return_sites = spec.sample_sites
            sites = (
                _coerce_return_sites(return_sites) if return_sites is not None else None
            )
            posterior_samples = device_get(
                _sample_forward(
                    self.inference.guide,
                    rng_keys=random.split(rng_key, num=num_samples),
                    return_sites=sites,
                    samples=None,
                    params={
                        **self.vi_result.params,
                        **(getattr(self.vi_result.state, "mutable_state", None) or {}),
                    },
                    intervention=None,
                    model_kwargs=kwargs,
                ),
            )
            if requested:
                _warn_missing_return_sites(sites or (), returned=posterior_samples)

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
        condition: dict | None = None,
        rng_key: Array | None = None,
        return_sites: str | Iterable[str] | None = None,
        return_datatree: bool = True,
        **kwargs: object,
    ) -> xr.DataTree | dict[str, npt.NDArray]:
        """Draw samples from the posterior predictive distribution.

        An alias of :meth:`~aimz.ImpactModel.predict_on_batch` with ``in_sample=True``;
        see it for the arguments, errors, and warnings.
        """
        return self.predict_on_batch(
            X,
            intervention=intervention,
            condition=condition,
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
        condition: dict | None = None,
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

        An alias of :meth:`~aimz.ImpactModel.predict` with ``in_sample=True``; see it
        for the arguments, errors, and warnings.
        """
        return self.predict(
            X,
            intervention=intervention,
            condition=condition,
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
        """Run one SVI step on a batch, keeping the state internally.

        A later step continues from that state, which has no guide parameters for a
        latent site that a new keyword argument adds; create a new model for one, with
        a new guide, since an autoguide keeps the sites of its first call.

        Args:
            X: Input array with observations on the leading axis.
            y: Output array with observations on the leading axis.
            rng_key: A pseudo-random number generator key, used only to initialize the
                SVI state when it is not set yet; by default an internal key is split.
            **kwargs: Additional arguments passed to the model.

        Returns:
            The updated SVI state and the loss.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument.
        """
        _group_kwargs(kwargs, forbid=(self.param_input, self.param_output))
        batch = {self.param_input: X, self.param_output: y, **kwargs}

        svi = cast("SVI", self.inference)
        # The setup runs the kernel outside the compiled step, on JAX arrays as the step
        # sees them, since `fit` may pass NumPy batches. A result set through
        # `vi_result` may come with an SVI object that needs no initialization, so the
        # trace does not depend on it. The step's static part keys the spec too, so a
        # step with known arguments neither binds nor converts them
        leaves, static = _partition(batch)
        if (
            static not in self._specs
            or self._vi_state is None
            or svi.constrain_fn is None
        ):
            batch = _to_jax(batch)
            leaves = _partition(batch)[0]
        if static not in self._specs:
            self._specs[static] = self._build_kernel_spec(
                signature(self.kernel).bind(**batch).arguments,
                with_output=True,
            )
        if self._vi_state is None or svi.constrain_fn is None:
            if rng_key is None:
                self._rng_key, rng_key = random.split(self._rng_key)
            state = svi.init(rng_key, **batch)
            self._vi_state = state if self._vi_state is None else self._vi_state
        # Array leaves are traced; every other value is static
        if self._fn_vi_update is None:
            svi = cast("SVI", self.inference)

            def update(
                state: SVIState,
                leaves: list,
                static: tuple,
            ) -> tuple[SVIState, Array]:
                return svi.update(state, **_combine(leaves, static))

            self._fn_vi_update = jit(update, static_argnums=2)
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
        """Fit the model to one batch of data.

        With SVI, runs the optimization on the batch and draws posterior samples from
        the guide; with MCMC, runs the sampler. Training continues from an existing SVI
        state; create a new model to start over, or to fit with a keyword argument that
        adds a latent site: that state has no guide parameters for it, so the posterior
        lacks it and the predictive methods draw it from its prior. For such a site,
        give the new model a new guide too, since an autoguide keeps the sites of its
        first call.

        Under SVI, each draw is a call of the guide, so a guide whose call is a point
        estimate gives that point in every draw; set draws made another way with
        :meth:`~aimz.ImpactModel.set_posterior_sample`.

        Args:
            X: Input array with observations on the leading axis.
            y: Output array with observations on the leading axis.
            num_steps: Number of optimization steps. Ignored for MCMC.
            num_samples: The number of posterior samples to draw. Ignored for MCMC.
            rng_key: A pseudo-random number generator key, split from an internal key by
                default.
            progress: Whether to display a progress bar. Ignored for MCMC.
            **kwargs: Additional arguments passed to the model; with SVI, arrays are
                traced and the rest static.

        Returns:
            The fitted model instance.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument.
            ValueError: If ``y`` does not share ``X``'s leading-axis size, or
                ``num_steps`` is less than one with SVI.
        """
        X, y = _validate_X_y_to_jax(X, y=y)

        _group_kwargs(kwargs, forbid=(self.param_input, self.param_output))
        args_bound = (
            signature(self.kernel)
            .bind(**{self.param_input: X, self.param_output: y, **_to_jax(kwargs)})
            .arguments
        )
        self._build_kernel_spec(args_bound, with_output=True)
        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)
        rng_key, rng_subkey = random.split(rng_key)
        # Release the placement of the posterior this call replaces
        self._streamer.place_posterior(None, sharding=None)
        if isinstance(self.inference, SVI):
            svi = self.inference
            if self._vi_state is not None and svi.constrain_fn is None:
                rng_key, rng_init = random.split(rng_key)
                svi.init(rng_init, **args_bound)
            if num_steps < 1:
                msg = "num_steps must be a positive integer."
                raise ValueError(msg)
            logger.info("Performing variational inference optimization")
            state = (
                svi.init(rng_subkey, **args_bound)
                if self._vi_state is None
                else self._vi_state
            )
            # The steps run as a scan that takes the arrays as traced arguments, as in
            # `train_on_batch`, in up to 20 chunks of one length, so the bar advances
            # between chunks and later calls with the same shapes reuse the program
            if self._fn_vi_run is None:

                def run(
                    state: SVIState,
                    leaves: list,
                    static: tuple,
                    length: int,
                ) -> tuple[SVIState, Array]:
                    def step(state: SVIState, _: None) -> tuple[SVIState, Array]:
                        return svi.update(state, **_combine(leaves, static))

                    return lax.scan(step, state, None, length=length)

                self._fn_vi_run = jit(run, static_argnums=(2, 3))
            leaves, static = _partition(args_bound)
            size = num_steps // max(d for d in range(1, 21) if num_steps % d == 0)
            losses: list = []
            with tqdm(
                total=num_steps, disable=not progress, dynamic_ncols=True
            ) as pbar:
                for start in range(0, num_steps + size, size):
                    if start < num_steps:
                        state, loss = self._fn_vi_run(state, leaves, static, size)
                        losses.append(loss)
                    # A chunk's losses are read while the next chunk runs, so the device
                    # never waits for the bar
                    if start:
                        i = start // size - 1
                        losses[i] = device_get(losses[i])
                        pbar.update(size)
                        pbar.set_postfix_str(
                            f"init loss: {losses[0][0]:.4f}, avg. loss "
                            f"[{start - size + 1}-{start}]: {losses[i].mean():.4f}",
                            refresh=False,
                        )
            self.vi_result: SVIRunResult = SVIRunResult(
                params=svi.get_params(state),
                state=state,
                losses=np.concatenate(losses),
            )

            # Clear the previous fit first: the draw below rejects diverged parameters
            self._is_fitted = False
            self._posterior = None
            logger.info("Drawing posterior samples (num_samples=%d)", num_samples)
            rng_key, rng_subkey = random.split(rng_key)
            self._posterior = _sample_forward(
                self.inference.guide,
                rng_keys=random.split(rng_subkey, num=num_samples),
                return_sites=cast("KernelSpec", self._kernel_spec).sample_sites,
                samples=None,
                params={
                    **self.vi_result.params,
                    **(getattr(self.vi_result.state, "mutable_state", None) or {}),
                },
                intervention=None,
                model_kwargs=args_bound,
            )
            self._num_chains = 1
            self._num_samples = num_samples
        elif isinstance(self.inference, MCMC):
            logger.info(
                "Drawing posterior samples (num_samples=%d)",
                self.inference.num_samples,
            )
            self.inference.run(rng_subkey, **args_bound)
            self.inference.transfer_states_to_host()
            self._posterior = dict(self.inference.get_samples())
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
        """Fit the model with variational inference over minibatches.

        The data is iterated in minibatches for a number of epochs, then posterior
        samples are drawn from the guide, called with the data in input order. With a
        data loader, the guide gets the last batch instead, so a latent drawn per
        observation covers only that batch. Training continues from an existing SVI
        state; create a new model to start over, or to fit with a keyword argument that
        adds a latent site: that state has no guide parameters for it, so the posterior
        lacks it and the predictive methods draw it from its prior. For such a site,
        give the new model a new guide too, since an autoguide keeps the sites of its
        first call.

        Each draw is a call of the guide, so a guide whose call is a point estimate
        gives that point in every draw; set draws made another way with
        :meth:`~aimz.ImpactModel.set_posterior_sample`.

        Args:
            X: Input array with observations on the leading axis, or a data loader
                yielding batch mappings keyed by kernel parameter names, including the
                observed output. A data loader is iterated once per epoch.
            y: Output array with observations on the leading axis. Must be ``None``
                if ``X`` is a data loader.
            num_samples: The number of posterior samples to draw.
            rng_key: A pseudo-random number generator key, split from an internal key by
                default.
            progress: Whether to display a progress bar and the average loss of each
                epoch.
            batch_size: Observations per optimization step; the whole data if ``None``.
                Ignored for a data loader, which batches itself.
            epochs: The number of passes over the data.
            shuffle: Whether to shuffle the data at each epoch when it is split into
                batches. Ignored for a data loader.
            **kwargs: Additional arguments passed to the model. Arrays aligned with
                ``X`` are batched with it; other values are passed whole, arrays
                traced and the rest static.

        Returns:
            The fitted model instance.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument, the
                inference method is MCMC, or ``y`` is passed with a data loader ``X``.
            ValueError: If ``y`` is missing when ``X`` is array-like, ``y`` does not
                share ``X``'s leading-axis size, or a data loader yields no batches
                in an epoch.

        Warns:
            FitWarning: If the batches are smaller than the data and the kernel does
                not scale the output site of a batch to the whole data, by a plate
                whose ``size`` is the number of observations or by the number of
                observations over the batch size, while a site outside the output's
                plate, such as a global latent or a penalty passed to ``factor``,
                weighs against it; a scale given per observation, as weights are,
                counts only when it is one on every observation. Also if the batches
                split a site that computes across observations, or a latent site per
                observation that the guide does not compute from the batch, such as one
                kept per position or plate index. The checks run on the first
                ``batch_size`` observations of an array input or of an
                :class:`~aimz.utils.data.ArrayLoader`; another data loader has no size
                to compare with. Of the sites drawn as one event over the observations,
                only an event as long as the batch is checked, so one padded beyond the
                observations, or offset from them, trains without the warning.
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
        # A loader's fields may supply the arguments `bind_partial` leaves open
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
        kwargs_extra = _to_jax(kwargs_extra)
        # A keyword array as long as a batch, but not as the data, is passed whole, so
        # every batch would get the same values; as in `fit_on_batch`, the fit fails
        # when the kernel does not run on all the rows with it
        if (
            isinstance(dataloader, ArrayLoader)
            and dataloader.batch_size < len(dataloader.dataset)
            and any(
                np.shape(leaf)[:1] == (dataloader.batch_size,)
                for leaf in _partition(kwargs_extra)[0]
            )
        ):
            arrays, static = _partition({**dataloader.dataset.arrays, **kwargs_extra})

            def run(arrays: list) -> None:
                with validation_enabled(is_validate=False):
                    trace(
                        substitute(
                            seed(self.kernel, rng_seed=self.rng_key),
                            substitute_fn=init_to_sample,
                        ),
                    ).get_trace(**_combine(arrays, static))

            eval_shape(run, arrays)
            del arrays
        # Commit model state only once the inputs are accepted
        self._rng_key = rng_key_model
        # Release the placement of the posterior this call replaces
        self._streamer.place_posterior(None, sharding=None)

        rng_key, rng_subkey = random.split(rng_key)
        # The scale a batch gives the output site tells whether the kernel scales it to
        # the whole data, as it must when a site outside the output's plate, such as a
        # global latent or a penalty passed to `factor`, weighs against it; otherwise
        # every term follows the batch
        if isinstance(dataloader, ArrayLoader) and dataloader.batch_size < len(
            dataloader.dataset,
        ):
            n_obs = len(dataloader.dataset)
            batch = _to_jax(
                {
                    k: _slice_rows(v, 0, dataloader.batch_size)
                    if isinstance(v, Array)
                    else v[: dataloader.batch_size]
                    for k, v in dataloader.dataset.arrays.items()
                },
            )
            # The first batch already rejected an output site that is not observed
            with validation_enabled(is_validate=False):
                model_trace = trace(
                    substitute(
                        seed(self.kernel, rng_seed=self.rng_key),
                        substitute_fn=init_to_sample,
                    ),
                ).get_trace(**{**batch, **kwargs_extra})
            site = model_trace.get(self.param_output, {})
            # The output's outermost plate is the one over the batch's rows
            frames = sorted(site.get("cond_indep_stack") or (), key=lambda f: f.dim)[:1]
            rows = {f.name for f in frames}
            # A scale given per row weights the rows, which leaves nothing to compare
            # unless every weight is one
            scale = site.get("scale")
            scale = 1.0 if scale is None or np.all(scale == 1) else scale
            expected = n_obs / dataloader.batch_size
            if (
                site.get("is_observed")
                # A plate whose size is all the observations scales a batch to them,
                # whatever weights the kernel adds
                and not any(
                    model_trace.get(f.name, {}).get("args", (None,))[0] == n_obs
                    for f in frames
                )
                and np.ndim(scale) == 0
                and not np.isclose(scale, expected)
                and (
                    outside := [
                        name
                        for name, v in model_trace.items()
                        if v["type"] == "sample"
                        and name != self.param_output
                        and not rows & {f.name for f in v.get("cond_indep_stack") or ()}
                    ]
                )
            ):
                msg = (
                    f"`fit` trains on batches of {dataloader.batch_size} of {n_obs} "
                    "observations, but the kernel scales the output site "
                    f"{self.param_output!r} by {scale:g} for a batch rather than "
                    f"by {n_obs}/{dataloader.batch_size} = {expected:g}, so a batch "
                    "does not weigh as the whole data against site(s) "
                    f"{', '.join(map(repr, outside))} outside the output's plate. "
                    "Scale it in the kernel, as a plate whose `size` is the number of "
                    "observations and whose `subsample_size` is the number in each "
                    "batch does, with any site per observation and any factor chosen "
                    "on purpose, as for tempering, inside that plate, or fit with "
                    "`fit_on_batch`."
                )
                warn(msg, category=FitWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
            del model_trace, site
            # Batches split a site that counts as computing across observations: one
            # recorded by control flow, or one drawn as an event whose size follows the
            # rows, which the build tells at twice the batch's rows, or all of them,
            # growing the batch's fields alone
            self._build_kernel_spec(
                signature(self.kernel).bind(**{**batch, **kwargs_extra}).arguments,
                with_output=True,
                rows_more=min(2 * dataloader.batch_size, n_obs),
                whole=kwargs_extra.keys(),
            )
            if coupled := sorted(self._coupled_sites):
                msg = (
                    f"`fit` trains on batches of {dataloader.batch_size} of {n_obs} "
                    "observations, but the kernel computes across observations at "
                    f"site(s) {', '.join(map(repr, coupled))}, which the batches "
                    "split. Fit with `fit_on_batch`, or with a `batch_size` that "
                    "covers the data."
                )
                warn(msg, category=FitWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
            # Batches also split a latent site in the output's plate whose guide value
            # reads none of the batch arrays, or follows the plate's indices: kept per
            # row position or plate index, it trains on whichever rows a batch brings.
            # The guide is traced, not run, with the parameters of a first step's setup
            # fixed. The setup runs on a copy of the guide where one can be made, so a
            # model whose guide holds its own state stays as it was if the warning is
            # raised as an error.
            plates = self._dims.get(self.param_output, ())[:1]
            latent = {
                k
                for k in cast("KernelSpec", self._kernel_spec).sample_sites
                if k not in self._observed_sites
                and set(plates) & set(self._dims.get(k, ()))
            }
            if latent:
                svi = cast("SVI", self.inference)
                state = self._vi_state
                if state is None or svi.constrain_fn is None:
                    try:
                        guide = copy(svi.guide)
                    except Exception:
                        guide = svi.guide
                    svi = SVI(
                        svi.model,
                        guide,
                        svi.optim,
                        svi.loss,
                        **svi.static_kwargs,
                    )
                    init = svi.init(rng_subkey, **{**batch, **kwargs_extra})
                    state = init if state is None else state
                    del guide, init
                params = {
                    **svi.get_params(state),
                    **(getattr(state, "mutable_state", None) or {}),
                }
                closed, values = make_jaxpr(
                    lambda rows, idx: {
                        k: v["value"]
                        for k, v in trace(
                            seed(
                                substitute(
                                    substitute(svi.guide, data=params),
                                    # Indices the guide sets itself, as from a batched
                                    # kwarg, are kept
                                    substitute_fn=lambda msg: (
                                        idx
                                        if msg["type"] == "plate"
                                        and msg["name"] == plates[0]
                                        and msg["value"] is None
                                        else None
                                    ),
                                ),
                                self.rng_key,
                            ),
                        )
                        .get_trace(**rows, **kwargs_extra)
                        .items()
                        if k in latent and v["type"] == "sample"
                    },
                    return_shape=True,
                )(batch, jnp.arange(dataloader.batch_size))
                reads = {id(v) for v in closed.jaxpr.invars[:-1]}
                follows = {id(closed.jaxpr.invars[-1])}
                for eqn in closed.jaxpr.eqns:
                    for seen in (reads, follows):
                        if any(id(v) in seen for v in eqn.invars):
                            seen.update(map(id, eqn.outvars))
                latent = {
                    k
                    for k, v in zip(sorted(values), closed.jaxpr.outvars, strict=True)
                    if id(v) not in reads or id(v) in follows
                }
                del svi, state, params, closed
            if latent:
                msg = (
                    f"`fit` trains on batches of {dataloader.batch_size} of {n_obs} "
                    "observations, but the guide keeps a value per observation for "
                    f"site(s) {', '.join(map(repr, sorted(latent)))} that it does not "
                    "compute from the batch, so the batches pair those values with "
                    "other observations. Fit with `fit_on_batch`, with a `batch_size` "
                    "that covers the data, or with a guide that computes such a site "
                    "from each observation's inputs."
                )
                warn(msg, category=FitWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
        # One batch covers the data, so it moves to the device once for every epoch
        if (
            isinstance(X, ArrayLike)
            and isinstance(dataloader, ArrayLoader)
            and len(dataloader) == 1
            and epochs > 0
        ):
            dataloader = [_to_jax(dataloader.dataset.arrays)]

        logger.info("Performing variational inference optimization")
        losses: list[npt.NDArray] = []
        for epoch in range(epochs):
            losses_epoch: list[npt.NDArray] = []
            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                total=len(dataloader) if isinstance(dataloader, ArrayLoader) else None,
                disable=not progress,
                dynamic_ncols=True,
            )
            pending: Array | None = None
            for batch in pbar:
                fields = dict(batch)
                if missing := names - fields.keys():
                    msg = f"The data loader has no field named {min(missing)!r}."
                    raise ValueError(msg)
                _, loss = self.train_on_batch(
                    fields.pop(self.param_input),
                    fields.pop(self.param_output),
                    **fields,
                    **kwargs_extra,
                    rng_key=rng_subkey,
                )
                if pending is not None:
                    loss_batch = device_get(pending)
                    losses_epoch.append(loss_batch)
                    pbar.set_postfix({"loss": f"{float(loss_batch):.4f}"})
                pending = loss
            if pending is not None:
                losses_epoch.append(device_get(pending))
            if not losses_epoch:
                msg = (
                    f"The data loader yielded no batches in epoch {epoch + 1}; pass a "
                    "loader that can be iterated once per epoch."
                )
                raise ValueError(msg)
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

        # Clear the previous fit first: the draw below rejects diverged parameters
        self._is_fitted = False
        self._posterior = None
        logger.info("Drawing posterior samples (num_samples=%d)", num_samples)
        rng_key, rng_subkey = random.split(rng_key)
        model_kwargs = (
            {self.param_input: X, self.param_output: y, **kwargs}
            if isinstance(X, ArrayLike)
            else _to_jax({**batch, **kwargs_extra})
        )
        # Release the training copy before the draw; assigned, as `epochs=0` binds none
        dataloader = pbar = batch = fields = None
        kwargs_extra = {}
        self._posterior = _sample_forward(
            self.inference.guide,
            rng_keys=random.split(rng_subkey, num=num_samples),
            return_sites=cast("KernelSpec", self._kernel_spec).sample_sites,
            samples=None,
            params={
                **cast("SVIRunResult", self.vi_result).params,
                **(getattr(self._vi_state, "mutable_state", None) or {}),
            },
            intervention=None,
            model_kwargs=model_kwargs,
        )
        self._num_chains = 1
        self._num_samples = num_samples
        self._is_fitted = True

        return self

    def is_fitted(self) -> bool:
        """Return whether the model holds posterior samples."""
        return self._is_fitted

    def describe(self) -> dict[str, object]:
        """Describe the kernel and the posterior the model holds.

        Returns:
            Plain Python values: the kernel's name and arguments, the input and output
            parameter names, the inference method, the fitted state with the number of
            chains and draws, and the sites known from the trace or the posterior. Each
            site maps to its ``kind`` (``"latent"``, ``"observed"``, or
            ``"deterministic"``), the name of the ``distribution`` it samples (``None``
            for a deterministic site and where no trace recorded it, as in a model an
            earlier version pickled until a call traces the site again), its ``dims``
            after ``chain`` and ``draw`` where the trace named them (``None`` for the
            default ``<site>_dim_<i>`` names), the ``upstream`` sample sites and the
            array ``arguments`` of the kernel that its distribution or its deterministic
            value is computed from, through any deterministic site between (``None``
            where no trace derived them), and, for a site in the posterior, the
            ``draw_shape`` of one draw. The relations are derived from the kernel, so
            they hold what it computes, not what a feature of an input means. They name
            the input by the kernel's own parameter name, and a site computed from a
            site that ``scan`` or ``cond`` records can list in its place everything that
            enters that ``scan`` or ``cond``, whether or not it feeds the site. The
            ``arguments`` add up over the calls traced so far, since a call lists only
            those it passes as arrays. A ``factor`` term counts as observed, and
            ``param`` sites are not listed; their values after an SVI fit are in
            :attr:`~aimz.ImpactModel.vi_result`. The ``card`` is the
            :class:`~aimz.model.CausalCard` as plain values, or ``None``.
        """
        spec = self._kernel_spec
        sample_sites = spec.sample_sites if spec else ()
        return_sites = spec.return_sites if spec else ()
        sites: dict[str, dict[str, object]] = {}
        for name in (*sample_sites, *return_sites):
            if name in sites:
                continue
            if name == self.param_output or name in self._observed_sites:
                kind = "observed"
            elif name in sample_sites:
                kind = "latent"
            else:
                kind = "deterministic"
            dims = self._dims.get(name)
            upstream = self._site_upstream.get(name)
            arguments = self._site_arguments.get(name)
            sites[name] = {
                "kind": kind,
                "distribution": self._site_dists.get(name),
                "dims": None if dims is None else list(dims),
                "upstream": None if upstream is None else sorted(upstream),
                "arguments": None if arguments is None else sorted(arguments),
            }
            if self._posterior is not None and name in self._posterior:
                sites[name]["draw_shape"] = [
                    int(n) for n in np.shape(self._posterior[name])[1:]
                ]

        return {
            "kernel": getattr(self.kernel, "__name__", type(self.kernel).__name__),
            "arguments": list(signature(self.kernel).parameters),
            "param_input": self.param_input,
            "param_output": self.param_output,
            "inference": type(self.inference).__name__,
            "fitted": self._is_fitted,
            "num_chains": self._num_chains,
            "num_samples": self._num_samples,
            "sites": sites,
            "card": None if self._card is None else self._card.to_dict(),
        }

    def set_posterior_sample(
        self,
        posterior_sample: dict[str, Array],
        *,
        num_chains: int = 1,
    ) -> Self:
        """Set posterior samples drawn elsewhere, in place of a fit.

        Args:
            posterior_sample: Posterior samples by site name, with the draws along the
                leading axis of each array. A deterministic site is kept, but the kernel
                computes its values instead of taking them from the posterior. An
                observed site other than the output is kept, and a call that omits its
                data takes the site's values from the posterior instead of drawing them.
            num_chains: Number of chains the draws are stacked from, chain by chain.
                Output trees then keep the chains along their ``chain`` dimension.

        Returns:
            The model instance, treated as fitted.

        Raises:
            ValueError: If ``posterior_sample`` is empty, a value is 0-D, the batch
                shapes in ``posterior_sample`` are inconsistent, or a key is ``chain``
                or ``draw``.

        Warns:
            AimzWarning: If the output site is in ``posterior_sample``; it is removed.
                Also if the kernel has been traced and ``posterior_sample`` holds sites
                that no trace of it records so far, such as a guide's ``_auto_latent``;
                leave out any the kernel never samples, since one as long as the number
                of observations can still make a call rerun under ``shard_axis="draw"``.
            OutputWarning: If the kernel has been traced and ``posterior_sample`` has
                no draws for one of its latent sample sites, which the predictive
                methods then draw from the prior.

        Note:
            The kernel is not traced here. Prior predictive sampling or a call with an
            array input traces it, and calls then return its deterministic
            sites by default, as on a fitted model; until then a call with a data
            loader knows no deterministic sites, which have to be requested through
            ``return_sites``. If the kernel has ``param`` sites, also set
            :attr:`~aimz.ImpactModel.vi_result`, or the predictive methods use their
            initial values.
        """
        if self.param_output in posterior_sample:
            posterior_sample = {
                k: v for k, v in posterior_sample.items() if k != self.param_output
            }
            msg = f"The output site {self.param_output!r} is removed."
            warn(msg, category=AimzWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
        if reserved := sorted({"chain", "draw"}.intersection(posterior_sample)):
            msg = (
                f"The site name(s) {', '.join(map(repr, reserved))} are reserved for a "
                "dimension of the output trees. Rename the site(s)."
            )
            raise ValueError(msg)
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
        (num_samples,) = batch_shapes.pop()
        spec = self._kernel_spec
        if spec is not None and spec.traced:
            latent = self._fitted_sites - {*self._observed_sites, self.param_output}
            if missing := sorted(latent - posterior_sample.keys()):
                msg = (
                    "`posterior_sample` has no draws for the latent site(s) "
                    f"{', '.join(map(repr, missing))}; the predictive methods draw "
                    "them from the prior."
                )
                warn(
                    msg,
                    category=OutputWarning,
                    skip_file_prefixes=_SKIP_FILE_PREFIXES,
                )
            if foreign := sorted(
                posterior_sample.keys() - {*spec.sample_sites, *spec.return_sites},
            ):
                msg = (
                    f"`posterior_sample` holds site(s) {', '.join(map(repr, foreign))} "
                    "that no trace of the kernel records so far; leave out any it "
                    "never samples, since one as long as the number of observations "
                    "can still make a call rerun under `shard_axis='draw'`."
                )
                warn(msg, category=AimzWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
        # Release the placement of the posterior this call replaces
        self._streamer.place_posterior(None, sharding=None)
        self._posterior = posterior_sample
        self._num_samples = num_samples
        self._num_chains = num_chains
        if self._kernel_spec is None or not self._kernel_spec.traced:
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
        condition: dict | None = None,
        rng_key: Array | None = None,
        in_sample: bool = True,
        return_sites: str | Iterable[str] | None = None,
        return_datatree: bool = True,
        **kwargs: object,
    ) -> xr.DataTree | dict[str, npt.NDArray]:
        """Predict the output based on the fitted model, for one in-memory batch.

        Suited to small inputs, to models whose posterior samples cannot stream, and to
        callers that want no files written.

        Args:
            X: Input array with observations on the leading axis.
            intervention: Replacement values by sample site name, applied while
                sampling.
            condition: Values by sample site name that the sites take while sampling,
                such as the observed history of an output that an autoregressive
                recursion reads; a conditioned site returns these values, and one
                whose values the posterior holds keeps its posterior draws.
            rng_key: A pseudo-random number generator key, split from an internal key by
                default.
            in_sample: Put the samples in the ``posterior_predictive`` group, or in
                ``predictions`` if ``False``.
            return_sites: Names of the sites to return; by default the output and the
                deterministic sites.
            return_datatree: Return an :external:class:`~xarray.DataTree`, or a
                :class:`dict` if ``False``.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Posterior predictive samples. Posterior samples are included if available
            when a :external:class:`~xarray.DataTree` is returned; the dictionary
            contains only the requested sites.

        Raises:
            NotFittedError: If the model is not fitted.
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument, or
                ``condition`` is not a mapping, as when it is meant for a kernel
                argument of that name, which has to be renamed.
            ValueError: If ``intervention`` or ``condition`` names a site that is not a
                sample site of the kernel, or a posterior site that is not intervened
                on does not fit the observations being evaluated, as a site with one
                value per observation does not fit other observations, and a site sized
                by a count, such as the number of groups, does not fit another count.

        Warns:
            OutputWarning: If an intervened site reaches the output only through sites
                whose values are taken from the posterior or intervened on, so that
                the draws do not respond to it, if a requested return site is
                intervened on, since it comes back with its own draws, or missing from
                the output, if a conditioned site is one the posterior holds, since it
                keeps its posterior draws, or if the output site does not have one value
                per observation of the input.
        """
        _check_is_fitted(self)
        X = cast("Array", _validate_X_y_to_jax(X))

        args_bound = _to_jax(self._bind_kernel_args(X, kwargs=kwargs))
        # The checks below and the output's sites and names follow a trace of the
        # kernel with the call's own arguments
        spec = self._call_spec(args_bound)
        _validate_intervention(intervention, kernel_spec=spec)
        _validate_condition(
            condition,
            kernel_spec=spec,
            fixed=(self._posterior or {}).keys(),
        )
        _warn_unreachable_intervention(
            intervention,
            output=self.param_output,
            parents=self._site_parents,
            upstream=self._site_upstream,
            fixed=(self._posterior or {}).keys(),
        )

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        sites = _coerce_return_sites(
            return_sites,
            intervention=intervention,
            spec=spec or self._kernel_spec,
        )
        samples = device_get(
            _sample_forward(
                self.kernel,
                rng_keys=random.split(rng_key, num=self._num_samples),
                return_sites=sites,
                samples=self._streamer.place_posterior(
                    self.posterior,
                    sharding=None,
                    # Unless the call's own trace failed
                    unread=self._recomputed_sites()
                    if self._specs.get(_partition(args_bound)[1])
                    else frozenset(),
                ),
                params={
                    **self.vi_result.params,
                    **(getattr(self.vi_result.state, "mutable_state", None) or {}),
                    **_to_jax(condition or {}),
                }
                if self.vi_result is not None
                else _to_jax(condition or {}),
                intervention=_to_jax(intervention or {}),
                model_kwargs=args_bound,
            ),
        )
        if return_sites is not None:
            _warn_missing_return_sites(sites, returned=samples)
        out = samples.get(self.param_output)
        if out is not None and np.shape(out)[1:2] != (len(X),):
            msg = (
                f"The output site {self.param_output!r} does not have one value per "
                "observation of the input."
            )
            warn(msg, category=OutputWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)

        if not return_datatree:
            return samples

        return _build_datatree(
            samples,
            group="posterior_predictive" if in_sample else "predictions",
            posterior=self.posterior,
            num_chains=self._num_chains,
            dims=self._dims,
            attrs=_draw_attrs(rng_key),
        )

    def predict(
        self,
        X: ArrayLike | ArrayLoader | Iterable[Mapping[str, Array | np.ndarray]],
        *,
        intervention: dict | None = None,
        condition: dict | None = None,
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

        The batches are sampled and written to a Zarr store concurrently, or kept in
        host memory with ``store="memory"``.

        Args:
            X: Input array with observations on the leading axis, or a data loader
                yielding batch mappings keyed by kernel parameter names.
            intervention: Replacement values by sample site name, applied while
                sampling.
            condition: Values by sample site name that the sites take while sampling,
                such as the observed history of an output that an autoregressive
                recursion reads; a conditioned site returns these values, and one
                whose values the posterior holds keeps its posterior draws. Each value
                is passed whole, to every batch of a data loader too.
            rng_key: A pseudo-random number generator key, split from an internal key by
                default.
            in_sample: Put the samples in the ``posterior_predictive`` group, or in
                ``predictions`` if ``False``.
            return_sites: Names of the sites to return; by default the output and the
                deterministic sites.
            shard_axis: ``"obs"`` shards the input across devices and replicates the
                posterior; ``"draw"`` shards the posterior and replicates the input,
                which must then be an array.
            batch_size: Observations per batch under ``"obs"``, draws per batch under
                ``"draw"``, and the chunk size of the stored results. Chosen
                automatically if ``None``; ignored for a data loader.
            store: ``"persistent"`` streams the batches to a Zarr store under
                ``output_dir`` and returns a lazy tree recording its ``artifact_path``;
                ``"memory"`` keeps the batches in host memory.
            output_dir: Directory for the Zarr store, created if needed; by default a
                temporary directory of the model, removed by
                :meth:`~aimz.ImpactModel.cleanup`,
                :meth:`~aimz.ImpactModel.cleanup_models`, or garbage collection. Each
                call writes its own subdirectory, recorded as the tree's
                ``artifact_path`` attribute.
            progress: Whether to display a progress bar.
            **kwargs: Additional arguments passed to the model. Arrays aligned with
                ``X`` are batched with it; other values are passed whole, arrays
                traced and the rest static.

        Returns:
            Posterior predictive samples. Posterior samples are included if available.

        Raises:
            NotFittedError: If the model is not fitted.
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument,
                ``shard_axis="draw"`` is used with a data loader ``X``, or
                ``condition`` is not a mapping, as when it is meant for a kernel
                argument of that name, which has to be renamed.
            ValueError: If ``shard_axis`` is not ``"obs"`` or ``"draw"``, ``store`` is
                not ``"persistent"`` or ``"memory"``, ``output_dir`` is passed with
                ``store="memory"``, ``X`` is 0-D or empty, ``intervention`` or
                ``condition`` names a site that is not a sample site of the kernel, or
                a posterior site that is not intervened on does not fit the observations
                being evaluated, as a site with one value per observation does not fit
                other observations, and a site sized by a count, such as the number of
                groups, does not fit another count.
            NotImplementedError: With a data loader, if a return site has no
                observation axis, unless it is intervened on or the kernel does not
                run on one observation fewer than a device evaluates (two when it
                evaluates one), or if its axis-1 size does not match the number of
                observations a device evaluates from a batch.

        Warns:
            OutputWarning: If an intervened site reaches the output only through sites
                whose values are taken from the posterior or intervened on, so that
                the draws do not respond to it, if a requested return site is
                intervened on, since it comes back with its own draws, or missing from
                the output, if a conditioned site is one the posterior holds, since it
                keeps its posterior draws, or if the output site does not have one value
                per observation of an array input; under ``shard_axis="obs"`` the call
                then reruns under
                ``shard_axis="draw"``. Also if the batches of a data loader split a
                kernel that computes across observations, once a call has traced the
                kernel.
            PerformanceWarning: If ``shard_axis="obs"`` cannot split an array input,
                as for a return site without an observation axis, a latent site with
                one value per observation or another size that follows the
                observations, or a ``condition`` value with one entry per observation,
                which is passed whole, so the call reruns under ``shard_axis="draw"``.
        """
        _check_is_fitted(self)
        _validate_shard_axis(shard_axis, X=X)
        _validate_batch_size(batch_size, X=X)
        _validate_store(store, output_dir=output_dir)
        _validate_aligned_inputs(X, y=None)

        # The checks below and planning follow a trace of the kernel with the call's
        # own arguments; a data loader has none before its batches, so it takes the
        # sites of every trace so far
        spec = self._kernel_spec
        if isinstance(X, ArrayLike):
            spec = self._call_spec(self._bind_kernel_args(X, kwargs=kwargs))
        _validate_intervention(intervention, kernel_spec=spec)
        _validate_condition(
            condition,
            kernel_spec=spec,
            fixed=(self._posterior or {}).keys(),
        )
        _warn_unreachable_intervention(
            intervention,
            output=self.param_output,
            parents=self._site_parents,
            upstream=self._site_upstream,
            fixed=(self._posterior or {}).keys(),
        )

        requested = return_sites is not None
        return_sites = _coerce_return_sites(
            return_sites,
            intervention=intervention,
            spec=spec or self._kernel_spec,
        )
        shard_axis, batch_size = _plan_execution(
            X,
            shard_axis=shard_axis,
            batch_size=batch_size,
            num_samples=self._num_samples,
            nbytes=_output_nbytes(self._site_sizes, return_sites),
            posterior=self.posterior,
            return_sites=return_sites,
            kwargs=kwargs,
            intervention=intervention,
            condition=condition,
            kernel=self.kernel,
            rng_key=self.rng_key,
            param_input=self.param_input,
            param_output=self.param_output,
            kernel_spec=self._kernel_spec,
            coupled_sites=self._coupled_sites,
            observed_sites=self._observed_sites,
            num_devices=self._num_devices,
            plan_cache=self._plan_cache,
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
                params={
                    **self.vi_result.params,
                    **(getattr(self.vi_result.state, "mutable_state", None) or {}),
                    **_to_jax(condition or {}),
                }
                if self.vi_result is not None
                else _to_jax(condition or {}),
                intervention=intervention,
                unread=self._recomputed_sites(),
            ),
            store=store,
            output_dir=output_dir,
            group=group,
            attrs=_draw_attrs(
                rng_key,
                X=X,
                shard_axis=shard_axis,
                batch_size=batch_size,
                num_devices=self._num_devices,
            ),
            requested=return_sites if requested else (),
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

        The effect is intervention minus baseline for every site of the shared
        predictive group, draw by draw. Precomputed outputs of
        :meth:`~aimz.ImpactModel.sample_prior_predictive` give the effect under the
        prior, also for an unfitted model; pass the same ``rng_key`` to both calls so
        the scenarios share their draws.

        Args:
            output_baseline: Precomputed output for the baseline scenario.
            output_intervention: Precomputed output for the intervention scenario.
            args_baseline: Arguments of the prediction method for the baseline
                scenario, keyed by its parameter names, as ``{"X": X}`` whatever the
                kernel calls its input, used when ``output_baseline`` is not given.
            args_intervention: Arguments of the prediction method for the intervention
                scenario, keyed the same way, used when ``output_intervention`` is not
                given.
            on_batch: Compute the scenarios with
                :meth:`~aimz.ImpactModel.predict_on_batch` instead of
                :meth:`~aimz.ImpactModel.predict`.

        Returns:
            The effect of the intervention, with the posterior samples unless the
            effect is under the prior. A scenario streamed to disk records its
            artifact path in the ``artifact_path_baseline`` or
            ``artifact_path_intervention`` attribute.

        Raises:
            NotFittedError: If the model is not fitted and ``output_baseline`` or
                ``output_intervention`` is not provided.
            TypeError: If a ``condition`` passed through ``args_baseline`` or
                ``args_intervention`` is not a mapping, as when it is meant for a
                kernel argument of that name, which has to be renamed.
            ValueError: If neither ``output_baseline`` nor ``args_baseline`` is
                provided, if neither ``output_intervention`` nor
                ``args_intervention`` is provided, if an ``intervention`` or
                ``condition`` passed through ``args_baseline`` or
                ``args_intervention`` names a site that is not a sample site of the
                kernel, if a posterior site that is not intervened on does not fit the
                observations a scenario evaluates, or if the scenarios do not share a
                predictive group.

        Warns:
            OutputWarning: If the scenarios differ in dimension sizes or coordinate
                labels, if they hold different posterior samples, or if the output,
                or any site under the prior, was not drawn in both with the same key,
                sharding strategy, and batching.
        """
        if output_baseline is None or output_intervention is None:
            _check_is_fitted(self)
        _predict = cast(
            "Callable[..., xr.DataTree]",
            self.predict_on_batch if on_batch else self.predict,
        )
        named = {
            name
            for name, p in signature(_predict).parameters.items()
            if p.kind is not p.VAR_KEYWORD
        }
        for output, args in (
            (output_baseline, args_baseline),
            (output_intervention, args_intervention),
        ):
            if output is None and args is not None:
                # As its call does, a scenario with an array input is checked against
                # a trace of the kernel with its own arguments
                spec = (
                    self._call_spec(
                        self._bind_kernel_args(
                            args["X"],
                            kwargs={k: v for k, v in args.items() if k not in named},
                        ),
                    )
                    if isinstance(args.get("X"), ArrayLike)
                    else self._kernel_spec
                )
                _validate_intervention(args.get("intervention"), kernel_spec=spec)
                # The scenario's own call warns for sites the posterior holds
                _validate_condition(
                    args.get("condition"),
                    kernel_spec=spec,
                    fixed=frozenset(),
                )

        if output_baseline is None and args_baseline is None:
            msg = "Either `output_baseline` or `args_baseline` must be provided."
            raise ValueError(msg)
        if output_intervention is None and args_intervention is None:
            msg = (
                "Either `output_intervention` or `args_intervention` must be provided."
            )
            raise ValueError(msg)

        overrides = {"return_datatree": True} if on_batch else {}

        # Lazily generated scenarios share one sampling key
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
        # A scenario reruns under `shard_axis="draw"` when its own sites cannot be split
        # by observation. A scenario drawn here with the other's key and an array input
        # follows it, so that their draws stay paired, and its first output is removed.
        axes = (
            dt_baseline[group].attrs.get("shard_axis"),
            dt_intervention[group].attrs.get("shard_axis"),
        )
        same_key = np.array_equal(
            np.asarray(dt_baseline[group].attrs.get("rng_key")),
            np.asarray(dt_intervention[group].attrs.get("rng_key")),
        )
        rerun = {"shard_axis": "draw", "batch_size": None}
        stale = None
        if (
            same_key
            and axes == ("obs", "draw")
            and output_baseline is None
            and isinstance(cast("dict", args_baseline)["X"], ArrayLike)
        ):
            stale = dt_baseline
            dt_baseline = _predict(**{**cast("dict", args_baseline), **rerun})
        elif (
            same_key
            and axes == ("draw", "obs")
            and output_intervention is None
            and isinstance(cast("dict", args_intervention)["X"], ArrayLike)
        ):
            stale = dt_intervention
            dt_intervention = _predict(**{**cast("dict", args_intervention), **rerun})
        if stale is not None and "artifact_path" in stale.attrs:
            rmtree(stale.attrs["artifact_path"], ignore_errors=True)

        # A site whose dimensions are named differently in the two scenarios, as when
        # only one of them returns a site that a default name skips, takes the
        # intervention's name for each such dimension, or a new default name where the
        # baseline already uses that one, and keeps its labels, so that its effect
        # aligns instead of broadcasting
        baseline = dt_baseline[group].to_dataset()
        intervention = dt_intervention[group].to_dataset()
        if renamed := sorted(
            str(site)
            for site in baseline.data_vars.keys() & intervention.data_vars.keys()
            if baseline[site].ndim == intervention[site].ndim
            and set(baseline[site].dims) != set(intervention[site].dims)
        ):
            taken = {
                str(name)
                for ds in (baseline, intervention)
                for name in (*ds.variables, *ds.dims)
            }
            used = {str(name) for name in (*baseline.variables, *baseline.dims)}
            arrays_baseline, arrays_intervention = {}, {}
            for site in renamed:
                names_baseline, names_intervention = {}, {}
                for old, new in zip(
                    [d for d in baseline[site].dims if d not in {"chain", "draw"}],
                    [d for d in intervention[site].dims if d not in {"chain", "draw"}],
                    strict=True,
                ):
                    if old == new:
                        continue
                    if new in used:
                        names_intervention[new] = next(
                            name
                            for i in count()
                            if (name := f"{site}_dim_{i}") not in taken
                        )
                        taken.add(names_intervention[new])
                    names_baseline[old] = names_intervention.get(new, new)
                    used.add(names_baseline[old])
                arrays_baseline[site] = baseline[site].rename(names_baseline)
                arrays_intervention[site] = intervention[site].rename(
                    names_intervention
                )
            baseline = baseline.drop_vars(list(arrays_baseline)).assign(arrays_baseline)
            intervention = intervention.drop_vars(list(arrays_intervention)).assign(
                arrays_intervention,
            )
            # A coordinate on a dimension no site uses any more would clash with a site
            # of its name
            baseline, intervention = (
                ds.drop_vars(
                    [
                        name
                        for name in ds.coords
                        if not set(ds.variables[name].dims) <= dims
                    ],
                )
                for ds, dims in (
                    (
                        ds,
                        {d for name in ds.data_vars for d in ds.variables[name].dims},
                    )
                    for ds in (baseline, intervention)
                )
            )
            msg = (
                f"The site(s) {', '.join(map(repr, renamed))} have other dimension "
                "names in the two scenarios, so their effects take the names of the "
                "intervention, or new default names where the baseline uses those."
            )
            warn(msg, category=OutputWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
        out = xr.DataTree(name="root")
        out[group] = intervention - baseline
        # The key and the batching decide the draws of the sites a call samples anew:
        # every site under the prior, and the output otherwise
        attrs_baseline = dt_baseline[group].attrs
        attrs_intervention = dt_intervention[group].attrs
        unmatched = [
            name
            for name in ("rng_key", "shard_axis", "num_devices", "batch_size")
            if not np.array_equal(
                np.asarray(attrs_baseline.get(name)),
                np.asarray(attrs_intervention.get(name)),
            )
        ]
        unpaired = [
            site
            for site in out[group].data_vars
            if group == "prior_predictive" or site == self.param_output
        ]
        if (
            "rng_key" in attrs_baseline
            and "rng_key" in attrs_intervention
            and unmatched
            and unpaired
        ):
            msg = (
                f"Baseline and intervention differ in "
                f"{', '.join(map(repr, unmatched))}; their draws of "
                f"{', '.join(map(repr, unpaired))} in group {group!r} are not paired."
            )
            warn(msg, category=OutputWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
        if self.posterior and group != "prior_predictive":
            out["posterior"] = _dict_to_datatree(
                self.posterior,
                num_chains=self._num_chains,
                dims=self._dims,
            )
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

    def build_scenario(
        self,
        args: Mapping[str, object],
        changes: Mapping[str, object],
        *,
        observations: ArrayLike | None = None,
    ) -> dict[str, object]:
        """Build a scenario from another by editing its inputs by name.

        A scenario is an argument dictionary of
        :meth:`~aimz.ImpactModel.estimate_effect`: the input under ``X`` and the
        keyword arguments of the kernel, with any ``intervention`` and ``condition``
        mapping. The names are those the card declares, so a treatment recorded in
        several features changes in all of them at once, and the arrays given are
        never touched: an edited input comes back as a copy and every other entry as
        given. Nothing is predicted, and nothing here establishes that a change is a
        treatment; :meth:`~aimz.ImpactModel.check_scenario` compares the two
        scenarios.

        Args:
            args: Arguments of the baseline scenario, with an array input.
            changes: The new values by name. A name is a treatment of the card, which
                edits every input it spans, a feature by its declared name, or as
                ``X[i]`` where no features are declared, the input ``X``, or a keyword
                array. A value is cast to the input's data type, which the edited
                copy keeps, and broadcast to the input's shape; a callable receives
                the input's current values and returns the new ones.
            observations: The observations to change, as a boolean mask or integer
                indices along the input's leading axis; the others keep their
                baseline values. By default every observation.

        Returns:
            The intervention scenario, with the keys of ``args``.

        Raises:
            TypeError: If ``args`` holds no array under ``X``, as with a data loader.
            ValueError: If the card declares features whose number differs from the
                input's, if a name is none of the card's treatments, the input ``X``,
                its features as ``changes`` names them and the keyword arrays, if a
                treatment spans a name that is none of these inputs, as a sample
                site is, which ``intervention`` sets, if a value does not broadcast
                to the input's shape, or if ``observations`` does not fit the input's
                observations or is given for an input without a value per
                observation.
        """
        own = {
            name
            for method in (self.predict, self.predict_on_batch)
            for name, p in signature(method).parameters.items()
            if p.kind is not p.VAR_KEYWORD
        }

        return _build_scenario(
            args,
            changes,
            observations=observations,
            card=self._card,
            kernel_spec=self._kernel_spec,
            own=own,
        )

    def check_scenario(
        self,
        args_baseline: Mapping[str, object],
        args_intervention: Mapping[str, object],
        *,
        by: Mapping[str, ArrayLike] | None = None,
        pattern: Iterable[str] | Mapping[str, Iterable[str]] | None = None,
        absent: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Compare two scenarios against the declared causal scope and the data.

        The scenarios are the argument dictionaries of
        :meth:`~aimz.ImpactModel.estimate_effect`: the input under ``X``, the keyword
        arguments of the kernel, and the ``intervention`` and ``condition`` mappings.
        Nothing is predicted. The scenarios are compared observation by observation,
        and the observations that change are compared with the baseline's. Every
        value returned is a fact about the scenarios and the data, not a verdict:
        what a share means for an answer is the caller's decision, and nothing here
        establishes that a contrast is causal.

        Args:
            args_baseline: Arguments of the baseline scenario, with an array input.
            args_intervention: Arguments of the intervention scenario, over the same
                observations.
            by: Label arrays an effect will be grouped by, keyed by name. Only the
                names are read, to flag those the card declares post-treatment.
            pattern: The inputs whose presence pattern the support share is
                computed over: names, each a group of its own, or a
                mapping of group name to the names whose presence it combines. By
                default the card's treatments, each over the names it spans that
                hold one value per observation, ``X`` standing for the input's
                features, else the inputs that changed and hold one, each a group
                of its own. A wide pattern makes support small, so group the
                inputs that belong together, such as the features one treatment
                is recorded in.
            absent: The value that counts as absence for an input, by name, for inputs
                whose zero is elsewhere after preprocessing; zero for every other
                input. A value is compared exactly, a Python float or a float64 in
                the input's precision, so give it as the input holds it.

        Returns:
            ``n_obs``, the observations compared. ``changed``, each input that
            differs, with the share of observations whose value differs: a feature of
            the input by its declared name or ``X[i]``, a keyword array
            by its argument name, a site of ``intervention`` or ``condition`` by its
            name, and a value that is not per observation with share one, compared
            leaf by leaf where it holds several arrays. A keyword argument given in
            only one scenario counts as changed, and so does a site that only one
            scenario sets: set it in both to have it compared observation by
            observation. Each entry is compared with the same entry of the other
            scenario, and a name that several share, as a site and the keyword
            argument of its name do, has changed where any of them has. Of the
            entries of such a name that hold values per observation, one stands for
            it in ``extrapolation``, over the observations where that entry differs,
            and in ``pattern`` and ``support``: the last that differs, or the last
            where none does, in the order in which a call lets them stand, a
            feature, the keyword argument, what ``condition`` sets, what
            ``intervention`` sets. An entry that changes as a whole voids those
            before it, a feature apart, and where none is left the name holds no
            value per observation.
            ``obs_changed``, the share of observations where anything differs.
            ``undeclared``, the changed names that no treatment of the card spans,
            empty when the card declares none. ``extrapolation``, for each changed
            numeric input with one value per observation, the share of its changed
            observations whose intervention value lies outside the range the
            baseline observes.
            ``pattern``, the groups the presence pattern is made of. ``support``, the
            share of changed observations whose intervention pattern occurs among the
            baseline's observations; ``None`` when nothing changed or no pattern
            applies. A changed name without one value per observation is in no
            group, so ``support`` says nothing about it. ``post_treatment``, the
            names in ``by`` that the card declares post-treatment.

        Raises:
            TypeError: If a scenario holds no array under ``X``, as with a data loader.
            ValueError: If the two inputs differ in shape, if the card declares
                features whose number differs from the input's, or if ``pattern`` or
                ``absent`` names an input without one value per observation in both
                scenarios.

        Warns:
            ScopeWarning: If the card declares treatments and an undeclared input
                changed, or if ``by`` names a variable the card declares
                post-treatment.
        """
        own = {
            name
            for method in (self.predict, self.predict_on_batch)
            for name, p in signature(method).parameters.items()
            if p.kind is not p.VAR_KEYWORD
        }

        return _check_scenario(
            args_baseline,
            args_intervention,
            by=by,
            pattern=pattern,
            absent=absent,
            card=self._card,
            own=own,
        )

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
        """Compute the log-likelihood of the output site under the given model.

        Other observed sites and ``factor`` terms of the kernel are not scored.

        The batches are computed and written to a Zarr store concurrently, or kept in
        host memory with ``store="memory"``.

        Args:
            X: Input array with observations on the leading axis, or a data loader
                yielding batch mappings keyed by kernel parameter names, including the
                observed output.
            y: Output array with observations on the leading axis. Must be ``None``
                if ``X`` is a data loader.
            shard_axis: ``"obs"`` shards the input across devices and replicates the
                posterior; ``"draw"`` shards the posterior and replicates the input,
                which must then be an array. Without posterior samples, ``"obs"`` is
                used.
            batch_size: Observations per batch under ``"obs"``, draws per batch under
                ``"draw"``, and the chunk size of the stored results. Chosen
                automatically if ``None``; ignored for a data loader.
            store: ``"persistent"`` streams the batches to a Zarr store under
                ``output_dir`` and returns a lazy tree recording its ``artifact_path``;
                ``"memory"`` keeps the batches in host memory.
            output_dir: Directory for the Zarr store, created if needed; by default a
                temporary directory of the model, removed by
                :meth:`~aimz.ImpactModel.cleanup`,
                :meth:`~aimz.ImpactModel.cleanup_models`, or garbage collection. Each
                call writes its own subdirectory, recorded as the tree's
                ``artifact_path`` attribute.
            progress: Whether to display a progress bar.
            **kwargs: Additional arguments passed to the model. Arrays aligned with
                ``X`` are batched with it; other values are passed whole, arrays
                traced and the rest static.

        Returns:
            Log-likelihood values, one per draw and observation, or one per draw when
            the output is drawn as one event over all the observations. Posterior
            samples are included if available.

        Raises:
            NotFittedError: If the model is not fitted.
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument,
                ``shard_axis="draw"`` is used with a data loader ``X``, or ``y`` is
                passed with a data loader ``X``.
            ValueError: If ``shard_axis`` is not ``"obs"`` or ``"draw"``, ``store`` is
                not ``"persistent"`` or ``"memory"``, ``output_dir`` is passed with
                ``store="memory"``, ``y`` is missing when ``X`` is array-like, ``y``
                does not share ``X``'s leading-axis size, or a posterior site does not
                fit the observations being evaluated, as a site with one value per
                observation does not fit other observations, and a site sized by a
                count, such as the number of groups, does not fit another count.
            NotImplementedError: If a return site's axis-1 size does not match the
                input batch size (``shard_axis="obs"`` only).

        Warns:
            OutputWarning: If the batches of a data loader split a kernel that
                computes across observations, once a call has traced the kernel.
            PerformanceWarning: If ``shard_axis="obs"`` cannot split an array input,
                as for a kernel that computes across observations or a posterior site
                as long as the number of observations or with another size that
                follows it, so the call reruns under ``shard_axis="draw"``.
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
        # A posterior set on a new model leaves the kernel untraced; planning needs its
        # sites. A kernel that runs on its batches but not on all the rows at once, such
        # as one whose subsampling plate is smaller than the input, plans untraced.
        if isinstance(X, ArrayLike) and not (
            self._kernel_spec and self._kernel_spec.traced
        ):
            try:
                self._build_kernel_spec(
                    _to_jax(
                        signature(self.kernel)
                        .bind(**{self.param_input: X, self.param_output: y, **kwargs})
                        .arguments,
                    ),
                    with_output=True,
                )
            except Exception as e:
                # A rule the kernel breaks, or a warning raised as an error, surfaces
                if isinstance(e, (KernelValidationError, AimzWarning)):
                    raise

        # Without a posterior there is nothing to shard by draw
        if not self.posterior:
            shard_axis = "obs"
        # One value of the default float type per observation, whatever the output's
        # own shape and type
        shard_axis, batch_size = _plan_execution(
            X,
            shard_axis=shard_axis,
            batch_size=batch_size,
            num_samples=self._num_samples,
            nbytes=(jnp.result_type(float).itemsize, 0),
            posterior=self.posterior,
            kwargs={**kwargs, self.param_output: y},
            kernel=self.kernel,
            rng_key=self.rng_key,
            param_input=self.param_input,
            param_output=self.param_output,
            kernel_spec=self._kernel_spec,
            coupled_sites=self._coupled_sites,
            observed_sites=self._observed_sites,
            num_devices=self._num_devices,
            plan_cache=self._plan_cache,
        )
        # One batch on one device keeps the rows of an output computed across them
        # whole, but under `obs` the log-likelihood of one drawn as an event over the
        # rows has no row axis to stream; `draw` keeps them whole too
        if (
            shard_axis == "obs"
            and self.posterior
            and isinstance(X, ArrayLike)
            and self.param_output in self._coupled_sites
        ):
            shard_axis, batch_size = _plan_execution(
                X,
                shard_axis="draw",
                batch_size=None,
                num_samples=self._num_samples,
                nbytes=(jnp.result_type(float).itemsize, 0),
                posterior=self.posterior,
                kernel=self.kernel,
                rng_key=self.rng_key,
                param_input=self.param_input,
                param_output=self.param_output,
                kernel_spec=self._kernel_spec,
                coupled_sites=self._coupled_sites,
                observed_sites=self._observed_sites,
                num_devices=self._num_devices,
                plan_cache=self._plan_cache,
            )
        # A keyword array as long as the rows each device evaluates, but not as the
        # input, is passed whole, so every device would score its rows with the same
        # values; as in `predict`, the call fails when the kernel does not run on all
        # the rows with it but does once the array has as many
        if shard_axis == "obs" and isinstance(X, ArrayLike):
            n_obs = len(cast("Sized", X))
            rows = -(-cast("int", batch_size) // self._num_devices)
            if rows != n_obs and any(
                np.shape(leaf)[:1] == (rows,)
                for leaf in _partition(_group_kwargs(kwargs, n_obs=n_obs)[1])[0]
            ):
                arrays, static = _partition(
                    {self.param_input: X, self.param_output: y, **kwargs},
                )
                # The outcome depends only on the call's shapes, constants and rows per
                # device, so a repeated call that passed reuses it
                check = (
                    "rows",
                    rows,
                    tuple((np.shape(a), jnp.result_type(a)) for a in arrays),
                    static,
                )
                if check not in self._plan_cache:

                    def run(arrays: list) -> None:
                        with validation_enabled(is_validate=False):
                            trace(
                                substitute(
                                    seed(self.kernel, rng_seed=self.rng_key),
                                    substitute_fn=init_to_sample,
                                ),
                            ).get_trace(**_combine(arrays, static))

                    try:
                        eval_shape(run, arrays)
                    # A kernel that does not run on all the rows anyway, such as one
                    # whose subsampling plate is smaller than the input, keeps its
                    # batches
                    except Exception:
                        try:
                            eval_shape(
                                run,
                                [
                                    ShapeDtypeStruct(
                                        (n_obs, *np.shape(leaf)[1:]),
                                        jnp.result_type(leaf),
                                    )
                                    if np.shape(leaf)[:1] == (rows,)
                                    else leaf
                                    for leaf in arrays
                                ],
                            )
                            misused = True
                        except Exception:
                            misused = False
                        if misused:
                            raise
                    self._plan_cache[check] = (frozenset(), False, {})

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
                kernel=self.kernel,
                posterior=self.posterior,
                params={
                    **self.vi_result.params,
                    **(getattr(self.vi_result.state, "mutable_state", None) or {}),
                }
                if self.vi_result is not None
                else None,
                y=y,
                unread=self._recomputed_sites(),
            ),
            store=store,
            output_dir=output_dir,
            group="log_likelihood",
        )

    def cleanup(self) -> None:
        """Remove the model's temporary directory, if it exists.

        An explicit ``output_dir`` is left alone. Garbage collection also removes the
        directory, but not at a guaranteed time.

        See Also:
            :meth:`~aimz.ImpactModel.cleanup_models`
        """
        if hasattr(self, "_temp_dir") and self._temp_dir is not None:
            temp_dir = self._temp_dir.name
            self._temp_dir.cleanup()
            self._temp_dir = None
            logger.info("Temporary directory cleaned up at: %s", temp_dir)

    @classmethod
    def cleanup_models(cls) -> None:
        """Remove the temporary directories of all :class:`~aimz.ImpactModel` instances.

        See Also:
            :meth:`~aimz.ImpactModel.cleanup`
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
