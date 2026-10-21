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
from copy import copy
from datetime import UTC, datetime
from functools import partial
from inspect import signature, stack
from itertools import count
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
    ShapeDtypeStruct,
    default_backend,
    device_get,
    eval_shape,
    jit,
    local_device_count,
    local_devices,
    make_jaxpr,
    make_mesh,
    random,
)
from jax.sharding import AxisType, NamedSharding, PartitionSpec
from jax.typing import ArrayLike
from numpyro import validation_enabled
from numpyro.distributions import Distribution, Independent
from numpyro.handlers import do, seed, substitute, trace
from numpyro.infer import MCMC, SVI, init_to_sample
from numpyro.infer.inspect import get_dependencies
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
from aimz.utils._kwargs import (
    _combine,
    _group_kwargs,
    _partition,
    _split_intervention,
    _split_intervention_fields,
    _to_jax,
)
from aimz.utils._validation import (
    _check_is_fitted,
    _validate_aligned_inputs,
    _validate_batch_size,
    _validate_condition,
    _validate_group,
    _validate_intervention,
    _validate_kernel_body,
    _validate_shard_axis,
    _validate_store,
    _validate_X_y_to_jax,
    _warn_unreachable_intervention,
)
from aimz.utils.data import ArrayLoader
from aimz.utils.data._input_setup import (
    _fits_single_batch,
    _resolve_batch_size,
    _setup_inputs,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sized

    from dask.array import Array as DaskArray

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
            rng_key: A typed key array from :external:func:`jax.random.key`.
            inference: An :external:class:`~numpyro.infer.svi.SVI` or
                :external:class:`~numpyro.infer.mcmc.MCMC` instance.
            param_input: Name of the ``kernel`` parameter for the input data.
            param_output: Name of the ``kernel`` parameter for the output data.

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
        self._coupled_sites: set[str] = set()
        self._site_parents: dict[str, set[str]] = {}
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

    def _init_runtime_attrs(self) -> None:
        """Initialize runtime attributes."""
        self._fn_vi_update: Callable | None = None
        # The return sites a streamed call cannot split, whether the kernel failed on
        # one row fewer, and the posterior sites whose size follows the rows, by the
        # call's shapes and constants, and the calls that passed `log_likelihood`'s
        # check of keyword arrays as long as each device's rows
        self._plan_cache: dict[
            tuple,
            tuple[frozenset[str], bool, dict[str, tuple[int, ...]]],
        ] = {}
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
        self.__dict__.setdefault("_coupled_sites", set())
        self.__dict__.setdefault("_site_parents", {})
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
    ) -> None:
        """Trace the kernel and record its sites, unless a sufficient spec exists.

        A spec traced without an observed output is upgraded by a trace with one,
        merging the sites, since a kernel may define sites only when data are supplied.
        The trace is abstract, unless the kernel reads a concrete value of an argument.
        A site marked for an event as long as the rows loses its mark when a trace at
        ``rows_more`` rows, twice the rows by default, gives it the same event length;
        the arguments named in ``whole`` keep their shape there, and a failed trace
        keeps every mark.
        """
        if self._kernel_spec:
            if with_output:
                if self._kernel_spec.output_observed:
                    return
            elif self._kernel_spec.traced:
                return
        leaves, static = _partition(args_bound)

        def sites_at(arrays: list) -> dict[str, dict]:
            sites: dict[str, dict] = {}

            def record(arrays: list) -> None:
                with validation_enabled(is_validate=False):
                    model_trace = trace(
                        substitute(
                            seed(self.kernel, rng_seed=self.rng_key),
                            substitute_fn=init_to_sample,
                        ),
                    ).get_trace(**_combine(arrays, static))
                # Each value keeps only its shape and type
                sites.update(
                    {
                        k: {
                            **v,
                            "value": ShapeDtypeStruct(
                                v["value"].shape,
                                v["value"].dtype,
                            ),
                        }
                        if hasattr(v.get("value"), "shape")
                        else v
                        for k, v in model_trace.items()
                    },
                )

            eval_shape(record, arrays)

            return sites

        # An abstract trace runs nothing; a kernel that reads a concrete value of an
        # argument, as through `int`, runs once instead
        try:
            model_trace = sites_at(leaves)
        except Exception:
            with validation_enabled(is_validate=False):
                model_trace = trace(
                    substitute(
                        seed(self.kernel, rng_seed=self.rng_key),
                        substitute_fn=init_to_sample,
                    ),
                ).get_trace(**args_bound)
        _validate_kernel_body(
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
        # Dimension names: the plates, outermost first, then the event dimensions named
        # in the site's `infer` dictionary, none of them a site's in this trace or an
        # earlier one. A site that `scan` records keeps the default names, as `scan`
        # stacks its values along a leading time axis and `cond` does not. Only sample
        # sites carry the mark of `scan`, so a deterministic site recorded by control
        # flow keeps them only in a trace with a marked sample site.
        rows = getattr(args_bound[self.param_input], "shape", ())[:1]
        prev = self._kernel_spec
        scanned = {
            k
            for k, v in model_trace.items()
            if "_scan_current_index" in (v.get("infer") or {})
        }
        taken = {"chain", "draw", *sample_sites, *return_sites}
        if prev is not None and prev.traced:
            taken |= {*prev.sample_sites, *prev.return_sites}
        events = []
        for k, v in model_trace.items():
            if v["type"] not in {"sample", "deterministic"}:
                continue
            frames = sorted(v.get("cond_indep_stack") or (), key=lambda f: f.dim)
            event_dims = (v.get("infer") or {}).get("event_dims", ())
            names = (*(f.name for f in frames), *event_dims)
            shape = getattr(v["value"], "shape", ())
            names_apply = (
                k not in scanned
                and not (
                    scanned
                    and v["type"] == "deterministic"
                    and v.get("_control_flow_done")
                )
                and 0 < len(names) <= len(shape)
                and all(isinstance(name, str) for name in names)
                and len(set(names)) == len(names)
                and not taken & set(names)
                and shape[: len(frames)] == tuple(f.size for f in frames)
            )
            if names_apply:
                self._dims[k] = names
            else:
                # Names that an earlier trace applied give way too
                self._dims.pop(k, None)
                if event_dims:
                    msg = (
                        f"The event dimension names {tuple(event_dims)!r} of site "
                        f"{k!r} cannot apply, so the site keeps the default dimension "
                        "names."
                    )
                    warn(
                        msg,
                        category=OutputWarning,
                        skip_file_prefixes=_SKIP_FILE_PREFIXES,
                    )
            # Values per draw: per observation when the leading axis follows the input
            # rows, else whole
            if hasattr(v["value"], "shape"):
                self._site_sizes[k] = (
                    (math.prod(shape[1:]), 0)
                    if shape[:1] == rows
                    else (0, math.prod(shape))
                )
            if v.get("is_observed"):
                self._observed_sites.add(k)
            # A site recorded by control flow, which NumPyro marks alike for `scan` and
            # `cond`, or drawn as one event over the rows, counts as computing across
            # observations. Each trace decides anew for the sites it records, so
            # a trace over all the rows clears a one-element event that a one-row trace
            # took for one over the rows.
            fn = v.get("fn")
            if v.get("_control_flow_done"):
                self._coupled_sites.add(k)
            elif (
                isinstance(fn, Distribution)
                and not isinstance(fn, Independent)
                and fn.batch_shape[:1] != rows
                and fn.event_shape[:1] == rows
            ):
                self._coupled_sites.add(k)
                events.append(k)
            else:
                self._coupled_sites.discard(k)
        # An event that keeps its length on more rows is global, as long as the rows
        # only by chance; a kernel that does not run on more rows keeps every mark
        if events:
            more = rows_more or 2 * math.prod(rows)
            try:
                grows = [
                    name not in whole
                    for name, value in args_bound.items()
                    for _ in _partition({name: value})[0]
                ]
                wider = sites_at(
                    [
                        ShapeDtypeStruct(
                            (more, *leaf.shape[1:])
                            if grow and leaf.shape[:1] == rows
                            else leaf.shape,
                            leaf.dtype,
                        )
                        for leaf, grow in zip(leaves, grows, strict=True)
                    ],
                )
            except Exception:
                wider = {}
            self._coupled_sites.difference_update(
                k
                for k in events
                if (site := wider.get(k))
                and getattr(site.get("fn"), "event_shape", ())[:1] == rows
                # The value too, since a distribution may take its event size from an
                # axis that the wider trace keeps, as `scale_tril` does
                and getattr(site["value"], "shape", ())[-len(site["fn"].event_shape) :]
                == site["fn"].event_shape
            )
        try:
            deps = get_dependencies(self.kernel, model_kwargs=dict(args_bound))
        except Exception:
            deps = {"prior_dependencies": {}}
        for k, v in cast("Mapping[str, Mapping]", deps["prior_dependencies"]).items():
            self._site_parents[k] = set(v) - {k}
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

    def _recomputed_sites(self) -> set[str]:
        """Return the posterior sites traced as deterministic with a value per row.

        A streamed call keeps off the devices those that its own trace also records as
        deterministic, since the kernel recomputes them there.
        """
        spec = cast("KernelSpec", self._kernel_spec)

        return {
            k
            for k in set(spec.return_sites) - set(spec.sample_sites)
            if k in (self._posterior or {}) and self._site_sizes.get(k, (0, 0))[0]
        }

    def _coerce_return_sites(
        self,
        return_sites: str | Iterable[str] | None,
        intervention: Mapping[str, object] | None = None,
    ) -> tuple[str, ...]:
        """Return the requested site names as a tuple, or the default return sites.

        Warns:
            OutputWarning: If a requested site was not seen in any trace so far; the
                name is passed through, since a kernel may define it only at sampling
                time. Also if a requested site is intervened on, since it comes back
                with its own draws, not the set value.
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
        if intervened := [site for site in requested if site in (intervention or {})]:
            msg = (
                f"Return site(s) {', '.join(map(repr, intervened))} are intervened on, "
                "so they come back with their own draws, not the set values. Read the "
                "effect of the intervention from the sites downstream of them."
            )
            warn(msg, category=OutputWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)

        return requested

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

    def _output_nbytes(self, return_sites: tuple[str, ...]) -> tuple[int, int]:
        """Return the output bytes per draw, per observation and for the other sites.

        Each value counts at the width of the default float type, since the trace holds
        the observed data, whose type the output need not share. A site the trace did
        not record counts as one value per observation.
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
        return_sites: tuple[str, ...] = (),
        kwargs: Mapping[str, object] | None = None,
        intervention: dict | None = None,
        condition: dict | None = None,
    ) -> tuple[Literal["obs", "draw"], int | None]:
        """Choose the sharding strategy and the batch size of a streamed call.

        A posterior site shaped ``(num_samples, n_obs, ...)`` cannot be split with the
        observation axis, nor can a kernel that computes across observations, nor a
        ``condition`` value with one entry per observation, which is passed whole: only
        a whole-input batch on a single device keeps them intact, so the call otherwise
        runs draw-parallel, with a warning. A call with such a ``condition`` value runs
        draw-parallel on a whole-input batch too, without a warning, since under
        ``"obs"`` the streamer can take a site set from a whole value for one without
        the observation axis. Deterministic posterior sites are exempt, since the kernel
        recomputes them, but observed ones are not, since a call without their data
        takes their draws, and neither is a site no trace records, which the kernel may
        sample under other arguments.

        Args:
            X: Input array, or a data loader, which batches itself.
            shard_axis: The requested sharding strategy.
            batch_size: The requested batch size, or ``None`` to choose it.
            num_samples: Number of draws the call produces.
            nbytes: Output bytes per draw, as :meth:`_output_nbytes` returns them.
            posterior: The posterior samples the call conditions on, if any.
            return_sites: The sites the call returns; one without the observation axis
                for the call's arguments makes it run draw-parallel.
            kwargs: The call's other kernel arguments.
            intervention: The call's intervention.
            condition: The call's condition.

        Returns:
            The sharding strategy and the batch size, ``None`` for a data loader.
        """
        if not isinstance(X, ArrayLike):
            # A data loader's batches split the rows, except one array loader batch on
            # one device, and no rerun can keep them whole
            if self._coupled_sites and not (
                self._num_devices == 1
                and isinstance(X, ArrayLoader)
                and X.batch_size >= len(X.dataset)
            ):
                msg = (
                    "The kernel computes across observations at site(s) "
                    f"{', '.join(map(repr, sorted(self._coupled_sites)))}, which the "
                    "batches of a data loader split. Pass the input as an array, which "
                    "then runs under `shard_axis='draw'`."
                )
                warn(
                    msg,
                    category=OutputWarning,
                    skip_file_prefixes=_SKIP_FILE_PREFIXES,
                )
            return shard_axis, None

        n_obs = len(cast("Sized", X))
        row_nbytes, rest_nbytes = nbytes
        min_aligned_ndim = 2
        spec = cast("KernelSpec", self._kernel_spec)
        deterministic = set(spec.return_sites) - set(spec.sample_sites)
        conditioned = sorted(
            site for site, v in (condition or {}).items() if np.shape(v)[:1] == (n_obs,)
        )
        local = sorted(
            site
            for site, v in (posterior or {}).items()
            if site not in deterministic
            and v.ndim >= min_aligned_ndim
            and v.shape[1] == n_obs
        )
        # Under `obs`, several devices or batches split the rows
        splits = self._num_devices > 1 or (
            batch_size < n_obs
            if batch_size is not None
            else not _fits_single_batch(n_obs, item_nbytes=num_samples * row_nbytes)
        )
        if shard_axis == "obs" and (self._coupled_sites or conditioned or local):
            if splits:
                reason = (
                    "The kernel computes across observations at site(s) "
                    f"{', '.join(map(repr, sorted(self._coupled_sites)))}, which "
                    "`shard_axis='obs'` would split"
                    if self._coupled_sites
                    else "The `condition` values of site(s) "
                    f"{', '.join(map(repr, conditioned))} have one entry per "
                    "observation and are passed whole, so `shard_axis='obs'` cannot "
                    "split them"
                    if conditioned
                    else "The posterior site(s) "
                    f"{', '.join(map(repr, local))} are as long as the input's rows in "
                    "each draw, so `shard_axis='obs'` cannot split them"
                )
                msg = (
                    f"{reason}; rerunning with `shard_axis='draw'`. Pass "
                    "`shard_axis='draw'` to silence this warning."
                )
                warn(
                    msg,
                    category=PerformanceWarning,
                    skip_file_prefixes=_SKIP_FILE_PREFIXES,
                )
                shard_axis, batch_size = "draw", None
            elif conditioned:
                # One batch keeps the rows whole, but under `obs` the streamer can
                # reject a site set from a whole `condition` value as having no
                # observation axis; `draw` keeps them whole and checks only the output
                shard_axis, batch_size = "draw", None
            elif batch_size is None:
                batch_size = n_obs
        # Posterior sites a call under `obs` replicates, deterministic and scalar ones
        # aside; their sizes matter only when the rows may be split or the batch is open
        latent = sorted(
            k
            for k, v in (posterior or {}).items()
            if k not in deterministic and v.ndim > 1 and (splits or batch_size is None)
        )
        if (return_sites or latent) and (
            shard_axis == "obs" or self.param_output in return_sites
        ):
            # A site without the observation axis, such as a global deterministic site,
            # has no rows to split, and a posterior site whose size follows the rows
            # without one value per row, such as a random walk's increments in a plate
            # of one row fewer, fits no device's rows. Abstract traces of this call, at
            # its rows and at one row fewer, as in a device batch, tell the sites that
            # follow the rows: an earlier trace may have had other arguments, and a
            # global site may match the row count. Under `draw`, which keeps the rows
            # whole and passes every argument whole, only the output is checked.
            n_split = n_obs if shard_axis == "obs" else None
            kwargs_rows, kwargs_const = _group_kwargs(kwargs or {}, n_obs=n_split)
            constants, fields = _split_intervention(intervention, n_split)
            constants = _to_jax(constants)
            leaves, static = _partition(kwargs_const)

            def values(rows: dict[str, object], leaves: list) -> dict[str, object]:
                model_kwargs, intervention_rows = _split_intervention_fields(rows)
                model = (
                    do(
                        self.kernel,
                        data=cast("dict", {**constants, **intervention_rows}),
                    )
                    if intervention
                    else self.kernel
                )
                model_trace = trace(
                    substitute(
                        seed(model, rng_seed=self.rng_key),
                        # A site that already holds a value, as under `substitute`,
                        # keeps it
                        substitute_fn=lambda msg: (
                            None if msg["value"] is not None else init_to_sample(msg)
                        ),
                    ),
                ).get_trace(**model_kwargs, **_combine(leaves, static))

                return {
                    k: v["value"]
                    for k, v in model_trace.items()
                    if k in return_sites or k in latent
                }

            rows = {
                k: ShapeDtypeStruct(
                    np.shape(v),
                    jnp.result_type(v if hasattr(v, "dtype") else np.asarray(v)),
                )
                for k, v in {self.param_input: X, **kwargs_rows, **fields}.items()
            }

            def shapes_at(size: int) -> dict[str, ShapeDtypeStruct]:
                return eval_shape(
                    values,
                    {
                        k: ShapeDtypeStruct((size, *v.shape[1:]), v.dtype)
                        for k, v in rows.items()
                    },
                    leaves,
                )

            # Whether each device evaluates one row fewer than the input, where draws
            # that fit those rows would pass
            fewer_rows = not return_sites and -(
                -min(
                    _resolve_batch_size(
                        batch_size,
                        axis_size=n_obs,
                        other_size=num_samples,
                        num_devices=self._num_devices,
                        item_nbytes=num_samples * row_nbytes,
                    ),
                    n_obs,
                )
                // self._num_devices
            ) == (n_obs - 1)
            # The traces depend only on the call's shapes and constants, and for a call
            # without return sites on the rows each device evaluates, so a repeated
            # call reuses their outcome
            key = (
                shard_axis,
                return_sites,
                tuple((k, v.shape, v.dtype) for k, v in rows.items()),
                tuple((np.shape(leaf), jnp.result_type(leaf)) for leaf in leaves),
                static,
                tuple(
                    (
                        k,
                        np.shape(v),
                        jnp.result_type(v if hasattr(v, "dtype") else np.asarray(v)),
                    )
                    for k, v in sorted(constants.items())
                ),
                tuple(sorted(intervention or {})),
                posterior is None,
                tuple((k, cast("dict", posterior)[k].shape[1:]) for k in latent),
                fewer_rows,
            )
            if key not in self._plan_cache:
                # A call without return sites, as `log_likelihood`, traces its rows
                # only for a latent site whose draws do not fit one row fewer, or when
                # each device evaluates one row fewer, which would take such draws
                shapes = shapes_at(n_obs) if return_sites else {}
                per_row = {
                    k
                    for k, v in shapes.items()
                    if k in return_sites and v.shape[:1] == (n_obs,)
                }
                follows, failed = {}, False
                try:
                    if (
                        (shapes or not return_sites)
                        and n_obs > 1
                        and shard_axis == "obs"
                    ):
                        fewer = shapes_at(n_obs - 1)
                        if not return_sites and (
                            fewer_rows
                            or any(
                                fewer[k].shape != cast("dict", posterior)[k].shape[1:]
                                for k in latent
                                if k in fewer
                            )
                        ):
                            shapes = shapes_at(n_obs)
                        per_row &= {
                            k for k, v in fewer.items() if v.shape[:1] == (n_obs - 1,)
                        }
                        follows = {
                            k: shapes[k].shape
                            for k in latent
                            if k in shapes
                            and k in fewer
                            and fewer[k].shape != shapes[k].shape
                        }
                        # The prior draws its global sites from one row
                        if posterior is None:
                            shapes_at(1)
                # A kernel that does not run on fewer rows, such as one whose plate size
                # is a fixed count, cannot be split either
                except Exception:
                    failed = True
                self._plan_cache[key] = (
                    frozenset(shapes.keys() & set(return_sites)) - per_row,
                    failed,
                    follows,
                )
            unsplit, failed, follows = self._plan_cache[key]
            # A posterior site whose draws do not fit the rows fails as on one batch,
            # though device batches with the fitted row count would take them
            for k, shape in sorted(follows.items()):
                draws = (posterior or {})[k].shape[1:]
                if (
                    k not in (intervention or {})
                    and k not in self._observed_sites
                    and len(draws) == len(shape)
                    and draws != shape
                ):
                    err_msg = (
                        f"Each posterior draw of site {k!r} has shape {draws}, which "
                        f"does not fit the shape {shape} the site has for the rows "
                        "being evaluated: a site with one value per observation fits "
                        "only the fitted observations, and a site sized by a count "
                        "read from the data fits only the fitted count."
                    )
                    raise ValueError(err_msg)
            # A posterior site follows the rows only if its draws fit them all
            rowed = sorted(
                k
                for k, shape in follows.items()
                if (posterior or {})[k].shape[1:] == shape
            )
            msg, category = None, PerformanceWarning
            if failed and return_sites:
                msg = (
                    "The kernel does not run on fewer rows than the input, so "
                    "`shard_axis='obs'` cannot split it; rerunning with "
                    "`shard_axis='draw'`. Pass `shard_axis='draw'` to silence this "
                    "warning."
                )
            whole = sorted(unsplit)
            if self.param_output in whole:
                msg = (
                    f"The output site {self.param_output!r} does not have one value "
                    "per row of the input"
                ) + (
                    ", so `shard_axis='obs'` cannot split it; rerunning with "
                    "`shard_axis='draw'`."
                    if shard_axis == "obs"
                    else "."
                )
                category = OutputWarning
            elif whole and msg is None and shard_axis == "obs":
                msg = (
                    f"Return site(s) {', '.join(map(repr, whole))} have no "
                    "observation axis, so `shard_axis='obs'` cannot split them; "
                    "rerunning with `shard_axis='draw'`. Pass `shard_axis='draw'`, "
                    "or leave the site(s) out of `return_sites`, to silence this "
                    "warning."
                )
            elif rowed and msg is None and splits:
                msg = (
                    f"The posterior site(s) {', '.join(map(repr, rowed))} have a size "
                    "that follows the input's rows in each draw, so `shard_axis='obs'` "
                    "cannot split them; rerunning with `shard_axis='draw'`. Pass "
                    "`shard_axis='draw'` to silence this warning."
                )
            elif rowed and msg is None and batch_size is None:
                # One batch on one device keeps the rows whole
                batch_size = n_obs
            if msg is not None:
                warn(msg, category=category, skip_file_prefixes=_SKIP_FILE_PREFIXES)
                if shard_axis == "obs":
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

    def _draw_attrs(
        self,
        rng_key: Array,
        *,
        X: object = None,
        shard_axis: str | None = None,
        batch_size: int | None = None,
    ) -> dict[str, object]:
        """Describe what the draws of a predictive call depend on.

        :meth:`~aimz.ImpactModel.estimate_effect` compares these attributes between two
        scenarios to tell whether their draws are paired. The values are strings,
        integers, or lists of integers, so a tree written to a file keeps them.
        """
        attrs: dict[str, object] = {"rng_key": random.key_data(rng_key).tolist()}
        if shard_axis is not None:
            attrs["shard_axis"] = shard_axis
        if shard_axis == "obs":
            attrs["num_devices"] = self._num_devices
            # Another loader than `ArrayLoader` has no fixed batch size to record
            if isinstance(X, ArrayLoader):
                batch_size = X.batch_size
            if batch_size is not None:
                attrs["batch_size"] = batch_size

        return attrs

    def _stream_to_datatree(
        self,
        write: Callable[[Path | None], dict[str, DaskArray] | None],
        *,
        store: str,
        output_dir: str | Path | None,
        group: str,
        attrs: Mapping[str, object] | None = None,
    ) -> xr.DataTree:
        """Run one streamed write and assemble its output tree.

        ``write`` streams into the artifact path, or into host memory for ``None``, and
        returns the retained site arrays, if any. A failed persistent write removes its
        artifact path.
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
            OutputWarning: If the output site does not have one value per row of the
                input, or a requested return site is intervened on, since it comes back
                with its own draws.

        See Also:
            :meth:`~aimz.ImpactModel.sample_prior_predictive`
        """
        X = cast("Array", _validate_X_y_to_jax(X))

        args_bound = _to_jax(self._bind_kernel_args(X, kwargs=kwargs))
        self._build_kernel_spec(args_bound, with_output=False)
        _validate_intervention(intervention, kernel_spec=self._kernel_spec)

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        prior_predictive_samples = device_get(
            _sample_forward(
                self.kernel,
                rng_keys=random.split(rng_key, num=num_samples),
                return_sites=self._coerce_return_sites(
                    return_sites,
                    intervention=intervention,
                ),
                samples=None,
                params=None,
                intervention=_to_jax(intervention or {}),
                model_kwargs=args_bound,
            ),
        )
        out = prior_predictive_samples.get(self.param_output)
        if out is not None and np.shape(out)[1:2] != (len(X),):
            msg = (
                f"The output site {self.param_output!r} does not have one value per "
                "row of the input."
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
            attrs=self._draw_attrs(rng_key),
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
                intervened on or the kernel does not run on one row fewer than a device
                evaluates (two rows when it evaluates one), or if its axis-1 size does
                not match the rows a device evaluates from a batch.

        Warns:
            OutputWarning: If a requested return site is intervened on, since it comes
                back with its own draws, or if the output site does not have one value
                per row of an array input; under ``shard_axis="obs"`` the call then
                reruns under ``shard_axis="draw"``. Also if the batches of a data
                loader split a kernel that computes across observations.
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
        if not (self._kernel_spec and self._kernel_spec.traced):
            probe = {
                k: (jnp.asarray(cast("Any", v)[:1]) if k in obs_names else v)
                for k, v in args_bound.items()
            }
            try:
                self._build_kernel_spec(_to_jax(probe), with_output=False)
            except Exception:
                self._build_kernel_spec(_to_jax(args_bound), with_output=False)
        _validate_intervention(intervention, kernel_spec=self._kernel_spec)

        return_sites = self._coerce_return_sites(
            return_sites,
            intervention=intervention,
        )
        shard_axis, batch_size = self._plan_execution(
            X,
            shard_axis=shard_axis,
            batch_size=batch_size,
            num_samples=num_samples,
            nbytes=self._output_nbytes(return_sites),
            posterior=None,
            return_sites=return_sites,
            kwargs=kwargs,
            intervention=intervention,
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
            attrs=self._draw_attrs(
                rng_key,
                X=X,
                shard_axis=shard_axis,
                batch_size=batch_size,
            ),
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
            if return_sites is None and spec is not None and spec.output_observed:
                return_sites = spec.sample_sites
            posterior_samples = device_get(
                _sample_forward(
                    self.inference.guide,
                    rng_keys=random.split(rng_key, num=num_samples),
                    return_sites=self._coerce_return_sites(return_sites)
                    if return_sites is not None
                    else None,
                    samples=None,
                    params={
                        **self.vi_result.params,
                        **(getattr(self.vi_result.state, "mutable_state", None) or {}),
                    },
                    intervention=None,
                    model_kwargs=kwargs,
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
        # sees them, since `fit` keeps its data in NumPy. A result set through
        # `vi_result` may come with an SVI object that needs no initialization, so the
        # trace does not depend on it
        if not (self._kernel_spec and self._kernel_spec.output_observed):
            self._build_kernel_spec(
                signature(self.kernel).bind(**_to_jax(batch)).arguments,
                with_output=True,
            )
        if self._vi_state is None or svi.constrain_fn is None:
            if rng_key is None:
                self._rng_key, rng_key = random.split(self._rng_key)
            state = svi.init(rng_key, **_to_jax(batch))
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
        """Fit the model to one batch of data.

        With SVI, runs the optimization on the batch and draws posterior samples from
        the guide; with MCMC, runs the sampler. Training continues from an existing SVI
        state; create a new model to start over, or to fit with a keyword argument that
        adds a latent site: that state has no guide parameters for it, so the posterior
        lacks it and the predictive methods draw it from its prior. For such a site,
        give the new model a new guide too, since an autoguide keeps the sites of its
        first call.

        Args:
            X: Input array with observations on the leading axis.
            y: Output array with observations on the leading axis.
            num_steps: Number of optimization steps. Ignored for MCMC.
            num_samples: The number of posterior samples to draw. Ignored for MCMC.
            rng_key: A pseudo-random number generator key, split from an internal key by
                default.
            progress: Whether to display a progress bar. Ignored for MCMC.
            **kwargs: Additional arguments passed to the model.

        Returns:
            The fitted model instance.

        Raises:
            TypeError: If :attr:`~aimz.ImpactModel.param_input` or
                :attr:`~aimz.ImpactModel.param_output` is passed as an argument.
            ValueError: If ``y`` does not share ``X``'s leading-axis size.
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
        if isinstance(self.inference, SVI):
            if self._vi_state is not None and self.inference.constrain_fn is None:
                rng_key, rng_init = random.split(rng_key)
                self.inference.init(rng_init, **args_bound)
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
            FitWarning: If the batches are smaller than the data and the kernel scales
                the output site for a batch by other than the number of observations
                over the batch size, as when it gives no scale, while a latent site
                outside the batch's plate, such as a global one, weighs against it, or
                has a site that computes across observations, or a
                latent site per observation that the guide does not compute from the
                batch, such as one kept per position or plate index, which the batches
                split. The checks run on the first ``batch_size`` observations of an
                array input or of an :class:`~aimz.utils.data.ArrayLoader`; another
                data loader has no size to compare with. Of the sites drawn as one
                event over the rows, only an event as long as the batch is checked, so
                one padded beyond the rows, or offset from them, trains without the
                warning.
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
        # Commit model state only once the inputs are accepted
        self._rng_key = rng_key_model

        rng_key, rng_subkey = random.split(rng_key)
        # The scale a batch gives the output site tells whether the kernel scales it to
        # the whole data, as it must when a latent site outside the batch's plate, such
        # as a global one, weighs against it; otherwise every term follows the batch
        if isinstance(dataloader, ArrayLoader) and dataloader.batch_size < len(
            dataloader.dataset,
        ):
            n_obs = len(dataloader.dataset)
            batch = {
                k: v[: dataloader.batch_size]
                for k, v in dataloader.dataset.arrays.items()
            }
            # The first batch already rejected an output site that is not observed
            with validation_enabled(is_validate=False):
                model_trace = trace(
                    substitute(
                        seed(self.kernel, rng_seed=self.rng_key),
                        substitute_fn=init_to_sample,
                    ),
                ).get_trace(**_to_jax({**batch, **kwargs_extra}))
            site = model_trace.get(self.param_output, {})
            # The output's outermost plate is the one over the batch's rows
            rows = {
                f.name
                for f in sorted(
                    site.get("cond_indep_stack") or (), key=lambda f: f.dim
                )[:1]
            }
            scale = float(np.mean(1.0 if site.get("scale") is None else site["scale"]))
            expected = n_obs / dataloader.batch_size
            if (
                site.get("is_observed")
                and not np.isclose(scale, expected)
                and any(
                    v["type"] == "sample"
                    and not v.get("is_observed")
                    and not rows & {f.name for f in v.get("cond_indep_stack") or ()}
                    for v in model_trace.values()
                )
            ):
                msg = (
                    f"`fit` trains on batches of {dataloader.batch_size} of {n_obs} "
                    "observations, but the kernel scales the output site "
                    f"{self.param_output!r} by {scale:g} for a batch rather than by "
                    f"{n_obs}/{dataloader.batch_size} = {expected:g}, so a batch does "
                    "not weigh as the whole data against the sites outside its plate. "
                    "Scale it in the kernel, as a plate whose `size` is the number of "
                    "observations and whose `subsample_size` is a batch's rows does, "
                    "or fit with `fit_on_batch`; a scale chosen on purpose, as for "
                    "tempering, can ignore this warning."
                )
                warn(msg, category=FitWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
            # Batches split a site that counts as computing across observations: one
            # recorded by control flow, or one drawn as an event whose size follows the
            # rows, which the build tells at twice the batch's rows, or all of them,
            # growing the batch's fields alone
            self._build_kernel_spec(
                signature(self.kernel)
                .bind(**_to_jax({**batch, **kwargs_extra}))
                .arguments,
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
                    init = svi.init(rng_subkey, **_to_jax({**batch, **kwargs_extra}))
                    state = init if state is None else state
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
                        .get_trace(**rows, **_to_jax(kwargs_extra))
                        .items()
                        if k in latent and v["type"] == "sample"
                    },
                    return_shape=True,
                )(_to_jax(batch), jnp.arange(dataloader.batch_size))
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
            model_kwargs=(
                {self.param_input: X, self.param_output: y, **kwargs}
                if isinstance(X, ArrayLike)
                else _to_jax({**batch, **kwargs_extra})
            ),
        )
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
            ``"deterministic"``), its ``dims`` after ``chain`` and ``draw`` where the
            trace named them (``None`` for the default ``<site>_dim_<i>`` names), and,
            for a site in the posterior, the ``draw_shape`` of one draw. A ``factor``
            term counts as observed, and ``param`` sites are not listed; their values
            after an SVI fit are in :attr:`~aimz.ImpactModel.vi_result`.
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
            sites[name] = {"kind": kind, "dims": None if dims is None else list(dims)}
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
                leave out any the kernel never samples, since one as long as the
                input's rows can still make a call rerun under ``shard_axis="draw"``.
            OutputWarning: If the kernel has been traced and ``posterior_sample`` has
                no draws for one of its latent sample sites, which the predictive
                methods then draw from the prior.

        Note:
            The kernel is not traced here. Prior predictive sampling or the first call
            with an array input traces it, and calls then return its deterministic
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
        (self._num_samples,) = batch_shapes.pop()
        spec = self._kernel_spec
        if spec is not None and spec.traced:
            latent = set(spec.sample_sites) - self._observed_sites - {self.param_output}
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
                    "never samples, since one as long as the input's rows can still "
                    "make a call rerun under `shard_axis='draw'`."
                )
                warn(msg, category=AimzWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
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
                on does not fit the rows being evaluated, as a site with one value per
                observation does not fit other observations, and a site sized by a
                count, such as the number of groups, does not fit another count.

        Warns:
            OutputWarning: If an intervened site reaches the output only through sites
                whose values are taken from the posterior or intervened on, so that
                the draws do not respond to it, if a requested return site is
                intervened on, since it comes back with its own draws, if a conditioned
                site is one the posterior holds, since it keeps its posterior draws, or
                if the output site does not have one value per row of the input.
        """
        _check_is_fitted(self)
        X = cast("Array", _validate_X_y_to_jax(X))

        args_bound = _to_jax(self._bind_kernel_args(X, kwargs=kwargs))
        # A posterior set on a new model leaves the kernel untraced; the checks below
        # and the output's sites and names need its trace
        if not (self._kernel_spec and self._kernel_spec.traced):
            self._build_kernel_spec(args_bound, with_output=False)
        _validate_intervention(intervention, kernel_spec=self._kernel_spec)
        _validate_condition(
            condition,
            kernel_spec=self._kernel_spec,
            fixed=(self._posterior or {}).keys(),
        )
        _warn_unreachable_intervention(
            intervention,
            output=self.param_output,
            parents=self._site_parents,
            fixed=(self._posterior or {}).keys(),
        )

        if rng_key is None:
            self._rng_key, rng_key = random.split(self._rng_key)

        samples = device_get(
            _sample_forward(
                self.kernel,
                rng_keys=random.split(rng_key, num=self._num_samples),
                return_sites=self._coerce_return_sites(
                    return_sites,
                    intervention=intervention,
                ),
                samples=self._streamer.place_posterior(self.posterior, sharding=None),
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
        out = samples.get(self.param_output)
        if out is not None and np.shape(out)[1:2] != (len(X),):
            msg = (
                f"The output site {self.param_output!r} does not have one value per "
                "row of the input."
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
            attrs=self._draw_attrs(rng_key),
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
                a posterior site that is not intervened on does not fit the rows being
                evaluated, as a site with one value per observation does not fit other
                observations, and a site sized by a count, such as the number of
                groups, does not fit another count.
            NotImplementedError: With a data loader, if a return site has no
                observation axis, unless it is intervened on or the kernel does not
                run on one row fewer than a device evaluates (two rows when it
                evaluates one), or if its axis-1 size does not match the rows a device
                evaluates from a batch.

        Warns:
            OutputWarning: If an intervened site reaches the output only through sites
                whose values are taken from the posterior or intervened on, so that
                the draws do not respond to it, if a requested return site is
                intervened on, since it comes back with its own draws, if a conditioned
                site is one the posterior holds, since it keeps its posterior draws, or
                if the output site does not have one value per row of an array input;
                under ``shard_axis="obs"`` the call then reruns under
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

        if isinstance(X, ArrayLike):
            args_bound = self._bind_kernel_args(X, kwargs=kwargs)
            # A posterior set on a new model leaves the kernel untraced; the checks
            # below and planning need its sites
            if not (self._kernel_spec and self._kernel_spec.traced):
                self._build_kernel_spec(_to_jax(args_bound), with_output=False)
        _validate_intervention(intervention, kernel_spec=self._kernel_spec)
        _validate_condition(
            condition,
            kernel_spec=self._kernel_spec,
            fixed=(self._posterior or {}).keys(),
        )
        _warn_unreachable_intervention(
            intervention,
            output=self.param_output,
            parents=self._site_parents,
            fixed=(self._posterior or {}).keys(),
        )

        return_sites = self._coerce_return_sites(
            return_sites,
            intervention=intervention,
        )
        shard_axis, batch_size = self._plan_execution(
            X,
            shard_axis=shard_axis,
            batch_size=batch_size,
            num_samples=self._num_samples,
            nbytes=self._output_nbytes(return_sites),
            posterior=self.posterior,
            return_sites=return_sites,
            kwargs=kwargs,
            intervention=intervention,
            condition=condition,
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
            attrs=self._draw_attrs(
                rng_key,
                X=X,
                shard_axis=shard_axis,
                batch_size=batch_size,
            ),
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
                rows a scenario evaluates, or if the scenarios do not share a
                predictive group.

        Warns:
            OutputWarning: If the scenarios differ in dimension sizes or coordinate
                labels, if they hold different posterior samples, or if the output,
                or any site under the prior, was not drawn in both with the same key,
                sharding strategy, and batching.
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
                # The scenario's own call warns for sites the posterior holds
                _validate_condition(
                    args.get("condition"),
                    kernel_spec=self._kernel_spec,
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

        _predict = cast(
            "Callable[..., xr.DataTree]",
            self.predict_on_batch if on_batch else self.predict,
        )
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
                fit the rows being evaluated, as a site with one value per observation
                does not fit other observations, and a site sized by a count, such as
                the number of groups, does not fit another count.
            NotImplementedError: If a return site's axis-1 size does not match the
                input batch size (``shard_axis="obs"`` only).

        Warns:
            OutputWarning: If the batches of a data loader split a kernel that
                computes across observations, once a call has traced the kernel.
            PerformanceWarning: If ``shard_axis="obs"`` cannot split an array input,
                as for a kernel that computes across observations or a posterior site
                as long as the input's rows or with another size that follows them, so
                the call reruns under ``shard_axis="draw"``.
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
        shard_axis, batch_size = self._plan_execution(
            X,
            shard_axis=shard_axis,
            batch_size=batch_size,
            num_samples=self._num_samples,
            nbytes=(jnp.result_type(float).itemsize, 0),
            posterior=self.posterior,
            kwargs={**kwargs, self.param_output: y},
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
            shard_axis, batch_size = self._plan_execution(
                X,
                shard_axis="draw",
                batch_size=None,
                num_samples=self._num_samples,
                nbytes=(jnp.result_type(float).itemsize, 0),
                posterior=self.posterior,
            )
        # A keyword array as long as the rows each device evaluates, but not as the
        # input, is passed whole, so every device would score its rows with the same
        # values; as in `predict`, the call fails when the kernel does not run on all
        # the rows with it but does once the array has as many
        if shard_axis == "obs" and isinstance(X, ArrayLike):
            n_obs = len(cast("Sized", X))
            rows = -(-min(cast("int", batch_size), n_obs) // self._num_devices)
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

        # Without a posterior every latent site is drawn from the prior, which needs a
        # seeded kernel; with one, the bare kernel keeps its identity for the cache
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
