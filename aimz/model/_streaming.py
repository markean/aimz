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

"""Streaming engine that runs the kernel across shards and writes to Zarr.

:class:`_OutputStreamer` owns the streaming subsystem extracted from
:class:`~aimz.ImpactModel`: it builds (and caches) the sharded callables, places the
posterior on devices, and drives the data- and draw-parallel write paths. The model
passes a stable :class:`_RuntimeContext` once and a per-call :class:`_WriteRequest` each
time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

from jax import Array, device_get, device_put, random
from tqdm.auto import tqdm
from zarr import open_group

from aimz.sampling._forward import _sample_forward
from aimz.utils._kwargs import _group_kwargs
from aimz.utils._output import (
    _select_write_strategy,
    _SliceWriteStrategy,
    _write_loop,
)
from aimz.utils.data import ArrayLoader
from aimz.utils.data._input_setup import _resolve_batch_size, _setup_inputs
from aimz.utils.data._sharding import (
    _create_sharded_log_likelihood,
    _create_sharded_sampler,
    _prepare_draw_chunk,
    _replicate,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sized
    from pathlib import Path

    import numpy as np
    from jax.sharding import Mesh, Sharding
    from jax.typing import ArrayLike

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RuntimeContext:
    """Stable per-model sharding configuration shared across write calls."""

    param_input: str
    param_output: str
    mesh: Mesh | None
    num_devices: int
    replicated_sharding: Sharding | None
    partitioned_sharding: Sharding | None


@dataclass(frozen=True)
class _WriteRequest:
    """The invariants of one streamed write job (shared by both write strategies)."""

    shard_axis: Literal["obs", "draw"]
    X: ArrayLike | ArrayLoader
    return_sites: tuple[str, ...]
    num_samples: int
    batch_size: int | None
    output_dir: Path
    progress: bool
    loader_rng_key: Array
    kwargs: dict[str, object]


class _Step(NamedTuple):
    """Per-item inputs for one sharded forward call, shared by both write strategies.

    A write strategy assembles a ``_Step`` for each item it streams (an observation
    batch or a draw chunk) and hands it to a ``compute`` closure that forwards it to the
    sharded sampler / log-likelihood function. The strategy fills the fields differently
    — whole vs. sliced — but the ``compute`` signature is identical, which keeps the
    strategy streamers kind-agnostic.
    """

    num: int
    """Draw count for the call: global count (data) or per-device count (draw)."""
    keys: Array | None
    """Per-draw keys for predictive sampling; ``None`` for log-likelihood."""
    samples: dict[str, Array]
    """Conditioning samples: the whole posterior (data) or one draw chunk (draw)."""
    x: Array
    """Input data: one observation batch (data) or the whole replicated input (draw)."""
    y: Array | None
    """Output data (log-likelihood only)."""
    tail: tuple
    """Array-kwargs and extra-kwargs values forwarded after ``x``."""


class _OutputStreamer:
    """Run the kernel across shards and stream the per-item outputs to Zarr.

    Constructed once per model from a :class:`_RuntimeContext`; owns the jit/shard_map
    callable cache and the posterior device-placement cache. Exposes
    :meth:`write_predictive` (predict / prior predictive) and
    :meth:`write_log_likelihood`, each dispatching to the data- or draw-parallel
    streamer by the request's ``shard_axis`` strategy.
    """

    def __init__(self, ctx: _RuntimeContext) -> None:
        """Initialize the streamer with its runtime context and empty caches.

        Args:
            ctx: Stable sharding configuration for the owning model.
        """
        self._ctx = ctx
        self._fn_cache: dict[tuple[str, str, int, int], Callable] = {}
        self._posterior_device_cache: dict[Sharding | None, dict[str, Array]] = {}
        self._posterior_device_src: dict[str, Array] | None = None

    def _cached_fn(
        self,
        kind: str,
        shard_axis: Literal["obs", "draw"],
        factory: Callable,
        n_kwargs_array: int,
        n_kwargs_extra: int,
    ) -> Callable:
        """Build a sharded callable once and cache it.

        Keyed by ``(kind, shard_axis, n_kwargs_array, n_kwargs_extra)``: the kwarg
        counts are baked into the ``shard_map`` ``in_specs``, so a call with a different
        arity needs its own callable rather than reusing a stale one. The callable is
        built with ``factory`` (which selects the partition specs) on first use.

        Args:
            kind: Cache namespace for the callable (e.g. ``"predict"``,
                ``"prior_predictive"``, ``"log_likelihood"``).
            shard_axis: Multi-device sharding strategy the callable is built for.
            factory: Builder that creates the sharded callable from the mesh and arity.
            n_kwargs_array: Number of array-like keyword arguments.
            n_kwargs_extra: Number of non-array (replicated) keyword arguments.

        Returns:
            The sharded callable for the given key, built on first use and cached.
        """
        key = (kind, shard_axis, n_kwargs_array, n_kwargs_extra)
        fn = self._fn_cache.get(key)
        if fn is None:
            fn = factory(
                self._ctx.mesh,
                n_kwargs_array=n_kwargs_array,
                n_kwargs_extra=n_kwargs_extra,
                shard_axis=shard_axis,
            )
            self._fn_cache[key] = fn

        return fn

    def _resolve_kwarg_key(
        self,
        req: _WriteRequest,
    ) -> tuple[tuple[str, ...], int, dict]:
        """Resolve the ordered kwarg names, array count, and extras for a stream.

        The names drive both the ``shard_map`` in-spec arity and the by-name binding
        in ``produce``, so they come from a single source and cannot drift. Array
        arguments come from the data loader's dataset fields when ``req.X`` is an
        :class:`~aimz.utils.data.ArrayLoader`; otherwise from the call kwargs. Non-array
        extras always come from the call kwargs.

        Args:
            req: The streamed write job.

        Returns:
            - The ordered ``kwargs_key`` (array names then extra names).
            - The number of array arguments.
            - The extras dict (the non-array call kwargs).

        Raises:
            ValueError: If array keyword arguments are passed alongside a data loader,
                or the loader has no field matching
                :attr:`~aimz.ImpactModel.param_input`.
        """
        kwargs_array, kwargs_extra = _group_kwargs(req.kwargs)
        if isinstance(req.X, ArrayLoader):
            if kwargs_array:
                msg = (
                    "Array keyword arguments are not supported alongside a data "
                    "loader; include them as fields of the loader's dataset instead: "
                    f"{sorted(kwargs_array)}."
                )
                raise ValueError(msg)
            fields = req.X.dataset.arrays
            if self._ctx.param_input not in fields:
                msg = (
                    f"The data loader has no field named {self._ctx.param_input!r} "
                    "for the model input; name the input array to match `param_input`."
                )
                raise ValueError(msg)
            array_names = tuple(
                name
                for name in fields
                if name not in (self._ctx.param_input, self._ctx.param_output)
            )
        else:
            array_names = tuple(kwargs_array)

        return (*array_names, *kwargs_extra), len(array_names), kwargs_extra

    def place_posterior(
        self,
        posterior: dict[str, Array] | None,
        sharding: Sharding | None,
    ) -> dict[str, Array]:
        """Return the posterior placed on devices, cached by ``sharding``.

        The cache is rebuilt whenever ``posterior`` is replaced (by identity); each
        distinct placement coexists in the same cache. Used by the data-parallel path
        to replicate the posterior once across calls and to keep a host-backed posterior
        device-resident; the draw-parallel path places its per-chunk slices itself.

        Args:
            posterior: The posterior samples to place, or empty/``None`` when unset.
            sharding: Placement to commit the posterior to, or ``None`` to place it on
                the default device (e.g. single-device).

        Returns:
            The posterior samples by variable name, placed on devices, or an empty
            dict when no posterior samples are set. The placement stays cached (and
            device-resident) until ``posterior`` is replaced.
        """
        if not posterior:
            return {}
        if self._posterior_device_src is not posterior:
            self._posterior_device_cache = {}
            # Keep the source posterior alive while its placements are cached, so the
            # identity check above stays meaningful until it is next replaced.
            self._posterior_device_src = posterior
        cache = self._posterior_device_cache
        if sharding not in cache:
            # sharding=None (no mesh): device_put converts a host-backed (NumPy)
            # posterior to the default device once, instead of it being re-transferred
            # on every downstream jit call; device-backed arrays pass through without a
            # copy.
            cache[sharding] = device_put(posterior, device=sharding)

        return cache[sharding]

    def write_predictive(
        self,
        req: _WriteRequest,
        *,
        kernel: Callable,
        rng_key: Array,
        group: str,
        posterior: dict[str, Array] | None,
    ) -> None:
        """Write predictive samples (predict / prior predictive) to ``req.output_dir``.

        Builds the predictive ``compute`` — one sharded-sampler call per item — and
        dispatches to the strategy streamer. ``shard_axis="draw"`` chunks the draw axis;
        ``shard_axis="obs"`` shards the observation axis and conditions every batch on
        the replicated posterior, or — for prior predictive — on the global prior
        samples drawn once from a single-element probe (a sharded probe would propagate
        the mesh axis onto global sites via JAX's sharding-in-types).

        Args:
            req: The streamed write job.
            kernel: Probabilistic model with `NumPyro`_ primitives.
            rng_key: Pseudo-random number generator key for sampling.
            group: Output group (``"posterior_predictive"``, ``"predictions"``, or
                ``"prior_predictive"``).
            posterior: The posterior to condition on (ignored for prior predictive).

        .. _NumPyro: https://num.pyro.ai/
        """
        kwargs_key, n_kwargs_array, kwargs_extra = self._resolve_kwarg_key(req)
        kind = "prior_predictive" if group == "prior_predictive" else "predict"
        fn = self._cached_fn(
            kind,
            shard_axis=req.shard_axis,
            factory=_create_sharded_sampler,
            n_kwargs_array=n_kwargs_array,
            n_kwargs_extra=len(kwargs_extra),
        )

        def compute(step: _Step) -> dict[str, Array]:
            return fn(
                kernel,
                step.num,
                step.keys,
                req.return_sites,
                step.samples,
                self._ctx.param_input,
                kwargs_key,
                step.x,
                *step.tail,
            )

        phase = "Prior" if group == "prior_predictive" else "Posterior"
        pbar = tqdm(
            desc=f"{phase} predictive sampling [{', '.join(req.return_sites)}]",
            disable=not req.progress,
            dynamic_ncols=True,
        )
        if req.shard_axis == "draw":
            chunk_posterior = (
                {}
                if group == "prior_predictive"
                else cast("dict[str, Array]", posterior)
            )
            self._write_draws(
                req,
                compute=compute,
                y=None,
                posterior=chunk_posterior,
                rng_key=rng_key,
                pbar=pbar,
            )
            return

        if group == "prior_predictive":
            # Single-element, unsharded probe draws the global prior samples once; the
            # return sites are dropped so they are redrawn per batch. Prior samples are
            # redrawn every call, so this is a one-shot (not cached) placement.
            probe, _ = _setup_inputs(
                X=req.X,
                y=None,
                param_input=self._ctx.param_input,
                param_output=self._ctx.param_output,
                rng_key=req.loader_rng_key,
                batch_size=1,
                num_samples=req.num_samples,
                shuffle=False,
                device=None,
                **req.kwargs,
            )
            batch, _ = next(iter(probe))
            rng_key, rng_subkey = random.split(rng_key)
            samples = _sample_forward(
                kernel,
                rng_keys=random.split(rng_subkey, num=req.num_samples),
                return_sites=None,
                samples=None,
                model_kwargs={**batch, **kwargs_extra},
            )
            samples = {k: v for k, v in samples.items() if k not in req.return_sites}
            if self._ctx.replicated_sharding is not None:
                samples = device_put(samples, device=self._ctx.replicated_sharding)
        else:
            samples = self.place_posterior(posterior, self._ctx.replicated_sharding)
        self._write_data(
            req,
            compute=compute,
            y=None,
            samples=samples,
            kwargs_key=kwargs_key,
            rng_key=rng_key,
            pbar=pbar,
        )

    def write_log_likelihood(
        self,
        req: _WriteRequest,
        *,
        kernel: Callable,
        posterior: dict[str, Array] | None,
        y: ArrayLike | None,
    ) -> None:
        """Write the log-likelihood of ``y`` under ``kernel`` to ``req.output_dir``.

        Builds the log-likelihood ``compute`` — one sharded call per item, keyed by the
        output site — and dispatches to the strategy streamer.

        Args:
            req: The streamed write job (its single return site is the output site).
            kernel: Probabilistic model with `NumPyro`_ primitives, already seeded.
            posterior: The posterior to condition on.
            y: Output data.

        .. _NumPyro: https://num.pyro.ai/
        """
        site = self._ctx.param_output
        kwargs_key, n_kwargs_array, kwargs_extra = self._resolve_kwarg_key(req)
        fn = self._cached_fn(
            "log_likelihood",
            shard_axis=req.shard_axis,
            factory=_create_sharded_log_likelihood,
            n_kwargs_array=n_kwargs_array,
            n_kwargs_extra=len(kwargs_extra),
        )

        def compute(step: _Step) -> dict[str, Array]:
            return {
                site: fn(
                    kernel,
                    step.samples,
                    self._ctx.param_input,
                    site,
                    kwargs_key,
                    step.x,
                    step.y,
                    *step.tail,
                ),
            }

        pbar = tqdm(
            desc=f"Computing log-likelihood [{site}]",
            disable=not req.progress,
            dynamic_ncols=True,
        )
        if req.shard_axis == "draw":
            self._write_draws(
                req,
                compute=compute,
                y=cast("ArrayLike", y),
                posterior=cast("dict[str, Array]", posterior),
                rng_key=None,
                pbar=pbar,
            )
        else:
            self._write_data(
                req,
                compute=compute,
                y=y,
                samples=self.place_posterior(
                    posterior,
                    self._ctx.replicated_sharding,
                ),
                kwargs_key=kwargs_key,
                rng_key=None,
                pbar=pbar,
            )

    def _write_data(
        self,
        req: _WriteRequest,
        compute: Callable[[_Step], dict[str, Array]],
        y: ArrayLike | None,
        samples: dict[str, Array],
        kwargs_key: tuple[str, ...],
        rng_key: Array | None,
        pbar: tqdm,
    ) -> None:
        """Stream over data-parallel batches and write to ``req.output_dir``.

        Shards the observation axis across devices and conditions every batch on the
        whole (replicated) ``samples``. For each batch it assembles a :class:`_Step`,
        runs ``compute``, trims the observation padding (axis 1), and drives
        ``_write_loop``. Kind-agnostic: ``compute`` carries the sampler / log-likelihood
        call. ``rng_key`` is ``None`` for log-likelihood (no per-draw keys).

        Args:
            req: The streamed write job.
            compute: Closure that runs the sharded sampler / log-likelihood on a
                :class:`_Step`.
            y: Output data, for log-likelihood; ``None`` otherwise.
            samples: The whole (replicated) posterior to condition every batch on.
            kwargs_key: Ordered kwarg names (array names then extra names) used to bind
                each batch's ``tail`` by name.
            rng_key: Per-batch key source, or ``None`` for log-likelihood.
            pbar: Progress bar to drive over the batches.
        """
        _, kwargs_extra = _group_kwargs(req.kwargs)
        dataloader, _ = _setup_inputs(
            X=req.X,
            y=y,
            param_input=self._ctx.param_input,
            param_output=self._ctx.param_output,
            rng_key=req.loader_rng_key,
            batch_size=req.batch_size,
            num_samples=req.num_samples,
            shuffle=False,
            device=self._ctx.partitioned_sharding,
            **req.kwargs,
        )
        n_batches = len(dataloader)
        pbar.reset(total=n_batches)
        if rng_key is None:
            subkeys = [None] * n_batches
        else:
            # One fresh key per batch; the sampler splits it into per-draw keys.
            rng_key, *subkeys = random.split(rng_key, num=n_batches + 1)
            if self._ctx.replicated_sharding is not None:
                subkeys = device_put(subkeys, device=self._ctx.replicated_sharding)

        def produce(item: object) -> dict[str, np.ndarray]:
            (batch, n_pad), subkey = cast("tuple", item)
            # Bind by name (array kwargs from the batch, extras from the call) in
            # `kwargs_key` order, so positions match the sharded callable's in-specs.
            tail = tuple(
                batch[name] if name in batch else kwargs_extra[name]
                for name in kwargs_key
            )
            step = _Step(
                num=req.num_samples,
                keys=subkey,
                samples=samples,
                x=batch[self._ctx.param_input],
                y=batch.get(self._ctx.param_output),
                tail=tail,
            )
            dict_arr = device_get(compute(step))

            return {
                site: arr[:, None] if arr.ndim == 1 else arr[:, : -n_pad or None]
                for site, arr in dict_arr.items()
            }

        _write_loop(
            items=zip(dataloader, subkeys, strict=True),
            n_items=n_batches,
            return_sites=req.return_sites,
            output_dir=req.output_dir,
            strategy=_select_write_strategy(
                open_group(req.output_dir, mode="w"),
                dataloader=dataloader,
            ),
            produce=produce,
            pbar=pbar,
        )

    def _write_draws(
        self,
        req: _WriteRequest,
        compute: Callable[[_Step], dict[str, Array]],
        y: ArrayLike | None,
        posterior: dict[str, Array],
        rng_key: Array | None,
        pbar: tqdm,
    ) -> None:
        """Stream over draw chunks and write to ``req.output_dir``.

        Shards the draw axis across devices and holds the whole input resident:
        replicates the input/output/array-kwargs once, splits the per-draw keys once,
        and for each chunk slices a posterior chunk (``_prepare_draw_chunk``), assembles
        a :class:`_Step`, runs ``compute``, trims to the chunk's true draw count
        (axis 0), and drives ``_write_loop``. Kind-agnostic: ``compute`` carries the
        sampler / log-likelihood call. ``rng_key`` is ``None`` for log-likelihood;
        ``posterior`` is empty for prior predictive (each chunk draws fresh).

        Args:
            req: The streamed write job.
            compute: Closure that runs the sharded sampler / log-likelihood on a
                :class:`_Step`.
            y: Output data, for log-likelihood; ``None`` otherwise.
            posterior: The posterior to slice per draw chunk; empty for prior predictive
                (each chunk draws fresh).
            rng_key: Per-draw key source, or ``None`` for log-likelihood.
            pbar: Progress bar to drive over the draw chunks.
        """
        batch_size = _resolve_batch_size(
            req.batch_size,
            axis_size=req.num_samples,
            other_size=len(cast("Sized", req.X)),
            num_devices=self._ctx.num_devices,
        )
        if req.batch_size is None:
            logger.info(
                "Using batch_size=%d (draws per chunk). Specify explicitly to better "
                "control memory usage.",
                batch_size,
            )
        pbar.reset(total=-(-req.num_samples // batch_size))
        kwargs_array, kwargs_extra = _group_kwargs(req.kwargs)
        # The public draw-parallel entry points reject data loaders, so `req.X` is
        # always an array here.
        x_dev = _replicate(
            cast("ArrayLike", req.X),
            sharding=self._ctx.replicated_sharding,
        )
        y_dev = (
            _replicate(y, sharding=self._ctx.replicated_sharding)
            if y is not None
            else None
        )
        tail = (
            *(
                _replicate(v, sharding=self._ctx.replicated_sharding)
                for v in kwargs_array.values()
            ),
            *kwargs_extra.values(),
        )
        draw_keys = (
            random.split(rng_key, num=req.num_samples) if rng_key is not None else None
        )

        def produce(item: object) -> dict[str, np.ndarray]:
            start = cast("int", item)
            stop = min(start + batch_size, req.num_samples)
            chunk_samples, chunk_keys, per_device = _prepare_draw_chunk(
                posterior,
                draw_keys=draw_keys,
                start=start,
                stop=stop,
                num_devices=self._ctx.num_devices,
                sharding=self._ctx.partitioned_sharding,
            )
            step = _Step(
                num=per_device,
                keys=chunk_keys,
                samples=chunk_samples,
                x=x_dev,
                y=y_dev,
                tail=tail,
            )

            return {s: a[: stop - start] for s, a in device_get(compute(step)).items()}

        _write_loop(
            items=range(0, req.num_samples, batch_size),
            n_items=-(-req.num_samples // batch_size),
            return_sites=req.return_sites,
            output_dir=req.output_dir,
            strategy=_SliceWriteStrategy(
                zarr_group=open_group(req.output_dir, mode="w"),
                total=req.num_samples,
                batch_size=batch_size,
                axis=0,
            ),
            produce=produce,
            pbar=pbar,
        )
