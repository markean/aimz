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

"""Streaming engine: runs the kernel across shards and streams the outputs.

:class:`_OutputStreamer` caches the sharded callables and the posterior placement, and
drives the data- and draw-parallel write paths into a Zarr store or host memory.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

from jax import Array, device_get, device_put, eval_shape, random, tree
from jax.typing import ArrayLike
from tqdm.auto import tqdm

from aimz.sampling._forward import _sample_forward
from aimz.utils._kwargs import (
    _partition,
    _split_intervention,
    _split_intervention_fields,
)
from aimz.utils._output import (
    _create_slice_strategy,
    _select_write_strategy,
    _write_loop,
)
from aimz.utils.data import ArrayLoader
from aimz.utils.data._input_setup import (
    _prepare_batch,
    _setup_inputs,
)
from aimz.utils.data._sharding import (
    _create_sharded_log_likelihood,
    _create_sharded_sampler,
    _prepare_draw_chunk,
    _replicate,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
    from pathlib import Path

    import numpy as np
    from dask.array import Array as DaskArray
    from jax.sharding import Mesh, Sharding

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
    X: ArrayLike | ArrayLoader | Iterable[Mapping[str, Array | np.ndarray]]
    return_sites: tuple[str, ...]
    num_samples: int
    batch_size: int | None
    artifact_path: Path | None
    """Zarr group path for the results; ``None`` accumulates them in host memory."""
    progress: bool
    loader_rng_key: Array
    kwargs: dict[str, object]
    dims: Mapping[str, Sequence[str]] = field(default_factory=dict)
    """Names of the dimensions after the draw axis, by site, for the Zarr arrays."""


class _Step(NamedTuple):
    """Per-item inputs of one sharded forward call, for a batch or a draw chunk."""

    num: int
    """Draw count for the call: global count (data) or per-device count (draw)."""
    keys: Array | None
    """Per-draw keys for predictive sampling; ``None`` for log-likelihood."""
    samples: dict[str, Array]
    """Conditioning samples: the whole posterior (data) or one draw chunk (draw)."""
    x: Array | np.ndarray
    """Input data: one observation batch (data) or the whole replicated input (draw)."""
    y: Array | np.ndarray | None
    """Output data (log-likelihood only)."""
    tail: tuple
    """Per-observation keyword argument values forwarded after ``x``."""


class _OutputStreamer:
    """Run the kernel across shards and stream the per-item outputs.

    One per model; caches the sharded callables and the posterior placement.
    """

    def __init__(self, ctx: _RuntimeContext) -> None:
        """Set the runtime context and empty caches."""
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
        n_kwargs_const: int,
    ) -> Callable:
        """Build a sharded callable once per key and cache it.

        The argument counts are baked into the ``shard_map`` in-specs, so each arity
        needs its own callable.
        """
        key = (kind, shard_axis, n_kwargs_array, n_kwargs_const)
        fn = self._fn_cache.get(key)
        if fn is None:
            fn = factory(
                self._ctx.mesh,
                n_kwargs_array=n_kwargs_array,
                n_kwargs_const=n_kwargs_const,
                shard_axis=shard_axis,
            )
            self._fn_cache[key] = fn

        return fn

    def _resolve_kwarg_key(
        self,
        req: _WriteRequest,
        batch: Mapping[str, Array | np.ndarray] | None = None,
    ) -> tuple[tuple[str, ...], dict]:
        """Return the per-observation keyword names and the call constants.

        The per-observation arguments are the fields of the first batch; every other
        keyword argument is a constant of the call. Draw streaming has no batch, so all
        of its keyword arguments are constants.
        """
        if batch is None:
            return (), dict(req.kwargs)
        kwargs_key = tuple(
            name
            for name in batch
            if name not in (self._ctx.param_input, self._ctx.param_output)
        )

        return kwargs_key, {k: v for k, v in req.kwargs.items() if k not in batch}

    def setup_stream(
        self,
        req: _WriteRequest,
        y: ArrayLike | None,
    ) -> tuple[
        Iterable[Mapping[str, Array | np.ndarray]],
        Iterator,
        dict[str, Array | np.ndarray],
    ]:
        """Open the observation stream and retain its validated first batch.

        Returns:
            The loader, its iterator with the first batch put back, and that batch.

        Raises:
            TypeError: If ``req.X`` is neither an array nor a loader of batch mappings.
            ValueError: If the loader is empty, or a keyword argument repeats a field.
        """
        loader, _ = _setup_inputs(
            X=req.X,
            y=y,
            param_input=self._ctx.param_input,
            param_output=self._ctx.param_output,
            rng_key=req.loader_rng_key,
            batch_size=req.batch_size,
            shuffle=False,
            **req.kwargs,
        )
        batches = iter(loader)
        try:
            first = next(batches)
        except StopIteration:
            msg = "The data loader must yield at least one nonempty batch."
            raise ValueError(msg) from None
        if not isinstance(first, Mapping):
            msg = (
                "`X` must be an array-like or a data loader yielding batch mappings, "
                f"got {type(req.X).__name__!r}."
            )
            raise TypeError(msg)
        first, _ = _prepare_batch(first, param_input=self._ctx.param_input)
        # A keyword argument named like a loader field would be ignored silently
        if not isinstance(req.X, ArrayLike) and (
            duplicates := sorted(first.keys() & req.kwargs.keys())
        ):
            msg = (
                f"Keyword argument(s) {duplicates} are also fields of the data loader; "
                "pass each argument either way, not both."
            )
            raise ValueError(msg)

        return loader, chain((first,), batches), first

    def place_posterior(
        self,
        posterior: dict[str, Array] | None,
        sharding: Sharding | None,
    ) -> dict[str, Array]:
        """Return the posterior placed on devices, cached by ``sharding``.

        The cache is rebuilt when ``posterior`` is replaced, by identity.
        """
        if not posterior:
            return {}
        if self._posterior_device_src is not posterior:
            self._posterior_device_cache = {}
            self._posterior_device_src = posterior
        cache = self._posterior_device_cache
        if sharding not in cache:
            # Placed once, instead of being transferred on every jit call
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
        params: Mapping[str, object] | None,
        intervention: dict | None = None,
        stream: (
            tuple[
                Iterable[Mapping[str, Array | np.ndarray]],
                Iterator,
                dict[str, Array | np.ndarray],
            ]
            | None
        ) = None,
    ) -> dict[str, DaskArray] | None:
        """Stream predictive samples to the request's destination.

        Under ``shard_axis="obs"`` every batch conditions on the replicated posterior,
        or, under the prior, on the global sites drawn once from a one-row probe, since
        a sharded probe would propagate the mesh axis onto global sites.

        Returns:
            The site arrays over the retained host batches when ``req.artifact_path``
            is ``None``; ``None`` for a Zarr-backed write.
        """
        # Per-observation intervention values are batched with the input as reserved
        # fields
        n_obs = (
            len(cast("Sized", req.X))
            if req.shard_axis == "obs" and isinstance(req.X, ArrayLike)
            else None
        )
        intervention, fields = _split_intervention(intervention, n_obs=n_obs)
        if fields:
            req = replace(req, kwargs={**req.kwargs, **fields})
        if req.shard_axis == "obs" and stream is None:
            stream = self.setup_stream(req, y=None)
        kwargs_key, kwargs_extra = self._resolve_kwarg_key(
            req,
            batch=stream[2] if stream is not None else None,
        )
        kwargs_const, kwargs_static = _partition(kwargs_extra)
        kwargs_const = [
            _replicate(v, sharding=self._ctx.replicated_sharding) for v in kwargs_const
        ]
        params = device_put(params or {}, device=self._ctx.replicated_sharding)
        kind = "prior_predictive" if group == "prior_predictive" else "predict"
        fn = self._cached_fn(
            kind,
            shard_axis=req.shard_axis,
            factory=_create_sharded_sampler,
            n_kwargs_array=len(kwargs_key),
            n_kwargs_const=len(kwargs_const),
        )

        def compute(step: _Step) -> dict[str, Array]:
            return fn(
                kernel,
                step.num,
                step.keys,
                req.return_sites,
                step.samples,
                params,
                intervention,
                self._ctx.param_input,
                kwargs_key,
                kwargs_static,
                step.x,
                *step.tail,
                *kwargs_const,
            )

        phase = "Prior" if group == "prior_predictive" else "Posterior"
        desc = f"{phase} predictive sampling [{', '.join(req.return_sites)}]"
        # Draw-parallel opened no observation stream
        if stream is None:
            chunk_posterior = (
                {}
                if group == "prior_predictive"
                else cast("dict[str, Array]", posterior)
            )
            return self._write_draws(
                req,
                compute=compute,
                y=None,
                posterior=chunk_posterior,
                rng_key=rng_key,
                desc=desc,
            )

        if group == "prior_predictive":
            first = stream[2]
            rng_key, rng_subkey = random.split(rng_key)
            rng_keys = random.split(rng_subkey, num=req.num_samples)

            def probe(batch: Mapping[str, object]) -> dict[str, Array]:
                model_kwargs, fields = _split_intervention_fields(batch)
                return _sample_forward(
                    kernel,
                    rng_keys=rng_keys,
                    return_sites=None,
                    samples=None,
                    params=params,
                    intervention={**intervention, **fields},
                    model_kwargs={**model_kwargs, **kwargs_extra},
                )

            samples = probe({name: arr[:1] for name, arr in first.items()})
            # A site whose shape follows the row count is per observation, so each
            # batch draws it
            shapes = eval_shape(
                probe,
                {name: arr[:1].repeat(2, axis=0) for name, arr in first.items()},
            )
            samples = {
                k: v
                for k, v in samples.items()
                if k not in req.return_sites and v.shape == shapes[k].shape
            }
            if self._ctx.replicated_sharding is not None:
                samples = device_put(samples, device=self._ctx.replicated_sharding)
        else:
            samples = self.place_posterior(posterior, self._ctx.replicated_sharding)

        return self._write_data(
            req,
            compute=compute,
            stream=stream,
            samples=samples,
            kwargs_key=kwargs_key,
            rng_key=rng_key,
            desc=desc,
        )

    def write_log_likelihood(
        self,
        req: _WriteRequest,
        *,
        kernel: Callable,
        posterior: dict[str, Array] | None,
        params: Mapping[str, object] | None,
        y: ArrayLike | None,
    ) -> dict[str, DaskArray] | None:
        """Stream the log-likelihood of the output site to the request's destination.

        Returns:
            The site arrays over the retained host batches when ``req.artifact_path``
            is ``None``; ``None`` for a Zarr-backed write.

        Raises:
            ValueError: If a loader's batches lack the output field.
        """
        site = self._ctx.param_output
        stream = self.setup_stream(req, y=y) if req.shard_axis == "obs" else None
        if stream is not None and site not in stream[2]:
            msg = (
                f"Log likelihood requires the observed output field {site!r} "
                "in each batch."
            )
            raise ValueError(msg)
        kwargs_key, kwargs_extra = self._resolve_kwarg_key(
            req,
            batch=stream[2] if stream is not None else None,
        )
        kwargs_const, kwargs_static = _partition(kwargs_extra)
        kwargs_const = [
            _replicate(v, sharding=self._ctx.replicated_sharding) for v in kwargs_const
        ]
        params = device_put(params or {}, device=self._ctx.replicated_sharding)
        fn = self._cached_fn(
            "log_likelihood",
            shard_axis=req.shard_axis,
            factory=_create_sharded_log_likelihood,
            n_kwargs_array=len(kwargs_key),
            n_kwargs_const=len(kwargs_const),
        )

        def compute(step: _Step) -> dict[str, Array]:
            return {
                site: fn(
                    kernel,
                    step.samples,
                    params,
                    self._ctx.param_input,
                    site,
                    kwargs_key,
                    kwargs_static,
                    step.x,
                    step.y,
                    *step.tail,
                    *kwargs_const,
                ),
            }

        desc = f"Computing log-likelihood [{site}]"
        # Draw-parallel opened no observation stream
        if stream is None:
            return self._write_draws(
                req,
                compute=compute,
                y=cast("ArrayLike", y),
                posterior=cast("dict[str, Array]", posterior),
                rng_key=None,
                desc=desc,
            )

        return self._write_data(
            req,
            compute=compute,
            stream=stream,
            samples=self.place_posterior(
                posterior,
                self._ctx.replicated_sharding,
            ),
            kwargs_key=kwargs_key,
            rng_key=None,
            desc=desc,
        )

    def _write_data(
        self,
        req: _WriteRequest,
        compute: Callable[[_Step], dict[str, Array]],
        stream: tuple[
            Iterable[Mapping[str, Array | np.ndarray]],
            Iterator,
            dict[str, Array | np.ndarray],
        ],
        samples: dict[str, Array],
        kwargs_key: tuple[str, ...],
        rng_key: Array | None,
        desc: str,
    ) -> dict[str, DaskArray] | None:
        """Stream over observation batches, each conditioned on the whole posterior.

        ``dispatch`` runs ``compute`` asynchronously; ``finalize`` waits and trims the
        observation padding.
        """
        dataloader, batches, first = stream
        # Only `ArrayLoader` guarantees its length and aligned chunks
        known_size = type(dataloader) is ArrayLoader
        n_batches = len(dataloader) if known_size else None
        # `ArrayLoader`'s last batch is padded to the first one's size, keeping one
        # compiled shape; other loaders vary their batches freely
        size = first[self._ctx.param_input].shape[0] if known_size else 0

        def dispatch(item: object) -> tuple[dict[str, Array], int]:
            nonlocal rng_key
            batch, n_valid = _prepare_batch(
                item,
                param_input=self._ctx.param_input,
                device=self._ctx.partitioned_sharding,
                size=size,
            )
            if batch.keys() != first.keys() or any(
                arr.shape[1:] != first[name].shape[1:] for name, arr in batch.items()
            ):
                msg = "Batch fields and non-observation shapes must remain consistent."
                raise ValueError(msg)
            subkey = None
            if rng_key is not None:
                # One key per batch, which the sampler splits into per-draw keys
                rng_key, subkey = random.split(rng_key)
                subkey = device_put(subkey, self._ctx.replicated_sharding)
            # In `kwargs_key` order, matching the callable's in-specs
            tail = tuple(batch[name] for name in kwargs_key)
            step = _Step(
                num=req.num_samples,
                keys=subkey,
                samples=samples,
                x=batch[self._ctx.param_input],
                y=batch.get(self._ctx.param_output),
                tail=tail,
            )
            out = compute(step)
            tree.map(lambda arr: arr.copy_to_host_async(), out)

            return out, n_valid

        def finalize(pending: object) -> dict[str, np.ndarray]:
            out, n_valid = cast("tuple[dict[str, Array], int]", pending)
            result = {}
            for site, value in device_get(out).items():
                result[site] = value[:, :n_valid]
            return result

        strategy = _select_write_strategy(
            req.artifact_path,
            total=len(dataloader.dataset) if known_size else None,
            batch_size=first[self._ctx.param_input].shape[0],
            dims=req.dims,
        )
        _write_loop(
            items=batches,
            n_items=n_batches,
            strategy=strategy,
            dispatch=dispatch,
            finalize=finalize,
            pbar=tqdm(
                desc=desc,
                total=n_batches,
                disable=not req.progress,
                dynamic_ncols=True,
            ),
        )

        return strategy.result()

    def _write_draws(
        self,
        req: _WriteRequest,
        compute: Callable[[_Step], dict[str, Array]],
        y: ArrayLike | None,
        posterior: dict[str, Array],
        rng_key: Array | None,
        desc: str,
    ) -> dict[str, DaskArray] | None:
        """Stream over draw chunks, holding the whole input replicated on every device.

        ``dispatch`` runs ``compute`` asynchronously on a padded posterior chunk;
        ``finalize`` waits and trims the padding draws.
        """
        batch_size = cast("int", req.batch_size)
        n_chunks = -(-req.num_samples // batch_size)
        # The entry points reject data loaders under `draw`
        x_dev = _replicate(
            cast("ArrayLike", req.X),
            sharding=self._ctx.replicated_sharding,
        )
        y_dev = (
            _replicate(y, sharding=self._ctx.replicated_sharding)
            if y is not None
            else None
        )
        draw_keys = (
            random.split(rng_key, num=req.num_samples) if rng_key is not None else None
        )

        def dispatch(item: object) -> tuple[dict[str, Array], int]:
            start = cast("int", item)
            stop = min(start + batch_size, req.num_samples)
            chunk_samples, chunk_keys, per_device = _prepare_draw_chunk(
                posterior,
                draw_keys=draw_keys,
                start=start,
                stop=stop,
                size=batch_size,
                num_devices=self._ctx.num_devices,
                sharding=self._ctx.partitioned_sharding,
            )
            step = _Step(
                num=per_device,
                keys=chunk_keys,
                samples=chunk_samples,
                x=x_dev,
                y=y_dev,
                tail=(),
            )
            out = compute(step)
            # Start the device-to-host copy as soon as the result is ready
            tree.map(lambda arr: arr.copy_to_host_async(), out)

            return out, stop - start

        def finalize(pending: object) -> dict[str, np.ndarray]:
            out, n_draws = cast("tuple[dict[str, Array], int]", pending)

            return {s: a[:n_draws] for s, a in device_get(out).items()}

        strategy = _create_slice_strategy(
            req.artifact_path,
            total=req.num_samples,
            batch_size=batch_size,
            axis=0,
            dims=req.dims,
        )
        _write_loop(
            items=range(0, req.num_samples, batch_size),
            n_items=n_chunks,
            strategy=strategy,
            dispatch=dispatch,
            finalize=finalize,
            pbar=tqdm(
                desc=desc,
                total=n_chunks,
                disable=not req.progress,
                dynamic_ncols=True,
            ),
        )

        return strategy.result()
