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

"""Module for handling output files."""

from __future__ import annotations

import logging
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from queue import Queue
from shutil import rmtree
from threading import Event, Thread
from typing import TYPE_CHECKING, Protocol, cast

import psutil
from dask import delayed
from dask.array import concatenate, from_delayed
from zarr import open_group
from zarr.codecs import BloscCodec

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterable,
        Mapping,
        MutableMapping,
    )

    import numpy as np
    from dask.array import Array as DaskArray
    from tqdm.auto import tqdm
    from zarr import Array, Group

    from aimz.utils.data import ArrayLoader


# Maximum in-flight compute steps in `_write_loop`. Depth 2 dispatches the next step
# before collecting the previous one, overlapping device compute with device-to-host
# transfer and the host-side write work; each in-flight step holds one output chunk on
# device, so raising this raises peak device memory proportionally.
_PIPELINE_DEPTH = 2
_QUEUE_SIZE_MAX = 128
# Ceiling on the automatically chosen writer-thread pool size. Guards against
# oversubscribing cores/disk; the effective count is further bounded by the write
# strategy's own ceiling and the number of items.
_WRITER_COUNT_MAX = 8
# Sentinel a concurrency-safe strategy returns from ``max_writers`` to signal "no
# strategy-imposed limit"; the effective cap is then the item count and CPU/ceiling.
_WRITER_COUNT_UNBOUNDED = 2**31 - 1


def _iter_pipelined(
    items: Iterable,
    dispatch: Callable[[object], object],
    finalize: Callable[[object], dict[str, np.ndarray]],
) -> Generator[dict[str, np.ndarray], None, None]:
    """Dispatch up to :data:`_PIPELINE_DEPTH` items ahead and finalize them in order.

    ``dispatch`` launches an item's computation (JAX dispatch is asynchronous, so it
    returns without waiting) and ``finalize`` blocks on the result, so keeping a
    bounded queue of in-flight steps lets the device compute item N+1 while the
    consumer collects and writes item N. Items are finalized in dispatch (FIFO)
    order.

    Args:
        items: Items to iterate (batches paired with keys, or draw-chunk starts).
        dispatch: Launches one item's computation and returns its in-flight handle.
        finalize: Blocks on an in-flight handle and returns its mapping of site name
            to array.

    Yields:
        Each item's mapping of site name to (post-slice) array, in item order.
    """
    pending: deque = deque()
    try:
        for item in items:
            pending.append(dispatch(item))
            if len(pending) >= _PIPELINE_DEPTH:
                yield finalize(pending.popleft())
        while pending:
            yield finalize(pending.popleft())
    except BaseException:
        pending.clear()
        raise


@dataclass(frozen=True)
class _StreamPlan:
    """Writer-pool sizing for one streamed write: pool size and shared queue depth."""

    n_writers: int
    queue_size: int


def _determine_writer_count(
    max_writers: int,
    num_items: int,
    requested: int | None = None,
) -> int:
    """Determine the writer-thread pool size for a stream.

    Without an explicit request the count defaults to the CPU count, capped by
    :data:`_WRITER_COUNT_MAX`. The result is always bounded by the strategy's own
    ceiling (``max_writers``; ``1`` for an order-sensitive strategy) and by
    ``num_items`` (never more writers than there are items to write), and floored at 1.

    Args:
        max_writers: The write strategy's ceiling on concurrent writers.
        num_items: Total number of items the producer will emit.
        requested: Explicit writer count, or ``None`` to choose automatically.

    Returns:
        The number of writer threads to start.
    """
    auto = min(cpu_count() or 1, _WRITER_COUNT_MAX)
    n = auto if requested is None else requested

    return max(1, min(n, max_writers, num_items))


def _plan_writers(
    max_writers: int,
    n_items: int,
    item_nbytes: int,
    n_sites: int,
    requested: int | None = None,
) -> _StreamPlan:
    """Plan the writer pool and shared-queue depth for a streamed write.

    The single place where the host-memory envelope and the concurrency bounds meet.
    Invariants, stated once:

    - In-flight host bytes — queued items, items being applied, and the pipelined
      producer's :data:`_PIPELINE_DEPTH` pre-collected batches — stay within the
      memory available at planning time.
    - The pool never exceeds the strategy's ceiling, the batch count, the CPU-derived
      automatic cap (or the explicit request), or what the memory envelope can feed.
    - Both results are floored at 1, so a bounded queue and at least one writer always
      exist; on an extremely tight envelope this floor (one queued item plus one being
      applied) is the documented minimum footprint.

    Args:
        max_writers: The write strategy's ceiling on concurrent writers.
        n_items: Total number of batches the producer will emit.
        item_nbytes: Bytes one batch commits across all of its sites.
        n_sites: Number of ``(site, payload)`` items each batch enqueues.
        requested: Explicit writer count, or ``None`` to choose automatically.

    Returns:
        The writer-pool plan.
    """
    # Batches of headroom the host affords beyond the pipeline's in-flight steps; NOT
    # clamped by `n_items`, so a small batch count with ample memory still gets a full
    # pool (every batch can be in flight at once).
    mem_batches = (
        psutil.virtual_memory().available // max(1, item_nbytes) - _PIPELINE_DEPTH
    )
    mem_slots = mem_batches * n_sites
    n_writers = max(
        1,
        min(
            _determine_writer_count(max_writers, n_items, requested=requested),
            # Reserve one queued slot so the producer can stay ahead of the pool.
            mem_slots - 1,
        ),
    )
    queue_size = max(
        1,
        min(_QUEUE_SIZE_MAX, n_items * n_sites, mem_slots) - n_writers,
    )
    plan = _StreamPlan(n_writers=n_writers, queue_size=queue_size)
    logger.debug(
        "Write plan: %d batches x %d site(s) (%d bytes/batch), %d writer(s), "
        "queue depth %d",
        n_items,
        n_sites,
        item_nbytes,
        plan.n_writers,
        plan.queue_size,
    )

    return plan


def _site_dtype(arr: np.ndarray) -> np.dtype | str:
    """Destination dtype for a site's array: bfloat16 is upcast to float32.

    Shared by every destination (Zarr and host memory) so the rule cannot drift.
    """
    return "float32" if arr.dtype == "bfloat16" else arr.dtype


def _validate_streamed_axis_size(
    arr: np.ndarray,
    *,
    site: str,
    axis: int,
    chunk_size: int,
) -> None:
    """Verify a site's first batch emits the streamed axis at the batch size.

    Contiguous batches must tile the streamed axis, so a site that does not emit it
    (e.g. a global site with no observation axis under ``axis=1``) cannot be streamed.

    Args:
        arr: The site's first (post-slice) batch array.
        site: The sample site name.
        axis: The streamed axis.
        chunk_size: Expected batch length along the streamed axis.

    Raises:
        NotImplementedError: If the site's streamed-axis size does not match
            ``chunk_size``.
    """
    if arr.shape[axis] != chunk_size:
        requirement = "the input batch size" if axis == 1 else "the draw chunk size"
        msg = (
            f"Slice writing requires each site's axis-{axis} size to match "
            f"{requirement}. Site {site!r} emitted shape {arr.shape} for a batch "
            f"of size {chunk_size}; this kernel is not currently supported under "
            "slice writing."
        )
        raise NotImplementedError(msg)


def _create_site_array(
    zarr_group: Group,
    site: str,
    arr: np.ndarray,
    axis: int,
    total: int,
    chunk: int,
) -> None:
    """Create one Zarr array for a site.

    The streamed ``axis`` is sized to ``total`` (``0`` for append-style growth, the
    full size for preallocated slice writing) and chunked by ``chunk``; every other
    axis is taken whole from ``arr``. The leading axis is always the draw axis, so
    ``dimension_names`` is ``("draw", "<site>_dim_0", ...)``.

    Args:
        zarr_group: The open Zarr group to create the array in.
        site: The sample site name (also the array name).
        arr: A representative (post-slice) sample array; its ``shape``, ``ndim``, and
            ``dtype`` are read.
        axis: The streamed axis (the one filled batch by batch).
        total: Full size of the streamed axis.
        chunk: Chunk length along the streamed axis.
    """
    shape = list(arr.shape)
    shape[axis] = total
    chunks = list(arr.shape)
    chunks[axis] = chunk
    zarr_group.create_array(
        name=site,
        shape=tuple(shape),
        dtype=_site_dtype(arr),
        chunks=tuple(chunks),
        dimension_names=(
            "draw",
            *tuple(f"{site}_dim_{i}" for i in range(arr.ndim - 1)),
        ),
        compressors=BloscCodec(cname="zstd", clevel=3, shuffle="shuffle"),
    )


class _WriteStrategy(Protocol):
    """How a batch of per-site samples is created and queued for writing.

    A strategy streams one **axis** of each site's destination array, filling it batch
    by batch while every other axis is written whole: ``axis=0`` streams the draw axis
    (draw-parallel), ``axis=1`` streams the observation axis (data-parallel). The
    strategy also owns where the results land.
    """

    @property
    def max_writers(self) -> int:
        """Maximum number of concurrent writer threads this strategy tolerates.

        ``1`` means the strategy is order-sensitive and must be written by a single
        consumer; a larger value means writes are order-independent and may be spread
        across a pool of workers.
        """

    @property
    def sink(self) -> Path | MutableMapping[str, list[np.ndarray]]:
        """Where writer threads land payloads.

        The path of a Zarr group (each worker opens it itself) or a shared
        in-memory mapping of site name to destination, indexed the same way.
        """

    def apply(self, array: Array | list[np.ndarray], item: object) -> None:
        """Write one queued item into a site's destination.

        Args:
            array: The site's destination (a Zarr array, or an in-memory batch
                list).
            item: The queued payload to write.
        """

    def create_arrays(self, site_arrays: Mapping[str, np.ndarray]) -> None:
        """Create any not-yet-created destination arrays for the sites in a batch.

        Args:
            site_arrays: Mapping of site name to the (post-slice) sample array emitted
                for the current batch.
        """

    def enqueue(
        self,
        queue: Queue,
        site_arrays: Mapping[str, np.ndarray],
    ) -> None:
        """Put a batch's per-site payloads onto the shared writer queue.

        Each payload is enqueued as a ``(site, payload)`` item so any worker in the
        pool can route it to the right destination. By default the payload is the
        site's batch array itself.

        Args:
            queue: The shared writer queue.
            site_arrays: Mapping of site name to the (post-slice) sample array
                emitted for the current batch.
        """
        for site, arr in site_arrays.items():
            queue.put((site, arr))

    def result(self) -> dict[str, DaskArray] | None:
        """The accumulated site arrays for an in-memory sink, or ``None``.

        Defaults to ``None``, as for the Zarr-backed strategies, whose results live
        at :attr:`sink`. An in-memory strategy overrides this to return its finished
        site arrays, Dask-backed over the retained host batches.
        """


class _AppendWriteStrategy(_WriteStrategy):
    """Grow each site's Zarr array by appending batches along the streamed axis.

    Requires no size information up front; the streamed-axis size emerges from the
    batches as they arrive. ``array.append`` mutates the array's length metadata and is
    order-sensitive, so this strategy is not concurrency-safe. It must be written by a
    single consumer (:attr:`max_writers` is ``1``).

    This is the designated fallback for data loaders with unknown length, broadening the
    accepted loaders beyond :class:`~aimz.utils.data.ArrayLoader`. Note the write side
    is not yet reachable by such loaders: see :func:`_select_write_strategy` for what
    the streaming path still assumes. If single-consumer appends ever become the
    bottleneck for that use case, a windowed hybrid (grow the array by a window of
    batches, slice-write concurrently within each window) can restore pool parallelism
    as a third strategy without touching :func:`_write_loop`.
    """

    def __init__(
        self,
        *,
        artifact_path: Path,
        batch_size: int,
        axis: int,
    ) -> None:
        """Initialize the append write strategy.

        Args:
            artifact_path: Path of the Zarr group to create site arrays in (opened
                for writing here).
            batch_size: Chunk length along the streamed axis.
            axis: The streamed axis to grow (``0`` for draws, ``1`` for observations).
        """
        self._artifact_path = artifact_path
        self._zarr_group = open_group(artifact_path, mode="w")
        self._chunk_size = batch_size
        self._axis = axis
        self._seen: set[str] = set()

    @property
    def max_writers(self) -> int:
        """A single writer: appends are order-sensitive and mutate array metadata."""
        return 1

    @property
    def sink(self) -> Path:
        """The Zarr group's path; each writer thread opens the group itself."""
        return self._artifact_path

    def apply(self, array: Array | list[np.ndarray], item: object) -> None:
        """Append the queued batch along the streamed axis.

        Args:
            array: The site's Zarr array.
            item: The batch array to append.
        """
        cast("Array", array).append(cast("np.ndarray", item), axis=self._axis)

    def create_arrays(self, site_arrays: Mapping[str, np.ndarray]) -> None:
        """Create zero-width Zarr arrays for sites not yet seen.

        Args:
            site_arrays: Mapping of site name to the (post-slice) sample array emitted
                for the current batch.
        """
        for site, arr in site_arrays.items():
            if site not in self._seen:
                _create_site_array(
                    self._zarr_group,
                    site=site,
                    arr=arr,
                    axis=self._axis,
                    total=0,
                    chunk=self._chunk_size,
                )
                self._seen.add(site)


class _SliceWriteStrategy(_WriteStrategy):
    """Write each batch into a fixed slice of a preallocated Zarr array along an axis.

    Requires the streamed-axis size up front. Every batch must emit a streamed-axis
    size equal to the batch size, so contiguous slices tile the full axis; a site that
    does not (e.g. a global site with no observation axis under ``axis=1``) raises
    :exc:`NotImplementedError` on the first batch. With ``axis=0`` it streams the draw
    axis (draw-parallel), where every site matches by construction; with ``axis=1`` the
    observation axis (data-parallel).
    """

    def __init__(
        self,
        *,
        artifact_path: Path,
        total: int,
        batch_size: int,
        axis: int,
    ) -> None:
        """Initialize the slice write strategy.

        Args:
            artifact_path: Path of the Zarr group to create site arrays in (opened
                for writing here).
            total: Full size of the streamed axis (preallocated up front).
            batch_size: Chunk length along the streamed axis.
            axis: The streamed axis to fill (``0`` for draws, ``1`` for observations).
        """
        self._artifact_path = artifact_path
        self._zarr_group = open_group(artifact_path, mode="w")
        self._total = total
        self._chunk_size = min(batch_size, total)
        self._axis = axis
        self._seen: set[str] = set()
        self._site_offsets: dict[str, int] = {}

    @property
    def max_writers(self) -> int:
        """No strategy limit: slice writes are position-addressed and disjoint.

        Each batch write targets a fixed, chunk-aligned slice (one Zarr chunk == one
        file on a local store), so writers never contend and completion order is
        irrelevant. The effective pool size is bounded elsewhere by the CPU count /
        :data:`_WRITER_COUNT_MAX` (or an explicit request) and the item count.
        """
        return _WRITER_COUNT_UNBOUNDED

    @property
    def sink(self) -> Path:
        """The Zarr group's path; each writer thread opens the group itself."""
        return self._artifact_path

    def apply(self, array: Array | list[np.ndarray], item: object) -> None:
        """Assign the queued ``(start, arr)`` batch into a fixed slice of the axis.

        Args:
            array: The site's (preallocated) Zarr array.
            item: A ``(start, arr)`` tuple; ``arr`` is written to the streamed-axis
                slice ``[start : start + arr.shape[axis])``.
        """
        start, arr = cast("tuple[int, np.ndarray]", item)
        idx: list = [slice(None)] * arr.ndim
        idx[self._axis] = slice(start, start + arr.shape[self._axis])
        cast("Array", array)[tuple(idx)] = arr

    def create_arrays(self, site_arrays: Mapping[str, np.ndarray]) -> None:
        """Preallocate full-size Zarr arrays for sites not yet seen.

        On the first call (the first batch), every site is verified to emit a
        streamed-axis size equal to the batch size; mismatches raise. After creation,
        each site is registered in ``self._site_offsets`` with offset zero so subsequent
        batches can be written to fixed slices.

        Args:
            site_arrays: Mapping of site name to the (post-slice) sample array emitted
                for the current batch.

        Raises:
            NotImplementedError: If any return site emits a streamed-axis size that does
                not match the batch size.
        """
        for site, arr in site_arrays.items():
            if site not in self._seen:
                _validate_streamed_axis_size(
                    arr,
                    site=site,
                    axis=self._axis,
                    chunk_size=self._chunk_size,
                )
                _create_site_array(
                    self._zarr_group,
                    site=site,
                    arr=arr,
                    axis=self._axis,
                    total=self._total,
                    chunk=self._chunk_size,
                )
                self._seen.add(site)
                self._site_offsets[site] = 0

    def enqueue(
        self,
        queue: Queue,
        site_arrays: Mapping[str, np.ndarray],
    ) -> None:
        """Enqueue each site's batch as ``(site, (start, arr))`` and advance its offset.

        Args:
            queue: The shared writer queue.
            site_arrays: Mapping of site name to the batch array to write.
        """
        for site, arr in site_arrays.items():
            start = self._site_offsets[site]
            queue.put((site, (start, arr)))
            self._site_offsets[site] = start + arr.shape[self._axis]


class _MemoryWriteStrategy(_WriteStrategy):
    """Retain each site's batches in host memory as the chunks of a Dask array.

    The finalized batch arrays are kept in arrival order (a single consumer) and
    :meth:`result` assembles each site into a lazy Dask array whose chunks are those
    resident batches, mirroring the chunk layout the Zarr-backed strategies persist.
    """

    def __init__(
        self,
        *,
        total: int,
        batch_size: int,
        axis: int,
    ) -> None:
        """Initialize the in-memory write strategy.

        Args:
            total: Full size of the streamed axis.
            batch_size: Chunk length along the streamed axis.
            axis: The streamed axis the batches tile (``0`` for draws, ``1`` for
                observations).
        """
        self._chunk_size = min(batch_size, total)
        self._axis = axis
        self._batches: dict[str, list[np.ndarray]] = {}

    @property
    def max_writers(self) -> int:
        """A single writer: batches are retained in arrival order."""
        return 1

    @property
    def sink(self) -> dict[str, list[np.ndarray]]:
        """The shared mapping of site name to its retained batches (never rebound)."""
        return self._batches

    def apply(self, array: Array | list[np.ndarray], item: object) -> None:
        """Retain the queued batch as one future chunk of the site's array.

        Args:
            array: The site's batch list.
            item: The batch array to retain.
        """
        cast("list[np.ndarray]", array).append(cast("np.ndarray", item))

    def create_arrays(self, site_arrays: Mapping[str, np.ndarray]) -> None:
        """Register batch lists for sites not yet seen.

        On the first call (the first batch), every site is verified to emit a
        streamed-axis size equal to the batch size; mismatches raise.

        Args:
            site_arrays: Mapping of site name to the (post-slice) sample array emitted
                for the current batch.

        Raises:
            NotImplementedError: If any return site emits a streamed-axis size that does
                not match the batch size.
        """
        for site, arr in site_arrays.items():
            if site not in self._batches:
                _validate_streamed_axis_size(
                    arr,
                    site=site,
                    axis=self._axis,
                    chunk_size=self._chunk_size,
                )
                self._batches[site] = []

    def result(self) -> dict[str, DaskArray]:
        """Assemble each site's retained batches into a lazy Dask array.

        Chunks reference the retained batches directly via
        :func:`dask.array.from_delayed` (:func:`dask.array.from_array` would copy
        them); bfloat16 batches are upcast lazily per the shared dtype rule.

        Returns:
            The finished site arrays, Dask-backed over host memory.
        """
        out = {}
        for site, batches in self._batches.items():
            arr = concatenate(
                [
                    from_delayed(delayed(b, pure=False), shape=b.shape, dtype=b.dtype)
                    for b in batches
                ],
                axis=self._axis,
            )
            dtype = _site_dtype(batches[0])
            out[site] = arr if arr.dtype == dtype else arr.astype(dtype)

        return out


def _create_slice_strategy(
    artifact_path: Path | None,
    *,
    total: int,
    batch_size: int,
    axis: int,
) -> _WriteStrategy:
    """Build the slice-writing strategy for a stream with a known streamed-axis size.

    The single place where the destination is chosen: Zarr-backed when an artifact
    path is given, host-memory accumulation otherwise.

    Args:
        artifact_path: Path of the Zarr group to create site arrays in, or ``None``
            to accumulate the results in host memory.
        total: Full size of the streamed axis.
        batch_size: Chunk length along the streamed axis.
        axis: The streamed axis to fill (``0`` for draws, ``1`` for observations).

    Returns:
        The write strategy to use.
    """
    if artifact_path is None:
        return _MemoryWriteStrategy(total=total, batch_size=batch_size, axis=axis)

    return _SliceWriteStrategy(
        artifact_path=artifact_path,
        total=total,
        batch_size=batch_size,
        axis=axis,
    )


def _select_write_strategy(
    artifact_path: Path | None,
    dataloader: ArrayLoader,
) -> _WriteStrategy:
    """Build the data-parallel write strategy for a data loader.

    Slice writing needs the batch count and dataset size up front; if either is
    unavailable the append strategy is used instead. Both stream the observation
    (axis-1) dimension.

    The append fallback anticipates loaders with unknown length, but it is reachable
    only by a loader that defines ``__len__`` while its dataset does not: the streaming
    path upstream still assumes a known batch count, so a fully length-less loader fails
    before strategy selection. Supporting such loaders means relaxing those assumptions
    (e.g. lazy per-batch key derivation, an unknown progress total, and ``n_items=None``
    planning) alongside an adapter for the batch format.

    Args:
        artifact_path: Path of the Zarr group to create site arrays in, or ``None``
            to accumulate the results in host memory.
        dataloader: The data loader the sample loop will iterate.

    Returns:
        The write strategy to use.

    Raises:
        NotImplementedError: If the results are to accumulate in host memory while the
            data loader's dataset size is unavailable (in-memory accumulation has no
            append fallback).
    """
    try:
        len(dataloader)
        n_obs = len(dataloader.dataset)
    except (TypeError, AttributeError):
        if artifact_path is None:
            msg = (
                "In-memory accumulation requires the data loader's dataset size up "
                "front. Provide a sized dataset or use the persistent store."
            )
            raise NotImplementedError(msg) from None
        return _AppendWriteStrategy(
            artifact_path=artifact_path,
            batch_size=dataloader.batch_size,
            axis=1,
        )

    return _create_slice_strategy(
        artifact_path,
        total=n_obs,
        batch_size=dataloader.batch_size,
        axis=1,
    )


def _writer(
    queue: Queue,
    sink: Path | MutableMapping[str, list[np.ndarray]],
    error_queue: Queue,
    stop: Event,
    apply: Callable[[Array | list[np.ndarray], object], None],
) -> None:
    """Background worker that writes queued ``(site, payload)`` items to the sink.

    One of a shared pool of interchangeable workers that all consume the same queue, so
    a worker is not bound to a single site: each item names the site to write. Runs in a
    loop, retrieving items and writing each into its site's array via ``apply``, exiting
    when a ``None`` sentinel is received. A path sink is a Zarr group each worker opens
    itself. A mapping sink is shared by reference and indexed directly. Concurrency
    safety rests on the strategy: the pool is only sized above one for
    order-independent, disjoint-write strategies (see
    :meth:`_WriteStrategy.max_writers`).

    If opening the group or a write fails, the error is logged, its details are put into
    ``error_queue``, and the shared ``stop`` event is set so every worker switches to
    drain mode — subsequent items are discarded (still marked done, so the bounded
    producer cannot block and ``queue.join()`` can finish) rather than written into a
    store that is being torn down.

    Args:
        queue: The shared queue of ``(site, payload)`` items (and ``None`` sentinels).
        sink: The path of the Zarr group (opened here, per worker), or a shared
            mapping of site name to destination array.
        error_queue: The queue to collect errors raised by the writer threads, each as a
            ``(site, exc, traceback)`` tuple (``site`` is ``None`` for an open failure).
        stop: Shared event; set on the first error to put the pool into drain mode.
        apply: Writes one queued payload into a site's destination array; the only
            behavior that differs between write strategies.
    """
    group = None
    try:
        group = open_group(sink, mode="r+") if isinstance(sink, Path) else sink
    except Exception as exc:
        # `stop.set()` first. It cannot fail, so the pool always enters drain mode even
        # if reporting or logging below raises (e.g. under memory pressure); the
        # suppress keeps this worker alive to drain the queue regardless.
        stop.set()
        with suppress(Exception):
            error_queue.put((None, exc, exc.__traceback__))
            logger.exception("Error opening output group")

    while True:
        item = queue.get()
        try:
            if item is None:
                return
            if stop.is_set():
                # Drain mode: discard the payload but keep the queue moving so the
                # bounded producer never blocks and the sentinels are still consumed.
                continue
            site, payload = cast("tuple[str, object]", item)
            try:
                array = cast("Group | Mapping", group)[site]
                apply(cast("Array | list[np.ndarray]", array), payload)
            except Exception as exc:
                stop.set()
                with suppress(Exception):
                    error_queue.put((site, exc, exc.__traceback__))
                    logger.exception("Error writing to site '%s'", site)
        finally:
            queue.task_done()


def _start_writer_threads(
    sink: Path | MutableMapping[str, list[np.ndarray]],
    apply: Callable[[Array | list[np.ndarray], object], None],
    n_writers: int,
    queue_size: int,
) -> tuple[list[Thread], Queue, Queue, Event]:
    """Start a shared pool of writer threads consuming one queue.

    Args:
        sink: The Zarr group path or shared in-memory mapping the workers write to.
        apply: Writes one queued payload into a site's destination array.
        n_writers: Number of writer threads in the pool.
        queue_size: Maximum size of the shared work queue.

    Returns:
        A tuple of the worker threads, the shared work queue, the shared error queue,
        and the shared stop event.
    """
    queue: Queue = Queue(queue_size)
    error_queue: Queue = Queue()
    stop = Event()
    threads = []
    try:
        for _ in range(n_writers):
            thread = Thread(
                target=_writer,
                args=(queue, sink, error_queue, stop),
                kwargs={"apply": apply},
            )
            thread.start()
            threads.append(thread)
    except BaseException:
        # A mid-pool `Thread.start()` failure (e.g. hitting a thread limit) would
        # otherwise orphan the already-started non-daemon workers on a queue the
        # caller never receives, blocking interpreter exit; unwind them first.
        for _ in threads:
            queue.put(None)
        for thread in threads:
            thread.join()
        raise

    return threads, queue, error_queue, stop


def _shutdown_writer_threads(
    threads: list[Thread],
    queue: Queue | None,
) -> None:
    """Signal the writer pool to stop and wait for its completion.

    One ``None`` sentinel is enqueued per worker; each worker consumes exactly one and
    exits, so any residual items (already ahead of the sentinels in the FIFO queue) are
    consumed first. Safe to call when no pool was started (``queue is None``).

    Args:
        threads: The worker threads to join.
        queue: The shared work queue, or ``None`` if no pool was started.
    """
    if queue is not None:
        for _ in threads:
            queue.put(None)
    for thread in threads:
        thread.join()


def _discard_partial_output(sink: Path | MutableMapping[str, list[np.ndarray]]) -> None:
    """Discard a failed stream's partial output.

    A path sink has its on-disk artifacts removed; a mapping sink is emptied
    eagerly, since a held traceback can keep the strategy alive and would
    otherwise pin the partial batches with it.

    Args:
        sink: The failed stream's destination.
    """
    if isinstance(sink, Path):
        rmtree(sink, ignore_errors=True)
        logger.warning("Cleaned up artifact path: %s", sink)
    else:
        for batches in sink.values():
            batches.clear()
        sink.clear()


def _write_loop(
    items: Iterable,
    n_items: int,
    strategy: _WriteStrategy,
    dispatch: Callable[[object], object],
    finalize: Callable[[object], dict[str, np.ndarray]],
    pbar: tqdm,
    num_writers: int | None = None,
) -> None:
    """Produce per-item site arrays and write them concurrently to the strategy's sink.

    Shared by the data- and draw-parallel write paths. Items are produced through
    :func:`_iter_pipelined`, which keeps consecutive items' computations in flight on
    the device and finalizes them in item order — preserving the offset bookkeeping the
    strategies rely on. Array creation/enqueuing is delegated to ``strategy`` and
    writing to a shared pool of background writer threads. The pool size is chosen from
    the strategy's :attr:`~_WriteStrategy.max_writers` ceiling (``1`` pins an
    order-sensitive strategy to a single consumer) and the item count; ``num_writers``
    overrides the automatic count.

    Args:
        items: Items to iterate (batches paired with keys, or draw-chunk starts).
        n_items: Number of items (used to size the writer queue and the pool).
        strategy: Write strategy that creates and enqueues each item's site arrays
            and owns the destination (:attr:`~_WriteStrategy.sink`).
        dispatch: Launches one item's computation and returns its in-flight handle.
        finalize: Blocks on an in-flight handle and returns its mapping of site name
            to array.
        pbar: Progress bar instance to display progress.
        num_writers: Explicit writer-thread pool size, or ``None`` (the production
            path) to choose automatically. Bounded by the strategy's ceiling and the
            item count; overriding is intended for tests.

    Raises:
        Exception: Any exception raised during production or writing is logged, the
            partial output at the strategy's sink is discarded (on-disk artifacts
            removed, in-memory batches released), and the exception is re-raised.
    """
    threads: list[Thread] = []
    queue: Queue | None = None
    error_queue: Queue | None = None
    stop: Event | None = None
    worker_err: tuple | None = None
    completed = False
    success = False
    producer = _iter_pipelined(items, dispatch=dispatch, finalize=finalize)
    try:
        for sliced in producer:
            strategy.create_arrays(sliced)
            if queue is None:
                plan = _plan_writers(
                    strategy.max_writers,
                    n_items=n_items,
                    item_nbytes=sum(int(arr.nbytes) for arr in sliced.values()),
                    n_sites=max(1, len(sliced)),
                    requested=num_writers,
                )
                threads, queue, error_queue, stop = _start_writer_threads(
                    sink=strategy.sink,
                    apply=strategy.apply,
                    n_writers=plan.n_writers,
                    queue_size=plan.queue_size,
                )
            strategy.enqueue(queue, site_arrays=sliced)
            if stop is not None and stop.is_set():
                if not cast("Queue", error_queue).empty():
                    worker_err = cast("Queue", error_queue).get()
                break
            pbar.update()
        if worker_err is None:
            pbar.set_description(
                "Writing in progress..."
                if isinstance(strategy.sink, Path)
                else "Collecting results...",
            )
        completed = True
    finally:
        _shutdown_writer_threads(threads, queue=queue)
        # Drop the in-flight pipeline results and the current batch explicitly.
        producer.close()
        with suppress(NameError):
            del sliced
        if worker_err is None and error_queue is not None and not error_queue.empty():
            worker_err = error_queue.get()
        # `stop` set without a reported error means a writer failed while reporting
        # (e.g. under memory pressure); treat it as a failure, never as a clean run.
        success = (
            completed and worker_err is None and (stop is None or not stop.is_set())
        )
        if not success:
            _discard_partial_output(strategy.sink)
        pbar.close()
    if worker_err is not None:
        _, exc, tb = worker_err
        raise exc.with_traceback(tb)
    if not success:
        msg = "A background writer thread failed without reporting an error."
        raise RuntimeError(msg)
