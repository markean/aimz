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
from typing import TYPE_CHECKING, Protocol, cast, override
from warnings import warn

import psutil
from dask import delayed
from dask.array import concatenate, from_delayed
from zarr import open_group
from zarr.codecs import BloscCodec

from aimz._exceptions import _SKIP_FILE_PREFIXES, PerformanceWarning
from aimz.utils._format import _group_dims

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterable,
        Mapping,
        MutableMapping,
        Sequence,
    )

    import numpy as np
    from dask.array import Array as DaskArray
    from jax import Array as JaxArray
    from tqdm.auto import tqdm
    from zarr import Array, Group
    from zarr.core.array_spec import ArrayConfigParams


# Compute steps in flight in `_write_loop`: the next step is dispatched before the
# previous one is collected, and each holds one output chunk on device
_PIPELINE_DEPTH = 2
# Ceiling on the items queued plus being written; a deeper queue adds no throughput
_QUEUE_SIZE_MAX = 16
# Ceiling on the automatic writer-thread count
_WRITER_COUNT_MAX = 8
# `max_writers` of a strategy that imposes no limit of its own
_WRITER_COUNT_UNBOUNDED = 2**31 - 1
# Every chunk is written, even one holding only the fill value, so a chunk missing on
# read means the artifact was removed, which `_zarr_to_datatree` reports as an error.
_ARRAY_CONFIG: ArrayConfigParams = {"write_empty_chunks": True}


def _iter_pipelined(
    items: Iterable,
    dispatch: Callable[[object], object],
    finalize: Callable[[object], dict[str, np.ndarray]],
) -> Generator[dict[str, np.ndarray], None, None]:
    """Dispatch up to :data:`_PIPELINE_DEPTH` items ahead and finalize them in order.

    JAX dispatch returns before the computation ends, so the device computes the next
    item while the consumer writes the previous one.
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
    num_items: int | None,
    requested: int | None = None,
) -> int:
    """Return the writer-thread count for a stream.

    The request, or the CPU count capped by :data:`_WRITER_COUNT_MAX`, bounded by the
    strategy's ``max_writers`` and the item count, and floored at one.
    """
    auto = min(cpu_count() or 1, _WRITER_COUNT_MAX)
    n = auto if requested is None else requested

    return max(1, min(n, max_writers, num_items if num_items is not None else n))


def _plan_writers(
    max_writers: int,
    n_items: int | None,
    item_nbytes: int,
    n_sites: int,
    requested: int | None = None,
    *,
    retained: bool = False,
) -> _StreamPlan:
    """Plan the writer pool and the depth of its shared queue.

    The items in flight on the host, plus the batches a memory store keeps, stay within
    the memory available at planning time; the pool never exceeds the strategy's
    ceiling, the item count, or what that memory can feed.

    Args:
        max_writers: The write strategy's ceiling on concurrent writers.
        n_items: Total batches, or ``None`` when the stream length is unknown.
        item_nbytes: Bytes the first batch commits across all sites.
        n_sites: Number of ``(site, payload)`` items each batch enqueues.
        requested: Explicit writer count, or ``None`` to choose automatically.
        retained: Whether the store keeps every batch on the host for the whole call.

    Warns:
        PerformanceWarning: If the batches the store keeps exceed the memory available.
    """
    available = psutil.virtual_memory().available
    resident = n_items * item_nbytes if retained and n_items is not None else 0
    if resident > available:
        msg = (
            "The output of this call exceeds the memory available, and "
            '`store="memory"` keeps all of it on the host; pass `store="persistent"` '
            "to write it out instead."
        )
        warn(msg, category=PerformanceWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
    # Headroom in batches beyond the pipeline's steps and the kept batches, not clamped
    # by `n_items`
    mem_batches = (available - resident) // max(1, item_nbytes) - _PIPELINE_DEPTH
    mem_slots = mem_batches * n_sites
    n_writers = max(
        1,
        min(
            _determine_writer_count(max_writers, n_items, requested=requested),
            # One queued slot is reserved for the producer to stay ahead of the pool
            mem_slots - 1,
        ),
    )
    queue_size = max(
        1,
        min(
            _QUEUE_SIZE_MAX,
            n_items * n_sites if n_items is not None else _QUEUE_SIZE_MAX,
            mem_slots,
        )
        - n_writers,
    )
    plan = _StreamPlan(n_writers=n_writers, queue_size=queue_size)
    logger.debug(
        "Write plan: %s batches x %d site(s) (%d bytes/batch), %d writer(s), "
        "queue depth %d",
        n_items,
        n_sites,
        item_nbytes,
        plan.n_writers,
        plan.queue_size,
    )

    return plan


def _site_dtype(arr: np.ndarray) -> np.dtype | str:
    """Return the destination dtype of a site's array: bfloat16 becomes float32."""
    return "float32" if arr.dtype == "bfloat16" else arr.dtype


def _validate_streamed_axis_size(
    arr: np.ndarray | JaxArray,
    *,
    site: str,
    axis: int,
    chunk_size: int,
) -> None:
    """Raise unless a site emits the streamed axis at the batch size.

    Raises:
        NotImplementedError: If the site has no streamed axis, or its size does not
            match ``chunk_size``.
    """
    if arr.ndim <= axis or arr.shape[axis] != chunk_size:
        requirement = "the input batch size" if axis == 1 else "the draw chunk size"
        msg = (
            f"Streaming requires each site's axis-{axis} size to match "
            f"{requirement}. Site {site!r} emitted shape {arr.shape} for a batch "
            f"of size {chunk_size}. "
        ) + (
            'Pass `shard_axis="draw"`, or leave the site out of `return_sites` where '
            "the method takes it."
            if axis == 1
            else "This kernel is not currently supported under streaming."
        )
        raise NotImplementedError(msg)


def _create_site_array(
    zarr_group: Group,
    site: str,
    arr: np.ndarray,
    axis: int,
    total: int,
    chunk: int,
    dims: Sequence[str],
) -> None:
    """Create a site's Zarr array, sized ``total`` and chunked by ``chunk`` on ``axis``.

    ``total`` is zero for append-style growth. The leading axis is the draw axis, so
    the dimension names are ``("draw", *dims)``.
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
        dimension_names=("draw", *dims),
        compressors=BloscCodec(cname="zstd", clevel=1, shuffle="shuffle"),
    )


class _WriteStrategy(Protocol):
    """How a batch of per-site samples is created and queued for writing.

    A strategy fills one axis of each site's destination batch by batch, ``axis=0`` the
    draws and ``axis=1`` the observations, and owns where the results land.
    """

    @property
    def max_writers(self) -> int:
        """Maximum concurrent writers; ``1`` for an order-sensitive strategy."""

    @property
    def sink(self) -> Path | MutableMapping[str, list[np.ndarray]]:
        """The Zarr group's path, or the shared mapping of site name to batches."""

    def apply(self, array: Array | list[np.ndarray], item: object) -> None:
        """Write one queued payload into a site's destination."""

    def create_arrays(self, site_arrays: Mapping[str, np.ndarray]) -> None:
        """Create the destination arrays of the sites not yet seen."""

    def enqueue(
        self,
        queue: Queue,
        site_arrays: Mapping[str, np.ndarray],
    ) -> None:
        """Put a batch's ``(site, payload)`` items onto the shared writer queue."""
        for site, arr in site_arrays.items():
            queue.put((site, arr))

    def result(self) -> dict[str, DaskArray] | None:
        """Return the accumulated site arrays of an in-memory sink, else ``None``."""
        return None


class _AppendWriteStrategy(_WriteStrategy):
    """Grow each site's Zarr array by appending batches along the streamed axis.

    Needs no size up front, so it serves generic data loaders. Appends change the
    array's length, so a single writer consumes them in order.
    """

    def __init__(
        self,
        *,
        artifact_path: Path,
        batch_size: int,
        axis: int,
        dims: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Open the Zarr group at ``artifact_path`` for writing."""
        self._artifact_path = artifact_path
        self._zarr_group = open_group(artifact_path, mode="w")
        self._chunk_size = batch_size
        self._axis = axis
        self._dims = dims or {}
        self._seen: set[str] = set()

    @property
    def max_writers(self) -> int:
        """A single writer: appends are order-sensitive."""
        return 1

    @property
    def sink(self) -> Path:
        """The Zarr group's path."""
        return self._artifact_path

    def apply(self, array: Array | list[np.ndarray], item: object) -> None:
        """Append the batch along the streamed axis."""
        cast("Array", array).with_config(_ARRAY_CONFIG).append(
            cast("np.ndarray", item),
            axis=self._axis,
        )

    def create_arrays(self, site_arrays: Mapping[str, np.ndarray]) -> None:
        """Create zero-length arrays for the sites not yet seen."""
        names = _group_dims(
            {site: arr.shape[1:] for site, arr in site_arrays.items()},
            self._dims,
        )
        for site, arr in site_arrays.items():
            if site not in self._seen:
                _create_site_array(
                    self._zarr_group,
                    site=site,
                    arr=arr,
                    axis=self._axis,
                    total=0,
                    chunk=self._chunk_size,
                    dims=names[site],
                )
                self._seen.add(site)


class _SliceWriteStrategy(_WriteStrategy):
    """Write each batch into its slice of a preallocated Zarr array.

    Needs the streamed-axis size up front, and every batch must emit that axis at the
    batch size, so the slices tile it; the first batch is checked.
    """

    def __init__(
        self,
        *,
        artifact_path: Path,
        total: int,
        batch_size: int,
        axis: int,
        dims: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Open the Zarr group at ``artifact_path`` for writing."""
        self._artifact_path = artifact_path
        self._zarr_group = open_group(artifact_path, mode="w")
        self._total = total
        self._chunk_size = min(batch_size, total)
        self._axis = axis
        self._dims = dims or {}
        self._seen: set[str] = set()
        self._site_offsets: dict[str, int] = {}

    @property
    def max_writers(self) -> int:
        """No limit: the slices are disjoint and chunk-aligned."""
        return _WRITER_COUNT_UNBOUNDED

    @property
    def sink(self) -> Path:
        """The Zarr group's path."""
        return self._artifact_path

    def apply(self, array: Array | list[np.ndarray], item: object) -> None:
        """Assign the queued ``(start, arr)`` batch to its slice of the axis."""
        start, arr = cast("tuple[int, np.ndarray]", item)
        idx: list = [slice(None)] * arr.ndim
        idx[self._axis] = slice(start, start + arr.shape[self._axis])
        cast("Array", array).with_config(_ARRAY_CONFIG)[tuple(idx)] = arr

    def create_arrays(self, site_arrays: Mapping[str, np.ndarray]) -> None:
        """Preallocate the arrays of the sites not yet seen, checking their batch size.

        Raises:
            NotImplementedError: If a site's streamed-axis size differs from the batch
                size.
        """
        names = _group_dims(
            {site: arr.shape[1:] for site, arr in site_arrays.items()},
            self._dims,
        )
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
                    dims=names[site],
                )
                self._seen.add(site)
                self._site_offsets[site] = 0

    @override
    def enqueue(
        self,
        queue: Queue,
        site_arrays: Mapping[str, np.ndarray],
    ) -> None:
        """Enqueue each site's batch with its offset, then advance the offset."""
        for site, arr in site_arrays.items():
            start = self._site_offsets[site]
            queue.put((site, (start, arr)))
            self._site_offsets[site] = start + arr.shape[self._axis]


class _MemoryWriteStrategy(_WriteStrategy):
    """Keep each site's batches in host memory as the chunks of a Dask array.

    A single consumer keeps them in arrival order, mirroring the persisted chunks.
    """

    def __init__(self, *, axis: int) -> None:
        """Set the streamed axis the batches tile."""
        self._axis = axis
        self._batches: dict[str, list[np.ndarray]] = {}

    @property
    def max_writers(self) -> int:
        """A single writer keeps the batches in arrival order."""
        return 1

    @property
    def sink(self) -> dict[str, list[np.ndarray]]:
        """The mapping of site name to its retained batches, never rebound."""
        return self._batches

    def apply(self, array: Array | list[np.ndarray], item: object) -> None:
        """Retain the batch as one future chunk of the site's array."""
        cast("list[np.ndarray]", array).append(cast("np.ndarray", item))

    def create_arrays(self, site_arrays: Mapping[str, np.ndarray]) -> None:
        """Register the batch lists of the sites not yet seen."""
        for site in site_arrays:
            self._batches.setdefault(site, [])

    @override
    def result(self) -> dict[str, DaskArray]:
        """Assemble each site's retained batches into a lazy Dask array.

        :func:`dask.array.from_delayed` references the batches without copying them.
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
    dims: Mapping[str, Sequence[str]] | None = None,
) -> _WriteStrategy:
    """Return the slice-writing strategy for a known streamed-axis size.

    Host memory when ``artifact_path`` is ``None``, a Zarr store otherwise.
    """
    if artifact_path is None:
        return _MemoryWriteStrategy(axis=axis)

    return _SliceWriteStrategy(
        artifact_path=artifact_path,
        total=total,
        batch_size=batch_size,
        axis=axis,
        dims=dims,
    )


def _select_write_strategy(
    artifact_path: Path | None,
    *,
    total: int | None,
    batch_size: int,
    dims: Mapping[str, Sequence[str]] | None = None,
) -> _WriteStrategy:
    """Return the observation-axis strategy for a regular loader or a generic stream.

    A generic stream, with ``total`` unknown, appends serially: its batch boundaries
    need not align with the Zarr chunks.
    """
    if artifact_path is None:
        return _MemoryWriteStrategy(axis=1)
    if total is None:
        return _AppendWriteStrategy(
            artifact_path=artifact_path,
            batch_size=batch_size,
            axis=1,
            dims=dims,
        )

    return _SliceWriteStrategy(
        artifact_path=artifact_path,
        total=total,
        batch_size=batch_size,
        axis=1,
        dims=dims,
    )


def _writer(
    queue: Queue,
    sink: Path | MutableMapping[str, list[np.ndarray]],
    error_queue: Queue,
    stop: Event,
    apply: Callable[[Array | list[np.ndarray], object], None],
) -> None:
    """Write queued ``(site, payload)`` items to the sink until a ``None`` sentinel.

    The workers of a pool share the queue, and each item names its site. On an error
    the worker puts ``(site, exc, traceback)`` on ``error_queue`` and sets ``stop``,
    after which every worker discards its items, still marking them done, so the
    producer never blocks and ``queue.join()`` completes.
    """
    group = None
    try:
        group = open_group(sink, mode="r+") if isinstance(sink, Path) else sink
    except Exception as exc:
        # `stop.set()` cannot fail, so the pool drains even if the report raises
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
    """Start a pool of writer threads over one queue.

    Returns:
        The threads, the work queue, the error queue, and the stop event.
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
        # Unwind the started workers, or they would block interpreter exit
        for _ in threads:
            queue.put(None)
        for thread in threads:
            thread.join()
        raise

    return threads, queue, error_queue, stop


def _shutdown_writer_threads(
    threads: list[Thread],
    queue: Queue | None,
    stop: Event | None,
    *,
    discard: bool,
) -> None:
    """Send one ``None`` sentinel per worker and wait for the pool to finish.

    The items ahead of the sentinels are written first, or discarded when ``discard``
    is set or the wait is interrupted.
    """
    if queue is None or stop is None:
        return
    sent = 0
    try:
        if discard:
            stop.set()
        for _ in threads:
            queue.put(None)
            sent += 1
        # The queue first: on Python 3.12 an interrupted `Thread.join()` marks a running
        # thread as stopped, so the joins below would return while writes are pending
        queue.join()
        for thread in threads:
            thread.join()
    except BaseException:
        stop.set()
        for _ in range(len(threads) - sent):
            queue.put(None)
        for thread in threads:
            thread.join()
        raise


def _discard_partial_output(sink: Path | MutableMapping[str, list[np.ndarray]]) -> None:
    """Remove a failed stream's partial output: the files, or the retained batches.

    The batches are emptied eagerly, since a held traceback can keep the strategy alive.
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
    n_items: int | None,
    strategy: _WriteStrategy,
    dispatch: Callable[[object], object],
    finalize: Callable[[object], dict[str, np.ndarray]],
    pbar: tqdm,
    num_writers: int | None = None,
) -> None:
    """Produce the site arrays of each item and write them through a writer pool.

    The items are produced in order through :func:`_iter_pipelined`; ``strategy``
    creates and enqueues their arrays, and a pool sized from ``num_writers``, or
    automatically, writes them. On any error the partial output is discarded and the
    error re-raised.
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
                    retained=isinstance(strategy, _MemoryWriteStrategy),
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
            pbar.set_postfix_str(
                "writing in progress..."
                if isinstance(strategy.sink, Path)
                else "collecting results...",
            )
        completed = True
    finally:
        _shutdown_writer_threads(threads, queue=queue, stop=stop, discard=not completed)
        pbar.set_postfix_str("")
        producer.close()
        with suppress(NameError):
            del sliced
        if worker_err is None and error_queue is not None and not error_queue.empty():
            worker_err = error_queue.get()
        # `stop` set without a report means a writer failed while reporting
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
