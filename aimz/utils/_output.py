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
from os import cpu_count
from queue import Queue
from shutil import rmtree
from threading import Event, Thread
from typing import TYPE_CHECKING, Protocol, cast

try:
    import psutil
except ImportError:
    psutil: ModuleType | None = None

from zarr import open_group

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from pathlib import Path
    from types import ModuleType

    import numpy as np
    from tqdm.auto import tqdm
    from zarr import Array, Group

    from aimz.utils.data import ArrayLoader


# Maximum in-flight compute steps in `_write_loop`. Depth 2 dispatches the next step
# before collecting the previous one, overlapping device compute with device-to-host
# transfer and the host-side write work; each in-flight step holds one output chunk on
# device, so raising this raises peak device memory proportionally.
_PIPELINE_DEPTH = 2
_QUEUE_SIZE_MAX = 128
_QUEUE_SIZE_FALLBACK_CAP = 4
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
) -> Iterator[dict[str, np.ndarray]]:
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
    for item in items:
        pending.append(dispatch(item))
        if len(pending) >= _PIPELINE_DEPTH:
            yield finalize(pending.popleft())
    while pending:
        yield finalize(pending.popleft())


def _determine_writer_queue_size(num_items: int, item_nbytes: int) -> int:
    """Determine the writer queue size from workload and host-memory bounds.

    Args:
        num_items: Total number of items the producer will emit.
        item_nbytes: Bytes the producer commits to the queue(s) per iteration.

    Returns:
        The writer queue size.
    """
    if psutil is None:
        queue_size = min(cpu_count() or 1, _QUEUE_SIZE_FALLBACK_CAP)
    else:
        queue_size = psutil.virtual_memory().available // item_nbytes

    return max(1, min(_QUEUE_SIZE_MAX, num_items, queue_size))


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
        dtype="float32" if arr.dtype == "bfloat16" else arr.dtype,
        chunks=tuple(chunks),
        dimension_names=(
            "draw",
            *tuple(f"{site}_dim_{i}" for i in range(arr.ndim - 1)),
        ),
    )


def _writer(
    queue: Queue,
    group_path: Path,
    error_queue: Queue,
    stop: Event,
    apply: Callable[[Array, object], None],
) -> None:
    """Background worker that writes queued ``(site, payload)`` items to Zarr.

    One of a shared pool of interchangeable workers that all consume the same queue, so
    a worker is not bound to a single site: each item names the site to write. Runs in a
    loop, retrieving items and writing each into its site's array via ``apply``, exiting
    when a ``None`` sentinel is received. Concurrency safety rests on the strategy: the
    pool is only sized above one for order-independent, disjoint-write strategies (see
    :meth:`_WriteStrategy.max_writers`).

    If opening the group or a write fails, the error is logged, its details are put into
    ``error_queue``, and the shared ``stop`` event is set so every worker switches to
    drain mode — subsequent items are discarded (still marked done, so the bounded
    producer cannot block and ``queue.join()`` can finish) rather than written into a
    store that is being torn down.

    Args:
        queue: The shared queue of ``(site, payload)`` items (and ``None`` sentinels).
        group_path: The path of the Zarr group.
        error_queue: The queue to collect errors raised by the writer threads, each as a
            ``(site, exc, traceback)`` tuple (``site`` is ``None`` for an open failure).
        stop: Shared event; set on the first error to put the pool into drain mode.
        apply: Writes one queued payload into a site's Zarr array; the only behavior
            that differs between write strategies.
    """
    group = None
    try:
        group = open_group(group_path, mode="r+")
    except Exception as exc:
        # `stop.set()` first — it cannot fail, so the pool always enters drain mode
        # even if reporting or logging below raises (e.g. under memory pressure); the
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
                apply(cast("Array", cast("Group", group)[site]), payload)
            except Exception as exc:
                stop.set()
                with suppress(Exception):
                    error_queue.put((site, exc, exc.__traceback__))
                    logger.exception("Error writing to site '%s'", site)
        finally:
            queue.task_done()


def _start_writer_threads(
    group_path: Path,
    apply: Callable[[Array, object], None],
    n_writers: int,
    queue_size: int,
) -> tuple[list[Thread], Queue, Queue, Event]:
    """Start a shared pool of writer threads consuming one queue.

    Args:
        group_path: The path to the Zarr group where data will be written.
        apply: Writes one queued payload into a site's Zarr array.
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
                args=(queue, group_path, error_queue, stop),
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


def _select_write_strategy(
    zarr_group: Group,
    dataloader: ArrayLoader,
) -> _WriteStrategy:
    """Build the data-parallel write strategy for a data loader.

    Slice writing needs the batch count and dataset size up front; if either is
    unavailable the append strategy is used instead. Both stream the observation
    (axis-1) dimension.

    Args:
        zarr_group: The open Zarr group to create site arrays in.
        dataloader: The data loader the sample loop will iterate.

    Returns:
        The write strategy to use.
    """
    try:
        len(dataloader)
        n_obs = len(dataloader.dataset)
    except (TypeError, AttributeError):
        return _AppendWriteStrategy(
            zarr_group=zarr_group,
            batch_size=dataloader.batch_size,
            axis=1,
        )

    return _SliceWriteStrategy(
        zarr_group=zarr_group,
        total=n_obs,
        batch_size=dataloader.batch_size,
        axis=1,
    )


class _WriteStrategy(Protocol):
    """How a batch of per-site samples is created and queued for writing.

    A strategy streams one **axis** of each site's Zarr array, filling it batch by
    batch while every other axis is written whole: ``axis=0`` streams the draw axis
    (draw-parallel), ``axis=1`` streams the observation axis (data-parallel).
    """

    @property
    def max_writers(self) -> int:
        """Maximum number of concurrent writer threads this strategy tolerates.

        ``1`` means the strategy is order-sensitive and must be written by a single
        consumer; a larger value means writes are order-independent and may be spread
        across a pool of workers.
        """

    def apply(self, array: Array, item: object) -> None:
        """Write one queued item into a site's Zarr array.

        Args:
            array: The site's Zarr array.
            item: The queued payload to write.
        """

    def create_arrays(self, site_arrays: Mapping[str, np.ndarray]) -> None:
        """Create any not-yet-created Zarr arrays for the sites in a batch.

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

        Each payload is enqueued as a ``(site, payload)`` item so any worker in the pool
        can route it to the right Zarr array.

        Args:
            queue: The shared writer queue.
            site_arrays: Mapping of site name to the (post-slice) sample array
                emitted for the current batch.
        """


class _AppendWriteStrategy(_WriteStrategy):
    """Grow each site's Zarr array by appending batches along the streamed axis.

    Requires no size information up front; the streamed-axis size emerges from the
    batches as they arrive. ``array.append`` mutates the array's length metadata and is
    order-sensitive, so this strategy is **not** concurrency-safe: it must be written by
    a single consumer (:attr:`max_writers` is ``1``). A single shared worker preserves
    per-site append order via FIFO consumption; the cost is that cross-site appends,
    which the previous one-thread-per-site model ran concurrently, are now serialized.
    This path is only selected for a data loader without a knowable length.
    """

    def __init__(
        self,
        *,
        zarr_group: Group,
        batch_size: int,
        axis: int,
    ) -> None:
        """Initialize the append write strategy.

        Args:
            zarr_group: The open Zarr group to create site arrays in.
            batch_size: Chunk length along the streamed axis.
            axis: The streamed axis to grow (``0`` for draws, ``1`` for observations).
        """
        self._zarr_group = zarr_group
        self._chunk_size = batch_size
        self._axis = axis
        self._seen: set[str] = set()

    @property
    def max_writers(self) -> int:
        """A single writer: appends are order-sensitive and mutate array metadata."""
        return 1

    def apply(self, array: Array, item: object) -> None:
        """Append the queued batch along the streamed axis.

        Args:
            array: The site's Zarr array.
            item: The batch array to append.
        """
        array.append(cast("np.ndarray", item), axis=self._axis)

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

    def enqueue(
        self,
        queue: Queue,
        site_arrays: Mapping[str, np.ndarray],
    ) -> None:
        """Put each site's batch array onto the shared writer queue.

        Args:
            queue: The shared writer queue.
            site_arrays: Mapping of site name to the batch array to append.
        """
        for site, arr in site_arrays.items():
            queue.put((site, arr))


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
        zarr_group: Group,
        total: int,
        batch_size: int,
        axis: int,
    ) -> None:
        """Initialize the slice write strategy.

        Args:
            zarr_group: The open Zarr group to create site arrays in.
            total: Full size of the streamed axis (preallocated up front).
            batch_size: Chunk length along the streamed axis.
            axis: The streamed axis to grow (``0`` for draws, ``1`` for observations).
        """
        self._zarr_group = zarr_group
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

    def apply(self, array: Array, item: object) -> None:
        """Assign the queued ``(start, arr)`` batch into a fixed slice of the axis.

        Args:
            array: The site's (preallocated) Zarr array.
            item: A ``(start, arr)`` tuple; ``arr`` is written to the streamed-axis
                slice ``[start : start + arr.shape[axis])``.
        """
        start, arr = cast("tuple[int, np.ndarray]", item)
        idx: list = [slice(None)] * arr.ndim
        idx[self._axis] = slice(start, start + arr.shape[self._axis])
        array[tuple(idx)] = arr

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
                if arr.shape[self._axis] != self._chunk_size:
                    requirement = (
                        "the input batch size"
                        if self._axis == 1
                        else "the draw chunk size"
                    )
                    msg = (
                        f"Slice writing requires each site's axis-{self._axis} size "
                        f"to match {requirement}. Site {site!r} emitted shape "
                        f"{arr.shape} for a batch of size {self._chunk_size}; this "
                        "kernel is not currently supported under slice writing."
                    )
                    raise NotImplementedError(msg)
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


def _write_loop(
    items: Iterable,
    n_items: int,
    artifact_path: Path,
    strategy: _WriteStrategy,
    dispatch: Callable[[object], object],
    finalize: Callable[[object], dict[str, np.ndarray]],
    pbar: tqdm,
    num_writers: int | None = None,
) -> None:
    """Produce per-item site arrays and write them concurrently to disk.

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
        artifact_path: Call-specific path where the Zarr group is written.
        strategy: Write strategy that creates and enqueues each item's site arrays.
        dispatch: Launches one item's computation and returns its in-flight handle.
        finalize: Blocks on an in-flight handle and returns its mapping of site name
            to array.
        pbar: Progress bar instance to display progress.
        num_writers: Explicit writer-thread pool size, or ``None`` (the production
            path) to choose automatically. Bounded by the strategy's ceiling and the
            item count; overriding is intended for tests.

    Raises:
        Exception: Any exception raised during production or writing is logged, the
            artifacts at ``artifact_path`` are removed, and the exception is re-raised.
    """
    threads: list[Thread] = []
    queue: Queue | None = None
    error_queue: Queue | None = None
    stop: Event | None = None
    worker_err: tuple | None = None
    completed = False
    success = False
    try:
        for sliced in _iter_pipelined(items, dispatch=dispatch, finalize=finalize):
            strategy.create_arrays(sliced)
            if queue is None:
                # One batch's worth of bytes fits `batch_budget` batches under the
                # host-memory/CPU bounds. The pipelined producer holds up to
                # `_PIPELINE_DEPTH` further batches in flight, so reserve those from
                # the budget first. The shared queue counts individual
                # `(site, payload)` items, so scale by the per-batch item count and
                # reserve `n_writers` slots for items being applied — keeping total
                # in-flight bytes within the single-writer envelope.
                batch_budget = _determine_writer_queue_size(
                    n_items,
                    item_nbytes=max(
                        1,
                        sum(int(arr.nbytes) for arr in sliced.values()),
                    ),
                )
                items_per_batch = max(1, len(sliced))
                slots = max(1, batch_budget - _PIPELINE_DEPTH) * items_per_batch
                n_writers = min(
                    _determine_writer_count(
                        strategy.max_writers,
                        n_items,
                        requested=num_writers,
                    ),
                    max(1, slots - 1),
                )
                queue_size = max(1, slots - n_writers)
                threads, queue, error_queue, stop = _start_writer_threads(
                    group_path=artifact_path,
                    apply=strategy.apply,
                    n_writers=n_writers,
                    queue_size=queue_size,
                )
            strategy.enqueue(queue, site_arrays=sliced)
            if stop is not None and stop.is_set():
                if not cast("Queue", error_queue).empty():
                    worker_err = cast("Queue", error_queue).get()
                break
            pbar.update()
        if worker_err is None:
            pbar.set_description("Writing in progress...")
        completed = True
    finally:
        _shutdown_writer_threads(threads, queue=queue)
        if worker_err is None and error_queue is not None and not error_queue.empty():
            worker_err = error_queue.get()
        # `stop` set without a reported error means a writer failed while reporting
        # (e.g. under memory pressure); treat it as a failure, never as a clean run.
        success = (
            completed and worker_err is None and (stop is None or not stop.is_set())
        )
        if not success:
            rmtree(artifact_path, ignore_errors=True)
            logger.warning("Cleaned up artifact path: %s", artifact_path)
        pbar.close()
    if worker_err is not None:
        _, exc, tb = worker_err
        raise exc.with_traceback(tb)
    if not success:
        msg = "A background writer thread failed without reporting an error."
        raise RuntimeError(msg)
