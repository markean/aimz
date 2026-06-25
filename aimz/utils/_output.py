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
from os import cpu_count
from queue import Queue
from shutil import rmtree
from threading import Thread
from typing import TYPE_CHECKING, Protocol, cast

try:
    import psutil
except ImportError:
    psutil: ModuleType | None = None

from zarr import open_group

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path
    from types import ModuleType

    import numpy as np
    from tqdm.auto import tqdm
    from zarr import Array, Group

    from aimz.utils.data import ArrayLoader


_QUEUE_SIZE_MAX = 128
_QUEUE_SIZE_FALLBACK_CAP = 4


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
    site: str,
    queue: Queue,
    group_path: Path,
    error_queue: Queue,
    apply: Callable[[Array, object], None],
) -> None:
    """Background worker that writes queued batches to a Zarr array.

    Runs in a loop, retrieving items from the queue and writing each into the site's
    dataset via ``apply``. It exits when a ``None`` sentinel is received. If opening the
    group or a write fails, the error is logged, its details are put into
    ``error_queue``, and the queue is drained (including the sentinel) so
    :func:`_shutdown_writer_threads`'s ``queue.join()`` can finish.

    Args:
        site: The name of the sample site (also the Zarr array name).
        queue: The queue to retrieve queued batches from.
        group_path: The path of the Zarr group.
        error_queue: The queue to collect errors raised by the writer thread.
        apply: Writes one queued item into the site's Zarr array; the only behavior that
            differs between write strategies.
    """
    try:
        group = open_group(group_path, mode="r+")
    except Exception as exc:
        logger.exception("Error opening output group for site '%s'", site)
        error_queue.put((site, exc, exc.__traceback__))
        while True:
            leftover = queue.get()
            queue.task_done()
            if leftover is None:
                break
        return

    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break

        try:
            apply(cast("Array", group[site]), item)
        except Exception as exc:
            logger.exception("Error writing to site '%s'", site)
            error_queue.put((site, exc, exc.__traceback__))
            # Drain remaining items including sentinel so queue.join() can finish
            while True:
                leftover = queue.get()
                queue.task_done()
                if leftover is None:
                    break
            break
        finally:
            queue.task_done()


def _start_writer_threads(
    sites: tuple[str, ...],
    group_path: Path,
    apply: Callable[[Array, object], None],
    queue_size: int,
) -> tuple[list[Thread], dict[str, Queue], Queue]:
    """Start writer threads and their corresponding queues for each site.

    Args:
        sites: Names of the return sites.
        group_path: The path to the Zarr group where data will be written.
        apply: Writes one queued item into a site's Zarr array.
        queue_size: Maximum size of each queue (per site).

    Returns:
        A tuple containing a list of threads and a dictionary mapping each site to its
        corresponding queue.
    """
    queues: dict[str, Queue] = {site: Queue(queue_size) for site in sites}
    threads = []
    error_queue: Queue = Queue()
    for site, queue in queues.items():
        thread = Thread(
            target=_writer,
            args=(site, queue, group_path, error_queue),
            kwargs={"apply": apply},
        )
        thread.start()
        threads.append(thread)

    return threads, queues, error_queue


def _shutdown_writer_threads(
    threads: list[Thread],
    queues: dict[str, Queue],
) -> None:
    """Signal writer threads to stop and wait for their completion.

    Args:
        threads: List of writer threads to join.
        queues: Mapping of site names to their respective queues.
    """
    for queue, thread in zip(queues.values(), threads, strict=True):
        if thread.is_alive():
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
        queues: dict[str, Queue],
        site_arrays: Mapping[str, np.ndarray],
    ) -> None:
        """Put a batch's per-site arrays onto their writer queues.

        Args:
            queues: Per-site writer queues, keyed by site name.
            site_arrays: Mapping of site name to the (post-slice) sample array
                emitted for the current batch.
        """


class _AppendWriteStrategy(_WriteStrategy):
    """Grow each site's Zarr array by appending batches along the streamed axis.

    Requires no size information up front; the streamed-axis size emerges from the
    batches as they arrive.
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
        queues: dict[str, Queue],
        site_arrays: Mapping[str, np.ndarray],
    ) -> None:
        """Put each site's batch array onto its writer queue.

        Args:
            queues: Per-site writer queues, keyed by site name.
            site_arrays: Mapping of site name to the batch array to append.
        """
        for site, arr in site_arrays.items():
            queues[site].put(arr)


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
        queues: dict[str, Queue],
        site_arrays: Mapping[str, np.ndarray],
    ) -> None:
        """Enqueue each site's batch as ``(start, arr)`` and advance its offset.

        Args:
            queues: Per-site writer queues, keyed by site name.
            site_arrays: Mapping of site name to the batch array to write.
        """
        for site, arr in site_arrays.items():
            start = self._site_offsets[site]
            queues[site].put((start, arr))
            self._site_offsets[site] = start + arr.shape[self._axis]


def _write_loop(
    items: Iterable,
    n_items: int,
    return_sites: tuple[str, ...],
    output_dir: Path,
    strategy: _WriteStrategy,
    produce: Callable[[object], dict[str, np.ndarray]],
    pbar: tqdm,
) -> None:
    """Produce per-item site arrays and write them concurrently to disk.

    Shared by the data- and draw-parallel write paths. For each item from ``items``,
    ``produce`` returns the (post-slice) per-site arrays; array creation/enqueuing is
    delegated to ``strategy`` and writing to background threads.

    Args:
        items: Items to iterate (batches paired with keys, or draw-chunk starts).
        n_items: Number of items (used to size the writer queues).
        return_sites: Names of variables (sites) to write.
        output_dir: Directory where outputs will be saved.
        strategy: Write strategy that creates and enqueues each item's site arrays.
        produce: Maps one item to its mapping of site name to array.
        pbar: Progress bar instance to display progress.

    Raises:
        Exception: Any exception raised during production or writing is logged, the
            output directory is cleaned up, and the exception is re-raised.
    """
    threads = []
    queues = {}
    error_queue = None
    worker_err: tuple | None = None
    success = False
    try:
        for item in items:
            sliced = produce(item)
            strategy.create_arrays(sliced)
            if error_queue is None:
                threads, queues, error_queue = _start_writer_threads(
                    return_sites,
                    group_path=output_dir,
                    apply=strategy.apply,
                    queue_size=_determine_writer_queue_size(
                        n_items,
                        item_nbytes=max(
                            1,
                            sum(int(arr.nbytes) for arr in sliced.values()),
                        ),
                    ),
                )
            strategy.enqueue(queues, site_arrays=sliced)
            if not error_queue.empty():
                worker_err = error_queue.get()
                break
            pbar.update()
        if worker_err is None:
            pbar.set_description("Writing in progress...")
            _shutdown_writer_threads(threads, queues=queues)
            success = True
    finally:
        if not success:
            logger.warning("Cleaning up output directory: %s", output_dir)
            _shutdown_writer_threads(threads, queues=queues)
            rmtree(output_dir, ignore_errors=True)
        pbar.close()
    if worker_err is not None:
        _, exc, tb = worker_err
        raise exc.with_traceback(tb)
