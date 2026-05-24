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
from threading import Thread
from typing import TYPE_CHECKING, Protocol, cast

try:
    import psutil
except ImportError:
    psutil: ModuleType | None = None

from zarr import open_group

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path
    from types import ModuleType

    import numpy as np
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
    *,
    draws: int,
    sample_axis_size: int,
    chunk_axis_size: int,
) -> None:
    """Create one Zarr array for a site.

    The draw axis is sized to ``draws``; the sample (axis-1) dimension is sized to
    ``sample_axis_size`` (``0`` for append-style growth, the full size for preallocated
    slice writing); any trailing data dimensions are preserved verbatim.
    ``dimension_names`` is ``("draw", "<site>_dim_0", ...)`` with one name per array
    dimension.

    Args:
        zarr_group: The open Zarr group to create the array in.
        site: The sample site name (also the array name).
        arr: A representative (post-slice) sample array; only ``shape[2:]``, ``ndim``,
            and ``dtype`` are read.
        draws: Size of the leading (draw) dimension.
        sample_axis_size: Size of the sample (axis-1) dimension.
        chunk_axis_size: Chunk length along the sample (axis-1) dimension.
    """
    zarr_group.create_array(
        name=site,
        shape=(draws, sample_axis_size, *arr.shape[2:]),
        dtype="float32" if arr.dtype == "bfloat16" else arr.dtype,
        chunks=(draws, chunk_axis_size, *arr.shape[2:]),
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
    *,
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
    for queue in queues.values():
        queue.put(None)
    for queue in queues.values():
        queue.join()
    for thread in threads:
        thread.join()


def _select_write_strategy(dataloader: ArrayLoader) -> type[_WriteStrategy]:
    """Select a write strategy from the loader's size capability.

    Slice writing needs the batch count and dataset size up front; if either is
    unavailable the append strategy is used instead.

    Args:
        dataloader: The data loader the sample loop will iterate.

    Returns:
        The write-strategy class to instantiate.
    """
    try:
        len(dataloader)
        len(dataloader.dataset)
    except (TypeError, AttributeError):
        return _AppendWriteStrategy

    return _SliceWriteStrategy


class _WriteStrategy(Protocol):
    """How a batch of per-site samples is created and queued for writing."""

    @staticmethod
    def apply(array: Array, item: object) -> None:
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
    """Grow each site's Zarr array by appending batches along axis 1.

    Requires no size information up front; the array shape emerges from the batches as
    they arrive.
    """

    @staticmethod
    def apply(array: Array, item: object) -> None:
        """Append the queued batch along the sample (axis-1) dimension.

        Args:
            array: The site's Zarr array.
            item: The batch array to append.
        """
        array.append(cast("np.ndarray", item), axis=1)

    def __init__(
        self,
        *,
        zarr_group: Group,
        draws: int,
        dataloader: ArrayLoader,
    ) -> None:
        """Initialize the append write strategy.

        Args:
            zarr_group: The open Zarr group to create site arrays in.
            draws: Size of the leading (draw) dimension for every site array.
            dataloader: The data loader the sample loop will iterate; its ``batch_size``
                is used as the Zarr chunk length along the sample (axis-1) dimension.
        """
        self._zarr_group = zarr_group
        self._draws = draws
        self._chunk_size = dataloader.batch_size
        self._seen: set[str] = set()

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
                    site,
                    arr,
                    draws=self._draws,
                    sample_axis_size=0,
                    chunk_axis_size=self._chunk_size,
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
    """Write each batch into a fixed slice of a preallocated Zarr array.

    Requires the dataset size up front. Every return site must emit an axis-1 size equal
    to the input batch size (the loader's configured batch size, less any sharding
    padding). Sites whose axis-1 size doesn't match raise :exc:`NotImplementedError` on
    the first batch.
    """

    @staticmethod
    def apply(array: Array, item: object) -> None:
        """Assign the queued ``(start, arr)`` batch into a fixed sample slice.

        Args:
            array: The site's (preallocated) Zarr array.
            item: A ``(start, arr)`` tuple; ``arr`` is written to
                ``array[:, start : start + arr.shape[1], ...]``.
        """
        start, arr = cast("tuple[int, np.ndarray]", item)
        array[:, start : start + arr.shape[1], ...] = arr

    def __init__(
        self,
        *,
        zarr_group: Group,
        draws: int,
        dataloader: ArrayLoader,
    ) -> None:
        """Initialize the slice write strategy.

        Args:
            zarr_group: The open Zarr group to create site arrays in.
            draws: Size of the leading (draw) dimension for every site array.
            dataloader: The data loader the sample loop will iterate; its ``batch_size``
                and ``len(dataloader.dataset)`` are used to preallocate each site's
                sample (axis-1) dimension.
        """
        self._zarr_group = zarr_group
        self._draws = draws
        self._chunk_size = dataloader.batch_size
        self._n_obs = len(dataloader.dataset)
        self._seen: set[str] = set()
        self._site_offsets: dict[str, int] = {}

    def create_arrays(self, site_arrays: Mapping[str, np.ndarray]) -> None:
        """Preallocate full-size Zarr arrays for sites not yet seen.

        On the first call (the first batch), every site is verified to emit an axis-1
        size equal to the input batch size; mismatches raise. After creation, each site
        is registered in ``self._site_offsets`` with offset zero so subsequent batches
        can be written to fixed slices.

        Args:
            site_arrays: Mapping of site name to the (post-slice) sample array emitted
                for the current batch.

        Raises:
            NotImplementedError: If any return site emits an axis-1 size that does not
                match the input batch size.
        """
        # The first call carries the first batch's sites; on full first batches the real
        # batch size equals ``self._chunk_size``, and when the dataset is smaller than
        # the configured batch the single batch's real size is ``self._n_obs``.
        expected_size = min(self._chunk_size, self._n_obs)
        for site, arr in site_arrays.items():
            if site not in self._seen:
                if arr.shape[1] != expected_size:
                    msg = (
                        "Slice writing requires the per-site axis-1 size to match the "
                        f"input batch size. Site {site!r} emitted shape {arr.shape} "
                        f"for a batch of size {expected_size}; this kernel is not "
                        "currently supported under slice writing."
                    )
                    raise NotImplementedError(msg)
                _create_site_array(
                    self._zarr_group,
                    site,
                    arr,
                    draws=self._draws,
                    sample_axis_size=self._n_obs,
                    chunk_axis_size=expected_size,
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
            self._site_offsets[site] = start + arr.shape[1]
