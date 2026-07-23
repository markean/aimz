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

"""Tests for the pipelined write loop and writer-thread pool in `aimz.utils._output`."""

import threading
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from tqdm.auto import tqdm
from zarr import open_group

from aimz.utils._output import (
    _QUEUE_SIZE_MAX,
    _WRITER_COUNT_MAX,
    _WRITER_COUNT_UNBOUNDED,
    _AppendWriteStrategy,
    _determine_writer_count,
    _plan_writers,
    _SliceWriteStrategy,
    _start_writer_threads,
    _write_loop,
    _writer,
)


@pytest.fixture
def fake_psutil(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a fake `psutil` with a controllable `virtual_memory().available`.

    Decouples the tests from whether the real `psutil` is installed, so the memory-aware
    branch runs deterministically in any environment.
    """
    fake = MagicMock()
    fake.virtual_memory.return_value.available = 10**12  # 1 TiB default
    monkeypatch.setattr("aimz.utils._output.psutil", fake)

    return fake


class TestPlanWriters:
    """Test class for the writer-pool planner :func:`_plan_writers`."""

    def test_small_item_count_gets_full_parallelism(
        self,
        fake_psutil: MagicMock,
    ) -> None:
        """Few batches with plentiful memory still get one writer per batch.

        The pipeline reservation applies to the memory bound only; it must not
        throttle a small batch count when memory is not the binding constraint.
        """
        del fake_psutil
        n_items = 3
        plan = _plan_writers(
            _WRITER_COUNT_UNBOUNDED,
            n_items=n_items,
            item_nbytes=1024,
            n_sites=1,
            requested=8,
        )
        assert plan.n_writers == n_items
        assert plan.queue_size >= 1

    def test_tight_memory_clamps_pool(self, fake_psutil: MagicMock) -> None:
        """A tight envelope collapses to the floor: one queued, one applying."""
        item_nbytes = 1024
        # Room for three batches; the pipeline reserves two, leaving one slot.
        fake_psutil.virtual_memory.return_value.available = 3 * item_nbytes
        plan = _plan_writers(
            _WRITER_COUNT_UNBOUNDED,
            n_items=100,
            item_nbytes=item_nbytes,
            n_sites=1,
            requested=8,
        )
        assert plan.n_writers == 1
        assert plan.queue_size == 1

    def test_absolute_queue_cap_binds(self, fake_psutil: MagicMock) -> None:
        """The absolute queue ceiling binds for huge workloads with tiny items."""
        del fake_psutil
        plan = _plan_writers(
            _WRITER_COUNT_UNBOUNDED,
            n_items=10**6,
            item_nbytes=1,
            n_sites=1,
        )
        assert plan.queue_size <= _QUEUE_SIZE_MAX


class TestWriteLoop:
    """Test class for the pipelined :func:`_write_loop`."""

    N_ITEMS = 4
    CHUNK = 2
    N_COLS = 5

    def _run(
        self,
        artifact_path: Path,
        log: list[tuple[str, int]],
        finalize_error_at: int | None = None,
    ) -> None:
        """Drive `_write_loop` over draw-style items with recording fakes."""

        class FinalizeError(RuntimeError):
            """Test exception for a failure while collecting a result."""

        def dispatch(item: object) -> object:
            log.append(("dispatch", cast("int", item)))
            return item

        def finalize(pending: object) -> dict[str, np.ndarray]:
            log.append(("finalize", cast("int", pending)))
            if pending == finalize_error_at:
                raise FinalizeError
            return {
                "y": np.full(
                    (self.CHUNK, self.N_COLS),
                    fill_value=pending,
                    dtype=np.float32,
                ),
            }

        _write_loop(
            items=range(self.N_ITEMS),
            n_items=self.N_ITEMS,
            artifact_path=artifact_path,
            strategy=_SliceWriteStrategy(
                zarr_group=open_group(artifact_path, mode="w"),
                total=self.N_ITEMS * self.CHUNK,
                batch_size=self.CHUNK,
                axis=0,
            ),
            dispatch=dispatch,
            finalize=finalize,
            pbar=tqdm(disable=True),
        )

    def test_dispatch_runs_ahead_and_order_is_preserved(self, tmp_path: Path) -> None:
        """Dispatch runs ahead of collection while results stay in item order."""
        artifact_path = tmp_path / "out"
        log: list[tuple[str, int]] = []

        self._run(artifact_path, log=log)

        # Pipelining engaged: item 1 was dispatched before item 0 was collected.
        assert log.index(("dispatch", 1)) < log.index(("finalize", 0))
        # Every item landed at its own offset (FIFO order, including the tail drain).
        written = np.asarray(open_group(artifact_path, mode="r")["y"])
        expected = np.repeat(
            np.arange(self.N_ITEMS, dtype=np.float32),
            self.CHUNK,
        )[:, None] * np.ones(self.N_COLS, dtype=np.float32)
        np.testing.assert_array_equal(written, expected)

    def test_finalize_failure_cleans_output_and_raises(self, tmp_path: Path) -> None:
        """An error while collecting a result propagates and removes the output."""
        artifact_path = tmp_path / "out"

        with pytest.raises(RuntimeError):
            self._run(artifact_path, log=[], finalize_error_at=1)

        assert not artifact_path.exists()


class TestDetermineWriterCount:
    """Test class for :func:`_determine_writer_count`."""

    def test_auto_uses_cpu_capped_by_ceiling(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The automatic count is `min(cpu_count, _WRITER_COUNT_MAX)`, item-bounded."""
        monkeypatch.setattr("aimz.utils._output.cpu_count", lambda: 1000)
        # A generous strategy ceiling and many items: the CPU ceiling binds.
        assert (
            _determine_writer_count(
                max_writers=_WRITER_COUNT_UNBOUNDED,
                num_items=10_000,
            )
            == _WRITER_COUNT_MAX
        )

    def test_explicit_request_honored_up_to_item_count(self) -> None:
        """An explicit request overrides the auto cap, bounded by the item count."""
        requested = 6
        # Plenty of items: the request is honored.
        assert (
            _determine_writer_count(
                max_writers=_WRITER_COUNT_UNBOUNDED,
                num_items=10,
                requested=requested,
            )
            == requested
        )
        # Fewer items than requested: never more writers than there are items.
        few_items = 3
        assert (
            _determine_writer_count(
                max_writers=_WRITER_COUNT_UNBOUNDED,
                num_items=few_items,
                requested=requested,
            )
            == few_items
        )

    def test_strategy_ceiling_binds(self) -> None:
        """An order-sensitive strategy (`max_writers=1`) pins the pool to one worker."""
        assert _determine_writer_count(max_writers=1, num_items=100, requested=8) == 1
        assert _determine_writer_count(max_writers=1, num_items=100) == 1

    def test_floor_binds(self) -> None:
        """The result never drops below 1, even for a zero/negative request."""
        assert (
            _determine_writer_count(
                max_writers=_WRITER_COUNT_UNBOUNDED,
                num_items=100,
                requested=0,
            )
            == 1
        )
        assert (
            _determine_writer_count(max_writers=_WRITER_COUNT_UNBOUNDED, num_items=0)
            == 1
        )


def test_writer_reports_open_group_failure_and_drains_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Writer startup errors are surfaced without leaving queued items stuck."""

    class StoreOpenError(OSError):
        """Test exception for a store-open failure."""

    def fail_open_group(*args: object, **kwargs: object) -> None:
        """Raise an error that mimics a filesystem or store-open failure."""
        raise StoreOpenError

    monkeypatch.setattr("aimz.utils._output.open_group", fail_open_group)
    queue: Queue = Queue(maxsize=2)
    error_queue: Queue = Queue()
    stop = Event()
    thread = Thread(
        target=_writer,
        args=(queue, tmp_path, error_queue, stop),
        kwargs={"apply": lambda _array, _item: None},
    )

    thread.start()
    queue.put(("site", object()))  # a real (site, payload) item
    queue.put(None)
    queue.join()
    thread.join(timeout=1)

    assert not thread.is_alive()
    # The open failure is a pool-level error, not tied to a site (site is None).
    site, exc, tb = error_queue.get_nowait()
    assert site is None
    assert isinstance(exc, StoreOpenError)
    assert tb is not None
    # The shared stop event is set so the whole pool switches to drain mode.
    assert stop.is_set()


def _batches(n_batches: int, chunk: int, sites: tuple[str, ...]) -> list[dict]:
    """Per-site batch dicts; batch ``k`` is filled with ``k`` to verify placement."""
    return [
        {site: np.full((chunk, 3), k, dtype=np.float32) for site in sites}
        for k in range(n_batches)
    ]


def _expected(n_batches: int, chunk: int) -> np.ndarray:
    """The array `_batches` produces once every batch sits at its own offset."""
    values = np.repeat(np.arange(n_batches, dtype=np.float32), chunk)
    return values[:, None] * np.ones(3, dtype=np.float32)


def _passthrough_finalize(pending: object) -> dict[str, np.ndarray]:
    """Identity finalize for tests whose dispatched item is already the site dict."""
    return cast("dict[str, np.ndarray]", pending)


def _run_pool(
    artifact_path: Path,
    strategy: _SliceWriteStrategy | _AppendWriteStrategy,
    batches: list[dict],
    num_writers: int,
) -> None:
    """Drive `_write_loop` with items that are already the finalized site dicts."""
    _write_loop(
        items=batches,
        n_items=len(batches),
        artifact_path=artifact_path,
        strategy=strategy,
        dispatch=lambda item: item,
        finalize=_passthrough_finalize,
        pbar=MagicMock(),
        num_writers=num_writers,
    )


class _WriteError(RuntimeError):
    """Test error for a deliberate writer failure."""


def test_pool_writes_all_items_with_multiple_writers(tmp_path: Path) -> None:
    """A pool of writers lands every batch of every site at its own offset."""
    store = tmp_path / "store"
    n_batches, chunk = 4, 2
    strategy = _SliceWriteStrategy(
        zarr_group=open_group(store, mode="w"),
        total=n_batches * chunk,
        batch_size=chunk,
        axis=0,
    )

    _run_pool(store, strategy, _batches(n_batches, chunk, ("y", "z")), num_writers=4)

    # Content equality proves each batch was written exactly once at its own offset,
    # regardless of the order in which the pool completed the writes.
    group = open_group(store, mode="r")
    for site in ("y", "z"):
        np.testing.assert_array_equal(
            np.asarray(group[site]),
            _expected(n_batches, chunk),
        )


def test_append_strategy_pinned_to_single_writer(tmp_path: Path) -> None:
    """The append strategy stays a single consumer even when more are requested."""
    store = tmp_path / "store"
    n_batches, chunk = 5, 2
    strategy = _AppendWriteStrategy(
        zarr_group=open_group(store, mode="w"),
        batch_size=chunk,
        axis=0,
    )

    # Requested high, but `max_writers=1` pins the pool to one worker, whose FIFO
    # consumption preserves the batch order the growing array depends on.
    _run_pool(store, strategy, _batches(n_batches, chunk, ("y",)), num_writers=8)

    np.testing.assert_array_equal(
        np.asarray(open_group(store, mode="r")["y"]),
        _expected(n_batches, chunk),
    )


def test_pool_error_propagates_and_cleans_up(tmp_path: Path) -> None:
    """A write error in the pool is re-raised with its traceback and cleans up."""
    artifact_path = tmp_path / "out"
    n_batches, chunk = 4, 2
    strategy = _SliceWriteStrategy(
        zarr_group=open_group(artifact_path, mode="w"),
        total=n_batches * chunk,
        batch_size=chunk,
        axis=0,
    )

    with (
        patch.object(strategy, "apply", side_effect=_WriteError),
        pytest.raises(_WriteError),
    ):
        _run_pool(
            artifact_path,
            strategy,
            _batches(n_batches, chunk, ("y", "z")),
            num_writers=4,
        )

    # On failure the artifact path is removed.
    assert not artifact_path.exists()


def test_partial_pool_startup_unwinds_started_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A mid-pool `Thread.start()` failure joins already-started workers.

    Without the unwind, the started non-daemon workers would block forever on a queue
    the caller never receives, hanging interpreter exit.
    """
    started = 0
    fail_at = 2

    # A targeted override of the module's `Thread` attribute; patching
    # `threading.Thread.start` globally would break Zarr's own worker threads.
    class _FlakyThread(Thread):
        def start(self) -> None:
            nonlocal started
            started += 1
            if started == fail_at:
                msg = "can't start new thread"
                raise RuntimeError(msg)
            super().start()

    monkeypatch.setattr("aimz.utils._output.Thread", _FlakyThread)
    store = tmp_path / "store"
    open_group(store, mode="w")

    with pytest.raises(RuntimeError, match="can't start new thread"):
        _start_writer_threads(
            group_path=store,
            apply=lambda _array, _item: None,
            n_writers=3,
            queue_size=4,
        )

    # The first (successfully started) worker was sentineled and joined; nothing from
    # the aborted pool is left alive.
    assert not any(
        thread.name.endswith("(_writer)") and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_pool_survives_failing_error_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A raising log call inside the writer's error handler cannot hang the stream.

    The error is enqueued before logging and both are guarded, so the original
    exception still propagates and the producer never deadlocks on a dead consumer.
    """

    def raising_exception(*args: object, **kwargs: object) -> None:
        msg = "logging failed"
        raise MemoryError(msg)

    monkeypatch.setattr("aimz.utils._output.logger.exception", raising_exception)
    artifact_path = tmp_path / "out"
    strategy = _SliceWriteStrategy(
        zarr_group=open_group(artifact_path, mode="w"),
        total=8,
        batch_size=2,
        axis=0,
    )

    with (
        patch.object(strategy, "apply", side_effect=_WriteError),
        pytest.raises(_WriteError),
    ):
        _run_pool(artifact_path, strategy, _batches(4, 2, ("y",)), num_writers=1)

    assert not artifact_path.exists()


def test_unreported_writer_failure_still_fails_the_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A writer that cannot even report its error must not yield a clean run.

    If `error_queue.put` itself fails (e.g. under memory pressure), the stop event is
    the only surviving failure signal; the write must raise and clean up rather than
    return partial output as success.
    """

    # A targeted override of the module's `Queue` attribute: the error queue is the
    # only unbounded queue the writer creates, so failing `put` on `maxsize == 0`
    # breaks error reporting while the bounded work queue keeps flowing.
    class _BrokenErrorQueue(Queue):
        def put(
            self,
            item: object,
            block: bool = True,  # noqa: FBT001, FBT002 -- stdlib signature
            timeout: float | None = None,
        ) -> None:
            if self.maxsize == 0:
                raise MemoryError
            super().put(item, block, timeout)

    monkeypatch.setattr("aimz.utils._output.Queue", _BrokenErrorQueue)
    artifact_path = tmp_path / "out"
    strategy = _SliceWriteStrategy(
        zarr_group=open_group(artifact_path, mode="w"),
        total=8,
        batch_size=2,
        axis=0,
    )

    with (
        patch.object(strategy, "apply", side_effect=_WriteError),
        pytest.raises(RuntimeError, match="without reporting an error"),
    ):
        _run_pool(artifact_path, strategy, _batches(4, 2, ("y",)), num_writers=1)

    assert not artifact_path.exists()
