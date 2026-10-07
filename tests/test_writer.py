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

from aimz import PerformanceWarning
from aimz.utils._output import (
    _PIPELINE_DEPTH,
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


class _FinalizeError(RuntimeError):
    """Test error for a failure while collecting a result."""


class _WriteError(RuntimeError):
    """Test error for a deliberate writer failure."""


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


def _run_pool(
    strategy: _SliceWriteStrategy | _AppendWriteStrategy,
    batches: list[dict],
    num_writers: int,
) -> None:
    """Drive `_write_loop` with items that are already the finalized site dicts."""
    _write_loop(
        items=batches,
        n_items=len(batches),
        strategy=strategy,
        dispatch=lambda item: item,
        finalize=lambda pending: cast("dict[str, np.ndarray]", pending),
        pbar=MagicMock(),
        num_writers=num_writers,
    )


@pytest.fixture
def fake_psutil(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a fake `psutil` with a controllable `virtual_memory().available`."""
    fake = MagicMock()
    fake.virtual_memory.return_value.available = 10**12  # 1 TiB default
    monkeypatch.setattr("aimz.utils._output.psutil", fake)

    return fake


def test_plan_writers_policy(fake_psutil: MagicMock) -> None:
    """The pool follows the batch count, the memory left, and the queue ceiling."""
    item_nbytes = 1024
    # Few batches with plentiful memory get one writer per batch
    n_items = 3
    plan = _plan_writers(
        _WRITER_COUNT_UNBOUNDED,
        n_items=n_items,
        item_nbytes=item_nbytes,
        n_sites=1,
        requested=8,
    )
    assert plan.n_writers == n_items
    assert plan.queue_size >= 1
    # The absolute queue ceiling binds for huge workloads with tiny items
    plan = _plan_writers(
        _WRITER_COUNT_UNBOUNDED, n_items=10**6, item_nbytes=1, n_sites=1
    )
    assert plan.queue_size <= _QUEUE_SIZE_MAX
    # Room for three batches, two reserved by the pipeline: one queued, one applying
    fake_psutil.virtual_memory.return_value.available = 3 * item_nbytes
    plan = _plan_writers(
        _WRITER_COUNT_UNBOUNDED,
        n_items=100,
        item_nbytes=item_nbytes,
        n_sites=1,
        requested=8,
    )
    assert (plan.n_writers, plan.queue_size) == (1, 1)
    # Kept batches shrink the queue and warn when they will not fit in memory
    n_items, room = 10, 20
    fake_psutil.virtual_memory.return_value.available = room * item_nbytes
    kept = _plan_writers(
        1, n_items=n_items, item_nbytes=item_nbytes, n_sites=1, retained=True
    )
    written = _plan_writers(1, n_items=n_items, item_nbytes=item_nbytes, n_sites=1)
    assert kept.n_writers == written.n_writers == 1
    assert kept.queue_size == room - n_items - _PIPELINE_DEPTH - 1
    assert kept.queue_size < written.queue_size
    fake_psutil.virtual_memory.return_value.available = 3 * item_nbytes
    with pytest.warns(PerformanceWarning, match="exceeds the memory available"):
        plan = _plan_writers(
            1, n_items=n_items, item_nbytes=item_nbytes, n_sites=1, retained=True
        )
    assert (plan.n_writers, plan.queue_size) == (1, 1)


def test_write_loop_pipelines_in_order(tmp_path: Path) -> None:
    """Dispatch runs ahead of collection, results land in order, an error cleans up."""
    n_items, chunk = 4, 2
    log: list[tuple[str, int]] = []

    def run(artifact_path: Path, finalize_error_at: int | None = None) -> None:
        def dispatch(item: object) -> object:
            log.append(("dispatch", cast("int", item)))
            return item

        def finalize(pending: object) -> dict[str, np.ndarray]:
            log.append(("finalize", cast("int", pending)))
            if pending == finalize_error_at:
                raise _FinalizeError
            return {"y": np.full((chunk, 3), fill_value=pending, dtype=np.float32)}

        _write_loop(
            items=range(n_items),
            n_items=n_items,
            strategy=_SliceWriteStrategy(
                artifact_path=artifact_path,
                total=n_items * chunk,
                batch_size=chunk,
                axis=0,
            ),
            dispatch=dispatch,
            finalize=finalize,
            pbar=tqdm(disable=True),
        )

    run(tmp_path / "out")
    # Pipelining engaged: item 1 was dispatched before item 0 was collected
    assert log.index(("dispatch", 1)) < log.index(("finalize", 0))
    # Every item landed at its own offset, FIFO order and the tail drain included
    written = np.asarray(open_group(tmp_path / "out", mode="r")["y"])
    np.testing.assert_array_equal(written, _expected(n_items, chunk))

    with pytest.raises(_FinalizeError):
        run(tmp_path / "failed", finalize_error_at=1)
    assert not (tmp_path / "failed").exists()


def test_determine_writer_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The count follows the CPUs, the request, the strategy ceiling, and the items."""
    # The automatic count is `min(cpu_count, _WRITER_COUNT_MAX)`, bounded by the items
    monkeypatch.setattr("aimz.utils._output.cpu_count", lambda: 1000)
    assert (
        _determine_writer_count(max_writers=_WRITER_COUNT_UNBOUNDED, num_items=10_000)
        == _WRITER_COUNT_MAX
    )
    # An explicit request overrides the automatic cap, never above the item count
    requested, few_items = 6, 3
    assert (
        _determine_writer_count(
            max_writers=_WRITER_COUNT_UNBOUNDED, num_items=10, requested=requested
        )
        == requested
    )
    assert (
        _determine_writer_count(
            max_writers=_WRITER_COUNT_UNBOUNDED,
            num_items=few_items,
            requested=requested,
        )
        == few_items
    )
    # An order-sensitive strategy pins the pool to one worker, and the floor is one
    assert _determine_writer_count(max_writers=1, num_items=100, requested=8) == 1
    assert (
        _determine_writer_count(
            max_writers=_WRITER_COUNT_UNBOUNDED, num_items=100, requested=0
        )
        == 1
    )
    assert (
        _determine_writer_count(max_writers=_WRITER_COUNT_UNBOUNDED, num_items=0) == 1
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


def test_writer_pool_lands_every_batch(tmp_path: Path) -> None:
    """Every batch of every site lands at its own offset, under either strategy."""
    n_batches, chunk = 4, 2
    strategies = {
        "slice": _SliceWriteStrategy(
            artifact_path=tmp_path / "slice",
            total=n_batches * chunk,
            batch_size=chunk,
            axis=0,
        ),
        # A single writer keeps the order the growing array depends on
        "append": _AppendWriteStrategy(
            artifact_path=tmp_path / "append", batch_size=chunk, axis=0
        ),
    }
    for name, strategy in strategies.items():
        _run_pool(strategy, _batches(n_batches, chunk, ("y", "z")), num_writers=4)
        # Each batch landed exactly once at its own offset, whatever the write order
        group = open_group(tmp_path / name, mode="r")
        for site in ("y", "z"):
            np.testing.assert_array_equal(
                np.asarray(group[site]), _expected(n_batches, chunk)
            )


def test_writer_pool_failure_raises_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A write error propagates and removes the output, even when reporting it fails."""

    def run(
        name: str,
        num_writers: int,
        exc: type[Exception] = _WriteError,
        match: str | None = None,
    ) -> None:
        artifact_path = tmp_path / name
        strategy = _SliceWriteStrategy(
            artifact_path=artifact_path, total=8, batch_size=2, axis=0
        )
        with (
            patch.object(strategy, "apply", side_effect=_WriteError),
            pytest.raises(exc, match=match),
        ):
            _run_pool(strategy, _batches(4, 2, ("y", "z")), num_writers=num_writers)
        assert not artifact_path.exists()

    run("plain", num_writers=4)

    # A raising log call inside the error handler must not hang the stream
    def raising_exception(*args: object, **kwargs: object) -> None:
        msg = "logging failed"
        raise MemoryError(msg)

    monkeypatch.setattr("aimz.utils._output.logger.exception", raising_exception)
    run("logging", num_writers=1)

    # A writer that cannot report its error must still fail the write; the error queue
    # is the only unbounded queue, so `put` fails only there
    class _BrokenErrorQueue(Queue):
        def put(self, item: object, *args: object, **kwargs: object) -> None:
            if self.maxsize == 0:
                raise MemoryError
            super().put(item, *args, **kwargs)

    monkeypatch.setattr("aimz.utils._output.Queue", _BrokenErrorQueue)
    run(
        "unreported",
        num_writers=1,
        exc=RuntimeError,
        match="without reporting an error",
    )


def test_partial_pool_startup_unwinds_started_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A mid-pool `Thread.start()` failure joins the workers already started."""
    started = 0
    fail_at = 2

    # Patching `threading.Thread.start` globally would break Zarr's own threads
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
            sink=store,
            apply=lambda _array, _item: None,
            n_writers=3,
            queue_size=4,
        )

    # Nothing from the aborted pool is left alive
    assert not any(
        thread.name.endswith("(_writer)") and thread.is_alive()
        for thread in threading.enumerate()
    )
