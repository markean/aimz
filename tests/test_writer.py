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
from threading import Event, Lock, Thread
from unittest.mock import MagicMock

import numpy as np
import pytest
from tqdm.auto import tqdm
from zarr import open_group

from aimz.utils import _output as output_mod
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


class _FakeGroup:
    """A stand-in for an open Zarr group whose ``group[site]`` returns the site name.

    This makes the ``array`` argument that ``_writer`` passes into ``strategy.apply``
    equal to the site name, so a recording strategy can attribute each applied payload
    to its site without a real Zarr store.
    """

    def __getitem__(self, site: str) -> str:
        return site


def _fake_open_group(*_args: object, **_kwargs: object) -> _FakeGroup:
    """Stand in for :func:`zarr.open_group`, returning a fake group."""
    return _FakeGroup()


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

    def test_full_pool_with_ample_memory(self, fake_psutil: MagicMock) -> None:
        """With ample memory and many batches, the automatic pool is CPU-derived."""
        del fake_psutil
        plan = _plan_writers(
            _WRITER_COUNT_UNBOUNDED,
            n_items=100,
            item_nbytes=1024,
            n_sites=1,
        )
        assert plan.n_writers == _determine_writer_count(
            _WRITER_COUNT_UNBOUNDED,
            num_items=100,
        )
        assert 1 <= plan.queue_size <= _QUEUE_SIZE_MAX

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

    def test_strategy_ceiling_pins_single_writer(self, fake_psutil: MagicMock) -> None:
        """An order-sensitive strategy keeps a single consumer regardless of request."""
        del fake_psutil
        plan = _plan_writers(1, n_items=100, item_nbytes=1024, n_sites=2, requested=8)
        assert plan.n_writers == 1

    def test_zero_size_items_do_not_crash(self, fake_psutil: MagicMock) -> None:
        """Zero-byte batches (empty sites) still produce a bounded, sane plan."""
        del fake_psutil
        plan = _plan_writers(
            _WRITER_COUNT_UNBOUNDED,
            n_items=4,
            item_nbytes=0,
            n_sites=1,
        )
        assert plan.n_writers >= 1
        assert plan.queue_size >= 1


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
            log.append(("dispatch", item))
            return item

        def finalize(pending: object) -> dict[str, np.ndarray]:
            log.append(("finalize", pending))
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
        written = open_group(artifact_path, mode="r")["y"][:]
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


class TestStrategyMaxWriters:
    """The write strategies advertise their concurrency tolerance correctly."""

    def test_slice_strategy_is_concurrency_safe(self) -> None:
        """Slice writing is position-addressed, so it tolerates a pool of writers."""
        strategy = _SliceWriteStrategy(
            zarr_group=MagicMock(),
            total=10,
            batch_size=2,
            axis=0,
        )
        assert strategy.max_writers > 1

    def test_append_strategy_is_single_consumer(self) -> None:
        """Append writing is order-sensitive, so it must stay single-consumer."""
        strategy = _AppendWriteStrategy(zarr_group=MagicMock(), batch_size=2, axis=1)
        assert strategy.max_writers == 1


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


def _slice_batches(n_batches: int, chunk: int, sites: tuple[str, ...]) -> list[dict]:
    """Build fake per-site batch dicts whose arrays have a real shape and nbytes."""
    return [
        {site: np.zeros((chunk, 3), dtype=np.float32) for site in sites}
        for _ in range(n_batches)
    ]


class _RecordingSlice(_SliceWriteStrategy):
    """A slice strategy that records applied `(site, payload)` items thread-safely.

    Inherits the real position-addressed `enqueue`/`create_arrays` bookkeeping (its
    ``zarr_group`` is a mock), but records each applied payload instead of touching a
    Zarr array, so the writer pool can be exercised without a real store.
    """

    def __init__(
        self,
        *,
        total: int,
        batch_size: int,
        axis: int,
        max_writers: int,
    ) -> None:
        super().__init__(
            zarr_group=MagicMock(),
            total=total,
            batch_size=batch_size,
            axis=axis,
        )
        self._max_writers = max_writers
        self._lock = Lock()
        self.applied: list[tuple[str, object]] = []

    @property
    def max_writers(self) -> int:
        return self._max_writers

    def apply(self, array: object, item: object) -> None:
        with self._lock:
            # `array` is the site name (see `_FakeGroup`); `item` is `(start, arr)`.
            self.applied.append((array, item))  # type: ignore[arg-type]


def test_pool_writes_all_items_with_multiple_writers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A pool of writers applies every batch of every site exactly once."""
    monkeypatch.setattr("aimz.utils._output.open_group", _fake_open_group)
    n_batches, chunk = 4, 2
    sites = ("y", "z")
    strategy = _RecordingSlice(
        total=n_batches * chunk,
        batch_size=chunk,
        axis=0,
        max_writers=4,
    )
    batches = _slice_batches(n_batches=n_batches, chunk=chunk, sites=sites)

    _write_loop(
        items=batches,
        n_items=len(batches),
        artifact_path=tmp_path,
        strategy=strategy,
        dispatch=lambda item: item,
        finalize=lambda pending: pending,
        pbar=MagicMock(),
        num_writers=4,
    )

    # Every batch of every site is applied exactly once, order-independent.
    assert len(strategy.applied) == n_batches * len(sites)
    applied_sites = [site for site, _ in strategy.applied]
    for site in sites:
        assert applied_sites.count(site) == n_batches
    # Every chunk-aligned slice offset appears exactly once per site.
    expected_starts = list(range(0, n_batches * chunk, chunk))
    starts_y = sorted(start for site, (start, _) in strategy.applied if site == "y")
    assert starts_y == expected_starts


def test_append_strategy_pinned_to_single_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The append strategy forces a single consumer even when more are requested."""
    monkeypatch.setattr("aimz.utils._output.open_group", _fake_open_group)

    applied: list[int] = []

    class _RecordingAppend(_AppendWriteStrategy):
        def apply(self, array: object, item: object) -> None:
            del array  # unused: `item` is the appended array
            applied.append(int(np.asarray(item).flat[0]))

    strategy = _RecordingAppend(zarr_group=MagicMock(), batch_size=2, axis=1)
    n_batches = 5
    batches = [
        {"y": np.full((3, 2), idx, dtype=np.float32)} for idx in range(n_batches)
    ]

    _write_loop(
        items=batches,
        n_items=len(batches),
        artifact_path=tmp_path,
        strategy=strategy,
        dispatch=lambda item: item,
        finalize=lambda pending: pending,
        pbar=MagicMock(),
        num_writers=8,  # requested high, but max_writers=1 pins to one worker
    )

    # A single consumer preserves per-site FIFO order.
    assert applied == list(range(n_batches))


def test_pool_error_propagates_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A write error in the pool is re-raised with its traceback and cleans up."""
    monkeypatch.setattr("aimz.utils._output.open_group", _fake_open_group)

    class BoomError(RuntimeError):
        """Test exception raised inside a writer thread."""

    class _BoomStrategy(_RecordingSlice):
        def apply(self, array: object, item: object) -> None:
            # `array` is the site name; fail on one site to trigger the error path.
            if array == "z":
                raise BoomError
            super().apply(array, item)

    artifact_path = tmp_path / "out"
    artifact_path.mkdir()
    n_batches, chunk = 4, 2
    strategy = _BoomStrategy(
        total=n_batches * chunk,
        batch_size=chunk,
        axis=0,
        max_writers=4,
    )
    batches = _slice_batches(n_batches=n_batches, chunk=chunk, sites=("y", "z"))

    with pytest.raises(BoomError):
        _write_loop(
            items=batches,
            n_items=len(batches),
            artifact_path=artifact_path,
            strategy=strategy,
            dispatch=lambda item: item,
            finalize=lambda pending: pending,
            pbar=MagicMock(),
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

    class _FlakyThread(Thread):
        def start(self) -> None:
            nonlocal started
            started += 1
            if started == fail_at:
                msg = "can't start new thread"
                raise RuntimeError(msg)
            super().start()

    monkeypatch.setattr("aimz.utils._output.Thread", _FlakyThread)
    monkeypatch.setattr("aimz.utils._output.open_group", _fake_open_group)

    with pytest.raises(RuntimeError, match="can't start new thread"):
        _start_writer_threads(
            group_path=tmp_path,
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
    monkeypatch.setattr("aimz.utils._output.open_group", _fake_open_group)

    def raising_exception(*args: object, **kwargs: object) -> None:
        msg = "logging failed"
        raise MemoryError(msg)

    monkeypatch.setattr("aimz.utils._output.logger.exception", raising_exception)

    class BoomError(RuntimeError):
        """Test exception raised inside a writer thread."""

    class _BoomStrategy(_RecordingSlice):
        def apply(self, _array: object, _item: object) -> None:
            raise BoomError

    artifact_path = tmp_path / "out"
    artifact_path.mkdir()
    strategy = _BoomStrategy(total=8, batch_size=2, axis=0, max_writers=1)
    batches = _slice_batches(n_batches=4, chunk=2, sites=("y",))

    with pytest.raises(BoomError):
        _write_loop(
            items=batches,
            n_items=len(batches),
            artifact_path=artifact_path,
            strategy=strategy,
            dispatch=lambda item: item,
            finalize=lambda pending: pending,
            pbar=MagicMock(),
            num_writers=1,
        )

    assert not artifact_path.exists()


def test_pool_size_clamped_by_memory_budget(
    fake_psutil: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A tight memory budget clamps the pool instead of breaking the envelope.

    With host memory for only one batch beyond the pipeline's reservation, the pool
    must not start more writers than the in-flight budget allows: one writer, one
    queued slot.
    """
    # Each batch is a (2, 3) float32 = 24 bytes; room for three batches, two of which
    # the pipelined producer reserves.
    fake_psutil.virtual_memory.return_value.available = 3 * 24
    monkeypatch.setattr("aimz.utils._output.open_group", _fake_open_group)

    captured: dict[str, int] = {}
    real_start = output_mod._start_writer_threads

    def capturing_start(*args: object, **kwargs: object) -> object:
        captured.update(
            n_writers=kwargs["n_writers"],  # type: ignore[dict-item]
            queue_size=kwargs["queue_size"],  # type: ignore[dict-item]
        )
        return real_start(*args, **kwargs)

    monkeypatch.setattr(
        "aimz.utils._output._start_writer_threads",
        capturing_start,
    )

    n_batches = 4
    strategy = _RecordingSlice(total=2 * n_batches, batch_size=2, axis=0, max_writers=8)
    batches = _slice_batches(n_batches=n_batches, chunk=2, sites=("y",))

    _write_loop(
        items=batches,
        n_items=len(batches),
        artifact_path=tmp_path,
        strategy=strategy,
        dispatch=lambda item: item,
        finalize=lambda pending: pending,
        pbar=MagicMock(),
        num_writers=8,
    )

    assert captured == {"n_writers": 1, "queue_size": 1}
    assert len(strategy.applied) == n_batches
