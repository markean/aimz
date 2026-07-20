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

"""Tests for the write loop and writer queues in :mod:`aimz.utils._output`."""

from pathlib import Path
from queue import Queue
from threading import Thread
from unittest.mock import MagicMock

import numpy as np
import pytest
from tqdm.auto import tqdm
from zarr import open_group

from aimz.utils._output import (
    _QUEUE_SIZE_FALLBACK_CAP,
    _QUEUE_SIZE_MAX,
    _determine_writer_queue_size,
    _SliceWriteStrategy,
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


class TestDetermineWriterQueueSize:
    """Test class for :func:`_determine_writer_queue_size`."""

    def test_queue_size_without_psutil(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fallback path: bounded by the CPU-derived cap when psutil is missing."""
        monkeypatch.setattr("aimz.utils._output.psutil", None)

        assert (
            1
            <= _determine_writer_queue_size(num_items=100, item_nbytes=1)
            <= _QUEUE_SIZE_FALLBACK_CAP
        )

    def test_queue_size_with_psutil(self, fake_psutil: MagicMock) -> None:
        """Memory-aware path: result respects the absolute cap on a normal host."""
        # Silence unused-argument warning; we just need the fixture to ensure the
        # memory-aware branch runs
        del fake_psutil
        assert (
            1
            <= _determine_writer_queue_size(num_items=100, item_nbytes=1024)
            <= _QUEUE_SIZE_MAX
        )

    def test_workload_cap_binds(self, fake_psutil: MagicMock) -> None:
        """`num_items` caps the queue size when smaller than memory/CPU bounds."""
        del fake_psutil
        assert _determine_writer_queue_size(num_items=1, item_nbytes=1024) == 1

    def test_absolute_cap_binds(self, fake_psutil: MagicMock) -> None:
        """The absolute ceiling binds for huge workloads with tiny items."""
        del fake_psutil
        assert (
            _determine_writer_queue_size(num_items=10**6, item_nbytes=1)
            == _QUEUE_SIZE_MAX
        )

    def test_floor_binds(self, fake_psutil: MagicMock) -> None:
        """The floor at 1 binds so `Queue(maxsize)` is always bounded."""
        # Tight available memory + huge item drives `available // item_nbytes` to zero;
        # The function's floor must keep `Queue(maxsize)` bounded.
        fake_psutil.virtual_memory.return_value.available = 1
        assert _determine_writer_queue_size(num_items=100, item_nbytes=10**18) == 1


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
            return_sites=("y",),
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
    queue = Queue(maxsize=1)
    error_queue = Queue()
    thread = Thread(
        target=_writer,
        args=("site", queue, tmp_path, error_queue),
        kwargs={"apply": lambda _array, _item: None},
    )

    thread.start()
    queue.put(object())
    queue.put(None)
    queue.join()
    thread.join(timeout=1)

    assert not thread.is_alive()
    site, exc, tb = error_queue.get_nowait()
    assert site == "site"
    assert isinstance(exc, StoreOpenError)
    assert tb is not None
