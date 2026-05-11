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

"""Tests for writer-thread queue sizing in :mod:`aimz.utils._output`."""

from unittest.mock import MagicMock

import pytest

from aimz.utils._output import (
    _QUEUE_SIZE_FALLBACK_CAP,
    _QUEUE_SIZE_MAX,
    _determine_writer_queue_size,
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
