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

"""Tests for the automatic batch size of the streaming methods."""

import jax.numpy as jnp
import pytest
from jax import local_device_count

from aimz.utils.data._input_setup import MAX_BYTES, _resolve_batch_size


def test_resolve_batch_size_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The automatic batch size follows the pool target, memory cap, and floor."""
    # Pin the CPU count so the pool-occupancy target is deterministic
    monkeypatch.setattr("aimz.utils.data._input_setup.cpu_count", lambda: 4)

    # 1M observations x 200 draws (~800 MB float32): the pool target binds, with
    # pool = min(4 cpus, cap) = 4 and 4 * 4 = 16 batches
    pool_target = 62_500
    assert (
        _resolve_batch_size(None, axis_size=1_000_000, other_size=200, num_devices=1)
        == pool_target
    )
    # 10M observations x 1000 draws (~40 GB float32): the per-batch memory ceiling
    # binds before the pool target, at 25M float32 elements over 1000 draws
    memory_cap = 25_000
    assert (
        _resolve_batch_size(None, axis_size=10_000_000, other_size=1000, num_devices=1)
        == memory_cap
    )
    # 1000 observations x 1000 draws (~4 MB float32): below the floor, the axis stays
    # whole
    axis_size = 1000
    assert (
        _resolve_batch_size(None, axis_size=axis_size, other_size=1000, num_devices=1)
        == axis_size
    )
    # A memory cap below the device count is floored at the device count
    num_devices = local_device_count()
    max_elements = MAX_BYTES // jnp.result_type(float).itemsize
    assert (
        _resolve_batch_size(
            None, axis_size=10, other_size=max_elements, num_devices=num_devices
        )
        == num_devices
    )
