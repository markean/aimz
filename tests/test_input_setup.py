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

"""Tests for :func:`~aimz.utils.data._input_setup._setup_inputs`."""

import jax.numpy as jnp
import pytest
from jax import local_device_count, make_mesh, random
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from aimz.utils.data._input_setup import (
    MAX_BYTES,
    _resolve_batch_size,
    _setup_inputs,
)


def test_batch_size_capped_when_exceeding_threshold() -> None:
    """Auto batch size is capped and aligned to num_devices."""
    num_devices = local_device_count()
    mesh = make_mesh(
        (num_devices,),
        axis_names=("obs",),
        axis_types=(AxisType.Explicit,),
    )
    device = NamedSharding(mesh, spec=PartitionSpec("obs"))

    n = 10
    # Mirror the implementation's dtype-aware element budget.
    max_elements = MAX_BYTES // jnp.result_type(float).itemsize
    num_samples = max_elements
    X = jnp.ones((n, 2))

    loader, _ = _setup_inputs(
        X=X,
        y=None,
        param_input="X",
        param_output="y",
        rng_key=random.key(0),
        batch_size=None,
        num_samples=num_samples,
        device=device,
    )

    # batch_size = max_elements // num_samples = max_elements // max_elements = 1.
    # Round down to nearest multiple of num_devices: 1 - 1 % num_devices = 0.
    # Floor at num_devices to avoid zero: max(0, num_devices) = num_devices.
    assert loader.batch_size == num_devices


def test_x_zero_dim_raises() -> None:
    """A 0-dimensional ``X`` raises ``ValueError``."""
    with pytest.raises(ValueError, match=r"`X` must have at least 1 dimension."):
        _setup_inputs(
            X=jnp.array(1.0),
            y=None,
            param_input="X",
            param_output="y",
            rng_key=random.key(0),
            batch_size=None,
            num_samples=1,
        )


def test_y_zero_dim_raises() -> None:
    """A 0-dimensional ``y`` raises ``ValueError``."""
    with pytest.raises(ValueError, match=r"`y` must have at least 1 dimension."):
        _setup_inputs(
            X=jnp.ones((4, 2)),
            y=jnp.array(1.0),
            param_input="X",
            param_output="y",
            rng_key=random.key(0),
            batch_size=None,
            num_samples=1,
        )


def test_x_wrong_type_raises() -> None:
    """A non-array, non-loader ``X`` (e.g. a Python list) raises ``TypeError``."""
    with pytest.raises(TypeError, match=r"`X` must be an array-like or a data loader"):
        _setup_inputs(
            X=[[1.0, 2.0], [3.0, 4.0]],
            y=None,
            param_input="X",
            param_output="y",
            rng_key=random.key(0),
            batch_size=None,
            num_samples=1,
        )


def test_non_array_kwargs_classified_as_extra() -> None:
    """Non-array keyword arguments are routed to extras, not the batched dataset."""
    loader, extra = _setup_inputs(
        X=jnp.ones((4, 2)),
        y=None,
        param_input="X",
        param_output="y",
        rng_key=random.key(0),
        batch_size=2,
        num_samples=1,
        family="gaussian",
    )

    assert set(loader.dataset.arrays) == {"X"}
    assert extra == {"family": "gaussian"}


class TestResolveBatchSize:
    """Test class for the automatic policy of :func:`_resolve_batch_size`."""

    @pytest.fixture(autouse=True)
    def four_cpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin the CPU count so the pool-occupancy target is deterministic."""
        monkeypatch.setattr("aimz.utils.data._input_setup.cpu_count", lambda: 4)

    def test_pool_target_splits_large_axis(self) -> None:
        """A large output is split into enough batches to occupy the writer pool."""
        # 1M observations x 200 draws (~800 MB float32): the pool target binds.
        # pool = min(4 cpus, cap) = 4; target batches = 4 * 4 = 16.
        batch = _resolve_batch_size(
            None,
            axis_size=1_000_000,
            other_size=200,
            num_devices=1,
        )
        expected = 62_500  # ceil(1_000_000 / 16)
        assert batch == expected

    def test_memory_cap_binds_for_huge_outputs(self) -> None:
        """The per-batch memory ceiling binds before the pool target for huge data."""
        # 10M observations x 1000 draws (~40 GB float32): cap = 25M elements / 1000.
        batch = _resolve_batch_size(
            None,
            axis_size=10_000_000,
            other_size=1000,
            num_devices=1,
        )
        expected = 25_000  # 25M float32 elements // 1000 draws
        assert batch == expected

    def test_floor_keeps_tiny_outputs_whole(self) -> None:
        """Outputs below the per-batch byte floor are not split at all."""
        # 1000 observations x 1000 draws (~4 MB float32): floor exceeds the axis.
        axis_size = 1000
        batch = _resolve_batch_size(
            None,
            axis_size=axis_size,
            other_size=1000,
            num_devices=1,
        )
        assert batch == axis_size
