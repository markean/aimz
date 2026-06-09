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

"""Tests for the input-setup utilities in :mod:`aimz.utils.data._input_setup`."""

import jax.numpy as jnp
import pytest
from jax import local_device_count, make_mesh, random
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from aimz.utils.data._input_setup import (
    MAX_ELEMENTS,
    _resolve_draw_batch_size,
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
    num_samples = MAX_ELEMENTS
    X = jnp.ones((n, 2))

    loader, _ = _setup_inputs(
        X=X,
        y=None,
        rng_key=random.key(0),
        batch_size=None,
        num_samples=num_samples,
        device=device,
    )

    # batch_size = MAX_ELEMENTS // num_samples = MAX_ELEMENTS // MAX_ELEMENTS = 1.
    # Round down to nearest multiple of num_devices: 1 - 1 % num_devices = 0.
    # Floor at num_devices to avoid zero: max(0, num_devices) = num_devices.
    assert loader.batch_size == num_devices


def test_resolve_draw_batch_size() -> None:
    """Draw chunking honors explicit sizes, the memory budget, and device floors."""
    num_devices = 3

    # An explicit batch size is used as given.
    explicit = 7
    assert (
        _resolve_draw_batch_size(
            explicit,
            num_samples=1000,
            n_obs=10,
            num_devices=num_devices,
        )
        == explicit
    )

    # The whole posterior is one chunk when it fits the memory budget.
    num_samples = 1000
    assert (
        _resolve_draw_batch_size(
            None,
            num_samples=num_samples,
            n_obs=10,
            num_devices=num_devices,
        )
        == num_samples
    )

    # Above the budget the chunk is capped to MAX_ELEMENTS // n_obs, rounded down to
    # a multiple of num_devices: (MAX_ELEMENTS // 50_000 = 500) -> 498.
    n_obs = 50_000
    capped = MAX_ELEMENTS // n_obs
    expected = capped - capped % num_devices
    assert (
        _resolve_draw_batch_size(
            None,
            num_samples=10_000,
            n_obs=n_obs,
            num_devices=num_devices,
        )
        == expected
    )

    # The cap is floored at num_devices and never exceeds num_samples.
    assert (
        _resolve_draw_batch_size(
            None,
            num_samples=10,
            n_obs=MAX_ELEMENTS,
            num_devices=num_devices,
        )
        == num_devices
    )


def test_x_zero_dim_raises() -> None:
    """A 0-dimensional ``X`` raises ``ValueError``."""
    with pytest.raises(ValueError, match=r"`X` must have at least 1 dimension."):
        _setup_inputs(
            X=jnp.array(1.0),
            y=None,
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
            rng_key=random.key(0),
            batch_size=None,
            num_samples=1,
        )
