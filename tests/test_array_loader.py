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

"""Tests for the `ArrayDataset` and `ArrayLoader` classes."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, random

from aimz import AimzWarning
from aimz.utils.data import ArrayDataset, ArrayLoader


def test_array_dataset() -> None:
    """A dataset needs aligned arrays and yields NumPy rows, or JAX rows on request."""
    with pytest.raises(ValueError, match=r"At least one array must be provided."):
        ArrayDataset()
    with pytest.raises(
        ValueError,
        match=r"All arrays must have the same leading-axis size.",
    ):
        ArrayDataset(X=np.ones((2, 3)), y=np.ones(3))

    X, y = np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 2])
    row = next(iter(ArrayDataset(X=X, y=y)))
    assert row.keys() == {"X", "y"}
    assert isinstance(row["X"], np.ndarray)
    np.testing.assert_array_equal(row["X"], X[0])
    np.testing.assert_array_equal(row["y"], y[0])
    row = next(iter(ArrayDataset(X=X, y=y, to_jax=True)))
    assert isinstance(row["X"], Array)
    assert jnp.array_equal(row["X"], X[0])
    assert jnp.array_equal(row["y"], y[0])


def test_array_loader() -> None:
    """A loader checks its batch size, converts a legacy key, and shuffles by key."""
    y = np.arange(100)
    with pytest.raises(ValueError, match="`batch_size` should be a positive integer"):
        ArrayLoader(ArrayDataset(y=y), rng_key=random.key(42), batch_size=0.5)
    with pytest.warns(AimzWarning, match="Legacy `uint32` PRNGKey detected"):
        ArrayLoader(ArrayDataset(y=y), rng_key=random.PRNGKey(42))

    # Same key, same batches; each epoch is a fresh permutation of every row
    loaders = [
        ArrayLoader(
            ArrayDataset(y=y), rng_key=random.key(42), batch_size=7, shuffle=True
        )
        for _ in range(2)
    ]
    epochs = [np.concatenate([batch["y"] for batch in loader]) for loader in loaders]
    np.testing.assert_array_equal(epochs[0], epochs[1])
    second_epoch = np.concatenate([batch["y"] for batch in loaders[0]])
    np.testing.assert_array_equal(np.sort(second_epoch), y)
    assert not np.array_equal(second_epoch, epochs[0])
