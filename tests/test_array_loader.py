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
from jax import random

from aimz.utils.data import ArrayDataset, ArrayLoader


class TestArrayDataset:
    """Tests class to ensure correct initialization and behavior."""
    def test_empty_array(self) -> None:
        """Initializing with no arrays raises a ValueError."""
        with pytest.raises(ValueError, match="At least one array must be provided."):
            ArrayDataset()

    def test_same_lengths(self) -> None:
        """All arrays must have the same length; otherwise, raise a ValueError."""
        X = jnp.array([[1, 2, 3], [4, 5, 6]])
        y = jnp.array([1, 2, 3])
        with pytest.raises(ValueError, match="All arrays must have the same length."):
            ArrayDataset(X=X, y=y)

    def test_no_jax_conversion(self) -> None:
        """Check that arrays remain NumPy arrays when `to_jax=False` is specified."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])
        dataset = ArrayDataset(X=X, y=y, to_jax=False)
        actual = next(iter(dataset))
        desired = {"X": np.array([1, 2, 3]), "y": np.array(1)}
        assert actual.keys() == desired.keys()
        for k in actual:
            np.testing.assert_array_equal(actual[k], desired[k])


class TestArrayLoader:
    """Tests class to ensure compatibility and correct handling."""
    def test_legacy_prng_key(self) -> None:
        """A legacy uint32 PRNGKey raises a UserWarning."""
        y = jnp.array([1, 2, 3])
        dataset = ArrayDataset(y=y)
        with pytest.warns(
            UserWarning,
            match="Legacy `uint32` PRNGKey detected; converting to a typed key array.",
        ):
            ArrayLoader(dataset, rng_key=random.PRNGKey(42))

    def test_array_loader(self) -> None:
        """Padding along unsupported axis in a 1D array raises a ValueError."""
        y = jnp.array([1, 2, 3])
        dataset = ArrayDataset(y=y)
        loader = ArrayLoader(dataset, rng_key=random.key(42))
        with pytest.raises(
            ValueError,
            match="Padding 1D arrays is only supported along axis 0.",
        ):
            loader.pad_array(y, n_pad=1, axis=1)
