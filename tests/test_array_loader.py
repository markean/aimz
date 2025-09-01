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
from conftest import lm
from jax import random
from jax.typing import ArrayLike
from numpyro.infer import SVI

from aimz import ImpactModel
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

    def test_invalid_batch_size(self) -> None:
        """Invalid `batch_size` raises a ValueError."""
        with pytest.raises(
            ValueError,
            match="`batch_size` should be a positive integer",
        ):
            ArrayLoader(
                dataset=ArrayDataset(X=np.array([[1, 2, 3], [4, 5, 6]])),
                rng_key=random.key(42),
                batch_size=0.5,
            )

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

    @pytest.mark.parametrize("vi", [lm], indirect=True)
    def test_fit_dataloader_y_not_none_error(
        self,
        synthetic_data: tuple[ArrayLike, ArrayLike],
        vi: SVI,
    ) -> None:
        """Passing a data loader as `X` and a non-None `y` raises an error."""
        X, y = synthetic_data
        im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
        dataloader = ArrayLoader(ArrayDataset(X=X, y=y), rng_key=random.key(42))
        with pytest.raises(
            TypeError,
            match="must be `None` when `X` is already a data loader",
        ):
            im.fit(X=dataloader, y=y)
        im.fit(X=dataloader)

    @pytest.mark.parametrize("vi", [lm], indirect=True)
    def test_fit_consistency_with_array_and_dataloader(
        self,
        synthetic_data: tuple[ArrayLike, ArrayLike],
        vi: SVI,
    ) -> None:
        """Calling `.fit()` with arrays or a data loader can yield identical results."""
        X, y = synthetic_data

        # Initialize ImpactModel (passing rng_key here has no effect since we pass one
        # to `.fit()`).
        rng_key = random.key(42)
        rng_key, rng_subkey = random.split(rng_key)
        rng_key, _ = random.split(rng_subkey)
        im_without_dataloader = ImpactModel(lm, rng_key=random.key(0), inference=vi)
        im_without_dataloader.fit(
            X=X,
            y=y,
            rng_key=rng_subkey,
            batch_size=3,
            progress=False,
            shuffle=True,
        )
        rng_key, rng_subkey = random.split(rng_key)
        losses_without_dataloader = im_without_dataloader.vi_result.losses
        mean_pred_without_dataloader = (
            im_without_dataloader.predict(X=X, rng_key=rng_subkey, batch_size=3)
            .posterior_predictive["y"]
            .mean(["chain", "draw"])
            .values
        )

        # Prepare loader and new ImpactModel
        rng_key = random.key(42)
        rng_key, rng_subkey = random.split(rng_key)
        rng_key, rng_loader_key = random.split(rng_subkey)
        im_with_dataloader = ImpactModel(lm, rng_key=random.key(0), inference=vi)
        im_with_dataloader.fit(
            X=ArrayLoader(
                ArrayDataset(X=X, y=y),
                rng_key=rng_loader_key,
                batch_size=3,
                shuffle=True,
            ),
            rng_key=rng_subkey,
            progress=False,
        )
        rng_key, rng_subkey = random.split(rng_key)

        losses_with_dataloader = im_with_dataloader.vi_result.losses
        mean_pred_with_dataloader = (
            im_with_dataloader.predict(
                X=ArrayLoader(
                    ArrayDataset(X=X),
                    rng_key=rng_loader_key,
                    batch_size=3,
                ),
                rng_key=rng_subkey,
            )
            .posterior_predictive["y"]
            .mean(["chain", "draw"])
            .values
        )

        assert jnp.allclose(losses_without_dataloader, losses_with_dataloader), (
            "Losses from fitting with raw arrays vs. fitting with a data loader "
            "can match, if the rng_key is properly set."
        )
        assert jnp.allclose(mean_pred_without_dataloader, mean_pred_with_dataloader), (
            "Posterior predictive samples from fitting with raw arrays vs. "
            "fitting with a data loader can match, if the rng_key is properly set."
        )
