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

"""Tests for the `.sample_prior_predictive()` method."""

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import pytest
from jax import Array, random

from aimz import ImpactModel
from tests.conftest import _make_svi, latent_intervention_model, lm

if TYPE_CHECKING:
    from numpyro.infer import SVI


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_sample_prior_predictive_lm(
    synthetic_data: tuple[Array, Array],
    vi: "SVI",
) -> None:
    """Test the `.sample_prior_predictive()` method of ImpactModel."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    im.fit_on_batch(X, y)
    msg = (
        r"The `batch_size` \(\d+\) is not divisible by the number of devices \(\d+\)\."
    )
    with pytest.warns(UserWarning, match=msg):
        samples = im.sample_prior_predictive(
            X=X,
            batch_size=len(X) // 2,
            num_samples=99,
        )

    assert samples.prior_predictive["y"].values.shape == (1, 99, len(X))

    # Test with `return_sites`
    with pytest.warns(UserWarning, match=msg), TemporaryDirectory() as tmp_dir:
        assert im.sample_prior_predictive(
            X=X,
            num_samples=99,
            batch_size=len(X) // 2,
            return_sites="y",
            output_dir=tmp_dir,
        ).prior_predictive["y"].values.shape == (1, 99, len(X))

    im.cleanup()


@pytest.mark.parametrize("shard_axis", ["obs", "draw"])
def test_sample_prior_predictive_unfitted_both_shard_axes(
    synthetic_data: tuple[Array, Array],
    shard_axis: str,
) -> None:
    """An unfitted model works under both modes (spec built from a one-row probe).

    The probe only drives kernel-spec discovery (before the streamer dispatch), so it
    is independent of the sharding strategy.
    """
    X, _ = synthetic_data
    num_samples = 10
    im = ImpactModel(lm, rng_key=random.key(0), inference=_make_svi(lm))
    try:
        dt = im.sample_prior_predictive(
            X,
            num_samples=num_samples,
            batch_size=99,
            progress=False,
            shard_axis=shard_axis,
        )
        pp = dt["prior_predictive"]
        assert pp["y"].sizes["draw"] == num_samples
        assert pp["y"].shape[-1] == len(X)
    finally:
        im.cleanup()


@pytest.mark.parametrize(
    ("shard_axis", "use_loader"),
    [("obs", False), ("obs", True), ("draw", False)],
)
def test_sample_prior_predictive_intervention(
    *,
    shard_axis: str,
    use_loader: bool,
) -> None:
    """Interventions change downstream prior draws across sharding modes."""
    X = np.arange(24, dtype=np.float32).reshape(12, 2) / 24
    im = ImpactModel(lm, rng_key=random.key(0), inference=_make_svi(lm))
    draws = []
    for value in (0.0, 1.0):
        inputs = (
            ({"X": X[start : start + 6]} for start in range(0, len(X), 6))
            if use_loader
            else X
        )
        dt = im.sample_prior_predictive(
            inputs,
            intervention={"w": np.full(2, value), "b": 2 * value},
            rng_key=random.key(1),
            num_samples=9,
            batch_size=6,
            shard_axis=shard_axis,
            store="memory",
            progress=False,
        )
        draws.append(dt["prior_predictive"]["y"].values)

    expected = np.broadcast_to(X.sum(axis=1) + 2, (1, 9, len(X)))
    np.testing.assert_allclose(draws[1] - draws[0], expected, atol=1e-6)


def test_sample_prior_predictive_per_observation_intervention() -> None:
    """A per-observation intervention array is sliced for the probe and each batch."""
    X = np.arange(40, dtype=np.float32).reshape(20, 2) / 40
    im = ImpactModel(
        latent_intervention_model,
        rng_key=random.key(0),
        inference=_make_svi(latent_intervention_model),
    )
    intervention = {"z": np.linspace(-1.0, 1.0, 20), "w": 0.5}
    streamed = im.sample_prior_predictive(
        X,
        intervention=intervention,
        num_samples=9,
        batch_size=6,
        store="memory",
        progress=False,
    )
    whole = im.sample_prior_predictive_on_batch(
        X,
        intervention=intervention,
        num_samples=9,
    )

    np.testing.assert_array_equal(
        streamed["prior_predictive"]["mu"].values,
        whole["prior_predictive"]["mu"].values,
    )
