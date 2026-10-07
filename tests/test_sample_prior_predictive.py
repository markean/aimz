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

from pathlib import Path

import numpy as np
import pytest
from jax import Array, random

from aimz import ImpactModel
from tests.conftest import latent_intervention_model, lm, make_svi


def test_sample_prior_predictive_unfitted(
    synthetic_data: tuple[Array, Array],
    tmp_path: Path,
) -> None:
    """An unfitted model samples under both strategies, into a given directory too."""
    X, _ = synthetic_data
    num_samples = 10
    im = ImpactModel(lm, rng_key=random.key(0), inference=make_svi(lm))
    try:
        for shard_axis in ("obs", "draw"):
            dt = im.sample_prior_predictive(
                X,
                num_samples=num_samples,
                batch_size=99,
                progress=False,
                shard_axis=shard_axis,
            )
            assert dt.prior_predictive["y"].shape == (1, num_samples, len(X))
        dt = im.sample_prior_predictive(
            X,
            num_samples=num_samples,
            return_sites="y",
            output_dir=tmp_path,
            progress=False,
        )
        assert set(dt.prior_predictive.data_vars) == {"y"}
        assert Path(dt.attrs["artifact_path"]).parent == tmp_path.resolve()
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
    im = ImpactModel(lm, rng_key=random.key(0), inference=make_svi(lm))
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
        inference=make_svi(latent_intervention_model),
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
