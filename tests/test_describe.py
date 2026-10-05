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

"""Tests for the `.describe()` method."""

import json

import jax.numpy as jnp
from jax import Array, random

from aimz import ImpactModel
from tests.conftest import _make_svi, latent_intervention_model, lm, mlm


def test_describe_workflow(synthetic_data: tuple[Array, Array]) -> None:
    """The description follows the trace and the posterior as the model gets them."""
    X, y = synthetic_data
    im = ImpactModel(
        latent_intervention_model,
        rng_key=random.key(0),
        inference=_make_svi(latent_intervention_model),
    )
    before = im.describe()
    assert before["arguments"] == ["X", "y"]
    assert before["inference"] == "SVI"
    assert before["fitted"] is False
    assert before["sites"] == {}

    # Fitting traces the kernel: every site gets a kind, and latent sites a draw shape
    num_samples = 20
    im.fit_on_batch(X, y, num_steps=10, num_samples=num_samples, progress=False)
    after = im.describe()
    assert after["fitted"]
    assert after["num_samples"] == num_samples
    assert {name: site["kind"] for name, site in after["sites"].items()} == {
        "w": "latent",
        "z": "latent",
        "y": "observed",
        "mu": "deterministic",
    }
    assert after["sites"]["w"]["draw_shape"] == []
    assert after["sites"]["z"]["draw_shape"] == [len(X)]
    # No plate, so the output trees use the default dimension names
    assert after["sites"]["mu"]["dims"] is None
    json.dumps(after)

    # A site inside a plate carries the plate's name
    im_plate = ImpactModel(mlm, rng_key=random.key(0), inference=_make_svi(mlm))
    im_plate.fit_on_batch(
        X, jnp.stack([y, y], axis=1), num_steps=10, num_samples=20, progress=False
    )
    assert im_plate.describe()["sites"]["y"]["dims"] == ["data"]

    # Samples set by hand on a model that was never traced: the sites come from them
    im_set = ImpactModel(lm, rng_key=random.key(0), inference=_make_svi(lm))
    im_set.set_posterior_sample(
        {"w": jnp.zeros((num_samples, 10)), "b": jnp.zeros(num_samples)},
    )
    described = im_set.describe()
    assert described["fitted"]
    assert described["num_samples"] == num_samples
    assert described["sites"]["w"] == {
        "kind": "latent",
        "dims": None,
        "draw_shape": [10],
    }
    assert described["sites"]["y"]["kind"] == "observed"
