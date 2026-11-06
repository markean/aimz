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
import numpyro.distributions as dist
from jax import Array, random
from numpyro import deterministic, plate, sample
from numpyro.contrib.control_flow import scan

from aimz import ImpactModel
from tests.conftest import latent_intervention_model, lm, make_svi, mlm


def two_observed(X: Array, t: Array | None = None, y: Array | None = None) -> None:
    """Model with the treatment observed through a keyword argument."""
    p = sample("p", dist.Beta(1.0, 1.0))
    with plate("obs", X.shape[0]):
        treat = sample("treat", dist.Bernoulli(p), obs=t)
        mu = deterministic("mu", treat + X[:, 0])
        sample("y", dist.Normal(mu, 1.0), obs=y)


def scanned(X: Array, y: Array | None = None) -> None:
    """Model whose observed site `scan` records."""
    a = sample("a", dist.Normal(0.0, 1.0))

    def step(carry: Array, xs: tuple[Array, Array | None]) -> tuple[Array, Array]:
        x, y_t = xs
        carry = 0.5 * carry + a * x
        sample("y", dist.Normal(carry, 1.0), obs=y_t)

        return carry, carry

    scan(step, 0.0, (X[:, 0], y))


def test_describe_workflow(synthetic_data: tuple[Array, Array]) -> None:
    """The description follows the trace and the posterior as the model gets them."""
    X, y = synthetic_data
    im = ImpactModel(
        latent_intervention_model,
        rng_key=random.key(0),
        inference=make_svi(latent_intervention_model),
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
    # The relations come from the kernel: the distribution under its `expand`, and
    # the sample sites and the array arguments each site is computed from
    assert {name: site["distribution"] for name, site in after["sites"].items()} == {
        "w": "Normal",
        "z": "Normal",
        "y": "Normal",
        "mu": None,
    }
    assert {name: site["upstream"] for name, site in after["sites"].items()} == {
        "w": [],
        "z": [],
        "y": ["z"],
        "mu": ["w", "z"],
    }
    assert {name: site["arguments"] for name, site in after["sites"].items()} == {
        "w": [],
        "z": [],
        "y": [],
        "mu": ["X"],
    }
    json.dumps(after)

    # A site inside a plate carries the plate's name
    im_plate = ImpactModel(mlm, rng_key=random.key(0), inference=make_svi(mlm))
    im_plate.fit_on_batch(
        X, jnp.stack([y, y], axis=1), num_steps=10, num_samples=20, progress=False
    )
    described = im_plate.describe()
    assert described["sites"]["y"]["dims"] == ["data"]
    # A `to_event` wrapper does not hide the distribution
    assert described["sites"]["y"]["distribution"] == "Normal"
    assert described["sites"]["y"]["upstream"] == ["sigma", "w"]

    # A sample site observed through a keyword argument is observed, not latent
    im_two = ImpactModel(
        two_observed,
        rng_key=random.key(0),
        inference=make_svi(two_observed),
    )
    im_two.fit_on_batch(
        X, y, t=jnp.ones(len(X)), num_steps=10, num_samples=20, progress=False
    )
    described = im_two.describe()
    assert {name: site["kind"] for name, site in described["sites"].items()} == {
        "p": "latent",
        "treat": "observed",
        "y": "observed",
        "mu": "deterministic",
    }
    # The distribution carries its factory's name; the observed treatment is a node
    # of the chain, and the deterministic site between passes it and the argument
    # through to the output
    assert described["sites"]["treat"]["distribution"] == "Bernoulli"
    assert described["sites"]["treat"]["upstream"] == ["p"]
    assert described["sites"]["treat"]["arguments"] == []
    assert described["sites"]["mu"]["upstream"] == ["treat"]
    assert described["sites"]["y"]["upstream"] == ["treat"]
    assert described["sites"]["y"]["arguments"] == ["X"]

    # A site that `scan` records has no derived relations
    im_scan = ImpactModel(scanned, rng_key=random.key(0), inference=make_svi(scanned))
    im_scan.fit_on_batch(X, y, num_steps=10, num_samples=20, progress=False)
    described = im_scan.describe()
    assert described["sites"]["a"]["distribution"] == "Normal"
    assert described["sites"]["y"]["upstream"] == []
    assert described["sites"]["y"]["arguments"] == []

    # Samples set by hand on a model that was never traced: the sites come from them
    im_set = ImpactModel(lm, rng_key=random.key(0), inference=make_svi(lm))
    im_set.set_posterior_sample(
        {"w": jnp.zeros((num_samples, 10)), "b": jnp.zeros(num_samples)},
    )
    described = im_set.describe()
    assert described["fitted"]
    assert described["num_samples"] == num_samples
    assert described["sites"]["w"] == {
        "kind": "latent",
        "distribution": None,
        "dims": None,
        "upstream": [],
        "arguments": [],
        "draw_shape": [10],
    }
    assert described["sites"]["y"]["kind"] == "observed"
