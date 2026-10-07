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

"""Tests for the `.train_on_batch()` method."""

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import Array, random
from numpyro import sample

from aimz import ImpactModel
from tests.conftest import lm_with_kwargs_array, make_svi


def test_train_on_batch_lm_with_kwargs_array(
    synthetic_data: tuple[Array, Array],
) -> None:
    """The loss decreases over repeated steps on one batch."""
    X, y = synthetic_data
    im = ImpactModel(
        lm_with_kwargs_array,
        rng_key=random.key(42),
        inference=make_svi(lm_with_kwargs_array),
    )
    losses = [float(im.train_on_batch(X=X, y=y, c=y)[1]) for _ in range(1000)]

    assert losses[-1] < losses[0]


def test_train_on_batch_different_extra_kwargs(
    synthetic_data: tuple[Array, Array],
) -> None:
    """Calls with different static (non-array) kwargs each compile their own update."""
    X, y = synthetic_data

    def kernel(
        X: Array,
        link: str = "identity",
        noise: str = "normal",
        y: Array | None = None,
    ) -> None:
        w = sample("w", dist.Normal(jnp.zeros(X.shape[1]), 1.0).to_event(1))
        mu = jnp.dot(X, w)
        mu = mu if link == "identity" else jnp.exp(mu)
        sample("y", dist.Normal(mu, 1.0), obs=y)

    im = ImpactModel(kernel, rng_key=random.key(42), inference=make_svi(kernel))
    # Each call passes a different set of string (non-array, static) kwargs
    im.train_on_batch(X=X, y=y, link="identity")
    im.train_on_batch(X=X, y=y, noise="normal")
    # The streaming methods take the same static kwargs
    im.sample_prior_predictive(
        X, link="identity", num_samples=2, store="memory", progress=False
    )
    num_configs = 2

    assert im._fn_vi_update._cache_size() == num_configs
