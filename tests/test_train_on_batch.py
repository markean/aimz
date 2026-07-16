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

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import sample
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from tests.conftest import lm_with_kwargs_array


@pytest.mark.parametrize("vi", [lm_with_kwargs_array], indirect=True)
def test_train_on_batch_lm_with_kwargs_array(
    synthetic_data: tuple[Array, Array],
    vi: SVI,
) -> None:
    """Test the `.train_on_batch()` method of `ImpactModel`."""
    X, y = synthetic_data
    im = ImpactModel(lm_with_kwargs_array, rng_key=random.key(42), inference=vi)

    for i in range(1000):
        _, loss = im.train_on_batch(X=X, y=y, c=y)
        if i == 0:
            first_loss = float(loss)
        last_loss = float(loss)

    assert last_loss < first_loss, (
        f"Loss did not decrease after training: first={first_loss}, last={last_loss}"
    )


def test_train_on_batch_different_extra_kwargs(
    synthetic_data: tuple[Array, Array],
) -> None:
    """Calls with different non-array kwargs each get their own static wrapper."""
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

    vi = SVI(
        kernel,
        guide=AutoNormal(kernel),
        optim=Adam(step_size=1e-3),
        loss=Trace_ELBO(),
    )
    im = ImpactModel(kernel, rng_key=random.key(42), inference=vi)

    # Each call passes a different set of string (non-array, static) kwargs
    im.train_on_batch(X=X, y=y, link="identity")
    im.train_on_batch(X=X, y=y, noise="normal")
    assert sorted(im._fn_vi_update) == [("link",), ("noise",)]
