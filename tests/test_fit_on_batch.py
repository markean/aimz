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

"""Tests for the `.fit_on_batch()` method."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import deterministic, sample
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from tests.conftest import lm


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_fit_svi(synthetic_data: tuple[Array, Array], vi: SVI) -> None:
    """Test the `.fit()` method of ImpactModel using SVI."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    im.fit_on_batch(X=X, y=y)
    assert im.is_fitted(), "Model fitting check failed"
    assert im.vi_result is not None, "VI result should not be `None`"
    first_loss = im.vi_result.losses[0]

    # Continue training to check if loss decreases
    im.fit_on_batch(X=X, y=y)
    last_loss = im.vi_result.losses[-1]
    assert last_loss < first_loss, (
        f"Loss did not decrease after training: first={first_loss}, last={last_loss}"
    )


def test_fit_mcmc_keeps_chains(synthetic_data: tuple[Array, Array]) -> None:
    """Outputs keep the sampler's chains until a posterior is injected."""
    X, y = synthetic_data

    def kernel(X: Array, y: Array | None = None) -> None:
        b = sample("b", dist.Normal(0.0, 1.0))
        mu = deterministic("mu", X.sum(axis=-1) + b)
        sample("y", dist.Normal(mu, 1.0), obs=y)

    im = ImpactModel(
        kernel,
        rng_key=random.key(42),
        inference=MCMC(NUTS(kernel), num_warmup=10, num_samples=5, num_chains=2),
    )
    im.fit_on_batch(X, y)
    b = im.inference.get_samples(group_by_chain=True)["b"]

    for dt in (
        im.predict_on_batch(X),
        im.predict(X, shard_axis="draw", progress=False),
    ):
        assert jnp.array_equal(dt.posterior["b"].values, b)
        assert jnp.allclose(
            dt.posterior_predictive["mu"].values,
            X.sum(axis=-1) + b[..., None],
        )
    im.cleanup()

    # Collapsed draws from both chains, in a count the chains do not divide.
    im.set_posterior_sample({"b": b.reshape(-1)[:7]})
    sizes = im.predict_on_batch(X).posterior_predictive.sizes
    assert (sizes["chain"], sizes["draw"]) == (1, 7)


def test_fit_on_batch_zero_dim_raises(synthetic_data: tuple[Array, Array]) -> None:
    """A 0-dimensional ``X`` or ``y`` raises ``ValueError``."""
    X, y = synthetic_data
    im = ImpactModel(
        lm,
        rng_key=random.key(42),
        inference=SVI(
            lm,
            guide=AutoNormal(lm),
            optim=Adam(step_size=1e-3),
            loss=Trace_ELBO(),
        ),
    )
    with pytest.raises(ValueError, match=r"`X` must have at least 1 dimension."):
        im.fit_on_batch(X=1.0, y=y)
    with pytest.raises(ValueError, match=r"`y` must have at least 1 dimension."):
        im.fit_on_batch(X=X, y=1.0)


def test_fit_on_batch_length_mismatch_raises(
    synthetic_data: tuple[Array, Array],
) -> None:
    """Mismatched leading-axis sizes between ``X`` and ``y`` raise ``ValueError``."""
    X, y = synthetic_data
    im = ImpactModel(
        lm,
        rng_key=random.key(42),
        inference=SVI(
            lm,
            guide=AutoNormal(lm),
            optim=Adam(step_size=1e-3),
            loss=Trace_ELBO(),
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"`X` and `y` must have the same leading-axis size.",
    ):
        im.fit_on_batch(X=X, y=y[:-1])
