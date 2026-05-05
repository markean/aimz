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

"""pytest configuration."""

from collections.abc import Callable, Iterator

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import sample
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from aimz import ImpactModel

numpyro.set_host_device_count(3)


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[Array, Array]:
    """Fixture for generating synthetic data.

    Returns:
        The input data and output data.
    """
    rng_key = random.key(42)
    key_w, key_b, key_x, key_e = random.split(rng_key, 4)

    w = random.normal(key_w, (10,))
    b = random.normal(key_b)

    X = random.normal(key_x, (100, 10))
    e = random.normal(key_e, (100,))
    y = jnp.dot(X, w) + b + e

    return X, y


@pytest.fixture(scope="module")
def mcmc(request: pytest.FixtureRequest) -> MCMC:
    """Fixture for creating an MCMC object.

    Args:
        request (pytest.FixtureRequest): The pytest request object to access the
            parameter.

    Returns:
        An MCMC object configured with the provided model.
    """
    model = request.param

    return MCMC(
        NUTS(model),
        num_warmup=100,
        num_samples=100,
        num_chains=1,
    )


@pytest.fixture(scope="module")
def vi(request: pytest.FixtureRequest) -> SVI:
    """Fixture for creating a variational inference object.

    Args:
        request (pytest.FixtureRequest): The pytest request object to access the
            parameter.

    Returns:
        A variational inference object configured with the provided model.
    """
    model = request.param

    return SVI(
        model,
        guide=AutoNormal(model),
        optim=numpyro.optim.Adam(step_size=1e-3),
        loss=Trace_ELBO(),
    )


def lm(X: Array, y: Array | None = None) -> None:
    """Linear regression model."""
    n_features = X.shape[1]
    w = sample("w", dist.Normal(jnp.zeros(n_features), jnp.ones(n_features)))
    b = sample("b", dist.Normal(0, 1))
    mu = jnp.dot(X, w) + b
    sigma = sample("sigma", dist.Exponential(1.0))
    sample("y", dist.Normal(mu, sigma), obs=y)


def lm_with_kwargs_array(X: Array, c: Array, y: Array | None = None) -> None:
    """Linear regression model with an extra array argument."""
    n_features = X.shape[1]
    w = sample("w", dist.Normal(jnp.zeros(n_features), jnp.ones(n_features)))
    b = sample("b", dist.Normal(0, 1))
    mu = jnp.dot(X, w) + b + c
    sigma = sample("sigma", dist.Exponential(1.0))
    sample("y", dist.Normal(mu, sigma), obs=y)


def mlm(X: Array, y: Array | None = None) -> None:
    """Multivariate linear regression model."""
    n_features = X.shape[1]
    n_targets = 2
    w = sample(
        "w",
        dist.Normal().expand([n_features, n_targets]).to_event(2),
    )
    sigma = sample("sigma", dist.Exponential())
    with numpyro.plate("data", X.shape[0]):
        sample("y", dist.Normal(X @ w, sigma).to_event(1), obs=y)


def latent_variable_model(X: Array, y: Array | None = None) -> None:
    """Latent variable model."""
    z = numpyro.sample(
        "z",
        dist.Normal(0.0, 1.0).expand([X.shape[0]]),
    ) + X.mean(axis=1)
    numpyro.sample("y", dist.Normal(z, 1.0), obs=y)


def _make_svi(model: Callable) -> SVI:

    return SVI(
        model,
        guide=AutoNormal(model),
        optim=numpyro.optim.Adam(step_size=1e-3),
        loss=Trace_ELBO(),
    )


def _make_mcmc(model: Callable) -> MCMC:

    return MCMC(NUTS(model), num_warmup=100, num_samples=100, num_chains=1)


@pytest.fixture(scope="module")
def im_lm_svi_fitted(synthetic_data: tuple[Array, Array]) -> Iterator[ImpactModel]:
    """`lm` fitted with SVI. Reusable for read-only tests."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=_make_svi(lm))
    im.fit(X=X, y=y, batch_size=len(X), progress=False)
    yield im
    im.cleanup()


@pytest.fixture(scope="module")
def im_lm_mcmc_fitted(synthetic_data: tuple[Array, Array]) -> Iterator[ImpactModel]:
    """`lm` fitted with MCMC. Reusable for read-only tests."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=_make_mcmc(lm))
    im.fit_on_batch(X=X, y=y)
    yield im
    im.cleanup()


@pytest.fixture(scope="module")
def im_lm_with_kwargs_svi_fitted(
    synthetic_data: tuple[Array, Array],
) -> Iterator[ImpactModel]:
    """`lm_with_kwargs_array` fitted with SVI. Reusable for read-only tests."""
    X, y = synthetic_data
    im = ImpactModel(
        lm_with_kwargs_array,
        rng_key=random.key(42),
        inference=_make_svi(lm_with_kwargs_array),
    )
    im.fit(X=X, y=y, c=y, batch_size=3, progress=False)
    yield im
    im.cleanup()


@pytest.fixture(scope="module")
def im_latent_var_svi_fitted(
    synthetic_data: tuple[Array, Array],
) -> Iterator[ImpactModel]:
    """`latent_variable_model` fitted with SVI. Reusable for read-only tests."""
    X, y = synthetic_data
    im = ImpactModel(
        latent_variable_model,
        rng_key=random.key(42),
        inference=_make_svi(latent_variable_model),
    )
    im.fit(X=X, y=y, batch_size=len(X), progress=False)
    yield im
    im.cleanup()
