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

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from jax.typing import ArrayLike
from numpyro import sample
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

numpyro.set_host_device_count(3)


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[Array, Array]:
    """Fixture for generating synthetic data.

    Returns:
        tuple[Array, Array]: A tuple containing the input data and output data.
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


def lm(X: ArrayLike, y: ArrayLike | None = None) -> None:
    """Linear regression model."""
    n_features = X.shape[1]

    # Priors for weights and bias
    w = sample("w", dist.Normal(jnp.zeros(n_features), jnp.ones(n_features)))
    b = sample("b", dist.Normal(0, 1))

    # Likelihood
    mu = jnp.dot(X, w) + b
    sigma = sample("sigma", dist.Exponential(1.0))
    sample("y", dist.Normal(mu, sigma), obs=y)


def lm_with_kwargs_array(
    X: ArrayLike,
    c: ArrayLike,
    y: ArrayLike | None = None,
) -> None:
    """Linear regression model with an extra array argument."""
    n_features = X.shape[1]

    # Priors for weights and bias
    w = sample("w", dist.Normal(jnp.zeros(n_features), jnp.ones(n_features)))
    b = sample("b", dist.Normal(0, 1))

    # Likelihood
    mu = jnp.dot(X, w) + b + c
    sigma = sample("sigma", dist.Exponential(1.0))
    sample("y", dist.Normal(mu, sigma), obs=y)


def latent_variable_model(X: ArrayLike, y: ArrayLike | None = None) -> None:
    """Latent variable model."""
    z = numpyro.sample(
        "z",
        dist.Normal(0.0, 1.0).expand([X.shape[0]]),
    ) + X.mean(axis=1)
    numpyro.sample("y", dist.Normal(z, 1.0), obs=y)
