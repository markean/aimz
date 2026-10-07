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

"""pytest configuration: shared data, kernels, inference, and fitted models."""

from collections.abc import Callable, Iterator
from pathlib import Path

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import deterministic, plate, sample
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from aimz import ImpactModel

numpyro.set_host_device_count(3)


@pytest.fixture(autouse=True)
def _chdir_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test in its temporary directory, so no file lands in the repository."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[Array, Array]:
    """Generate 100 observations of 10 features with a linear Gaussian outcome."""
    key_w, key_b, key_x, key_e = random.split(random.key(42), 4)
    w = random.normal(key_w, (10,))
    b = random.normal(key_b)
    X = random.normal(key_x, (100, 10))
    e = random.normal(key_e, (100,))

    return X, jnp.dot(X, w) + b + e


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
    with plate("data", size=100, subsample_size=X.shape[0]):
        sample("y", dist.Normal(mu, sigma), obs=y)


def mlm(X: Array, y: Array | None = None) -> None:
    """Multivariate linear regression model."""
    n_features = X.shape[1]
    n_targets = 2
    w = sample("w", dist.Normal().expand([n_features, n_targets]).to_event(2))
    sigma = sample("sigma", dist.Exponential())
    with plate("data", X.shape[0]):
        sample("y", dist.Normal(X @ w, sigma).to_event(1), obs=y)


def lm_subsample(X: Array, y: Array | None = None) -> None:
    """Linear regression whose plate scales a batch to the 100 rows of the data."""
    n_features = X.shape[1]
    w = sample("w", dist.Normal(jnp.zeros(n_features), jnp.ones(n_features)))
    b = sample("b", dist.Normal(0, 1))
    sigma = sample("sigma", dist.Exponential(1.0))
    with plate("data", size=100, subsample_size=X.shape[0]):
        mu = jnp.dot(X, w) + b
        sample("y", dist.Normal(mu, sigma), obs=y)


def latent_variable_model(X: Array, y: Array | None = None) -> None:
    """Latent variable model."""
    z = sample("z", dist.Normal(0.0, 1.0).expand([X.shape[0]])) + X.mean(axis=1)
    sample("y", dist.Normal(z, 1.0), obs=y)


def multidim_latent_model(X: Array, y: Array | None = None) -> None:
    """Model with a rank-3 observation-aligned latent `(num_samples, n_obs, k)`."""
    z = sample("z", dist.Normal(0.0, 1.0).expand([X.shape[0], 2]).to_event(2))
    mu = z.mean(axis=-1) + X.mean(axis=-1)
    sample("y", dist.Normal(mu, 1.0), obs=y)


def latent_intervention_model(X: Array, y: Array | None = None) -> None:
    """Latent variable model with a deterministic site downstream of its latent."""
    w = sample("w", dist.Normal(0.0, 1.0))
    z = sample("z", dist.Normal(0.0, 1.0).expand([X.shape[0]]))
    deterministic("mu", z + X[:, 1] + w)
    sample("y", dist.Normal(z, 1.0), obs=y)


def make_svi(model: Callable, *, step_size: float = 1e-3) -> SVI:
    """Mean-field SVI for ``model`` with an Adam optimizer."""
    return SVI(
        model,
        guide=AutoNormal(model),
        optim=numpyro.optim.Adam(step_size=step_size),
        loss=Trace_ELBO(),
    )


@pytest.fixture(scope="module")
def im_lm_svi_fitted(synthetic_data: tuple[Array, Array]) -> Iterator[ImpactModel]:
    """`lm` fitted with SVI, for read-only tests."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=make_svi(lm))
    im.fit(X=X, y=y, batch_size=len(X), progress=False)
    yield im
    im.cleanup()


@pytest.fixture(scope="module")
def im_lm_with_kwargs_svi_fitted(
    synthetic_data: tuple[Array, Array],
) -> Iterator[ImpactModel]:
    """`lm_with_kwargs_array` fitted with SVI, for read-only tests."""
    X, y = synthetic_data
    im = ImpactModel(
        lm_with_kwargs_array,
        rng_key=random.key(42),
        inference=make_svi(lm_with_kwargs_array),
    )
    im.fit(X=X, y=y, c=y, batch_size=3, progress=False)
    yield im
    im.cleanup()


@pytest.fixture(scope="module")
def im_latent_var_svi_fitted(
    synthetic_data: tuple[Array, Array],
) -> Iterator[ImpactModel]:
    """`latent_variable_model` fitted with SVI, for read-only tests."""
    X, y = synthetic_data
    im = ImpactModel(
        latent_variable_model,
        rng_key=random.key(42),
        inference=make_svi(latent_variable_model),
    )
    im.fit(X=X, y=y, batch_size=len(X), progress=False)
    yield im
    im.cleanup()
