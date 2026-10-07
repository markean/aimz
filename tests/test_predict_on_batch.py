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

"""Tests for the `.predict_on_batch()` method."""

import numpy as np
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import deterministic, plate, sample
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel, OutputWarning
from tests.conftest import _make_svi, mlm


def test_predict_on_batch_lm_with_kwargs_array(
    synthetic_data: tuple[Array, Array],
    im_lm_with_kwargs_svi_fitted: ImpactModel,
) -> None:
    """Test the `.predict_on_batch()` method of ImpactModel."""
    X, y = synthetic_data
    rng_key = random.key(0)
    dt = im_lm_with_kwargs_svi_fitted.predict_on_batch(
        X=X,
        c=y,
        rng_key=rng_key,
        return_sites="y",
    )
    assert set(dt.posterior_predictive.data_vars) == {"y"}
    assert dt.posterior_predictive["y"].sizes["data"] == X.shape[0]

    # `.sample_posterior_predictive_on_batch()` is an alias for `.predict_on_batch()`.
    samples = im_lm_with_kwargs_svi_fitted.sample_posterior_predictive_on_batch(
        X=X,
        c=y,
        rng_key=rng_key,
        return_sites=["y"],
        return_datatree=False,
    )
    np.testing.assert_array_equal(samples["y"], dt.posterior_predictive["y"][0])


def test_predict_on_batch_x_zero_dim_raises(
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """A 0-dimensional ``X`` raises ``ValueError``."""
    with pytest.raises(ValueError, match=r"`X` must have at least 1 dimension."):
        im_lm_svi_fitted.predict_on_batch(X=1.0)


def test_predict_on_batch_mlm() -> None:
    """`.predict_on_batch()` works with a multivariate linear regression model."""
    n_obs, n_features, n_targets = 100, 3, 2
    rng_key = random.key(42)
    rng_key, rng_subkey = random.split(rng_key)
    X = random.normal(rng_subkey, (n_obs, n_features))
    rng_key, rng_subkey = random.split(rng_key)
    w = random.normal(rng_subkey, (n_features, n_targets))
    rng_key, rng_subkey = random.split(rng_key)
    e = random.normal(rng_subkey, (n_obs, n_targets))
    y = X @ w + e

    rng_key, rng_subkey = random.split(rng_key)
    im = ImpactModel(
        mlm,
        rng_key=rng_subkey,
        inference=SVI(
            mlm,
            guide=AutoNormal(mlm),
            optim=Adam(step_size=1e-2),
            loss=Trace_ELBO(),
        ),
    )
    im.fit_on_batch(X=X, y=y, num_steps=10, num_samples=10, progress=False)
    out = im.predict_on_batch(X=X)

    assert out["posterior_predictive"]["y"].sizes["data"] == n_obs
    assert out["posterior_predictive"]["y"].sizes["y_dim_0"] == n_targets


def test_predict_on_batch_warns_on_dimension_names_that_cannot_apply() -> None:
    """Names a kernel gives but cannot use give way to the defaults with a warning."""

    def kernel(X: Array, y: Array | None = None) -> None:
        w = sample(
            "w", dist.Normal().expand([2]).to_event(1), infer={"event_dims": ["f"]}
        )
        # Too many names for the site's one dimension
        v = sample(
            "v", dist.Normal().expand([2]).to_event(1), infer={"event_dims": ["f", "g"]}
        )
        # A name shared with `w` at another length
        u = sample(
            "u", dist.Normal().expand([3]).to_event(1), infer={"event_dims": ["f"]}
        )
        with plate("obs", X.shape[0]):
            mu = deterministic("mu", X @ w + v.sum() + u.sum())
            sample("y", dist.Normal(mu, 1.0), obs=y)

    X = np.linspace(-1.0, 1.0, 20).reshape(10, 2)
    y = X.sum(axis=1)
    im = ImpactModel(kernel, rng_key=random.key(0), inference=_make_svi(kernel))
    # The trace drops names that outnumber the site's dimensions
    with pytest.warns(OutputWarning, match="names .* of site 'v' cannot apply"):
        im.fit_on_batch(X, y, num_steps=5, num_samples=4, progress=False)
    # Within the posterior group, one name with two lengths sends both sites back to
    # the defaults, while the plate name still applies
    with pytest.warns(OutputWarning, match="'u', 'w' keep the default|'w', 'u' keep"):
        dt = im.predict_on_batch(X)
    assert dt.posterior["w"].dims == ("chain", "draw", "w_dim_0")
    assert dt.posterior["u"].dims == ("chain", "draw", "u_dim_0")
    assert dt.posterior["v"].dims == ("chain", "draw", "v_dim_0")
    assert dt.posterior_predictive["mu"].dims == ("chain", "draw", "obs")
