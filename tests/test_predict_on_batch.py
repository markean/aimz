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

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import deterministic, plate, sample

from aimz import ImpactModel, OutputWarning
from tests.conftest import make_svi, mlm


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


def test_predict_on_batch_mlm(synthetic_data: tuple[Array, Array]) -> None:
    """A multivariate output keeps its target axis as a named dimension."""
    X, y = synthetic_data
    im = ImpactModel(mlm, rng_key=random.key(42), inference=make_svi(mlm))
    im.fit_on_batch(
        X, jnp.stack([y, y], axis=1), num_steps=10, num_samples=10, progress=False
    )
    sizes = im.predict_on_batch(X).posterior_predictive["y"].sizes

    assert (sizes["data"], sizes["y_dim_0"]) == (len(X), 2)


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
    im = ImpactModel(kernel, rng_key=random.key(0), inference=make_svi(kernel))
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

    # A name shared with a site of one branch gives way in every trace
    def branched(X: Array, y: Array | None = None, *, observed: bool = False) -> None:
        w = sample(
            "w", dist.Normal().expand([4]).to_event(1), infer={"event_dims": ["q"]}
        )
        with plate("obs", X.shape[0]):
            sample("y", dist.Normal(w.sum() + X[:, 0], 1.0), obs=y)
        if (y is not None) is observed:
            deterministic("q", 2 * w)

    im = ImpactModel(branched, rng_key=random.key(0), inference=make_svi(branched))
    with pytest.warns(OutputWarning, match="names .* of site 'w' cannot apply"):
        im.sample_prior_predictive_on_batch(X, num_samples=4)
    with pytest.warns(OutputWarning, match="names .* of site 'w' cannot apply"):
        im.fit_on_batch(X, y, num_steps=5, num_samples=4, progress=False)
    dt = im.sample_prior_predictive_on_batch(X, num_samples=4, return_sites=["q", "w"])
    assert dt.prior_predictive["w"].dims == ("chain", "draw", "w_dim_0")
    assert "q" in dt.prior_predictive.data_vars
    # With the site in the other branch, the fit drops the name stored before
    im = ImpactModel(branched, rng_key=random.key(0), inference=make_svi(branched))
    im.sample_prior_predictive_on_batch(X, num_samples=4, observed=True)
    with pytest.warns(OutputWarning, match="names .* of site 'w' cannot apply"):
        im.fit_on_batch(X, y, num_steps=5, num_samples=4, progress=False, observed=True)
    dt = im.predict_on_batch(X, observed=True)
    assert dt.posterior["w"].dims == ("chain", "draw", "w_dim_0")
    # A fit before any prior trace stores the name, which the tree then drops
    im = ImpactModel(branched, rng_key=random.key(0), inference=make_svi(branched))
    im.fit_on_batch(X, y, num_steps=5, num_samples=4, progress=False)
    with pytest.warns(OutputWarning) as record:
        dt = im.sample_prior_predictive_on_batch(
            X, num_samples=4, return_sites=["q", "w"]
        )
    assert any("'q' would name a site" in str(w.message) for w in record)
    assert dt.prior_predictive["w"].dims == ("chain", "draw", "w_dim_0")
    assert "q" in dt.prior_predictive.data_vars


def test_predict_on_batch_default_names_skip_sites(
    synthetic_data: tuple[Array, Array],
) -> None:
    """A default dimension name that is a site of the group skips to the next one."""
    X, y = synthetic_data

    def named_like_default(X: Array, y: Array | None = None) -> None:
        b = sample("b", dist.Normal().expand([3]).to_event(1))
        with plate("obs", X.shape[0]):
            mu = deterministic("mu", X[:, :1] + b)
            sample("y", dist.Normal(mu.sum(-1), 1.0), obs=y)
        deterministic("mu_dim_0", b.sum())

    im = ImpactModel(
        named_like_default,
        rng_key=random.key(0),
        inference=make_svi(named_like_default),
    )
    im.fit_on_batch(X, y, num_steps=5, num_samples=4, progress=False)
    with pytest.warns(OutputWarning, match="'mu' skip those that name a site"):
        dt = im.predict_on_batch(X)
    assert dt.posterior_predictive["mu"].dims == ("chain", "draw", "obs", "mu_dim_1")
    assert "mu_dim_0" in dt.posterior_predictive.data_vars
