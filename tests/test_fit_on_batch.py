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

import warnings

import numpy as np
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import sample
from numpyro.infer import MCMC, NUTS

from aimz import ImpactModel
from tests.conftest import lm, make_svi


def test_fit_on_batch_continues_training(synthetic_data: tuple[Array, Array]) -> None:
    """A second call, or a result set through `vi_result`, continues the training."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=make_svi(lm))
    im.fit_on_batch(X, y, num_steps=1000, num_samples=10, progress=False)
    assert im.is_fitted()
    first_loss = im.vi_result.losses[0]
    im.fit_on_batch(X, y, num_steps=1000, num_samples=10, progress=False)
    assert im.vi_result.losses[-1] < first_loss

    # The models hold inference objects that never trained themselves; a first loss
    # near the end of the earlier run shows the training continued from its state
    trained = make_svi(lm).run(
        random.key(0), num_steps=2000, X=X, y=y, progress_bar=False
    )
    for method, kwargs in (
        ("fit_on_batch", {"num_steps": 20}),
        ("fit", {"batch_size": len(X)}),
    ):
        im = ImpactModel(lm, rng_key=random.key(1), inference=make_svi(lm))
        im.vi_result = trained
        getattr(im, method)(X, y, num_samples=10, progress=False, **kwargs)
        assert im.vi_result.losses[0] < trained.losses[-50:].mean() * 1.2


def test_fit_on_batch_mcmc_keeps_chains(synthetic_data: tuple[Array, Array]) -> None:
    """Outputs keep the sampler's chains on every path until a posterior is injected."""
    X, y = synthetic_data

    def kernel(X: Array, y: Array | None = None) -> None:
        b = sample("b", dist.Normal(0.0, 1.0))
        sample("y", dist.Normal(X.sum(axis=-1) + b, 1.0), obs=y)

    mcmc = MCMC(NUTS(kernel), num_warmup=10, num_samples=5, num_chains=2)
    im = ImpactModel(kernel, rng_key=random.key(42), inference=mcmc)
    im.fit_on_batch(X, y)
    b = im.inference.get_samples(group_by_chain=True)["b"]
    try:
        # A rerun under `draw` would warn and bypass the obs path
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            trees = {
                "posterior_predictive": im.predict_on_batch(X),
                "predictions": im.predict(X, in_sample=False, progress=False),
            }
        trees["posterior_predictive"] = im.predict(X, shard_axis="draw", progress=False)
        for group, dt in trees.items():
            np.testing.assert_array_equal(dt.posterior["b"].values, b)
            sizes = dt[group].sizes
            assert (sizes["chain"], sizes["draw"]) == (2, 5)
        for shard_axis in ("obs", "draw"):
            out = im.log_likelihood(X, y, shard_axis=shard_axis, progress=False)
            np.testing.assert_allclose(
                out.log_likelihood["y"].transpose("chain", "draw", ...).values,
                dist.Normal(X.sum(axis=-1) + b[..., None], 1.0).log_prob(y),
                rtol=1e-6,
            )
    finally:
        im.cleanup()

    # Collapsed draws from both chains, in a count the chains do not divide
    im.set_posterior_sample({"b": b.reshape(-1)[:7]})
    sizes = im.predict_on_batch(X).posterior_predictive.sizes
    assert (sizes["chain"], sizes["draw"]) == (1, 7)


def test_fit_on_batch_input_validation(synthetic_data: tuple[Array, Array]) -> None:
    """0-D arrays and mismatched leading axes are rejected on the on-batch paths."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=make_svi(lm))
    with pytest.raises(ValueError, match=r"`X` must have at least 1 dimension."):
        im.fit_on_batch(X=1.0, y=y)
    with pytest.raises(ValueError, match=r"`y` must have at least 1 dimension."):
        im.fit_on_batch(X=X, y=1.0)
    with pytest.raises(ValueError, match="must have the same leading-axis size"):
        im.fit_on_batch(X=X, y=y[:-1])
