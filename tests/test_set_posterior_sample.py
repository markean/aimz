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

"""Tests for the `.set_posterior_sample()` method."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest
from jax import Array, random
from jax import numpy as jnp
from numpyro.infer import Predictive
from numpyro.infer.svi import SVIRunResult

from aimz import ImpactModel, OutputWarning
from tests.conftest import lm

if TYPE_CHECKING:
    from pathlib import Path

    from numpyro.infer import SVI


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_empty_posterior_sample(vi: SVI) -> None:
    """Empty posterior sample raises ValueError."""
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    msg = "`posterior_sample` cannot be empty."
    with pytest.raises(ValueError, match=msg):
        im.set_posterior_sample({})


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_set_posterior_sample(
    synthetic_data: tuple[Array, Array],
    vi: SVI,
    tmp_path: Path,
) -> None:
    """Test the `.set_posterior_sample()` method of ImpactModel."""
    X, y = synthetic_data

    rng_key = random.key(42)
    rng_key, rng_subkey = random.split(rng_key)
    vi_result = vi.run(rng_subkey, num_steps=1000, X=X, y=y)

    posterior = Predictive(vi.guide, params=vi_result.params, num_samples=100)
    rng_key, rng_subkey = random.split(rng_key)
    posterior_sample = posterior(rng_subkey)

    im = ImpactModel(lm, rng_key=rng_key, inference=vi)
    im.vi_result = vi_result
    # Use the same key for reproducibility
    posterior_samples = im.sample(num_samples=100, rng_key=rng_subkey).posterior.sel(
        chain=0,
    )
    im.set_posterior_sample({k: v.values for k, v in posterior_samples.items()})
    assert im.is_fitted(), "Model fitting check failed"
    assert isinstance(im.vi_result, SVIRunResult)
    assert posterior_sample.keys() == im.posterior.keys()
    for key in posterior_sample:
        assert jnp.allclose(posterior_sample[key], im.posterior[key])

    posterior_samples = im.sample(num_samples=100).posterior.sel(chain=0)
    im.set_posterior_sample({k: v.values for k, v in posterior_samples.items()})
    for key in posterior_sample:
        # Without the `rng_key` argument, we get different posterior samples
        assert not jnp.allclose(posterior_sample[key], im.posterior[key])

    # Check that prediction works after setting the posterior sample
    im.predict_on_batch(X)
    im.predict(X, batch_size=33, output_dir=tmp_path, progress=False)


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_inconsistent_batch_shapes(vi: SVI) -> None:
    """Setting a posterior sample with inconsistent batch shapes raises ValueError."""
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    with pytest.raises(
        ValueError,
        match="Inconsistent batch shapes found in `posterior_sample`",
    ):
        im.set_posterior_sample({"a": jnp.ones((100, 10)), "b": jnp.ones((200,))})


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_chain_layout(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    vi: SVI,
) -> None:
    """Output trees keep the chains the draws are stacked from."""
    X, _ = synthetic_data
    num_chains = 4
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    im.set_posterior_sample(im_lm_svi_fitted.posterior, num_chains=num_chains)
    assert im.predict_on_batch(X).posterior_predictive.sizes["chain"] == num_chains

    # Once the kernel is traced, a posterior without one of its latent sites warns; an
    # untraced model has nothing to compare it with
    partial = {k: v for k, v in im_lm_svi_fitted.posterior.items() if k != "sigma"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ImpactModel(lm, rng_key=random.key(0), inference=vi).set_posterior_sample(
            partial,
        )
    im.sample_prior_predictive_on_batch(X, num_samples=2)
    with pytest.warns(OutputWarning, match="no draws for the latent site"):
        im.set_posterior_sample(partial)
