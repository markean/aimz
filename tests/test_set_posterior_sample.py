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

import warnings
from pathlib import Path

import jax.numpy as jnp
import pytest
from jax import Array, random
from numpyro.infer import Predictive
from numpyro.infer.svi import SVIRunResult

from aimz import ImpactModel, OutputWarning
from tests.conftest import lm, make_svi


def test_set_posterior_sample(
    synthetic_data: tuple[Array, Array],
    tmp_path: Path,
) -> None:
    """Samples set by hand are validated, kept as given, and used for prediction."""
    X, y = synthetic_data
    vi = make_svi(lm)
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    with pytest.raises(ValueError, match=r"`posterior_sample` cannot be empty\."):
        im.set_posterior_sample({})
    with pytest.raises(
        ValueError,
        match="Inconsistent batch shapes found in `posterior_sample`",
    ):
        im.set_posterior_sample({"a": jnp.ones((100, 10)), "b": jnp.ones((200,))})

    num_samples = 100
    rng_key, rng_subkey = random.split(random.key(0))
    vi_result = vi.run(rng_subkey, num_steps=100, X=X, y=y, progress_bar=False)
    _, rng_subkey = random.split(rng_key)
    expected = Predictive(vi.guide, params=vi_result.params, num_samples=num_samples)(
        rng_subkey,
    )
    # Under the same key, `sample` reproduces NumPyro's draws from the guide
    im.vi_result = vi_result
    posterior = im.sample(num_samples=num_samples, rng_key=rng_subkey).posterior
    im.set_posterior_sample({k: v.values for k, v in posterior.sel(chain=0).items()})
    assert im.is_fitted()
    assert isinstance(im.vi_result, SVIRunResult)
    assert expected.keys() == im.posterior.keys()
    for site, draws in expected.items():
        assert jnp.allclose(draws, im.posterior[site])
    # Without the key, the internal key gives other draws
    posterior = im.sample(num_samples=num_samples).posterior
    im.set_posterior_sample({k: v.values for k, v in posterior.sel(chain=0).items()})
    for site, draws in expected.items():
        assert not jnp.allclose(draws, im.posterior[site])
    # Prediction works after setting the posterior sample
    im.predict_on_batch(X)
    im.predict(X, batch_size=33, output_dir=tmp_path, progress=False)


def test_chain_layout(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Output trees keep the chains the draws are stacked from."""
    X, _ = synthetic_data
    num_chains = 4
    im = ImpactModel(lm, rng_key=random.key(42), inference=make_svi(lm))
    im.set_posterior_sample(im_lm_svi_fitted.posterior, num_chains=num_chains)
    assert im.predict_on_batch(X).posterior_predictive.sizes["chain"] == num_chains

    # A traced kernel warns about a missing latent site; an untraced one cannot
    partial = {k: v for k, v in im_lm_svi_fitted.posterior.items() if k != "sigma"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ImpactModel(
            lm, rng_key=random.key(0), inference=make_svi(lm)
        ).set_posterior_sample(partial)
    im.sample_prior_predictive_on_batch(X, num_samples=2)
    with pytest.warns(OutputWarning, match="no draws for the latent site"):
        im.set_posterior_sample(partial)
