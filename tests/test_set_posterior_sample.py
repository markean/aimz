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

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from conftest import lm
from jax import numpy as jnp
from jax import random
from numpyro.infer import Predictive
from numpyro.infer.svi import SVIRunResult

from aimz.model import ImpactModel
from aimz.utils._validation import _is_fitted

if TYPE_CHECKING:
    import jax
    from numpyro.infer import SVI


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_set_posterior_sample(
    synthetic_data: tuple["jax.Array", "jax.Array"],
    vi: "SVI",
) -> None:
    """Test the `.set_posterior_sample()` method of `ImpactModel`."""
    X, y = synthetic_data

    rng_key = random.key(42)
    rng_key, rng_subkey = random.split(key=rng_key)
    vi_result = vi.run(rng_subkey, num_steps=1000, X=X, y=y)

    posterior = Predictive(vi.guide, params=vi_result.params, num_samples=100)
    rng_key, rng_subkey = random.split(rng_key)
    posterior_sample = posterior(rng_subkey)

    im = ImpactModel(lm, rng_key=rng_key, vi=vi)
    im.guide = vi.guide
    assert isinstance(im.guide, Callable)
    im.vi_result = vi_result
    # Use the same key for reproducibility
    im.set_posterior_sample(im.sample(num_samples=100, rng_key=rng_subkey))
    assert _is_fitted(im), "Model fitting check failed"
    assert isinstance(im.vi_result, SVIRunResult)
    assert posterior_sample.keys() == im.posterior_sample_.keys()
    for key in posterior_sample:
        assert jnp.allclose(posterior_sample[key], im.posterior_sample_[key])

    im.set_posterior_sample(im.sample(num_samples=100))
    for key in posterior_sample:
        # Without the `rng_key` argument, we get different posterior samples
        assert not jnp.allclose(posterior_sample[key], im.posterior_sample_[key])
