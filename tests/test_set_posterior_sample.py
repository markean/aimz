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

from typing import TYPE_CHECKING

import pytest
from conftest import lm
from jax import numpy as jnp
from jax import random
from jax.typing import ArrayLike
from numpyro.infer import Predictive
from numpyro.infer.svi import SVIRunResult

from aimz import ImpactModel
from aimz.utils._validation import _is_fitted

if TYPE_CHECKING:
    from numpyro.infer import SVI


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_set_posterior_sample(
    synthetic_data: tuple[ArrayLike, ArrayLike],
    vi: "SVI",
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
    posterior_samples = im.sample(num_samples=100, rng_key=rng_subkey).posterior
    im.set_posterior_sample({k: v.values for k, v in posterior_samples.items()})
    assert _is_fitted(im), "Model fitting check failed"
    assert isinstance(im.vi_result, SVIRunResult)
    assert posterior_sample.keys() == im.posterior.keys()
    for key in posterior_sample:
        assert jnp.allclose(posterior_sample[key], im.posterior[key])

    posterior_samples = im.sample(num_samples=100).posterior
    im.set_posterior_sample({k: v.values for k, v in posterior_samples.items()})
    for key in posterior_sample:
        # Without the `rng_key` argument, we get different posterior samples
        assert not jnp.allclose(posterior_sample[key], im.posterior[key])


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_inconsistent_batch_shapes(vi: "SVI") -> None:
    """Setting a posterior sample with inconsistent batch shapes raises ValueError."""
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    with pytest.raises(
        ValueError,
        match="Inconsistent batch shapes found in posterior_sample",
    ):
        im.set_posterior_sample({"a": jnp.ones((100, 10)), "b": jnp.ones((200,))})
