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

"""Tests for PRNG key consistency."""

import jax.numpy as jnp
import pytest
from conftest import lm
from jax import random
from jax.typing import ArrayLike
from numpyro.infer import SVI

from aimz import ImpactModel


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_rng_key_consistency(
    synthetic_data: tuple[ArrayLike, ArrayLike],
    vi: SVI,
) -> None:
    """Test that the ImpactModel's internal rng_key remains unchanged.

    Verifies that providing an explicit PRNG key to method calls does not mutate the
    model's internal random state.
    """
    X, y = synthetic_data
    rng_key = random.key(42)
    im = ImpactModel(lm, rng_key=rng_key, inference=vi)
    assert jnp.allclose(im.rng_key, rng_key)
    im.train_on_batch(X=X, y=y, rng_key=rng_key)
    assert jnp.allclose(im.rng_key, rng_key)
    im.fit_on_batch(X=X, y=y, rng_key=rng_key, progress=False)
    assert jnp.allclose(im.rng_key, rng_key)
    im.fit(X=X, y=y, rng_key=rng_key, batch_size=3, progress=False)
    assert jnp.allclose(im.rng_key, rng_key)
    im.log_likelihood(X=X, y=y, batch_size=3, progress=False)
    assert jnp.allclose(im.rng_key, rng_key)
    im.predict_on_batch(X=X, rng_key=rng_key)
    assert jnp.allclose(im.rng_key, rng_key)
    im.predict(X=X, rng_key=rng_key, batch_size=3, progress=False)
    assert jnp.allclose(im.rng_key, rng_key)
