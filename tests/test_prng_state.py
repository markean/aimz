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

import numpy as np
import pytest
from jax import Array, random

from aimz import AimzWarning, ImpactModel
from tests.conftest import lm_subsample, make_svi


def test_rng_key_consistency(synthetic_data: tuple[Array, Array]) -> None:
    """An explicit key leaves the model's internal key unchanged on every method.

    A legacy ``uint32`` key given to the constructor is converted with a warning.
    """
    X, y = synthetic_data
    with pytest.warns(AimzWarning, match="Legacy `uint32` PRNGKey detected"):
        im = ImpactModel(
            lm_subsample,
            rng_key=random.PRNGKey(42),
            inference=make_svi(lm_subsample),
        )
    rng_key = random.key(42)
    key_data = random.key_data(rng_key)
    np.testing.assert_array_equal(random.key_data(im.rng_key), key_data)
    im.train_on_batch(X=X, y=y, rng_key=rng_key)
    np.testing.assert_array_equal(random.key_data(im.rng_key), key_data)
    im.fit_on_batch(
        X=X, y=y, rng_key=rng_key, num_steps=10, num_samples=10, progress=False
    )
    np.testing.assert_array_equal(random.key_data(im.rng_key), key_data)
    im.fit(X=X, y=y, rng_key=rng_key, batch_size=3, progress=False)
    np.testing.assert_array_equal(random.key_data(im.rng_key), key_data)
    im.log_likelihood(X=X, y=y, batch_size=3, progress=False)
    np.testing.assert_array_equal(random.key_data(im.rng_key), key_data)
    im.predict_on_batch(X=X, rng_key=rng_key)
    np.testing.assert_array_equal(random.key_data(im.rng_key), key_data)
    im.predict(X=X, rng_key=rng_key, batch_size=3, progress=False)
    np.testing.assert_array_equal(random.key_data(im.rng_key), key_data)
    im.cleanup()
