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

"""Tests for the `.cleanup_models()` method."""

import pytest
from jax import random
from jax.typing import ArrayLike
from numpyro.infer import SVI

from aimz import ImpactModel
from tests.conftest import lm


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_predict_after_cleanup(
    synthetic_data: tuple[ArrayLike, ArrayLike],
    vi: SVI,
) -> None:
    """Ensure class-level cleanup removes temporary directories for all instances."""
    X, y = synthetic_data

    im1 = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    im1.fit_on_batch(X=X, y=y)

    im2 = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    im2.fit_on_batch(X=X, y=y)

    im1.predict(X, batch_size=3)
    im2.predict(X, batch_size=3)

    ImpactModel.cleanup_models()

    assert im1.temp_dir is None
    assert im2.temp_dir is None
