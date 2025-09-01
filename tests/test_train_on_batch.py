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

"""Tests for the `.train_on_batch()` method."""

import pytest
from conftest import lm_with_kwargs_array
from jax import random
from jax.typing import ArrayLike
from numpyro.infer import SVI

from aimz import ImpactModel


@pytest.mark.parametrize("vi", [lm_with_kwargs_array], indirect=True)
def test_train_on_batch_lm_with_kwargs_array(
    synthetic_data: tuple[ArrayLike, ArrayLike],
    vi: SVI,
) -> None:
    """Test the `.train_on_batch()` method of `ImpactModel`."""
    X, y = synthetic_data
    im = ImpactModel(lm_with_kwargs_array, rng_key=random.key(42), inference=vi)

    for i in range(1000):
        _, loss = im.train_on_batch(X=X, y=y, c=y)
        if i == 0:
            first_loss = float(loss)
        last_loss = float(loss)

    assert last_loss < first_loss, (
        f"Loss did not decrease after training: first={first_loss}, last={last_loss}"
    )
