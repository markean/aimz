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

from typing import TYPE_CHECKING

import pytest
from conftest import lm
from jax import random
from numpyro.infer import SVI

from aimz.model import ImpactModel
from aimz.utils._validation import _is_fitted

if TYPE_CHECKING:
    import jax


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_fit_lm(synthetic_data: tuple["jax.Array", "jax.Array"], vi: SVI) -> None:
    """Test the `.fit()` method of `ImpactModel`."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), vi=vi)
    im.fit_on_batch(X=X, y=y)
    assert _is_fitted(im), "Model fitting check failed"
    assert im.vi_result is not None, "VI result should not be `None`"
    first_loss = im.vi_result.losses[0]

    # Continue training to check if loss decreases
    im.fit_on_batch(X=X, y=y)
    last_loss = im.vi_result.losses[-1]
    assert last_loss < first_loss, (
        f"Loss did not decrease after training: first={first_loss}, last={last_loss}"
    )
