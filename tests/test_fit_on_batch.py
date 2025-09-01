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

import pytest
from conftest import lm
from jax import random
from jax.typing import ArrayLike
from numpyro.infer import MCMC, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from aimz.utils._validation import _is_fitted


def test_fit_on_batch_nan_warning(synthetic_data: tuple[ArrayLike, ArrayLike]) -> None:
    """Test that `.fit_on_batch()` emits a RuntimeWarning when NaN is in the loss."""
    X, y = synthetic_data
    im = ImpactModel(
        lm,
        rng_key=random.key(42),
        inference=SVI(
            lm,
            guide=AutoNormal(lm),
            optim=Adam(step_size=1e3),
            loss=Trace_ELBO(),
        ),
    )
    with pytest.warns(RuntimeWarning):
        im.fit_on_batch(X, y)


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_fit_svi(synthetic_data: tuple[ArrayLike, ArrayLike], vi: SVI) -> None:
    """Test the `.fit()` method of ImpactModel using SVI."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
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


@pytest.mark.parametrize("mcmc", [lm], indirect=True)
def test_fit_mcmc(synthetic_data: tuple[ArrayLike, ArrayLike], mcmc: MCMC) -> None:
    """Test the `.fit()` method of ImpactModel using MCMC."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=mcmc)
    im.fit_on_batch(X=X, y=y)
    assert _is_fitted(im), "Model fitting check failed"
