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

"""Tests for the `.sample()` method."""

from typing import TYPE_CHECKING

import pytest
from jax import random
from jax.typing import ArrayLike
from numpyro.infer import MCMC

from aimz import ImpactModel
from tests.conftest import lm

if TYPE_CHECKING:
    from numpyro.infer import SVI


@pytest.mark.parametrize("mcmc", [lm], indirect=True)
def test_missing_param_output(
    synthetic_data: tuple[ArrayLike, ArrayLike],
    mcmc: MCMC,
) -> None:
    """Missing `param_output` argument raises TypeError."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=mcmc)
    im.fit_on_batch(X=X, y=y)
    with pytest.raises(TypeError):
        im.sample(rng_key=random.key(42), X=X)


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_sample_with_vi(
    synthetic_data: tuple[ArrayLike, ArrayLike],
    vi: "SVI",
) -> None:
    """Test the `.sample()` method of ImpactModel with SVI."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    im.fit_on_batch(X=X, y=y)

    num_samples = 7
    samples = im.sample(
        num_samples=num_samples,
        rng_key=random.key(42),
        return_sites="b",
        X=X,
        y=y,
    ).posterior

    # Check shapes for all sampled sites
    for var in samples.data_vars:
        assert samples[var].values.shape[1] == num_samples, (
            f"Incorrect number of samples for site {var}"
        )

    samples_dict = im.sample(
        num_samples=num_samples,
        rng_key=random.key(42),
        return_sites=["w", "b", "sigma"],
        return_datatree=False,
        X=X,
        y=y,
    )

    for k, v in samples_dict.items():
        assert v.shape[0] == num_samples, f"Incorrect number of samples for site {k}"


@pytest.mark.parametrize("mcmc", [lm], indirect=True)
def test_sample_with_mcmc(
    synthetic_data: tuple[ArrayLike, ArrayLike],
    mcmc: MCMC,
) -> None:
    """Test the `.sample()` method of ImpactModel with MCMC."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=mcmc)
    im.fit_on_batch(X=X, y=y)

    num_samples = 7
    # rng_key is ignored for MCMC; sampling uses the post_warmup_state
    samples = im.sample(
        num_samples=num_samples,
        rng_key=random.key(42),
        X=X,
        y=y,
    ).posterior

    assert im.inference.num_samples == num_samples

    # Check shapes for all sampled sites
    for var in samples.data_vars:
        assert samples[var].values.shape[1] == num_samples, (
            f"Incorrect number of samples for site {var}"
        )

    samples_dict = im.sample(
        num_samples=num_samples,
        rng_key=random.key(42),
        return_datatree=False,
        X=X,
        y=y,
    )

    for k, v in samples_dict.items():
        assert v.shape[0] == num_samples, f"Incorrect number of samples for site {k}"
