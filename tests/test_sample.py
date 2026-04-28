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

import pytest
from jax import Array, random

from aimz import ImpactModel


def test_missing_param_output(
    synthetic_data: tuple[Array, Array],
    im_lm_mcmc_fitted: ImpactModel,
) -> None:
    """Missing `param_output` argument raises TypeError."""
    X, _ = synthetic_data
    with pytest.raises(TypeError):
        im_lm_mcmc_fitted.sample(rng_key=random.key(42), X=X)


def test_sample_with_vi(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Test the `.sample()` method of ImpactModel with SVI."""
    X, y = synthetic_data
    num_samples = 7
    samples = im_lm_svi_fitted.sample(
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

    samples_dict = im_lm_svi_fitted.sample(
        num_samples=num_samples,
        rng_key=random.key(42),
        return_sites=["w", "b", "sigma"],
        return_datatree=False,
        X=X,
        y=y,
    )

    for k, v in samples_dict.items():
        assert v.shape[0] == num_samples, f"Incorrect number of samples for site {k}"


def test_sample_with_mcmc(
    synthetic_data: tuple[Array, Array],
    im_lm_mcmc_fitted: ImpactModel,
) -> None:
    """Test the `.sample()` method of ImpactModel with MCMC."""
    X, y = synthetic_data
    num_samples = 7
    # rng_key is ignored for MCMC; sampling uses the post_warmup_state
    samples = im_lm_mcmc_fitted.sample(
        num_samples=num_samples,
        rng_key=random.key(42),
        X=X,
        y=y,
    ).posterior

    assert im_lm_mcmc_fitted.inference.num_samples == num_samples

    # Check shapes for all sampled sites
    for var in samples.data_vars:
        assert samples[var].values.shape[1] == num_samples, (
            f"Incorrect number of samples for site {var}"
        )

    samples_dict = im_lm_mcmc_fitted.sample(
        num_samples=num_samples,
        rng_key=random.key(42),
        return_datatree=False,
        X=X,
        y=y,
    )

    for k, v in samples_dict.items():
        assert v.shape[0] == num_samples, f"Incorrect number of samples for site {k}"
