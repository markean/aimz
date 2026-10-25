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

"""Tests for the `.sample_prior_predictive_on_batch()` method."""

import numpy as np
import pytest
from jax import Array, random

from aimz import ImpactModel
from tests.conftest import lm, lm_subsample, make_svi


def test_sample_prior_predictive_on_batch(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Prior draws follow the request and interventions; kernel errors pass through."""
    X, _ = synthetic_data
    samples = im_lm_svi_fitted.sample_prior_predictive_on_batch(
        X=X,
        num_samples=99,
        return_sites="y",
    )
    assert samples.prior_predictive["y"].values.shape == (1, 99, len(X))

    samples_dict = im_lm_svi_fitted.sample_prior_predictive_on_batch(
        X=X,
        num_samples=99,
        return_datatree=False,
        return_sites=["b", "y", "sigma"],
    )
    assert isinstance(samples_dict, dict)
    assert samples_dict["y"].shape == (99, len(X))
    assert im_lm_svi_fitted.kernel_spec.traced
    assert im_lm_svi_fitted.kernel_spec.output_observed

    # Prior interventions change downstream draws without fitting the model
    X = np.arange(24, dtype=np.float32).reshape(12, 2) / 24
    im = ImpactModel(lm, rng_key=random.key(0), inference=make_svi(lm))
    draws = []
    for value in (0.0, 1.0):
        dt = im.sample_prior_predictive_on_batch(
            X,
            intervention={"w": np.full(2, value), "b": 2 * value},
            rng_key=random.key(1),
            num_samples=9,
        )
        draws.append(dt["prior_predictive"]["y"].values)

    expected = np.broadcast_to(X.sum(axis=1) + 2, (1, 9, len(X)))
    np.testing.assert_allclose(draws[1] - draws[0], expected, atol=1e-6)

    # A new model passes on the error of a kernel that cannot run on the input
    X = np.ones((300, 2), dtype=np.float32)
    im = ImpactModel(
        lm_subsample,
        rng_key=random.key(0),
        inference=make_svi(lm_subsample),
    )
    with pytest.raises(ValueError, match="Incompatible shapes"):
        im.sample_prior_predictive_on_batch(X)
