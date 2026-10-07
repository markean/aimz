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
from numpyro.infer import MCMC, NUTS

from aimz import ImpactModel
from tests.conftest import lm, make_svi


def test_sample(synthetic_data: tuple[Array, Array]) -> None:
    """`sample` draws from the guide, or continues the chain, in the number asked."""
    X, y = synthetic_data
    num_samples = 7
    im = ImpactModel(lm, rng_key=random.key(42), inference=make_svi(lm))
    im.fit_on_batch(X, y, num_steps=10, num_samples=10, progress=False)
    posterior = im.sample(
        num_samples=num_samples, rng_key=random.key(42), return_sites="b", X=X, y=y
    ).posterior
    assert posterior.sizes["draw"] == num_samples
    samples = im.sample(
        num_samples=num_samples,
        rng_key=random.key(42),
        return_sites=["w", "b", "sigma"],
        return_datatree=False,
        X=X,
        y=y,
    )
    assert {k: v.shape[0] for k, v in samples.items()} == dict.fromkeys(
        ("w", "b", "sigma"), num_samples
    )

    num_samples_fit = 10
    mcmc = MCMC(
        NUTS(lm), num_warmup=10, num_samples=num_samples_fit, progress_bar=False
    )
    im = ImpactModel(lm, rng_key=random.key(42), inference=mcmc)
    im.fit_on_batch(X, y)
    with pytest.raises(TypeError, match="must be provided"):
        im.sample(rng_key=random.key(42), X=X)
    # The chain continues from its last state for this call only; the key is ignored
    posterior = im.sample(
        num_samples=num_samples, rng_key=random.key(42), X=X, y=y
    ).posterior
    assert posterior.sizes["draw"] == num_samples
    assert im.inference.num_samples == num_samples_fit
    samples = im.sample(num_samples=num_samples, return_datatree=False, X=X, y=y)
    assert all(v.shape[0] == num_samples for v in samples.values())
