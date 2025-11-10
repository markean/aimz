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

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
import xarray as xr
from jax import random
from jax.typing import ArrayLike
from numpyro import sample
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from aimz._exceptions import KernelValidationError
from tests.conftest import lm


def test_kernel_without_output(synthetic_data: tuple[ArrayLike, ArrayLike]) -> None:
    """Kernel without output sample site raises an error."""

    def kernel(X: ArrayLike, y: ArrayLike | None = None) -> None:
        sample("z", dist.Delta(y if y is not None else jnp.zeros(len(X))), obs=y)

    X, _ = synthetic_data
    im = ImpactModel(
        kernel,
        rng_key=random.key(42),
        inference=SVI(
            kernel,
            guide=AutoNormal(kernel),
            optim=Adam(step_size=1e-3),
            loss=Trace_ELBO(),
        ),
    )

    with pytest.raises(KernelValidationError):
        im.sample_prior_predictive_on_batch(X)


@pytest.mark.parametrize("vi", [lm], indirect=True)
class TestKernelParameterValidation:
    """Test class for validating parameter compatibility with the kernel."""

    def test_invalid_parameter(
        self,
        synthetic_data: tuple[ArrayLike, ArrayLike],
        vi: "SVI",
    ) -> None:
        """An invalid parameter raise an error."""
        X, y = synthetic_data
        im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
        with pytest.raises(TypeError):
            im.sample_prior_predictive_on_batch(X=X, y=y)


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_sample_prior_predictive_on_batch_lm(
    synthetic_data: tuple[ArrayLike, ArrayLike],
    vi: "SVI",
) -> None:
    """Test the `.sample_prior_predictive_on_batch()` method of ImpactModel."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    im.fit_on_batch(X, y)
    samples = im.sample_prior_predictive_on_batch(X=X, num_samples=99, return_sites="y")

    assert isinstance(samples, xr.DataTree)
    assert samples.prior_predictive["y"].values.shape == (1, 99, len(X))

    samples_dict = im.sample_prior_predictive_on_batch(
        X=X,
        num_samples=99,
        return_datatree=False,
        return_sites=["b", "y", "sigma"],
    )

    assert isinstance(samples_dict, dict)
    assert samples_dict["y"].shape == (99, len(X))
    assert im.kernel_spec.traced
    assert im.kernel_spec.output_observed
