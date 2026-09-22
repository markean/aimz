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
import numpy as np
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import sample
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from aimz._exceptions import KernelValidationError
from tests.conftest import _make_svi, lm


def test_kernel_without_output(synthetic_data: tuple[Array, Array]) -> None:
    """Kernel without output sample site raises an error."""

    def kernel(X: Array, y: Array | None = None) -> None:
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
        synthetic_data: tuple[Array, Array],
        vi: "SVI",
    ) -> None:
        """An invalid parameter raise an error."""
        X, y = synthetic_data
        im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
        with pytest.raises(TypeError):
            im.sample_prior_predictive_on_batch(X=X, y=y)


def test_sample_prior_predictive_on_batch_lm(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Test the `.sample_prior_predictive_on_batch()` method of ImpactModel."""
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


@pytest.mark.parametrize("return_datatree", [True, False])
def test_sample_prior_predictive_on_batch_intervention(
    *,
    return_datatree: bool,
) -> None:
    """Prior interventions change downstream draws without fitting the model."""
    X = np.arange(24, dtype=np.float32).reshape(12, 2) / 24
    im = ImpactModel(lm, rng_key=random.key(0), inference=_make_svi(lm))
    draws = []
    for value in (0.0, 1.0):
        samples = im.sample_prior_predictive_on_batch(
            X,
            intervention={"w": np.full(2, value), "b": 2 * value},
            rng_key=random.key(1),
            num_samples=9,
            return_datatree=return_datatree,
        )
        draws.append(
            samples["y"]
            if isinstance(samples, dict)
            else samples["prior_predictive"]["y"].values
        )

    shape = (1, 9, len(X)) if return_datatree else (9, len(X))
    expected = np.broadcast_to(X.sum(axis=1) + 2, shape)
    np.testing.assert_allclose(draws[1] - draws[0], expected, atol=1e-6)
