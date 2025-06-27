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

"""Tests for the `.fit()` method."""

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from conftest import lm
from jax import random
from numpyro import deterministic, sample
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
from sklearn.exceptions import DataConversionWarning

from aimz._exceptions import KernelValidationError
from aimz.model import ImpactModel
from aimz.utils._validation import _is_fitted

if TYPE_CHECKING:
    import jax


class TestKernelSignatureValidation:
    """Test class for validating parameter compatibility with the kernel signature."""

    def test_extra_parameters(self) -> None:
        """Extra parameters not present in the kernel raise an error."""

        def kernel(X: "jax.Array", y: "jax.Array | None" = None) -> None:
            pass

        im = ImpactModel(
            kernel,
            rng_key=random.key(42),
            vi=SVI(
                kernel,
                guide=AutoNormal(kernel),
                optim=Adam(step_size=1e-3),
                loss=Trace_ELBO(),
            ),
        )
        with pytest.raises(TypeError):
            im.fit(X=jnp.ones((10, 1)), y=jnp.ones((10,)), extra=True)

    def test_missing_parameters(self) -> None:
        """Missing required parameters in the kernel raise an error."""

        def kernel(X: "jax.Array", arg: object, y: "jax.Array | None" = None) -> None:
            pass

        im = ImpactModel(
            kernel,
            rng_key=random.key(42),
            vi=SVI(
                kernel,
                guide=AutoNormal(kernel),
                optim=Adam(step_size=1e-3),
                loss=Trace_ELBO(),
            ),
        )
        with pytest.raises(TypeError):
            im.fit(X=jnp.ones((10, 1)), y=jnp.ones((10,)))


class TestKernelBodyValidation:
    """Test class for validating parameter compatibility with the kernel body."""

    def test_missing_output_site(self) -> None:
        """Missing output sample site in the kernel raises an error."""

        def kernel(X: "jax.Array", y: "jax.Array | None" = None) -> None:
            sample("z", dist.Normal(0.0, 1.0), obs=y)

        im = ImpactModel(
            kernel,
            rng_key=random.key(42),
            vi=SVI(
                kernel,
                guide=AutoNormal(kernel),
                optim=Adam(step_size=1e-3),
                loss=Trace_ELBO(),
            ),
        )
        with pytest.raises(KernelValidationError):
            im.fit(X=jnp.ones((10, 1)), y=jnp.ones((10,)))

    def test_sample_output_site(self) -> None:
        """Raises error if output site is not a sample site."""

        def kernel(X: "jax.Array", y: "jax.Array | None" = None) -> None:
            deterministic("y", jnp.zeros_like(y))

        im = ImpactModel(
            kernel,
            rng_key=random.key(42),
            vi=SVI(
                kernel,
                guide=AutoNormal(kernel),
                optim=Adam(step_size=1e-3),
                loss=Trace_ELBO(),
            ),
        )
        with pytest.raises(KernelValidationError):
            im.fit(X=jnp.ones((10, 1)), y=jnp.ones((10,)))

    def test_unobserved_output_site(self) -> None:
        """Raises error if output site is not observed."""

        def kernel(X: "jax.Array", y: "jax.Array | None" = None) -> None:
            sample("y", dist.Normal(0.0, 1.0))

        im = ImpactModel(
            kernel,
            rng_key=random.key(42),
            vi=SVI(
                kernel,
                guide=AutoNormal(kernel),
                optim=Adam(step_size=1e-3),
                loss=Trace_ELBO(),
            ),
        )
        with pytest.raises(KernelValidationError):
            im.fit(X=jnp.ones((10, 1)), y=jnp.ones((10,)))

    def test_parameter_site_conflict(self) -> None:
        """Raises an error if a parameter name conflicts with a site name."""

        def kernel(X: "jax.Array", y: "jax.Array | None" = None) -> None:
            sample("X", dist.Normal(0.0, 1.0))
            sample("y", dist.Normal(0.0, 1.0), obs=y)

        im = ImpactModel(
            kernel,
            rng_key=random.key(42),
            vi=SVI(
                kernel,
                guide=AutoNormal(kernel),
                optim=Adam(step_size=1e-3),
                loss=Trace_ELBO(),
            ),
        )
        with pytest.raises(KernelValidationError):
            im.fit(X=jnp.ones((10, 1)), y=jnp.ones((10,)))


def test_fit_unexpected_y_shape() -> None:
    """Calling `.fit()` with an unexpected shape of `y` raises a warning."""

    def kernel(X: "jax.Array", y: "jax.Array | None" = None) -> None:
        sample("y", dist.Normal(0.0, 1.0), obs=y)

    im = ImpactModel(
        kernel,
        rng_key=random.key(42),
        vi=SVI(
            kernel,
            guide=AutoNormal(kernel),
            optim=Adam(step_size=1e-3),
            loss=Trace_ELBO(),
        ),
    )
    with pytest.warns(DataConversionWarning):
        im.fit(X=jnp.zeros((10, 2)), y=jnp.zeros((10, 1)))


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_fit_lm(synthetic_data: tuple["jax.Array", "jax.Array"], vi: SVI) -> None:
    """Test the `.fit()` method of `ImpactModel`."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), vi=vi)
    im.fit(X=X, y=y)
    assert _is_fitted(im), "Model fitting check failed"
