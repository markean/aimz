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

"""Tests for the `.predict_on_batch()` method."""

from typing import TYPE_CHECKING

import numpyro.distributions as dist
import pytest
from conftest import lm, lm_with_kwargs_array
from jax import random
from numpyro import sample
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz._exceptions import NotFittedError
from aimz.model import ImpactModel

if TYPE_CHECKING:
    import jax


def test_model_not_fitted() -> None:
    """Calling `.predict_on_batch()` on an unfitted model raises an error."""

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
    with pytest.raises(NotFittedError):
        im.predict_on_batch(None)


class TestKernelParameterValidation:
    """Test class for validating parameter compatibility with the kernel."""

    @pytest.mark.parametrize("vi", [lm], indirect=True)
    def test_invalid_parameter(
        self,
        synthetic_data: tuple["jax.Array", "jax.Array"],
        vi: SVI,
    ) -> None:
        """An invalid parameter raise an error."""
        X, y = synthetic_data
        im = ImpactModel(lm, rng_key=random.key(42), vi=vi)
        im.fit(X=X, y=y, progress=False)
        with pytest.raises(TypeError):
            im.predict_on_batch(X=X, y=y)

    @pytest.mark.parametrize("vi", [lm], indirect=True)
    def test_extra_parameters(
        self,
        synthetic_data: tuple["jax.Array", "jax.Array"],
        vi: SVI,
    ) -> None:
        """Extra parameters not present in the kernel raise an error."""
        X, y = synthetic_data
        im = ImpactModel(lm, rng_key=random.key(42), vi=vi)
        im.fit(X=X, y=y, progress=False)
        with pytest.raises(TypeError):
            im.predict_on_batch(X=X, y=y, extra=True)

    def test_missing_parameters(
        self,
        synthetic_data: tuple["jax.Array", "jax.Array"],
    ) -> None:
        """Missing required parameters in the kernel raise an error."""
        X, y = synthetic_data
        arg = True

        def kernel(X: "jax.Array", arg: object, y: "jax.Array | None" = None) -> None:
            sample("y", dist.Normal(0.0, 1.0), obs=y)

        vi = SVI(
            kernel,
            guide=AutoNormal(kernel),
            optim=Adam(step_size=1e-3),
            loss=Trace_ELBO(),
        )
        im = ImpactModel(kernel, rng_key=random.key(42), vi=vi)
        im.fit(X=X, arg=arg, y=y, progress=False)
        with pytest.raises(TypeError):
            im.predict_on_batch(X=X)


@pytest.mark.parametrize("vi", [lm_with_kwargs_array], indirect=True)
def test_predict_on_batch_lm_with_kwargs_array(
    synthetic_data: tuple["jax.Array", "jax.Array"],
    vi: SVI,
) -> None:
    """Test the `.predict_on_batch()` method of `ImpactModel`."""
    X, y = synthetic_data
    im = ImpactModel(lm_with_kwargs_array, rng_key=random.key(42), vi=vi)
    im.fit(X=X, y=y, c=y, progress=False)
    im.predict_on_batch(X=X, c=y)
