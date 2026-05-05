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

import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import sample
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from aimz._exceptions import NotFittedError
from tests.conftest import mlm


def test_model_not_fitted() -> None:
    """Calling `.predict_on_batch()` on an unfitted model raises an error."""

    def kernel(X: Array, y: Array | None = None) -> None:
        pass

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
    with pytest.raises(NotFittedError):
        im.predict_on_batch(None)


class TestKernelParameterValidation:
    """Test class for validating parameter compatibility with the kernel."""

    def test_invalid_parameter(
        self,
        synthetic_data: tuple[Array, Array],
        im_lm_svi_fitted: ImpactModel,
    ) -> None:
        """An invalid parameter raise an error."""
        X, y = synthetic_data
        with pytest.raises(TypeError):
            im_lm_svi_fitted.predict_on_batch(X=X, y=y)

    def test_extra_parameters(
        self,
        synthetic_data: tuple[Array, Array],
        im_lm_svi_fitted: ImpactModel,
    ) -> None:
        """Extra parameters not present in the kernel raise an error."""
        X, y = synthetic_data
        with pytest.raises(TypeError):
            im_lm_svi_fitted.predict_on_batch(X=X, y=y, extra=True)

    def test_missing_parameters(self, synthetic_data: tuple[Array, Array]) -> None:
        """Missing required parameters in the kernel raise an error."""
        X, y = synthetic_data
        arg = True

        def kernel(X: Array, arg: object, y: Array | None = None) -> None:
            sample("y", dist.Normal(0.0, 1.0), obs=y)

        vi = SVI(
            kernel,
            guide=AutoNormal(kernel),
            optim=Adam(step_size=1e-3),
            loss=Trace_ELBO(),
        )
        im = ImpactModel(kernel, rng_key=random.key(42), inference=vi)
        im.fit(X=X, arg=arg, y=y, batch_size=3)
        with pytest.raises(TypeError):
            im.predict_on_batch(X=X)


def test_predict_on_batch_lm_with_kwargs_array(
    synthetic_data: tuple[Array, Array],
    im_lm_with_kwargs_svi_fitted: ImpactModel,
) -> None:
    """Test the `.predict_on_batch()` method of ImpactModel."""
    X, y = synthetic_data
    im_lm_with_kwargs_svi_fitted.predict_on_batch(X=X, c=y, return_sites="y")

    # `.sample_posterior_predictive_on_batch()` is an alias for `.predict_on_batch()`.
    im_lm_with_kwargs_svi_fitted.sample_posterior_predictive_on_batch(
        X=X,
        c=y,
        return_sites=["y"],
        return_datatree=False,
    )


def test_predict_on_batch_x_zero_dim_raises(
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """A 0-dimensional ``X`` raises ``ValueError``."""
    with pytest.raises(ValueError, match=r"`X` must have at least 1 dimension."):
        im_lm_svi_fitted.predict_on_batch(X=1.0)


def test_predict_on_batch_mlm() -> None:
    """`.predict_on_batch()` works with a multivariate linear regression model."""
    n_obs, n_features, n_targets = 100, 3, 2
    rng_key = random.key(42)
    rng_key, rng_subkey = random.split(rng_key)
    X = random.normal(rng_subkey, (n_obs, n_features))
    rng_key, rng_subkey = random.split(rng_key)
    w = random.normal(rng_subkey, (n_features, n_targets))
    rng_key, rng_subkey = random.split(rng_key)
    e = random.normal(rng_subkey, (n_obs, n_targets))
    y = X @ w + e

    rng_key, rng_subkey = random.split(rng_key)
    im = ImpactModel(
        mlm,
        rng_key=rng_subkey,
        inference=SVI(
            mlm,
            guide=AutoNormal(mlm),
            optim=Adam(step_size=1e-2),
            loss=Trace_ELBO(),
        ),
    )
    im.fit_on_batch(X=X, y=y)
    out = im.predict_on_batch(X=X)

    assert out["posterior_predictive"]["y"].sizes["y_dim_0"] == n_obs
    assert out["posterior_predictive"]["y"].sizes["y_dim_1"] == n_targets
