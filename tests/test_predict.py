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

"""Tests for the `.predict()` method."""

import numpyro.distributions as dist
import pytest
from conftest import latent_variable_model, lm
from jax import random
from jax.typing import ArrayLike
from numpyro import sample
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from aimz._exceptions import NotFittedError


def test_model_not_fitted() -> None:
    """Calling `.predict()` on an unfitted model raises an error."""

    def kernel(X: ArrayLike, y: ArrayLike | None = None) -> None:
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
        im.predict(None)


@pytest.mark.parametrize("vi", [latent_variable_model], indirect=True)
def test_predict_fall_back(
    synthetic_data: tuple[ArrayLike, ArrayLike],
    vi: SVI,
) -> None:
    """Calling `.predict()` warns and falls back on an incompatible model."""
    X, y = synthetic_data
    im = ImpactModel(latent_variable_model, rng_key=random.key(42), inference=vi)
    im.fit(X=X, y=y, batch_size=len(X))
    msg = "One or more posterior sample shapes are not compatible"
    with pytest.warns(UserWarning, match=msg):
        im.predict(X=X, batch_size=len(X), progress=False)


class TestKernelParameterValidation:
    """Test class for validating parameter compatibility with the kernel."""

    @pytest.mark.parametrize("vi", [lm], indirect=True)
    def test_invalid_parameter(
        self,
        synthetic_data: tuple[ArrayLike, ArrayLike],
        vi: SVI,
    ) -> None:
        """An invalid parameter raise an error."""
        X, y = synthetic_data
        im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
        im.fit(X=X, y=y, batch_size=3)
        with pytest.raises(TypeError):
            im.predict(X=X, y=y)

    @pytest.mark.parametrize("vi", [lm], indirect=True)
    def test_extra_parameters(
        self,
        synthetic_data: tuple[ArrayLike, ArrayLike],
        vi: SVI,
    ) -> None:
        """Extra parameters not present in the kernel raise an error."""
        X, y = synthetic_data
        im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
        im.fit(X=X, y=y, batch_size=3)
        with pytest.raises(TypeError):
            im.predict(X=X, extra=True)

    def test_missing_parameters(
        self,
        synthetic_data: tuple[ArrayLike, ArrayLike],
    ) -> None:
        """Missing required parameters in the kernel raise an error."""
        X, y = synthetic_data
        arg = True

        def kernel(X: ArrayLike, arg: object, y: ArrayLike | None = None) -> None:
            sample("y", dist.Normal(0.0, 1.0), obs=y)

        vi = SVI(
            kernel,
            guide=AutoNormal(kernel),
            optim=Adam(step_size=1e-3),
            loss=Trace_ELBO(),
        )
        im = ImpactModel(kernel, rng_key=random.key(42), inference=vi)
        im.fit(X=X, y=y, arg=arg, batch_size=3)
        with pytest.raises(TypeError):
            im.predict(X=X)


class TestBatchSize:
    """Test class related to batch size specification."""

    @pytest.mark.parametrize("vi", [lm], indirect=True)
    def test_default_batch_size(
        self,
        synthetic_data: tuple[ArrayLike, ArrayLike],
        vi: SVI,
    ) -> None:
        """Warns if `batch_size` is not explicitly set."""
        X, y = synthetic_data
        im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
        im.fit(X=X, y=y, batch_size=3)
        # NOTE: An additional warning about batch size not being divisible by the number
        # of devices may also be raised.
        with pytest.warns(UserWarning, match=".*"):
            im.predict(X=X, progress=False)


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_predict_after_cleanup(
    synthetic_data: tuple[ArrayLike, ArrayLike],
    vi: SVI,
) -> None:
    """Test `.predict()` recreates tempdir after `.cleanup()`."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    # NOTE: An additional warning about batch size not being divisible by the number
    # of devices may also be raised.
    with pytest.warns(UserWarning, match=".*"):
        im.fit(X=X, y=y)
    im.fit(X=X, y=y, batch_size=3)
    msg = (
        r"The `batch_size` \(\d+\) is not divisible by the number of devices \(\d+\)\."
    )
    with pytest.warns(UserWarning, match=msg):
        im.predict(X=X, batch_size=len(X) // 2, progress=False)
    temp_dir_before = im.temp_dir.name
    im.cleanup()
    with pytest.warns(UserWarning, match=msg):
        im.predict(X=X, batch_size=len(X) // 2, progress=False)
    temp_dir_after = im.temp_dir.name

    assert temp_dir_before != temp_dir_after

    # `.sample_posterior_predictive()` is an alias for `.predict()`
    im.cleanup()
    with pytest.warns(UserWarning, match=msg):
        im.sample_posterior_predictive(X=X, batch_size=len(X) // 2, progress=False)
