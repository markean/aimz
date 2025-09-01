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

"""Tests for the `.log_likelihood()` method."""

import pytest
from conftest import lm
from jax import random
from jax.typing import ArrayLike
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from aimz._exceptions import NotFittedError


def test_model_not_fitted() -> None:
    """Calling `.log_likelihood()` on an unfitted model raises an error."""

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
        im.log_likelihood(None, None)


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
            im.log_likelihood(X=X, y=y)
