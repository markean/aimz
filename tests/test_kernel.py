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

"""Tests for the model kernel."""

import pytest
from jax import random
from jax.typing import ArrayLike
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from aimz._exceptions import KernelValidationError


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
class TestModelKernel:
    """Test class for kernel validation."""

    def test_kernel_with_args(self) -> None:
        """Kernel with *args raises an error."""

        def kernel(X: ArrayLike, y: ArrayLike | None = None, *args: tuple) -> None:
            pass

        with pytest.raises(KernelValidationError):
            ImpactModel(
                kernel,
                rng_key=random.key(42),
                inference=SVI(
                    kernel,
                    guide=AutoNormal(kernel),
                    optim=Adam(step_size=1e-3),
                    loss=Trace_ELBO(),
                ),
            )

    def test_kernel_with_kwargs(self) -> None:
        """Kernel with **kwargs raises an error."""

        def kernel(
            X: ArrayLike,
            y: ArrayLike | None = None,
            **kwargs: object,
        ) -> None:
            pass

        with pytest.raises(KernelValidationError):
            ImpactModel(
                kernel,
                rng_key=random.key(42),
                inference=SVI(
                    kernel,
                    guide=AutoNormal(kernel),
                    optim=Adam(step_size=1e-3),
                    loss=Trace_ELBO(),
                ),
            )

    def test_kernel_with_no_input(self) -> None:
        """Kernel without input raises an error."""

        def kernel(x: object, y: ArrayLike | None = None) -> None:
            pass

        with pytest.raises(KernelValidationError):
            ImpactModel(
                kernel,
                rng_key=random.key(42),
                inference=SVI(
                    kernel,
                    guide=AutoNormal(kernel),
                    optim=Adam(step_size=1e-3),
                    loss=Trace_ELBO(),
                ),
            )

    def test_kernel_with_no_ouput(self) -> None:
        """Kernel without output raises an error."""

        def kernel(X: ArrayLike, yy: object) -> None:
            pass

        with pytest.raises(KernelValidationError):
            ImpactModel(
                kernel,
                rng_key=random.key(42),
                inference=SVI(
                    kernel,
                    guide=AutoNormal(kernel),
                    optim=Adam(step_size=1e-3),
                    loss=Trace_ELBO(),
                ),
            )

    def test_kernel_with_default_input(self) -> None:
        """Kernel with a `None` default input parameter raises an error."""

        def kernel(X: ArrayLike | None = None, y: ArrayLike | None = None) -> None:
            pass

        with pytest.raises(KernelValidationError):
            ImpactModel(
                kernel,
                rng_key=random.key(42),
                inference=SVI(
                    kernel,
                    guide=AutoNormal(kernel),
                    optim=Adam(step_size=1e-3),
                    loss=Trace_ELBO(),
                ),
            )

    def test_kernel_with_non_default_output(self) -> None:
        """Kernel with a non-`None` default output parameter raises an error."""

        def kernel(X: ArrayLike, y: ArrayLike) -> None:
            pass

        with pytest.raises(KernelValidationError):
            ImpactModel(
                kernel,
                rng_key=random.key(42),
                inference=SVI(
                    kernel,
                    guide=AutoNormal(kernel),
                    optim=Adam(step_size=1e-3),
                    loss=Trace_ELBO(),
                ),
            )
