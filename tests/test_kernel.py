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

import warnings
from collections.abc import Callable

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from jax.typing import ArrayLike
from numpyro import sample
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from aimz._exceptions import KernelValidationError


def lm_dual_branch(X: Array, y: Array | None = None) -> None:
    """Dual-branch kernel: output-named factor when scoring, sampling otherwise."""
    n_features = X.shape[1]
    w = sample("w", dist.Normal(jnp.zeros(n_features), jnp.ones(n_features)))
    sigma = sample("sigma", dist.Exponential(1.0))
    mu = numpyro.deterministic("mu", jnp.dot(X, w))
    if y is not None:
        with numpyro.plate("data", X.shape[0]):
            numpyro.factor("y", dist.Normal(mu, sigma).log_prob(y))
    else:
        with numpyro.plate("data", X.shape[0]):
            z = sample("z", dist.Bernoulli(logits=mu))
            numpyro.deterministic("y", z * mu)


def _make_im(model: Callable) -> ImpactModel:

    return ImpactModel(
        model,
        rng_key=random.key(42),
        inference=SVI(
            model,
            guide=AutoNormal(model),
            optim=Adam(step_size=1e-3),
            loss=Trace_ELBO(),
        ),
    )


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


class TestDualBranchKernel:
    """Prior predictive sampling and site visibility for dual-branch kernels."""

    def test_sample_prior_predictive_before_fit(
        self,
        synthetic_data: tuple[Array, Array],
    ) -> None:
        """The deterministic output of the data-free branch validates before a fit."""
        X, _ = synthetic_data
        im = _make_im(lm_dual_branch)
        dt = im.sample_prior_predictive_on_batch(X, num_samples=10)
        assert "y" in dt["prior_predictive"]
        # The output leads the default return sites and is listed once.
        assert im.kernel_spec.return_sites == ("y", "mu")

    def test_fit_merges_spec_and_returns_sampling_site(
        self,
        synthetic_data: tuple[Array, Array],
    ) -> None:
        """Sites discovered before fitting stay known and requestable after it."""
        X, y = synthetic_data
        im = _make_im(lm_dual_branch)
        im.sample_prior_predictive_on_batch(X, num_samples=5)
        im.fit_on_batch(X, y, num_steps=50, num_samples=20, progress=False)
        assert "z" in im.kernel_spec.sample_sites
        assert im.kernel_spec.return_sites == ("y", "mu")
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*not seen in any trace.*")
            dt = im.predict_on_batch(X, return_sites=["z"])
        assert "z" in dt["posterior_predictive"]
