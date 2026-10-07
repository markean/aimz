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

"""Tests for kernel validation and site discovery."""

import warnings

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import deterministic, sample

from aimz import ImpactModel, KernelValidationError
from tests.conftest import lm, make_svi


def lm_dual_branch(X: Array, y: Array | None = None) -> None:
    """Dual-branch kernel: output-named factor when scoring, sampling otherwise."""
    n_features = X.shape[1]
    w = sample("w", dist.Normal(jnp.zeros(n_features), jnp.ones(n_features)))
    sigma = sample("sigma", dist.Exponential(1.0))
    mu = deterministic("mu", jnp.dot(X, w))
    if y is not None:
        with numpyro.plate("data", X.shape[0]):
            numpyro.factor("y", dist.Normal(mu, sigma).log_prob(y))
    else:
        with numpyro.plate("data", X.shape[0]):
            z = sample("z", dist.Bernoulli(logits=mu))
            deterministic("y", z * mu)


def test_kernel_signature_validation() -> None:
    """A kernel with an unusable signature, or an unsupported inference, is rejected."""

    def with_args(X: Array, y: Array | None = None, *args: object) -> None:
        pass

    def with_kwargs(X: Array, y: Array | None = None, **kwargs: object) -> None:
        pass

    def without_input(x: Array, y: Array | None = None) -> None:
        pass

    def without_output(X: Array, yy: object) -> None:
        pass

    def input_with_default(X: Array | None = None, y: Array | None = None) -> None:
        pass

    def output_without_default(X: Array, y: Array) -> None:
        pass

    for kernel in (
        with_args,
        with_kwargs,
        without_input,
        without_output,
        input_with_default,
        output_without_default,
    ):
        with pytest.raises(KernelValidationError):
            ImpactModel(kernel, rng_key=random.key(42), inference=make_svi(kernel))
    with pytest.raises(TypeError, match="Unsupported inference object"):
        ImpactModel(lm, rng_key=random.key(42), inference=None)


def test_kernel_body_validation(synthetic_data: tuple[Array, Array]) -> None:
    """A kernel whose trace lacks a usable output site is rejected at the first call."""
    X, y = synthetic_data

    def without_output_site(X: Array, y: Array | None = None) -> None:
        sample("z", dist.Normal(0.0, 1.0), obs=y)

    def deterministic_output(X: Array, y: Array | None = None) -> None:
        deterministic("y", jnp.zeros_like(y))

    def unobserved_output(X: Array, y: Array | None = None) -> None:
        sample("y", dist.Normal(0.0, 1.0))

    def site_named_as_argument(X: Array, y: Array | None = None) -> None:
        sample("X", dist.Normal(0.0, 1.0))
        sample("y", dist.Normal(0.0, 1.0), obs=y)

    def site_with_slash(X: Array, y: Array | None = None) -> None:
        mu = sample("x/y", dist.Normal())
        sample("y", dist.Normal(mu, 1.0), obs=y)

    for kernel in (
        without_output_site,
        deterministic_output,
        unobserved_output,
        site_named_as_argument,
        site_with_slash,
    ):
        im = ImpactModel(kernel, rng_key=random.key(42), inference=make_svi(kernel))
        with pytest.raises(KernelValidationError):
            im.fit(X, y, batch_size=len(X), progress=False)

    # Without an output, the prior predictive trace still needs a sample or
    # deterministic site named after it
    def delta_only(X: Array, y: Array | None = None) -> None:
        sample("z", dist.Delta(y if y is not None else jnp.zeros(len(X))), obs=y)

    im = ImpactModel(delta_only, rng_key=random.key(42), inference=make_svi(delta_only))
    with pytest.raises(KernelValidationError):
        im.sample_prior_predictive_on_batch(X)


def test_dual_branch_kernel_sites(synthetic_data: tuple[Array, Array]) -> None:
    """Sites found before a fit stay known and requestable after it."""
    X, y = synthetic_data
    im = ImpactModel(
        lm_dual_branch,
        rng_key=random.key(42),
        inference=make_svi(lm_dual_branch),
    )
    # The deterministic output of the data-free branch validates before a fit and
    # leads the default return sites once
    dt = im.sample_prior_predictive_on_batch(X, num_samples=10)
    assert "y" in dt["prior_predictive"]
    assert im.kernel_spec.return_sites == ("y", "mu")

    im.fit_on_batch(X, y, num_steps=50, num_samples=20, progress=False)
    assert "z" in im.kernel_spec.sample_sites
    assert im.kernel_spec.return_sites == ("y", "mu")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*not seen in any trace.*")
        dt = im.predict_on_batch(X, return_sites=["z"])
    assert "z" in dt["posterior_predictive"]
