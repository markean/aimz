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
from functools import partial

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import deterministic, handlers, sample
from numpyro.infer import SVI

from aimz import ImpactModel, KernelValidationError, OutputWarning
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

    def positional_only(X: Array, /, y: Array | None = None) -> None:
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
        positional_only,
        without_input,
        without_output,
        input_with_default,
        output_without_default,
    ):
        with pytest.raises(KernelValidationError):
            ImpactModel(kernel, rng_key=random.key(42), inference=make_svi(kernel))

    @handlers.scale(scale=1.0)
    @handlers.condition(data={})
    def decorated(X: Array, y: Array | None = None) -> None:
        lm(X, y)

    kernel = partial(decorated)
    im = ImpactModel(kernel, rng_key=random.key(42), inference=make_svi(kernel))
    assert im.describe()["arguments"] == ["X", "y"]
    with pytest.raises(TypeError, match="Unsupported inference object"):
        ImpactModel(lm, rng_key=random.key(42), inference=None)
    vi = make_svi(lm)
    with pytest.raises(ValueError, match="static keyword arguments"):
        ImpactModel(
            lm,
            rng_key=random.key(42),
            inference=SVI(lm, vi.guide, vi.optim, vi.loss, sd=1.0),
        )


def test_kernel_body_validation(synthetic_data: tuple[Array, Array]) -> None:
    """A kernel with an unusable site name or output site is rejected when traced."""
    X, y = synthetic_data

    def without_output_site(X: Array, y: Array | None = None) -> None:
        sample("z", dist.Normal(0.0, 1.0), obs=y)

    def deterministic_output(X: Array, y: Array | None = None) -> None:
        deterministic("y", jnp.zeros_like(y))

    def unobserved_output(X: Array, y: Array | None = None) -> None:
        sample("y", dist.Normal(0.0, 1.0))

    def site_with_slash(X: Array, y: Array | None = None) -> None:
        mu = sample("x/y", dist.Normal())
        sample("y", dist.Normal(mu, 1.0), obs=y)

    def site_named_draw(X: Array, y: Array | None = None) -> None:
        mu = sample("draw", dist.Normal())
        sample("y", dist.Normal(mu, 1.0), obs=y)

    for kernel in (
        without_output_site,
        deterministic_output,
        unobserved_output,
        site_with_slash,
        site_named_draw,
    ):
        im = ImpactModel(kernel, rng_key=random.key(42), inference=make_svi(kernel))
        with pytest.raises(KernelValidationError):
            im.fit(X, y, batch_size=len(X), progress=False)

    # `obs_mask` records the output as a deterministic site
    def masked_output(X: Array, y: Array | None = None) -> None:
        mask = jnp.ones(len(X), dtype=bool)
        with numpyro.plate("obs", X.shape[0]):
            sample("y", dist.Normal(0.0, 1.0), obs=y, obs_mask=mask)

    im = ImpactModel(
        masked_output,
        rng_key=random.key(42),
        inference=make_svi(masked_output),
    )
    with pytest.raises(KernelValidationError, match="mask its distribution instead"):
        im.fit(X, y, batch_size=len(X), progress=False)

    # The prior predictive trace needs a sample or deterministic output site
    def delta_only(X: Array, y: Array | None = None) -> None:
        sample("z", dist.Delta(y if y is not None else jnp.zeros(len(X))), obs=y)

    im = ImpactModel(delta_only, rng_key=random.key(42), inference=make_svi(delta_only))
    with pytest.raises(KernelValidationError):
        im.sample_prior_predictive_on_batch(X)

    # A plate named like an output dimension gives way to the default names
    def plate_named_chain(X: Array, y: Array | None = None) -> None:
        with numpyro.plate("chain", X.shape[0]):
            sample("y", dist.Normal(0.0, 1.0), obs=y)

    im = ImpactModel(
        plate_named_chain,
        rng_key=random.key(42),
        inference=make_svi(plate_named_chain),
    )
    with pytest.warns(OutputWarning, match="plate name 'chain'"):
        dt = im.sample_prior_predictive_on_batch(X, num_samples=10)
    assert dt.prior_predictive["y"].dims == ("chain", "draw", "y_dim_0")


def test_dual_branch_kernel_sites(synthetic_data: tuple[Array, Array]) -> None:
    """Sites found before a fit stay known and requestable after it."""
    X, y = synthetic_data
    im = ImpactModel(
        lm_dual_branch,
        rng_key=random.key(42),
        inference=make_svi(lm_dual_branch),
    )
    # The data-free branch validates before a fit, its output listed once
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
