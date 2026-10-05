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

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import sample
from numpyro.infer import MCMC, NUTS

from aimz import ImpactModel, PerformanceWarning


def test_empty_posterior(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty posterior produces a single-draw log-likelihood without crashing."""
    X, y = synthetic_data
    monkeypatch.setattr(im_lm_svi_fitted, "_posterior", None)

    out = im_lm_svi_fitted.log_likelihood(X=X, y=y, progress=False)

    assert out.log_likelihood["y"].sizes["draw"] == 1


def test_subsampled_kernel(
    synthetic_data: tuple[Array, Array],
    im_lm_subsample_svi_fitted: ImpactModel,
) -> None:
    """A kernel subsampling inside a plate is evaluated without an rng key.

    Subsample indices are pinned deterministically rather than drawn at random, so
    the bare (unseeded) kernel traces cleanly and every passed observation is scored
    on both sharding paths.
    """
    X, y = synthetic_data
    im = im_lm_subsample_svi_fitted

    out = im.log_likelihood(X=X, y=y, batch_size=30, progress=False)
    out_draw = im.log_likelihood(
        X=X,
        y=y,
        shard_axis="draw",
        batch_size=250,
        progress=False,
    )

    assert out.log_likelihood["y"].shape == (1, 1000, len(X))
    assert np.isfinite(out.log_likelihood["y"].values).all()
    # Plate indices never gather here, so the sharding paths agree.
    assert np.allclose(
        out.log_likelihood["y"].values,
        out_draw.log_likelihood["y"].values,
        atol=1e-6,
    )


def test_subsampled_kernel_batch_exceeds_plate(
    synthetic_data: tuple[Array, Array],
    im_lm_subsample_svi_fitted: ImpactModel,
) -> None:
    """A batch larger than the declared plate size is rejected loudly."""
    X, y = synthetic_data
    X_big, y_big = jnp.tile(X, (6, 1)), jnp.tile(y, 6)

    with pytest.raises(ValueError, match="declares size=100"):
        im_lm_subsample_svi_fitted.log_likelihood(
            X=X_big,
            y=y_big,
            batch_size=600,
            progress=False,
        )


def test_log_likelihood_requires_y(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """`log_likelihood()` requires `y` for an array `X`."""
    X, _ = synthetic_data
    with pytest.raises(ValueError, match="`y` is required"):
        im_lm_svi_fitted.log_likelihood(X, progress=False)


def test_log_likelihood_requires_output_field(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """`log_likelihood()` requires the output field in every loader batch."""
    X, _ = synthetic_data
    with pytest.raises(ValueError, match="requires the observed output field 'y'"):
        im_lm_svi_fitted.log_likelihood(
            iter([{"X": X}]),
            store="memory",
            progress=False,
        )


def test_explicit_batch_size_not_divisible_by_devices(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """An explicit batch size that is not a device multiple triggers a warning."""
    X, y = synthetic_data
    msg = (
        r"The `batch_size` \(\d+\) is not divisible by the number of devices "
        r"\(\d+\)\."
    )
    with pytest.warns(PerformanceWarning, match=msg):
        im_lm_svi_fitted.log_likelihood(X=X, y=y, batch_size=2, progress=False)


def test_log_likelihood_mcmc_keeps_chains(synthetic_data: tuple[Array, Array]) -> None:
    """The log-likelihood keeps the sampler's chains under both sharding paths."""
    X, y = synthetic_data

    def kernel(X: Array, y: Array | None = None) -> None:
        b = sample("b", dist.Normal(0.0, 1.0))
        sample("y", dist.Normal(X.sum(axis=-1) + b, 1.0), obs=y)

    im = ImpactModel(
        kernel,
        rng_key=random.key(42),
        inference=MCMC(NUTS(kernel), num_warmup=10, num_samples=5, num_chains=2),
    )
    im.fit_on_batch(X, y)
    b = im.inference.get_samples(group_by_chain=True)["b"]
    try:
        for shard_axis in ("obs", "draw"):
            out = im.log_likelihood(X, y, shard_axis=shard_axis, progress=False)
            np.testing.assert_allclose(
                out.log_likelihood["y"].transpose("chain", "draw", ...).values,
                dist.Normal(X.sum(axis=-1) + b[..., None], 1.0).log_prob(y),
            )
    finally:
        im.cleanup()
