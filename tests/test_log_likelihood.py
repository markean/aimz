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
from numpyro import plate, sample
from numpyro.infer.util import log_likelihood

from aimz import ImpactModel
from tests.conftest import lm_subsample, make_svi


@pytest.mark.filterwarnings("ignore:subsample_size does not match")
def test_log_likelihood_subsampled_kernel(synthetic_data: tuple[Array, Array]) -> None:
    """Subsampling kernels score every row; a batch past a fixed-size plate raises."""
    X, y = synthetic_data
    im = ImpactModel(
        lm_subsample,
        rng_key=random.key(42),
        inference=make_svi(lm_subsample),
    )
    im.fit(X, y, batch_size=20, progress=False)
    try:
        out = im.log_likelihood(X, y, batch_size=30, progress=False)
        out_draw = im.log_likelihood(
            X, y, shard_axis="draw", batch_size=250, progress=False
        )
        assert out.log_likelihood["y"].shape == (1, 1000, len(X))
        assert np.isfinite(out.log_likelihood["y"].values).all()
        # Plate indices never gather here, so the sharding paths agree
        np.testing.assert_allclose(
            out.log_likelihood["y"].values,
            out_draw.log_likelihood["y"].values,
            rtol=1e-5,
            atol=1e-6,
        )
        with pytest.raises(ValueError, match="declares size=100"):
            im.log_likelihood(
                jnp.tile(X, (6, 1)), jnp.tile(y, 6), batch_size=600, progress=False
            )
    finally:
        im.cleanup()

    # A plate sized by the rows it is given takes them all, as on a device's share
    def gathering(X: Array, y: Array | None = None, k: int | None = None) -> None:
        b = sample("b", dist.Normal())
        with plate("obs", X.shape[0], subsample_size=k) as idx:
            sample("y", dist.Normal(b * X[idx], 1.0), obs=None if y is None else y[idx])

    im = ImpactModel(gathering, rng_key=random.key(0), inference=make_svi(gathering))
    im.set_posterior_sample({"b": jnp.ones(4)})
    x = jnp.arange(30.0) / 10
    out = im.log_likelihood(
        x, 2 * x, k=len(x), batch_size=12, store="memory", progress=False
    )
    np.testing.assert_allclose(
        out.log_likelihood["y"].values[0],
        np.tile(-0.5 * np.log(2 * np.pi) - 0.5 * x**2, (4, 1)),
        rtol=1e-5,
    )


def test_log_likelihood_without_posterior(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without posterior draws, both strategies score a single draw."""
    X, y = synthetic_data
    monkeypatch.setattr(im_lm_svi_fitted, "_posterior", None)
    for shard_axis in ("obs", "draw"):
        out = im_lm_svi_fitted.log_likelihood(
            X, y, shard_axis=shard_axis, batch_size=99, progress=False
        )
        assert out.log_likelihood["y"].sizes["draw"] == 1


def test_log_likelihood_requires_output(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """The output is required, `y` for an array or a field for a loader, per row."""
    X, _ = synthetic_data
    with pytest.raises(ValueError, match="`y` is required"):
        im_lm_svi_fitted.log_likelihood(X, progress=False)
    with pytest.raises(ValueError, match="requires the observed output field 'y'"):
        im_lm_svi_fitted.log_likelihood(
            iter([{"X": X}]), store="memory", progress=False
        )

    # A loader streams only an output with one value per row
    def joint(X: Array, y: Array | None = None) -> None:
        mu = sample("mu", dist.Normal())
        sample("y", dist.MultivariateNormal(mu + X[:, 0], jnp.eye(len(X))), obs=y)

    im = ImpactModel(joint, rng_key=random.key(0), inference=make_svi(joint))
    im.set_posterior_sample({"mu": jnp.zeros(5)})
    with pytest.raises(NotImplementedError, match="match the input batch size"):
        im.log_likelihood(
            iter([{"X": X[:60], "y": X[:60, 0]}]), store="memory", progress=False
        )


def test_log_likelihood_event_over_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """An output drawn as one event over all the rows is scored under `draw`."""

    def gp(X: Array, y: Array | None = None) -> None:
        ls = sample("ls", dist.LogNormal(0.0, 0.3))
        cov = jnp.exp(-0.5 * ((X[:, None] - X[None, :]) / ls) ** 2)
        k = cov + 1e-2 * jnp.eye(X.shape[0])
        sample("y", dist.MultivariateNormal(jnp.zeros(X.shape[0]), k), obs=y)

    X = np.linspace(0.0, 3.0, 12, dtype=np.float32)
    y = np.sin(X)
    posterior = {"ls": np.linspace(0.5, 1.5, 4, dtype=np.float32)}
    expected = log_likelihood(gp, posterior, X, y)["y"]
    # One device and one batch keep the rows whole, yet only `draw` streams a
    # log-likelihood without a row axis
    monkeypatch.setattr("aimz.model.impact_model.local_device_count", lambda: 1)
    im = ImpactModel(gp, rng_key=random.key(0), inference=make_svi(gp))
    im.set_posterior_sample(posterior)
    dt = im.log_likelihood(X, y, store="memory", progress=False)
    np.testing.assert_allclose(
        dt.log_likelihood["y"].values.ravel(),
        expected,
        rtol=1e-5,
    )
