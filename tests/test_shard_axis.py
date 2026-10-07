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

"""Tests for the `shard_axis` multi-device sharding strategy (`obs`/`draw`)."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, local_device_count, random

from aimz import ImpactModel, PerformanceWarning
from aimz.utils.data import ArrayDataset, ArrayLoader
from tests.conftest import latent_variable_model, make_svi, multidim_latent_model


def _n_draws(im: ImpactModel) -> int:
    """Return the posterior draw count via the public `posterior` property."""
    return len(next(iter(im.posterior.values())))


def test_local_latent_model(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
) -> None:
    """A local latent streams under `draw`; `obs` warns and reruns under it."""
    X, y = synthetic_data
    im = im_latent_var_svi_fitted
    n = _n_draws(im)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dt = im.predict(
            X,
            batch_size=len(X),
            progress=False,
            shard_axis="draw",
            return_sites=("y", "z"),
        )
        ll = im.log_likelihood(
            X, y, batch_size=len(X), progress=False, shard_axis="draw"
        )
    # `z` is shaped (num_samples, n_obs), which the obs path cannot stream
    assert dt["posterior_predictive"]["y"].shape == (1, n, len(X))
    assert dt["posterior_predictive"]["z"].shape == (1, n, len(X))
    assert ll["log_likelihood"]["y"].shape == (1, n, len(X))

    with pytest.warns(PerformanceWarning, match="rerunning with"):
        dt = im.predict(X, batch_size=len(X), progress=False, shard_axis="obs")
    assert dt["posterior_predictive"]["y"].shape == (1, n, len(X))
    with pytest.warns(PerformanceWarning, match="rerunning with"):
        ll = im.log_likelihood(X, y, batch_size=len(X), progress=False)
    assert ll["log_likelihood"]["y"].shape == (1, n, len(X))

    # Under the prior each chunk draws a fresh local latent, varying across rows
    z = np.asarray(
        im.sample_prior_predictive(
            X,
            num_samples=200,
            batch_size=len(X) // 4,
            progress=False,
            shard_axis="draw",
            return_sites=("y", "z"),
        )["prior_predictive"]["z"],
    )
    min_prior_std = 0.5
    assert z.shape[-1] == len(X)
    assert z.std(axis=-1).mean() > min_prior_std


def test_predict_data_reruns_draw_on_rank3_local_latent(
    synthetic_data: tuple[Array, Array],
) -> None:
    """A rank-3 observation-aligned latent also reruns under `draw` and round-trips."""
    X, y = synthetic_data
    im = ImpactModel(
        multidim_latent_model,
        rng_key=random.key(0),
        inference=make_svi(multidim_latent_model),
    )
    im.fit(X=X, y=y, batch_size=len(X), progress=False)
    try:
        with pytest.warns(PerformanceWarning, match="rerunning with"):
            dt = im.predict(
                X,
                batch_size=len(X),
                progress=False,
                shard_axis="obs",
                return_sites=("y", "z"),
            )
        pp = dt["posterior_predictive"]
        assert pp["y"].shape == (1, _n_draws(im), len(X))
        # The rank-3 latent streams back as (draw, n_obs, 2).
        assert pp["z"].shape == (1, _n_draws(im), len(X), 2)
    finally:
        im.cleanup()


@pytest.mark.filterwarnings("ignore:One or more posterior sample shapes")
def test_plan_execution_aligned_posterior(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An aligned posterior needs the whole input whenever the obs axis would split."""
    X, _ = synthetic_data
    im = im_latent_var_svi_fitted

    def plan(batch_size: int | None) -> tuple[str, int | None]:
        return im._plan_execution(
            X,
            shard_axis="obs",
            batch_size=batch_size,
            num_samples=im._num_samples,
            nbytes=im._output_nbytes(("y",)),
            posterior=im.posterior,
        )

    # On several devices the observation axis is always sharded
    monkeypatch.setattr(im, "_num_devices", 3)
    assert plan(len(X) // 4)[0] == "draw"
    assert plan(len(X))[0] == "draw"
    # On one device the whole input is pinned when it fits; a smaller explicit batch
    # still falls back
    monkeypatch.setattr(im, "_num_devices", 1)
    assert plan(None) == ("obs", len(X))
    assert plan(len(X))[0] == "obs"
    assert plan(len(X) // 2)[0] == "draw"
    monkeypatch.setattr(im, "_num_samples", 10**12)
    assert plan(None)[0] == "draw"


def test_predict_draw_global_model(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Under `draw` a global model streams any site, invariant to the chunking."""
    X, _ = synthetic_data
    im = im_lm_svi_fitted
    n = _n_draws(im)
    key = random.key(11)
    # A per-draw scalar site streams beside the per-observation output
    pp = im.predict(
        X,
        rng_key=key,
        progress=False,
        shard_axis="draw",
        return_sites=("y", "sigma"),
    )["posterior_predictive"]
    assert pp["y"].shape == (1, n, len(X))
    assert pp["sigma"].dims == ("chain", "draw")

    # The draws do not depend on the chunk size
    chunked = im.predict(
        X,
        rng_key=key,
        batch_size=max(1, n // 4),
        progress=False,
        shard_axis="draw",
        return_sites="y",
    )
    np.testing.assert_array_equal(
        np.asarray(pp["y"]),
        np.asarray(chunked["posterior_predictive"]["y"]),
    )

    # With the noise pinned near zero, both strategies agree draw by draw
    data = im.predict(
        X,
        intervention={"sigma": 1e-6},
        batch_size=99,
        progress=False,
        shard_axis="obs",
    )
    draw = im.predict(
        X,
        intervention={"sigma": 1e-6},
        batch_size=len(X),
        progress=False,
        shard_axis="draw",
    )
    np.testing.assert_allclose(
        np.asarray(data["posterior_predictive"]["y"]),
        np.asarray(draw["posterior_predictive"]["y"]),
        atol=1e-4,
    )


def test_predict_draw_num_samples_less_than_devices(
    synthetic_data: tuple[Array, Array],
) -> None:
    """`num_samples` smaller than the device count still round-trips."""
    X, y = synthetic_data
    n_draws = 2
    im = ImpactModel(
        latent_variable_model,
        rng_key=random.key(0),
        inference=make_svi(latent_variable_model),
    )
    im.fit(X=X, y=y, num_samples=n_draws, batch_size=len(X), progress=False)
    try:
        dt = im.predict(X, batch_size=len(X), progress=False, shard_axis="draw")
        assert dt["posterior_predictive"]["y"].sizes["draw"] == n_draws
    finally:
        im.cleanup()


def test_streaming_argument_validation(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Invalid arguments are rejected up front, before any output is written."""
    X, y = synthetic_data
    im = im_lm_svi_fitted
    for method, args in (
        ("predict", (X,)),
        ("sample_posterior_predictive", (X,)),
        ("sample_prior_predictive", (X,)),
        ("log_likelihood", (X, y)),
    ):
        with pytest.raises(ValueError, match="shard_axis"):
            getattr(im, method)(*args, progress=False, shard_axis="rows")
    # The draw path would otherwise step `range` by a non-positive batch size
    for bad_batch_size in (-1, 0):
        with pytest.raises(ValueError, match="positive integer"):
            im.predict(X, batch_size=bad_batch_size, progress=False)
    # A length-1 `y` would otherwise broadcast under `draw`
    for shard_axis in ("obs", "draw"):
        with pytest.raises(ValueError, match="leading-axis size"):
            im.log_likelihood(X, y[:1], shard_axis=shard_axis, progress=False)
    with pytest.raises(ValueError, match="at least 1 dimension"):
        im.log_likelihood(X, np.float32(0.5), shard_axis="draw", progress=False)
    # Draw-parallel replicates the whole input, so it takes an array, not a loader
    loader = ArrayLoader(ArrayDataset(X=np.asarray(X)), rng_key=random.key(0))
    with pytest.raises(TypeError, match="not a data loader"):
        im.predict(loader, progress=False, shard_axis="draw")


def test_predict_obs_shards_draw_independent_noise(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """`shard_axis='obs'` draws independent noise on every device shard."""
    X, _ = synthetic_data
    n_devices = local_device_count()
    if n_devices == 1:
        pytest.skip("Needs more than one device to compare shards.")
    # With a constant input, the only variation left is the likelihood noise
    dt = im_lm_svi_fitted.predict(
        jnp.tile(X[:1], reps=(n_devices, 1)),
        batch_size=n_devices,
        shard_axis="obs",
        progress=False,
    )
    draws = np.asarray(dt["posterior_predictive"].to_dataset()["y"])[0]
    for i in range(1, n_devices):
        assert not np.array_equal(draws[:, 0], draws[:, i])
