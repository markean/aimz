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

import numpy as np
import pytest
from jax import Array, random

from aimz import ImpactModel
from aimz.utils.data import ArrayDataset, ArrayLoader
from tests.conftest import (
    _make_svi,
    latent_variable_model,
    multidim_latent_model,
)


def _n_draws(im: ImpactModel) -> int:
    """Return the posterior draw count via the public `posterior` property."""
    return len(next(iter(im.posterior.values())))


def test_predict_draw_local_latent_no_fallback(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
) -> None:
    """`shard_axis='draw'` streams a local-latent model without a rerun warning."""
    X, _ = synthetic_data
    im = im_latent_var_svi_fitted
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dt = im.predict(
            X,
            batch_size=len(X),
            progress=False,
            shard_axis="draw",
            return_sites=("y", "z"),
        )
    pp = dt["posterior_predictive"]
    assert pp["y"].sizes["draw"] == _n_draws(im)
    assert pp["y"].shape[-1] == len(X)
    # `z` is a local latent of shape (num_samples, n_obs) — the case that the
    # data-parallel path cannot stream.
    assert pp["z"].sizes["draw"] == _n_draws(im)
    assert pp["z"].shape[-1] == len(X)


def test_predict_data_reruns_draw_on_local_latent(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
) -> None:
    """`shard_axis='obs'` warns and reruns under draw for a local-latent model."""
    X, _ = synthetic_data
    im = im_latent_var_svi_fitted
    with pytest.warns(UserWarning, match="rerunning with"):
        dt = im.predict(
            X,
            batch_size=len(X),
            progress=False,
            shard_axis="obs",
        )
    pp = dt["posterior_predictive"]
    assert pp["y"].sizes["draw"] == _n_draws(im)
    assert pp["y"].shape[-1] == len(X)


def test_predict_data_reruns_draw_on_rank3_local_latent(
    synthetic_data: tuple[Array, Array],
) -> None:
    """`shard_axis='obs'` reruns under draw for a rank-3 observation-aligned latent.

    The posterior site `z` is shaped `(num_samples, n_obs, 2)`; the detector must
    treat any `ndim >= 2` site with `shape[1] == n_obs` as observation-aligned, not
    only rank-2 sites. `z` also round-trips through the draw write path as rank-3.
    """
    X, y = synthetic_data
    im = ImpactModel(
        multidim_latent_model,
        rng_key=random.key(0),
        inference=_make_svi(multidim_latent_model),
    )
    im.fit(X=X, y=y, batch_size=len(X), progress=False)
    try:
        with pytest.warns(UserWarning, match="rerunning with"):
            dt = im.predict(
                X,
                batch_size=len(X),
                progress=False,
                shard_axis="obs",
                return_sites=("y", "z"),
            )
        pp = dt["posterior_predictive"]
        assert pp["y"].sizes["draw"] == _n_draws(im)
        assert pp["y"].shape[-1] == len(X)
        # The rank-3 latent streams back as (draw, n_obs, 2).
        assert pp["z"].sizes["draw"] == _n_draws(im)
        assert pp["z"].shape[-2:] == (len(X), 2)
    finally:
        im.cleanup()


@pytest.mark.parametrize("n_devices", [1, 3])
def test_plan_obs_batching_explicit_batch(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
    n_devices: int,
) -> None:
    """An aligned posterior needs the whole input when the obs axis is split.

    On multiple devices that is always (the axis is sharded); on one device only
    when an explicit `batch_size` is smaller than the observation count.
    """
    X, _ = synthetic_data
    im = im_latent_var_svi_fitted
    monkeypatch.setattr(im, "_num_devices", n_devices)
    assert im._plan_obs_batching(X, batch_size=len(X) // 4) == "fallback"
    assert im._plan_obs_batching(X, batch_size=len(X)) == (
        "fallback" if n_devices > 1 else "proceed"
    )


def test_predict_draw_padding_round_trip(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Draw padding to a device multiple is trimmed back to `num_samples`.

    With 3 host devices (conftest) and 1000 draws the count pads to 1002 and is
    trimmed back to 1000.
    """
    X, _ = synthetic_data
    dt = im_lm_svi_fitted.predict(
        X,
        batch_size=len(X),
        progress=False,
        shard_axis="draw",
    )
    assert dt["posterior_predictive"]["y"].sizes["draw"] == _n_draws(im_lm_svi_fitted)


def test_predict_draw_default_batch_size(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Omitting `batch_size` under draw auto-resolves it from the draws and input."""
    X, _ = synthetic_data
    dt = im_lm_svi_fitted.predict(X, shard_axis="draw", progress=False)
    pp = dt["posterior_predictive"]
    assert pp["y"].sizes["draw"] == _n_draws(im_lm_svi_fitted)
    assert pp["y"].shape[-1] == len(X)


def test_predict_draw_scalar_return_site(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """A scalar-per-draw (rank-1) return site streams under `shard_axis='draw'`.

    With >1 host device the draw-mode `out_specs` must shard only the leading draw
    axis; a length-2 spec would reject the rank-1 `sigma` site at trace time.
    """
    X, _ = synthetic_data
    im = im_lm_svi_fitted
    dt = im.predict(
        X,
        batch_size=len(X),
        progress=False,
        shard_axis="draw",
        return_sites=("y", "sigma"),
    )
    pp = dt["posterior_predictive"]
    # `sigma` is a global scalar latent: one value per draw, no observation axis.
    assert pp["sigma"].sizes["draw"] == _n_draws(im)
    assert "sigma_dim_0" not in pp["sigma"].dims
    # `y` (per-observation) still streams correctly alongside it.
    assert pp["y"].sizes["draw"] == _n_draws(im)
    assert pp["y"].shape[-1] == len(X)


def test_predict_draw_num_samples_less_than_devices(
    synthetic_data: tuple[Array, Array],
) -> None:
    """`num_samples` smaller than the device count still round-trips."""
    X, y = synthetic_data
    n_draws = 2
    im = ImpactModel(
        latent_variable_model,
        rng_key=random.key(0),
        inference=_make_svi(latent_variable_model),
    )
    im.fit(X=X, y=y, num_samples=n_draws, batch_size=len(X), progress=False)
    try:
        dt = im.predict(X, batch_size=len(X), progress=False, shard_axis="draw")
        assert dt["posterior_predictive"]["y"].sizes["draw"] == n_draws
    finally:
        im.cleanup()


def test_predict_data_vs_draw_means_close(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """For a global model both strategies use the same posterior draws.

    The per-observation predictive mean is therefore the same up to the per-draw
    likelihood noise, which averages out over the draws.
    """
    X, _ = synthetic_data
    # A batch size divisible by the 3 host devices avoids the divisibility warning.
    data = np.asarray(
        im_lm_svi_fitted.predict(
            X,
            batch_size=99,
            progress=False,
            shard_axis="obs",
        )["posterior_predictive"]["y"],
    )
    draw = np.asarray(
        im_lm_svi_fitted.predict(
            X,
            batch_size=len(X),
            progress=False,
            shard_axis="draw",
        )["posterior_predictive"]["y"],
    )
    assert data.shape == draw.shape
    np.testing.assert_allclose(
        data.mean(axis=(0, 1)),
        draw.mean(axis=(0, 1)),
        atol=0.5,
    )


def test_log_likelihood_draw_local_latent(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
) -> None:
    """`log_likelihood(shard_axis='draw')` works on a local-latent model."""
    X, y = synthetic_data
    im = im_latent_var_svi_fitted
    dt = im.log_likelihood(X, y, batch_size=len(X), progress=False, shard_axis="draw")
    ll = dt["log_likelihood"]["y"]
    assert ll.sizes["draw"] == _n_draws(im)
    assert ll.shape[-1] == len(X)


def test_log_likelihood_data_reruns_draw_on_local_latent(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
) -> None:
    """`shard_axis='obs'` warns and reruns under draw for a local-latent model."""
    X, y = synthetic_data
    with pytest.warns(UserWarning, match="rerunning with"):
        dt = im_latent_var_svi_fitted.log_likelihood(
            X,
            y,
            batch_size=len(X),
            progress=False,
            shard_axis="obs",
        )
    assert dt["log_likelihood"]["y"].shape[-1] == len(X)


def test_log_likelihood_draw_empty_posterior_parity(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no posterior, `shard_axis='draw'` matches the single-draw data path."""
    X, y = synthetic_data
    monkeypatch.setattr(im_lm_svi_fitted, "_posterior", None)
    # A batch size divisible by the 3 host devices avoids the divisibility warning.
    dt = im_lm_svi_fitted.log_likelihood(
        X,
        y,
        batch_size=99,
        progress=False,
        shard_axis="draw",
    )
    assert dt["log_likelihood"]["y"].sizes["draw"] == 1


def test_predict_draw_chunk_size_invariant(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Draw output is invariant to the per-chunk draw count (split-once-then-slice).

    Keys are split once over all draws and sliced per chunk, so the same `rng_key`
    yields identical draws no matter how many chunks the draw axis is split into.
    """
    X, _ = synthetic_data
    key = random.key(11)
    n = _n_draws(im_lm_svi_fitted)
    a = np.asarray(
        im_lm_svi_fitted.predict(
            X,
            rng_key=key,
            batch_size=n,
            progress=False,
            shard_axis="draw",
        )["posterior_predictive"]["y"],
    )
    b = np.asarray(
        im_lm_svi_fitted.predict(
            X,
            rng_key=key,
            batch_size=max(1, n // 4),
            progress=False,
            shard_axis="draw",
        )["posterior_predictive"]["y"],
    )
    np.testing.assert_array_equal(a, b)


def test_sample_prior_predictive_draw_local_latent(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
) -> None:
    """Prior predictive draw-parallel redraws a local latent fresh per observation.

    No plate and no conditioning: the whole input is held resident, so each chunk draws
    a complete fresh prior sample and the local latent varies across the observation
    axis (prior std close to one).
    """
    X, _ = synthetic_data
    pp = im_latent_var_svi_fitted.sample_prior_predictive(
        X,
        num_samples=200,
        batch_size=len(X) // 4,
        progress=False,
        shard_axis="draw",
        return_sites=("y", "z"),
    )["prior_predictive"]
    z = np.asarray(pp["z"])
    min_prior_std = 0.5
    assert z.shape[-1] == len(X)
    assert pp["y"].shape[-1] == len(X)
    assert z.std(axis=-1).mean() > min_prior_std


class TestValidation:
    """Invalid arguments are rejected up front, before any output is written."""

    def test_invalid_shard_axis_value_raises(
        self,
        synthetic_data: tuple[Array, Array],
        im_lm_svi_fitted: ImpactModel,
    ) -> None:
        """An unknown `shard_axis` value raises, not silently run as data-parallel."""
        X, y = synthetic_data
        im = im_lm_svi_fitted
        with pytest.raises(ValueError, match="shard_axis"):
            im.predict(X, progress=False, shard_axis="rows")
        with pytest.raises(ValueError, match="shard_axis"):
            im.sample_posterior_predictive(X, progress=False, shard_axis="rows")
        with pytest.raises(ValueError, match="shard_axis"):
            im.log_likelihood(X, y, progress=False, shard_axis="rows")
        with pytest.raises(ValueError, match="shard_axis"):
            im.sample_prior_predictive(
                X,
                num_samples=2,
                progress=False,
                shard_axis="rows",
            )

    @pytest.mark.parametrize("shard_axis", ["draw", "obs"])
    @pytest.mark.parametrize("bad_batch_size", [-1, 0])
    def test_predict_rejects_nonpositive_batch_size(
        self,
        synthetic_data: tuple[Array, Array],
        im_lm_svi_fitted: ImpactModel,
        shard_axis: str,
        bad_batch_size: int,
    ) -> None:
        """A non-positive `batch_size` is rejected up front on both parallel paths.

        The draw path would otherwise step `range` by it (empty -> silent empty
        result for `-1`) or divide by it (`ZeroDivisionError` for `0`).
        """
        X, _ = synthetic_data
        with pytest.raises(ValueError, match="positive integer"):
            im_lm_svi_fitted.predict(
                X,
                shard_axis=shard_axis,
                batch_size=bad_batch_size,
                progress=False,
            )

    def test_log_likelihood_draw_rejects_mismatched_y(
        self,
        synthetic_data: tuple[Array, Array],
        im_lm_svi_fitted: ImpactModel,
    ) -> None:
        """A length-mismatched `y` is rejected up front, the same way both paths are.

        The draw path replicates `X`/`y` independently and would otherwise broadcast a
        length-1 `y` across every observation, silently returning wrong log-likelihoods.
        """
        X, y = synthetic_data
        im = im_lm_svi_fitted
        with pytest.raises(ValueError, match="leading-axis size"):
            im.log_likelihood(
                X,
                y[:1],
                shard_axis="draw",
                batch_size=len(X),
                progress=False,
            )
        # Parity: the data path raises the same error from the same entry-point check.
        with pytest.raises(ValueError, match="leading-axis size"):
            im.log_likelihood(
                X,
                y[:1],
                shard_axis="obs",
                batch_size=len(X),
                progress=False,
            )

    def test_log_likelihood_draw_rejects_0d_y(
        self,
        synthetic_data: tuple[Array, Array],
        im_lm_svi_fitted: ImpactModel,
    ) -> None:
        """A 0-D `y` is rejected before any output directory is created."""
        X, _ = synthetic_data
        with pytest.raises(ValueError, match="at least 1 dimension"):
            im_lm_svi_fitted.log_likelihood(
                X,
                np.float32(0.5),
                shard_axis="draw",
                batch_size=len(X),
                progress=False,
            )

    def test_predict_draw_rejects_data_loader(
        self,
        synthetic_data: tuple[Array, Array],
        im_lm_svi_fitted: ImpactModel,
    ) -> None:
        """Draw-parallel requires an array `X`, not a data loader."""
        X, _ = synthetic_data
        loader = ArrayLoader(
            ArrayDataset(X=np.asarray(X)),
            rng_key=random.key(0),
            batch_size=10,
        )
        with pytest.raises(TypeError, match="not a data loader"):
            im_lm_svi_fitted.predict(loader, progress=False, shard_axis="draw")


def test_aligned_posterior_pins_whole_input_on_single_device(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On one device, an aligned posterior that fits memory avoids the draw fallback.

    Automatic batching splits the observation axis for I/O parallelism, which an
    observation-aligned posterior cannot tolerate; the whole-input batch is pinned
    instead of warning and rerunning draw-parallel, preserving the pre-split behavior.
    """
    X, _ = synthetic_data
    im = im_latent_var_svi_fitted
    monkeypatch.setattr(im, "_num_devices", 1)

    # Fits the budget on a single device: pin the whole input, no fallback.
    assert im._plan_obs_batching(X, batch_size=None) == "whole"
    # An explicit smaller batch is the caller's contract and still forces the fallback.
    assert im._plan_obs_batching(X, batch_size=max(1, len(X) // 2)) == "fallback"
