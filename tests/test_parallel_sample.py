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

"""Tests for draw-parallel (draws-sharded) ``predict`` / ``log_likelihood``."""

import warnings

import numpy as np
import pytest
from jax import Array, random

from aimz import ImpactModel
from aimz.utils.data import ArrayDataset, ArrayLoader
from tests.conftest import _make_svi, latent_variable_model, mlm


def _n_draws(im: ImpactModel) -> int:
    """Return the posterior draw count via the public ``posterior`` property."""
    return len(next(iter(im.posterior.values())))


def test_predict_draw_local_latent_no_fallback(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
) -> None:
    """`parallel='draw'` streams a local-latent model without a rerun warning."""
    X, _ = synthetic_data
    im = im_latent_var_svi_fitted
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dt = im.predict(
            X,
            batch_size=len(X),
            progress=False,
            parallel="draw",
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
    """`parallel='data'` warns and reruns under draw for a local-latent model."""
    X, _ = synthetic_data
    im = im_latent_var_svi_fitted
    with pytest.warns(UserWarning, match="rerunning with"):
        dt = im.predict(
            X,
            batch_size=len(X),
            progress=False,
            parallel="data",
        )
    pp = dt["posterior_predictive"]
    assert pp["y"].sizes["draw"] == _n_draws(im)
    assert pp["y"].shape[-1] == len(X)


def test_predict_draw_padding_round_trip(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Draw padding to a device multiple is trimmed back to ``num_samples``.

    With 3 host devices (conftest) and 1000 draws the count pads to 1002 and is
    trimmed back to 1000.
    """
    X, _ = synthetic_data
    dt = im_lm_svi_fitted.predict(
        X,
        batch_size=len(X),
        progress=False,
        parallel="draw",
    )
    assert dt["posterior_predictive"]["y"].sizes["draw"] == _n_draws(im_lm_svi_fitted)


def test_predict_draw_num_samples_less_than_devices(
    synthetic_data: tuple[Array, Array],
) -> None:
    """``num_samples`` smaller than the device count still round-trips."""
    X, y = synthetic_data
    n_draws = 2
    im = ImpactModel(
        latent_variable_model,
        rng_key=random.key(0),
        inference=_make_svi(latent_variable_model),
    )
    im.fit(X=X, y=y, num_samples=n_draws, batch_size=len(X), progress=False)
    try:
        dt = im.predict(X, batch_size=len(X), progress=False, parallel="draw")
        assert dt["posterior_predictive"]["y"].sizes["draw"] == n_draws
    finally:
        im.cleanup()


def test_predict_draw_deterministic(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """The same ``rng_key`` yields identical draws (host pre-split keys)."""
    X, _ = synthetic_data
    key = random.key(7)
    a = np.asarray(
        im_lm_svi_fitted.predict(
            X,
            rng_key=key,
            batch_size=len(X),
            progress=False,
            parallel="draw",
        )["posterior_predictive"]["y"],
    )
    b = np.asarray(
        im_lm_svi_fitted.predict(
            X,
            rng_key=key,
            batch_size=len(X),
            progress=False,
            parallel="draw",
        )["posterior_predictive"]["y"],
    )
    np.testing.assert_array_equal(a, b)


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
            parallel="data",
        )["posterior_predictive"]["y"],
    )
    draw = np.asarray(
        im_lm_svi_fitted.predict(
            X,
            batch_size=len(X),
            progress=False,
            parallel="draw",
        )["posterior_predictive"]["y"],
    )
    assert data.shape == draw.shape
    np.testing.assert_allclose(
        data.mean(axis=(0, 1)),
        draw.mean(axis=(0, 1)),
        atol=0.5,
    )


def test_predict_draw_with_kwargs_array(
    synthetic_data: tuple[Array, Array],
    im_lm_with_kwargs_svi_fitted: ImpactModel,
) -> None:
    """Array kwargs are replicated (not obs-sharded) under draw-parallelism."""
    X, y = synthetic_data
    dt = im_lm_with_kwargs_svi_fitted.predict(
        X,
        c=y,
        batch_size=len(X),
        progress=False,
        parallel="draw",
        return_sites="y",
    )
    assert dt["posterior_predictive"]["y"].shape[-1] == len(X)


def test_predict_draw_mlm() -> None:
    """A multi-output model (ndim>2 sites) works under draw-parallelism."""
    n_targets = 2
    X, y = synthetic_data_mlm()
    im = ImpactModel(mlm, rng_key=random.key(0), inference=_make_svi(mlm))
    im.fit(X=X, y=y, batch_size=len(X), progress=False)
    try:
        dt = im.predict(X, batch_size=len(X), progress=False, parallel="draw")
        out = dt["posterior_predictive"]["y"]
        assert out.shape[-2] == len(X)
        assert out.shape[-1] == n_targets
    finally:
        im.cleanup()


def test_predict_draw_mcmc(
    synthetic_data: tuple[Array, Array],
    im_lm_mcmc_fitted: ImpactModel,
) -> None:
    """Sample-parallel prediction works with an MCMC-fitted posterior."""
    X, _ = synthetic_data
    dt = im_lm_mcmc_fitted.predict(
        X,
        batch_size=len(X),
        progress=False,
        parallel="draw",
    )
    assert dt["posterior_predictive"]["y"].sizes["draw"] == _n_draws(im_lm_mcmc_fitted)


def test_log_likelihood_draw_local_latent(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
) -> None:
    """`log_likelihood(parallel='draw')` works on a local-latent model."""
    X, y = synthetic_data
    im = im_latent_var_svi_fitted
    dt = im.log_likelihood(X, y, batch_size=len(X), progress=False, parallel="draw")
    ll = dt["log_likelihood"]["y"]
    assert ll.sizes["draw"] == _n_draws(im)
    assert ll.shape[-1] == len(X)


def test_log_likelihood_draw_empty_posterior_parity(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no posterior, `parallel='draw'` matches the single-draw data path."""
    X, y = synthetic_data
    monkeypatch.setattr(im_lm_svi_fitted, "_posterior", None)
    # A batch size divisible by the 3 host devices avoids the divisibility warning.
    dt = im_lm_svi_fitted.log_likelihood(
        X,
        y,
        batch_size=99,
        progress=False,
        parallel="draw",
    )
    assert dt["log_likelihood"]["y"].sizes["draw"] == 1


def test_predict_draw_chunk_size_invariant(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Draw output is invariant to the per-chunk draw count (split-once-then-slice).

    Keys are split once over all draws and sliced per chunk, so the same ``rng_key``
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
            parallel="draw",
        )["posterior_predictive"]["y"],
    )
    b = np.asarray(
        im_lm_svi_fitted.predict(
            X,
            rng_key=key,
            batch_size=max(1, n // 4),
            progress=False,
            parallel="draw",
        )["posterior_predictive"]["y"],
    )
    np.testing.assert_array_equal(a, b)


def test_predict_draw_rejects_data_loader(
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
        im_lm_svi_fitted.predict(loader, progress=False, parallel="draw")


def test_log_likelihood_data_falls_back_on_local_latent(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
) -> None:
    """`log_likelihood(parallel='data')` warns and reruns under draw for a local."""
    X, y = synthetic_data
    msg = "rerunning with"
    with pytest.warns(UserWarning, match=msg):
        dt = im_latent_var_svi_fitted.log_likelihood(
            X,
            y,
            batch_size=len(X) // 4,
            progress=False,
            parallel="data",
        )
    assert dt["log_likelihood"]["y"].shape[-1] == len(X)


def test_prior_predictive_draw_local_latent(
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
        parallel="draw",
        return_sites=("y", "z"),
    )["prior_predictive"]
    z = np.asarray(pp["z"])
    min_prior_std = 0.5
    assert z.shape[-1] == len(X)
    assert pp["y"].shape[-1] == len(X)
    assert z.std(axis=-1).mean() > min_prior_std


def test_prior_predictive_data_vs_draw_marginals_close(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """For a global model both prior-predictive strategies sample the same marginal.

    The strategies use different RNG schemes, so the comparison is on the well-averaged
    marginal location and spread rather than noisy per-observation means.
    """
    X, _ = synthetic_data
    key = random.key(13)
    data = np.asarray(
        im_lm_svi_fitted.sample_prior_predictive(
            X,
            num_samples=500,
            rng_key=key,
            batch_size=len(X) - 1,
            progress=False,
            parallel="data",
        )["prior_predictive"]["y"],
    )
    draw = np.asarray(
        im_lm_svi_fitted.sample_prior_predictive(
            X,
            num_samples=500,
            rng_key=key,
            batch_size=99,
            progress=False,
            parallel="draw",
        )["prior_predictive"]["y"],
    )
    assert data.shape == draw.shape
    np.testing.assert_allclose(data.mean(), draw.mean(), atol=0.1)
    np.testing.assert_allclose(data.std(), draw.std(), rtol=0.1)


def test_predict_draw_intervention(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """An intervention (`do` handler) shifts predictions under draw-parallelism."""
    X, _ = synthetic_data
    im = im_lm_svi_fitted
    key = random.key(3)
    shift = 100.0
    base = np.asarray(
        im.predict(
            X,
            rng_key=key,
            batch_size=len(X),
            progress=False,
            parallel="draw",
        )["posterior_predictive"]["y"],
    )
    shifted = np.asarray(
        im.predict(
            X,
            intervention={"b": shift},
            rng_key=key,
            batch_size=len(X),
            progress=False,
            parallel="draw",
        )["posterior_predictive"]["y"],
    )
    assert shifted.mean() - base.mean() > shift / 2


def test_sample_posterior_predictive_draw(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """The `sample_posterior_predictive` alias accepts `parallel='draw'`."""
    X, _ = synthetic_data
    im = im_lm_svi_fitted
    dt = im.sample_posterior_predictive(
        X,
        batch_size=len(X),
        progress=False,
        parallel="draw",
    )
    pp = dt["posterior_predictive"]
    assert pp["y"].sizes["draw"] == _n_draws(im)
    assert pp["y"].shape[-1] == len(X)


def test_predict_draw_return_sites(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
) -> None:
    """`return_sites` filters the stored sites under draw-parallelism."""
    X, _ = synthetic_data
    dt = im_latent_var_svi_fitted.predict(
        X,
        batch_size=len(X),
        progress=False,
        parallel="draw",
        return_sites="y",
    )
    pp = dt["posterior_predictive"]
    assert "y" in pp.data_vars
    assert "z" not in pp.data_vars


def test_log_likelihood_draw_rejects_data_loader(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Draw-parallel `log_likelihood` requires an array `X`, not a data loader."""
    X, y = synthetic_data
    loader = ArrayLoader(
        ArrayDataset(X=np.asarray(X), y=np.asarray(y)),
        rng_key=random.key(0),
        batch_size=10,
    )
    with pytest.raises(TypeError, match="not a data loader"):
        im_lm_svi_fitted.log_likelihood(loader, progress=False, parallel="draw")


def test_prior_predictive_draw_rejects_data_loader(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Draw-parallel `sample_prior_predictive` requires an array `X`."""
    X, _ = synthetic_data
    loader = ArrayLoader(
        ArrayDataset(X=np.asarray(X)),
        rng_key=random.key(0),
        batch_size=10,
    )
    with pytest.raises(TypeError, match="not a data loader"):
        im_lm_svi_fitted.sample_prior_predictive(
            loader,
            num_samples=2,
            progress=False,
            parallel="draw",
        )


def test_invalid_parallel_value_raises(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """An unknown `parallel` value raises instead of silently running data-parallel."""
    X, y = synthetic_data
    im = im_lm_svi_fitted
    with pytest.raises(ValueError, match="parallel"):
        im.predict(X, progress=False, parallel="rows")
    with pytest.raises(ValueError, match="parallel"):
        im.sample_posterior_predictive(X, progress=False, parallel="rows")
    with pytest.raises(ValueError, match="parallel"):
        im.log_likelihood(X, y, progress=False, parallel="rows")
    with pytest.raises(ValueError, match="parallel"):
        im.sample_prior_predictive(X, num_samples=2, progress=False, parallel="rows")


def synthetic_data_mlm() -> tuple[Array, Array]:
    """Generate a small multi-output dataset for the ``mlm`` model."""
    rng_key = random.key(123)
    key_x, key_w, key_e = random.split(rng_key, 3)
    n_obs, n_features, n_targets = 30, 3, 2
    x = random.normal(key_x, (n_obs, n_features))
    w = random.normal(key_w, (n_features, n_targets))
    e = random.normal(key_e, (n_obs, n_targets))
    return x, x @ w + e
