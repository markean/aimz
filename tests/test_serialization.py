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

"""Tests for saving and loading functionality of models."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cloudpickle
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import plate, sample
from numpyro.infer import MCMC, NUTS

from aimz import ImpactModel, PerformanceWarning
from tests.conftest import make_svi

if TYPE_CHECKING:
    import xarray as xr


def poisson(X: Array, y: Array | None = None) -> None:
    """Poisson regression model."""
    b = sample("b", dist.Normal())
    with plate("obs", X.shape[0]):
        sample("y", dist.Poisson(jnp.exp(X[:, 0] + b)), obs=y)


def _load_predict(path: Path, X: Array, y: Array) -> np.ndarray:
    with path.open("rb") as f:
        im = cloudpickle.load(f)
    # Continuing training first uses the keys nested in the inference state
    im.train_on_batch(X, y)
    out = cast("xr.DataTree", im.predict_on_batch(X))

    return np.asarray(out["posterior_predictive"]["y"])


def test_save_load(
    im_lm_svi_fitted: ImpactModel,
    synthetic_data: tuple[Array, Array],
    tmp_path: Path,
) -> None:
    """A pickled model round-trips its posterior and predictions without a refit.

    The posterior samples survive the round trip unchanged, and the loaded model
    predicts draw-for-draw identically under the same explicit PRNG key.
    """
    X, _ = synthetic_data
    p = tmp_path / "model.pkl"
    with p.open("wb") as f:
        cloudpickle.dump(im_lm_svi_fitted, f)
    with p.open("rb") as f:
        im = cloudpickle.load(f)

    assert isinstance(im, ImpactModel)
    assert im.posterior is not None
    assert im_lm_svi_fitted.posterior is not None
    assert set(im.posterior) == set(im_lm_svi_fitted.posterior)
    for site, draws in im_lm_svi_fitted.posterior.items():
        np.testing.assert_array_equal(
            np.asarray(im.posterior[site]),
            np.asarray(draws),
        )
    expected = cast(
        "xr.DataTree",
        im_lm_svi_fitted.predict_on_batch(X, rng_key=random.key(0)),
    )
    actual = cast("xr.DataTree", im.predict_on_batch(X, rng_key=random.key(0)))
    np.testing.assert_array_equal(
        np.asarray(actual["posterior_predictive"]["y"]),
        np.asarray(expected["posterior_predictive"]["y"]),
    )


def test_load_poisson_new_process(
    synthetic_data: tuple[Array, Array],
    tmp_path: Path,
) -> None:
    """A pickled model with a Poisson likelihood trains and predicts in a new process.

    After a training step, the loaded model predicts with its internal PRNG key
    draw-for-draw identically to the original, instead of rejecting the unpickled keys.
    """
    X, _ = synthetic_data
    y = random.poisson(random.key(1), 1.0, (len(X),))
    im = ImpactModel(poisson, rng_key=random.key(42), inference=make_svi(poisson))
    im.fit_on_batch(X, y, num_steps=10, progress=False)
    p = tmp_path / "model.pkl"
    with p.open("wb") as f:
        cloudpickle.dump(im, f)
    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as executor:
        actual = executor.submit(_load_predict, p, X, y).result()

    expected = cast("xr.DataTree", im.predict_on_batch(X))
    np.testing.assert_array_equal(
        actual,
        np.asarray(expected["posterior_predictive"]["y"]),
    )


def test_load_parallel_chains_on_fewer_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel chains loaded on fewer devices than chains fall back to sequential."""
    mcmc = MCMC(NUTS(poisson), num_warmup=10, num_samples=10, num_chains=2)
    im = ImpactModel(poisson, rng_key=random.key(42), inference=mcmc)
    data = cloudpickle.dumps(im)
    monkeypatch.setattr("aimz.model.impact_model.local_device_count", lambda: 1)
    with pytest.warns(
        PerformanceWarning,
        match=r"not enough devices to run parallel chains: expected 2 but got 1\.",
    ):
        im = cloudpickle.loads(data)

    assert im.inference.chain_method == "sequential"
