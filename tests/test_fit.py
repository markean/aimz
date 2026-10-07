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

"""Tests for the `.fit()` method."""

import warnings

import numpy as np
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import sample
from numpyro.infer import MCMC, NUTS

from aimz import FitWarning, ImpactModel
from aimz.utils.data import ArrayDataset, ArrayLoader
from tests.conftest import lm, lm_subsample, make_svi


def test_fit_argument_errors(synthetic_data: tuple[Array, Array]) -> None:
    """MCMC inference, and kernel arguments that do not bind, are rejected."""
    X, y = synthetic_data
    mcmc = MCMC(NUTS(lm), num_warmup=10, num_samples=10, progress_bar=False)
    im = ImpactModel(lm, rng_key=random.key(42), inference=mcmc)
    with pytest.raises(TypeError, match="not supported for MCMC"):
        im.fit(X, y, batch_size=len(X))

    def kernel(X: Array, arg: object, y: Array | None = None) -> None:
        pass

    im = ImpactModel(kernel, rng_key=random.key(42), inference=make_svi(kernel))
    with pytest.raises(TypeError, match="missing a required argument: 'arg'"):
        im.fit(X, y, batch_size=len(X), progress=False)
    with pytest.raises(TypeError, match="unexpected keyword argument 'extra'"):
        im.fit(X, y, arg=True, extra=True, batch_size=len(X), progress=False)


def test_fit_nan_warning(synthetic_data: tuple[Array, Array]) -> None:
    """A NaN loss warns, then the diverged posterior draw raises."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=make_svi(lm, step_size=1e3))
    with (
        pytest.warns(FitWarning, match="Loss contains NaN or Inf"),
        pytest.raises(ValueError, match="invalid loc parameter"),
    ):
        im.fit(X, y, batch_size=len(X), epochs=3, progress=False)

    with (
        pytest.warns(FitWarning, match="Loss contains NaN or Inf"),
        pytest.raises(ValueError, match="invalid loc parameter"),
    ):
        im.fit_on_batch(X, y, progress=False)


def test_fit_warns_on_unscaled_minibatches(synthetic_data: tuple[Array, Array]) -> None:
    """Batches smaller than the data warn unless the kernel scales the output site."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=make_svi(lm))
    msg = "trains on batches of 20 of 100 observations, but the kernel gives the"

    # An array input and an `ArrayLoader` are checked on their first batch.
    with pytest.warns(FitWarning, match=msg):
        im.fit(X, y, batch_size=20, epochs=1, progress=False)
    loader = ArrayLoader(ArrayDataset(X=X, y=y), rng_key=random.key(0), batch_size=20)
    with pytest.warns(FitWarning, match=msg):
        im.fit(loader, epochs=1, progress=False)

    # The whole data as one batch, and a kernel that scales the batch, pass silently.
    im_scaled = ImpactModel(
        lm_subsample,
        rng_key=random.key(42),
        inference=make_svi(lm_subsample),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FitWarning)
        im.fit(X, y, batch_size=len(X), epochs=1, progress=False)
        im_scaled.fit(X, y, batch_size=20, epochs=1, progress=False)


def test_fit_loader_with_custom_param_names(
    synthetic_data: tuple[Array, Array],
) -> None:
    """Fit, predict, and log-likelihood accept a loader keyed by custom names."""
    X, y = synthetic_data

    def kernel(x: Array, obs: Array | None = None) -> None:
        b = sample("b", dist.Normal(0.0, 1.0))
        sample("obs", dist.Normal(b + x.sum(axis=-1), 1.0), obs=obs)

    im = ImpactModel(
        kernel,
        rng_key=random.key(42),
        inference=make_svi(kernel),
        param_input="x",
        param_output="obs",
    )
    loader = ArrayLoader(
        ArrayDataset(x=X, obs=y), rng_key=random.key(0), batch_size=len(X)
    )
    im.fit(loader, num_samples=10, progress=False)
    dt_pred = im.predict(loader, store="memory", progress=False)
    dt_ll = im.log_likelihood(loader, store="memory", progress=False)

    assert dt_pred.posterior_predictive["obs"].shape == (1, 10, len(X))
    assert dt_ll.log_likelihood["obs"].shape == (1, 10, len(X))
    with pytest.raises(ValueError, match="no field named 'obs'"):
        im.fit(ArrayLoader(ArrayDataset(x=X), rng_key=random.key(0)), progress=False)
    with pytest.raises(ValueError, match="no field named 'obs'"):
        im.fit([{"x": X}], progress=False)


def test_fit_rejected_call_leaves_state_untouched(
    synthetic_data: tuple[Array, Array],
) -> None:
    """A rejected `fit` changes neither the key nor the draw count."""
    X, y = synthetic_data
    num_samples = 5
    im = ImpactModel(lm, rng_key=random.key(42), inference=make_svi(lm))
    im.fit(X=X, y=y, num_samples=num_samples, batch_size=len(X), progress=False)
    key_before = random.key_data(im.rng_key)
    loader = ArrayLoader(ArrayDataset(X=X, y=y), rng_key=random.key(0))

    with pytest.raises(TypeError, match="must be `None` when `X` is already"):
        im.fit(X=loader, y=y, progress=False)
    with pytest.raises(TypeError, match="unexpected keyword argument 'c'"):
        im.fit(X=X, y=y, c=1.0, progress=False)

    np.testing.assert_array_equal(random.key_data(im.rng_key), key_before)
    assert im._num_samples == num_samples
    out = im.predict(X, store="memory", progress=False)
    assert out.posterior_predictive["y"].sizes["draw"] == num_samples

    # A one-shot iterator is exhausted after its first epoch
    with pytest.raises(ValueError, match="yielded no batches in epoch 2"):
        im.fit(iter([{"X": X, "y": y}]), epochs=2, progress=False)


def test_fit_array_matches_loader(synthetic_data: tuple[Array, Array]) -> None:
    """Fitting on arrays or on the equivalent data loader gives the same result."""
    X, y = synthetic_data
    rng_key = random.key(1)
    im_array = ImpactModel(
        lm_subsample,
        rng_key=random.key(0),
        inference=make_svi(lm_subsample),
    )
    im_array.fit(X, y, rng_key=rng_key, batch_size=20, shuffle=False, progress=False)
    loader = ArrayLoader(
        ArrayDataset(X=X, y=y), rng_key=random.key(0), batch_size=20, shuffle=False
    )
    im_loader = ImpactModel(
        lm_subsample,
        rng_key=random.key(0),
        inference=make_svi(lm_subsample),
    )
    im_loader.fit(loader, rng_key=rng_key, progress=False)

    np.testing.assert_allclose(im_array.vi_result.losses, im_loader.vi_result.losses)
    np.testing.assert_array_equal(
        im_array.predict_on_batch(X, rng_key=rng_key).posterior_predictive["y"],
        im_loader.predict_on_batch(X, rng_key=rng_key).posterior_predictive["y"],
    )
