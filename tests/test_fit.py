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

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import param, plate, sample
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.optim import Adam

from aimz import FitWarning, ImpactModel
from aimz.utils.data import ArrayDataset, ArrayLoader
from tests.conftest import lm, lm_subsample, lm_with_kwargs_array, make_svi


def test_fit_argument_errors(synthetic_data: tuple[Array, Array]) -> None:
    """MCMC inference, and kernel arguments that do not bind or fit the rows, raise."""
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

    # A keyword array as long as a batch, but not as the data, is not reused
    im = ImpactModel(
        lm_with_kwargs_array,
        rng_key=random.key(42),
        inference=make_svi(lm_with_kwargs_array),
    )
    with pytest.raises(TypeError, match="incompatible shapes"):
        im.fit(X, y, c=jnp.zeros(20), batch_size=20, progress=False)


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


def test_fit_minibatch_warnings(synthetic_data: tuple[Array, Array]) -> None:
    """Batches warn for a misscaled output, a coupled site, and a guide's row values."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=make_svi(lm))
    msg = "trains on batches of 20 of 100 observations, but the kernel scales the"

    # An array input and an `ArrayLoader` are checked on their first batch.
    with pytest.warns(FitWarning, match=msg):
        im.fit(X, y, batch_size=20, epochs=1, progress=False)
    loader = ArrayLoader(ArrayDataset(X=X, y=y), rng_key=random.key(0), batch_size=20)
    with pytest.warns(FitWarning, match=msg):
        im.fit(loader, epochs=1, progress=False)

    # A scaled kernel with a latent site per row warns, since the batches split it.
    def per_row(X: Array, y: Array | None = None) -> None:
        with plate("data", size=100, subsample_size=X.shape[0]):
            z = sample("z", dist.Normal())
            sample("y", dist.Normal(X[:, 0] + z, 1.0), obs=y)

    # Raised as an error, the warning leaves a model with an automatic guide as it was,
    # so the fit it names works on the same model.
    im_rows = ImpactModel(per_row, rng_key=random.key(42), inference=make_svi(per_row))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FitWarning)
        with pytest.raises(FitWarning, match="keeps a value per observation"):
            im_rows.fit(X, y, batch_size=20, epochs=1, progress=False)
    im_rows.fit_on_batch(X, y, num_steps=1, progress=False)

    # An event over the rows is split, though with no other site its scale does not
    # matter; a global event as long as a batch is not split, even when an argument
    # passed whole sizes it.
    def coupled(X: Array, y: Array | None = None) -> None:
        sample("y", dist.MultivariateNormal(X[:, 0], jnp.eye(X.shape[0])), obs=y)

    def global_event(
        X: Array, y: Array | None = None, loc: Array | None = None
    ) -> None:
        w = sample("w", dist.MultivariateNormal(loc, jnp.eye(20)))
        with plate("data", size=100, subsample_size=X.shape[0]):
            sample("y", dist.Normal(X[:, 0] + w.sum(), 1.0), obs=y)

    im_coupled = ImpactModel(
        coupled,
        rng_key=random.key(42),
        inference=make_svi(coupled),
    )
    with pytest.warns(FitWarning, match="computes across observations at site"):
        im_coupled.fit(X, y, batch_size=20, epochs=1, progress=False)
    im_global = ImpactModel(
        global_event,
        rng_key=random.key(42),
        inference=make_svi(global_event),
    )

    # A guide that computes the site from each row's inputs trains on batches.
    def amortized(X: Array, y: Array | None = None) -> None:
        w = param("w", jnp.zeros(X.shape[1]))
        with plate("data", size=100, subsample_size=X.shape[0]):
            sample("z", dist.Normal(X @ w, 1.0))

    im_amortized = ImpactModel(
        per_row,
        rng_key=random.key(42),
        inference=SVI(per_row, amortized, Adam(1e-3), Trace_ELBO()),
    )

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
        im_amortized.fit(X, y, batch_size=20, epochs=1, progress=False)
        im_global.fit(X, y, batch_size=20, epochs=1, progress=False, loc=jnp.zeros(20))


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
    with pytest.raises(TypeError, match="`y` must be an array-like, got 'list'"):
        im.fit(X=X, y=y.tolist(), progress=False)

    np.testing.assert_array_equal(random.key_data(im.rng_key), key_before)
    assert im._num_samples == num_samples
    out = im.predict(X, store="memory", progress=False)
    assert out.posterior_predictive["y"].sizes["draw"] == num_samples

    # A one-shot iterator is exhausted after its first epoch
    with pytest.raises(ValueError, match="yielded no batches in epoch 2"):
        im.fit(iter([{"X": X, "y": y}]), epochs=2, progress=False)
    # Batches that are not mappings, as the rows of a list, are rejected
    with pytest.raises(TypeError, match="yielding batch mappings, got 'list'"):
        im.fit(X.tolist(), progress=False)


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

    np.testing.assert_allclose(
        im_array.vi_result.losses, im_loader.vi_result.losses, rtol=1e-5
    )
    np.testing.assert_array_equal(
        im_array.predict_on_batch(X, rng_key=rng_key).posterior_predictive["y"],
        im_loader.predict_on_batch(X, rng_key=rng_key).posterior_predictive["y"],
    )


def test_fit_draws_guide_on_data_in_order(synthetic_data: tuple[Array, Array]) -> None:
    """The guide draws the posterior from the data in input order, not shuffled."""
    X, y = synthetic_data

    def kernel(X: Array, y: Array | None = None) -> None:
        w = sample("w", dist.Normal().expand([X.shape[1]]).to_event(1))
        with plate("obs", X.shape[0]):
            z = sample("z", dist.Normal())
            sample("y", dist.Normal(X @ w + z, 0.1), obs=y)

    # Each row's latent comes from that row's data
    def guide(X: Array, y: Array | None = None) -> None:
        w_loc = param("w_loc", jnp.zeros(X.shape[1]))
        sample("w", dist.Normal(w_loc, 0.01).to_event(1))
        with plate("obs", X.shape[0]):
            sample("z", dist.Normal(y - X @ w_loc, 0.01))

    svi = SVI(kernel, guide, Adam(0.01), Trace_ELBO())
    im = ImpactModel(kernel, rng_key=random.key(0), inference=svi)
    im.fit(X, y, num_samples=20, progress=False)
    w_loc = im.vi_result.params["w_loc"]
    np.testing.assert_allclose(im.posterior["z"].mean(axis=0), y - X @ w_loc, atol=0.02)
