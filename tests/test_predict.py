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

"""Tests for the `.predict()` method."""

import warnings
from collections.abc import Iterator
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax import Array, device_count, random
from numpyro import deterministic, param, plate, sample
from numpyro.contrib.control_flow import scan
from numpyro.primitives import mutable

from aimz import ImpactModel, OutputWarning, PerformanceWarning
from aimz.model._streaming import _OutputStreamer, _RuntimeContext
from aimz.utils.data import ArrayDataset, ArrayLoader
from tests.conftest import latent_intervention_model, lm, make_svi


def _iter_batches(
    X: Array,
    sizes: list[int],
    y: Array | None = None,
) -> Iterator[dict[str, Array]]:
    """Yield consecutive batches of the given sizes as a one-shot generator."""
    start = 0
    for size in sizes:
        batch = {"X": X[start : start + size]}
        if y is not None:
            batch["y"] = y[start : start + size]
        yield batch
        start += size


def test_predict_site_without_observation_axis(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What `obs` cannot split reruns under `draw`; a loader raises instead."""
    X, _ = synthetic_data
    # A mesh-less streamer forces the single-device path
    monkeypatch.setattr(
        im_lm_svi_fitted,
        "_streamer",
        _OutputStreamer(
            _RuntimeContext(
                param_input=im_lm_svi_fitted.param_input,
                param_output=im_lm_svi_fitted.param_output,
                mesh=None,
                num_devices=1,
                replicated_sharding=None,
                partitioned_sharding=None,
            ),
        ),
    )
    with pytest.warns(PerformanceWarning, match="'sigma' have no observation axis"):
        dt = im_lm_svi_fitted.predict(
            X=X,
            return_sites="sigma",
            batch_size=3,
            progress=False,
        )
    assert dt.posterior_predictive["sigma"].dims == ("chain", "draw")
    with pytest.warns(PerformanceWarning, match="'sigma' have no observation axis"):
        im_lm_svi_fitted.predict(X=X[:1], return_sites="sigma", progress=False)
    # A loader raises, even where each batch has as many rows as the site has values
    with pytest.raises(NotImplementedError, match="'w' have no observation axis"):
        im_lm_svi_fitted.predict(
            X=_iter_batches(X, [10] * 10),
            return_sites="w",
            progress=False,
        )

    def counted(X: Array, y: Array | None = None, n: int = 100) -> None:
        mu = sample("mu", dist.Normal())
        with plate("obs", n):
            sample("y", dist.Normal(mu + X[:n, 0], 1.0), obs=y)

    im = ImpactModel(counted, rng_key=random.key(0), inference=make_svi(counted))
    im.set_posterior_sample({"mu": jnp.zeros(5)})
    # A plate sized by the number of rows does not run on fewer rows
    with pytest.warns(PerformanceWarning, match="does not run on fewer rows"):
        dt = im.predict(X, progress=False)
    assert dt.posterior_predictive.attrs["shard_axis"] == "draw"
    # An output sized by another count does not follow the rows
    with pytest.warns(OutputWarning, match="'y' does not have one value per row"):
        dt = im.predict(X, n=3, progress=False)
    assert dt.posterior_predictive["y"].shape == (1, 5, 3)
    # So does it under `draw`, which keeps the rows whole, and in the on-batch methods
    with pytest.warns(OutputWarning, match="'y' does not have one value per row"):
        im.predict(X, n=3, shard_axis="draw", progress=False)
    with pytest.warns(OutputWarning, match="'y' does not have one value per row"):
        im.predict_on_batch(X, n=3)
    with pytest.warns(OutputWarning, match="'y' does not have one value per row"):
        im.sample_prior_predictive_on_batch(X, n=3, num_samples=2)
    # A loader streams it when each device evaluates as many rows as the count
    dt = im.predict(
        _iter_batches(X, [3 * device_count()] * 2),
        n=3,
        progress=False,
    )
    assert dt.posterior_predictive["y"].shape == (1, 5, 6 * device_count())


def test_predict_warns_on_unknown_return_site(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """`.predict()` warns on a return site absent from the kernel trace."""
    X, _ = synthetic_data
    with pytest.warns(OutputWarning, match=r"not seen in any trace so far: 'typo'"):
        im_lm_svi_fitted.predict(X=X, return_sites="typo", progress=False)


def test_predict_after_cleanup(
    synthetic_data: tuple[Array, Array],
    tmp_path: Path,
) -> None:
    """A model recreates its temporary directory after `cleanup`."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=make_svi(lm))
    im.fit_on_batch(X, y, num_steps=10, num_samples=10, progress=False)
    # Fifty rows do not split over the three host devices
    msg = (
        r"The `batch_size` \(\d+\) is not divisible by the number of devices \(\d+\)\."
    )
    with pytest.warns(PerformanceWarning, match=msg):
        im.predict(X, batch_size=len(X) // 2, progress=False)
    temp_dir_before = im.temp_dir
    assert "Inference method: SVI" in str(im)
    assert "inference_method=SVI" in repr(im)
    im.cleanup()
    im.predict(X, batch_size=99, progress=False)
    assert im.temp_dir is not None
    assert im.temp_dir != temp_dir_before
    im.cleanup()

    # `.sample_posterior_predictive()` is an alias for `.predict()`.
    dt = im.sample_posterior_predictive(
        X,
        return_sites=["y"],
        batch_size=99,
        output_dir=tmp_path,
        progress=False,
    )
    assert set(dt.posterior_predictive.data_vars) == {"y"}


def test_predict_per_observation_intervention() -> None:
    """A per-observation intervention array is batched and sharded with ``X``."""
    X = np.arange(40, dtype=np.float32).reshape(20, 2) / 40
    im = ImpactModel(
        latent_intervention_model,
        rng_key=random.key(0),
        inference=make_svi(latent_intervention_model),
    )
    im.set_posterior_sample({"w": np.zeros(5)})
    intervention = {"z": np.linspace(-1.0, 1.0, 20), "w": 0.5}
    streamed = im.predict(
        X,
        intervention=intervention,
        return_sites="mu",
        batch_size=6,
        store="memory",
        progress=False,
    )
    whole = im.predict_on_batch(X, intervention=intervention, return_sites="mu")

    np.testing.assert_array_equal(
        streamed["posterior_predictive"]["mu"].values,
        whole["posterior_predictive"]["mu"].values,
    )


@pytest.mark.filterwarnings("ignore:subsample_size does not match")
def test_predict_subsample_plate_keeps_row_order() -> None:
    """Rows gathered through a plate subsampling all of them stay in order."""

    def kernel(X: Array, y: Array | None = None, k: int | None = None) -> None:
        b = sample("b", dist.Normal())
        with plate("obs", X.shape[0], subsample_size=k) as idx:
            mu = deterministic("mu", b * X[idx])
            sample("y", dist.Normal(mu, 1.0), obs=y)

    X = jnp.arange(30.0)
    im = ImpactModel(kernel, rng_key=random.key(0), inference=make_svi(kernel))
    im.set_posterior_sample({"b": jnp.ones(4)})
    # Each device's share of the rows is smaller than the subsample size
    dt = im.predict(
        X, k=len(X), return_sites="mu", batch_size=12, store="memory", progress=False
    )

    np.testing.assert_array_equal(
        dt["posterior_predictive"]["mu"].values[0], np.tile(X, (4, 1))
    )


def test_predict_intervention_reuses_compiled_program(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Repeated calls with an equal intervention reuse the compiled program."""
    X, _ = synthetic_data
    im = im_lm_svi_fitted
    im.predict(X, intervention={"b": 1.0}, store="memory", progress=False)
    fn = im._streamer._fn_cache["predict", "obs", 0, 0]
    cache_size = fn._cache_size()
    im.predict(X, intervention={"b": 1.0}, store="memory", progress=False)

    assert fn._cache_size() == cache_size


@pytest.mark.parametrize("shard_axis", ["obs", "draw"])
def test_predict_param_site(
    synthetic_data: tuple[Array, Array],
    shard_axis: str,
) -> None:
    """Predictive methods use the `param` values and mutable state learned by SVI."""
    X, y = synthetic_data

    def kernel(X: Array, y: Array | None = None) -> None:
        w = param("w", jnp.zeros(X.shape[1]))
        state = mutable("state", {"b": jnp.zeros(())})
        if y is not None:
            state["b"] = jnp.ones(())
        sigma = sample("sigma", dist.Exponential(1.0))
        mu = deterministic("mu", jnp.dot(X, w) + state["b"])
        sample("y", dist.Normal(mu, sigma), obs=y)

    im = ImpactModel(kernel, rng_key=random.key(42), inference=make_svi(kernel))
    im.fit_on_batch(X, y, num_steps=10, num_samples=4, progress=False)
    w = im.vi_result.params["w"]
    assert w.any()
    mu = np.broadcast_to(jnp.dot(X, w) + 1.0, (4, len(X)))

    batch_size = 30 if shard_axis == "obs" else 4
    on_batch = im.predict_on_batch(X, return_sites="mu", return_datatree=False)
    np.testing.assert_allclose(on_batch["mu"], mu, rtol=1e-5)
    dt = im.predict(
        X,
        return_sites="mu",
        shard_axis=shard_axis,
        batch_size=batch_size,
        store="memory",
        progress=False,
    )
    np.testing.assert_allclose(dt.posterior_predictive["mu"].values[0], mu, rtol=1e-5)
    dt = im.log_likelihood(
        X,
        y,
        shard_axis=shard_axis,
        batch_size=batch_size,
        store="memory",
        progress=False,
    )
    np.testing.assert_allclose(
        dt.log_likelihood["y"].values[0],
        dist.Normal(mu, im.posterior["sigma"][:, None]).log_prob(y),
        rtol=1e-4,
    )

    # A refit passes the new values to the same compiled program
    fn = im._streamer._fn_cache["predict", shard_axis, 0, 0]
    cache_size = fn._cache_size()
    im.fit_on_batch(X, y, num_steps=10, num_samples=4, progress=False)
    dt = im.predict(
        X,
        return_sites="mu",
        shard_axis=shard_axis,
        batch_size=batch_size,
        store="memory",
        progress=False,
    )
    np.testing.assert_allclose(
        dt.posterior_predictive["mu"].values[0],
        np.broadcast_to(jnp.dot(X, im.vi_result.params["w"]) + 1.0, (4, len(X))),
        rtol=1e-5,
    )
    assert fn._cache_size() == cache_size

    # Prior predictive sampling keeps the initial values
    prior = im.sample_prior_predictive_on_batch(
        X,
        num_samples=2,
        return_sites="mu",
        return_datatree=False,
    )
    assert not prior["mu"].any()


def test_predict_generator_matches_array(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """A one-shot generator of regular batches reproduces the array result exactly."""
    X, _ = synthetic_data
    im = im_lm_svi_fitted
    rng_key = random.key(7)
    ref = im.predict(X, rng_key=rng_key, batch_size=6, progress=False)
    # The last batch of 4 is padded to the 3 devices, and the store appends
    via = im.predict(
        _iter_batches(X, sizes=[6] * 16 + [4]), rng_key=rng_key, progress=False
    )

    np.testing.assert_array_equal(
        ref.posterior_predictive["y"].values,
        via.posterior_predictive["y"].values,
    )


@pytest.mark.parametrize(
    ("X", "exc", "match"),
    [
        ({"X": np.ones((4, 10))}, TypeError, "must be an array-like or a data loader"),
        ([np.ones((4, 10))], TypeError, "must be an array-like or a data loader"),
        ([], ValueError, "at least one nonempty batch"),
        (
            [{"X": np.ones((4, 10))}, {"X": np.ones((4, 10)), "y": np.ones(4)}],
            ValueError,
            "must remain consistent",
        ),
    ],
    ids=["mapping", "arrays", "empty", "inconsistent"],
)
def test_predict_loader_batch_contract(
    X: object,
    exc: type[Exception],
    match: str,
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """An input that is neither an array nor a loader of consistent batches raises."""
    with pytest.raises(exc, match=match):
        im_lm_svi_fitted.predict(X, store="memory", progress=False)


def test_predict_loader_binds_fields_by_name(
    synthetic_data: tuple[Array, Array],
    im_lm_with_kwargs_svi_fitted: ImpactModel,
) -> None:
    """A loader's extra array field binds by name; misused fields raise."""
    X, y = synthetic_data
    im = im_lm_with_kwargs_svi_fitted
    rng_key = random.key(7)
    ref = im.predict(X, c=y, rng_key=rng_key, batch_size=3, progress=False)
    loader = ArrayLoader(ArrayDataset(X=X, c=y), rng_key=random.key(0), batch_size=3)
    via = im.predict(loader, rng_key=rng_key, progress=False)
    np.testing.assert_allclose(
        ref.posterior_predictive["y"].values,
        via.posterior_predictive["y"].values,
        rtol=1e-5,
    )

    with pytest.raises(ValueError, match="also fields of the data loader"):
        im.predict(loader, c=y, progress=False)
    no_input = ArrayLoader(ArrayDataset(c=y), rng_key=random.key(0))
    with pytest.raises(ValueError, match="no field named 'X'"):
        im.predict(no_input, progress=False)


def test_predict_container_keyword_argument() -> None:
    """A container keyword argument is passed whole, a ragged one included."""

    def kernel(X: Array, pair: tuple, y: Array | None = None) -> None:
        offset, weights = pair
        b = sample("b", dist.Normal())
        with plate("obs", X.shape[0]):
            mu = deterministic("mu", b * X[:, 0] + offset.sum() + weights.sum())
            sample("y", dist.Normal(mu, 1.0), obs=y)

    X = np.linspace(-1.0, 1.0, 12).reshape(12, 1)
    pair = (np.ones(3), np.ones((2, 2)))
    im = ImpactModel(kernel, rng_key=random.key(0), inference=make_svi(kernel))
    im.fit_on_batch(X, X[:, 0], pair=pair, num_steps=5, num_samples=5, progress=False)
    dt = im.predict(X, pair=pair, store="memory", progress=False)
    np.testing.assert_allclose(
        dt.posterior_predictive["mu"],
        im.predict_on_batch(X, pair=pair).posterior_predictive["mu"],
        rtol=1e-6,
    )
    ragged_rows = [np.ones(i % 3 + 1) for i in range(len(X))]
    with pytest.raises(ValueError, match="one entry per observation"):
        im.predict(X, pair=ragged_rows, store="memory", progress=False)


def test_predict_blocked_intervention() -> None:
    """An intervention that the posterior draws of a latent site block warns."""

    def centered(X: Array, y: Array | None = None) -> None:
        z = sample("z", dist.Normal(0.0, 1.0))
        m = sample("m", dist.Normal(3.0 * z, 0.1))
        with plate("obs", size=X.shape[0]):
            mu = deterministic("mu", m + X[:, 0])
            sample("y", dist.Normal(mu, 0.1), obs=y)

    X = np.linspace(-1.0, 1.0, 20).reshape(20, 1)
    y = 3.0 + X[:, 0]
    im = ImpactModel(centered, rng_key=random.key(0), inference=make_svi(centered))
    im.fit_on_batch(X, y, num_steps=10, num_samples=5, progress=False)
    # `z` reaches `y` only through `m`, whose posterior draws block it
    msg = "reach the output site 'y' only through"
    with pytest.warns(OutputWarning, match=msg):
        im.predict(X, intervention={"z": 0.0}, store="memory", progress=False)
    with pytest.warns(OutputWarning, match=msg):
        im.predict_on_batch(X, intervention={"z": 0.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        im.predict_on_batch(X, intervention={"m": 0.0})
    # A requested site that is intervened on comes back with its own draws
    with pytest.warns(OutputWarning, match="'m' are intervened on"):
        dt = im.predict_on_batch(X, intervention={"m": 0.0}, return_sites=["m", "mu"])
    assert (dt.posterior_predictive["m"] != 0).all()
    # Once `m` is drawn from the prior instead of the posterior, `z` passes through
    with pytest.warns(OutputWarning, match="no draws for the latent site"):
        im.set_posterior_sample({"z": np.zeros(5)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        im.predict_on_batch(X, intervention={"z": 0.0})


def test_predict_condition(monkeypatch: pytest.MonkeyPatch) -> None:
    """A conditioned output gives an autoregressive recursion its observed history."""

    def ar1(X: Array, y: Array | None = None) -> None:
        phi = sample("phi", dist.Uniform(-1.0, 1.0))

        def step(prev: Array, inputs: tuple[Array, Array | None]) -> tuple:
            x, y_t = inputs
            mu = deterministic("mu", phi * prev + 0.0 * x)
            return sample("y", dist.Normal(mu, 0.1), obs=y_t), None

        scan(step, 0.0, (X[:, 0], y))

    X = np.zeros((12, 1), dtype=np.float32)
    y = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    im = ImpactModel(ar1, rng_key=random.key(0), inference=make_svi(ar1))
    im.fit_on_batch(X, y, num_steps=10, num_samples=4, progress=False)
    im.set_posterior_sample({"phi": np.full(4, 0.5)})
    expected = np.broadcast_to(0.5 * np.concatenate([[0.0], y[:-1]]), (1, 4, 12))
    dt = im.predict_on_batch(X, condition={"y": y}, return_sites="mu")
    np.testing.assert_allclose(dt.posterior_predictive["mu"].values, expected)
    with pytest.warns(PerformanceWarning, match="computes across observations"):
        dt = im.predict(
            X,
            condition={"y": y},
            return_sites="mu",
            store="memory",
            progress=False,
        )
    np.testing.assert_allclose(dt.posterior_predictive["mu"].values, expected)
    with pytest.raises(ValueError, match="Condition site"):
        im.predict_on_batch(X, condition={"mu": y})
    # A value meant for a kernel argument named `condition` is not a mapping
    with pytest.raises(TypeError, match="rename it"):
        im.predict_on_batch(X, condition=y)
    with pytest.warns(OutputWarning, match="keep their posterior draws"):
        im.predict_on_batch(X, condition={"phi": 0.0})

    # Without `scan`, a value with one entry per observation keeps the rows whole
    X = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(12, 2)
    im = ImpactModel(lm, rng_key=random.key(0), inference=make_svi(lm))
    im.fit_on_batch(X, y, num_steps=10, num_samples=4, progress=False)
    with pytest.warns(PerformanceWarning, match="`condition` values of site"):
        dt = im.predict(X, condition={"y": y}, store="memory", progress=False)
    np.testing.assert_array_equal(
        dt.posterior_predictive["y"].values,
        np.broadcast_to(y, (1, 4, 12)),
    )
    # On one device, a whole batch splits nothing, so `draw` runs without a warning
    monkeypatch.setattr("aimz.model.impact_model.local_device_count", lambda: 1)
    im = ImpactModel(lm, rng_key=random.key(0), inference=make_svi(lm))
    im.fit_on_batch(X, y, num_steps=10, num_samples=4, progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dt = im.predict(X, condition={"y": y}, store="memory", progress=False)
    assert dt.posterior_predictive.attrs["shard_axis"] == "draw"
    np.testing.assert_array_equal(
        dt.posterior_predictive["y"].values,
        np.broadcast_to(y, (1, 4, 12)),
    )


def test_predict_untraced_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """A kernel that the call traces first is planned like a fitted one."""

    def gp(X: Array, y: Array | None = None) -> None:
        ls = sample("ls", dist.LogNormal(0.0, 0.3))
        cov = jnp.exp(-0.5 * ((X[:, None] - X[None, :]) / ls) ** 2)
        k = cov + 1e-2 * jnp.eye(X.shape[0])
        sample("y", dist.MultivariateNormal(jnp.zeros(X.shape[0]), k), obs=y)

    X = np.linspace(0.0, 3.0, 12, dtype=np.float32)
    im = ImpactModel(gp, rng_key=random.key(0), inference=make_svi(gp))
    im.set_posterior_sample({"ls": np.ones(4)})
    # One event over all the rows keeps them whole
    with pytest.warns(PerformanceWarning, match="computes across observations"):
        dt = im.predict(X, store="memory", progress=False)
    assert dt.posterior_predictive.attrs["shard_axis"] == "draw"
    loader = ArrayLoader(
        ArrayDataset(X=X),
        rng_key=random.key(1),
        batch_size=6,
        shuffle=False,
    )
    with pytest.warns(OutputWarning, match="batches of a data loader split"):
        im.predict(loader, store="memory", progress=False)

    # A deterministic site of an imported posterior is recomputed, so nothing reruns
    def linear(X: Array, y: Array | None = None) -> None:
        w = sample("w", dist.Normal(0.0, 1.0).expand([2]).to_event(1))
        with plate("obs", X.shape[0]):
            sample("y", dist.Normal(deterministic("mu", X @ w), 1.0), obs=y)

    X = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(12, 2)
    w = np.ones((4, 2), dtype=np.float32)
    im = ImpactModel(linear, rng_key=random.key(0), inference=make_svi(linear))
    im.set_posterior_sample({"w": w, "mu": w @ X.T})
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        dt = im.predict(X, batch_size=6, store="memory", progress=False)
    assert dt.posterior_predictive.attrs["shard_axis"] == "obs"

    # On one device, one batch of an array loader splits nothing, so it does not warn
    monkeypatch.setattr("aimz.model.impact_model.local_device_count", lambda: 1)
    X = np.linspace(0.0, 3.0, 12, dtype=np.float32)
    im = ImpactModel(gp, rng_key=random.key(0), inference=make_svi(gp))
    im.set_posterior_sample({"ls": np.ones(4)})
    im.predict(X, store="memory", progress=False)
    loader = ArrayLoader(
        ArrayDataset(X=X),
        rng_key=random.key(1),
        batch_size=12,
        shuffle=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", OutputWarning)
        im.predict(loader, store="memory", progress=False)


def test_predict_global_event() -> None:
    """An event over the rows keeps its mark; a global one as long as them does not."""

    # A kernel that does not run on more rows keeps the mark of an event over them
    def anchored(X: Array, y: Array | None = None) -> None:
        ls = sample("ls", dist.LogNormal(0.0, 0.3))
        cov = jnp.exp(-0.5 * ((X[:, None] - X[None, :]) / ls) ** 2)
        k = cov + 1e-2 * jnp.eye(X.shape[0])
        sample("y", dist.MultivariateNormal(jnp.zeros(X.shape[0]), k), obs=y)
        # Offsets from a fixed grid of anchor points, which only twelve rows fit
        deterministic("offset", X - jnp.linspace(0.0, 3.0, 12))

    X = np.linspace(0.0, 3.0, 12, dtype=np.float32)
    im = ImpactModel(anchored, rng_key=random.key(0), inference=make_svi(anchored))
    im.set_posterior_sample({"ls": np.ones(4)})
    with pytest.warns(PerformanceWarning, match="computes across observations"):
        im.predict(X, store="memory", progress=False)

    # A global event as long as the first call's rows is no event over them
    def weights(X: Array, y: Array | None = None) -> None:
        w = sample("w", dist.MultivariateNormal(jnp.zeros(6), jnp.eye(6)))
        with plate("obs", X.shape[0]):
            sample("y", dist.Normal(deterministic("mu", X @ w), 1.0), obs=y)

    im = ImpactModel(weights, rng_key=random.key(0), inference=make_svi(weights))
    im.set_posterior_sample({"w": np.ones((4, 6))})
    with pytest.warns(PerformanceWarning, match="as long as the input's rows"):
        im.predict(np.ones((6, 6)), return_sites="mu", store="memory", progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        dt = im.predict(
            np.ones((12, 6)),
            return_sites="mu",
            store="memory",
            progress=False,
        )
    assert dt.posterior_predictive.attrs["shard_axis"] == "obs"

    # An event over the rows sized by a covariance factor passed whole keeps its mark
    def factored(X: Array, y: Array | None = None, factor: Array | None = None) -> None:
        s = sample("s", dist.HalfNormal(1.0))
        sample("y", dist.MultivariateNormal(s * jnp.sin(X), scale_tril=factor), obs=y)

    X = np.linspace(0.0, 3.0, 12, dtype=np.float32)
    cov = np.exp(-0.5 * (X[:, None] - X[None, :]) ** 2) + 1e-2 * np.eye(12)
    im = ImpactModel(factored, rng_key=random.key(0), inference=make_svi(factored))
    im.set_posterior_sample({"s": np.ones(4)})
    with pytest.warns(PerformanceWarning, match="computes across observations"):
        dt = im.predict(
            X,
            factor=np.linalg.cholesky(cov).astype(np.float32),
            store="memory",
            progress=False,
        )
    assert dt.posterior_predictive.attrs["shard_axis"] == "draw"


def test_predict_recomputed_sites() -> None:
    """A deterministic site stays off the devices only in a call that recomputes it."""

    # A latent that an argument fixed in the first call, as long as its rows
    def switched(X: Array, y: Array | None = None, fixed: float | None = None) -> None:
        g = (
            deterministic("g", jnp.full(3, fixed))
            if fixed is not None
            else sample("g", dist.Normal().expand([3]).to_event(1))
        )
        with plate("obs", X.shape[0]):
            mu = deterministic("mu", X + g.sum())
            sample("y", dist.Normal(mu, 1.0), obs=y)

    im = ImpactModel(switched, rng_key=random.key(0), inference=make_svi(switched))
    im.set_posterior_sample({"g": np.ones((4, 3))})
    im.predict(
        np.zeros(3), fixed=0.0, return_sites="mu", store="memory", progress=False
    )
    X = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    # A repeated call reuses the outcome of its trace
    for _ in range(2):
        dt = im.predict(X, return_sites="mu", store="memory", progress=False)
        np.testing.assert_allclose(
            dt.posterior_predictive["mu"].values,
            np.broadcast_to(X + 3.0, (1, 4, 12)),
        )

    # A call whose own trace fails, as on a batch past a subsampling plate, keeps all
    def subsampled(X: Array, y: Array | None = None) -> None:
        b = sample("b", dist.Normal())
        with plate("data", size=12, subsample_size=X.shape[0]):
            mu = deterministic("mu", b + X)
            sample("y", dist.Normal(mu, 1.0), obs=y)

    X = np.linspace(0.0, 1.0, 24, dtype=np.float32)
    im = ImpactModel(subsampled, rng_key=random.key(0), inference=make_svi(subsampled))
    im.set_posterior_sample({"b": np.ones(4), "mu": np.ones((4, 24))})
    im.predict(X[:12], return_sites="mu", store="memory", progress=False)
    loader = ArrayLoader(
        ArrayDataset(X=X),
        rng_key=random.key(1),
        batch_size=24,
        shuffle=False,
    )
    dt = im.predict(loader, return_sites="mu", store="memory", progress=False)
    np.testing.assert_allclose(
        dt.posterior_predictive["mu"].values,
        np.broadcast_to(1.0 + X, (1, 4, 24)),
        rtol=1e-6,
    )
