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
from jax import Array, random
from numpyro import deterministic, param, plate, sample
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


def test_predict_rejects_unsupported_size(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`.predict()` rejects return sites whose axis-1 size doesn't match the batch.

    Streaming requires every return site to emit an axis-1 size equal to the input
    batch size; ``sigma`` in ``lm`` does not satisfy this.
    """
    X, _ = synthetic_data
    # Force the single-device (unsharded) path by swapping in a mesh-less streamer.
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
    with pytest.raises(NotImplementedError, match="match the input batch size"):
        im_lm_svi_fitted.predict(
            X=X,
            return_sites="sigma",
            batch_size=3,
            progress=False,
        )


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
    """A one-shot generator of regular batches reproduces the array result exactly.

    The last batch of 4 is padded to the 3 host devices and trimmed on the way out,
    and the persistent store takes the append strategy since the generator has no
    length.
    """
    X, _ = synthetic_data
    im = im_lm_svi_fitted
    rng_key = random.key(7)
    ref = im.predict(X, rng_key=rng_key, batch_size=6, progress=False)
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
    )

    with pytest.raises(ValueError, match="also fields of the data loader"):
        im.predict(loader, c=y, progress=False)
    no_input = ArrayLoader(ArrayDataset(c=y), rng_key=random.key(0))
    with pytest.raises(ValueError, match="no field named 'X'"):
        im.predict(no_input, progress=False)


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
    # `z` reaches `y` only through `m`, whose posterior draws do not respond to it,
    # while an intervention on `m` itself takes effect
    msg = "reach the output site 'y' only through"
    with pytest.warns(OutputWarning, match=msg):
        im.predict(X, intervention={"z": 0.0}, store="memory", progress=False)
    with pytest.warns(OutputWarning, match=msg):
        im.predict_on_batch(X, intervention={"z": 0.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        im.predict_on_batch(X, intervention={"m": 0.0})
    # Once `m` is drawn from the prior instead of the posterior, `z` passes through
    with pytest.warns(OutputWarning, match="no draws for the latent site"):
        im.set_posterior_sample({"z": np.zeros(5)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        im.predict_on_batch(X, intervention={"z": 0.0})
