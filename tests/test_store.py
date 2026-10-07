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

"""Tests for the `store` option of the streaming methods."""

from itertools import product
from pathlib import Path
from queue import Queue

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
import xarray as xr
from jax import Array, random
from numpyro import deterministic, plate, sample
from zarr.errors import ChunkNotFoundError

from aimz import ImpactModel
from aimz.utils._output import _MemoryWriteStrategy
from tests.conftest import lm, make_svi

GROUPS = {
    "predict": "posterior_predictive",
    "sample_prior_predictive": "prior_predictive",
    "log_likelihood": "log_likelihood",
}


@pytest.mark.parametrize("shard_axis", ["obs", "draw"])
@pytest.mark.parametrize(
    "method", ["predict", "sample_prior_predictive", "log_likelihood"]
)
def test_persistent_matches_memory(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    method: str,
    shard_axis: str,
) -> None:
    """Both stores return the same tree with the same chunks under either strategy."""
    X, y = synthetic_data
    kwargs: dict[str, object] = {
        "shard_axis": shard_axis,
        "batch_size": 30,
        "progress": False,
    }
    if method == "log_likelihood":
        args = (X, y)
    else:
        args = (X,)
        kwargs["rng_key"] = random.key(7)
    persistent = getattr(im_lm_svi_fitted, method)(*args, **kwargs)
    memory = getattr(im_lm_svi_fitted, method)(*args, store="memory", **kwargs)

    xr.testing.assert_equal(persistent, memory)
    group = GROUPS[method]
    assert memory[group]["y"].chunks == persistent[group]["y"].chunks
    assert "artifact_path" not in memory.attrs
    assert "artifact_path" not in memory[group].attrs


def test_persistent_store_records_draw_attributes(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """The store carries the group's attributes, so the files alone rebuild the tree."""
    X, _ = synthetic_data
    rng_key, batch_size = random.key(7), 30
    dt = im_lm_svi_fitted.predict(
        X, rng_key=rng_key, batch_size=batch_size, progress=False
    )
    ds = xr.open_zarr(dt.attrs["artifact_path"], consolidated=False)

    assert ds.attrs["num_chains"] == 1
    assert ds.attrs["batch_size"] == batch_size
    np.testing.assert_array_equal(ds.attrs["rng_key"], random.key_data(rng_key))


def test_memory_store_leaves_filesystem_untouched(
    synthetic_data: tuple[Array, Array],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory-store calls never materialize the temp directory, even on failure."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=make_svi(lm))
    im.fit_on_batch(X=X, y=y, num_steps=10, num_samples=10, progress=False)

    assert im.temp_dir is None

    dt = im.predict(X, batch_size=30, store="memory", progress=False)

    assert im.temp_dir is None
    assert "artifact_path" not in dt.attrs

    def boom(*args: object, **kwargs: object) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(im._streamer, "write_predictive", boom)

    with pytest.raises(RuntimeError, match="boom"):
        im.predict(X, store="memory", batch_size=30, progress=False)

    assert im.temp_dir is None


def test_persistent_read_raises_after_artifact_removal(
    synthetic_data: tuple[Array, Array],
) -> None:
    """A chunk holding only the fill value reads back; a removed artifact raises."""
    X, y = synthetic_data
    t = jnp.zeros(len(X)).at[60:].set(1.0)

    def kernel(X: Array, t: Array, y: Array | None = None) -> None:
        b = sample("b", dist.Normal())
        with plate("n", size=X.shape[0]):
            # Exactly +0.0 where `t` is 0, so the first two chunks hold only the fill
            # value
            deterministic("effect", t * b**2)
            sample("y", dist.Normal(X.sum(axis=-1) + b * t, 1.0), obs=y)

    im = ImpactModel(kernel, rng_key=random.key(42), inference=make_svi(kernel))
    im.fit_on_batch(X=X, y=y, t=t, num_steps=10, num_samples=10, progress=False)
    dt = im.predict(X, t=t, batch_size=30, progress=False)

    np.testing.assert_array_equal(dt["posterior_predictive"]["effect"][..., :60], 0.0)

    im.cleanup()

    with pytest.raises(ChunkNotFoundError):
        dt["posterior_predictive"]["effect"].load()


def test_interrupted_memory_stream_releases_partial_batches(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted stream frees its retained batches despite a held traceback.

    An exception's traceback keeps the interrupted call stack (including the write
    strategy) reachable (e.g. a notebook's post-mortem state), so the failure path
    must empty the retained batches rather than rely on the frames dying.
    """
    X, _ = synthetic_data
    interrupt_at_batch = 2
    calls = {"n": 0}
    unpatched = _MemoryWriteStrategy.enqueue

    def interrupt_on_second_batch(
        self: _MemoryWriteStrategy,
        queue: Queue,
        site_arrays: dict,
    ) -> None:
        unpatched(self, queue, site_arrays)
        calls["n"] += 1
        if calls["n"] >= interrupt_at_batch:
            raise KeyboardInterrupt

    monkeypatch.setattr(_MemoryWriteStrategy, "enqueue", interrupt_on_second_batch)

    with pytest.raises(KeyboardInterrupt) as excinfo:
        im_lm_svi_fitted.predict(X, batch_size=30, store="memory", progress=False)

    # Walk the held traceback to the frames owning the strategy and verify its
    # retained batches were released.
    sinks = []
    tb = excinfo.tb
    while tb is not None:
        obj = tb.tb_frame.f_locals.get("strategy")
        if isinstance(obj, _MemoryWriteStrategy):
            sinks.append(obj.sink)
        tb = tb.tb_next

    assert sinks
    assert all(not sink for sink in sinks)


def test_store_validation(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    tmp_path: Path,
) -> None:
    """An unknown `store`, or an `output_dir` with the memory store, raises."""
    X, y = synthetic_data
    for method in (
        "predict",
        "sample_posterior_predictive",
        "sample_prior_predictive",
        "log_likelihood",
    ):
        args = (X, y) if method == "log_likelihood" else (X,)
        with pytest.raises(ValueError, match="`store` must be either"):
            getattr(im_lm_svi_fitted, method)(*args, store="rows")
        with pytest.raises(ValueError, match="`output_dir` must be `None`"):
            getattr(im_lm_svi_fitted, method)(
                *args, store="memory", output_dir=str(tmp_path)
            )


def test_estimate_effect_store_combinations(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Any combination of stores gives the same effect, recording persistent paths."""
    X, _ = synthetic_data
    rng_key = random.key(13)
    effects = {}
    for stores in product(("persistent", "memory"), repeat=2):
        effects[stores] = im_lm_svi_fitted.estimate_effect(
            args_baseline={
                "X": X,
                "rng_key": rng_key,
                "store": stores[0],
                "batch_size": 30,
                "progress": False,
            },
            args_intervention={
                "X": X,
                "intervention": {"b": 0.0},
                "rng_key": rng_key,
                "store": stores[1],
                "batch_size": 30,
                "progress": False,
            },
        )

    reference = effects["persistent", "persistent"]
    for (store_baseline, store_intervention), effect in effects.items():
        xr.testing.assert_equal(reference, effect)
        assert ("artifact_path_baseline" in effect.attrs) == (
            store_baseline == "persistent"
        )
        assert ("artifact_path_intervention" in effect.attrs) == (
            store_intervention == "persistent"
        )
