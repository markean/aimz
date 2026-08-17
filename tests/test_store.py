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

from queue import Queue

import pytest
import xarray as xr
from jax import Array, random
from numpyro.infer import SVI

from aimz import ImpactModel
from tests.conftest import lm


def test_predict_persistent_matches_memory_obs(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """`store` option returns the same tree (obs-parallel)."""
    X, _ = synthetic_data
    rng_key = random.key(7)

    dt_persistent = im_lm_svi_fitted.predict(
        X,
        rng_key=rng_key,
        batch_size=30,
        progress=False,
    )
    dt_mem = im_lm_svi_fitted.predict(
        X,
        rng_key=rng_key,
        batch_size=30,
        store="memory",
        progress=False,
    )

    xr.testing.assert_equal(dt_persistent, dt_mem)
    assert (
        dt_mem["posterior_predictive"]["y"].chunks
        == dt_persistent["posterior_predictive"]["y"].chunks
    )
    assert "artifact_path" not in dt_mem.attrs
    assert "artifact_path" not in dt_mem["posterior_predictive"].attrs


def test_predict_persistent_matches_memory_draw(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """`store` option returns the same tree (draw-parallel)."""
    X, _ = synthetic_data
    rng_key = random.key(11)

    dt_persistent = im_lm_svi_fitted.predict(
        X,
        rng_key=rng_key,
        shard_axis="draw",
        batch_size=30,
        progress=False,
    )
    dt_mem = im_lm_svi_fitted.predict(
        X,
        rng_key=rng_key,
        shard_axis="draw",
        batch_size=30,
        store="memory",
        progress=False,
    )

    xr.testing.assert_equal(dt_persistent, dt_mem)
    assert (
        dt_mem["posterior_predictive"]["y"].chunks
        == dt_persistent["posterior_predictive"]["y"].chunks
    )
    assert "artifact_path" not in dt_mem.attrs
    assert "artifact_path" not in dt_mem["posterior_predictive"].attrs


def test_sample_prior_predictive_persistent_matches_memory_obs(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """`store` option returns the same tree (obs-parallel)."""
    X, _ = synthetic_data
    rng_key = random.key(3)

    dt_persistent = im_lm_svi_fitted.sample_prior_predictive(
        X,
        num_samples=300,
        rng_key=rng_key,
        batch_size=30,
        progress=False,
    )
    dt_mem = im_lm_svi_fitted.sample_prior_predictive(
        X,
        num_samples=300,
        rng_key=rng_key,
        batch_size=30,
        store="memory",
        progress=False,
    )

    xr.testing.assert_equal(dt_persistent, dt_mem)
    assert (
        dt_mem["prior_predictive"]["y"].chunks
        == dt_persistent["prior_predictive"]["y"].chunks
    )


def test_sample_prior_predictive_persistent_matches_memory_draw(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """`store` option returns the same tree (draw-parallel)."""
    X, _ = synthetic_data
    rng_key = random.key(5)

    dt_persistent = im_lm_svi_fitted.sample_prior_predictive(
        X,
        num_samples=300,
        rng_key=rng_key,
        shard_axis="draw",
        batch_size=30,
        progress=False,
    )
    dt_mem = im_lm_svi_fitted.sample_prior_predictive(
        X,
        num_samples=300,
        rng_key=rng_key,
        shard_axis="draw",
        batch_size=30,
        store="memory",
        progress=False,
    )

    xr.testing.assert_equal(dt_persistent, dt_mem)
    assert (
        dt_mem["prior_predictive"]["y"].chunks
        == dt_persistent["prior_predictive"]["y"].chunks
    )


def test_log_likelihood_persistent_matches_memory_obs(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """`store` option returns the same tree (obs-parallel)."""
    X, y = synthetic_data

    dt_persistent = im_lm_svi_fitted.log_likelihood(
        X,
        y,
        batch_size=30,
        progress=False,
    )
    dt_mem = im_lm_svi_fitted.log_likelihood(
        X,
        y,
        batch_size=30,
        store="memory",
        progress=False,
    )

    xr.testing.assert_equal(dt_persistent, dt_mem)
    assert (
        dt_mem["log_likelihood"]["y"].chunks
        == dt_persistent["log_likelihood"]["y"].chunks
    )


def test_log_likelihood_persistent_matches_memory_draw(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """`store` option returns the same tree (draw-parallel)."""
    X, y = synthetic_data

    dt_persistent = im_lm_svi_fitted.log_likelihood(
        X,
        y,
        shard_axis="draw",
        batch_size=30,
        progress=False,
    )
    dt_mem = im_lm_svi_fitted.log_likelihood(
        X,
        y,
        shard_axis="draw",
        batch_size=30,
        store="memory",
        progress=False,
    )

    xr.testing.assert_equal(dt_persistent, dt_mem)
    assert (
        dt_mem["log_likelihood"]["y"].chunks
        == dt_persistent["log_likelihood"]["y"].chunks
    )


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_memory_store_leaves_filesystem_untouched(
    synthetic_data: tuple[Array, Array],
    vi: SVI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory-store calls never materialize the temp directory, even on failure."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    im.fit_on_batch(X=X, y=y)

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


def test_interrupted_memory_stream_releases_partial_batches(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted stream frees its retained batches despite a held traceback.

    An exception's traceback keeps the interrupted call stack — including the write
    strategy — reachable (e.g. a notebook's post-mortem state), so the failure path
    must empty the retained batches rather than rely on the frames dying.
    """
    X, _ = synthetic_data

    from aimz.utils import _output  # noqa: PLC0415

    interrupt_at_batch = 2
    calls = {"n": 0}
    unpatched = _output._MemoryWriteStrategy.enqueue

    def interrupt_on_second_batch(
        self: _output._MemoryWriteStrategy,
        queue: Queue,
        site_arrays: dict,
    ) -> None:
        unpatched(self, queue, site_arrays)
        calls["n"] += 1
        if calls["n"] >= interrupt_at_batch:
            raise KeyboardInterrupt

    monkeypatch.setattr(
        _output._MemoryWriteStrategy,
        "enqueue",
        interrupt_on_second_batch,
    )

    with pytest.raises(KeyboardInterrupt) as excinfo:
        im_lm_svi_fitted.predict(X, batch_size=30, store="memory", progress=False)

    # Walk the held traceback to the frames owning the strategy and verify its
    # retained batches were released.
    sinks = []
    tb = excinfo.tb
    while tb is not None:
        obj = tb.tb_frame.f_locals.get("strategy")
        if isinstance(obj, _output._MemoryWriteStrategy):
            sinks.append(obj.sink)
        tb = tb.tb_next

    assert sinks
    assert all(not sink for sink in sinks)


@pytest.mark.parametrize(
    "method",
    [
        "predict",
        "sample_posterior_predictive",
        "sample_prior_predictive",
        "log_likelihood",
    ],
)
class TestValidation:
    """Test class for `store` validation across the four streaming methods."""

    @staticmethod
    def _call(
        im: ImpactModel,
        method: str,
        X: Array,
        y: Array,
        **kwargs: object,
    ) -> None:
        """Invoke one of the streaming methods with its required arguments."""
        if method == "log_likelihood":
            getattr(im, method)(X, y, **kwargs)
        else:
            getattr(im, method)(X, **kwargs)

    def test_invalid_store(
        self,
        synthetic_data: tuple[Array, Array],
        im_lm_svi_fitted: ImpactModel,
        method: str,
    ) -> None:
        """An unknown `store` raises an error."""
        X, y = synthetic_data
        with pytest.raises(ValueError, match="`store` must be either"):
            self._call(im_lm_svi_fitted, method, X, y, store="rows")

    def test_output_dir_with_memory_store(
        self,
        synthetic_data: tuple[Array, Array],
        im_lm_svi_fitted: ImpactModel,
        method: str,
        tmp_path: object,
    ) -> None:
        """`store="memory"` combined with an explicit `output_dir` raises an error."""
        X, y = synthetic_data
        with pytest.raises(ValueError, match="`output_dir` must be `None`"):
            self._call(
                im_lm_svi_fitted,
                method,
                X,
                y,
                store="memory",
                output_dir=str(tmp_path),
            )


def test_estimate_effect_store_combinations_match(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """All four `store` combinations return the same effect tree."""
    X, _ = synthetic_data
    rng_key_baseline = random.key(13)
    rng_key_intervention = random.key(17)

    effects = [
        im_lm_svi_fitted.estimate_effect(
            args_baseline={
                "X": X,
                "rng_key": rng_key_baseline,
                "store": store_baseline,
                "batch_size": 30,
                "progress": False,
            },
            args_intervention={
                "X": X,
                "intervention": {"b": 0.0},
                "rng_key": rng_key_intervention,
                "store": store_intervention,
                "batch_size": 30,
                "progress": False,
            },
        )
        for store_baseline in ("persistent", "memory")
        for store_intervention in ("persistent", "memory")
    ]

    for effect in effects[1:]:
        xr.testing.assert_equal(effects[0], effect)


def test_estimate_effect_memory_records_no_artifact_paths(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Effects from two memory-store scenarios record no artifact-path attrs."""
    X, _ = synthetic_data

    effect = im_lm_svi_fitted.estimate_effect(
        args_baseline={
            "X": X,
            "store": "memory",
            "batch_size": 30,
            "progress": False,
        },
        args_intervention={
            "X": X,
            "intervention": {"b": 0.0},
            "store": "memory",
            "batch_size": 30,
            "progress": False,
        },
    )

    assert "posterior_predictive" in effect.children
    assert "artifact_path_baseline" not in effect.attrs
    assert "artifact_path_intervention" not in effect.attrs


def test_estimate_effect_mixed_stores_records_persistent_side_only(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Mixing stores records only the persistent scenario's artifact path."""
    X, _ = synthetic_data

    effect = im_lm_svi_fitted.estimate_effect(
        args_baseline={"X": X, "batch_size": 30, "progress": False},
        args_intervention={
            "X": X,
            "intervention": {"b": 0.0},
            "store": "memory",
            "batch_size": 30,
            "progress": False,
        },
    )

    assert "artifact_path_baseline" in effect.attrs
    assert "artifact_path_intervention" not in effect.attrs
