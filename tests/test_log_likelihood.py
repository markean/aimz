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

"""Tests for the `.log_likelihood()` method."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, random

from aimz import ImpactModel
from tests.conftest import lm_subsample, make_svi


def test_log_likelihood_subsampled_kernel(synthetic_data: tuple[Array, Array]) -> None:
    """A kernel subsampling in a plate scores every row; a batch past its plate raises.

    Subsample indices are pinned deterministically rather than drawn at random, so the
    bare (unseeded) kernel traces cleanly on both sharding paths.
    """
    X, y = synthetic_data
    im = ImpactModel(
        lm_subsample,
        rng_key=random.key(42),
        inference=make_svi(lm_subsample),
    )
    im.fit(X, y, batch_size=20, progress=False)
    try:
        out = im.log_likelihood(X, y, batch_size=30, progress=False)
        out_draw = im.log_likelihood(
            X, y, shard_axis="draw", batch_size=250, progress=False
        )
        assert out.log_likelihood["y"].shape == (1, 1000, len(X))
        assert np.isfinite(out.log_likelihood["y"].values).all()
        # Plate indices never gather here, so the sharding paths agree
        np.testing.assert_allclose(
            out.log_likelihood["y"].values,
            out_draw.log_likelihood["y"].values,
            atol=1e-6,
        )
        with pytest.raises(ValueError, match="declares size=100"):
            im.log_likelihood(
                jnp.tile(X, (6, 1)), jnp.tile(y, 6), batch_size=600, progress=False
            )
    finally:
        im.cleanup()


def test_log_likelihood_without_posterior(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without posterior draws, both strategies score a single draw."""
    X, y = synthetic_data
    monkeypatch.setattr(im_lm_svi_fitted, "_posterior", None)
    for shard_axis in ("obs", "draw"):
        out = im_lm_svi_fitted.log_likelihood(
            X, y, shard_axis=shard_axis, batch_size=99, progress=False
        )
        assert out.log_likelihood["y"].sizes["draw"] == 1


def test_log_likelihood_requires_output(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """The observed output is required: `y` for an array, a field for a loader."""
    X, _ = synthetic_data
    with pytest.raises(ValueError, match="`y` is required"):
        im_lm_svi_fitted.log_likelihood(X, progress=False)
    with pytest.raises(ValueError, match="requires the observed output field 'y'"):
        im_lm_svi_fitted.log_likelihood(
            iter([{"X": X}]), store="memory", progress=False
        )
