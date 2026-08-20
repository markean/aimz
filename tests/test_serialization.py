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

from pathlib import Path
from typing import TYPE_CHECKING, cast

import cloudpickle
import numpy as np
from jax import Array, random

from aimz import ImpactModel

if TYPE_CHECKING:
    import xarray as xr


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
    for site, sample in im_lm_svi_fitted.posterior.items():
        np.testing.assert_array_equal(
            np.asarray(im.posterior[site]),
            np.asarray(sample),
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
