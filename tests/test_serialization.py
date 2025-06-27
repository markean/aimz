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
from typing import TYPE_CHECKING

import dill
import pytest
from conftest import lm
from jax import random

from aimz.model import ImpactModel

if TYPE_CHECKING:
    import jax
    from numpyro.infer import SVI


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_save_load(
    synthetic_data: tuple["jax.Array", "jax.Array"],
    vi: "SVI",
    tmp_path: "Path",
) -> None:
    """Test saving and loading an ImpactModel without errors."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), vi=vi)
    im.fit(X=X, y=y, progress=False)

    p = tmp_path / "model.dill"
    with Path.open(p, "wb") as f:
        dill.dump(im, f)
    del im  # Simulate reloading from scratch
    with Path.open(p, "rb") as f:
        im = dill.load(f)

    assert isinstance(im, ImpactModel)
