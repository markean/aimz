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

import cloudpickle

from aimz import ImpactModel


def test_save_load(im_lm_svi_fitted: ImpactModel, tmp_path: Path) -> None:
    """Test saving and loading an ImpactModel without errors."""
    p = tmp_path / "model.pkl"
    with p.open("wb") as f:
        cloudpickle.dump(im_lm_svi_fitted, f)
    with p.open("rb") as f:
        im = cloudpickle.load(f)

    assert isinstance(im, ImpactModel)
