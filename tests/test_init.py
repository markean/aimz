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

"""Tests for initializing the `ImpactModel` class."""

import pytest
from conftest import lm
from jax import random

from aimz import ImpactModel


def test_unsupported_inference_method() -> None:
    """ImpactModel initialization with unsupported inference method raises TypeError."""
    with pytest.raises(TypeError, match="Unsupported inference object"):
        ImpactModel(lm, rng_key=random.key(42), inference=None)
