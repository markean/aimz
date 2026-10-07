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

"""Tests for the `.cleanup_models()` method."""

import cloudpickle
from jax import Array, random

from aimz import ImpactModel
from tests.conftest import lm, make_svi


def test_cleanup_models_reaches_every_instance(
    synthetic_data: tuple[Array, Array],
) -> None:
    """Class-level cleanup removes the temporary directory of every registered model."""
    X, y = synthetic_data
    models = [
        ImpactModel(lm, rng_key=random.key(42), inference=make_svi(lm))
        for _ in range(2)
    ]
    for im in models:
        im.fit_on_batch(X=X, y=y, num_steps=10, num_samples=10, progress=False)
    # A cloudpickle-restored model must register itself again
    models.append(cloudpickle.loads(cloudpickle.dumps(models[0])))
    assert models[-1] in ImpactModel._models

    for im in models:
        im.predict(X, batch_size=99, progress=False)
        assert im.temp_dir is not None
    ImpactModel.cleanup_models()
    assert all(im.temp_dir is None for im in models)
