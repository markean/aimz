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

"""MLflow pyfunc wrapper for aimz models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from mlflow.pyfunc import PyFuncModel

if TYPE_CHECKING:
    from jax.typing import ArrayLike

    from aimz.model.impact_model import ImpactModel
    from aimz.utils.data import ArrayLoader


class _ImpactModelWrapper(PyFuncModel):
    """MLflow PyFunc wrapper for ImpactModel."""

    def __init__(self, model: ImpactModel) -> None:
        self.model = model

    def get_raw_model(self) -> ImpactModel:
        """Return the underlying ImpactModel instance."""
        return self.model

    def predict(self, model_input: object, params: dict | None = None):
        """Run predictions using the wrapped ImpactModel."""
        if isinstance(model_input, dict):
            return self.model.predict(**model_input, **(params or {}))
        return self.model.predict(
            cast("ArrayLike | ArrayLoader", model_input),
            **(params or {}),
        )


def _load_pyfunc(path: str) -> _ImpactModelWrapper:
    """Load the ImpactModel as an MLflow PyFunc wrapper."""
    return _ImpactModelWrapper(_load_model(path))


def _load_model(path: str) -> ImpactModel:
    """Load an ImpactModel instance from the specified path."""
    import cloudpickle

    with Path(path).open("rb") as f:
        return cloudpickle.load(f)
