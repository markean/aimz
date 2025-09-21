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

"""MLflow logging for aimz models."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import ModelInfo
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring

if TYPE_CHECKING:
    from aimz.model.impact_model import ImpactModel

FLAVOR_NAME = "aimz"


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    model: "ImpactModel",
    conda_env: dict | None = None,
    code_paths: list | None = None,
    registered_model_name: str | None = None,
    signature: ModelSignature | None = None,
    input_example: ModelInputExample = None,
    await_registration_for: int | None = DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements: Iterable[str] | str | None = None,
    extra_pip_requirements: Iterable[str] | str | None = None,
    metadata: dict | None = None,
    name: str | None = None,
    params: dict[str, object] | None = None,
    model_type: str | None = None,
    model_id: str | None = None,
    step: int = 0,
    tags: dict[str, object] | None = None,
) -> ModelInfo:
    """Log an aimz model as an MLflow artifact for the current run..

    Args:
        model: An aimz model (an instance of :class:`~aimz.ImpactModel`) to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: If given, each time a model is trained, it is registered
            as a new model version of the registered model with this name. The
            registered model is created if it does not already exist.
        signature: {{ signature }}
        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to
            finish being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify ``0`` or :obj:`None` to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        name: {{ name }}
        params: {{ params }}
        model_type: {{ model_type }}
        model_id: {{ model_id }}
        step: {{ step }}
        tags: {{ tags }}

    Returns:
        A :external:class:`~mlflow.models.model.ModelInfo` instance that contains the
        metadata of the logged model.
    """
    import aimz.mlflow

    return Model.log(
        artifact_path=None,
        flavor=aimz.mlflow,
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
        metadata=metadata,
        model=model,
        name=name,
        model_type=model_type,
        params=params,
        tags=tags,
        step=step,
        model_id=model_id,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
    )
