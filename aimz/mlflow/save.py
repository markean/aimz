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

"""MLflow model saving for aimz."""

from __future__ import annotations

import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from mlflow import pyfunc
from mlflow.models import (
    Model,
    ModelInputExample,
    ModelSignature,
    infer_pip_requirements,
    infer_signature,
)
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.autologging_utils import (
    INPUT_EXAMPLE_SAMPLE_ROWS,
)
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

if TYPE_CHECKING:
    from collections.abc import Iterable

    from aimz.model.impact_model import ImpactModel
FLAVOR_NAME = "aimz"


def get_default_conda_env(*, include_cloudpickle: bool = False) -> dict[str, object]:
    """Return the default Conda environment for saving an aimz model.

    Args:
        include_cloudpickle: If ``True``, include ``cloudpickle`` in the environment.

    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`~aimz.mlflow.save_model` and :func:`~aimz.mlflow.log_model`.
    """
    return _mlflow_conda_env(
        additional_pip_deps=get_default_pip_requirements(
            include_cloudpickle=include_cloudpickle,
        ),
    )


def get_default_pip_requirements(*, include_cloudpickle: bool = False) -> list[str]:
    """Return the default pip requirements for saving an aimz model.

    Args:
        include_cloudpickle: If ``True``, include ``cloudpickle`` in the requirements
            list.

    Returns:
        The default pip requirements for MLflow Models produced by calls to
        :func:`~aimz.mlflow.save_model` and :func:`~aimz.mlflow.log_model`.
    """
    out = [_get_pinned_requirement("aimz")]
    if include_cloudpickle:
        out.append(_get_pinned_requirement("cloudpickle"))

    return out


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    model: ImpactModel,
    path: str | Path,
    conda_env: dict | None = None,
    code_paths: list | None = None,
    mlflow_model: Model | None = None,
    signature: ModelSignature | None = None,
    input_example: ModelInputExample = None,
    pip_requirements: Iterable[str] | str | None = None,
    extra_pip_requirements: Iterable[str] | str | None = None,
    metadata: dict | None = None,
) -> None:
    """Save an aimz model to a path on the local file system.

    Args:
        model: An aimz model (an instance of :class:`~aimz.ImpactModel`) to be saved.
        path (str | Path): Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :class:`mlflow.models.Model` this flavor is being added to.
        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
    """
    import cloudpickle

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = Path(path).resolve()
    _validate_and_prepare_target_save_path(path)
    model_data_subpath = "model.pkl"
    model_data_path = path / model_data_subpath
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    with model_data_path.open("wb") as out:
        cloudpickle.dump(model, out)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(
        mlflow_model,
        input_example=input_example,
        path=str(path),
    )
    if signature is None and saved_example is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            signature = infer_signature(
                input_example,
                params={
                    "batch_size": INPUT_EXAMPLE_SAMPLE_ROWS,
                    "output_dir": temp_dir,
                    "progress": False,
                },
            )
    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata

    flavor_options = {
        "version": version("aimz"),
        "model_class": _get_fully_qualified_class_name(model),
        "serialization_format": "cloudpickle",
        "pickled_model": model_data_subpath,
    }
    mlflow_model.add_flavor(FLAVOR_NAME, code=code_dir_subpath, **flavor_options)
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="aimz.mlflow.pyfunc",
        data=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(path / MLMODEL_FILE_NAME)

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements(include_cloudpickle=True)
            # To ensure `_load_pyfunc` can successfully load the model during the
            # dependency inference, `mlflow_model.save` must be called beforehand to
            # save an MLmodel file.
            inferred_reqs = infer_pip_requirements(
                path,
                flavor=FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with Path.open(path / _CONDA_ENV_FILE_NAME, "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary.
    if pip_constraints:
        write_to(path / _CONSTRAINTS_FILE_NAME, "\n".join(pip_constraints))

    # Save `requirements.txt`.
    write_to(path / _REQUIREMENTS_FILE_NAME, "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(path / _PYTHON_ENV_FILE_NAME)
