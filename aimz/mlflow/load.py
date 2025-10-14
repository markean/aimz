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

"""MLflow model loading for aimz."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
)

from aimz.mlflow.pyfunc import _load_model

if TYPE_CHECKING:
    from aimz.model.impact_model import ImpactModel

FLAVOR_NAME = "aimz"


def load_model(model_uri: str, dst_path: str | None = None) -> ImpactModel:
    """Load an aimz model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
            artifact-locations>`_.

        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        An aimz model (an instance of :class:`~aimz.ImpactModel`).

    """
    local_model_path = _download_artifact_from_uri(model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(local_model_path, FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    model_artifacts_path = Path(local_model_path) / flavor_conf["pickled_model"]

    return _load_model(model_artifacts_path)
