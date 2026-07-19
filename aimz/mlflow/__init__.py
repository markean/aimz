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

"""The ``aimz.mlflow`` module provides an API for logging and loading aimz models.

This module exports aimz models with the following flavors:

aimz (native) format
    This is the main flavor that can be loaded back into aimz.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import annotations

import logging
import pickle
from importlib.metadata import version
from inspect import getsource
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import mlflow
import numpy as np
import yaml
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.entities.logged_model_input import LoggedModelInput
from mlflow.environment_variables import MLFLOW_ALLOW_PICKLE_DESERIALIZATION
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _Example, _save_example
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.fluent import _initialize_logged_model
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.autologging_utils import (
    INPUT_EXAMPLE_SAMPLE_ROWS,
    MlflowAutologgingQueueingClient,
    autologging_integration,
    batch_metrics_logger,
    get_mlflow_run_params_for_fn_args,
    resolve_input_example_and_signature,
    safe_patch,
)
from mlflow.utils.databricks_utils import (
    is_in_databricks_model_serving_environment,
    is_in_databricks_runtime,
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
    _add_code_from_conf_to_system_path,
    _copy_extra_files,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

from aimz.utils._validation import _is_arraylike

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType
    from typing import IO

    from mlflow.models import ModelInputExample, ModelSignature
    from mlflow.models.model import ModelInfo
    from mlflow.tracking.fluent import ActiveRun
    from numpyro.infer.svi import SVI, SVIRunResult

    from aimz.model.impact_model import ImpactModel

FLAVOR_NAME = "aimz"

SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"

SUPPORTED_SERIALIZATION_FORMATS = [SERIALIZATION_FORMAT_CLOUDPICKLE]

_logger = logging.getLogger(__name__)


def get_default_pip_requirements(*, include_cloudpickle: bool = False) -> list[str]:
    """Return the default pip requirements for MLflow Models produced by this flavor.

    Args:
        include_cloudpickle: If ``True``, include ``cloudpickle`` in the requirements
            list.

    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("aimz")]
    if include_cloudpickle:
        pip_deps.append(_get_pinned_requirement("cloudpickle"))

    return pip_deps


def get_default_conda_env(*, include_cloudpickle: bool = False) -> dict[str, object]:
    """Return the default Conda environment for MLflow Models produced by this flavor.

    Args:
        include_cloudpickle: If ``True``, include ``cloudpickle`` in the environment.

    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(
        additional_pip_deps=get_default_pip_requirements(
            include_cloudpickle=include_cloudpickle,
        ),
    )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    model: ImpactModel,
    path: str | Path,
    conda_env: dict | None = None,
    code_paths: list | None = None,
    mlflow_model: Model | None = None,
    signature: ModelSignature | Literal[False] | None = None,
    input_example: ModelInputExample | None = None,
    pip_requirements: Iterable[str] | str | None = None,
    extra_pip_requirements: Iterable[str] | str | None = None,
    metadata: dict | None = None,
    extra_files: list | None = None,
) -> None:
    """Save an aimz model to a path on the local file system.

    Args:
        model: aimz model (an instance of :class:`~aimz.ImpactModel`) to be saved.
        path: Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        extra_files: {{ extra_files }}

    .. code-block:: python
        :caption: Example

        from pathlib import Path

        import aimz.mlflow
        from aimz import ImpactModel

        # Train the model
        im = ImpactModel(...).fit(X, y)

        # Save the model
        path = "model"
        aimz.mlflow.save_model(im, path)

        # Load model for inference
        loaded_model = aimz.mlflow.load_model(Path.cwd() / path)
        print(loaded_model.predict(X[:5]))
    """
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = Path(path).resolve()
    _validate_and_prepare_target_save_path(path)
    model_data_subpath = "model.pkl"
    model_data_path = path / model_data_subpath
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(mlflow_model, input_example, str(path))

    if signature is None and saved_example is not None:
        wrapped_model = _AimzModelWrapper(model)
        signature = _infer_signature_from_input_example(saved_example, wrapped_model)
    elif signature is False:
        signature = None

    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata

    # Save an aimz model
    _save_model(model, model_data_path, SERIALIZATION_FORMAT_CLOUDPICKLE)

    model_class = _get_fully_qualified_class_name(model)

    extra_files_config = _copy_extra_files(extra_files, path)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="aimz.mlflow",
        data=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        aimz_version=version("aimz"),
        model_class=model_class,
        serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
        code=code_dir_subpath,
        **extra_files_config,
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
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path,
                FLAVOR_NAME,
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

    with (path / _CONDA_ENV_FILE_NAME).open("w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(path / _CONSTRAINTS_FILE_NAME, "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(path / _REQUIREMENTS_FILE_NAME, "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(path / _PYTHON_ENV_FILE_NAME)


def _dump_model(pickle_lib: ModuleType, model: ImpactModel, out: IO[bytes]) -> None:
    # Using python's default protocol to optimize compatibility.
    # Otherwise cloudpickle uses latest protocol leading to incompatibilities.
    # See https://github.com/mlflow/mlflow/issues/5419
    pickle_lib.dump(model, out, protocol=pickle.DEFAULT_PROTOCOL)


def _save_model(
    model: ImpactModel,
    output_path: Path,
    serialization_format: str,
) -> None:
    """Serialize an aimz model to the specified output path.

    Args:
        model: The aimz model to serialize.
        output_path: The file path to which to write the serialized model.
        serialization_format: The format in which to serialize the model. This should
            be one of the following:
            ``aimz.mlflow.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    with output_path.open("wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            _dump_model(cloudpickle, model, out)
        else:
            msg = f"Unrecognized serialization format: {serialization_format}"
            raise MlflowException(message=msg, error_code=INTERNAL_ERROR)


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    model: ImpactModel,
    artifact_path: str | None = None,
    conda_env: dict | None = None,
    code_paths: list | None = None,
    registered_model_name: str | None = None,
    signature: ModelSignature | Literal[False] | None = None,
    input_example: ModelInputExample | None = None,
    await_registration_for: int | None = DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements: Iterable[str] | str | None = None,
    extra_pip_requirements: Iterable[str] | str | None = None,
    metadata: dict | None = None,
    extra_files: list | None = None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    model_type: str | None = None,
    step: int = 0,
    model_id: str | None = None,
    **kwargs: object,
) -> ModelInfo:
    """Log an aimz model as an MLflow artifact for the current run.

    Args:
        model: aimz model (an instance of :class:`~aimz.ImpactModel`) to be saved.
        artifact_path: Deprecated. Use `name` instead.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: If given, create a model version under
            ``registered_model_name``, also creating a registered model if one
            with the given name does not exist.
        signature: {{ signature }}
        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to
            finish being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        extra_files: {{ extra_files }}
        name: {{ name }}
        params: {{ params }}
        tags: {{ tags }}
        model_type: {{ model_type }}
        step: {{ step }}
        model_id: {{ model_id }}
        kwargs: Extra arguments to pass to :py:func:`mlflow.models.Model.log`.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains
        the metadata of the logged model.

    .. code-block:: python
        :caption: Example

        import mlflow

        import aimz.mlflow
        from aimz import ImpactModel

        # Train the model
        im = ImpactModel(...).fit(X, y)

        # Log the model
        with mlflow.start_run() as run:
            model_info = aimz.mlflow.log_model(im, name="model")

        # Fetch the logged model artifacts
        client = mlflow.MlflowClient()
        artifacts = [f.path for f in client.list_artifacts(run.info.run_id, "model")]
        print(f"artifacts: {artifacts}")

    .. code-block:: text
        :caption: Output

        artifacts: ['model/MLmodel',
                    'model/conda.yaml',
                    'model/model.pkl',
                    'model/python_env.yaml',
                    'model/requirements.txt']
    """
    import aimz.mlflow

    return Model.log(
        artifact_path=artifact_path,
        name=name,
        flavor=aimz.mlflow,
        registered_model_name=registered_model_name,
        model=model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        extra_files=extra_files,
        params=params,
        tags=tags,
        model_type=model_type,
        step=step,
        model_id=model_id,
        **kwargs,
    )


def _load_model_from_local_file(
    path: str | Path,
    serialization_format: str,
) -> ImpactModel:
    """Load an aimz model saved as an MLflow artifact on the local file system.

    Args:
        path: Local filesystem path to the MLflow Model saved with the ``aimz`` flavor
        serialization_format: The format in which the model was serialized. This should
            be one of the following:
            ``aimz.mlflow.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        msg = (
            f"Unrecognized serialization format: {serialization_format}. Please "
            f"specify one of the following supported formats: "
            f"{SUPPORTED_SERIALIZATION_FORMATS}."
        )
        raise MlflowException(message=msg, error_code=INVALID_PARAMETER_VALUE)

    if (
        not MLFLOW_ALLOW_PICKLE_DESERIALIZATION.get()
        and not is_in_databricks_runtime()
        and not is_in_databricks_model_serving_environment()
    ):
        msg = (
            "Deserializing model using pickle is disallowed, but this model is saved "
            "in pickle format. To address this issue, you need to set environment "
            "variable 'MLFLOW_ALLOW_PICKLE_DESERIALIZATION' to 'true'."
        )
        raise MlflowException(msg)

    with Path(path).open("rb") as f:
        import cloudpickle

        return cloudpickle.load(f)


def _load_model(path: str | Path) -> ImpactModel:
    """Load Model Implementation.

    Args:
        path: Local filesystem path to
            the MLflow Model's ``model.pkl`` artifact or
            the top-level MLflow Model directory.
    """
    path = Path(path)
    model_dir = path.parent if path.is_file() else path
    flavor_conf = _get_flavor_configuration(
        model_path=model_dir,
        flavor_name=FLAVOR_NAME,
    )

    aimz_model_path = model_dir / flavor_conf["pickled_model"]
    serialization_format = flavor_conf.get(
        "serialization_format",
        SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    return _load_model_from_local_file(aimz_model_path, serialization_format)


def _load_pyfunc(path: str) -> _AimzModelWrapper:
    """Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``aimz`` flavor.
    """
    return _AimzModelWrapper(_load_model(path))


def load_model(model_uri: str, dst_path: str | None = None) -> ImpactModel:
    """Load an aimz model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        An aimz model (an instance of :class:`~aimz.ImpactModel`).

    .. code-block:: python
        :caption: Example

        import aimz.mlflow

        # Load model
        im = aimz.mlflow.load_model("runs:/<mlflow_run_id>/model")

        # Make predictions; returns an xarray.DataTree of posterior predictive samples
        predictions = im.predict(X)
    """
    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri,
        output_path=dst_path,
    )
    flavor_conf = _get_flavor_configuration(local_model_path, FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    return _load_model(path=local_model_path)


class _AimzModelWrapper:
    def __init__(self, aimz_model: ImpactModel) -> None:
        self.aimz_model = aimz_model

    def get_raw_model(self) -> ImpactModel:
        """Return the underlying model."""
        return self.aimz_model

    def predict(self, data: object, params: dict[str, Any] | None = None):
        """Run predictions using the wrapped ImpactModel.

        Args:
            data: Model input data. A mapping is unpacked into keyword arguments, so
                prediction keyword arguments pass through the pyfunc boundary.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        if isinstance(data, dict):
            return self.aimz_model.predict(
                **cast("dict[str, Any]", data),
                **(params or {}),
            )
        return self.aimz_model.predict(cast("Any", data), **(params or {}))


@autologging_integration(FLAVOR_NAME)
def autolog(
    *,
    log_input_examples: bool = False,
    log_model_signatures: bool = True,
    log_models: bool = True,
    log_datasets: bool = True,
    disable: bool = False,
    exclusive: bool = False,
    disable_for_unsupported_versions: bool = False,
    silent: bool = False,
    registered_model_name: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> None:
    """Enable (or disable) and configure autologging from aimz to MLflow.

    Logs the following:

        - parameters specified in :meth:`~aimz.ImpactModel.fit` and
          :meth:`~aimz.ImpactModel.fit_on_batch`, together with ``param_input``,
          ``param_output``, ``inference_method``, and ``optimizer``.
        - the evidence lower bound (ELBO) loss on each optimization step.
        - the source code of the kernel function used by the model.
        - trained model, including:
            - an example of valid input.
            - inferred signature of the inputs and outputs of the model.

    Autologging is performed when you call :meth:`~aimz.ImpactModel.fit` or
    :meth:`~aimz.ImpactModel.fit_on_batch`.

    Args:
        log_input_examples: If ``True``, input examples from training datasets are
            collected and logged along with aimz model artifacts during training. If
            ``False``, input examples are not logged.
            Note: Input examples are MLflow model attributes
            and are only collected if ``log_models`` is also ``True``.
        log_model_signatures: If ``True``,
            :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
            describing model inputs and outputs are collected and logged along
            with aimz model artifacts during training. If ``False``,
            signatures are not logged.
            Note: Model signatures are MLflow model attributes
            and are only collected if ``log_models`` is also ``True``.
        log_models: If ``True``, trained models are logged as MLflow model artifacts.
            If ``False``, trained models are not logged.
            Input examples and model signatures, which are attributes of MLflow models,
            are also omitted when ``log_models`` is ``False``.
        log_datasets: If ``True``, train dataset information is logged to MLflow
            Tracking if applicable. If ``False``, dataset information is not logged.
        disable: If ``True``, disables the aimz autologging integration. If ``False``,
            enables the aimz autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent
            runs. If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions
            of aimz that have not been tested against this version of the MLflow client
            or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during
            aimz autologging. If ``False``, show all events and warnings during aimz
            autologging.
        registered_model_name: If given, each time a model is trained, it is registered
            as a new model version of the registered model with this name.
            The registered model is created if it does not already exist.
        extra_tags: A dictionary of extra tags to set on each managed run created by
            autologging.
    """
    from aimz.model.impact_model import ImpactModel

    def patch_fit(
        original: Callable,
        self: ImpactModel,
        *args: object,
        **kwargs: object,
    ) -> ImpactModel:
        """Patch for the fitting method to log information.

        Args:
            original (Callable): The original method.
            self (ImpactModel): The instance being fitted.
            *args (object): Positional arguments for the method.
            **kwargs (object): Keyword arguments for the method.
        """
        autologging_client = MlflowAutologgingQueueingClient()
        run_id = cast("ActiveRun", mlflow.active_run()).info.run_id

        # Log the source code of the kernel function as an artifact.
        mlflow.log_text(getsource(self.kernel), artifact_file="model.py")

        params = _run_params(self, original, args, kwargs)
        autologging_client.log_params(run_id=run_id, params=params)

        param_logging_operations = autologging_client.flush(synchronous=False)

        # Obtain a copy of a model input example from the training dataset prior to
        # model training for subsequent use during model logging, ensuring that the
        # input example and inferred model signature to not include any mutations from
        # model training.
        input_example = None
        input_example_exc = None
        try:
            input_example = _get_input_example(self, args, kwargs)
        except Exception as e:
            input_example_exc = e

        model_id = None
        if log_models:
            model_id = _initialize_logged_model(
                "model",
                params={k: str(v) for k, v in params.items()},
                flavor=FLAVOR_NAME,
            ).model_id

        # Whether to automatically log the training dataset as a dataset artifact.
        if log_datasets:
            try:
                context_tags = context_registry.resolve_tags()
                source = CodeDatasetSource(tags=context_tags)
                _log_aimz_dataset(self, args, kwargs, source, "train", model_id)
            except Exception as e:
                _logger.warning(
                    "Failed to log dataset information to MLflow. Reason: %s",
                    e,
                )

        with batch_metrics_logger(run_id, model_id=model_id) as metrics_logger:
            # training model
            model = original(self, *args, **kwargs)

            # aimz does not expose training callbacks, so the per-step ELBO losses are
            # recorded from the fit result once training completes.
            if params["inference_method"] == "SVI":
                losses = np.asarray(cast("SVIRunResult", self.vi_result).losses)
                for step, loss in enumerate(losses):
                    metrics_logger.record_metrics({"elbo_loss": float(loss)}, step)

        # `num_samples` is only known after training completes.
        autologging_client.log_params(
            run_id=run_id,
            params={"num_samples": self._num_samples},
        )
        post_training_logging_operations = autologging_client.flush(synchronous=False)
        if model_id is not None:
            mlflow.MlflowClient().log_model_params(
                model_id,
                {"num_samples": str(self._num_samples)},
            )

        # Whether to automatically log the trained model based on boolean flag.
        if log_models:
            _log_model_with_signature(
                model,
                model_id,
                input_example,
                input_example_exc,
                log_input_examples=log_input_examples,
                log_model_signatures=log_model_signatures,
                registered_model_name=registered_model_name,
            )

        param_logging_operations.await_completion()
        post_training_logging_operations.await_completion()

        return model

    safe_patch(
        FLAVOR_NAME,
        ImpactModel,
        "fit_on_batch",
        patch_fit,
        manage_run=True,
        extra_tags=extra_tags,
    )
    safe_patch(
        FLAVOR_NAME,
        ImpactModel,
        "fit",
        patch_fit,
        manage_run=True,
        extra_tags=extra_tags,
    )


def _run_params(
    model: ImpactModel,
    original: Callable,
    args: tuple,
    kwargs: dict,
) -> dict[str, object]:
    """Collect the parameters to log for a fitting-method call.

    Args:
        model (ImpactModel): The model instance being fitted.
        original (Callable): The original fitting method.
        args (tuple): Positional arguments passed to the fitting method.
        kwargs (dict): Keyword arguments passed to the fitting method.

    Returns:
        The model attributes and explicitly passed fitting-method arguments to log as
        parameters.
    """
    params = {
        "param_input": model.param_input,
        "param_output": model.param_output,
        "inference_method": type(model.inference).__name__,
    }
    if params["inference_method"] == "SVI":
        params["optimizer"] = type(cast("SVI", model.inference).optim).__name__

    unlogged_params = ["X", "y", "num_samples", "progress", "kwargs"]
    params_to_log_for_fn = get_mlflow_run_params_for_fn_args(
        original,
        args,
        {k: v for k, v in kwargs.items() if not _is_arraylike(v)},
        unlogged_params,
    )
    return {**params, **params_to_log_for_fn}


def _get_input_example(
    model: ImpactModel,
    args: tuple,
    kwargs: dict,
) -> dict[str, np.ndarray] | np.ndarray:
    """Copy an input example from the first several rows of the training data.

    The example is copied so that it does not include any mutations from model
    training.

    Args:
        model (ImpactModel): The model instance being fitted.
        args (tuple): Positional arguments passed to the fitting method.
        kwargs (dict): Keyword arguments passed to the fitting method.

    Returns:
        A copy of the first few rows of the training data.
    """
    from aimz.utils.data import ArrayLoader

    X = kwargs["X"] if "X" in kwargs else args[0]
    if isinstance(X, ArrayLoader):
        return {
            k: np.array(v[:INPUT_EXAMPLE_SAMPLE_ROWS])
            for k, v in X.dataset.arrays.items()
            if k not in ("y", model.param_output) and v is not None
        }
    input_example = {
        "X": np.array(np.asarray(X)[:INPUT_EXAMPLE_SAMPLE_ROWS]),
        **{
            k: np.array(np.asarray(v)[:INPUT_EXAMPLE_SAMPLE_ROWS])
            for k, v in kwargs.items()
            if k not in ("y", "rng_key") and _is_arraylike(v)
        },
    }
    if len(input_example) == 1:
        return input_example["X"]
    return input_example


def _log_model_with_signature(
    model: ImpactModel,
    model_id: str | None,
    input_example: dict[str, np.ndarray] | np.ndarray | None,
    input_example_exc: Exception | None,
    *,
    log_input_examples: bool,
    log_model_signatures: bool,
    registered_model_name: str | None,
) -> None:
    """Log the fitted model, resolving its input example and signature.

    Args:
        model (ImpactModel): The fitted model to log.
        model_id: The ID of the logged model to which the artifacts belong.
        input_example: An input example copied from the training data prior to
            training, or ``None`` if collecting it failed.
        input_example_exc: The exception raised while collecting the input example,
            if any.
        log_input_examples: Whether to log the input example along with the model.
        log_model_signatures: Whether to log the model signature along with the model.
        registered_model_name: If given, the model is registered as a new model
            version of the registered model with this name.
    """

    def get_input_example() -> dict[str, np.ndarray] | np.ndarray | None:
        if input_example_exc is not None:
            raise input_example_exc
        return input_example

    def infer_model_signature(input_example: object) -> ModelSignature | None:
        # `ImpactModel.predict` returns an `xarray.DataTree`, which schema inference
        # does not support, so the signature is inferred through the same helper used
        # at save time, with the `progress` parameter recorded in the signature.
        return _infer_signature_from_input_example(
            _Example((input_example, {"progress": False})),
            _AimzModelWrapper(model),
        )

    # Will only resolve `input_example` and `signature` if `log_models` is `True`.
    input_example, signature = resolve_input_example_and_signature(
        get_input_example,
        infer_model_signature,
        log_input_examples,
        log_model_signatures,
        _logger,
    )

    log_model(
        model,
        name="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
        model_id=model_id,
    )


def _log_aimz_dataset(
    aimz_model: ImpactModel,
    args: tuple,
    kwargs: dict,
    source: CodeDatasetSource,
    context: str,
    model_id: str | None,
    name: str | None = None,
) -> None:
    """Log the dataset information to MLflow.

    Args:
        aimz_model (ImpactModel): The model instance being fitted.
        args (tuple): Positional arguments passed to the fitting method.
        kwargs (dict): Keyword arguments passed to the fitting method.
        source (CodeDatasetSource): The dataset source to record.
        context (str): The context tag of the dataset (e.g. ``"train"``).
        model_id: The ID of the logged model to link the dataset to, if any.
        name: The name of the dataset, if any.
    """
    from aimz.utils.data import ArrayLoader

    X = kwargs["X"] if "X" in kwargs else args[0]
    if isinstance(X, ArrayLoader):
        features = {
            k: np.asarray(v)
            for k, v in X.dataset.arrays.items()
            if k not in ("y", aimz_model.param_output) and v is not None
        }
        if len(features) == 1:
            features = next(iter(features.values()))
        label = X.dataset.arrays.get("y")
    else:
        features = {
            "X": np.asarray(X),
            **{
                k: np.asarray(v)
                for k, v in kwargs.items()
                if k not in ("y", "rng_key") and _is_arraylike(v)
            },
        }
        if len(features) == 1:
            features = features["X"]
        label = (
            kwargs.get("y") if "y" in kwargs else (args[1] if len(args) > 1 else None)
        )

    if label is None:
        dataset = from_numpy(features=features, source=source, name=name)
    else:
        dataset = from_numpy(
            features=features,
            targets=np.asarray(label),
            source=source,
            name=name,
        )

    model = LoggedModelInput(model_id=model_id) if model_id else None
    mlflow.log_input(dataset, context, model=model)
