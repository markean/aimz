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

"""The ``aimz.mlflow`` module logs and loads aimz models.

Models are exported with two flavors:

aimz (native) format
    The main flavor, loaded back into aimz.
:py:mod:`mlflow.pyfunc`
    For generic pyfunc inference. Predictions are an :py:class:`xarray.DataTree`, or
    with ``return_datatree=False`` in ``params`` a dictionary of arrays, which a
    scoring server serializes.
"""

from __future__ import annotations

import logging
import pickle
from importlib.metadata import version
from inspect import getsource, signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import mlflow
import numpy as np
import yaml
from jax import random
from jax.typing import ArrayLike
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
from mlflow.types import ParamSchema, ParamSpec
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.autologging_utils import (
    INPUT_EXAMPLE_SAMPLE_ROWS,
    MlflowAutologgingQueueingClient,
    autologging_integration,
    batch_metrics_logger,
    get_autologging_config,
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
from numpyro.infer import MCMC, SVI

from aimz.utils._kwargs import _is_per_observation
from aimz.utils._validation import _is_arraylike

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType
    from typing import IO

    import xarray as xr
    from mlflow.models import ModelInputExample, ModelSignature
    from mlflow.models.model import ModelInfo
    from mlflow.tracking.fluent import ActiveRun
    from numpyro.infer.svi import SVIRunResult

    from aimz.model.impact_model import ImpactModel

FLAVOR_NAME = "aimz"

SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"

SUPPORTED_SERIALIZATION_FORMATS = [SERIALIZATION_FORMAT_CLOUDPICKLE]

# A child of MLflow's logger, so autologging messages are shown by default and
# muted by `silent=True` like those of the built-in flavors.
_logger = logging.getLogger("mlflow.aimz")


def get_default_pip_requirements(*, include_cloudpickle: bool = False) -> list[str]:
    """Return the pip requirements that :func:`save_model` and :func:`log_model` record.

    Args:
        include_cloudpickle: Whether to include ``cloudpickle``.

    Returns:
        The pinned requirements of ``aimz``, ``jax``, and ``numpyro``.
    """
    pip_deps = [
        _get_pinned_requirement("aimz"),
        _get_pinned_requirement("jax"),
        _get_pinned_requirement("numpyro"),
    ]
    if include_cloudpickle:
        pip_deps.append(_get_pinned_requirement("cloudpickle"))

    return pip_deps


def get_default_conda_env(*, include_cloudpickle: bool = False) -> dict[str, object]:
    """Return the Conda environment :func:`save_model` and :func:`log_model` record.

    Args:
        include_cloudpickle: Whether to include ``cloudpickle``.

    Returns:
        The environment built on :func:`get_default_pip_requirements`.
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
        mlflow_model: :py:class:`mlflow.models.Model` this flavor is being added to.
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
        im = ImpactModel(...).fit(X, y, batch_size=32, epochs=5)

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
        # Signature inference runs a prediction; the model's key stays unchanged
        rng_key = model.rng_key
        try:
            signature = _infer_signature_from_input_example(
                saved_example,
                _AimzModelWrapper(model),
            )
        finally:
            model._rng_key = rng_key
    elif signature is False:
        signature = None

    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata

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
            # The inference loads the model, so the MLmodel file must exist already
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

    if pip_constraints:
        write_to(path / _CONSTRAINTS_FILE_NAME, "\n".join(pip_constraints))
    write_to(path / _REQUIREMENTS_FILE_NAME, "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(path / _PYTHON_ENV_FILE_NAME)


def _dump_model(pickle_lib: ModuleType, model: ImpactModel, out: IO[bytes]) -> None:
    # The default protocol, for compatibility (mlflow/mlflow#5419)
    pickle_lib.dump(model, out, protocol=pickle.DEFAULT_PROTOCOL)


def _save_model(
    model: ImpactModel,
    output_path: Path,
    serialization_format: str,
) -> None:
    """Pickle the model to ``output_path`` in the given serialization format.

    Raises:
        MlflowException: If the serialization format is unknown.
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
        kwargs: Extra arguments to pass to :py:meth:`mlflow.models.Model.log`.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains
        the metadata of the logged model.

    .. code-block:: python
        :caption: Example

        import mlflow

        import aimz.mlflow
        from aimz import ImpactModel

        # Train the model
        im = ImpactModel(...).fit(X, y, batch_size=32, epochs=5)

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
    """Unpickle a model saved with the ``aimz`` flavor.

    Raises:
        MlflowException: If the serialization format is unknown, or pickle
            deserialization is disabled through ``MLFLOW_ALLOW_PICKLE_DESERIALIZATION``.
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
    """Load the model of an MLflow Model directory, or of its ``model.pkl``."""
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
    """Load the pyfunc wrapper; called by :func:`mlflow.pyfunc.load_model`."""
    return _AimzModelWrapper(_load_model(path))


def load_model(model_uri: str | Path, dst_path: str | None = None) -> ImpactModel:
    """Load an aimz model from a local path, a run, or the model registry.

    Args:
        model_uri: The URI of the MLflow model, such as a local path,
            ``runs:/<run_id>/<path>``, or ``models:/<name>/<version>``.
        dst_path: An existing local directory to download the model artifact to; by
            default a local output path is created.

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
        artifact_uri=str(model_uri),
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

    def predict(
        self,
        data: object,
        params: dict[str, Any] | None = None,
    ) -> xr.DataTree | dict[str, np.ndarray]:
        """Predict with the wrapped model.

        Args:
            data: The input, or a mapping of keyword arguments of
                :meth:`~aimz.ImpactModel.predict`.
            params: Further keyword arguments. ``return_datatree=False`` returns the
                predictive group as a dictionary of arrays, which a scoring server
                serializes, and an integer ``seed`` sets the sampling key.

        Returns:
            The predictions.
        """
        kwargs: dict[str, Any] = {
            "store": "memory",
            "progress": False,
            **(cast("dict[str, Any]", data) if isinstance(data, dict) else {"X": data}),
            **(params or {}),
        }
        return_datatree = kwargs.pop("return_datatree", True)
        if (seed := kwargs.pop("seed", None)) is not None:
            kwargs["rng_key"] = random.key(seed)
        dt = self.aimz_model.predict(**kwargs)
        if return_datatree:
            return dt
        group = (
            "posterior_predictive" if kwargs.get("in_sample", True) else "predictions"
        )

        return {str(site): var.values for site, var in dt[group].data_vars.items()}


def _log_kernel_source(model: ImpactModel) -> None:
    """Log the kernel's source code as an artifact."""
    try:
        mlflow.log_text(getsource(model.kernel), artifact_file="model.py")
    except Exception:
        _logger.exception(
            "Failed to log the kernel source code. aimz autologging will ignore the "
            "failure and continue."
        )


def _log_elbo_losses(model: ImpactModel, run_id: str, model_id: str | None) -> None:
    """Log the ELBO losses of the latest optimization as metrics by step."""
    try:
        with batch_metrics_logger(run_id, model_id=model_id) as metrics_logger:
            losses = np.asarray(cast("SVIRunResult", model.vi_result).losses)
            for step, loss in enumerate(losses):
                metrics_logger.record_metrics({"elbo_loss": float(loss)}, step)
    except Exception:
        _logger.exception(
            "Failed to log the ELBO losses. aimz autologging will ignore the failure "
            "and continue."
        )


def _run_params(
    model: ImpactModel,
    original: Callable,
    args: tuple,
    kwargs: dict,
) -> dict[str, object]:
    """Return the model attributes and fitting arguments to log as parameters."""
    from aimz.utils.data import ArrayLoader

    params = {
        "param_input": model.param_input,
        "param_output": model.param_output,
        "inference_method": type(model.inference).__name__,
    }
    unlogged_params = ["X", "y", "rng_key", "num_samples", "progress", "kwargs"]
    if isinstance(model.inference, SVI):
        params["optimizer"] = type(model.inference.optim).__name__
    elif isinstance(model.inference, MCMC):
        params["num_chains"] = model.inference.num_chains
        params["num_warmup"] = model.inference.num_warmup
        # `fit_on_batch` ignores `num_steps` for MCMC.
        unlogged_params.append("num_steps")

    params_to_log_for_fn = get_mlflow_run_params_for_fn_args(
        original,
        args,
        {k: v for k, v in kwargs.items() if not _is_arraylike(v)},
        unlogged_params,
    )
    X = kwargs["X"] if "X" in kwargs else args[0]
    if isinstance(X, ArrayLoader):
        params_to_log_for_fn |= {"batch_size": X.batch_size, "shuffle": X.shuffle}
    elif not isinstance(X, ArrayLike):
        # Another data loader batches and orders the data itself
        params_to_log_for_fn.pop("batch_size", None)
        params_to_log_for_fn.pop("shuffle", None)
    return {**params, **params_to_log_for_fn}


def _get_input_example(
    model: ImpactModel,
    args: tuple,
    kwargs: dict,
) -> dict[str, np.ndarray] | np.ndarray:
    """Copy the first rows of the training data as an input example.

    Raises:
        TypeError: If the training data is a data loader other than an
            :class:`~aimz.utils.data.ArrayLoader`.
    """
    from aimz.utils.data import ArrayLoader

    X = kwargs["X"] if "X" in kwargs else args[0]
    if isinstance(X, ArrayLoader):
        input_example = {
            k: np.array(v[:INPUT_EXAMPLE_SAMPLE_ROWS])
            for k, v in X.dataset.arrays.items()
            if k != model.param_output
        }
        if len(input_example) == 1:
            return next(iter(input_example.values()))
        return input_example
    if not isinstance(X, ArrayLike):
        msg = "A data loader other than an ArrayLoader has no arrays to copy"
        raise TypeError(msg)
    n_obs = len(np.asarray(X))
    input_example = {
        "X": np.array(np.asarray(X)[:INPUT_EXAMPLE_SAMPLE_ROWS]),
        # Arrays aligned with `X` are sliced with it; the others are call constants
        **{
            k: np.array(
                np.asarray(v)[:INPUT_EXAMPLE_SAMPLE_ROWS]
                if _is_per_observation(v, n_obs)
                else np.asarray(v),
            )
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
    params: dict[str, object],
    *,
    log_input_examples: bool,
    log_model_signatures: bool,
) -> None:
    """Log the fitted model with its input example and signature, when asked for.

    ``params`` are recorded in the signature with their values as defaults.
    """

    def get_input_example() -> dict[str, np.ndarray] | np.ndarray | None:
        if input_example_exc is not None:
            raise input_example_exc
        return input_example

    def infer_model_signature(input_example: object) -> ModelSignature | None:
        # Schema inference does not support a DataTree, so the signature is inferred as
        # at save time, with the model's key kept unchanged
        rng_key = model.rng_key
        try:
            signature = _infer_signature_from_input_example(
                _Example((input_example, params)),
                _AimzModelWrapper(model),
            )
        finally:
            model._rng_key = rng_key
        if signature is not None and signature.params is not None:
            # The seed has no default: a call sets it or the draws stay unseeded
            signature.params = ParamSchema(
                [*signature.params.params, ParamSpec("seed", "long", default=None)],
            )

        return signature

    input_example, signature = resolve_input_example_and_signature(
        get_input_example,
        infer_model_signature,
        log_input_examples,
        log_model_signatures,
        _logger,
    )

    registered_model_name = get_autologging_config(
        FLAVOR_NAME,
        "registered_model_name",
        None,
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
    """Log the training data as a run input, when it holds arrays."""
    from aimz.utils.data import ArrayLoader

    X = kwargs["X"] if "X" in kwargs else args[0]
    if isinstance(X, ArrayLoader):
        features = {
            k: np.asarray(v)
            for k, v in X.dataset.arrays.items()
            if k != aimz_model.param_output
        }
        if len(features) == 1:
            features = next(iter(features.values()))
        label = X.dataset.arrays.get(aimz_model.param_output)
    elif not isinstance(X, ArrayLike):
        # Another data loader holds no arrays to record without consuming it
        return
    else:
        features = {
            "X": np.asarray(X),
            **{
                k: np.asarray(v)
                for k, v in kwargs.items()
                if k not in ("y", "rng_key") and _is_per_observation(v, len(X))
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
    """Enable, configure, or disable autologging from aimz to MLflow.

    Each call to :meth:`~aimz.ImpactModel.fit` or :meth:`~aimz.ImpactModel.fit_on_batch`
    logs its arguments with ``param_input``, ``param_output``, ``inference_method``,
    and the ``optimizer`` of an SVI or the ``num_chains`` and ``num_warmup`` of an
    MCMC; the ELBO loss of each SVI step; the kernel's source code; the training data
    as a run input, when it holds arrays; and the fitted model, with an input example
    and an inferred signature.

    Args:
        log_input_examples: Whether to log an input example with the model.
        log_model_signatures: Whether to log a
            :py:class:`ModelSignature <mlflow.models.ModelSignature>` with the model.
        log_models: Whether to log the fitted model, with its input example and
            signature.
        log_datasets: Whether to log the training data as a run input.
        disable: Whether to disable the integration.
        exclusive: Whether to keep autologged content out of user-created runs.
        disable_for_unsupported_versions: Accepted for parity with MLflow's own
            integrations; MLflow's version gate does not cover aimz.
        silent: Whether to suppress MLflow's event logs and warnings during
            autologging.
        registered_model_name: A registered model to add each fitted model to as a
            new version, created if it does not exist.
        extra_tags: Tags to set on each run that autologging creates.
    """
    from aimz.model.impact_model import ImpactModel

    def patch_fit(
        original: Callable,
        self: ImpactModel,
        *args: object,
        **kwargs: object,
    ) -> ImpactModel:
        """Log the call, the losses, the data, and the model around ``original``."""
        autologging_client = MlflowAutologgingQueueingClient()
        run_id = cast("ActiveRun", mlflow.active_run()).info.run_id
        _log_kernel_source(self)

        params = _run_params(self, original, args, kwargs)
        autologging_client.log_params(run_id=run_id, params=params)

        param_logging_operations = autologging_client.flush(synchronous=False)

        # Copied before training, so the example and the signature see no mutation
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

        # The losses are read from the fit result once it ends, also when the fit
        # raises after the optimization
        vi_result = self.vi_result
        try:
            model = original(self, *args, **kwargs)
        finally:
            if self.vi_result is not vi_result:
                _log_elbo_losses(self, run_id, model_id)

        # `num_samples` is known only after training
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

        if log_models:
            _log_model_with_signature(
                model,
                model_id,
                input_example,
                input_example_exc,
                # Scalar kernel keywords keep their training values as the defaults
                {
                    "progress": False,
                    "return_datatree": True,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in signature(original).parameters
                        and isinstance(v, (bool, int, float, str))
                    },
                },
                log_input_examples=log_input_examples,
                log_model_signatures=log_model_signatures,
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
