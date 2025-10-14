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

"""MLflow autologging integration for aimz."""

import logging
import tempfile
from collections.abc import Callable
from inspect import getsource
from typing import TYPE_CHECKING, cast

import numpy as np
from mlflow import log_metric, log_param, log_params, log_text
from mlflow.models import infer_signature
from mlflow.tracking.fluent import _initialize_logged_model
from mlflow.utils.autologging_utils import (
    INPUT_EXAMPLE_SAMPLE_ROWS,
    autologging_integration,
    get_autologging_config,
    log_fn_args_as_params,
    resolve_input_example_and_signature,
    safe_patch,
)
from sklearn.utils.validation import _is_arraylike

if TYPE_CHECKING:
    from numpyro.infer.svi import SVIRunResult

FLAVOR_NAME = "aimz"

_logger = logging.getLogger(__name__)


@autologging_integration(FLAVOR_NAME)
def autolog(
    *,
    log_input_examples: bool = False,
    log_model_signatures: bool = True,
    log_models: bool = True,
    disable: bool = False,
    silent: bool = False,
    registered_model_name: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> None:
    """Enable and configure autologging for aimz with MLflow.

    Autologging is performed when you call:
        - :meth:`~aimz.ImpactModel.fit_on_batch`.
        - :meth:`~aimz.ImpactModel.fit`.

    Logs the following:
        - Selected arguments to the methods, together with ``param_input``,
          ``param_output``, ``inference_method``, and ``optimizer`` as parameters.
        - The final evidence lower bound (ELBO) loss as a metric.
        - The source code of the kernel function used in the model as a text artifact.
        - An MLflow Model containing the fitted estimator as an artifact.

    Args:
        log_input_examples: If ``True``, input examples from training datasets are
            collected and logged along with model artifacts during training. If
            ``False``, input examples are not logged. Note: Input examples are MLflow
            model attributes and are only collected if ``log_models`` is also ``True``.
        log_model_signatures: If ``True``,
            :external:class:`ModelSignatures <mlflow.models.ModelSignature>` describing
            model inputs and outputs are collected and logged along with model artifacts
            during training. If ``False``, signatures are not logged. Note: Model
            signatures are MLflow model attributes and are only collected if
            ``log_models`` is also ``True``.
        log_models: If ``True``, trained models are logged as MLflow model artifacts. If
            ``False``, trained models are not logged. Input examples and model
            signatures, which are attributes of MLflow models, are also omitted when
            ``log_models`` is ``False``.
        disable: If ``True``, disables all supported autologging integrations. If
            ``False``, enables all supported autologging integrations.
        silent: If ``True``, suppress all event logs and warnings from MLflow during
            autologging setup and training execution. If ``False``, show all events and
            warnings during autologging setup and training execution.
        registered_model_name: If given, each time a model is trained, it is registered
            as a new model version of the registered model with this name. The
            registered model is created if it does not already exist.
        extra_tags: A dictionary of extra tags to set on each managed run created by
            autologging.
    """
    from aimz.mlflow.log import log_model
    from aimz.model.impact_model import ImpactModel
    from aimz.utils.data import ArrayLoader

    def patch_fit(
        original: Callable,
        self: ImpactModel,
        *args: object,
        **kwargs: object,
    ) -> ImpactModel:
        """Patch for the fitting method to log information.

        Args:
            original (Callable): The original method.
            self (ImpactModel): The instance being initialized.
            *args (object): Positional arguments for the method.
            **kwargs (object): Keyword arguments for the method.
        """
        model_source = getsource(self.kernel)
        log_text(model_source, artifact_file="model.py")

        unlogged = ["X", "y", "num_samples", "progress", "kwargs"]
        params = {
            "param_input": self.param_input,
            "param_output": self.param_output,
            "inference_method": type(self.inference).__name__,
        }
        if params["inference_method"] == "SVI":
            params["optimizer"] = type(self.inference.optim).__name__
        log_params(params)
        log_fn_args_as_params(
            original,
            args=args,
            kwargs={k: v for k, v in kwargs.items() if not _is_arraylike(v)},
            unlogged=unlogged,
        )

        model = original(self, *args, **kwargs)

        log_param("num_samples", self._num_samples)
        if params["inference_method"] == "SVI":
            losses = cast("SVIRunResult", self.vi_result).losses
            log_metric("elbo_loss", value=losses[-1])

        if log_models:
            model_id = _initialize_logged_model("model", flavor=FLAVOR_NAME).model_id
            # Will only resolve `input_example` and `signature` if `log_models` is
            # `True`.
            with tempfile.TemporaryDirectory() as temp_dir:
                X = kwargs.get("X") if "X" in kwargs else args[0]
                if isinstance(X, ArrayLoader):
                    input_example = {
                        k: np.asarray(v[:INPUT_EXAMPLE_SAMPLE_ROWS])
                        for k, v in X.dataset.arrays.items()
                        if k != self.param_output
                    }
                else:
                    input_example = {
                        "X": np.asarray(X)[:INPUT_EXAMPLE_SAMPLE_ROWS],
                        **{
                            k: np.asarray(v)[:INPUT_EXAMPLE_SAMPLE_ROWS]
                            for k, v in kwargs.items()
                            if k != self.param_output and _is_arraylike(v)
                        },
                    }
                    if len(input_example) == 1:
                        input_example = input_example["X"]
                input_example, signature = resolve_input_example_and_signature(
                    get_input_example=lambda: input_example,
                    infer_model_signature=lambda x: infer_signature(
                        x,
                        params={
                            "batch_size": INPUT_EXAMPLE_SAMPLE_ROWS,
                            "output_dir": temp_dir,
                            "progress": False,
                        },
                    ),
                    log_input_example=log_input_examples,
                    log_model_signature=log_model_signatures,
                    logger=_logger,
                )
            registered_model_name = get_autologging_config(
                FLAVOR_NAME,
                config_key="registered_model_name",
                default_value=None,
            )
            log_model(
                model,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
                model_id=model_id,
            )
            if params["inference_method"] == "SVI":
                log_metric("elbo_loss", value=losses[-1], model_id=model_id)

        return model

    safe_patch(
        FLAVOR_NAME,
        destination=ImpactModel,
        function_name="fit_on_batch",
        patch_function=patch_fit,
        manage_run=True,
        extra_tags=extra_tags,
    )

    safe_patch(
        FLAVOR_NAME,
        destination=ImpactModel,
        function_name="fit",
        patch_function=patch_fit,
        manage_run=True,
        extra_tags=extra_tags,
    )
