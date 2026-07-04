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

"""Tests for the MLflow integration."""

from pathlib import Path

import numpy as np
import pytest
from jax import Array, random
from numpyro.infer import SVI

from aimz import ImpactModel
from aimz.utils.data import ArrayDataset, ArrayLoader
from tests.conftest import lm

pytest.importorskip("mlflow")

import mlflow.models
import mlflow.pyfunc

from aimz.mlflow import (
    autolog,
    get_default_conda_env,
    load_model,
    save_model,
)


@pytest.fixture(autouse=True)
def _isolate_mlflow_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep MLflow's file output inside the test's temporary directory.

    Saving/loading and autologging create a tracking store (and artifact directory)
    relative to the working directory; chdir into ``tmp_path`` and pin the tracking URI
    there so nothing is written into the working tree.
    """
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")


# A served model auto-batches (no fixed batch size in its signature), emitting a
# device-divisibility performance hint irrelevant to this round-trip.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pyfunc_round_trip_predicts(
    im_lm_svi_fitted: ImpactModel,
    synthetic_data: tuple[Array, Array],
    tmp_path: Path,
) -> None:
    """Save an aimz model, reload it through the pyfunc flavor, and predict.

    Exercises the flavor round-trip: save with signature inference from the
    ``input_example``, :func:`mlflow.pyfunc.load_model`, and prediction through the
    wrapper's :meth:`~aimz.ImpactModel.predict` delegation.
    """
    X, _ = synthetic_data
    save_model(im_lm_svi_fitted, tmp_path / "model", input_example=np.asarray(X[:5]))

    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))

    assert isinstance(loaded.get_raw_model(), ImpactModel)

    # The tensor signature enforces a NumPy input; `progress` defaults to False from the
    # signature (MLflow injects it), so no params are needed here.
    out = loaded.predict(np.asarray(X))

    assert out["posterior_predictive"]["y"].sizes["y_dim_0"] == len(X)


def test_pyfunc_predict_with_dict_input(
    im_lm_svi_fitted: ImpactModel,
    synthetic_data: tuple[Array, Array],
    tmp_path: Path,
) -> None:
    """Saved without an ``input_example`` (no signature), a dict input is unpacked.

    The wrapper forwards a mapping as ``predict(**model_input)``, so prediction keyword
    arguments pass through the pyfunc boundary.
    """
    X, _ = synthetic_data
    save_model(im_lm_svi_fitted, tmp_path / "model")

    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))

    out = loaded.predict({"X": np.asarray(X), "batch_size": 3, "progress": False})

    assert out["posterior_predictive"]["y"].sizes["y_dim_0"] == len(X)


def test_load_model_returns_raw_impact_model(
    im_lm_svi_fitted: ImpactModel,
    tmp_path: Path,
) -> None:
    """:func:`~aimz.mlflow.load_model` reloads the underlying :class:`ImpactModel`."""
    save_model(im_lm_svi_fitted, tmp_path / "model")

    reloaded = load_model(str(tmp_path / "model"))

    assert isinstance(reloaded, ImpactModel)


def test_save_model_with_conda_env_and_metadata(
    im_lm_svi_fitted: ImpactModel,
    tmp_path: Path,
) -> None:
    """An explicit ``conda_env`` is honored and ``metadata`` is recorded."""
    conda_env = get_default_conda_env(include_cloudpickle=True)

    save_model(
        im_lm_svi_fitted,
        tmp_path / "model",
        conda_env=conda_env,
        metadata={"key": "value"},
    )

    model = mlflow.models.Model.load(str(tmp_path / "model"))
    assert model.metadata == {"key": "value"}
    assert (tmp_path / "model" / "conda.yaml").exists()


def test_save_model_with_pip_requirements_and_constraints(
    im_lm_svi_fitted: ImpactModel,
    tmp_path: Path,
) -> None:
    """Explicit ``pip_requirements`` (with a constraint) write both files."""
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("example-package==1.0.0\n")

    save_model(
        im_lm_svi_fitted,
        tmp_path / "model",
        pip_requirements=[f"-c {constraints}", "example-package"],
    )

    assert (tmp_path / "model" / "requirements.txt").exists()
    assert (tmp_path / "model" / "constraints.txt").exists()


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_autolog_logs_model_when_rng_key_passed(
    synthetic_data: tuple[Array, Array],
    vi: SVI,
) -> None:
    """Autologging still logs the fitted model when ``rng_key`` is passed."""
    X, y = synthetic_data
    autolog()
    try:
        im = ImpactModel(lm, rng_key=random.key(0), inference=vi)
        with mlflow.start_run() as run:
            im.fit_on_batch(X=X, y=y, rng_key=random.key(1), num_steps=10)
        logged = mlflow.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            output_format="list",
        )
    finally:
        autolog(disable=True)

    assert len(logged) == 1


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_autolog_logs_model_with_loader_input(
    synthetic_data: tuple[Array, Array],
    vi: SVI,
) -> None:
    """Autologging builds the input example (excluding the label) from a data loader."""
    X, y = synthetic_data
    autolog()
    try:
        im = ImpactModel(lm, rng_key=random.key(0), inference=vi)
        loader = ArrayLoader(
            ArrayDataset(X=X, y=y),
            rng_key=random.key(1),
            batch_size=3,
        )
        with mlflow.start_run() as run:
            im.fit(loader, num_samples=10, progress=False)
        logged = mlflow.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            output_format="list",
        )
    finally:
        autolog(disable=True)

    assert len(logged) == 1
