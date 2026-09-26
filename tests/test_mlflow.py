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

import logging
from functools import partial
from logging.handlers import BufferingHandler
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import yaml
from jax import Array, random
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from aimz.utils.data import ArrayDataset, ArrayLoader
from tests.conftest import _make_svi, lm

if TYPE_CHECKING:
    import xarray as xr

pytest.importorskip("mlflow")

import mlflow.models
import mlflow.pyfunc
from mlflow.entities import LoggedModelStatus
from mlflow.exceptions import MlflowException

from aimz.mlflow import (
    _get_input_example,
    autolog,
    get_default_conda_env,
    load_model,
    log_model,
    save_model,
)


@pytest.fixture(autouse=True)
def _isolate_mlflow_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep MLflow's file output inside the test's temporary directory.

    Saving/loading and autologging create a tracking store (and artifact directory)
    relative to the working directory; chdir into ``tmp_path`` and pin the tracking URI
    there so nothing is written into the working tree. Autologging errors are re-raised
    instead of being logged as warnings, so a failed autolog step fails the test.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_AUTOLOGGING_TESTING", "true")
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
    rng_key = random.key_data(im_lm_svi_fitted.rng_key)
    save_model(
        im_lm_svi_fitted,
        tmp_path / "model",
        input_example=(np.asarray(X[:5]), {"progress": False}),
    )
    # Signature inference runs a prediction but leaves the model's key unchanged
    np.testing.assert_array_equal(random.key_data(im_lm_svi_fitted.rng_key), rng_key)

    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))

    assert isinstance(loaded.get_raw_model(), ImpactModel)
    # Signature inference swallows all exceptions upstream; assert it actually
    # produced a signature, or a broken wrapper wiring would pass silently.
    assert loaded.metadata.signature is not None

    # The tensor signature enforces a NumPy input; `progress` defaults to False from the
    # signature params (MLflow injects it), so no params are needed here.
    out = cast("xr.DataTree", loaded.predict(np.asarray(X)))

    assert out["posterior_predictive"]["y"].sizes["y_dim_0"] == len(X)
    # Predictions default to the in-memory store, which has no artifact on disk
    assert "artifact_path" not in out.attrs


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

    out = cast(
        "xr.DataTree",
        loaded.predict({"X": np.asarray(X), "batch_size": 3, "progress": False}),
    )

    assert out["posterior_predictive"]["y"].sizes["y_dim_0"] == len(X)


def test_save_model_with_conda_env_and_metadata(
    im_lm_svi_fitted: ImpactModel,
    tmp_path: Path,
) -> None:
    """An explicit ``conda_env`` is accepted and ``metadata`` is recorded."""
    conda_env = get_default_conda_env(include_cloudpickle=True)

    save_model(
        im_lm_svi_fitted,
        tmp_path / "model",
        conda_env=conda_env,
        metadata={"key": "value"},
    )

    model = mlflow.models.Model.load(str(tmp_path / "model"))
    assert model.metadata == {"key": "value"}
    assert yaml.safe_load((tmp_path / "model" / "conda.yaml").read_text()) == conda_env


def test_save_model_with_extra_files(
    im_lm_svi_fitted: ImpactModel,
    tmp_path: Path,
) -> None:
    """``extra_files`` are forwarded and recorded in the flavor configuration."""
    extra = tmp_path / "notes.txt"
    extra.write_text("hello")

    save_model(im_lm_svi_fitted, tmp_path / "model", extra_files=[str(extra)])

    model = mlflow.models.Model.load(str(tmp_path / "model"))
    assert "extra_files" in model.flavors["aimz"]


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

    assert "example-package" in (tmp_path / "model" / "requirements.txt").read_text()
    assert (tmp_path / "model" / "constraints.txt").exists()


def test_save_model_signature_false_disables_inference(
    im_lm_svi_fitted: ImpactModel,
    synthetic_data: tuple[Array, Array],
    tmp_path: Path,
) -> None:
    """``signature=False`` disables inference even when an example is provided."""
    X, _ = synthetic_data
    save_model(
        im_lm_svi_fitted,
        tmp_path / "model",
        signature=False,
        input_example=np.asarray(X[:5]),
    )

    model = mlflow.models.Model.load(str(tmp_path / "model"))

    assert model.signature is None


def test_log_model_round_trip_outputs_match(
    im_lm_svi_fitted: ImpactModel,
    synthetic_data: tuple[Array, Array],
) -> None:
    """A logged model reloads predicting identically to the original.

    Across a :func:`~aimz.mlflow.log_model` -> :func:`~aimz.mlflow.load_model` round
    trip, the reloaded model must reproduce predictions draw-for-draw under the same
    explicit PRNG key, without a refit.
    """
    X, _ = synthetic_data
    with mlflow.start_run():
        info = log_model(im_lm_svi_fitted, name="model")

    reloaded = load_model(info.model_uri)

    expected = cast(
        "xr.DataTree",
        im_lm_svi_fitted.predict_on_batch(X, rng_key=random.key(0)),
    )
    actual = cast("xr.DataTree", reloaded.predict_on_batch(X, rng_key=random.key(0)))
    np.testing.assert_array_equal(
        np.asarray(actual["posterior_predictive"]["y"]),
        np.asarray(expected["posterior_predictive"]["y"]),
    )


def test_load_model_disallowed_when_pickle_deserialization_disabled(
    im_lm_svi_fitted: ImpactModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both loaders raise when ``MLFLOW_ALLOW_PICKLE_DESERIALIZATION`` is disabled.

    No other test exercises the gate: every other load runs with the permissive
    default, so a dropped gate would otherwise go unnoticed.
    """
    save_model(im_lm_svi_fitted, tmp_path / "model")

    monkeypatch.setenv("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "false")

    with pytest.raises(MlflowException, match="pickle is disallowed"):
        load_model(str(tmp_path / "model"))
    with pytest.raises(MlflowException, match="pickle is disallowed"):
        mlflow.pyfunc.load_model(str(tmp_path / "model"))


def test_load_model_rejects_unrecognized_serialization_format(
    im_lm_svi_fitted: ImpactModel,
    tmp_path: Path,
) -> None:
    """Loading fails fast when the flavor config declares an unknown format.

    Exercises the only reader of the ``serialization_format`` flavor key: a model
    written by a future aimz version (or a tampered MLmodel file) must raise a
    structured error instead of blindly unpickling ``model.pkl``.
    """
    save_model(im_lm_svi_fitted, tmp_path / "model")
    mlmodel = mlflow.models.Model.load(str(tmp_path / "model"))
    mlmodel.flavors["aimz"]["serialization_format"] = "unsupported_format"
    mlmodel.save(str(tmp_path / "model" / "MLmodel"))

    with pytest.raises(
        MlflowException,
        match="Unrecognized serialization format",
    ) as exc_info:
        load_model(str(tmp_path / "model"))

    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


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
        rng_key = random.key_data(im.rng_key)
        with mlflow.start_run() as run:
            im.fit_on_batch(X=X, y=y, rng_key=random.key(1), num_steps=10)
        logged = mlflow.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            output_format="list",
        )
    finally:
        autolog(disable=True)

    assert len(logged) == 1
    assert logged[0].status == LoggedModelStatus.READY
    # Neither the keyed fit nor signature inference advances the model's key
    np.testing.assert_array_equal(random.key_data(im.rng_key), rng_key)


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_autolog_input_example_snapshot_copies_multi_input(
    synthetic_data: tuple[Array, Array],
    vi: SVI,
) -> None:
    """A multi-array fit call yields a dict example copied before training.

    The label and PRNG key are excluded, and the example rows are copies, so
    in-place changes to the training arrays after the snapshot (i.e. during
    training) cannot leak into the logged example.
    """
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(0), inference=vi)
    z = np.zeros(len(X), dtype=np.float32)

    example = _get_input_example(
        im,
        (),
        {"X": np.asarray(X), "y": np.asarray(y), "z": z, "rng_key": random.key(1)},
    )

    assert isinstance(example, dict)
    assert set(example) == {"X", "z"}
    z[:] = -1.0
    assert example["z"][0] == 0.0


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_autolog_logs_elbo_history_dataset_and_model_params(
    synthetic_data: tuple[Array, Array],
    vi: SVI,
) -> None:
    """The ELBO curve, training dataset, and params are logged once, model-linked."""
    num_steps = 10
    X, y = synthetic_data
    autolog()
    try:
        im = ImpactModel(lm, rng_key=random.key(0), inference=vi)
        with mlflow.start_run() as run:
            im.fit_on_batch(X=X, y=y, num_steps=num_steps)
        client = mlflow.MlflowClient()
        history = client.get_metric_history(run.info.run_id, "elbo_loss")
        run_data = client.get_run(run.info.run_id)
        artifacts = [f.path for f in client.list_artifacts(run.info.run_id)]
        logged = mlflow.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            output_format="list",
        )
        info = mlflow.models.get_model_info(f"models:/{logged[0].model_id}")
    finally:
        autolog(disable=True)

    # One point per SVI step, with no duplicated final-loss entry
    assert sorted(m.step for m in history) == list(range(num_steps))
    # The metrics are also attached to the logged model entity
    assert sum(m.key == "elbo_loss" for m in logged[0].metrics or []) == num_steps
    # The training data is logged as a run input tagged as the train context
    assert run_data.inputs is not None
    dataset_inputs = run_data.inputs.dataset_inputs
    assert len(dataset_inputs) == 1
    assert [t.value for t in dataset_inputs[0].tags] == ["train"]
    # The training parameters are attached to the logged model entity as well
    assert logged[0].params["num_steps"] == str(num_steps)
    # num_samples is attached post-fit via a separate log_model_params call
    assert logged[0].params["num_samples"] == str(im._num_samples)
    # The kernel source is logged, and the model carries an inferred signature
    assert "model.py" in artifacts
    assert info.signature is not None


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_autolog_logs_model_with_loader_input(
    synthetic_data: tuple[Array, Array],
    vi: SVI,
) -> None:
    """Autologging logs the fitted model when fitting on a data loader."""
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
    assert logged[0].status == LoggedModelStatus.READY
    # The loader's own batch size is logged, not fit's ignored argument
    assert logged[0].params["batch_size"] == "3"


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_autolog_manages_run_without_model(
    synthetic_data: tuple[Array, Array],
    vi: SVI,
) -> None:
    """Without an active run, autologging creates, tags, and ends its own run.

    With ``log_models=False``, no model is logged.
    """
    X, y = synthetic_data
    autolog(log_models=False, extra_tags={"key": "value"})
    try:
        im = ImpactModel(lm, rng_key=random.key(0), inference=vi)
        im.fit_on_batch(X=X, y=y, num_steps=10)
        run = mlflow.last_active_run()
        assert run is not None
        logged = mlflow.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            output_format="list",
        )
    finally:
        autolog(disable=True)

    assert run.info.status == "FINISHED"
    assert run.data.tags["key"] == "value"
    assert logged == []


def test_autolog_logs_mcmc_sampler_settings(
    synthetic_data: tuple[Array, Array],
) -> None:
    """MCMC fits log the sampler settings instead of the ignored ``num_steps``."""
    X, y = synthetic_data
    autolog()
    try:
        im = ImpactModel(
            lm,
            rng_key=random.key(0),
            inference=MCMC(NUTS(lm), num_warmup=10, num_samples=5, num_chains=2),
        )
        with mlflow.start_run() as run:
            im.fit_on_batch(X=X, y=y)
        params = mlflow.MlflowClient().get_run(run.info.run_id).data.params
    finally:
        autolog(disable=True)

    assert params["num_chains"] == "2"
    assert "num_steps" not in params


def test_autolog_logs_elbo_when_fit_raises(
    synthetic_data: tuple[Array, Array],
) -> None:
    """A fit raising after optimization keeps its ELBO curve.

    The inference is a subclass of SVI, which is logged as SVI, with its optimizer.
    """

    class _SVI(SVI):
        pass

    num_steps = 10
    X, y = synthetic_data
    autolog()
    try:
        im = ImpactModel(
            lm,
            rng_key=random.key(0),
            inference=_SVI(
                lm,
                guide=AutoNormal(lm),
                optim=Adam(step_size=1e3),
                loss=Trace_ELBO(),
            ),
        )
        # The diverged parameters are rejected by the posterior draw after optimization
        with (
            pytest.warns(RuntimeWarning),
            pytest.raises(ValueError, match="invalid loc parameter"),
            mlflow.start_run() as run,
        ):
            im.fit_on_batch(X=X, y=y, num_steps=num_steps)
        client = mlflow.MlflowClient()
        history = client.get_metric_history(run.info.run_id, "elbo_loss")
        params = client.get_run(run.info.run_id).data.params
    finally:
        autolog(disable=True)

    assert sorted(m.step for m in history) == list(range(num_steps))
    assert params["optimizer"] == "Adam"


def test_autolog_reports_unreadable_kernel_source(
    synthetic_data: tuple[Array, Array],
) -> None:
    """A kernel whose source cannot be read is reported and the model still logged.

    The error goes through MLflow's logger, so it is shown by default like the messages
    of the built-in flavors.
    """
    X, y = synthetic_data
    # The source of a partial cannot be retrieved
    kernel = partial(lm)
    handler = BufferingHandler(capacity=100)
    handler.setLevel(logging.ERROR)
    logging.getLogger("mlflow").addHandler(handler)
    autolog()
    try:
        im = ImpactModel(kernel, rng_key=random.key(0), inference=_make_svi(kernel))
        with mlflow.start_run() as run:
            im.fit_on_batch(X=X, y=y, num_steps=10)
        logged = mlflow.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            output_format="list",
        )
    finally:
        autolog(disable=True)
        logging.getLogger("mlflow").removeHandler(handler)

    assert any(
        "Failed to log the kernel source code" in record.getMessage()
        for record in handler.buffer
    )
    assert len(logged) == 1
    assert logged[0].status == LoggedModelStatus.READY
