MLflow Integration
==================

MLflow_ is an open-source platform for managing the end-to-end machine learning lifecycle (experiment tracking, model packaging, and deployment).
aimz provides first-class MLflow support with built-in automatic logging and model management, making it easy to integrate into production workflows.
This page shows how to use the ``aimz.mlflow`` subpackage, including autologging, customizing logged information, and saving and loading models for reproducible inference.

The integration offers two complementary layers:

1. Autologging via :func:`~aimz.mlflow.autolog` that patches :meth:`~aimz.ImpactModel.fit` and :meth:`~aimz.ImpactModel.fit_on_batch` to record parameters, metrics, datasets, artifacts, and an MLflow Model.
2. Low-level helpers (:func:`~aimz.mlflow.save_model`, :func:`~aimz.mlflow.log_model`, :func:`~aimz.mlflow.load_model`) that mirror the MLflow flavor contract and enable manual control.

.. note::
   MLflow is an optional dependency and is **not installed** with aimz.
   Install it with ``pip install "aimz[mlflow]"``.


Auto Logging
------------
Enable autologging before you instantiate or fit a model (inside an active MLflow run or letting autolog manage the run):

.. code-block:: python

    import mlflow

    import aimz.mlflow
    from aimz import ImpactModel

    # Optional: set your MLflow tracking server URI
    # mlflow.set_tracking_uri("<your-tracking-server-uri>")

    # Optional: set an MLflow experiment
    # mlflow.set_experiment("<your-experiment-name>")

    aimz.mlflow.autolog()  # enable aimz autologging

    X, y = ...  # your training data

    with mlflow.start_run():  # optional: autolog will create a managed run if absent
        im = ImpactModel(...)
        im.fit(X, y)  # parameters, metrics, artifacts, and model logged automatically


When :func:`~aimz.mlflow.autolog` is active and you call :meth:`~aimz.ImpactModel.fit` or :meth:`~aimz.ImpactModel.fit_on_batch`, the following are captured:

Parameters
~~~~~~~~~~
* Selected non-array-like arguments of :meth:`~aimz.ImpactModel.fit` or :meth:`~aimz.ImpactModel.fit_on_batch`, including default values of arguments not passed (array-like inputs are excluded to avoid large parameter payloads)
* ``batch_size`` and ``shuffle`` of the :class:`~aimz.utils.data.ArrayLoader`, when one is passed to :meth:`~aimz.ImpactModel.fit` in place of arrays
* ``param_input`` and ``param_output``
* ``inference_method`` (the class name of the ``inference`` attribute)
* ``optimizer`` (SVI only; the class name of the optimizer stored in the ``optim`` attribute of ``inference``)
* ``num_chains`` and ``num_warmup`` (MCMC only; the settings of ``inference``)
* ``num_samples`` (recorded post-fit; for MCMC, the total across chains)

Parameters are recorded on the run and attached to the logged model entity (if a model artifact is logged).

Metrics
~~~~~~~
* The ELBO loss curve (SVI only) logged step-aware as ``elbo_loss``, one point per optimization step, attached to both the run and the logged model entity (if a model artifact is logged)

Datasets
~~~~~~~~
* The training data passed to :meth:`~aimz.ImpactModel.fit` or :meth:`~aimz.ImpactModel.fit_on_batch` is logged as a run input (disable with ``log_datasets=False``) and linked to the logged model entity (if a model artifact is logged)

Artifacts
~~~~~~~~~
* ``model.py``: the source code of the ``kernel`` attribute used by the model, e.g.,

  .. code-block:: python

    def model(X, y=None):
        ...
* Contents inside the MLflow Model artifact:

  - Pickled model (requires ``cloudpickle``; logged if ``log_models=True``)
  - Conda / requirements / Python environment descriptors (for reproducibility)
  - Optional input example and signature (if ``log_input_examples`` or ``log_model_signatures`` are enabled and ``log_models=True``), where:

    + An input example is copied from the first few rows of the data passed to :meth:`~aimz.ImpactModel.fit` or :meth:`~aimz.ImpactModel.fit_on_batch`, before training starts.
    + If the first positional argument (``X``) is an :class:`~aimz.utils.data.ArrayLoader`, the example is built from its underlying arrays except for the output variable.
    + A signature is inferred by running a short forward pass through :meth:`~aimz.ImpactModel.predict` on the input example, with the ``progress`` parameter recorded in the signature so it can be passed at inference time.


.. note::
   The autologging implementation may evolve. Pin versions in production pipelines for stability.


Custom Logging
--------------
For more control over what is recorded, use :func:`~aimz.mlflow.save_model` or :func:`~aimz.mlflow.log_model` directly instead of autologging.
Here is an example to save and reload a model manually:

.. code-block:: python

    import numpy as np

    from aimz import ImpactModel
    from aimz.mlflow import save_model, load_model

    # Train the model
    im = ImpactModel(...).fit(X, y)

    # Save the model to a local path; the signature is inferred by running the
    # model on the input example. Pass a (data, params) tuple to also record
    # prediction parameters in the signature.
    save_model(
        im,
        path="./model_aimz",
        input_example=(np.asarray(X[:5]), {"progress": False}),
    )

    # Reload the model and make predictions
    loaded_model = load_model("./model_aimz")
    preds = loaded_model.predict(X_new)

The input example must be NumPy arrays.
If the kernel takes additional array inputs, pass a dict of arrays such as ``{"X": X[:5], "z": z[:5]}`` as the input example.
Pass ``signature=False`` to disable signature inference entirely.
Extra files can be bundled into the model directory with the ``extra_files`` argument of :func:`~aimz.mlflow.save_model` and :func:`~aimz.mlflow.log_model`.

.. note::
   aimz models are serialized with ``cloudpickle``.
   Loading honors MLflow's ``MLFLOW_ALLOW_PICKLE_DESERIALIZATION`` environment variable: if it is set to ``false``, loading pickled models is refused.
   Only load models from sources you trust.

Logging directly to an active MLflow run:

.. code-block:: python

    import mlflow

    from aimz import ImpactModel
    from aimz.mlflow import load_model, log_model

    # Example training data (z: additional array input)
    X, y, z = ...

    # Train the model
    im = ImpactModel(...).fit(X, y, z=z)

    with mlflow.start_run():
        # Log custom parameters
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("batch_size", 100)

        # Log custom metrics
        mlflow.log_metric("training_time_sec", 120.5)

        # Log the model
        model_info = log_model(im, name="model")


    # Reload the logged model from the MLflow tracking server for inference
    model_uri = f"models:/{model_info.model_id}"
    loaded_model = load_model(model_uri)

    # Make predictions with the loaded model
    preds = loaded_model.predict(X, z=z)


PyFunc Interface
----------------
Models saved or logged with aimz.mlflow can be loaded as generic MLflow PyFunc models.
You can use :func:`mlflow.pyfunc.load_model` to load them and call ``predict`` in a standard way.

.. code-block:: python

    import mlflow.pyfunc

    # Load the model as a generic PyFunc model
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)

    # Using the PyFunc interface:
    # For multiple array inputs, pass a dict of arrays
    preds = pyfunc_model.predict({"X": X_new, "z": z_new})

    # Or access the underlying ImpactModel directly
    preds = pyfunc_model.get_raw_model().predict(X=X_new, z=z_new)

Dict input works only for models saved with a dict input example or without a signature.
A model saved with an array input example accepts array input only.
Under the hood the pyfunc wrapper delegates to :meth:`~aimz.ImpactModel.predict`.


Environment & Dependencies
--------------------------
When saving a model with aimz.mlflow, both a Conda environment (``conda.yaml``) and a ``python_env.yaml`` are exported, along with pinned requirements.
Helper functions:

* :func:`~aimz.mlflow.get_default_pip_requirements`
* :func:`~aimz.mlflow.get_default_conda_env`

provide the minimal set of packages (optionally including ``cloudpickle``) needed to unpickle the model.
Additional dependencies required for inference may be automatically added by inspecting the model during saving.
