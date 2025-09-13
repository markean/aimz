Model Persistence
=================

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/markean/aimz/blob/main/docs/notebooks/model_persistence.ipynb
    :alt: Open In Colab

\

Model persistence allows you to save a trained model to disk and reload it later for inference or continued training.
This documentation shows how to serialize and deserialize an :py:class:`~aimz.ImpactModel` instance using `cloudpickle <https://pypi.org/project/cloudpickle/>`__, which extends the standard ``pickle`` module to handle a wide range of Python objects, including closures and local functions.
An alternative is `dill <https://pypi.org/project/dill/>`__, which offers similar functionality.

.. note::

   For MLflow users, see the :doc:`mlflow` page for details on saving and loading
   models with MLflow.


Model Training
--------------

.. jupyter-execute::
    :hide-output:

    from pathlib import Path

    import cloudpickle
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from jax import random
    from jax.typing import ArrayLike
    from numpyro import optim, sample
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    from aimz import ImpactModel


    def model(X: ArrayLike, y: ArrayLike | None = None) -> None:
        """Linear regression model."""
        w = sample("w", dist.Normal().expand((X.shape[1],)))
        b = sample("b", dist.Normal())
        mu = jnp.dot(X, w) + b
        sigma = sample("sigma", dist.Exponential())
        sample("y", dist.Normal(mu, sigma), obs=y)


    rng_key = random.key(42)
    rng_key, rng_key_w, rng_key_b, rng_key_x, rng_key_e = random.split(rng_key, 5)
    w = random.normal(rng_key_w, (10,))
    b = random.normal(rng_key_b)
    X = random.normal(rng_key_x, (1000, 10))
    e = random.normal(rng_key_e, (1000,))
    y = jnp.dot(X, w) + b + e

    rng_key, rng_subkey = random.split(rng_key)
    im = ImpactModel(
        model,
        rng_key=rng_subkey,
        inference=SVI(
            model,
            guide=AutoNormal(model),
            optim=optim.Adam(step_size=1e-3),
            loss=Trace_ELBO(),
        ),
    )
    im.fit_on_batch(X, y, progress=False);

Serialization
-------------

Save a trained :py:class:`~aimz.ImpactModel` (and optionally its input data) to disk for later use:

.. jupyter-execute::

    with Path.open("model.pkl", "wb") as f:
        cloudpickle.dump((im, X, y), f)

Deserialization
---------------

Load a previously saved :py:class:`~aimz.ImpactModel` (and optionally its input data) from disk in a fresh new session or different runtime environment.
To use the loaded model correctly, the same dependencies, imports, and any constants or variables that the ``model`` relied on when it was saved must be available.
Any JAX array—whether part of the :py:class:`~aimz.ImpactModel` or the input data—will be placed on the default device.

.. jupyter-execute::
    :hide-output:

    from pathlib import Path

    import cloudpickle
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from numpyro import sample

    with Path.open("model.pkl", "rb") as f:
        im, X, y = cloudpickle.load(f)

Model Usage
-----------

.. jupyter-execute::

    # Resume training from the previous SVI state
    im.fit_on_batch(X, y, progress=False)

    # Predict using the loaded model
    im.predict_on_batch(X)

Resources
---------

* ``dill`` `documentation <https://dill.readthedocs.io/en/latest/>`__
* ``jax Array`` `serialization <https://docs.jax.dev/en/latest/jax.numpy.html#copying-and-serialization>`__
