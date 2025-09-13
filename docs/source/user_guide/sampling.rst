Explicit Sampling
=================

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/markean/aimz/blob/main/docs/notebooks/sampling.ipynb
    :alt: Open In Colab

\

aimz provides three sets of explicit sampling methods from the :py:class:`~aimz.ImpactModel` class, similar to `PyMC samplers <https://www.pymc.io/projects/docs/en/stable/api/samplers.html>`__:

1. **Prior Predictive Sampling**: :py:meth:`~aimz.ImpactModel.sample_prior_predictive_on_batch` and :py:meth:`~aimz.ImpactModel.sample_prior_predictive`.
2. **Posterior Sampling**: :py:meth:`~aimz.ImpactModel.sample`.
3. **Posterior Predictive Sampling**: :py:meth:`~aimz.ImpactModel.sample_posterior_predictive_on_batch` and :py:meth:`~aimz.ImpactModel.sample_posterior_predictive`.

By default, these methods return results as an :external:py:class:`xarray.DataTree`, with the relevant group labeled as ``prior_predictive``, ``posterior``, or ``posterior_predictive``.
For some methods, setting ``return_datatree=False`` instead returns a :py:class:`dict`.

The prior predictive sampling methods perform forward sampling based on the model’s prior specification in the ``kernel`` and are not part of the standard training and inference workflow (:py:meth:`~aimz.ImpactModel.fit`/:py:meth:`~aimz.ImpactModel.predict`), making them particularly useful for conducting prior predictive checks.

Unlike :py:meth:`~aimz.ImpactModel.fit` or :py:meth:`~aimz.ImpactModel.fit_on_batch`, :py:meth:`~aimz.ImpactModel.sample` does not modify the internal ``posterior`` attribute.
It is primarily intended for drawing posterior samples from a fitted model using variational inference.
Users can update the internal posterior manually by passing the samples obtained from :py:meth:`~aimz.ImpactModel.sample` to :py:meth:`~aimz.ImpactModel.set_posterior_sample` without retraining the model.

The posterior predictive sampling methods serve as convenient aliases for :py:meth:`~aimz.ImpactModel.predict_on_batch` and :py:meth:`~aimz.ImpactModel.predict`, respectively.

.. jupyter-execute::
    :hide-output:

    import jax.numpy as jnp
    import numpyro.distributions as dist
    import xarray as xr
    from arviz_plots import plot_ppc_dist, style
    from jax import random
    from jax.typing import ArrayLike
    from numpyro import optim, plate, sample
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    from aimz import ImpactModel

    style.use("arviz-variat")

\

A minimal linear regression model and synthetic data are defined as an example below.

.. jupyter-execute::
    :hide-output:

    def model(X: ArrayLike, y: ArrayLike | None = None) -> None:
        """Linear regression model."""
        w = sample("w", dist.Normal().expand((X.shape[1],)))
        b = sample("b", dist.Normal())
        mu = jnp.dot(X, w) + b
        sigma = sample("sigma", dist.Exponential())
        with plate("data", size=X.shape[0]):
            sample("y", dist.Normal(mu, sigma), obs=y)


    rng_key = random.key(42)
    rng_key, rng_key_w, rng_key_b, rng_key_x, rng_key_e = random.split(rng_key, 5)
    w = random.normal(rng_key_w, (5,))
    b = random.normal(rng_key_b)
    X = random.normal(rng_key_x, (1000, 5))
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


Prior Predictive Sampling
-------------------------

Before training the model, we draw prior predictive samples and visualize the prior predictive distribution:

.. jupyter-execute::

    dt = im.sample_prior_predictive_on_batch(X, num_samples=100)
    plot_ppc_dist(dt, var_names="y", group="prior_predictive")
    dt


Posterior Sampling
------------------

We first train the model using variational inference, drawing only a single posterior sample for demonstration purposes.
After fitting, we call :py:meth:`~aimz.ImpactModel.sample` to generate 100 posterior samples for further analysis.
Setting ``return_datatree=False`` ensures that the results are returned as a dictionary rather than an :external:py:class:`xarray.DataTree`.

.. jupyter-execute::

    im.fit_on_batch(X, y, num_samples=1, progress=False)
    posterior_samples = im.sample(num_samples=100, return_datatree=False)

\

We pass posterior samples to :py:meth:`~aimz.ImpactModel.set_posterior_sample` to update the model’s internal ``posterior``:

.. jupyter-execute::

    im.set_posterior_sample(posterior_samples);


Posterior Predictive Sampling
-----------------------------

We draw posterior predictive samples from the fitted model using :py:meth:`~aimz.ImpactModel.sample_posterior_predictive_on_batch`, though the same results can be obtained with :py:meth:`~aimz.ImpactModel.predict_on_batch` (or :py:meth:`~aimz.ImpactModel.predict`).
The posterior group now contains 100 posterior samples.

.. jupyter-execute::

    dt_posterior_predictive = im.sample_posterior_predictive_on_batch(X)
    dt_posterior_predictive

\

We join the ``posterior_predictive`` group from ``dt_posterior_predictive`` to the ``dt`` containing the ``prior_predictive`` group, and also add the ``observed_data`` as a new group to visualize the posterior predictive distribution.

.. jupyter-execute::

    # Add posterior predictive samples as a new group
    dt["/posterior_predictive"] = dt_posterior_predictive.posterior_predictive

    # Create a dataset for observed data and add as a new group
    ds = xr.Dataset({"y": xr.DataArray(y, dims=["y_dim_0"])})
    dt["/observed_data"] = xr.DataTree(ds)

    # Plot the posterior predictive distribution
    plot_ppc_dist(dt, var_names="y")

    # Display the combined DataTree
    dt
