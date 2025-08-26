MCMC
====

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/markean/aimz/blob/main/docs/notebooks/mcmc.ipynb
    :alt: Open In Colab

\

While aimz is primarily designed around variational inference and predictive sampling, it also provides support for MCMC methods via the `NumPyro backend <https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.mcmc.MCMC>`__, using the same aimz interface (e.g., :py:meth:`~aimz.model.ImpactModel.fit_on_batch` and :py:meth:`~aimz.model.ImpactModel.predict_on_batch`).
This enables users to apply MCMC to more complex models where variational inference may be less effective and dataset sizes are relatively small.

.. jupyter-execute::
    :hide-output:

    import jax.numpy as jnp
    import numpyro.distributions as dist
    from jax import random
    from jax.typing import ArrayLike
    from numpyro import plate, sample
    from numpyro.infer import MCMC, NUTS

    from aimz.model import ImpactModel


Model and Data
--------------

We set up a linear regression model and create synthetic data for both features and targets as an example.

.. jupyter-execute::

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
    w = random.normal(rng_key_w, (10,))
    b = random.normal(rng_key_b)
    X = random.normal(rng_key_x, (1000, 10))
    e = random.normal(rng_key_e, (1000,))
    y = jnp.dot(X, w) + b + e


MCMC Sampling and Prediction
----------------------------

MCMC sampling can be performed using the :py:class:`~aimz.model.ImpactModel` class by setting the ``inference`` argument to :external:py:class:`~numpyro.infer.mcmc.MCMC`.
Users can configure the sampler, warm-up steps, and other MCMC-specific parameters.
Calling :py:meth:`~aimz.model.ImpactModel.fit_on_batch()` initiates the sampling process.
Internally, aimz executes the sampler via the :external:py:meth:`~numpyro.infer.mcmc.MCMC.run` method and stores the posterior samples using :external:py:meth:`~numpyro.infer.mcmc.MCMC.get_samples`.

Note that calling :py:meth:`~aimz.model.ImpactModel.fit` with :external:py:class:`~numpyro.infer.mcmc.MCMC` as the inference method will raise a :exc:`TypeError`, as this method is intended for mini-batch training or subsampling.
Regardless of the number of chains (``num_chains``) used, the posterior samples are combined across chains to ensure compatibility with the rest of the aimz interface.
Posterior predictive sampling can be performed using the :py:meth:`~aimz.model.ImpactModel.predict` or :py:meth:`~aimz.model.ImpactModel.predict_on_batch` methods.

.. jupyter-execute::

    rng_key, rng_subkey = random.split(rng_key)
    im = ImpactModel(
        model,
        rng_key=rng_subkey,
        inference=MCMC(NUTS(model), num_warmup=500, num_samples=500),
    )
    im.fit_on_batch(X, y)
    im.inference.print_summary()
    im.predict_on_batch(X)


Using External MCMC Samples
---------------------------

Users can run MCMC sampling directly using NumPyro and then insert the posterior samples into an :py:class:`~aimz.model.ImpactModel` instance using the :py:meth:`~aimz.model.ImpactModel.set_posterior_sample` method for downstream analysis.
For example:

.. jupyter-execute::

    mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=1000)
    rng_key, rng_subkey = random.split(rng_key)
    mcmc.run(rng_key, X, y)

    im.set_posterior_sample(mcmc.get_samples())
    im.predict_on_batch(X)
