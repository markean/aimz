Model Comparison
================

The tree returned by :meth:`~aimz.ImpactModel.log_likelihood` has the structure `ArviZ`_ expects for leave-one-out cross-validation: a ``log_likelihood`` group holding the pointwise log-likelihood with ``chain`` and ``draw`` dimensions, next to the ``posterior`` group.
This page computes PSIS-LOO with `arviz-stats <https://arviz-stats.readthedocs.io/>`__ and ranks candidate models, with no conversion step.

.. note::

   ArviZ functions operate on loaded arrays, so call :external:meth:`~xarray.DataTree.load` on the tree first.


Candidate Models
----------------
Two regression kernels are fitted to data generated with a quadratic term, so the second kernel is the better model.

.. jupyter-execute::
    :hide-output:

    import logging

    import arviz_stats as azs
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from jax import random
    from jax.typing import ArrayLike
    from numpyro import optim, sample
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    from aimz import ImpactModel

    logging.basicConfig(level=logging.INFO, force=True)


    def linear(X: ArrayLike, y: ArrayLike | None = None) -> None:
        """Linear regression model."""
        w = sample("w", dist.Normal().expand((X.shape[1],)).to_event(1))
        sigma = sample("sigma", dist.Exponential())
        sample("y", dist.Normal(jnp.dot(X, w), sigma), obs=y)


    def quadratic(X: ArrayLike, y: ArrayLike | None = None) -> None:
        """Linear regression model with a quadratic term in the first feature."""
        w = sample("w", dist.Normal().expand((X.shape[1],)).to_event(1))
        v = sample("v", dist.Normal())
        sigma = sample("sigma", dist.Exponential())
        sample("y", dist.Normal(jnp.dot(X, w) + v * X[:, 0] ** 2, sigma), obs=y)


    rng_key = random.key(42)
    rng_key, rng_key_x, rng_key_e = random.split(rng_key, 3)
    X = random.normal(rng_key_x, (300, 2))
    y = (
        X @ jnp.array([1.0, -0.5])
        + 0.8 * X[:, 0] ** 2
        + 0.5 * random.normal(rng_key_e, (300,))
    )

    models = {}
    for name, kernel in (("linear", linear), ("quadratic", quadratic)):
        im = ImpactModel(
            kernel,
            rng_key=random.key(0),
            inference=SVI(
                kernel,
                guide=AutoNormal(kernel),
                optim=optim.Adam(step_size=1e-2),
                loss=Trace_ELBO(),
            ),
        )
        models[name] = im.fit_on_batch(
            X,
            y,
            num_steps=3000,
            num_samples=500,
            progress=False,
        )


Pointwise Log-Likelihood
------------------------
:meth:`~aimz.ImpactModel.log_likelihood` evaluates every posterior draw on every observation.
The memory store keeps the result in host memory, and :external:meth:`~xarray.DataTree.load` materializes it for ArviZ.

.. jupyter-execute::

    trees = {
        name: im.log_likelihood(X, y, store="memory", progress=False).load()
        for name, im in models.items()
    }
    trees["quadratic"]


Leave-One-Out Cross-Validation
------------------------------
``azs.loo`` estimates the expected log pointwise predictive density (ELPD) with Pareto-smoothed importance sampling.

.. jupyter-execute::

    azs.loo(trees["quadratic"])


Ranking Models
--------------
``azs.compare`` ranks the candidates by ELPD and reports the differences with their standard errors.

.. jupyter-execute::

    azs.compare(trees)

.. note::

   :meth:`~aimz.ImpactModel.log_likelihood` requires a posterior covering every latent site of the kernel.
   The Pareto ``k`` diagnostics describe the fitted posterior as used, so for a variational posterior they reflect the approximation as well as the model.
