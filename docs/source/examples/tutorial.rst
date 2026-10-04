Tutorial: Estimating a Treatment Effect
=======================================

This tutorial walks through the core aimz workflow on simulated data with a known treatment effect.
You write a model, fit it, estimate the effect of a treatment with :meth:`~aimz.ImpactModel.estimate_effect`, and check that the estimate recovers the truth.

.. jupyter-execute::

    import logging

    import numpy as np
    import numpyro.distributions as dist
    from jax import random
    from numpyro import deterministic, plate, sample
    from numpyro.infer import MCMC, NUTS

    from aimz import ImpactModel

    import arviz_stats as azs

    logging.basicConfig(level=logging.INFO, force=True)


Simulating Data with a Known Effect
-----------------------------------

We simulate 1,000 units with two covariates ``X``, a binary treatment ``trt``, and an outcome ``y``.
The treatment raises the outcome by exactly 2.0, the true average treatment effect (ATE).
Units with a larger first covariate are more likely to be treated, and the same covariate also raises the outcome, so it confounds the effect of the treatment.

.. jupyter-execute::

    rng = np.random.default_rng(123)
    n_obs = 1_000
    true_ate = 2.0

    X = rng.normal(size=(n_obs, 2))
    trt = rng.binomial(1, p=1 / (1 + np.exp(-X[:, 0])))
    y = 1.0 + 1.5 * X[:, 0] - 0.5 * X[:, 1] + true_ate * trt + rng.normal(size=n_obs)

Because of the confounding, the naive difference in mean outcomes between treated and untreated units overstates the effect.

.. jupyter-execute::

    naive = y[trt == 1].mean() - y[trt == 0].mean()
    print(f"Naive difference: {naive:.2f}")


Writing the Model
-----------------

aimz works with a NumPyro_ model, called a kernel, that takes the input ``X`` and the outcome ``y``.
The outcome defaults to ``None``, so the same kernel fits observed outcomes and predicts new ones.
Other data, such as the treatment ``trt``, are passed to the kernel as keyword arguments.
This kernel regresses the outcome on the covariates and the treatment, and records the expected outcome of each unit in the deterministic site ``mu``.

.. jupyter-execute::

    def model(X, trt, y=None):
        intercept = sample("intercept", dist.Normal(0.0, 5.0))
        beta = sample("beta", dist.Normal(0.0, 5.0).expand([X.shape[1]]))
        tau = sample("tau", dist.Normal(0.0, 5.0))
        sigma = sample("sigma", dist.HalfNormal(5.0))
        with plate("obs", size=X.shape[0]):
            mu = deterministic("mu", intercept + X @ beta + tau * trt)
            sample("y", dist.Normal(mu, sigma), obs=y)


Fitting the Model
-----------------

:class:`~aimz.ImpactModel` wraps the kernel with a random key and an inference method, here the No-U-Turn sampler.
:meth:`~aimz.ImpactModel.fit_on_batch` fits the model to the whole data set at once and stores the posterior draws.

.. jupyter-execute::
    :hide-output:

    im = ImpactModel(
        model,
        rng_key=random.key(0),
        inference=MCMC(NUTS(model), num_warmup=500, num_samples=1_000),
    )
    im.fit_on_batch(X, y, trt=trt)


Estimating the Effect
---------------------

:meth:`~aimz.ImpactModel.estimate_effect` predicts the outcome under a baseline and an intervention scenario and returns their difference (intervention minus baseline) for every draw.
Here the baseline sets ``trt`` to 0 for every unit and the intervention sets it to 1.
Both scenarios are drawn with the same random key, so their draws differ only through the treatment.

.. jupyter-execute::
    :hide-output:

    effect = im.estimate_effect(
        args_baseline={"X": X, "trt": 0.0},
        args_intervention={"X": X, "trt": 1.0},
    )

The result is a :class:`~xarray.DataTree`.
Its ``posterior_predictive`` group holds the effect on ``mu`` and ``y`` for every draw and unit, and its ``posterior`` group holds the posterior draws.

.. jupyter-execute::

    effect

Averaging the effect on the expected outcome over the units gives one ATE per posterior draw.
The tree reads its draws lazily, so we load the averages before summarizing them with ArviZ_.

.. jupyter-execute::

    ate = effect.posterior_predictive["mu"].mean(dim="mu_dim_0").load()
    lower, upper = azs.hdi(ate, prob=0.95).values
    print(f"Posterior mean ATE: {ate.mean().item():.2f}")
    print(f"95% HDI: [{lower:.2f}, {upper:.2f}]")

The interval contains the true ATE of 2.0, while the naive difference lies far outside it.
The model recovers the effect because it adjusts for the confounder; with real data, including every confounder is an assumption you make when you write the model.
In this linear model the ATE also equals the coefficient ``tau``, but the same workflow applies to any model, including ones where no single parameter holds the effect.


Where the Draws Are Stored
--------------------------

:meth:`~aimz.ImpactModel.estimate_effect` generated both scenarios with :meth:`~aimz.ImpactModel.predict`, which streams the draws to Zarr_ stores in the model's temporary directory instead of holding them in memory.
The ``artifact_path_baseline`` and ``artifact_path_intervention`` attributes of the effect tree record where they are.
When you no longer need them, :meth:`~aimz.ImpactModel.cleanup` removes the temporary directory.

.. jupyter-execute::

    im.cleanup()


Next Steps
----------

- :doc:`../user_guide/intervention` shows how to fix sample sites with the ``intervention`` argument and how to estimate effects under the prior.
- :doc:`../user_guide/estimands` summarizes unit-level effects into average effects for the treated or a subgroup, and into a relative effect.
- :doc:`../user_guide/streaming_and_on_batch` compares the streaming methods with their ``*_on_batch`` counterparts, including :meth:`~aimz.ImpactModel.fit` for training on minibatches.
- :doc:`../user_guide/sharding` explains how the streaming methods split the work across devices.
- :doc:`The other examples <index>` apply the workflow to real data sets.
