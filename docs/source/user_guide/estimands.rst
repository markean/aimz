Effect Estimands
================

:meth:`~aimz.ImpactModel.estimate_effect` returns the difference between two scenarios for every unit and posterior draw.
This guide shows how to summarize these unit-level effects into average effects for all units, for the treated units, or for a subgroup, and into a relative lift.
It also covers which site to summarize, how an intervention interacts with the posterior draws of latent sites, and how to sweep a lever without recompiling.


Example Model
-------------

The examples use simulated conversion data: 3,000 customers in three segments, with one covariate, a binary treatment, and a binary outcome.
Customers in segment 2 are the most likely to be treated and respond the most, while customers in segment 0 are the least likely to be treated and do not respond at all.

.. jupyter-execute::

    import logging

    import numpy as np
    import numpyro.distributions as dist
    import xarray as xr
    from jax import nn, random
    from numpyro import deterministic, plate, sample
    from numpyro.infer import MCMC, NUTS

    from aimz import ImpactModel

    import arviz_stats as azs

    logging.basicConfig(level=logging.INFO, force=True)

    rng = np.random.default_rng(0)
    n_obs = 3_000

    X = rng.normal(size=(n_obs, 1))
    segment = rng.integers(0, 3, size=n_obs)
    trt = rng.binomial(1, p=np.array([0.2, 0.5, 0.8])[segment])
    logits = (
        np.array([-1.5, -1.0, -0.5])[segment]
        + 0.8 * X[:, 0]
        + np.array([0.0, 0.5, 1.0])[segment] * trt
    )
    y = rng.binomial(1, p=1 / (1 + np.exp(-logits)))

The kernel records the conversion probability of each customer in the deterministic site ``p``.
After fitting it, we estimate the effect of treating every customer against treating none.

.. jupyter-execute::
    :hide-output:

    def model(X, segment, trt, y=None):
        alpha = sample("alpha", dist.Normal(0.0, 2.0).expand([3]))
        beta = sample("beta", dist.Normal(0.0, 2.0))
        tau = sample("tau", dist.Normal(0.0, 2.0).expand([3]))
        with plate("obs", size=X.shape[0]):
            logits = alpha[segment] + beta * X[:, 0] + tau[segment] * trt
            p = deterministic("p", nn.sigmoid(logits))
            sample("y", dist.Bernoulli(probs=p), obs=y)


    im = ImpactModel(
        model,
        rng_key=random.key(0),
        inference=MCMC(NUTS(model), num_warmup=500, num_samples=1_000),
    )
    im.fit_on_batch(X, y, segment=segment, trt=trt)

    effect = im.estimate_effect(
        args_baseline={"X": X, "segment": segment, "trt": np.zeros(n_obs)},
        args_intervention={"X": X, "segment": segment, "trt": np.ones(n_obs)},
    )


Expected and Predictive Effects
-------------------------------

The effect tree holds a difference for every return site: here ``p``, the conversion probability, and ``y``, the simulated conversion.
A customer's effect on ``y`` is the difference between two simulated conversions, so it is -1, 0, or 1 in every draw.

.. jupyter-execute::

    np.unique(effect.posterior_predictive["y"])

Averaged over many customers, both sites estimate the same average treatment effect (ATE), but the draws from ``y`` also carry the noise of the simulated outcomes.

.. jupyter-execute::

    ate_p = effect.posterior_predictive["p"].mean("p_dim_0").load()
    ate_y = effect.posterior_predictive["y"].mean("y_dim_0").load()
    print(f"ATE from p: {ate_p.mean().item():.3f} (sd {ate_p.std().item():.3f})")
    print(f"ATE from y: {ate_y.mean().item():.3f} (sd {ate_y.std().item():.3f})")

Summarize a deterministic site such as ``p`` for the expected effect, in particular for unit-level effects and small groups.
Summarize the outcome site when the question is about the outcomes themselves, such as the number of extra conversions a campaign would bring.


Average Effects
---------------

Each estimand averages the unit-level effects over a set of units, separately in every draw, so its draws carry the posterior uncertainty.
The ATE averages over all customers, the average treatment effect on the treated (ATT) over the customers who were treated, and a conditional average treatment effect (CATE) over a subgroup, here each segment.

.. jupyter-execute::

    ite = effect.posterior_predictive["p"].load()
    cate = ite.assign_coords(segment=("p_dim_0", segment)).groupby("segment").mean()
    estimands = xr.Dataset(
        {
            "ATE": ite.mean("p_dim_0"),
            "ATT": ite.isel(p_dim_0=trt == 1).mean("p_dim_0"),
            "CATE": cate,
        },
    )
    azs.summary(estimands, kind="stats", ci_prob=0.95, ci_kind="hdi", round_to=3)

The treated customers come mostly from segment 2, which responds the most, so the ATT exceeds the ATE.


Relative Lift
-------------

A relative effect, such as the lift in conversions, is a ratio of the scenario totals, so it needs the level of each scenario rather than their difference.
Predict both scenarios with one key so they stay paired, sum each over the customers, and take the ratio in every draw.

.. jupyter-execute::
    :hide-output:

    key = random.key(1)
    baseline = im.predict(X, segment=segment, trt=np.zeros(n_obs), rng_key=key)
    treated = im.predict(X, segment=segment, trt=np.ones(n_obs), rng_key=key)

.. jupyter-execute::

    total_baseline = baseline.posterior_predictive["p"].sum("p_dim_0")
    total_treated = treated.posterior_predictive["p"].sum("p_dim_0")
    lift = (total_treated / total_baseline - 1).load()
    lower, upper = azs.hdi(lift, prob=0.95).values
    print(f"Lift: {lift.mean().item():.1%} (95% HDI: {lower:.1%} to {upper:.1%})")

The same two trees give the difference with ``im.estimate_effect(output_baseline=baseline, output_intervention=treated)``.
Averaging the unit-level ratios instead answers a different question: it gives heavy weight to customers with a small baseline probability, and with ``y`` it divides by zero wherever a baseline draw is 0.


Interventions on Latent Sites
-----------------------------

An intervention replaces the value of a sample site for everything downstream of it in the kernel.
The predictive methods also condition on the posterior draws of every latent site, so a latent site downstream of the intervened one keeps its posterior draws, and the intervention does not pass through it.
In the kernel below, ``z`` affects the outcome only through the latent site ``m``.

.. jupyter-execute::

    def centered(X, y=None):
        z = sample("z", dist.Normal(0.0, 1.0))
        m = sample("m", dist.Normal(3.0 * z, 0.1))
        with plate("obs", size=X.shape[0]):
            mu = deterministic("mu", m + X[:, 0])
            sample("y", dist.Normal(mu, 0.1), obs=y)

In the non-centered form of the same kernel, ``m`` is computed from ``z`` and a separate noise site, so it is not a sample site of its own.

.. jupyter-execute::

    def noncentered(X, y=None):
        z = sample("z", dist.Normal(0.0, 1.0))
        m_noise = sample("m_noise", dist.Normal(0.0, 0.1))
        with plate("obs", size=X.shape[0]):
            mu = deterministic("mu", 3.0 * z + m_noise + X[:, 0])
            sample("y", dist.Normal(mu, 0.1), obs=y)

Both kernels fit the same data, and each estimates the effect of setting ``z`` to 0.

.. jupyter-execute::
    :hide-output:

    X_small = rng.normal(size=(200, 1))
    y_small = 3.0 + X_small[:, 0] + rng.normal(scale=0.1, size=200)

    ates = {}
    for kernel in (centered, noncentered):
        im_z = ImpactModel(
            kernel,
            rng_key=random.key(0),
            inference=MCMC(NUTS(kernel), num_warmup=500, num_samples=1_000),
        )
        im_z.fit_on_batch(X_small, y_small)
        effect_z = im_z.estimate_effect(
            args_baseline={"X": X_small},
            args_intervention={"X": X_small, "intervention": {"z": 0.0}},
        )
        ates[kernel.__name__] = effect_z.posterior_predictive["mu"].mean("mu_dim_0").load()

.. jupyter-execute::

    for name, ate_z in ates.items():
        print(f"{name}: {ate_z.mean().item():.2f}")

In the centered kernel the effect is exactly zero, because ``m`` keeps its posterior draws.
In the non-centered kernel the intervention changes ``m``, while each draw keeps its posterior value of ``m_noise``.
Write the kernel so that every path an intervention should change runs through deterministic computations rather than through latent sample sites.


Sweeping a Lever
----------------

A response curve or a budget sweep calls :meth:`~aimz.ImpactModel.estimate_effect` once for each value of a lever.
A Python number passed as a keyword argument is static, so each new value compiles the kernel again; pass the value as a NumPy or JAX array instead, and the whole sweep reuses one compiled program (see :ref:`streaming-keyword-arguments`).
For a kernel that takes a ``price`` argument:

.. code-block:: python

    curve = {}
    for price in (8.0, 9.0, 10.0, 11.0, 12.0):
        effect = im.estimate_effect(
            args_baseline={"X": X, "price": np.float32(10.0)},
            args_intervention={"X": X, "price": np.float32(price)},
        )
        curve[price] = effect.posterior_predictive["y"].mean("y_dim_0").load()
