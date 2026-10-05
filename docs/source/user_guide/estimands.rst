Effect Estimands
================

:meth:`~aimz.ImpactModel.estimate_effect` returns the difference between two scenarios for every unit and posterior draw.
This guide shows how to summarize these unit-level effects into average effects for all units, for the treated units, or for a subgroup, and into a relative effect.
It also covers which site to summarize, how an intervention interacts with the posterior draws of latent sites, and how to sweep a lever without recompiling.


Example Model
-------------

The examples use simulated data: 3,000 units in three groups, with one covariate, a binary treatment, and a binary outcome.
Units in group 2 are the most likely to be treated and respond the most, while units in group 0 are the least likely to be treated and do not respond at all.

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
    group = rng.integers(0, 3, size=n_obs)
    trt = rng.binomial(1, p=np.array([0.2, 0.5, 0.8])[group])
    logits = (
        np.array([-1.5, -1.0, -0.5])[group]
        + 0.8 * X[:, 0]
        + np.array([0.0, 0.5, 1.0])[group] * trt
    )
    y = rng.binomial(1, p=1 / (1 + np.exp(-logits)))

The kernel records the outcome probability of each unit in the deterministic site ``p``.
After fitting it, we estimate the effect of treating every unit against treating none.

.. jupyter-execute::
    :hide-output:

    def model(X, group, trt, y=None):
        alpha = sample("alpha", dist.Normal(0.0, 2.0).expand([3]))
        beta = sample("beta", dist.Normal(0.0, 2.0))
        tau = sample("tau", dist.Normal(0.0, 2.0).expand([3]))
        with plate("obs", size=X.shape[0]):
            logits = alpha[group] + beta * X[:, 0] + tau[group] * trt
            p = deterministic("p", nn.sigmoid(logits))
            sample("y", dist.Bernoulli(probs=p), obs=y)


    im = ImpactModel(
        model,
        rng_key=random.key(0),
        inference=MCMC(NUTS(model), num_warmup=500, num_samples=1_000),
    )
    im.fit_on_batch(X, y, group=group, trt=trt)

    effect = im.estimate_effect(
        args_baseline={"X": X, "group": group, "trt": np.zeros(n_obs)},
        args_intervention={"X": X, "group": group, "trt": np.ones(n_obs)},
    )


Expected and Predictive Effects
-------------------------------

The effect tree holds a difference for every return site: here ``p``, the outcome probability, and ``y``, the simulated outcome.
A unit's effect on ``y`` is the difference between two simulated outcomes, so it is -1, 0, or 1 in every draw.

.. jupyter-execute::

    np.unique(effect.posterior_predictive["y"])

Averaged over many units, both sites estimate the same average treatment effect (ATE), but the draws from ``y`` also carry the noise of the simulated outcomes.

.. jupyter-execute::

    ate_p = effect.posterior_predictive["p"].mean("obs").load()
    ate_y = effect.posterior_predictive["y"].mean("obs").load()
    print(f"ATE from p: {ate_p.mean().item():.3f} (sd {ate_p.std().item():.3f})")
    print(f"ATE from y: {ate_y.mean().item():.3f} (sd {ate_y.std().item():.3f})")

Summarize a deterministic site such as ``p`` for the expected effect, in particular for unit-level effects and small groups.
Summarize the outcome site when the question is about the outcomes themselves, such as the number of additional positive outcomes if every unit were treated.


Average Effects
---------------

Each estimand averages the unit-level effects over a set of units, separately in every draw, so its draws carry the posterior uncertainty.
The ATE averages over all units, the average treatment effect on the treated (ATT) over the units that were treated, and a conditional average treatment effect (CATE) over a subgroup, here each group.

.. jupyter-execute::

    ite = effect.posterior_predictive["p"].load()
    cate = ite.assign_coords(group=("obs", group)).groupby("group").mean()
    estimands = xr.Dataset(
        {
            "ATE": ite.mean("obs"),
            "ATT": ite.isel(obs=trt == 1).mean("obs"),
            "CATE": cate,
        },
    )
    azs.summary(estimands, kind="stats", ci_prob=0.95, ci_kind="hdi", round_to=3)

The treated units come mostly from group 2, which responds the most, so the ATT exceeds the ATE.


Relative Effects
----------------

A relative effect, such as the percentage change in the outcome rate, is a ratio of the scenario totals, so it needs the level of each scenario rather than their difference.
Predict both scenarios with one key so they stay paired, sum each over the units, and take the ratio in every draw.

.. jupyter-execute::
    :hide-output:

    key = random.key(1)
    baseline = im.predict(X, group=group, trt=np.zeros(n_obs), rng_key=key)
    treated = im.predict(X, group=group, trt=np.ones(n_obs), rng_key=key)

.. jupyter-execute::

    total_baseline = baseline.posterior_predictive["p"].sum("obs")
    total_treated = treated.posterior_predictive["p"].sum("obs")
    relative = (total_treated / total_baseline - 1).load()
    lower, upper = azs.hdi(relative, prob=0.95).values
    print(f"Relative effect: {relative.mean().item():.1%} (95% HDI: {lower:.1%} to {upper:.1%})")

The same two trees give the difference with ``im.estimate_effect(output_baseline=baseline, output_intervention=treated)``.
Averaging the unit-level ratios instead answers a different question: it gives heavy weight to units with a small baseline probability, and with ``y`` it divides by zero wherever a baseline draw is 0.


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
        ates[kernel.__name__] = effect_z.posterior_predictive["mu"].mean("obs").load()

.. jupyter-execute::

    for name, ate_z in ates.items():
        print(f"{name}: {ate_z.mean().item():.2f}")

In the centered kernel the effect is exactly zero, because ``m`` keeps its posterior draws.
In the non-centered kernel the intervention changes ``m``, while each draw keeps its posterior value of ``m_noise``.
Write the kernel so that every path an intervention should change runs through deterministic computations rather than through latent sample sites.


Sweeping a Lever
----------------

A response curve compares each value of a lever with one baseline.
Predict the baseline once, pass it as ``output_baseline``, and give every scenario the same ``rng_key``.
Each value is then paired with the baseline and with the other values, and the baseline is not generated again for each of them.
A Python number passed as a keyword argument is static, so each new value compiles the kernel again; pass the value as a NumPy or JAX array instead, and the whole sweep reuses one compiled program (see :ref:`streaming-keyword-arguments`).
For a kernel that takes a ``dose`` argument:

.. code-block:: python

    key = random.key(2)
    baseline = im.predict(X, dose=np.float32(0.0), rng_key=key)
    curve = {}
    for dose in (0.5, 1.0, 1.5, 2.0):
        effect = im.estimate_effect(
            output_baseline=baseline,
            args_intervention={"X": X, "dose": np.float32(dose), "rng_key": key},
        )
        curve[dose] = effect.posterior_predictive["y"].mean("obs").load()
