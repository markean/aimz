Uplift Modeling with Custom Likelihood
======================================

This example uses the `Hillstrom email marketing dataset
<https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html>`_
to estimate the causal effect of two email campaigns on customer conversion and
spending.
We build two `NumPyro`_ models, one for conversion and one for spend (a logistic
regression and a hurdle model with a custom likelihood via
:external:func:`~numpyro.primitives.factor`, respectively), fit them through the
:class:`~aimz.ImpactModel` interface, and use :meth:`~aimz.ImpactModel.estimate_effect`
to compute treatment effects.

.. jupyter-execute::
    :hide-output:

    import arviz_base as az
    import arviz_plots as azp
    import arviz_stats as azs
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    import numpyro.distributions as dist
    import numpyro.distributions.distribution
    import pandas as pd
    from jax import Array, random
    from numpyro import deterministic, factor, plate, sample
    from numpyro.infer import MCMC, NUTS

    from aimz import ImpactModel

    # Force JAX to use CPU even if GPU is available
    jax.config.update("jax_platform_name", "cpu")
    # Set the number of CPU devices JAX sees (for CPU-based parallelism)
    jax.config.update("jax_num_cpu_devices", 2)

    # Configure the inline backend for high-resolution figures
    %config InlineBackend.figure_format = "retina"

    # Set the style for ArviZ plots
    azp.style.use("arviz-variat")

    # Set a random seed for reproducibility
    rng_key = random.key(532)


The Hillstrom Dataset
---------------------

The dataset comes from a randomized experiment at an e-commerce company, where 64,000
customers who had made a purchase within the past twelve months were randomly assigned
to one of three groups:

- **Men's E-Mail**: received an email promoting men's merchandise.
- **Women's E-Mail**: received an email promoting women's merchandise.
- **No E-Mail**: control group, received no email.

Two weeks after the email was sent, three outcomes were recorded: whether the customer
visited the website, whether they made a purchase, and how much they spent.

.. jupyter-execute::

    df = pd.read_csv(
        "http://www.minethatdata.com/"
        "Kevin_Hillstrom_MineThatData_E-MailAnalytics"
        "_DataMiningChallenge_2008.03.20.csv",
    )

\

The dataset contains the following 11 columns:

- **Covariates** (pre-treatment customer attributes):

  - ``recency``: months since last purchase (integer, 1--12).
  - ``history``: total dollar value spent in the past year (continuous).
  - ``mens``: 1 if the customer purchased men's merchandise in the past year, 0
    otherwise.
  - ``womens``: 1 if the customer purchased women's merchandise in the past year, 0
    otherwise.
  - ``newbie``: 1 if the customer is new (first purchase within twelve months), 0
    otherwise.
  - ``zip_code``: residential area classification (Urban / Suburban / Rural).
  - ``channel``: purchase channel in the past year (Phone / Web / Multichannel).

- **Treatment**: ``segment``, one of "Mens E-Mail", "Womens E-Mail", or "No E-Mail".

- **Outcomes** (measured during two weeks following the email):

  - ``visit``: 1 if the customer visited the website, 0 otherwise.
  - ``conversion``: 1 if the customer made a purchase, 0 otherwise.
  - ``spend``: dollar amount spent (0 for non-purchasers).


The bar charts below show the raw conversion rates and average spend by treatment arm.

.. jupyter-execute::

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    segments = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]

    # Conversion rate
    conv = df.groupby("segment")["conversion"].mean().reindex(segments)
    conv.plot.bar(ax=axes[0], color=[f"C{i}" for i in range(len(conv))])
    axes[0].set(
        xlabel="Segment",
        ylabel="Conversion Rate",
        title="Conversion Rate by Segment",
    )
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    axes[0].tick_params(axis="x", rotation=0)

    # Average spend
    spend = df.groupby("segment")["spend"].mean().reindex(segments)
    spend.plot.bar(ax=axes[1], color=[f"C{i}" for i in range(len(spend))])
    axes[1].set(
        xlabel="Segment", ylabel="Average Spend",
        title="Average Spend by Segment",
    )
    axes[1].tick_params(axis="x", rotation=0);

The email groups show higher conversion rates and spend than the control group.

.. jupyter-execute::

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # All customers
    for i, seg in enumerate(segments):
        subset = df.loc[df["segment"] == seg, "spend"]
        axes[0].hist(subset, label=seg, color=f"C{i}", density=True)
    axes[0].set(xlabel="Spend", ylabel="Density")
    axes[0].set_title("Spend Distribution (All Customers)")
    axes[0].legend()

    # Buyers only
    for i, seg in enumerate(segments):
        subset = df.loc[(df["segment"] == seg) & (df["spend"] > 0), "spend"]
        axes[1].hist(subset, label=seg, color=f"C{i}", density=True)
    axes[1].set(xlabel="Spend", ylabel="Density")
    axes[1].set_title("Spend Distribution (Buyers Only)")
    axes[1].legend();

\

The left panel confirms that the spend distribution is heavily zero-inflated: the vast
majority of customers do not purchase.
The right panel, restricted to buyers, shows a right-skewed but roughly continuous
distribution across all three segments.

Before modeling, we check that covariates are balanced across treatment arms, as
expected in a randomized experiment.

.. jupyter-execute::

    balance_num = df.groupby("segment")[
        ["recency", "history", "mens", "womens", "newbie"]
    ].mean()

    balance_cat = pd.concat(
        [
            pd.crosstab(df["segment"], df[col], normalize="index").rename(
                columns=lambda v, c=col: f"{c}: {v}",
            )
            for col in ["zip_code", "channel"]
        ],
        axis=1,
    )

    balance_num.join(balance_cat).round(2).T

\

The means (for numeric covariates) and within-row proportions (for categorical
covariates) are nearly identical across segments, confirming that the randomization
worked as intended.

.. note::

   Regarding causal identification, randomization ensures that treatment assignment is
   independent of potential outcomes, giving us **ignorability** (no confounding) and
   **positivity** by design. **Consistency** (a customer's observed outcome equals the
   potential outcome under the assigned treatment) and **no interference** (one
   customer's treatment does not affect another's outcome) are not guaranteed by
   randomization and must be argued on domain grounds.

We encode the treatment segment as an integer, one-hot encode the categorical
covariates, standardize the continuous ones for better sampling, and pack everything
into JAX arrays.

.. jupyter-execute::

    # Map segment names to integer IDs for modeling
    df["segment"] = df["segment"].map({name: i for i, name in enumerate(segments)})

    # Dummy-encode categorical covariates
    df = pd.get_dummies(
        df,
        columns=["zip_code", "channel"],
        drop_first=True,
        dtype=int,
    )

    # Standardize continuous covariates for better sampling
    cols_to_standardize = ["recency", "history"]
    df[cols_to_standardize] = (
        df[cols_to_standardize] - df[cols_to_standardize].mean()
    ) / df[cols_to_standardize].std(ddof=0)

    # Build JAX arrays
    segment = jnp.asarray(df["segment"].to_numpy(), dtype=jnp.int32)
    X = jnp.asarray(
        df[
            ["recency", "history", "mens", "womens", "newbie"]
            + [c for c in df.columns if c.startswith(("zip_code_", "channel_"))]
        ].to_numpy(),
        dtype=jnp.float32,
    )
    y_conversion = jnp.asarray(df["conversion"].to_numpy(), dtype=jnp.int32)
    y_spend = jnp.asarray(df["spend"].to_numpy(), dtype=jnp.float32)


Model 1: Conversion
-------------------

We model conversion as a Bayesian logistic regression.
The linear predictor includes a global intercept, covariate effects, and a
segment-specific shift that captures the treatment effect.
The covariate coefficients are shared across treatment arms (no treatment x covariate
interactions), so the treatment effect is homogeneous on the logit scale.

.. jupyter-execute::

    n_obs, n_features = X.shape


    def conversion_model(X: Array, segment: Array, y: Array | None = None) -> None:
        alpha = sample("alpha", dist.Normal(0.0, 1.0))
        beta = sample("beta", dist.Normal(0.0, 1.0).expand([n_features]))
        tau = sample("tau", dist.Normal(0.0, 1.0).expand([len(segments)]))

        logit = alpha + X @ beta + tau[segment]

        with plate("obs", n_obs):
            sample("y", dist.Bernoulli(logits=logit), obs=y)

\

The parameter ``tau[0]`` corresponds to the control group (No E-Mail), ``tau[1]``
to the Men's E-Mail, and ``tau[2]`` to the Women's E-Mail. The global intercept
``alpha`` and segment shifts ``tau`` are not separately identifiable, but the
treatment-effect contrasts :math:`\tau_1 - \tau_0` and :math:`\tau_2 - \tau_0` are
always identified.
While those contrasts live on the logit scale, our target estimand is the **Average
Treatment Effect (ATE)** on the outcome scale, computed by predicting potential
outcomes under counterfactual treatment assignments and averaging over observations.

We wrap the model in an :class:`~aimz.ImpactModel`, which provides a unified interface
for fitting, prediction, and treatment-effect estimation.
Calling :meth:`~aimz.ImpactModel.fit_on_batch` runs the configured inference engine
(here MCMC with the No-U-Turn Sampler) on the entire dataset in one pass.

.. jupyter-execute::
    :hide-output:

    rng_key, rng_subkey = random.split(rng_key)
    im_conv = ImpactModel(
        conversion_model,
        rng_key=rng_subkey,
        inference=MCMC(
            NUTS(conversion_model),
            num_warmup=500,
            num_samples=500,
            num_chains=2,
        ),
    )

    im_conv.fit_on_batch(X, y_conversion, segment=segment)

\

After fitting, the underlying `NumPyro`_ inference object is accessible via
:attr:`~aimz.ImpactModel.inference`.
We use it here to print MCMC diagnostics.

.. jupyter-execute::

    azs.summary(az.from_numpyro(im_conv.inference))

\

Before estimating treatment effects, we run a posterior predictive check to verify
that the model reproduces the observed conversion rates, overall and per treatment arm.
:meth:`~aimz.ImpactModel.predict_on_batch` generates posterior predictive samples and
returns an :class:`~xarray.DataTree`.

.. jupyter-execute::

    dt_conv = im_conv.predict_on_batch(X, segment=segment)
    dt_conv

\

.. jupyter-execute::

    pp_conv = dt_conv.posterior_predictive["y"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes = axes.flatten()

    # Overall
    obs_overall = float(y_conversion.mean())
    pred_overall = pp_conv.mean(dim="y_dim_0").to_numpy().flatten()
    axes[0].hist(pred_overall, bins=20, color="C0")
    axes[0].axvline(
        obs_overall,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Obs: {obs_overall:.3f}",
    )
    axes[0].set_title("All")
    axes[0].legend()

    # Per treatment arm
    for arm_id, arm_name in enumerate(segments):
        mask = np.asarray(segment == arm_id)
        obs_arm = float(y_conversion[mask].mean())
        pred_arm = pp_conv.isel(y_dim_0=mask).mean(dim="y_dim_0").to_numpy().flatten()
        axes[arm_id + 1].hist(pred_arm, bins=20, color=f"C{arm_id + 1}")
        axes[arm_id + 1].axvline(
            obs_arm,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Obs: {obs_arm:.3f}",
        )
        axes[arm_id + 1].set_title(arm_name)
        axes[arm_id + 1].legend()

    fig.supxlabel("Mean Predicted Rate")
    fig.supylabel("Frequency")
    fig.suptitle(
        "Posterior Predictive Check: Conversion",
        fontsize=18,
        fontweight="bold",
        y=1.05,
    );

\

The observed rate (red dashed line) falls within the bulk of the posterior predictive
distribution for each arm, indicating an adequate fit.


Estimating Treatment Effects on Conversion
------------------------------------------

With the fitted model in hand, we now estimate treatment effects.
:meth:`~aimz.ImpactModel.estimate_effect` takes two scenarios, each a dict of keyword
arguments that would be passed to the underlying prediction method.
For every posterior draw, it generates predictions under both scenarios, subtracts
baseline from intervention, and returns per-observation differences as an
:class:`~xarray.DataTree`.

Because ``segment`` is a function argument rather than a
:func:`~numpyro.primitives.sample` site, each counterfactual scenario is
specified by passing the desired treatment value directly.

.. jupyter-execute::

    effect_conv_mens = im_conv.estimate_effect(
        args_baseline={
            "X": X,
            "segment": jnp.zeros(n_obs, dtype=jnp.int32),
        },
        args_intervention={
            "X": X,
            "segment": jnp.ones(n_obs, dtype=jnp.int32),
        },
        on_batch=True,
    )

    effect_conv_womens = im_conv.estimate_effect(
        args_baseline={
            "X": X,
            "segment": jnp.zeros(n_obs, dtype=jnp.int32),
        },
        args_intervention={
            "X": X,
            "segment": 2 * jnp.ones(n_obs, dtype=jnp.int32),
        },
        on_batch=True,
    )

\

Averaging the per-observation differences over all customers gives the ATE posterior,
one value per draw, fully propagating parameter uncertainty.

.. jupyter-execute::

    ate_conv_mens = effect_conv_mens.posterior_predictive["y"].mean(dim="y_dim_0")
    ate_conv_womens = effect_conv_womens.posterior_predictive["y"].mean(dim="y_dim_0")

\

We plot the ATE posteriors for both campaigns alongside the posterior mean and a
zero-effect reference.

.. jupyter-execute::

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")

    for i, (ax, ate, label) in enumerate(
        zip(
            axes,
            [ate_conv_mens, ate_conv_womens],
            ["Mens E-Mail vs. No E-Mail", "Womens E-Mail vs. No E-Mail"],
            strict=True,
        ),
    ):
        pp_ate = ate.to_numpy().flatten()
        ax.hist(pp_ate, bins=20, color=f"C{i}")
        mean = pp_ate.mean()
        ax.axvline(
            mean,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean:.3f}",
        )
        ax.axvline(0, color="gray", linestyle=":")
        ax.set_title(label)
        ax.legend()

    fig.supxlabel("ATE (Conversion Rate)")
    fig.supylabel("Frequency")
    fig.suptitle(
        "Treatment Effects: Conversion",
        fontsize=18,
        fontweight="bold",
        y=1.1,
    );

\

Both posterior distributions sit entirely above zero, providing strong evidence that
both campaigns increase conversion relative to the control group.
The Men's E-Mail effect is somewhat larger, though the posteriors overlap substantially.


Model 2: Spend
--------------

Spend is zero for most customers and right-skewed among buyers.
We use a **hurdle model** with two components: a Bernoulli gate for whether the customer
purchases at all, and a log-normal distribution for the spend amount conditional on
purchasing.

This model demonstrates how :class:`~aimz.ImpactModel` handles custom likelihoods.
Because the hurdle likelihood does not decompose into a single
:func:`~numpyro.primitives.sample` statement, we compute the log-likelihood manually and
register it with :func:`~numpyro.primitives.factor` during fitting.
At prediction time, we switch to the generative form: sample ``purchase`` from
the Bernoulli, sample ``amount`` from the log-normal distribution, and return their
product as ``y``.

.. jupyter-execute::

    def spend_model(X: Array, segment: Array, y: Array | None = None) -> None:
        # --- Hurdle component: P(spend > 0) ---
        alpha_h = sample("alpha_h", dist.Normal(-5.0, 1.0))
        beta_h = sample("beta_h", dist.Normal(0.0, 1.0).expand([n_features]))
        tau_h = sample("tau_h", dist.Normal(0.0, 1.0).expand([len(segments)]))
        logit = alpha_h + X @ beta_h + tau_h[segment]

        # --- Amount component: log(spend) | spend > 0 ---
        alpha_a = sample("alpha_a", dist.Normal(5.0, 1.0))
        beta_a = sample("beta_a", dist.Normal(0.0, 1.0).expand([n_features]))
        tau_a = sample("tau_a", dist.Normal(0.0, 1.0).expand([len(segments)]))
        sigma = sample("sigma", dist.HalfNormal(1.0))
        mu = alpha_a + X @ beta_a + tau_a[segment]

        if y is not None:
            is_zero = y == 0.0
            with plate("obs", n_obs):
                log_lik = jnp.where(
                    is_zero,
                    jax.nn.log_sigmoid(-logit),
                    jax.nn.log_sigmoid(logit)
                    + dist.LogNormal(mu, sigma).log_prob(jnp.where(is_zero, 1.0, y)),
                )
                factor("y", log_lik)
        else:
            with plate("obs", n_obs):
                purchase = sample("purchase", dist.Bernoulli(logits=logit))
                amount = sample("amount", dist.LogNormal(mu, sigma))
                deterministic("y", purchase * amount)

\

The intercepts are centered on domain-appropriate values for each component: a low
baseline purchase probability for the hurdle, and a plausible log-scale spend level for
the amount.

We fit the hurdle model using the same MCMC configuration as before.

.. jupyter-execute::
    :hide-output:

    rng_key, rng_subkey = random.split(rng_key)
    im_spend = ImpactModel(
        spend_model,
        rng_key=rng_subkey,
        inference=MCMC(
            NUTS(spend_model),
            num_warmup=500,
            num_samples=500,
            num_chains=2,
        ),
    )

    im_spend.fit_on_batch(X, y_spend, segment=segment)

\

MCMC diagnostics:

.. jupyter-execute::

    azs.summary(az.from_numpyro(im_spend.inference))

\

Following the same workflow as the conversion model, we check that the predicted spend
matches the observed values per arm.

.. jupyter-execute::

    dt_spend = im_spend.predict_on_batch(X, segment=segment)
    pp_spend = dt_spend.posterior_predictive["y"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes = axes.flatten()

    # Overall
    obs_overall = float(y_spend.mean())
    pred_overall = pp_spend.mean(dim="y_dim_0").to_numpy().flatten()
    axes[0].hist(pred_overall, bins=20, color="C0")
    axes[0].axvline(
        obs_overall,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Obs: {obs_overall:.3f}",
    )
    axes[0].set_title("All")
    axes[0].legend()

    # Per treatment arm
    for arm_id, arm_name in enumerate(segments):
        mask = np.asarray(segment == arm_id)
        obs_arm = float(y_spend[mask].mean())
        pred_arm = pp_spend.isel(y_dim_0=mask).mean(dim="y_dim_0").to_numpy().flatten()
        axes[arm_id + 1].hist(pred_arm, bins=20, color=f"C{arm_id + 1}")
        axes[arm_id + 1].axvline(
            obs_arm,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Obs: {obs_arm:.3f}",
        )
        axes[arm_id + 1].set_title(arm_name)
        axes[arm_id + 1].legend()

    fig.supxlabel("Mean Predicted Spend")
    fig.supylabel("Frequency")
    fig.suptitle(
        "Posterior Predictive Check: Spend",
        fontsize=18,
        fontweight="bold",
        y=1.05,
    );


Estimating Treatment Effects on Spend
-------------------------------------

We repeat the same counterfactual procedure as for conversion, now using the spend
model.

.. jupyter-execute::

    effect_spend_mens = im_spend.estimate_effect(
        args_baseline={
            "X": X,
            "segment": jnp.zeros(n_obs, dtype=jnp.int32),
        },
        args_intervention={
            "X": X,
            "segment": jnp.ones(n_obs, dtype=jnp.int32),
        },
        on_batch=True,
    )

    effect_spend_womens = im_spend.estimate_effect(
        args_baseline={
            "X": X,
            "segment": jnp.zeros(n_obs, dtype=jnp.int32),
        },
        args_intervention={
            "X": X,
            "segment": 2 * jnp.ones(n_obs, dtype=jnp.int32),
        },
        on_batch=True,
    )

\

Averaging per-observation spend differences gives the ATE posterior in dollars.

.. jupyter-execute::

    ate_spend_mens = effect_spend_mens.posterior_predictive["y"].mean(dim="y_dim_0")
    ate_spend_womens = effect_spend_womens.posterior_predictive["y"].mean(dim="y_dim_0")

\

We visualize the spend ATE posteriors for both campaigns.

.. jupyter-execute::

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")

    for i, (ax, ate, label) in enumerate(
        zip(
            axes,
            [ate_spend_mens, ate_spend_womens],
            ["Mens E-Mail vs. No E-Mail", "Womens E-Mail vs. No E-Mail"],
            strict=True,
        ),
    ):
        pp_ate = ate.to_numpy().flatten()
        ax.hist(pp_ate, bins=20, color=f"C{i}")
        mean = pp_ate.mean()
        ax.axvline(
            mean,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean:.3f}",
        )
        ax.axvline(0, color="gray", linestyle=":")
        ax.set_title(label)
        ax.legend()

    fig.supxlabel("ATE (Spend)")
    fig.supylabel("Frequency")
    fig.suptitle(
        "Treatment Effects: Spend",
        fontsize=18,
        fontweight="bold",
        y=1.1,
    );


Comparison
----------

We compare treatment effects across both outcomes to check whether the campaigns that
drive more conversions also drive more revenue.

.. jupyter-execute::

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Conversion effects
    for i, ate in enumerate([ate_conv_mens, ate_conv_womens]):
        pp_ate = ate.to_numpy().flatten()
        mean = pp_ate.mean()
        hdi = azs.hdi(pp_ate)
        axes[0].errorbar(
            mean, i,
            xerr=[[mean - hdi[0]], [hdi[1] - mean]],
            fmt="o", capsize=5, color="C0", markersize=8,
        )
    axes[0].set_yticks(range(2))
    axes[0].set_yticklabels(["Men's vs. No E-Mail", "Women's vs. No E-Mail"])
    axes[0].axvline(0, color="gray", linestyle=":")
    axes[0].set_xlabel("ATE (Conversion Rate)")
    axes[0].set_title("Conversion")

    # Spend effects
    for i, ate in enumerate([ate_spend_mens, ate_spend_womens]):
        pp_ate = ate.to_numpy().flatten()
        mean = pp_ate.mean()
        hdi = azs.hdi(pp_ate)
        axes[1].errorbar(
            mean, i,
            xerr=[[mean - hdi[0]], [hdi[1] - mean]],
            fmt="o", capsize=5, color="C1", markersize=8,
        )
    axes[1].axvline(0, color="gray", linestyle=":")
    axes[1].set_xlabel("ATE (Spend)")
    axes[1].set_title("Spend")

    fig.suptitle("Treatment Effect Comparison", fontsize=18, fontweight="bold");

\

Both campaigns increase conversion and spend relative to the control, and the Men's
campaign shows a larger effect on both outcomes.


References
----------

- Freedman, D. A. (2008). On regression adjustments to experimental data.
  *Advances in Applied Mathematics*, 40(2), 180–193.
- Hillstrom, K. (2008). The MineThatData E-Mail Analytics and Data Mining Challenge.
  `MineThatData Blog
  <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html>`_.
