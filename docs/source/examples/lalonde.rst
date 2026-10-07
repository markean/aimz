Effect Estimation with Intervention
====================================

This example uses the `Lalonde job training dataset <https://users.nber.org/~rdehejia/nswdata2.html>`_, a classic benchmark for causal inference with observational confounding, to estimate the effect of a job training program on earnings.
The model encodes treatment as a `NumPyro`_ :func:`~numpyro.primitives.sample` site and uses the ``intervention`` keyword to apply `NumPyro`_'s :external:class:`~numpyro.handlers.do` handler, fixing treatment to specific values and generating counterfactual predictions without rewriting the model.

The model includes a treatment x covariate interaction, decomposing the overall **average treatment effect** (ATE) into subgroup-specific **conditional average treatment effects** (CATEs) to ask whether the program helped those without a high-school degree more than those with one.

.. jupyter-execute::

    import logging

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro.distributions as dist
    import pandas as pd
    import xarray as xr
    from jax import Array, random
    from numpyro import deterministic, plate, sample
    from numpyro.infer import MCMC, NUTS

    from aimz import ImpactModel

    import arviz_base as az
    import arviz_plots as azp
    import arviz_stats as azs

    logging.basicConfig(level=logging.INFO, force=True)

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


The Lalonde Dataset
-------------------

The dataset comes from an observational study of a job training program.
A total of 614 individuals were observed, of whom 185 received training and 429 did not.
The outcome is earnings in 1978 (``re78``), measured in thousands of dollars.

.. code-block:: python

    url = (
        "https://raw.githubusercontent.com/robjellis/lalonde/"
        "master/lalonde_data.csv"
    )
    df = pd.read_csv(url)

    # Scale earnings to $k
    df["re75"] = df["re75"] / 1_000
    df["re78"] = df["re78"] / 1_000

.. jupyter-execute::
    :hide-code:
    :hide-output:

    from io import StringIO
    import warnings

    import requests
    from urllib3.exceptions import InsecureRequestWarning

    warnings.filterwarnings("ignore", category=InsecureRequestWarning)

    url = (
        "https://raw.githubusercontent.com/robjellis/lalonde/"
        "master/lalonde_data.csv"
    )
    r = requests.get(url, verify=False)
    df = pd.read_csv(StringIO(r.text))

    # Scale earnings to $k
    df["re75"] = df["re75"] / 1_000
    df["re78"] = df["re78"] / 1_000

\

The dataset contains the following columns:

- **Covariates** (pre-treatment):

  - ``educ``: years of education.
  - ``age``: age in years.
  - ``re75``: earnings in 1975, in thousands of dollars.
  - ``black``, ``hispan``: race/ethnicity indicators.
  - ``married``: marital status indicator.
  - ``nodegree``: 1 if no high-school degree, 0 otherwise.

- **Treatment**: ``treat``, 1 if the individual received job training.

- **Outcome**: ``re78``, earnings in 1978 (dollars, scaled to $k above).

The naive difference in mean earnings between the treated and control groups is negative, as if the program *reduced* earnings:

.. jupyter-execute::

    covariates = ["educ", "age", "re75", "black", "hispan", "married", "nodegree"]
    naive_ate = (
        df.loc[df["treat"] == 1, "re78"].mean() - df.loc[df["treat"] == 0, "re78"].mean()
    )
    print(f"Naive ATE: ${naive_ate:.3f}k")

\

This is due to confounding: the treated group had lower prior earnings, less education, and other systematic differences.

.. jupyter-execute::

    df.groupby("treat")[covariates].mean().round(2).T

\

We standardize the continuous covariates for better sampling and pack the covariates, the treatment, and the outcome into JAX arrays.
The education indicator is kept as a plain array to label the subgroups later.

.. jupyter-execute::

    nodegree = df["nodegree"].to_numpy()

    # Standardize continuous covariates for better sampling
    cols_to_standardize = ["educ", "age", "re75"]
    df[cols_to_standardize] = (
        df[cols_to_standardize] - df[cols_to_standardize].mean()
    ) / df[cols_to_standardize].std(ddof=0)

    # Build JAX arrays
    X = jnp.asarray(df[covariates].to_numpy(), dtype=jnp.float32)
    y_treat = jnp.asarray(df["treat"].to_numpy(), dtype=jnp.int32)
    y_earnings = jnp.asarray(df["re78"].to_numpy(), dtype=jnp.float32)


Model: Heterogeneous Treatment Effects
---------------------------------------

The model includes a treatment x ``nodegree`` interaction, which allows the treatment effect to differ between those without a high-school degree (``nodegree = 1``) and those with one (``nodegree = 0``).
We employ a normal likelihood for simplicity, which keeps the treatment effect directly interpretable in dollars.
The treatment variable ``treat`` is a :func:`~numpyro.primitives.sample` site, observed during fitting (``obs=y_treat``) and intervened on during counterfactual prediction via ``intervention``.

.. jupyter-execute::

    n_obs, n_features = X.shape
    nodegree_idx = covariates.index("nodegree")


    def earnings_model(
        X: Array,
        y: Array | None = None,
        y_treat: Array | None = None,
    ) -> None:
        # Treatment sub-model: makes treat a sample site so that the
        # intervention keyword can fix its value via do().
        p_treat = sample("p_treat", dist.Beta(1.0, 1.0))
        with plate("obs", size=n_obs):
            treat = sample("treat", dist.Bernoulli(probs=p_treat), obs=y_treat)

        # Outcome model with heterogeneous treatment effect.
        # Priors are weakly informative relative to the data scale.
        intercept = sample("intercept", dist.Normal(0.0, 5.0))
        beta_treat = sample("beta_treat", dist.Normal(0.0, 2.0))
        beta_interact = sample("beta_interact", dist.Normal(0.0, 2.0))
        beta_cov = sample(
            "beta_cov",
            dist.Normal(0.0, 1.0).expand([n_features]),
        )
        sigma = sample("sigma", dist.HalfNormal(5.0))

        nodegree = X[:, nodegree_idx]
        mu = intercept + (beta_treat + beta_interact * nodegree) * treat + X @ beta_cov
        with plate("obs", size=n_obs):
            deterministic("mu_earnings", mu)
            sample("y", dist.Normal(mu, sigma), obs=y)

\

The linear predictor expands to ``intercept + beta_treat * treat + beta_interact * treat * nodegree + X @ beta_cov``.
The treatment effect for a degree holder (``nodegree = 0``) is ``beta_treat``, and for a non-degree holder (``nodegree = 1``) it is ``beta_treat + beta_interact``.

The treatment sub-model is intentionally simple: it does not model the treatment assignment mechanism.
Its purpose is to make ``treat`` a sample site so that ``intervention`` can fix treatment values via `NumPyro`_'s :external:class:`~numpyro.handlers.do` handler.
Causal identification relies on the outcome regression: under conditional ignorability and correct specification of the outcome model, the average over individual-level counterfactual predictions recovers the ATE.
This is sometimes called **g-computation**.
If the outcome model is misspecified, the resulting ATE may be biased; in practice, flexible outcome models or doubly robust estimators can reduce this risk.

We fit using MCMC with the No-U-Turn Sampler.

.. jupyter-execute::
    :hide-output:

    rng_key, rng_subkey = random.split(rng_key)
    im = ImpactModel(
        earnings_model,
        rng_key=rng_subkey,
        inference=MCMC(
            NUTS(earnings_model),
            num_warmup=500,
            num_samples=500,
            num_chains=2,
        ),
    )

    im.fit_on_batch(X, y_earnings, y_treat=y_treat)


Estimating Treatment Effects via ``intervention``
-------------------------------------------------

Because ``treat`` is a :func:`~numpyro.primitives.sample` site rather than a regular function argument, we use the ``intervention`` keyword inside the scenario dicts.
This triggers `NumPyro`_'s :external:class:`~numpyro.handlers.do` handler, which severs the incoming edges to the ``treat`` node, generating counterfactual predictions under fixed treatment values.

.. jupyter-execute::

    effect = im.estimate_effect(
        args_baseline={
            "X": X,
            "intervention": {"treat": jnp.zeros(n_obs, dtype=jnp.int32)},
        },
        args_intervention={
            "X": X,
            "intervention": {"treat": jnp.ones(n_obs, dtype=jnp.int32)},
        },
        on_batch=True,
    )

\

The result contains individual-level differences (intervention − baseline) for ``mu_earnings``.
Averaging over all observations in every draw gives the overall ATE.
Because the model includes a treatment x ``nodegree`` interaction, the individual-level effects also vary by education, and averaging within each group gives the CATEs.
A coordinate on ``obs`` labels every observation with its group, so one ``groupby`` yields both.

.. jupyter-execute::

    ite = effect.posterior_predictive["mu_earnings"]
    cate = ite.assign_coords(nodegree=("obs", nodegree)).groupby("nodegree").mean("obs")
    estimands = xr.Dataset({"ATE": ite.mean("obs"), "CATE": cate})
    azs.summary(estimands, kind="stats", ci_prob=0.95, ci_kind="hdi", round_to=2)

\

The posterior mean ATE is positive, indicating that the training program increased earnings on average.
Its interval is wide and includes zero, reflecting the small sample size, high variance of individual earnings, and imbalanced treatment groups.

.. jupyter-execute::

    fig, ax = plt.subplots(figsize=(8, 4))

    labels = ["Overall ATE", "CATE: Degree", "CATE: No Degree"]
    draws = [estimands["ATE"], cate.sel(nodegree=0), cate.sel(nodegree=1)]
    for i, (da, label) in enumerate(zip(draws, labels, strict=True)):
        mean = da.mean().item()
        lower, upper = azs.hdi(da, prob=0.95).values
        ax.errorbar(
            mean,
            i,
            xerr=[[mean - lower], [upper - mean]],
            fmt="o",
            capsize=5,
            color=f"C{i}",
            markersize=8,
        )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="gray", linestyle=":")
    ax.set_xlabel("Treatment Effect ($k)")
    ax.set_title("Treatment Effect Comparison", fontweight="bold");

\

The plot compares the overall ATE with the subgroup CATEs.
This decomposition is a direct consequence of the interaction term; the same :meth:`~aimz.ImpactModel.estimate_effect` call produces both through post-processing.
For this linear model, the subgroup CATEs could also be read directly from the coefficients, but the workflow shown here generalizes to models where the treatment effect has no closed-form expression.

All three intervals include zero, so the data do not provide strong evidence that the program increased earnings for either subgroup.
The degree-holder CATE has a larger point estimate than the no-degree CATE, but its interval is also wider, in part because there are fewer degree holders in the treated group.
The two CATEs overlap substantially, meaning the data do not support a confident claim of treatment effect heterogeneity by education level.

The subgroup CATEs are determined by the interaction structure in the model, not discovered from the data nonparametrically.
A richer model with additional interactions or flexible components could reveal different patterns of heterogeneity.
The raw earnings distributions below hint at the heterogeneity the interaction term is meant to capture: they differ more across treatment status for those without a degree than for those with one.

.. jupyter-execute::

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for treat_val, color, label in [(0, "C0", "Control"), (1, "C1", "Treated")]:
        subset = df.loc[df["treat"] == treat_val, "re78"]
        axes[0].hist(subset, alpha=0.5, color=color, label=label, density=True)
    axes[0].set(xlabel="Earnings 1978 ($k)", ylabel="Density")
    axes[0].legend()

    # Breakdown by nodegree x treatment
    for (nd, treat), grp in df.groupby(["nodegree", "treat"]):
        label = f"{'No Degree' if nd else 'Degree'}, {'Treated' if treat else 'Control'}"
        axes[1].hist(grp["re78"], alpha=0.5, label=label, density=True)
    axes[1].set(xlabel="Earnings 1978 ($k)", ylabel="Density")
    axes[1].legend();


Model Checks
------------

MCMC diagnostics:

.. jupyter-execute::

    summary = azs.summary(az.from_numpyro(im.inference))
    summary.loc[~summary.index.str.startswith("mu_earnings")]

\

The posterior predictive check below compares the observed and predicted mean earnings overall and by treatment arm.
The same coordinate idiom groups the predictions by arm.

.. jupyter-execute::

    dt = im.predict_on_batch(X, y_treat=y_treat)
    pp_earn = dt.posterior_predictive["y"]
    pred_arm = (
        pp_earn.assign_coords(treat=("obs", np.asarray(y_treat))).groupby("treat").mean("obs")
    )
    obs_arm = df.groupby("treat")["re78"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overall
    obs_mean = float(y_earnings.mean())
    axes[0].hist(pp_earn.mean("obs").to_numpy().flatten(), bins=30, color="C0")
    axes[0].axvline(
        obs_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Obs: {obs_mean:.2f}",
    )
    axes[0].set(xlabel="Mean Predicted Earnings ($k)", title="All")
    axes[0].legend()

    # Per treatment arm
    for arm, color in [(0, "C0"), (1, "C1")]:
        axes[1].hist(
            pred_arm.sel(treat=arm).to_numpy().flatten(),
            bins=30,
            color=color,
            alpha=0.5,
            label="Treated" if arm else "Control",
        )
        axes[1].axvline(obs_arm[arm], color=color, linestyle="--", linewidth=2)
    axes[1].set(xlabel="Mean Predicted Earnings ($k)", title="By Treatment")
    axes[1].legend();


References
----------

- Dehejia, R. and Wahba, S. (1999). Causal Effects in Non-Experimental Studies: Reevaluating the Evaluation of Training Programs. *Journal of the American Statistical Association*, 94(448), 1053--1062.
- Dehejia, R. and Wahba, S. (2002). Propensity Score Matching Methods for Non-Experimental Causal Studies. *Review of Economics and Statistics*, 84(1), 151--161.
- Lalonde, R. (1986). Evaluating the Econometric Evaluations of Training Programs. *American Economic Review*, 76(4), 604--620.
