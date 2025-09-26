.. _NumPyro: https://num.pyro.ai/

Interventions & Effect Estimation
=================================
This guide covers two closely related functionalities:

* The ``intervention`` argument available in predictive methods like :meth:`~aimz.ImpactModel.predict`, :meth:`~aimz.ImpactModel.predict_on_batch`, and their posterior predictive counterparts.
  Internally, this enables hard (``do``) interventions on specified sample sites using NumPyro_'s :external:class:`~numpyro.handlers.do` effect handler to generate counterfactual draws without rewriting the model.

* The :meth:`~aimz.ImpactModel.estimate_effect` method, which computes the elementwise difference between an intervention (counterfactual) scenario and a baseline (factual) scenario to quantify causal or policy impact.

Typical workflow:

1. Generate one predictive result under factual conditions (optionally also using ``intervention`` if you want to hold certain sites at specific values).
2. Generate another predictive result under a modified ``intervention`` mapping.
3. Pass both results (or the argument dictionaries to generate them lazily) to :meth:`~aimz.ImpactModel.estimate_effect` to obtain the effect output.

Each scenario is a :class:`~xarray.DataTree` produced by the prediction API or materialized on-demand via argument dictionaries.


Interventions
-------------
The ``intervention`` argument is a mapping (``dict[str, ArrayLike]``) from sample site name to a replacement value; during predictive sampling each listed site is fixed, enabling counterfactual or policy analysis.
Values must broadcast to the site’s per‑observation shape (e.g., intervening on a length‑``N`` vector site generally requires shape ``(N,)``).
You can modify multiple sites at once; any not specified follow their posterior (or prior) distribution.

Setting ``in_sample=True`` stores draws under ``posterior_predictive`` while ``in_sample=False`` stores them under ``predictions``—the group must match between baseline and intervention scenarios when computing effects.
Deterministic downstream sites automatically reflect the intervened values.

.. code-block:: python

    # Minimal sketch of a model exposing a stochastic site 'z'
    def model(X, Z, y=None):
        ...
        # site we may choose to override at prediction time
        z = numpyro.sample("z", ...)
        ...

    # Fit (details elided);
    im = ImpactModel(model, ...).fit_on_batch(...)

    # Baseline scenario: set 'z' to its observed/factual value Z
    baseline = im.predict_on_batch(X, intervention={"z": Z})

    # Modified scenario: counterfactual where we overwrite 'z' with zeros
    modified = im.predict_on_batch(
        X,
        intervention={"z": jnp.zeros_like(Z)},
    )


Effect Estimation
-----------------
The :meth:`~aimz.ImpactModel.estimate_effect` method computes an elementwise difference between two predictive scenarios (``intervention - baseline``) and returns a single-group :class:`~xarray.DataTree` that preserves sampling dimensions.

One baseline and one intervention scenario must be provided, either eagerly (``output_baseline`` / ``output_intervention``) or lazily through argument dictionaries (``args_baseline`` / ``args_intervention``).
Mixing is allowed; for example, a precomputed baseline can be supplied with ``output_baseline`` while the intervention is generated lazily with ``args_intervention`` (or the reverse).
Both scenarios must come from the same predictive group (both ``posterior_predictive`` or both ``predictions``) with matching variable sets and shapes.

The result contains that shared group name and each variable is the elementwise difference

.. math:: \text{intervention} - \text{baseline}

retaining leading ``draw`` / ``chain`` dimensions.

Eager (precomputed scenarios)::

    impact = im.estimate_effect(
        output_baseline=baseline,
        output_intervention=modified,
    )

Lazy (defer prediction)::

    impact = im.estimate_effect(
        args_baseline={
            "X": X,
            "intervention": {"z": Z},
            "in_sample": False,
        },
        args_intervention={
            "X": X,
            "intervention": {"z": jnp.zeros_like(Z)},
            "in_sample": False,
        },
    )

Mixed (precomputed baseline, lazy intervention)::

    impact = im.estimate_effect(
        output_baseline=baseline,
        args_intervention={
            "X": X,
            "intervention": {"z": jnp.zeros_like(Z)},
            "in_sample": False,
        },
    )

The returned :class:`~xarray.DataTree` captures the elementwise difference for every variable present in the predictive group.
Any subsequent summary (e.g. mean, intervals) can be computed using Xarray, ArviZ, or standard NumPy / JAX utilities.


Example: Causal Network with Confounder
---------------------------------------
This example illustrates a simple causal network. The variable ``Z`` has a direct causal effect on the outcome ``Y``, while both are influenced by a shared confounder, ``C``.
An additional variable, ``X``, is an observed exogenous factor that influences ``Z`` but has no direct effect on ``Y``.

Our objective is to estimate the causal effect of ``Z`` (or alternatively ``X``) on ``Y``, while properly accounting for the confounding influence of ``C``.
We assume the following generative model for the observed data:

Model
~~~~~

.. jupyter-execute::

    import jax.numpy as jnp
    import numpyro.distributions as dist
    from jax import nn, random
    from jax.typing import ArrayLike
    from numpyro import optim, plate, sample
    from numpyro.infer import SVI, Trace_ELBO, init_to_feasible
    from numpyro.infer.autoguide import AutoNormal

    from aimz import ImpactModel


    def model(X: ArrayLike, C: ArrayLike, y: ArrayLike | None = None) -> None:
        # Observed confounder
        c = sample("c", dist.Exponential(), obs=C)

        # Priors for coefficients in the structural model
        # C -> Z and C -> Y
        beta_cz = sample("beta_cz", dist.Normal())
        beta_cy = sample("beta_cy", dist.Normal())

        # X -> Z and Z -> Y
        beta_xz = sample("beta_xz", dist.Normal())
        beta_zy = sample("beta_zy", dist.Normal())

        # Intercepts
        beta_z = sample("beta_z", dist.Normal())
        beta_y = sample("beta_y", dist.Normal())

        # Observation noise for Z
        sigma = sample("sigma", dist.Exponential())

        # Plate over data
        with plate("data", X.shape[0]):
            mu_z = beta_z + beta_cz * c + beta_xz * X.squeeze(axis=1)
            z = sample("z", dist.LogNormal(mu_z, sigma))

            logits = beta_y + beta_cy * c + beta_zy * z
            sample("y", dist.Bernoulli(logits=logits), obs=y)


Simulating Data under a Known Structural Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We generate synthetic data consistent with the assumed structure:

- `C` is drawn from an exponential distribution.
- `X` is a count variable from a Poisson distribution.
- `Z` is generated as a noisy exponential function of `C` and `X`.
- `Y` is a binary outcome influenced by both `C` and `Z` through a logistic model.

.. jupyter-execute::

    # Create a pseudo-random number generator key for JAX
    rng_key = random.key(42)

    # Sample C from an Exponential distribution
    rng_key, rng_subkey = random.split(rng_key)
    C = random.exponential(rng_subkey, shape=(100,))

    # Sample X from a Poisson distribution
    rng_key, rng_subkey = random.split(rng_key)
    X = random.poisson(rng_subkey, lam=1, shape=(100, 1))

    # Generate Z influenced by C and X
    rng_key, rng_subkey = random.split(rng_key)
    mu_z = -1.0 + 0.5 * C - 1.5 * X.squeeze()
    sigma_z = 10.0  # Add substantial noise to reduce correlation between C and Z
    Z = jnp.exp(random.normal(rng_subkey, shape=(100,)) * sigma_z + mu_z)

    # Generate Y from a logistic regression on C and Z
    rng_key, rng_subkey = random.split(rng_key)
    logits = -2.0 + 5.0 * C + 0.1 * Z
    p = nn.sigmoid(logits)
    y = random.bernoulli(rng_subkey, p=p).astype(jnp.int32)


Fitting the Model and Estimating Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We fit the model using stochastic variational inference.
Once trained, we perform a counterfactual analysis to isolate the effect of `Z` on `Y`.

- `dt_factual` represents predictions under the factual setting (with observed `Z`).
- `dt_counterfactual` represents predictions under a counterfactual intervention where `Z` is set to zero.

.. note::

    Local latent variable requires :meth:`~aimz.ImpactModel.predict_on_batch` here.
    Prefer :meth:`~aimz.ImpactModel.predict` whenever it is compatible with the model.

Comparing these two distributions allows us to estimate the effect of `Z` on `Y`, adjusted for the influence of `C`.

.. jupyter-execute::
    :hide-output:

    im = ImpactModel(
        model,
        rng_key=rng_key,
        inference=SVI(
            model,
            guide=AutoNormal(model, init_loc_fn=init_to_feasible()),
            optim=optim.Adam(step_size=1e-3),
            loss=Trace_ELBO(),
        ),
    )
    im.fit_on_batch(X, y, C=C)

    # Predict under factual (Z) and counterfactual (zeroed Z) scenarios
    dt_factual = im.predict_on_batch(X, C=C, intervention={"z": Z})
    dt_counterfactual = im.predict_on_batch(
        X,
        C=C,
        intervention={"z": jnp.zeros_like(Z)},
    )

    # Estimate effect of intervening on Z while conditioning on C
    impact = im.estimate_effect(
        output_baseline=dt_factual,
        output_intervention=dt_counterfactual,
    )
    impact

.. jupyter-execute::
    :hide-code:

    impact
