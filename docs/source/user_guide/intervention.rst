Interventions & Effect Estimation
=================================
Interventions let you explore how a model's predictions change when specified variables are set to chosen values.
You can use them with prior predictive sampling to explore the implications of your assumptions, or with posterior predictive sampling to incorporate what the model has learned from data.
For prior or posterior predictive scenarios, :meth:`~aimz.ImpactModel.estimate_effect` computes the difference between a baseline and an intervention scenario while preserving the individual draws.


Interventions
-------------
The ``intervention`` argument maps sample site names to replacement values, for example ``{"z": 0.0}``.
The predictive methods apply this mapping through NumPyro_'s :external:class:`~numpyro.handlers.do` effect handler, so downstream computations use the specified values without requiring changes to the kernel.
This includes deterministic downstream sites.
You can intervene on multiple sites in the same call.

Values must broadcast to the shape expected by downstream computations.
A scalar sets a site to the same value for every observation, and an array whose leading axis matches the number of observations sets a different value for each observation.
For example, a length-``N`` site can take a replacement array of shape ``(N,)``.
The streaming methods batch and shard such arrays together with an array ``X``, so per-observation interventions also work on inputs processed in several batches.
Any other value is applied whole to every batch, as is every value when ``X`` is a data loader.
The rule goes by shape alone, so a global site whose length happens to equal the number of observations is also treated as per-observation; pass ``shard_axis="draw"`` or use an ``_on_batch`` method to apply it whole.

The choice of predictive method determines how the remaining sites are sampled and where results appear in the returned :class:`~xarray.DataTree`:

* **Prior predictive:** :meth:`~aimz.ImpactModel.sample_prior_predictive` and :meth:`~aimz.ImpactModel.sample_prior_predictive_on_batch` sample from the model's priors without requiring a fitted model.
  Results are stored in ``prior_predictive``.
* **Posterior predictive:** :meth:`~aimz.ImpactModel.predict` and :meth:`~aimz.ImpactModel.predict_on_batch` use the stored posterior samples, as do their :meth:`~aimz.ImpactModel.sample_posterior_predictive` and :meth:`~aimz.ImpactModel.sample_posterior_predictive_on_batch` counterparts.
  Results are stored in ``posterior_predictive`` when ``in_sample=True`` and in ``predictions`` when ``in_sample=False``.

Suppose the kernel defines a sample site named ``z`` that influences the outcome.
Before fitting, you can explore the model's prior predictions with ``z`` fixed at zero:

.. code-block:: python

    im = ImpactModel(model, ...)  # Supply the inference configuration.
    prior = im.sample_prior_predictive_on_batch(X, intervention={"z": 0.0})

After fitting, the same mapping applies the intervention while using posterior samples for the other latent sites.
Here, the baseline holds ``z`` at one and the modified scenario holds it at zero, both drawn with the same key:

.. code-block:: python

    im.fit_on_batch(X, y)

    key = random.key(1)
    baseline = im.predict_on_batch(
        X,
        intervention={"z": 1.0},
        in_sample=False,
        rng_key=key,
    )
    modified = im.predict_on_batch(
        X,
        intervention={"z": 0.0},
        in_sample=False,
        rng_key=key,
    )


Effect Estimation
-----------------
The :meth:`~aimz.ImpactModel.estimate_effect` method compares two predictive scenarios by computing their elementwise difference (``intervention - baseline``).
It accepts scenarios in ``posterior_predictive``, ``predictions``, or ``prior_predictive``, and requires a fitted model only to generate a scenario from an argument dictionary.
The returned :class:`~xarray.DataTree` contains the differences in the shared predictive group, preserving the ``chain`` and ``draw`` dimensions, and includes the stored posterior samples when available, except for prior predictive scenarios.

One baseline and one intervention scenario must be provided, either eagerly (``output_baseline`` / ``output_intervention``) or lazily through argument dictionaries (``args_baseline`` / ``args_intervention``).
Mixing is allowed; for example, a precomputed baseline can be supplied with ``output_baseline`` while the intervention is generated lazily with ``args_intervention`` (or the reverse).
Use the same predictive group, variable sets, and shapes for both scenarios so their draws can be compared.
Scenarios drawn by the same method with the same ``rng_key`` are paired: they use the same random numbers, so their draws differ only through the intervention.
Two lazily generated scenarios share a key automatically; for precomputed or mixed scenarios, pass the same ``rng_key`` to each, as in the examples below.

Eager (precomputed scenarios)::

    effect = im.estimate_effect(
        output_baseline=baseline,
        output_intervention=modified,
    )

Lazy (defer prediction)::

    effect = im.estimate_effect(
        args_baseline={
            "X": X,
            "intervention": {"z": 1.0},
            "in_sample": False,
        },
        args_intervention={
            "X": X,
            "intervention": {"z": 0.0},
            "in_sample": False,
        },
    )

Mixed (precomputed baseline, lazy intervention)::

    effect = im.estimate_effect(
        output_baseline=baseline,
        args_intervention={
            "X": X,
            "intervention": {"z": 0.0},
            "in_sample": False,
            "rng_key": key,
        },
        on_batch=True,
    )

Prior predictive (precomputed scenarios, no fitted model needed)::

    key = random.key(0)
    prior_baseline = im.sample_prior_predictive_on_batch(
        X,
        intervention={"z": 1.0},
        rng_key=key,
    )
    prior_modified = im.sample_prior_predictive_on_batch(
        X,
        intervention={"z": 0.0},
        rng_key=key,
    )
    effect = im.estimate_effect(
        output_baseline=prior_baseline,
        output_intervention=prior_modified,
    )

This gives the effect the model implies before seeing any data.
Pass the same ``rng_key`` to both calls so the two scenarios share their prior draws and differ only by the intervention.

.. note::

   A lazily generated scenario (``args_baseline`` / ``args_intervention``) runs the streaming :meth:`~aimz.ImpactModel.predict` internally with its default persistent store, writing artifacts under the model's temporary directory (unless an ``output_dir`` or ``store`` entry is included in the argument dictionary).
   Because the intermediate trees are not returned, the effect tree's ``artifact_path_baseline`` / ``artifact_path_intervention`` attributes are the only handle to those artifacts; see :doc:`cleanup` for managing them.
   Pass ``on_batch=True`` to compute both scenarios in memory without writing to disk.

The returned :class:`~xarray.DataTree` captures the elementwise difference for every variable present in the predictive group.
Any subsequent summary (e.g. mean, intervals) can be computed using Xarray, ArviZ, or standard NumPy / JAX utilities.
See :doc:`estimands` for common summaries, such as the ATE, ATT, CATE, and relative effect.

.. note::

   :meth:`~aimz.ImpactModel.estimate_effect` computes the predictive contrast between two scenarios under structural interventions, propagating the full posterior (or prior) uncertainty through the difference.
   Whether this contrast admits a causal interpretation depends on the structural assumptions encoded in the model (the kernel): causal identification is a property of the model specification, not the estimation procedure.
   When the user-defined model encodes appropriate causal assumptions, such as conditioning on confounders and specifying correct functional relationships, this contrast corresponds to a causal effect estimate.


Example: Causal Network with Confounder
---------------------------------------
The following example uses posterior predictive interventions to estimate effects in a simple causal network.
The variable ``Z`` has a direct causal effect on the outcome ``Y``, while both are influenced by a shared confounder, ``C``.
An additional variable, ``X``, is an observed exogenous factor that influences ``Z`` but has no direct effect on ``Y``.

Our objective is to estimate the causal effect of ``Z`` (or alternatively ``X``) on ``Y``, while properly accounting for the confounding influence of ``C``.
We assume the following generative model for the observed data:

Model
~~~~~

.. jupyter-execute::

    import logging

    import jax.numpy as jnp
    import numpyro.distributions as dist
    from jax import nn, random
    from jax.typing import ArrayLike
    from numpyro import optim, plate, sample
    from numpyro.infer import SVI, Trace_ELBO, init_to_feasible
    from numpyro.infer.autoguide import AutoNormal

    from aimz import ImpactModel

    logging.basicConfig(level=logging.INFO, force=True)


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
        with plate("obs", size=X.shape[0]):
            mu_z = beta_z + beta_cz * c + beta_xz * X.squeeze(axis=1)
            z = sample("z", dist.LogNormal(mu_z, sigma))

            logits = beta_y + beta_cy * c + beta_zy * z
            sample("y", dist.Bernoulli(logits=logits), obs=y)


Simulating Data under a Known Structural Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We generate synthetic data consistent with the assumed structure:

- ``C`` is drawn from an exponential distribution.
- ``X`` is a count variable from a Poisson distribution.
- ``Z`` is generated as a noisy exponential function of ``C`` and ``X``.
- ``Y`` is a binary outcome influenced by both ``C`` and ``Z`` through a logistic model.

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
Once trained, we perform a counterfactual analysis to isolate the effect of ``Z`` on ``Y``.

- ``dt_factual`` represents predictions under the factual setting (with observed ``Z``).
- ``dt_counterfactual`` represents predictions under a counterfactual intervention where ``Z`` is set to zero.

.. note::

    The example uses :meth:`~aimz.ImpactModel.predict_on_batch` for brevity.
    :meth:`~aimz.ImpactModel.predict` also handles this model's local latent variable: with the default ``batch_size`` on a single device it processes the whole input as one batch, and otherwise it warns and reruns under ``shard_axis="draw"``.
    See :ref:`model compatibility <faq-model-compatibility>` for details.

Comparing these two distributions allows us to estimate the effect of ``Z`` on ``Y``, adjusted for the influence of ``C``.

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

    # Predict under factual (Z) and counterfactual (zeroed Z) scenarios with one key
    rng_key, rng_subkey = random.split(rng_key)
    dt_factual = im.predict_on_batch(
        X,
        C=C,
        intervention={"z": Z},
        rng_key=rng_subkey,
    )
    dt_counterfactual = im.predict_on_batch(
        X,
        C=C,
        intervention={"z": jnp.zeros_like(Z)},
        rng_key=rng_subkey,
    )

    # Estimate effect of intervening on Z while conditioning on C
    effect = im.estimate_effect(
        output_baseline=dt_factual,
        output_intervention=dt_counterfactual,
    )
    effect

.. jupyter-execute::
    :hide-code:

    effect
