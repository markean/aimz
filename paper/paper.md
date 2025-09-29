---
title: 'aimz: Scalable probabilistic impact modeling'
tags:
  - Python
  - JAX
  - Bayesian inference
  - uncertainty quantification
  - probabilistic modeling
  - uplift modeling
authors:
  - name: Eunseop Kim
    orcid: 0009-0000-2138-788X
    affiliation: 1
affiliations:
 - name: Eli Lilly and Company, United States
   index: 1
date: 29 September 2025
bibliography: paper.bib
---

# Summary

`aimz` is a Python library for scalable probabilistic impact modeling, enabling assessment of intervention effects on outcomes while providing an intuitive interface for fitting Bayesian models, drawing posterior samples, generating large-scale posterior predictive simulations, and estimating interventional effects with minimal boilerplate.
It combines the usability of general machine learning APIs with the flexibility of probabilistic programming through a single high-level object (`ImpactModel`).
Built atop JAX [@jax2018github] and NumPyro [@phan2019composable], it supports (minibatch) stochastic variational inference (SVI) and Markov chain Monte Carlo sampling, just-in-time (JIT)-compiled parallel predictive streaming to chunked Zarr [@alistair_miles_2020_3773450] stores exposed through Xarray [@hoyer2017xarray], and first-class intervention handling for effect estimation.
Integrated MLflow [@Zaharia_Accelerating_the_Machine_2018] support enables experiment tracking and model lineage.
These design choices reduce bespoke glue code and enable reproducible, high-throughput analyses on large datasets, while supporting rapid iteration and experimentation.

# Statement of need

Applied analytics workflows often require: (1) fitting probabilistic models to large scale datasets, (2) generating posterior and posterior predictive samples for calibrated uncertainty, and (3) estimating intervention effects under explicit structural modifications.
While core probabilistic programming frameworks (e.g., NumPyro, PyMC [@pymc2023], Stan [@carpenter2017stan]) offer mature inference algorithms, recurring engineering tasks—such as streaming large predictive draws to disk, structuring outputs, coordinating intervention scenarios, managing device resources, and logging experiments—are typically reimplemented in an ad hoc manner.
General machine learning libraries (e.g., scikit‑learn [@scikit-learn]) lack native Bayesian sampling or causal intervention semantics, while many causal inference toolkits emphasize causal graph discovery or fixed-form treatment effect routines rather than scalable sampling workflows.

`aimz` consolidates these infrastructural concerns within a single object (`ImpactModel`) that provides: probabilistic model tracing and argument binding; SVI or MCMC with automatic posterior sample management; JIT‑compiled, sharded posterior predictive sampling with concurrent streaming to Zarr stores surfaced as Xarray objects; structured intervention ("do-operation") application; and optional MLflow integration for experiment tracking.
The familiar "`fit` / `predict` / `sample`" interface further eases integration with machine learning tooling, emerging Model Context Protocol pipelines, and generative AI assistants that assume estimator-like semantics.
This unification reduces redundant glue code and minimizes potential failure points in applications such as marketing mix modeling, policy evaluation, and attribution of program impacts.

Existing impact or uplift modeling libraries (e.g., domain‑specific frameworks such as Meridian [@meridian_github], Robyn [@robyn], or PyMC-Marketing [@pymc-marketing]) typically standardize on a constrained family of time‑series or marketing response models and a fixed inference stack, making it difficult to deviate from their built-in assumptions without forking code or re-implementing infrastructure.
`aimz` instead pursues generality—accepting arbitrary NumPyro model functions and multiple inference strategies—while still avoiding boilerplate through streamed predictive simulation, structured outputs, and intervention orchestration.
By elevating scalable posterior predictive simulation and intervention effect estimation to first-class capabilities, `aimz` lowers the barrier between exploratory probabilistic modeling and production-grade Bayesian impact analysis.

## References
