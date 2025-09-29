---
title: 'aimz: Scalable probabilistic impact modeling'
tags:
  - Python
  - JAX
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

`aimz` is a Python library for scalable probabilistic impact modeling, providing an intuitive interface for fitting Bayesian models, drawing posterior samples, generating large-scale posterior predictive simulations, and estimating interventional effects with minimal boilerplate.
It combines the usability of general ML APIs with the flexibility of probabilistic programming through a single high-level object (`ImpactModel`).
Built atop JAX [@jax2018github] and NumPyro [@phan2019composable], it supports intuitive kernel specification, minibatch SVI (or MCMC), JIT‑compiled parallel predictive streaming to chunked Zarr stores [@alistair_miles_2020_3773450] exposed as Xarray [@hoyer2017xarray] objects, and first-class intervention handling for effect estimation.
Integrated MLflow [@Zaharia_Accelerating_the_Machine_2018] support enables experiment tracking and model lineage.
These design choices reduce bespoke glue code and enable reproducible, high-throughput Bayesian impact analyses on large datasets.

# Statement of need

Applied analytics teams often need unified pipelines that (1) fit probabilistic models on large tabular datasets, (2) generate posterior and posterior predictive samples for uncertainty quantification, and (3) estimate interventional effects under explicit structural modifications.
Core probabilistic frameworks (NumPyro, PyMC [@salvatier2016probabilistic], Stan [@carpenter2017stan]) expose powerful inference primitives but leave recurrent engineering tasks—batched posterior predictive streaming, structured disk persistence, effect estimation orchestration, resource detection, and experiment logging—to bespoke, project‑specific code.
General machine learning libraries (e.g., scikit‑learn [@scikit-learn]) do not natively integrate Bayesian sampling or causal intervention semantics, while causal packages frequently emphasize identification estimators over scalable posterior workflows.

`aimz` emphasizes an end‑to‑end workflow: model definition (NumPyro primitives), fitting (SVI or MCMC), posterior sampling, posterior predictive generation, interventional ("do‑operation") effect estimation, and artifact streaming to chunked Zarr stores [@zarr] exposed as Xarray DataTrees [@hoyer2017xarray]. Deterministic sites are captured so probabilities or other intermediates can be returned without redundant forward passes. Parallel I/O and sharded sampling enable efficient scaling, while optional MLflow integration [@mlflow] supports experiment tracking and model provenance. By packaging common infrastructure—input validation, device detection, batching, structured output management, and intervention handling—`aimz` reduces repetitive glue code and accelerates reproducible Bayesian impact analyses in domains such as marketing mix modeling, policy evaluation, and program attribution.

`aimz` addresses this gap by centering a single object (`ImpactModel`) that encapsulates:

- Probabilistic kernel tracing and argument specification.
- Minibatch SVI (or MCMC) with automatic posterior sampling and loss diagnostics.
- Sharded, JIT‑compiled posterior predictive generation with concurrent disk streaming to Zarr (suitable for very large *n*).
- Intervention (do‑operation) application via effect handlers, enabling direct difference computations on aligned predictive groups.
- Structured Xarray DataTree outputs, facilitating downstream statistical summaries or integration with ArviZ [@kumar2019arviz].
- Optional MLflow autologging for parameters, losses, signatures, and serialized models.

This consolidation reduces boilerplate and potential error surfaces, promoting reproducibility and clarity in impact analysis pipelines. The current scope focuses on dense 2D tabular data—a high‑leverage regime for many real‑world business and policy applications. Planned extensions include native support for higher‑dimensional or sparse modalities. Community feedback and issue submissions are encouraged to guide prioritization.

By framing intervention effect estimation and predictive simulation as first‑class tasks with scalable execution, `aimz` lowers the operational barrier between exploratory modeling and production‑grade Bayesian impact analytics.

## References
