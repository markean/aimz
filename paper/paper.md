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

`aimz` is a Python library for probabilistic impact modeling, a method for estimating how actions or interventions affect outcomes while capturing uncertainty.
It provides an intuitive interface for fitting Bayesian models, drawing posterior samples, generating large-scale posterior predictive simulations, and estimating interventional effects with minimal boilerplate.
Researchers and analysts can explore “what-if” scenarios to understand potential effects under different decisions or interventions, generating predictions that reflect the full range of possible outcomes.
`aimz` supports advanced inference methods with efficient, reproducible workflows, making it easy to run large-scale simulations and serialize structured outputs.
These design choices reduce bespoke glue code, enable high-throughput analyses, and support rapid iteration and experimentation.

It combines the usability of general machine learning APIs with the flexibility of probabilistic programming through a single high-level object (`ImpactModel`).
Built atop JAX [@jax2018github] and NumPyro [@phan2019composable], it supports (minibatch) stochastic variational inference (SVI) and Markov chain Monte Carlo sampling, just-in-time (JIT)-compiled parallel predictive streaming to chunked Zarr [@alistair_miles_2020_3773450] stores exposed through Xarray [@hoyer2017xarray], and first-class intervention handling for effect estimation.
Integrated MLflow [@zaharia2018accelerating] support enables experiment tracking and model lineage.

# Statement of need

Standard Bayesian workflows often encounter significant engineering friction when transitioning from model specification to large-scale impact evaluation.
While core probabilistic programming frameworks offer mature inference algorithms, they lack native infrastructure for the repetitive engineering tasks essential to production-grade impact analysis: streaming predictive draws without exceeding system memory, managing device-aware sharding for simulation, and coordinating complex interventional scenarios.
General machine learning libraries lack the calibrated uncertainty quantification provided by Bayesian sampling, while many causal inference toolkits focus on graph discovery rather than high-throughput predictive workflows.
`aimz` addresses these gaps by consolidating model tracing, sharded predictive sampling, and experiment lineage into a single estimator-like object.
This reduces the bespoke engineering effort required to connect flexible statistical research with high-throughput data pipelines, supporting reliable and reproducible probabilistic analyses at scale.

# State of the field

The landscape of Bayesian software is largely divided between low-level flexibility and high-level rigidity.
Core probabilistic programming languages like NumPyro, PyMC [@pymc2023], and Stan [@carpenter2017stan] provide the flexible primitives necessary for custom research but require users to manually handle scaling, parallelization, and other performance considerations.
Conversely, domain-specific frameworks like Meridian [@meridian_github], Robyn [@robyn], or PyMC-Marketing [@pymc-marketing] can offer robust end-to-end pipelines but are specialized for marketing mix modeling.
These tools frequently enforce fixed model architectures (e.g., specific adstock or saturation transformations) and opaque internal logic that are difficult to adapt for broader scientific research, such as evaluating clinical care gaps---where individual-level discrepancies between recommended best practices and actual care require custom hierarchical specifications and structural modifications.

`aimz` is designed to target a middle ground by providing Bayesian infrastructure in an estimator-like form.
It does not prescribe a specific model form; instead, it provides a scalable execution layer for user-defined models.
While libraries like CausalPy [@causalpy_github] provide sophisticated, high-level interfaces for a variety of quasi-experimental designs, they are typically optimized for exploratory analysis and causal identification.
`aimz` distinguishes itself by focusing on high-throughput engineering for large-scale probabilistic simulations and on structured, reusable artifact management.
This makes `aimz` uniquely suited for production-grade Bayesian workflows where reproducibility, experiment lineage, and the parallelized simulation of custom interventions are critical requirements.

# Software design

`aimz` is designed to support scalable Bayesian analyses in applied settings, allowing users to iterate quickly across model specifications, inference settings, and intervention scenarios, as well as to handle large datasets and produce reproducible artifacts.
It combines the usability of general machine learning APIs with the flexibility of probabilistic programming through a single high-level object (`ImpactModel`).
Built atop JAX [@jax2018github] and NumPyro [@phan2019composable], `aimz` leverages modeling flexibility, accelerator-native execution, and composable program transformations such as vectorization and sharded execution.
It supports (minibatch) stochastic variational inference (SVI) and Markov chain Monte Carlo sampling, just-in-time–compiled parallel predictive streaming to chunked Zarr [@alistair_miles_2020_3773450] stores exposed through Xarray [@hoyer2017xarray], and first-class intervention handling for effect estimation.
Integrated MLflow [@zaharia2018accelerating] support enables experiment tracking and model lineage.

`aimz` prioritizes performance as a key consideration.
Predictive sampling and effect estimation are organized around JIT-accelerated execution, data-parallel sharding, and streaming results to disk-backed, chunked storage, enabling sustained throughput for large simulation workloads without requiring all draws to be retained in memory.
This architectural choice comes with trade-offs: only models that are compatible with sharded execution can fully leverage these optimizations.
Additional orchestration around execution and I/O is required, which `aimz` manages through a small set of high-level methods, enabling users to focus on analysis while maintaining a consistent interface.
At the user interface level, `aimz` adopts an estimator-like API centered on the `ImpactModel` class, which takes a NumPyro model function as its “kernel” (the primary user-specified argument) and exposes familiar estimator methods while still supporting a wide range of model specifications.

`aimz` is also designed for integration: results are materialized as structured artifacts (e.g., Xarray objects backed by Zarr stores) and can optionally be logged via MLflow for experiment tracking and lineage.
The combination of a stable, method-based interface and standardized outputs makes `aimz` well suited for AI-enabled workflows, where agentic tools can invoke a small set of operations and reliably consume the resulting artifacts.

# Research impact statement

`aimz` enables a class of Bayesian modeling and probabilistic analyses that are otherwise costly to implement and difficult to reproduce at scale: including fitting flexible probabilistic models, generating large posterior predictive simulations, and estimating intervention effects under explicit structural modifications.
Its research impact is therefore primarily infrastructural---lowering the engineering barrier to rigorous uncertainty-aware effect estimation---while remaining general enough to support diverse model specifications and inference strategies.

Within Eli Lilly and Company, `aimz` has supported internal analytics workflows that require scalable posterior and posterior predictive sampling, consistent output structures, and experiment traceability.
In this setting, the library has reduced duplicated “glue” code for streaming predictive draws, coordinating intervention scenarios, and tracking model lineage, thereby improving iteration speed and reproducibility across analyses.
Its stable estimator-like interface and structured, reproducible outputs also make it well suited for integration with AI-enabled workflows, where agentic tools can leverage standardized artifacts for downstream tasks.

Beyond immediate use, `aimz` is designed as reusable research infrastructure: it exposes arbitrary NumPyro model functions through a stable estimator-like interface and standardizes artifacts (samples, predictions, and effect estimates) in formats that integrate cleanly with the broader scientific Python ecosystem (e.g., Xarray/Zarr) and with experiment tracking via MLflow.
This makes it easier for researchers and applied scientists to share models and results, compare alternative specifications, and operationalize workflows on larger datasets than would be practical with bespoke scripts.
Early external adoption is reflected in package index downloads: since its initial public release in June 2025, `aimz` has been downloaded over 10,000 times via PyPI and conda-forge as of January 2026.

# AI usage disclosure

Microsoft Copilot was used solely to assist with drafting code docstrings and adding type hints. All suggestions generated by Copilot were carefully reviewed and edited to ensure accuracy and consistency with the implemented functionality. No other generative AI tools were used in the development of the software, the writing of the manuscript, or the preparation of supporting materials.

# Acknowledgements

The author acknowledges support from Eli Lilly and Company.

# References
