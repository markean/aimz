# aimz: Scalable probabilistic impact modeling


## Overview
**aimz** is a Python library for flexible and scalable probabilistic impact modeling to assess the effects of interventions on outcomes of interest.
Designed to work with user-defined models with probabilistic primitives, the library builds on [NumPyro](https://num.pyro.ai/en/stable/), [JAX](https://jax.readthedocs.io/en/latest/), [Xarray](https://xarray.dev/), and [Zarr](https://zarr.readthedocs.io/en/stable/) to enable efficient inference workflows.


## Features
- An intuitive API that combines ease of use from ML frameworks with the flexibility of probabilistic modeling.
- Scalable computation via parallelism and distributed data processing—no manual orchestration required.
- Variational inference as the primary inference engine, supporting custom optimization strategies and results.
- Support for interventional causal inference for modeling counterfactuals and causal relations.


## Workflow
1. Outline the model, considering the data generating process, latent variables, and causal relationships, if any.
2. Translate the model into a **kernel** (i.e., a function) using NumPyro and JAX.
3. Integrate the kernel into the provided API to train the model and perform inference.
