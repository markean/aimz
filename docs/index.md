# aimz: Flexible probabilistic impact modeling at scale
[![Python](https://img.shields.io/pypi/pyversions/aimz.svg)](https://pypi.org/project/aimz/)
[![PyPI version](https://img.shields.io/pypi/v/aimz)](https://pypi.org/project/aimz/)
[![codecov](https://codecov.io/gh/markean/aimz/graph/badge.svg?token=34OH7KQBXE)](https://codecov.io/gh/markean/aimz)


## Overview
**aimz** is a Python library for flexible and scalable probabilistic impact modeling to assess the effects of interventions on outcomes of interest.
Designed to work with user-defined models with probabilistic primitives, the library builds on [NumPyro](https://num.pyro.ai/en/stable/), [JAX](https://jax.readthedocs.io/en/latest/), [Xarray](https://xarray.dev/), and [Zarr](https://zarr.readthedocs.io/en/stable/) to enable efficient inference workflows.


## Features
- An intuitive API that combines ease of use from ML frameworks with the flexibility of probabilistic modeling.
- Scalable computation via parallelism and distributed data processing—no manual orchestration required.
- Variational inference as the primary inference engine, supporting custom optimization strategies and results.
- Support for interventional causal inference for modeling counterfactuals and causal relations.


## Installation
CPU (default):
```sh
pip install -U aimz
```

GPU (NVIDIA, CUDA 12):
```sh
pip install -U "aimz[gpu]"
```
This installs `jax[cuda12]` with the version specified by the package. However, to ensure you have the latest compatible version of JAX with CUDA 12, it is recommended to update JAX separately after installation:
```sh
pip install -U "jax[cuda12]"
```
Refer to the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for up-to-date compatibility and driver requirements.


## Workflow
1. Outline the model, considering the data generating process, latent variables, and causal relationships, if any.
2. Translate the model into a **kernel** (i.e., a function) using NumPyro and JAX.
3. Integrate the kernel into the provided API to train the model and perform inference.