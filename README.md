# aimz: Scalable probabilistic impact modeling

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![Run Pytest](https://github.com/markean/aimz/actions/workflows/coverage.yaml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/aimz)](https://pypi.org/project/aimz/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/aimz.svg)](https://anaconda.org/conda-forge/aimz)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://pypi.org/project/aimz/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/markean/aimz/graph/badge.svg?token=34OH7KQBXE)](https://codecov.io/gh/markean/aimz)
[![DOI](https://zenodo.org/badge/1009062911.svg)](https://doi.org/10.5281/zenodo.16101876)

[**Installation**](https://aimz.readthedocs.io/stable/getting_started/installation.html) |
[**Tutorial**](https://aimz.readthedocs.io/latest/getting_started/tutorial.html) |
[**User Guide**](https://aimz.readthedocs.io/latest/user_guide/index.html) |
[**FAQs**](https://aimz.readthedocs.io/latest/faq.html) |
[**Changelog**](https://aimz.readthedocs.io/latest/changelog.html)

## Overview

aimz is a Python library for scalable probabilistic impact modeling—estimating how interventions affect outcomes while quantifying uncertainty. It provides an intuitive interface for fitting Bayesian models, drawing posterior samples, generating large-scale posterior predictive simulations, and estimating intervention effects with minimal boilerplate.

## Key capabilities

- **Flexible model specification:**
Built on [NumPyro](https://num.pyro.ai/en/stable/) and [JAX](https://jax.readthedocs.io/en/latest/), bring a NumPyro model as a "kernel"—aimz does not enforce a fixed architecture.
- **Scalable predictive sampling:**
JIT-compiled, sharded sampling streams results to chunked [Zarr](https://zarr.readthedocs.io/en/stable/) stores, enabling large-scale posterior predictive simulations.
- **Structured outputs:**
Predictions, samples, and effect estimates are materialized as [Xarray](https://xarray.dev/) objects backed by Zarr, integrating cleanly with the scientific Python ecosystem.
- **Intervention handling:**
Specify interventions declaratively and estimate effects from posterior predictive distributions.
- **Experiment tracking:**
[MLflow](https://mlflow.org/) integration for logging runs, parameters, metrics, and model artifacts with full lineage.

## Installation

Install aimz using either `pip` or `conda`:

```sh
pip install -U aimz
```

```sh
conda install -c conda-forge aimz
```

For additional details, see the full [installation guide](https://aimz.readthedocs.io/stable/getting_started/installation.html).

## Quick start

```python
from aimz import ImpactModel

# Define a probabilistic model (kernel) using NumPyro primitives
def model(X, y=None):
    ...

# Load or prepare data
X, y = ...

# Initialize ImpactModel with SVI or MCMC inference
im = ImpactModel(
    model,
    rng_key=...,      # e.g., jax.random.key(0)
    inference=...,    # e.g., SVI (or MCMC)
)

# Fit model and draw posterior samples
im.fit(X, y)

# Generate posterior predictive samples
dt = im.predict(X)

# Estimate intervention effects
dt_baseline = im.predict(X)
dt_intervention = im.predict(X, intervention={"treatment": 1.0})
effect = im.estimate_effect(dt_baseline, dt_intervention)
```

## Contributing

See the [Contributing Guide](https://aimz.readthedocs.io/latest/development/contributing.html) to get started.
