# aimz: Scalable probabilistic impact modeling

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![Run Pytest](https://github.com/markean/aimz/actions/workflows/coverage.yaml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/aimz)](https://pypi.org/project/aimz/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/aimz.svg)](https://anaconda.org/conda-forge/aimz)
[![Python](https://img.shields.io/pypi/pyversions/aimz.svg)](https://pypi.org/project/aimz/)
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

aimz is a Python library for scalable probabilistic impact modeling, enabling assessment of intervention effects on outcomes with a streamlined interface for fitting, sampling, prediction, and effect estimation—minimal boilerplate, accelerated execution, and powered by [NumPyro](https://num.pyro.ai/en/stable/), [JAX](https://jax.readthedocs.io/en/latest/), [Xarray](https://xarray.dev/), and [Zarr](https://zarr.readthedocs.io/en/stable/).

## Features

- Intuitive API combining the ease of use from ML frameworks with the flexibility of probabilistic modeling.
- Accelerated computation via parallelism and distributed data.
- Support for interventional causal inference for counterfactuals and causal effects.
- MLflow integration for experiment tracking and model management.

## Installation

Install aimz using either `pip` or `conda`:

```sh
pip install -U aimz
```

```sh
conda install -c conda-forge aimz
```

For additional details, see the full [installation guide](https://aimz.readthedocs.io/stable/getting_started/installation.html).

## Usage

```python
from aimz import ImpactModel

# Define probabilistic model (kernel) using Numpyro primitives
def model(X, y=None):
    ...

# Load or prepare data
X, y = ...

# Initialize ImpactModel
im = ImpactModel(
    model,
    rng_key=...,      # e.g., jax.random.key(0)
    inference=...,    # e.g., SVI (or MCMC)
)

# Fit model and draw posterior samples
im.fit(X, y)

# Make predictions or posterior predictive samples
dt = im.predict(X)
```

## Contributing

See the [Contributing Guide](https://aimz.readthedocs.io/latest/development/contributing.html) to get started.
