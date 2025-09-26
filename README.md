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
[**Documentation**](https://aimz.readthedocs.io/) |
[**FAQs**](https://aimz.readthedocs.io/latest/faq.html) |
[**Changelog**](https://aimz.readthedocs.io/latest/changelog.html)

## Overview

**aimz** is a Python library for flexible and scalable probabilistic impact modeling to assess the effects of interventions on outcomes of interest.
Designed to work with user-defined models with probabilistic primitives, the library builds on [NumPyro](https://num.pyro.ai/en/stable/), [JAX](https://jax.readthedocs.io/en/latest/), [Xarray](https://xarray.dev/), and [Zarr](https://zarr.readthedocs.io/en/stable/) to enable efficient inference workflows.

## Features

- An intuitive API that combines ease of use from ML frameworks with the flexibility of probabilistic modeling.
- Scalable computation via parallelism and distributed data processing—no manual orchestration required.
- Variational inference as the primary inference engine, supporting custom optimization strategies and results.
- Support for interventional causal inference for modeling counterfactuals and causal relations.
- MLflow integration for experiment tracking and model management.

## Usage

### Workflow

1. Outline the model, considering the data generating process, latent variables, and causal relationships, if any.
2. Translate the model into a **kernel** (i.e., a function) using NumPyro and JAX.
3. Integrate the kernel into the provided API to train the model and perform inference.

### Example: Regression Using a scikit-learn-like Workflow

This example demonstrates a simple regression model following a typical ML workflow. The `ImpactModel` class provides `.fit()` and `.fit_on_batch()` for variational inference and posterior sampling, and `.predict()` and `.predict_on_batch()` for posterior predictive sampling. The optional `.cleanup()` removes posterior predictive samples saved as temporary files.

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from jax.typing import ArrayLike
from numpyro import optim, plate, sample
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from aimz import ImpactModel

# Load California Housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# NumPyro model: linear regression
def model(X: ArrayLike, y: ArrayLike | None = None) -> None:
    """Bayesian linear regression."""
    n_features = X.shape[1]

    # Priors for weights, bias, and observation noise
    w = sample("w", dist.Normal(jnp.zeros(n_features), jnp.ones(n_features)))
    b = sample("b", dist.Normal())
    sigma = sample("sigma", dist.Exponential())

    # Plate over data
    mu = jnp.dot(X, w) + b
    with plate("data", X.shape[0]):
        sample("y", dist.Normal(mu, sigma), obs=y)


# Wrap with ImpactModel
im = ImpactModel(
    model,
    rng_key=random.key(42),
    inference=SVI(
        model,
        guide=AutoNormal(model),
        optim=optim.Adam(step_size=1e-3),
        loss=Trace_ELBO(),
    ),
)

# Fit the model: variational inference followed by posterior sampling
im.fit_on_batch(X_train, y_train)

# Predict on new data using posterior predictive sampling
idata = im.predict(X_test)

# Clean up posterior predictive samples saved to disk during `.predict()`
im.cleanup()
```

> The training step can be skipped if pre-trained variational inference results or posterior samples are available. These can be integrated into the `ImpactModel`, allowing `.predict()` to be available subsequently.

## Contributing

See the [Contributing Guide](https://aimz.readthedocs.io/latest/development/contributing.html) to get started.
