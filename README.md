# aimz: Scalable probabilistic impact modeling

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![Run Pytest](https://github.com/markean/aimz/actions/workflows/coverage.yaml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/aimz)](https://pypi.org/project/aimz/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/aimz.svg)](https://anaconda.org/conda-forge/aimz)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://pypi.org/project/aimz/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/markean/aimz/graph/badge.svg?token=34OH7KQBXE)](https://codecov.io/gh/markean/aimz)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.09738/status.svg)](https://doi.org/10.21105/joss.09738)
[![DOI](https://zenodo.org/badge/1009062911.svg)](https://doi.org/10.5281/zenodo.16101876)

[**Installation**](https://aimz.readthedocs.io/stable/getting_started/installation.html) |
[**Tutorial**](https://aimz.readthedocs.io/latest/getting_started/tutorial.html) |
[**User Guide**](https://aimz.readthedocs.io/latest/user_guide/index.html) |
[**FAQs**](https://aimz.readthedocs.io/latest/faq.html) |
[**Changelog**](https://aimz.readthedocs.io/latest/changelog.html)

## Overview

aimz is a Python library for scalable probabilistic impact modeling—estimating how interventions affect outcomes while quantifying uncertainty.

It provides a high-level, object-oriented interface on top of [NumPyro](https://num.pyro.ai/en/stable/) and [JAX](https://jax.readthedocs.io/en/latest/) for building, fitting, and scaling Bayesian models: a user-defined NumPyro model is wrapped as a "kernel" inside a single class, augmented with capabilities for scalable predictive sampling, structured outputs, and experiment tracking.

## Key capabilities

- **Object-oriented interface for NumPyro models:**
Bring any NumPyro model as a "kernel" and access `fit`, `predict`, `sample`, and related methods through a single class—aimz does not enforce a fixed architecture.
- **Scalable predictive sampling:**
JIT-compiled, sharded sampling streams results to chunked [Zarr](https://zarr.readthedocs.io/en/stable/) stores, enabling large-scale posterior predictive simulations that do not need to fit in memory.
- **Structured outputs:**
Predictions, samples, and effect estimates are materialized as [Xarray](https://xarray.dev/) objects backed by Zarr, integrating cleanly with the scientific Python ecosystem.
- **Intervention handling and impact modeling:**
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

## Citation

If you use aimz in your work, please cite the accompanying paper in the [Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.09738):

```bibtex
@article{Kim2026,
  title        = {aimz: Scalable probabilistic impact modeling},
  author       = {Kim, Eunseop},
  year         = 2026,
  journal      = {Journal of Open Source Software},
  publisher    = {The Open Journal},
  volume       = 11,
  number       = 120,
  pages        = 9738,
  doi          = {10.21105/joss.09738},
  url          = {https://doi.org/10.21105/joss.09738}
}
```
