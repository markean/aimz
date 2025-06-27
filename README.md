# aimz: Flexible probabilistic impact modeling at scale
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)]()
[![Coverage Status](https://coveralls.io/repos/github/markean/aimz/badge.svg?branch=main)](https://coveralls.io/github/markean/aimz?branch=main)


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


## Usage
### Workflow
1. Outline the model, considering the data generating process, latent variables, and causal relationships, if any.
2. Translate the model into a **kernel** (i.e., a function) using NumPyro and JAX.
3. Integrate the kernel into the provided API to train the model and perform inference.

### Example 1: Regression Using a scikit-learn-like Workflow
This example demonstrates a simple regression model following a typical ML workflow. The `ImpactModel` class provides `.fit()` for variational inference and posterior sampling, and `.predict()` for posterior predictive sampling. The optional `.cleanup()` removes posterior predictive samples saved as temporary files.
```python
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import random
from numpyro import optim, plate, sample
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from aimz.model import ImpactModel

# Load California Housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# NumPyro model: linear regression
def model(X: np.ndarray, y: np.ndarray | None = None) -> None:
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
    vi=SVI(
        model,
        guide=AutoNormal(model),
        optim=optim.Adam(step_size=1e-3),
        loss=Trace_ELBO(),
    ),
)

# Fit the model: variational inference followed by posterior sampling
im.fit(X_train, y_train)

# Predict on new data using posterior predictive sampling
idata = im.predict(X_test)

# Clean up posterior predictive samples saved to disk during `.predict()`
im.cleanup()
```
> The `.fit()` step can be skipped if pre-trained variational inference results or posterior samples are available. These can be integrated into the `ImpactModel`, allowing `.predict()` to be available subsequently.

### Example 2: Causal Network with Confounder
This example illustrates a simple causal network. The variable `Z` has a direct causal effect on the outcome `Y`, while both are influenced by a shared confounder, `C`. An additional variable, `X`, is an observed exogenous factor that influences `Z` but has no direct effect on `Y`.

Our objective is to estimate the causal effect of `Z` (or alternatively `X`) on `Y`, while properly accounting for the confounding influence of `C`. We assume the following generative model for the observed data:

```python
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import nn, random
from numpyro import optim, plate, sample
from numpyro.infer import SVI, Trace_ELBO, init_to_feasible
from numpyro.infer.autoguide import AutoNormal

from aimz.model import ImpactModel


# NumPyro model: Z and y are influenced by C and X, with Z mediating part of y
def model(X: jax.Array, C: jax.Array, y: jax.Array | None = None) -> None:
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
    with plate("data", X.shape[0]):
        mu_z = beta_z + beta_cz * c + beta_xz * X.squeeze(axis=1)
        z = sample("z", dist.LogNormal(mu_z, sigma))

        logits = beta_y + beta_cy * c + beta_zy * z
        sample("y", dist.Bernoulli(logits=logits), obs=y)
```

#### Simulating data under a known structural model
We generate synthetic data consistent with the assumed causal structure:
- `C` is drawn from an exponential distribution.
- `X` is a count variable from a Poisson distribution.
- `Z` is generated as a noisy exponential function of `C` and `X`.
- `Y` is a binary outcome influenced by both `C` and `Z` through a logistic model.

```python
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
```

#### Fitting the model and estimating causal effects
We fit the model using stochastic variational inference. Once trained, we perform a counterfactual analysis to isolate the effect of `Z` on `Y`.
- `idata_factual` represents predictions under the factual setting (with observed `Z`).
- `idata_counterfactual` represents predictions under a counterfactual intervention where `Z` is set to zero.
Comparing these two distributions allows us to estimate the causal effect of `Z` on `Y`, adjusted for the influence of `C`.

```python
# Fit the model with SVI
im = ImpactModel(
    model,
    rng_key=rng_key,
    vi=SVI(
        model,
        guide=AutoNormal(model, init_loc_fn=init_to_feasible()),
        optim=optim.Adam(step_size=1e-3),
        loss=Trace_ELBO(),
    ),
)
im.fit(X, y, C=C)

# Predict under factual (Z) and counterfactual (zeroed Z) scenarios
idata_factual = im.predict_on_batch(X, C=C, intervention={"z": Z})
idata_counterfactual = im.predict_on_batch(
    X,
    C=C,
    intervention={"z": jnp.zeros_like(Z)},
)

# Estimate causal effect of intervening on Z while conditioning on C
impact = im.estimate_effect(
    output_baseline=idata_factual,
    output_intervention=idata_counterfactual,
)
```
> Local latent variable requires `.predict_on_batch()` here. Prefer `.predict()` whenever it is compatible with the model.


## Getting Help
For feature requests, assistance, or any inquiries, contact maintainers or open an issue/pull request.


## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.
