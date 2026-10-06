---
name: aimz
description: >-
  Use whenever code imports `aimz` or mentions `ImpactModel`, `estimate_effect`, or
  `aimz.mlflow`, and whenever a treatment effect, lift, impact, or a what-if or
  counterfactual scenario is to be computed with a model built on aimz, even if the
  request names none of its methods. Covers what aimz requires of the NumPyro model it
  wraps, the call defaults that suit an agent, how to express a scenario depending on
  where the treatment enters the model, how to keep two scenarios paired, how to
  summarize an effect, the mistakes that return a number without an error, and when
  not to report an effect. aimz changes between minor releases, so read this file
  instead of relying on memory of its API.
---

# aimz

aimz wraps a NumPyro model that the user writes, called the kernel, in one class, `ImpactModel`. It fits the kernel, draws predictions into an `xarray.DataTree`, and subtracts two scenarios with `estimate_effect`. It ships no model, does not check that a fit converged, and does not decide whether a contrast is causal.

This file ships inside the installed package, so it describes the version that `aimz.__version__` reports. The documentation is at https://aimz.readthedocs.io, with an index for agents at https://aimz.readthedocs.io/stable/llms.txt.

## The calls in order

The kernel and the inference object are NumPyro's and the user's to write. aimz adds the calls around them:

```python
import jax
import numpy as np

from aimz import ImpactModel


def model(X, trt, y=None): ...  # the kernel


im = ImpactModel(model, rng_key=jax.random.key(0), inference=...)  # SVI or MCMC
im.fit_on_batch(X, y, trt=trt, progress=False)

shared = {"X": X, "return_sites": "mu", "store": "memory", "progress": False}
effect = im.estimate_effect(
    args_baseline={**shared, "trt": np.zeros(n_obs)},
    args_intervention={**shared, "trt": np.ones(n_obs)},
)
ate = effect.posterior_predictive["mu"].mean("obs").load()  # one value per draw
lower, upper = ate.quantile([0.025, 0.975]).to_numpy()
```

Here the treatment `trt` is a keyword argument of the kernel, and `mu` is a deterministic site that the kernel records inside `plate("obs", ...)`. The tutorial at https://aimz.readthedocs.io/stable/examples/tutorial.html runs a sequence like this on simulated data with a known effect. Real data have no truth to compare an estimate with, so when the result matters, first run the same kernel and scenario code on simulated data in the same way.

## What aimz requires of a kernel

- **Signature.** The kernel takes the input as `X` and the output as `y=None`, or under other names declared with `param_input` and `param_output` when the `ImpactModel` is created. The output is a sample site named after the output argument and observed through it (`obs=y`).
- **Keyword arguments.** Any other data are keyword arguments of the kernel, passed by name in every call: `im.fit_on_batch(X, y, trt=trt)`, `im.predict(X, trt=trt)`. An array whose leading axis matches `X` holds one value per observation and is batched with `X`. Any other array is passed whole. A Python number, boolean, or string is static, and each new value compiles the kernel again, so pass a value that changes between calls as an array.
- **Independent rows.** The kernel computes each row from that row's inputs alone, because `fit` shuffles the rows and trains on batches of them, and the streaming methods (`predict`, `sample_prior_predictive`, `log_likelihood`) split the rows into batches and across devices.
- **Latent sites per observation.** A latent site sampled once per observation makes `predict` and `log_likelihood` warn with a `PerformanceWarning` and rerun under `shard_axis="draw"` when they split the input. Pass `shard_axis="draw"` to go there directly. A kernel built on `scan` is not supported.
- **A site to report.** Record the quantity to report, such as the expected outcome, in a `deterministic` site inside the plate over the observations. It is returned by default with the output, an effect read from it carries no simulation noise, and its observation dimension takes the name of the plate.
- **Inference.** With `MCMC`, fit with `fit_on_batch`. `fit` trains `SVI` on minibatches and raises for `MCMC`.

## Defaults for an agent

Use these unless the task says otherwise.

- Turn the two warning categories that concern correctness into errors before the first aimz call:

  ```python
  import warnings

  import aimz

  warnings.simplefilter("error", aimz.FitWarning)
  warnings.simplefilter("error", aimz.OutputWarning)
  ```

  `FitWarning` reports a loss that is not finite, or a minibatch fit whose kernel weighs each batch as the whole data. `OutputWarning` reports an output that is not what the call asked for: a requested site that is missing, a posterior set without one of the kernel's latent sites, scenarios that cover different rows, draws that are not paired, a baseline from before a refit. `PerformanceWarning` concerns speed only and can stay a warning.
- Pass `store="memory"` to the streaming methods and in the argument dictionaries of `estimate_effect`. By default each call writes its draws to a Zarr store in a temporary directory that the model owns, and reading the result after `im.cleanup()`, or after the model is garbage-collected, raises an error. For a small input, the `*_on_batch` methods and `estimate_effect(..., on_batch=True)` also work in memory; they take neither `store` nor `progress`.
- Pass `progress=False` to `fit`, `fit_on_batch`, and the streaming methods, which otherwise print a progress bar. An `MCMC` object prints its own unless it is created with `progress_bar=False`.
- Pass `return_sites` with the name of the deterministic site to report. By default a call returns the output and every deterministic site, for every draw and observation.
- Pass `rng_key=jax.random.key(seed)` to a call whose result has to be repeatable; for `estimate_effect`, put the same key in both argument dictionaries. Without it each call advances the model's own key, so the same call twice gives different draws of the output. Under the default `shard_axis="obs"`, a streaming method's draws of the output also depend on `batch_size` and on the number of devices, so pass `batch_size` as well, or use `shard_axis="draw"`, which depends on neither. A deterministic site is computed from the posterior draws and the inputs, so none of this changes it.
- Give the inputs as NumPy or JAX arrays with the observations on the leading axis. Trees from the streaming methods are lazy: reduce first, then call `.load()` on the reduced result.

## Where the treatment lives

How a scenario is expressed depends on how the kernel receives the treatment, so find that out first: `im.describe()` lists the kernel's arguments and its sites with their kinds.

| The treatment is | Express a scenario by |
|---|---|
| a sample site of the kernel | `intervention={"<site>": value}` |
| a keyword argument of the kernel | passing another value for that argument |
| part of `X` | passing a modified `X`, repeating any preprocessing done outside the kernel |

`intervention` takes sample sites only. Naming a kernel argument or a deterministic site raises a `ValueError` that lists the kernel's sample sites. Unless the name is misspelled, that error means the treatment belongs to one of the other two rows. A scalar sets the site to one value for every observation, and an array whose leading axis matches `X` sets one value per observation.

A treatment that the kernel does not contain cannot be varied, so no scenario of the fitted model answers a question about it.

## Effects

`estimate_effect` returns intervention minus baseline for every returned site, chain, draw, and observation. The result is in the predictive group of its scenarios: `posterior_predictive`, `predictions` when they were drawn with `in_sample=False`, or `prior_predictive`.

- **Pairing.** Two scenarios given as argument dictionaries are drawn with one key, so their draws differ only through the treatment. Scenarios computed beforehand and passed as `output_baseline` and `output_intervention` are paired only when both come from the same method with the same `rng_key`, `shard_axis`, and `batch_size`. Otherwise the output site carries the noise of both scenarios: its average effect barely moves, while the unit-level effects and their intervals become far too wide. aimz raises an `OutputWarning` for this, and for a baseline computed before a refit. A deterministic site is paired whatever the keys.
- **Summaries.** Reduce over the observations within each draw first, then summarize across the draws: average or sum over the observation dimension, then take the mean and the quantiles over `chain` and `draw`. For a subgroup, select its rows before reducing, as in `.isel(obs=mask)`.
- **Dimension names.** A dimension is named after the plate around the site, such as `obs` for `plate("obs", ...)`. A dimension outside any plate is named `<site>_dim_<i>`, so two sites that share no plate share no dimension, and arithmetic between them broadcasts to one value per pair of rows.
- **Relative effects.** A relative effect is a ratio of the scenarios' per-draw totals, not the mean of unit-level ratios. `estimate_effect` only subtracts, so predict the two scenarios with one `rng_key`, sum each over the observations, and divide.

## Loading a fitted model

- A pickled model loads with `cloudpickle.load`. A kernel defined in a module is stored by reference, so that module has to be importable. A kernel defined in `__main__` is stored with the model.
- A model saved or logged through `aimz.mlflow` loads with `aimz.mlflow.load_model(uri)`, which returns the `ImpactModel`. `mlflow.pyfunc.load_model(uri)` only predicts, and a model saved with a signature drops what the signature does not record, such as `intervention`, so do not build scenarios on it.
- Then `im.describe()` tells whether the model holds posterior draws, and lists its arguments and its sites with their kinds and the names of their dimensions.

## Mistakes that return a number

None of these raises an error or a warning.

| Mistake | What happens | Do instead |
|---|---|---|
| Training `SVI` for too few steps | A narrow interval around a wrong effect. Only a loss that is not finite warns. | Check that `im.vi_result.losses` has stopped falling. Calling `fit` or `fit_on_batch` again continues the training. |
| Calling `fit` or `fit_on_batch` again to start over | The training continues from the state it had reached. | Create a new `ImpactModel`. |
| Computing across rows in the kernel, as in a cumulative sum, a lag, or a mean over the rows | Wrong numbers once the rows are shuffled or split, which depends on the input size, `batch_size`, and the number of devices. | Give each row what it needs as its own input, or keep the rows whole and in order: `fit_on_batch`, the `*_on_batch` methods, `shard_axis="draw"`. |
| Intervening on a site that reaches the outcome only through another latent sample site | The effect is exactly zero, because the site downstream keeps its posterior draws. | Write the kernel so that the path runs through deterministic computations. |

## When not to report an effect

Withhold the number and say why when any of these holds:

- A `FitWarning` or an `OutputWarning` came from the fit or from a scenario and its cause was not removed.
- The fit was not checked. For `SVI`, look at `im.vi_result.losses`; for `MCMC`, at the sampler's own diagnostics, as in `im.inference.print_summary()`.
- The kernel does not contain the treatment, or the scenario sets it to values far from anything in the data the model was fitted on. aimz does not flag extrapolation.
- A run of the same kernel and scenario code on data simulated with a known effect did not recover it.
- The request or the project leaves open the baseline, the intervention, the units, the site, or which input holds the treatment and on what scale, and the choice changes the answer. Ask, or state the choice with the number.

With every number that is reported:

- State the two scenarios, the units, the site, and the summary. The sign is intervention minus baseline.
- Call the interval a posterior interval of this model. A mean-field guide such as `AutoNormal` tends to make it too narrow.
- Say that reading the contrast as causal rests on the assumptions written into the kernel, which aimz does not check.
- When the interval includes zero, say that this model and data did not detect an effect, not that there is none.

Passing these checks does not certify a number. They catch known failures and nothing else.
