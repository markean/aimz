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

- **Signature.** The kernel takes the input as `X` and the output as `y=None`, or under other names declared with `param_input` and `param_output` when the `ImpactModel` is created. The methods that take an input, and the argument dictionaries of `estimate_effect`, still take the input as `X` and the output as `y`, while data loader batches and the keyword arguments of `sample` use the kernel's own names. The output is a sample site named after the output argument and observed through it (`obs=y`); for missing outputs, mask its distribution, `sample("y", d.mask(mask), obs=y)`, since `obs_mask` turns it into a deterministic site. A `factor` term named after the output passes too, but has nothing to draw, so when `y` is None, record the output from a generative form: otherwise the `*_on_batch` methods return it with an empty last axis, and the streaming methods can raise an error, as they do with the default store. The input takes no default, and the signature has no `*args`, `**kwargs`, or positional-only parameters, since aimz passes every argument by name. A kernel without covariates takes another array with one value per observation as its input, through `param_input`. No site name contains `/`: pass `divider="."` to `handlers.scope` or `scope_divider="."` to `random_nnx_module`, and wrap a kernel that uses `random_flax_module` in a handler that renames its sites.
- **Keyword arguments.** Any other data are keyword arguments of the kernel, passed by name in every call: `im.fit_on_batch(X, y, trt=trt)`, `im.predict(X, trt=trt)`. A method's parameters other than `X` and `y`, such as `intervention`, `condition`, and `rng_key`, never reach the kernel, so do not give the kernel an argument with one of those names. An array whose leading axis matches `X` holds one value per observation and is batched with `X`. Any other array is passed whole, and so is a dictionary, so give each per-observation array its own keyword argument, and keep an array that only happens to be as long as `X` whole by binding it to the kernel with `functools.partial` and passing the bound kernel to the `ImpactModel`, its inference object, and any autoguide, or by wrapping it in a dictionary. A list or tuple with an entry per observation is batched as the array NumPy stacks it into, and raises a `ValueError` when NumPy cannot stack it, so pad its entries into an array, or wrap it in a dictionary if its length matches `X` only by chance; any other list or tuple is passed whole. A Python number, boolean, or string is static, and each new value compiles the kernel again, so pass a value that changes between calls as an array. Pass a count that sizes a site, such as the number of groups, as a Python int, with its fitted value at prediction: a posterior site of another size makes prediction raise a `ValueError` naming it.
- **Independent observations.** The kernel computes each observation from that observation's inputs alone, because `fit` with a `batch_size` below the number of observations trains on shuffled batches of them, and the streaming methods (`predict`, `sample_prior_predictive`, `log_likelihood`) split the observations into batches and across devices. A kernel with a site recorded by `scan`, or by `cond`, which aimz counts the same way, makes the streaming methods warn with a `PerformanceWarning` and rerun under `shard_axis="draw"` for an array input, which keeps the observations whole and in order; a data loader cannot rerun, so with one they warn with an `OutputWarning` instead, once a call has traced the kernel: after `set_posterior_sample` on a new model, make one call with an array input first.
- **Latent sites per observation.** A latent site sampled once per observation, or with another size that follows the number of observations, such as a random walk's increments in a plate of one observation fewer, makes `predict` and `log_likelihood` warn with a `PerformanceWarning` and rerun under `shard_axis="draw"` when they split an array input. Pass `shard_axis="draw"` to go there directly. `fit` with a `batch_size` below the number of observations warns with a `FitWarning` for such a site in the output's plate when the guide does not compute it from the batch, as when it keeps a value per position or plate index, such as `AutoNormal`, since its batches then pair those values with other observations; a guide that computes the site from each observation's inputs, or does not sample it, trains on batches without the warning, and so does a site in another plate, such as a random walk's increments, so fit one with `fit_on_batch` or a `batch_size` that covers the data. Its draws belong to the observations the model was fitted on, so do not predict other observations with them: an array input with another number of observations makes the predictive methods and `log_likelihood` raise a `ValueError` naming the site, unless it is named in `intervention`, but one with the same number silently reuses the fitted values, as does a data loader whose batches give each device the fitted number of observations. A data loader cannot rerun under `shard_axis="draw"`, so a call with one raises for the fitted observations too whenever its batches or the devices split them; pass them as an array. To predict other observations, draw the site from its prior: set a posterior without it, or, if it is the only latent site, use `sample_prior_predictive`, which runs `param` sites at their initial values. If it is the only latent site and the kernel has learned `param` sites, such as a network, pass its draws for the new observations to `set_posterior_sample` instead, or, for a guide that computes the site from the inputs, the draws of `sample` on the new inputs. A latent site inside the body of `scan` follows NumPyro's `scan` instead, intervened on or not: an input with fewer observations than the fitted ones raises a `RuntimeError`, and one with more takes the fitted values for the leading observations and draws the rest.
- **A site to report.** Record the quantity to report, such as the expected outcome, in a `deterministic` site inside the plate over the observations. It is returned by default with the output, an effect read from it carries no simulation noise, and its observation dimension takes the name of the plate.
- **Inference.** With `MCMC`, fit with `fit_on_batch`. `fit` trains `SVI` on minibatches and raises for `MCMC`. For `MCMC.run` options such as `extra_fields`, or another inference method such as `SteinVI`, run it natively and pass its draws to `set_posterior_sample`, with `num_chains` set to the chains or particles. A custom `SVI` guide takes the kernel's arguments under the same names, since aimz passes them by keyword; other names or positional-only parameters fail with a `TypeError`.

## Defaults for an agent

Use these unless the task says otherwise.

- Turn the two warning categories that concern correctness into errors before the first aimz call:

  ```python
  import warnings

  import aimz

  warnings.simplefilter("error", aimz.FitWarning)
  warnings.simplefilter("error", aimz.OutputWarning)
  ```

  `FitWarning` reports a loss that is not finite, or a minibatch fit whose kernel does not scale the output of a batch by the number of observations over `batch_size` (a scale given per observation, as weights are, counts only when it is one on every observation), as a plate with `size` the number of observations and `subsample_size=X.shape[0]` does, while a site outside that plate, such as a global latent, a per-observation latent drawn outside it, or a penalty passed to `factor`, weighs against the batch, or whose batches split a site computed across observations (an event over the observations counts only when it is as long as the batch) or a latent site per observation that the guide does not compute from the batch. `OutputWarning` reports an output that is not what the call asked for: a requested site that is missing or intervened on, an output site without one value per observation of the input, dimension names that cannot apply, a posterior set without one of the kernel's latent sites, an intervention that reaches the output only through posterior draws, a data loader whose batches split a kernel that computes across observations, a conditioned site that keeps its posterior draws, scenarios that cover different observations, draws that are not paired, a baseline from before a refit. `PerformanceWarning` concerns speed or memory and can stay a warning.
- Pass `store="memory"` to the streaming methods and in the argument dictionaries of `estimate_effect`. By default each call writes its draws to a Zarr store in a temporary directory that the model owns, and reading the result after `im.cleanup()`, or after the model is garbage-collected, raises an error. For a small input, the `*_on_batch` methods and `estimate_effect(..., on_batch=True)` also work in memory; they take neither `store` nor `progress`.
- Pass `progress=False` to `fit`, `fit_on_batch`, and the streaming methods, which otherwise print a progress bar. An `MCMC` object prints its own unless it is created with `progress_bar=False`.
- Pass `return_sites` with the name of the deterministic site to report. By default a call returns the output and every deterministic site, for every draw and observation.
- Pass `rng_key=jax.random.key(seed)` to a call whose result has to be repeatable; for `estimate_effect`, put the same key in both argument dictionaries. Without it each call advances the model's own key, so the same call twice gives different draws of the output. Under the default `shard_axis="obs"`, a streaming method's draws of the output also depend on `batch_size` and on the number of devices, so pass `batch_size` as well, or use `shard_axis="draw"`, which depends on neither. A deterministic site is computed from the posterior draws and the inputs, so none of this changes it.
- Give the inputs as NumPy or JAX arrays with the observations on the leading axis. Trees from the streaming methods are lazy: reduce first, then call `.load()` on the reduced result.

## Where the treatment lives

How a scenario is expressed depends on how the kernel receives the treatment, so find that out first: `im.describe()` lists the kernel's arguments and its sites with their kinds, though not its `param` sites, whose values after an `SVI` fit are in `im.vi_result.params`, and a `factor` term shows as observed.

| The treatment is | Express a scenario by |
|---|---|
| a sample site of the kernel | `intervention={"<site>": value}` |
| a keyword argument of the kernel | passing another value for that argument |
| part of `X` | passing a modified `X`, repeating any preprocessing done outside the kernel |

`intervention` takes sample sites only. Naming a kernel argument or a deterministic site raises a `ValueError` that lists the kernel's sample sites. Unless the name is misspelled, that error means the treatment belongs to one of the other two rows. A site that the kernel already sets with its own `handlers.do` cannot be named either: two `do` handlers on one site recurse in NumPyro, so the call raises a `RecursionError`; remove the kernel's `do` or pass its value as a keyword argument. A scalar sets the site to one value for every observation, and an array whose leading axis matches `X` sets one value per observation. A value is shared by every draw, so to change one component of a vector site and keep the draws of the others, pass the change as a keyword argument that the kernel applies to that component. A site named in both `intervention` and `return_sites` comes back with its own draws, not the set value: under the posterior predictive, for a site the posterior holds, its posterior draws, with the fitted number of observations for a site with one value per observation, and otherwise the data passed for the site or fresh draws from its distribution. The call warns with an `OutputWarning`, and its effect in `estimate_effect` is zero or noise, so leave it out of `return_sites` and read the effect from the sites downstream of it.

A latent sample site downstream of the intervened one keeps its posterior draws and does not pass the intervention on, so an intervention that reaches the output only through such sites changes nothing; `predict` and `predict_on_batch`, and so `estimate_effect`, warn with an `OutputWarning`. Write the kernel so that the path runs through deterministic computations instead. A keyword argument or a part of `X` that reaches the output only through such a site, and an intervention whose only path runs through an observed site whose data are passed, change nothing either, with no warning: the effect is exactly zero.

A sample site inside the body of `scan` is not intervened on at all, since NumPyro's `do` does not reach it, and no warning says so. On a `scan` kernel the `OutputWarning` above does not fire either when the path to the output runs through a latent site inside the body. To vary a treatment inside the body, pass it as a keyword argument of the kernel and add it to the output through deterministic computations; read only into the distribution of a latent site there, it changes nothing either.

A treatment that the kernel does not contain cannot be varied, so no scenario of the fitted model answers a question about it.

## Effects

`estimate_effect` returns intervention minus baseline for every returned site, chain, draw, and observation. The result is in the predictive group of its scenarios: `posterior_predictive`, `predictions` when they were drawn with `in_sample=False`, or `prior_predictive`.

- **Pairing.** Two scenarios given as argument dictionaries are drawn with one key, so their draws differ only through the treatment. Scenarios computed beforehand and passed as `output_baseline` and `output_intervention` are paired only when both come from the same method with the same `rng_key`, `shard_axis`, and `batch_size`. Otherwise the output site carries the noise of both scenarios: its average effect barely moves, while the unit-level effects and their intervals become far too wide. aimz raises an `OutputWarning` for this, and for a baseline computed before a refit. A deterministic site is paired whatever the keys.
- **Summaries.** Reduce over the observations within each draw first, then summarize across the draws: average or sum over the observation dimension, then take the mean and the quantiles over `chain` and `draw`. For a subgroup, select its observations before reducing, as in `.isel(obs=mask)`.
- **Dimension names.** A dimension is named after the plate around the site, such as `obs` for `plate("obs", ...)`. A dimension outside any plate is named `<site>_dim_<i>`, so two sites that share no plate share no dimension, and arithmetic between them broadcasts to one value per pair of entries. In a kernel with a sample site inside the body of `scan`, every site that `scan` records keeps the default names, and so does a deterministic site inside `cond`. In a kernel whose `scan` bodies record deterministic sites only, those sites do not carry the plate name on the plate axis: when the number of steps equals the plate size, their leading time axis takes it instead, so do not align them with other sites by name.
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
| Refitting with a keyword argument that adds a latent site, such as a switch that turns on an intercept | Under `SVI`, the training continues from a state without the site, so the posterior lacks it and the predictive methods draw it from its prior; only NumPyro's "Found vars in model but not guide" warning hints at it. | Create a new `ImpactModel` with a new `SVI` object and guide, since an autoguide keeps the sites of its first call. |
| Computing across observations in the kernel, as in a cumulative sum, a lag, or a mean over the observations | Wrong numbers once the observations are shuffled or split, which depends on the input size, `batch_size`, and the number of devices. | Give each observation what it needs as its own input, or keep the observations whole and in order: `fit_on_batch`, the `*_on_batch` methods, `shard_axis="draw"`. |
| Forecasting with a kernel whose recursion reads the output | The recursion runs free at prediction, since the output cannot be passed; an intervention on it does not condition either. | Pass the observed history as `condition={"y": y_observed}` to `predict` or `predict_on_batch`; inside `scan`, a history shorter than `X` conditions the leading steps and forecasts the rest. |
| Reading an enumerated discrete site as each observation's class or imputed value, such as a mixture assignment or a missing discrete covariate, even with the output conditioned | The site comes from its prior given each draw, as from NumPyro's `Predictive` without `infer_discrete`. | Pass the site's draws from `Predictive(..., infer_discrete=True)` to `set_posterior_sample` with the other posterior sites. |
| Intervening on a sample site inside the body of `scan` | A zero effect: the intervention changes no draw. | Pass the treatment as a keyword argument of the kernel and add it to the output through deterministic computations in the body. |

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
