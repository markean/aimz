Frequently Asked Questions
==========================


What is a kernel?
-----------------
A kernel in aimz is a user-defined `NumPyro`_ model (a stochastic function or :class:`~collections.abc.Callable`) built with primitives like :external:func:`~numpyro.primitives.sample` and :external:func:`~numpyro.primitives.deterministic`.
Its signature and body define the inputs and output (e.g., ``X``, ``y``, ...), encoding the probabilistic structure: priors, likelihood, and latent variables.


How do I use different argument names than ``X`` and ``y``?
-----------------------------------------------------------
By default, aimz expects your kernel signature to include parameters named ``X`` (input) and ``y`` (output).
If you use different names (e.g. ``features`` / ``target`` or ``covariates`` / ``outcome``), declare them when instantiating :class:`~aimz.ImpactModel`:

.. code-block:: python

    def kernel(features, extra, target=None):
        ...

    im = ImpactModel(
        kernel,
        ...,
        param_input="features",
        param_output="target",
    )

If you see an error like:

``Kernel must accept 'X', 'y' as argument(s). Modify the kernel signature or set `param_input` and `param_output` accordingly.``

it means you neither matched the defaults nor overrode them.
Fix it by renaming your arguments to ``X`` / ``y`` or supplying ``param_input`` / ``param_output`` as shown above.

The methods that take an input keep their own ``X`` and ``y`` parameters whatever the kernel calls them, so pass the input as ``X`` and the output as ``y``, also in the argument dictionaries of :meth:`~aimz.ImpactModel.estimate_effect`, as in ``{"X": X}``; the kernel's own names, such as ``features=X``, raise a ``TypeError``.
Their other parameters, such as ``intervention``, ``condition``, and ``rng_key``, do not reach the kernel either, so rename a kernel argument that shares a name with one; a ``condition`` that is not a mapping raises a ``TypeError`` saying so.
The batches of a data loader, in contrast, are keyed by the kernel's own names (see :doc:`user_guide/dataloader`), and so are the keyword arguments of :meth:`~aimz.ImpactModel.sample`, which it passes to the kernel under MCMC and to the guide under SVI.


.. _faq-kernel-contract:

What does aimz require of a kernel's signature and sites?
---------------------------------------------------------
aimz calls the kernel itself, so it relies on a few rules, each checked with a :class:`~aimz.KernelValidationError` when the model is created or first traces the kernel, except that the output is checked to be an observed sample site only when the kernel is first traced with output data, as in :meth:`~aimz.ImpactModel.fit`:

* The input argument (``X``, or ``param_input``) takes no default, since every call passes it and the batches split along its leading axis.
* The output argument (``y``, or ``param_output``) defaults to ``None``, since aimz predicts by calling the kernel without it.
* The output is a sample site named after the output argument and observed through it, as in ``sample("y", ..., obs=y)``, since aimz finds the output by that name; a likelihood named otherwise, such as ``sample("obs", ..., obs=y)``, needs renaming.
* The signature takes no ``*args``, ``**kwargs``, or positional-only parameters, since aimz passes every argument by name.
* No site name contains ``/``, which :external:class:`~xarray.DataTree` does not allow in variable names, and no sample or deterministic site is named ``chain`` or ``draw``, the dimensions of the draws.

For missing outputs, mask the output's distribution, as in ``sample("y", d.mask(mask), obs=y)``, which leaves the masked observations out of the likelihood; with ``obs_mask``, NumPyro records the output as a deterministic site, which the rules above reject.
A custom likelihood given as a :external:func:`~numpyro.primitives.factor` term named after the output passes as the output, since NumPyro records a ``factor`` term as an observed sample site, but it has nothing to draw, so a kernel that keeps it when the output is ``None`` makes the ``*_on_batch`` methods return the output with an empty last axis, as NumPyro's ``Predictive`` does, while the streaming methods can raise an error instead, as they do with the default store.
Record the output from a generative form in that case, as the spend model of :doc:`examples/hillstrom` does.
A custom guide takes the kernel's arguments under the same names, since aimz passes them to the guide by keyword, as to the kernel: a guide with other names or positional-only parameters, which NumPyro's own ``SVI.run`` accepts by passing them positionally, fails with a ``TypeError`` when aimz fits it.

A kernel without an input matrix, such as the eight schools model, takes another array with one value per row as its input:

.. code-block:: python

    def eight_schools(sigma, y=None):  # sigma: the standard error of each school
        ...

    im = ImpactModel(eight_schools, ..., param_input="sigma")

Names with a ``/`` come from :external:class:`~numpyro.handlers.scope` with its default divider and from NumPyro's module helpers built on it.
Pass ``divider="."`` to ``scope``, or ``scope_divider="."`` to :external:func:`~numpyro.contrib.module.random_nnx_module` or :external:func:`~numpyro.contrib.module.random_eqx_module`; :external:func:`~numpyro.contrib.module.random_flax_module` takes no divider, so wrap the kernel in a handler that renames its sites.


Do I need to know NumPyro_ to use aimz?
---------------------------------------
Yes.
The aimz package builds on `NumPyro`_'s primitives and effect handlers.
You should be comfortable writing a model function, defining a guide (for SVI) or configuring MCMC, and reading model traces.
The library focuses on orchestration, not abstracting away core probabilistic modeling concepts.


.. _faq-model-compatibility:

Can I use aimz with any `NumPyro`_ model?
-----------------------------------------
No.
Most conventional models with global latents and a plate-based structure work out of the box.
The core requirement is that every sample site in the model must have a **static shape** that does not change with the data.
A site sized by a count read from the data, such as the number of groups, breaks this rule, and data with another count make prediction raise the ``ValueError`` described below for local latents; pass the count as a keyword argument with its fitted value.
The forward sampler used internally requires every traced site to have a fixed shape across all sample iterations, and the default observation-sharded execution additionally requires the posterior to be **replicable** across devices.

The most common pattern that violates this requirement is a **local latent variable**.
A :external:func:`~numpyro.primitives.sample` call inside a :external:class:`~numpyro.primitives.plate` without ``obs=`` produces a posterior whose shape grows with the plate size along its second (observation) axis, for example ``(num_samples, n_obs)``, or ``(num_samples, n_obs, k)`` for a multi-output site.
The default observation-sharded execution splits the observations between batches and devices, so such a posterior, which covers all of them, cannot be replicated there.

For **local latent variables**, :meth:`~aimz.ImpactModel.predict` and :meth:`~aimz.ImpactModel.log_likelihood` detect the incompatible posterior shape, warn, and automatically rerun under draw-parallel sharding (``shard_axis="draw"``), which never splits the observation axis.
They tell such a site by the shape of its draws, so a global latent site whose draws happen to be as long as the input's rows reruns the call too; its results stay correct, and passing ``shard_axis="draw"`` skips the check.
:meth:`~aimz.ImpactModel.fit` with a ``batch_size`` smaller than the input warns with a :class:`~aimz.FitWarning` when the guide does not compute such a site from the batch, as when it keeps a value per position or plate index like :external:class:`~numpyro.infer.autoguide.AutoNormal`, since its batches then pair those values with other observations.
A guide that computes the site from each observation's inputs, such as an amortized encoder, trains on batches without the warning.
:meth:`~aimz.ImpactModel.sample_prior_predictive` has no posterior to replicate and needs no rerun: under ``shard_axis="obs"`` it draws local latents within each batch and shares only the global latents across batches.
See :ref:`user-guide-sharding` for how the two sharding strategies differ, when to choose each, and a worked comparison of a compatible kernel and one that triggers the rerun.
A local latent's posterior applies only to the observations the model was fitted on.
For an array input with another number of observations, prediction and :meth:`~aimz.ImpactModel.log_likelihood` raise a ``ValueError`` that names the site, unless the site is intervened on.
The fitted values are reused without notice for an input with the same number of observations, and, under ``shard_axis="obs"``, for one split into device batches with exactly the fitted number of rows, padding included.
A data loader is checked one device batch at a time, so a loader batch that gives each device exactly the fitted number of rows passes too.
A data loader cannot rerun under ``shard_axis="draw"``, so a call with one raises the error for the fitted observations too whenever its batches or the devices split them; pass them as an array instead.
The rerun recognizes a local latent by the second axis of its posterior, so for one whose observations lie on another axis, as in nested plates, the call raises the error for the fitted observations too whenever its batches or the devices split them, unless ``shard_axis="draw"`` is passed.
To predict other observations, draw the site from the prior: set a posterior without it, or, if it is the only latent site, use :meth:`~aimz.ImpactModel.sample_prior_predictive`, which runs ``param`` sites at their initial values.
If it is the only latent site of a kernel with learned ``param`` sites, such as a network, pass draws of it for the new observations to :meth:`~aimz.ImpactModel.set_posterior_sample` instead, drawn from its prior or, for a guide that computes the site from the inputs, by :meth:`~aimz.ImpactModel.sample` with the new inputs; the learned values stay in use.
A latent site inside the body of :external:func:`~numpyro.contrib.control_flow.scan` follows NumPyro's scan instead, intervened on or not: an input with fewer rows than the fitted ones raises NumPyro's ``RuntimeError``, and one with more rows takes the fitted values for the leading rows and draws the rest.

A model with nested plates whose :external:func:`~numpyro.primitives.sample` sites are all observed (``obs=``) or whose latent shapes are fixed remains compatible; a model with a single plate containing one unobserved :external:func:`~numpyro.primitives.sample` site triggers the draw-parallel rerun.

A kernel must also compute each observation from its own inputs (see :ref:`faq-cross-observation`).

If you encounter an unsupported pattern (ideally with a minimal reproducible example), please `open an issue <https://github.com/markean/aimz/issues/new>`_ or submit a PR.
We plan to broaden coverage based on user needs.


Does aimz ship built-in model templates?
----------------------------------------
No.
This is intentional to keep the library lightweight and avoid prescribing a specific modeling style.
Future recipes or example galleries may be provided separately, but the library itself does not include canonical model classes.


What kinds of data can aimz handle?
-----------------------------------
aimz accepts NumPy or JAX arrays of any shape with at least one dimension; the leading axis is treated as the observation axis.
This covers tabular inputs (``(n, d)``), 1D inputs (``(n,)``), and higher-rank inputs such as sequences (``(n, seq_len, d)``) or images (``(n, h, w, c)``).
Additional named arrays whose leading axis matches the input are treated as per-observation data, and other arrays, such as a vector of prior scales, are passed whole (see :ref:`streaming-keyword-arguments`).
The output variable has the same flexibility: it can be 1D for scalar targets, 2D for multi-output regression, or higher-rank as the model requires, provided its leading axis matches the input.
Ragged or nested structures are not currently supported as the input or the output; a keyword argument that holds one is passed whole, except a list or tuple with one entry per observation, which is batched with the input when NumPy can stack it into an array and otherwise raises a ``ValueError``.
If native support for a specific structure is important for your use case, opening an issue helps prioritize it, and contributions are welcome.


.. _faq-cross-observation:

Can my kernel compute across observations?
------------------------------------------
Only when the observation axis stays whole and in order.
aimz treats the leading axis as the observation axis and assumes observations are independent given the model parameters, so it may split and reorder that axis to scale.
:meth:`~aimz.ImpactModel.fit` trains on batches of the rows and, when ``batch_size`` splits them, shuffles them by default.
The streaming methods (:meth:`~aimz.ImpactModel.predict`, :meth:`~aimz.ImpactModel.sample_posterior_predictive`, :meth:`~aimz.ImpactModel.sample_prior_predictive`, and :meth:`~aimz.ImpactModel.log_likelihood`) process the input in batches and, under the default ``shard_axis="obs"``, split each batch across devices.
A kernel that computes across observations, for example through a cumulative sum, a moving average, a value from a previous row, or a statistic such as the mean over all rows, then sees only part of the rows or sees them out of order.
This happens without an error or a warning, unless the kernel's trace shows the dependence: a site recorded by :external:func:`~numpyro.contrib.control_flow.scan`, or a site drawn as one event over all the rows, such as a :external:class:`~numpyro.distributions.continuous.MultivariateNormal` over the observations.
For such a kernel and an array input, the streaming methods warn with a :class:`~aimz.PerformanceWarning` and rerun under ``shard_axis="draw"``, which keeps the rows whole and in order, while a data loader still splits them, with an :class:`~aimz.OutputWarning`, and so does :meth:`~aimz.ImpactModel.fit` with a ``batch_size`` smaller than the input, which warns with a :class:`~aimz.FitWarning`.
The check needs a trace of the kernel, which prior predictive sampling and the first call with an array input run, so on a model whose posterior was set with :meth:`~aimz.ImpactModel.set_posterior_sample`, a data loader given to the other streaming methods before either splits such a kernel without the warning.
:meth:`~aimz.ImpactModel.fit` and the streaming methods count only an event exactly as long as the rows they trace, the batch for :meth:`~aimz.ImpactModel.fit`, so an event over the rows that is padded beyond them, as to a multiple of eight when their number is not a multiple of eight, or offset from them, as by an anchor point, gets neither the warning of :meth:`~aimz.ImpactModel.fit` nor the rerun of the streaming methods.
A global event that only happens to be as long as those rows, such as a weight vector with as many entries, counts neither, since a second trace on more rows shows that its length does not follow them.
A site recorded by :external:func:`~numpyro.contrib.control_flow.cond` counts as one recorded by ``scan``, so a kernel with such a site reruns and warns the same way, even when it computes each row from that row's inputs alone.
Whether a split happens depends on the input size, ``batch_size``, and the number of devices, so such a kernel can give correct results on a small input on one device and wrong results on a larger input or on several devices.
A plate inside the body of :external:func:`~numpyro.contrib.control_flow.scan` whose ``subsample_size`` exceeds the rows it receives, as when the kernel gathers rows through its indices and :meth:`~aimz.ImpactModel.predict` or :meth:`~aimz.ImpactModel.sample_prior_predictive` splits the input, takes the rows in a random order that differs between draws, so give it no ``subsample_size`` at prediction or pass ``shard_axis="draw"``.

There are two ways to make such a kernel safe:

* **Make each observation self-contained**

  Pass what each observation needs from the rest of the input as part of its own inputs, for example a precomputed statistic repeated for every row, or ``L`` related rows gathered along an extra axis to give shape ``(n, L, d)``.
  Observations are then independent, so every method and execution mode gives the same result.

* **Keep the observation axis whole and in order**

  Train with :meth:`~aimz.ImpactModel.fit_on_batch`, or with :meth:`~aimz.ImpactModel.fit` and a ``batch_size`` that covers the whole input, as the default does.
  For the other methods, use the ``*_on_batch`` variants or pass ``shard_axis="draw"`` to the streaming methods, which then split the draws instead of the observations and require an array input.


.. _faq-forecasting:

How do I forecast from the observed history?
--------------------------------------------
The predictive methods call the kernel without the output, so a kernel whose recursion reads the output, as an autoregressive :external:func:`~numpyro.contrib.control_flow.scan` does, runs free at prediction: each step follows the steps it drew, not the observed ones.
The output cannot be passed instead, since it is a reserved parameter, and an intervention on it does not condition the recursion either.
To predict each step from the observed history, pass the history as ``condition`` to :meth:`~aimz.ImpactModel.predict` or :meth:`~aimz.ImpactModel.predict_on_batch`, also in the argument dictionaries of :meth:`~aimz.ImpactModel.estimate_effect`, as in ``im.predict(X, condition={"y": y_observed})``: the output takes those values while sampling, so the recursion reads them, also inside the body of ``scan``, and the output comes back with them.
Inside ``scan``, a history shorter than ``X`` conditions only the leading steps, and the steps beyond it are drawn, each from the steps before it, which forecasts past the history; a longer one raises NumPyro's ``RuntimeError``.
``condition`` takes any sample site, but a site whose values the posterior holds keeps its posterior draws, with an :class:`~aimz.OutputWarning`, so set a latent site through ``intervention`` instead.
Its values are passed whole: under ``shard_axis="obs"``, a value with one entry per row of ``X`` makes the call run under ``shard_axis="draw"``, with a :class:`~aimz.PerformanceWarning` when ``"obs"`` would split the rows, as a ``scan`` kernel does anyway, and a data loader passes such a value to each of its batches, which it does not fit.
The history can also be passed as data: as a keyword argument that the recursion reads in place of the output, or as lag features in ``X``, such as the previous value of the output in a column.


Can I use my own data loader?
-----------------------------
Yes.
The streaming methods accept any finite iterable that yields batches as mappings from kernel parameter names to NumPy or JAX arrays, such as a generator wrapping a PyTorch ``DataLoader``.
Batches may differ in size, no length is required, and padding for sharding and device placement are handled by aimz.
:meth:`~aimz.ImpactModel.fit` accepts such a loader too and iterates it once per epoch, so a loader for several epochs has to start a new pass each time it is iterated.
See :doc:`user_guide/dataloader` for the batch contract and examples.


Can I use aimz for general-purpose Bayesian inference?
------------------------------------------------------
Yes.
aimz is a flexible, object-oriented interface to `NumPyro`_ and supports a wide range of Bayesian modeling tasks (regression, classification, uncertainty quantification, and predictive simulation), even if your application doesn't involve interventions or causal analysis.


Can I use posterior samples generated elsewhere?
------------------------------------------------
Yes, you do not need to train a model from scratch and sample posteriors.
After initializing an :class:`~aimz.ImpactModel` with your model, call :meth:`~aimz.ImpactModel.set_posterior_sample` with a dictionary mapping site names to arrays.
Each array must share the same leading dimension (number of draws), and the dictionary must not be empty.
For draws from several chains, pass ``num_chains`` as well, so the output trees keep the chains.
Once injected, the model is treated as fitted, and the prediction, log-likelihood, and posterior predictive methods will use the supplied samples.
For :meth:`~aimz.ImpactModel.log_likelihood`, the samples must cover every latent site of the kernel (a partial posterior raises an error), whereas the predictive methods draw any missing sites fresh, which :meth:`~aimz.ImpactModel.set_posterior_sample` warns about once the kernel has been traced.


When should I use the ``*_on_batch`` variants?
----------------------------------------------
Use the batch-specific variants only when you need explicit, single-batch control (e.g., custom training loops, micro-benchmarking, or integrating with external schedulers).
The higher-level methods handle internal batching, iteration, shuffling, streaming, and aggregation automatically and are preferred for typical workflows.
See :doc:`user_guide/streaming_and_on_batch` for a detailed comparison of both approaches and guidance on when to use each.


How do I control which variables (sites) are sampled?
-----------------------------------------------------
By default, prediction and sampling methods use the set of return sites cached in :attr:`~aimz.model.KernelSpec.return_sites`, typically the model output plus any deterministic sites discovered during the first trace.
To override this behavior, pass ``return_sites=(...)`` explicitly to the relevant methods.


How to ensure reproducible results?
-----------------------------------
:class:`~aimz.ImpactModel` requires an explicit JAX pseudo-random number generator key for initialization.
Using the same initial key ensures that all subsequent stochastic operations are reproducible.
Stochastic methods accept an optional ``rng_key`` for per-call determinism.
If provided, it affects only that call and does not modify the model's internal key.
If omitted, a new subkey is derived internally, so repeated calls may produce different results.
Under the default ``shard_axis="obs"``, the streaming methods draw each batch with its own key, so their results also depend on ``batch_size`` and the number of devices; under ``shard_axis="draw"`` they depend on neither.
When ``batch_size`` is not given, the automatic choice can also depend on the machine's CPU count, so pass ``batch_size`` explicitly to reproduce results across machines.
A predictive output records the key and the batching that its draws depend on in the attributes of its group (see :ref:`output-attributes`).
To fully reproduce results, log the initial seed along with other artifacts.


Why do some methods return :class:`~xarray.DataTree`?
-----------------------------------------------------
A :class:`~xarray.DataTree` organizes heterogeneous groups (``posterior``, ``posterior_predictive``, ``predictions``) with labeled dimensions and coordinates, facilitating I/O, slicing, and downstream analysis.
It can also be passed to `ArviZ`_ functions.
The dimensions of a site are named after the plates around it, for example ``obs`` for a site inside ``plate("obs", ...)``.
Its other dimensions are named ``<site>_dim_<i>``, numbered from zero among them, so a site of shape ``(obs, 3)`` has the dimensions ``obs`` and ``<site>_dim_0``.
To name such a dimension of a sample site, list it in the site's ``infer={"event_dims": [...]}``; a deterministic site takes no ``infer``, so its other dimensions keep the default names.
Names that cannot apply, such as a plate named ``chain`` or ``draw``, a name shared with a site, more names than the site has dimensions, or one name with two lengths in a group, give way to the default names with an :class:`~aimz.OutputWarning`.
A site inside the body of :external:func:`~numpyro.contrib.control_flow.scan` keeps the default names, and event dimension names listed in its ``infer`` give way with an :class:`~aimz.OutputWarning`.
A deterministic site there keeps them only in a kernel with a sample site inside the body of a ``scan``; otherwise it is named like any other site, so when the number of steps equals the size of every plate around it, the plate names apply from its leading time axis on, each one axis early, and otherwise it keeps the default names.
In a kernel with such a sample site, a deterministic site inside :external:func:`~numpyro.contrib.control_flow.cond` also keeps the default names.
A default name that is itself a site of the group skips to the next free number, also with an :class:`~aimz.OutputWarning`.
The plate names come from a trace of the kernel, which fitting and prior predictive sampling run, as does the first call with an array input on a model never traced before, so a posterior set with :meth:`~aimz.ImpactModel.set_posterior_sample` on a new model keeps the default names only in calls with a data loader made before then.
Trees from the streaming methods are lazy, and ArviZ functions operate on loaded arrays, so call :external:meth:`~xarray.DataTree.load` on the tree first, or on a reduced result when the tree does not fit in memory.
If desired, you can pass ``return_datatree=False`` to methods such as :meth:`~aimz.ImpactModel.predict_on_batch` to return a plain dictionary instead.


Why do I not see a ``posterior`` group in the output?
-----------------------------------------------------
It appears in the returned :class:`~xarray.DataTree` only if posterior samples are available (fitted or injected).


Where is the persistent output written?
---------------------------------------
Currently, persistent-store outputs (the streaming methods' default) are written under the directory passed via ``output_dir``.
If ``output_dir=None``, a temporary directory is created (accessible via :attr:`~aimz.ImpactModel.temp_dir`) and removed when the model is cleaned up (either explicitly with :meth:`~aimz.ImpactModel.cleanup` or when the instance is garbage collected).
The returned :class:`~xarray.DataTree` records the call's artifact path in an ``artifact_path`` attribute on both the root tree and the group node; its parent is the ``output_dir`` (or temporary root) it was written under.
Calls made with ``store="memory"`` write nothing and return a Dask-backed tree whose chunks live in host memory, with no ``artifact_path`` attribute, and leave :attr:`~aimz.ImpactModel.temp_dir` untouched.


Does serialization persist the posterior samples?
-------------------------------------------------
Yes.
Pickling (or MLflow integration via :mod:`aimz.mlflow`) preserves the posterior samples (if set) and the cached :class:`~aimz.model.KernelSpec` so retracing / re-fitting is unnecessary upon load.
See :doc:`user_guide/model_persistence` or :doc:`user_guide/mlflow` for more details.
