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

``Kernel must accept 'X' and 'y' as argument(s). Modify the kernel signature or set `param_input` and `param_output` accordingly.``

it means you neither matched the defaults nor overrode them.
Fix it by renaming your arguments to ``X`` / ``y`` or supplying ``param_input`` / ``param_output`` as shown above.


Do I need to know NumPyro_ to use aimz?
---------------------------------------
Yes.
The aimz package builds on `NumPyro`_’s primitives and effect handlers.
You should be comfortable writing a model function, defining a guide (for SVI) or configuring MCMC, and reading model traces.
The library focuses on orchestration, not abstracting away core probabilistic modeling concepts.


.. _faq-model-compatibility:

Can I use aimz with any `NumPyro`_ model?
-----------------------------------------
No.
Most conventional models with global latents and a plate-based structure work out of the box.
The core requirement is that every sample site in the model must have a **static shape** that does not change with the data.
The forward sampler used internally requires every traced site to have a fixed shape across all sample iterations, and the default observation-sharded execution additionally requires the posterior to be **replicable** across devices.

Several modeling patterns can violate this requirement.
The two most common are:

* **Local latent variables**

  A :external:func:`~numpyro.primitives.sample` call inside a :external:class:`~numpyro.primitives.plate` without ``obs=`` produces a posterior whose shape grows with the plate size along its second (observation) axis, for example ``(num_samples, n_obs)``, or ``(num_samples, n_obs, k)`` for a multi-output site.
  Because ``n_obs`` can differ between training and prediction, such a posterior cannot be replicated under the default observation-sharded execution.

* **The** :external:func:`~numpyro.contrib.control_flow.scan` **primitive**

  :external:func:`~numpyro.contrib.control_flow.scan` is commonly used for sequential or autoregressive models (e.g., state-space models, time-series forecasting).
  The number of sites it creates typically grows with the sequence length, making the trace shape dynamic for the same reason.

For **local latent variables**, :meth:`~aimz.ImpactModel.predict` and :meth:`~aimz.ImpactModel.log_likelihood` detect the incompatible posterior shape, warn, and automatically rerun under draw-parallel sharding (``shard_axis="draw"``), which never splits the observation axis.
See :ref:`user-guide-sharding` for how the two sharding strategies differ, when to choose each, and a worked comparison of a compatible kernel and one that triggers the rerun.
Draw-parallel sharding does not lift the static-shape requirement, so :external:func:`~numpyro.contrib.control_flow.scan`-based models remain unsupported, and a posterior that is fundamentally incompatible with the new input still fails with a shape mismatch.

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
Multiple named arrays are supported as long as they share the same leading-axis size.
The output variable has the same flexibility: it can be 1D for scalar targets, 2D for multi-output regression, or higher-rank as the model requires, provided its leading axis matches the input.
Ragged or nested structures are not currently supported.
If native support for a specific structure is important for your use case, opening an issue helps prioritize it, and contributions are welcome.


.. _faq-cross-observation:

Can my kernel compute across observations?
------------------------------------------
Only when the observation axis stays whole and in order.
aimz treats the leading axis as the observation axis and assumes observations are independent given the model parameters, so it may split and reorder that axis to scale.
:meth:`~aimz.ImpactModel.fit` shuffles the rows by default and trains on batches of them.
The streaming methods (:meth:`~aimz.ImpactModel.predict`, :meth:`~aimz.ImpactModel.sample_posterior_predictive`, :meth:`~aimz.ImpactModel.sample_prior_predictive`, and :meth:`~aimz.ImpactModel.log_likelihood`) process the input in batches and, under the default ``shard_axis="obs"``, split each batch across devices.
A kernel that computes across observations, for example through a cumulative sum, a moving average, a value from a previous row, or a statistic such as the mean over all rows, then sees only part of the rows or sees them out of order.
This happens without an error or a warning.
Whether a split happens depends on the input size, ``batch_size``, and the number of devices, so such a kernel can give correct results on a small input on one device and wrong results on a larger input or on several devices.

There are two ways to make such a kernel safe:

* **Make each observation self-contained**

  Pass what each observation needs from the rest of the input as part of its own inputs, for example a precomputed statistic repeated for every row, or ``L`` related rows gathered along an extra axis to give shape ``(n, L, d)``.
  Observations are then independent, so every method and execution mode gives the same result.

* **Keep the observation axis whole and in order**

  Train with :meth:`~aimz.ImpactModel.fit_on_batch`, or with :meth:`~aimz.ImpactModel.fit` using ``shuffle=False`` and a ``batch_size`` that covers the whole input.
  For the other methods, use the ``*_on_batch`` variants or pass ``shard_axis="draw"`` to the streaming methods, which then split the draws instead of the observations and require an array input.


Can I use my own data loader?
-----------------------------
Yes.
The streaming methods accept any finite iterable that yields batches as mappings from kernel parameter names to NumPy or JAX arrays, such as a generator wrapping a PyTorch ``DataLoader``.
Batches may differ in size, no length is required, and padding for sharding and device placement are handled by aimz.
:meth:`~aimz.ImpactModel.fit` accepts arrays or the built-in :class:`~aimz.utils.data.ArrayLoader`; for other loaders, write a training loop with :meth:`~aimz.ImpactModel.train_on_batch`.
See :doc:`user_guide/dataloader` for the batch contract and examples.


Can I use aimz for general-purpose Bayesian inference?
------------------------------------------------------
Yes.
aimz is a flexible, object-oriented interface to `NumPyro`_ and supports a wide range of Bayesian modeling tasks (regression, classification, uncertainty quantification, and predictive simulation), even if your application doesn’t involve interventions or causal analysis.


Can I use posterior samples generated elsewhere?
------------------------------------------------
Yes, you do not need to train a model from scratch and sample posteriors.
After initializing an :class:`~aimz.ImpactModel` with your model, call :meth:`~aimz.ImpactModel.set_posterior_sample` with a dictionary mapping site names to arrays.
Each array must share the same leading dimension (number of draws), and the dictionary must not be empty.
Once injected, the model is treated as fitted, and the prediction, log-likelihood, and posterior predictive methods will use the supplied samples.
For :meth:`~aimz.ImpactModel.log_likelihood`, the samples must cover every latent site of the kernel (a partial posterior raises an error), whereas the predictive methods draw any missing sites fresh.


When should I use the ``*_on_batch`` variants?
----------------------------------------------
Use the batch-specific variants only when you need explicit, single-batch control (e.g., custom training loops, micro‑benchmarking, or integrating with external schedulers).
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
If provided, it affects only that call and does not modify the model’s internal key.
If omitted, a new subkey is derived internally, so repeated calls may produce different results.
To fully reproduce results, log the initial seed along with other artifacts.


Why do some methods return :class:`~xarray.DataTree`?
-----------------------------------------------------
A :class:`~xarray.DataTree` organizes heterogeneous groups (``posterior``, ``posterior_predictive``, ``predictions``) with labeled dimensions and coordinates, facilitating I/O, slicing, and downstream analysis.
It can also be passed directly to `ArviZ`_ functions.
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
