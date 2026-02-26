.. _NumPyro: https://num.pyro.ai/

Frequently Asked Questions
==========================


What is a kernel?
-----------------
A kernel in aimz is a user-defined NumPyro_ model (a stochastic function or :class:`~collections.abc.Callable`) built with primitives like :external:func:`~numpyro.primitives.sample` and :external:func:`~numpyro.primitives.deterministic`.
Its signature and body define the inputs and output (e.g., ``X``, ``y``, ...), encoding the probabilistic structure—priors, likelihood, and latent variables.


How do I use different argument names than ``X`` and ``y``?
-----------------------------------------------------------
By default, aimz expects your kernel signature to include parameters named ``X`` (input) and ``y`` (output).
If you use different names—e.g. ``features`` / ``target`` or ``covariates`` / ``outcome``—declare them when instantiating :class:`~aimz.ImpactModel`:

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
The aimz package builds on NumPyro_’s primitives and effect handlers.
You should be comfortable writing a model function, defining a guide (for SVI) or configuring MCMC, and reading model traces.
The library focuses on orchestration, not abstracting away core probabilistic modeling concepts.


Can I use aimz with any NumPyro_ model?
---------------------------------------
No.
Most conventional models with global latents and a plate-based structure work out of the box.
The core requirement is that every sample site in the model must have a **static shape** that does not change with the data.

On multi-device systems, :meth:`~aimz.ImpactModel.predict` uses `data parallelism <https://jax.readthedocs.io/en/latest/distributed_data_loading.html#data-parallelism>`_: the input data is sharded across devices along axis 0 (the observation dimension), while posterior samples are **replicated** on every device.
The forward sampler used internally also requires every traced site to have a fixed shape across all sample iterations.

Several modeling patterns can violate this requirement.
The two most common are:

* **Local latent variables**

  A :external:func:`~numpyro.primitives.sample` call inside a :external:class:`~numpyro.primitives.plate` without ``obs=`` produces a posterior whose shape grows with the plate size (e.g., ``(num_samples, n_obs)``).
  Because ``n_obs`` can differ between training and prediction, the replicated-posterior contract breaks.

* **The** :external:func:`~numpyro.contrib.control_flow.scan` **primitive**

  :external:func:`~numpyro.contrib.control_flow.scan` is commonly used for sequential or autoregressive models (e.g., state-space models, time-series forecasting).
  The number of sites it creates typically grows with the sequence length, making the trace shape dynamic for the same reason.

When this incompatibility is detected, :meth:`~aimz.ImpactModel.predict` issues a warning and automatically falls back to :meth:`~aimz.ImpactModel.predict_on_batch`, which processes data in a single batch without sharding.
However, if the posterior shapes are fundamentally incompatible with the new input, the forward pass will still fail with a shape mismatch.

Note that a model with nested plates whose :external:func:`~numpyro.primitives.sample` sites are all observed (``obs=``) or whose latent shapes are fixed remains compatible.
In contrast, a model with a single plate containing one unobserved :external:func:`~numpyro.primitives.sample` site triggers the fallback.
The following examples illustrate the compatibility with :meth:`~aimz.ImpactModel.predict`:

.. code-block:: python

    import numpyro.distributions as dist
    from numpyro import plate, sample

    X, y = ...


    # Compatible with .predict(): all latents are global
    def kernel(X, y=None):
        ...
        alpha = sample("alpha", dist.Normal())
        beta = sample("beta", dist.Normal().expand([X.shape[1]]))
        with plate("obs", X.shape[0]):
            mu = alpha + X @ beta
            sample("y", dist.Normal(mu), obs=y)


    # Falls back to .predict_on_batch(): `mu` is a local latent
    # with posterior shape (num_samples, X.shape[0])
    def kernel(X, y=None):
        ...
        with plate("obs", X.shape[0]):
            mu = sample("mu", dist.Normal())
            sample("y", dist.Normal(mu), obs=y)

If you encounter an unsupported pattern—ideally with a minimal reproducible example—please `open an issue <https://github.com/markean/aimz/issues/new>`_ or submit a PR.
We plan to broaden coverage based on user needs.

Non-static site shapes
^^^^^^^^^^^^^^^^^^^^^^
On multi-device systems, :meth:`~aimz.ImpactModel.predict` relies on `data parallelism <https://jax.readthedocs.io/en/latest/distributed_data_loading.html#data-parallelism>`_: the input data is sharded across devices along axis 0 (the observation dimension), while all other parameters of the model, including posterior samples, are replicated across all devices.
This requires that posterior sample sites have a global shape, meaning their shape is static and does not depend on data.

A common case where this requirement is violated is models with local latents, where the posterior sample shape grows linearly with the number of observations, resulting in shape ``(num_samples, n_obs)``.
As both the memory footprint and parameter dimensionality grow with the dataset, such models become inherently difficult to scale and distribute.
The following examples illustrate the compatibility with :meth:`~aimz.ImpactModel.predict`.

.. code-block:: python

    import numpyro.distributions as dist
    from numpyro import plate, sample

    X, y = ...


    # Compatible with .predict(): all latents are global
    def kernel(X, y=None):
        ...
        alpha = sample("alpha", dist.Normal())
        beta = sample("beta", dist.Normal().expand([X.shape[1]]))
        with plate("obs", X.shape[0]):
            mu = alpha + X @ beta
            sample("y", dist.Normal(mu), obs=y)


    # Incompatible with .predict(): `mu` is a local latent
    # with posterior shape (num_samples, X.shape[0])
    def kernel(X, y=None):
        ...
        with plate("obs", X.shape[0]):
            mu = sample("mu", dist.Normal())
            sample("y", dist.Normal(mu), obs=y)



Does aimz ship built-in model templates?
----------------------------------------
No.
This is intentional to keep the library lightweight and avoid prescribing a specific modeling style.
Future recipes or example galleries may be provided separately, but the library itself does not include canonical model classes.


What kinds of data can aimz handle?
-----------------------------------
Current functionality is optimized for tall tabular array inputs: one or more two-dimensional numeric arrays of shape ``(n, d)`` (NumPy / JAX), along with a one-dimensional output variable of shape ``(n,)``.
Multiple named arrays are supported as long as they share the same leading dimension ``n``.
Support for other modalities—including images, text, sequences with temporal axes, or ragged/nested structures—is on the roadmap.
In some cases, these can already be adapted by reshaping to 2D during preprocessing and reversing the reshape inside the model.
If native support for a specific structure is important for your use case, opening an issue helps prioritize it, and contributions are welcome.


Can I use aimz for general-purpose Bayesian inference?
------------------------------------------------------
Yes.
While aimz is designed to streamline probabilistic impact modeling workflows—such as intervention analysis and causal effect estimation—it also functions as a flexible Bayesian inference toolkit.
You can use aimz for a wide range of Bayesian modeling tasks supported by NumPyro, including regression, classification, uncertainty quantification, and predictive simulation, even if your application doesn’t involve interventions or causal analysis.


Can I use posterior samples generated elsewhere?
------------------------------------------------
Yes—you do not need to train a model from scratch and sample posteriors.
After initializing an :class:`~aimz.ImpactModel` with your model, call :meth:`~aimz.ImpactModel.set_posterior_sample` with a dictionary mapping site names to arrays.
Each array must share the same leading dimension (number of draws), and the dictionary must not be empty.
Once injected, the model is treated as fitted, and the prediction, log-likelihood, and posterior predictive methods will use the supplied samples.


When should I use the ``*_on_batch`` variants?
----------------------------------------------
Use the batch-specific variants only when you need explicit, single-batch control (e.g., custom training loops, micro‑benchmarking, or integrating with external schedulers).
The higher-level methods handle internal batching, iteration, shuffling, streaming, and aggregation automatically and are preferred for typical workflows.
See :doc:`user_guide/disk_and_on_batch` for a detailed comparison of both approaches and guidance on when to use each.


How do I control which variables (sites) are sampled?
-----------------------------------------------------
By default, prediction and sampling methods use the set of return sites cached in :attr:`~aimz.model.KernelSpec.return_sites`—typically the model output plus any deterministic sites discovered during the first trace.
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
It can also be easily converted to an :external:class:`arviz.InferenceData` object using :external:func:`arviz.from_datatree`.
If desired, you can pass ``return_datatree=False`` to methods such as :meth:`~aimz.ImpactModel.predict_on_batch` to return a plain dictionary instead.


Why do I not see a ``posterior`` group in the output?
-----------------------------------------------------
It appears in the returned :class:`~xarray.DataTree` only if posterior samples are available (fitted or injected).


Where is the on-disk output written?
------------------------------------
All outputs are written under the directory passed via ``output_dir``.
If ``output_dir=None``, a temporary directory is created (accessible via
:attr:`~aimz.ImpactModel.temp_dir`) and removed when the model is cleaned up
(either explicitly with :meth:`~aimz.ImpactModel.cleanup` or when the instance is
garbage collected).
Each group in the returned :class:`~xarray.DataTree` stores its own artifact path
in an ``output_dir`` attribute, and the root tree includes the top-level path.


Does serialization persist the posterior samples?
-------------------------------------------------
Yes.
Pickling (or MLflow integration via :mod:`aimz.mlflow`) preserves the posterior samples (if set) and the cached :class:`~aimz.model.KernelSpec` so retracing / re-fitting is unnecessary upon load.
See :doc:`user_guide/model_persistence` or :doc:`user_guide/mlflow` for more details.
