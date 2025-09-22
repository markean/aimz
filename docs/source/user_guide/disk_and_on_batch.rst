.. _Dask: https://www.dask.org/
.. _ArviZ: https://python.arviz.org/

Disk-Backed vs. On-Batch Methods
================================

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/markean/aimz/blob/main/docs/notebooks/disk_and_on_batch.ipynb
   :alt: Open In Colab

\

This page explains and compares the two complementary execution styles provided by :class:`~aimz.ImpactModel`:

* **Disk-backed** (default) methods iterate over the input in chunks, materialize results incrementally, and persist structured artifacts (Zarr‑backed :external:class:`xarray.DataTree` plus metadata) to a temporary or user-specified output directory.
* **On-batch** (``*_on_batch`` suffix) methods execute a single, fully in-memory pass and can optionally return a plain :class:`dict` instead of a :external:class:`xarray.DataTree`. The naming mirrors the Keras convention to signal an immediate, single-batch, memory-resident operation.


Why Disk-Backed by Default
--------------------------
The non-``*_on_batch`` methods default to a disk-backed (chunked) execution model for several reasons:

* Posterior predictive and prior predictive tensors can scale as ``(#samples x #dims x #posterior_samples x ...)``.
  Even moderate increases in any axis (time, spatial units, parameter samples) can exceed host or accelerator RAM.
* Using ``batch_size`` with chunked iteration limits peak memory and prevents out-of-memory errors.
* Persisted Zarr arrays with metadata (coords, dims, attrs) create an artifact you can reopen without rerunning inference.
* The :external:class:`xarray.DataTree` + Zarr format integrates with scientific Python tools such as Dask_ and ArviZ_.
* Summaries (means, HDIs, residual PPC stats) can be computed lazily over chunked storage without first materializing dense arrays.
* One API works for both small experiments and large-scale use cases.


Comparison
----------
Disk-backed variants target larger datasets, enable chunked processing, multi-device parallelism, and stable artifact generation.
These methods build internal data loaders, iterate in chunks, and decouple sampling from file I/O, enabling concurrent execution.
Outputs consolidate into a single :external:class:`xarray.DataTree` backed by Zarr files for post-hoc analysis.
On-batch variants, in contrast, favor minimal overhead, immediate return, and greater flexibility when posterior sample shapes are not shard-friendly.

Feature Summary
^^^^^^^^^^^^^^^
============================= ==================================== =======================================================
Feature                       Disk-backed (default)                On-batch (``*_on_batch``)
============================= ==================================== =======================================================
Typical dataset size          Medium → large                       Small → moderate
Supported use cases           Standard models                      Broader model support
Peak memory usage             Chunk-bounded                        Full batch resident
Writes to disk                Yes                                  No
Return type                   :external:class:`xarray.DataTree`    :external:class:`xarray.DataTree` or :class:`dict`
                                                                   (via ``return_datatree=False``)
Custom batch sizing           Yes (``batch_size``)                 No (single pass)
Device parallelism (sharding) Yes                                  No
Automatic fallback            Yes (may auto‑delegate to on‑batch)  No (final mode)
Latency (small data)          Higher (I/O + orchestration)         Minimal
============================= ==================================== =======================================================

Capability Matrix
^^^^^^^^^^^^^^^^^
=============================== ===================================================== ==============================================================
Capability                      Disk-backed (default)                                 On-batch (``*_on_batch``)
=============================== ===================================================== ==============================================================
Full dataset training           :meth:`~aimz.ImpactModel.fit`                         :meth:`~aimz.ImpactModel.fit_on_batch`
Single training step            N/A                                                   :meth:`~aimz.ImpactModel.train_on_batch`
Prior predictive sampling       :meth:`~aimz.ImpactModel.sample_prior_predictive`     :meth:`~aimz.ImpactModel.sample_prior_predictive_on_batch`
Posterior sampling              :meth:`~aimz.ImpactModel.sample`                      N/A
Posterior predictive sampling   :meth:`~aimz.ImpactModel.predict` or                  :meth:`~aimz.ImpactModel.predict_on_batch` or
                                :meth:`~aimz.ImpactModel.sample_posterior_predictive` :meth:`~aimz.ImpactModel.sample_posterior_predictive_on_batch`
Log-likelihood computation      :meth:`~aimz.ImpactModel.log_likelihood`              N/A
Effect estimation               :meth:`~aimz.ImpactModel.estimate_effect`             (consumes outputs above)
=============================== ===================================================== ==============================================================


Quick Recommendations
---------------------
* Moderate or large data, or need persisted outputs: use disk-backed (e.g., :meth:`~aimz.ImpactModel.fit`, :meth:`~aimz.ImpactModel.predict`).
* Small data, rapid iteration, CI, or read-only / ephemeral filesystem: use on-batch (``*_on_batch``).
* If :meth:`~aimz.ImpactModel.predict` issues a fallback warning, call :meth:`~aimz.ImpactModel.predict_on_batch` directly.
  This occurs when the model or posterior sample shapes are incompatible with shard-based chunked execution.
* Custom training loop: iterate with :meth:`~aimz.ImpactModel.train_on_batch`.
* Need multi-device (sharding) execution: disk-backed.
* Need raw NumPy/dict outputs (no :external:class:`xarray.DataTree`): on-batch with ``return_datatree=False``.

.. note::

   For MCMC inference, only :meth:`~aimz.ImpactModel.fit_on_batch` or :meth:`~aimz.ImpactModel.sample` is supported for training and posterior sampling,
   as MCMC is incompatible with epoch-based or chunked batch processing. See :doc:`mcmc` for more details.


Example: :meth:`~aimz.ImpactModel.predict` with Fallback Warning
----------------------------------------------------------------

A common scenario for the fallback warning occurs when the model contains **local latent variables**, which can make posterior sample shapes incompatible with shard-based parallel execution.
The example below illustrates this case.

.. jupyter-execute::
    :hide-output:

    import jax.numpy as jnp
    import numpyro.distributions as dist
    from jax import random
    from jax.typing import ArrayLike
    from numpyro import optim, plate, sample
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    from aimz import ImpactModel


    def model(X: ArrayLike, y: ArrayLike | None = None) -> None:
        # Model includes a local latent variable
        sigma = sample("sigma", dist.Exponential().expand((X.shape[0],)))
        with plate("data", size=X.shape[0]):
            sample("y", dist.Normal(0.0, sigma), obs=y)


    rng_key = random.key(42)
    rng_key, rng_key_X, rng_key_y = random.split(rng_key, 3)
    X = random.normal(rng_key_X, (100, 2))
    y = random.normal(rng_key_y, (100,))


    im = ImpactModel(
        model,
        rng_key=rng_key,
        inference=SVI(
            model,
            guide=AutoNormal(model),
            optim=optim.Adam(step_size=1e-3),
            loss=Trace_ELBO(),
        ),
    # This internally calls the `.run()` method of `SVI`
    ).fit_on_batch(X, y)

.. jupyter-execute::
    :stderr:

    # Calling `.predict()` triggers a fallback warning
    im.predict(X)


Performance Tips
----------------
* Tune ``batch_size`` appropriately; it also determines the chunk size for Zarr-backed arrays.
* Monitor disk usage, as chunk sizes scale with ``batch_size`` and ``num_samples``.
* Reduce ``num_samples`` first for faster iteration.
* Use on-batch methods in tests to minimize I/O overhead.
