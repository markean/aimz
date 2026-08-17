Streaming vs. On-Batch Methods
==============================

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/markean/aimz/blob/main/docs/notebooks/disk_and_on_batch.ipynb
   :alt: Open In Colab

\

This page explains and compares the two complementary execution styles provided by :class:`~aimz.ImpactModel`:

* **Streaming** (default) methods iterate over the input in chunks, materialize results incrementally, and can distribute the computation across devices. Where the streamed results accumulate is a separate choice — the *result store*: a persisted Zarr_ artifact (default) or host memory; see :ref:`result-store`.
* **On-batch** (``*_on_batch`` suffix) methods execute a single, fully in-memory pass and can optionally return a plain :class:`dict` instead of a :external:class:`xarray.DataTree`. The naming mirrors the Keras convention to signal an immediate, single-batch, memory-resident operation.


Why Streaming by Default
------------------------
The non-``*_on_batch`` methods default to a streaming (chunked) execution model for several reasons:

* Posterior predictive and prior predictive tensors can scale as ``(#samples x #dims x #posterior_samples x ...)``.
  Even moderate increases in any axis (time, spatial units, parameter samples) can exceed host or accelerator RAM.
* Using ``batch_size`` with chunked iteration limits peak memory and prevents out-of-memory errors.
* The persistent store creates an artifact you can reopen without rerunning inference. Coordinates and attributes are re-derived when the tree is rebuilt rather than stored on disk (see :ref:`reopening-persisted-outputs`).
* The :external:class:`xarray.DataTree` + Zarr_ format integrates with scientific Python tools such as Dask_ and ArviZ_.
* Summaries (means, HDIs, residual PPC stats) can be computed lazily, chunk by chunk, without first materializing dense arrays.
* One API works for both small experiments and large-scale use cases.


Comparison
----------
Streaming variants target larger datasets, enable chunked processing, multi-device parallelism, and — with the persistent store — stable artifact generation.
These methods build internal data loaders, iterate in chunks, and decouple sampling from result handling, enabling concurrent execution.
Outputs consolidate into a single lazy, Dask_-backed :external:class:`xarray.DataTree` whose chunks live in a Zarr_ store or directly in host memory, depending on the result store.
On-batch variants, in contrast, favor minimal overhead, immediate return, and greater flexibility when posterior sample shapes are not shard-friendly.

.. seealso::

   :doc:`sharding` explains how multi-device sharding works and how to choose the ``shard_axis`` strategy referenced throughout this page.

Feature Summary
^^^^^^^^^^^^^^^
.. list-table::
   :header-rows: 1
   :widths: 22 27 27 24

   * - Feature
     - Streaming, persistent store (default)
     - Streaming, memory store
     - On-batch (``*_on_batch``)
   * - Typical dataset size
     - Medium to large
     - Medium to large (result fits host RAM)
     - Small to moderate
   * - Supported use cases
     - Standard models
     - Standard models
     - Broader model support
   * - Peak memory usage
     - Chunk-bounded
     - Full result resident
     - Full batch resident
   * - Writes to disk
     - Yes
     - No
     - No
   * - Result survives the process
     - Yes (reopenable Zarr_ artifact)
     - No
     - No
   * - Return type
     - Lazy :external:class:`xarray.DataTree` (Dask_ over Zarr_ chunks)
     - Lazy :external:class:`xarray.DataTree` (Dask_ over resident chunks)
     - Eager :external:class:`xarray.DataTree`, or :class:`dict` via ``return_datatree=False``
   * - Custom batch sizing
     - Yes (``batch_size``)
     - Yes (``batch_size``)
     - No (single pass)
   * - Device parallelism (sharding)
     - Yes
     - Yes
     - No
   * - Automatic rerun
     - Yes (reruns as ``draw``)
     - Yes (reruns as ``draw``)
     - No (final mode)
   * - Latency (small data)
     - Higher (I/O + orchestration)
     - Moderate (orchestration)
     - Minimal

Capability Matrix
^^^^^^^^^^^^^^^^^
=============================== ===================================================== ==============================================================
Capability                      Streaming (default)                                   On-batch (``*_on_batch``)
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


.. _result-store:

Choosing a Result Store
-----------------------
Every streaming method accepts a ``store`` argument selecting where the streamed batches accumulate; the batched (and sharded) execution is identical either way.

Persistent (default)
^^^^^^^^^^^^^^^^^^^^
``store="persistent"`` streams batches into a Zarr_ store under ``output_dir`` (a model-owned temporary directory by default) and returns a lazy, Dask_-backed :external:class:`xarray.DataTree` recording the artifact's location in its ``artifact_path`` attribute.
The artifact outlives the call: it can be reopened without rerunning inference (see :ref:`reopening-persisted-outputs`), and the temporary directory is managed through :meth:`~aimz.ImpactModel.cleanup` (see :doc:`cleanup`).

.. _in-memory-results:

Memory
^^^^^^
``store="memory"`` retains the streamed batches in host memory instead of writing them:

.. code-block:: python

    dt = im.predict(X, store="memory")

The returned :external:class:`xarray.DataTree` has the same groups, dimensions, coordinates, and chunk structure as the persistent result — a lazy, Dask_-backed tree — except that its chunks *are* the resident batch arrays.
Downstream summaries, group-bys, quantiles, and :meth:`~aimz.ImpactModel.estimate_effect` contrasts still parallelize across chunks exactly as they do for the persistent tree, but read directly from memory with no file I/O or chunk decoding.
Call :external:meth:`~xarray.DataTree.load` on the result (or :external:meth:`~xarray.DataArray.load` on a variable) to materialize plain eager NumPy arrays — for example when iterating on many small selections, where per-operation Dask_ scheduling overhead dominates.

Use it when the complete result comfortably fits in host memory.
``output_dir`` does not apply and must be left ``None``.
Unlike the on-batch methods, ``store="memory"`` still batches — and, on multiple devices, shards — the computation, so it handles inputs that a single unbatched pass cannot.


Quick Recommendations
---------------------
* Moderate or large data: use the streaming methods (e.g., :meth:`~aimz.ImpactModel.fit`, :meth:`~aimz.ImpactModel.predict`).
* Need results beyond the model's lifetime: keep the default persistent store and pass an explicit ``output_dir``.
* Results fit in RAM and you iterate on summaries, group-bys, or plots: pass ``store="memory"`` to skip the Zarr_ layer — same lazy Dask_ tree, chunks resident in memory — while keeping batched (and sharded) execution; see :ref:`in-memory-results`.
* Small data, rapid iteration, CI, or read-only / ephemeral filesystem: use on-batch (``*_on_batch``), or streaming with ``store="memory"``, which also writes nothing.
* If :meth:`~aimz.ImpactModel.predict` warns that posterior sample shapes are not compatible with ``shard_axis="obs"``, it automatically reruns under ``shard_axis="draw"``.
  Pass ``shard_axis="draw"`` explicitly to silence the warning.
  For posterior shapes that remain incompatible with chunked execution, call :meth:`~aimz.ImpactModel.predict_on_batch` directly.
* Custom training loop: iterate with :meth:`~aimz.ImpactModel.train_on_batch`.
* Need multi-device (sharding) execution: streaming; see :doc:`sharding` for choosing ``shard_axis``.
* Need raw NumPy/dict outputs (no :external:class:`xarray.DataTree`): on-batch with ``return_datatree=False``.

.. note::

   For MCMC inference, only :meth:`~aimz.ImpactModel.fit_on_batch` or :meth:`~aimz.ImpactModel.sample` is supported for training and posterior sampling,
   as MCMC is incompatible with epoch-based or chunked batch processing. See :doc:`mcmc` for more details.


Example: :meth:`~aimz.ImpactModel.predict` with an Automatic Rerun
-------------------------------------------------------------------

A common scenario for the rerun warning occurs when the model contains **local latent variables**, which make posterior sample shapes incompatible with data-parallel (observation-sharded) execution.
:meth:`~aimz.ImpactModel.predict` then warns and reruns under ``shard_axis="draw"``.
The example below illustrates this case.

.. jupyter-execute::
    :hide-output:

    import logging

    import jax.numpy as jnp
    import numpyro.distributions as dist
    from jax import random
    from jax.typing import ArrayLike
    from numpyro import optim, plate, sample
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    from aimz import ImpactModel

    logging.basicConfig(level=logging.INFO, force=True)


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

    # Calling `.predict()` warns and reruns under shard_axis="draw"
    im.predict(X)


.. _reopening-persisted-outputs:

Reopening Persisted Outputs
---------------------------
When you pass an explicit ``output_dir``, each persistent-store call writes one subdirectory containing a Zarr_ group with one array per return site.
Its path is recorded in the returned tree's ``artifact_path`` attribute (the attribute is recorded for temporary-root outputs as well).
Only the sampled arrays and their dimension names are persisted.
Coordinates and attributes are not stored on disk.

To reconstruct the same :external:class:`xarray.DataTree` from the files alone, mirror that read-time step:

.. code-block:: python

    import numpy as np
    import xarray as xr

    # The per-call directory written under `output_dir` (the tree's `artifact_path`)
    store = ...

    # If relevant, add the leading `chain` axis and coordinates as aimz does on read
    ds = xr.open_zarr(store, consolidated=False).expand_dims(dim="chain", axis=0)
    ds = ds.assign_coords({dim: np.arange(ds.sizes[dim]) for dim in ds.sizes})

    dt = xr.DataTree(name="root")
    # Pick a group name for downstream use
    dt["posterior_predictive"] = xr.DataTree(ds)


The ``posterior`` subtree is likewise not stored in a predictive output: aimz attaches it from the
in-memory model when it builds the tree. Persist the model itself (see :doc:`model_persistence`) or
keep :meth:`~aimz.ImpactModel.sample`'s return value if you need the posterior alongside the files.


Performance Tips
----------------
* Tune ``batch_size`` appropriately; it also determines the chunk size for Zarr_-backed arrays.
* Monitor disk usage, as chunk sizes scale with ``batch_size`` and ``num_samples``.
* Reduce ``num_samples`` first for faster iteration.
* Use on-batch methods in tests to minimize I/O overhead.
