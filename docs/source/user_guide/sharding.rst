.. _user-guide-sharding:

Multi-Device Execution and Sharding
===================================

The disk-backed predictive methods of :class:`~aimz.ImpactModel` can distribute work across multiple devices (CPUs, GPUs, or TPUs) by *sharding*, which splits one of the computation's axes across the available devices and runs the pieces in parallel.
aimz shards over whatever devices JAX already exposes and does not select or configure them itself.
These methods accept a ``shard_axis`` argument that selects the strategy.
The on-batch (``*_on_batch``) variants run a single in-memory pass and do not shard; see :doc:`disk_and_on_batch` for the broader disk-backed vs. on-batch comparison.

.. note::

   ``shard_axis`` chooses which axis is split across devices, so it has no effect on a single device.
   The ``shard_axis="obs"`` to ``shard_axis="draw"`` rerun described below is separate: it triggers whenever an observation-aligned posterior would be separated from the observations it indexes (across devices, or across batches when the input is chunked), so it can happen even on a single device.


The Two Strategies
------------------

A predictive pass has two axes worth sharding: the **observation axis** of the input (axis 0) and the **draw axis** of the posterior.
``shard_axis`` picks between them.

Data parallelism (``shard_axis="obs"``, default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The input is sharded across devices along axis 0 (the observation axis), while the posterior samples are **replicated** on every device.
This is the standard `data-parallel <https://jax.readthedocs.io/en/latest/distributed_data_loading.html#data-parallelism>`__ pattern: each device holds the full posterior and processes a slice of the observations.
Because it streams the input (an array or an :class:`~aimz.utils.data.ArrayLoader`), it scales to inputs larger than device memory.
It does require a **replicable** posterior: every sample site must have a static shape that does not depend on the number of observations, which holds when all latents are global.

Draw parallelism (``shard_axis="draw"``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The drawn samples are sharded across devices in chunks of ``batch_size`` draws, while the whole input is held **resident** on every device, so it must be an in-memory array rather than a data loader.
Because the observation axis is never split, a posterior whose shape grows with the observations (a local latent of shape ``(num_samples, n_obs)`` or higher rank) always matches the full input, however it is written.


Choosing a Strategy
-------------------

``shard_axis="obs"`` (the default) suits models with global latents and inputs that are large or supplied through a data loader.
Prefer ``shard_axis="draw"`` when the model has local latents, or when the input is small relative to the number of draws so replicating it on every device stays cheap.

Choosing a ``batch_size`` that is a multiple of :external:func:`jax.local_device_count` keeps the shards even and avoids padding the final batch.


Automatic Rerun for Incompatible Posteriors
-------------------------------------------

When the posterior sample shape is observation-aligned (the hallmark of a **local latent variable**), it cannot be replicated under ``shard_axis="obs"``.
Rather than failing, :meth:`~aimz.ImpactModel.predict` and :meth:`~aimz.ImpactModel.log_likelihood` detect this, emit a warning, and rerun under ``shard_axis="draw"``, keeping results streamed to disk and memory-bounded.
Passing ``shard_axis="draw"`` explicitly avoids the discarded first attempt.
See :ref:`faq-model-compatibility` for the model patterns that trigger a rerun and those that stay unsupported.
The kernels below contrast a compatible model with one that triggers the rerun; a runnable version is in :doc:`disk_and_on_batch`.

.. code-block:: python

    import numpyro.distributions as dist
    from numpyro import plate, sample

    X, y = ...


    # Compatible with shard_axis="obs": all latents are global.
    def kernel(X, y=None):
        ...
        alpha = sample("alpha", dist.Normal())
        beta = sample("beta", dist.Normal().expand([X.shape[1]]))
        with plate("obs", X.shape[0]):
            mu = alpha + X @ beta
            sample("y", dist.Normal(mu), obs=y)


    # `mu` is a local latent with posterior shape (num_samples, X.shape[0]),
    # so the default shard_axis="obs" reruns under shard_axis="draw".
    def kernel(X, y=None):
        ...
        with plate("obs", X.shape[0]):
            mu = sample("mu", dist.Normal())
            sample("y", dist.Normal(mu), obs=y)


When There Is No Posterior to Shard
-----------------------------------

* For :meth:`~aimz.ImpactModel.predict` and :meth:`~aimz.ImpactModel.log_likelihood`, a model with no posterior samples has nothing to shard along the draw axis, so the data-parallel path runs regardless of ``shard_axis``.
* :meth:`~aimz.ImpactModel.sample_prior_predictive` has no posterior to shard: under ``shard_axis="draw"`` it shards the prior draws instead, drawing a fresh prior sample per chunk against the whole input (request latent sites via ``return_sites``); under ``shard_axis="obs"`` it streams over observations.


Constraints
-----------

Draw-parallel sharding does **not** lift the core static-shape requirement: every traced site must have a shape that is fixed across draws.
In particular, :external:func:`~numpyro.contrib.control_flow.scan`-based models, whose number of sites grows with the sequence length, remain unsupported under either strategy.
