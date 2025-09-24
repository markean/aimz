Training & Inference with Data Loaders
======================================

This guide explains how to use the built-in :class:`~aimz.utils.data.ArrayDataset` and :class:`~aimz.utils.data.ArrayLoader`, how they integrate with high-level methods like :meth:`~aimz.ImpactModel.fit` / :meth:`~aimz.ImpactModel.predict`, and how to construct fully custom training or inference loops (e.g., integrating a PyTorch ``DataLoader``).


Built-in Dataset & Loader
-------------------------
:class:`~aimz.utils.data.ArrayDataset`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It wraps one or more named arrays passed as keyword-only arguments.
All arrays must share the same leading dimension (the sample axis).
By default inputs are converted to JAX arrays (set ``to_jax=False`` to skip conversion).

.. code-block:: python

   from aimz.utils.data import ArrayDataset

   X, y = ...   # X and y are array-like
   dataset = ArrayDataset(X=X, y=y)
   len(dataset)         # total number of samples
   sample = dataset[0]  # {'X': X[0], 'y': y[0]} (dict of field -> element)


:class:`~aimz.utils.data.ArrayLoader`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It consumes an :class:`~aimz.utils.data.ArrayDataset` and produces an iterator of ``(batch_dict, n_pad)`` pairs:

* ``batch_dict`` maps each field name to a (possibly padded) mini-batch array.
* ``n_pad`` is the number of synthetic examples added so the (possibly last) batch size is divisible by the number of local devices.
  When no device is specified (``device=None``), no padding is performed and ``n_pad`` is always ``0``.
  If set, batch is padded (if needed) then moved via :func:`jax.device_put`.
  Padding uses :external:func:`jax.numpy.pad` with ``mode="edge"`` (it repeats the last row) so shapes align for sharded computations; callers can ignore those rows (track via ``n_pad``).
  The ``batch_size`` must be a positive integer, and if using a device or sharding it is best to choose a multiple of :external:func:`jax.local_device_count()` to avoid padding.

.. code-block:: python

   import jax
   from jax import random
   from aimz.utils.data import ArrayDataset, ArrayLoader

   # Suppose local_device_count() == 8 and batch_size == 10 -> padded to 16
   loader = ArrayLoader(
     ArrayDataset(X=X, y=y),
     rng_key=random.key(0),
     batch_size=10,
     shuffle=True,
     device=jax.devices()[0],  # or a Sharding spec
   )

   for batch, n_pad in loader:
       # batch is a dict: {'X': ..., 'y': ...}; n_pad == 6
       ...


.. note::
    :class:`~aimz.utils.data.ArrayDataset` and :class:`~aimz.utils.data.ArrayLoader` are lightweight utilities for working with in-memory (JAX) arrays.
    They are intentionally minimal and primarily used internally to enable batching, optional shuffling, and (when required) padding for device sharding.
    The user can use them directly, but they are not meant to be a comprehensive data pipeline abstraction.
    For out-of-core datasets, implement a generator that streams data in chunks from disk or cloud storage.


Integration with High-Level Methods
-----------------------------------
High-level methods (:meth:`~aimz.ImpactModel.fit`, :meth:`~aimz.ImpactModel.predict`) accept either raw arrays (``X``, ``y``, etc.) or an :class:`~aimz.utils.data.ArrayLoader`.
Passing a loader gives finer control over batch size, ordering, and shuffling.
Any model-level device or sharding configuration takes precedence over the loader's ``device`` argument.
If the user pass raw arrays instead, :meth:`~aimz.ImpactModel.fit` may internally construct a temporary loader with heuristic batching.

.. code-block:: python

    from numpyro.infer import SVI

    from aimz import ImpactModel

    # Set up variational inference strategy
    vi = SVI(model, ...)

    # Initialize ImpactModel with a model, random key, and SVI object
    im = ImpactModel(model, rng_key=random.key(0), svi=vi)

    # Use a prepared ArrayLoader for explicit batching/shuffling
    im.fit(loader, epochs=10)

    # Predictions also accept a loader for consistent batching
    preds = im.predict(loader)


Custom Training Loops with :meth:`~aimz.ImpactModel.train_on_batch`
-------------------------------------------------------------------
For fine-grained control (e.g., custom scheduling, gradient accumulation, or early stopping), a custom training loop can be built with :meth:`~aimz.ImpactModel.train_on_batch`.

.. code-block:: python

    im = ImpactModel(...)

    for epoch in range(num_epochs):
        for batch, n_pad in loader:  # `n_pad` may be > 0 when padded
            if n_pad > 0:
                # Optionally handle or ignore the extra padded rows
                ...

            # Perform one update step on this batch
            im.train_on_batch(**batch)
            ...

        # (Optional) validation, logging, early stop checks


Using Other DataLoader Implementations
--------------------------------------
You are not restricted to the built-in loader.
Any iterable that yields a mapping (field name → array) per batch works with a custom loop, provided the arrays are convertible via :external:func:`jax.numpy.asarray`.

.. code-block:: python

    im = ImpactModel(...)

    # PyTorch DataLoader example (CPU → JAX conversion per batch)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    losses = []
    for epoch in range(num_epochs):
        for X_batch, y_batch in loader:
            batch = {"X": jnp.asarray(X_batch), "y": jnp.asarray(y_batch)}
            _, loss = im.train_on_batch(**batch)
            losses.append(loss)


After a manual training loop you can populate the model state so downstream calls (prediction, posterior predictive sampling) work the same as after :meth:`~aimz.ImpactModel.fit`:

1. Set :attr:`~aimz.ImpactModel.vi_result` to a structure containing the final parameters and loss history.
2. Draw posterior samples with :meth:`~aimz.ImpactModel.sample` (``return_datatree=False`` to get a raw dictionary instead of a :external:class:`~xarray.DataTree`).
3. Register the samples  via :meth:`~aimz.ImpactModel.set_posterior_sample`.

.. code-block:: python

    from typing import NamedTuple

    from jax import Array


    class SVIRunResult(NamedTuple):
        params: dict[str, Array]
        losses: list[float]

    # Store final VI parameters and the collected loss trace (assumes `losses` list built above)
    im.vi_result = SVIRunResult(im.inference.get_params(im._vi_state), losses)

    # Obtain posterior samples
    posterior_sample = im.sample(return_datatree=False)

    # Register the samples so predictive methods can use them
    im.set_posterior_sample(posterior_sample)

You can reuse the same loop pattern for prediction or likelihood evaluation:

.. code-block:: python

    # Collect per-batch posterior predictive means for target 'y'
    batch_means = []
    for X_batch, _ in loader:
        preds = im.predict_on_batch(X_batch, return_datatree=False)
        # preds['y'] shape: (num_draws, batch_size, ...); average over draws
        batch_means.append(preds["y"].mean(axis=0))

    # Stitch back together along the sample axis
    posterior_predictive_mean = jnp.concatenate(batch_means, axis=0)
    # ... further metrics / evaluation


See Also
--------
* `PyTorch DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__ – Widely used reference implementation.
* `Grain <https://google-grain.readthedocs.io/>`__ – JAX-native scalable input pipeline.
* `Dataloader for JAX <https://birkhoffg.github.io/jax-dataloader/>`__ – Minimal NumPy/JAX DataLoader.
