Data Loaders
============

The streaming methods (:meth:`~aimz.ImpactModel.predict`, :meth:`~aimz.ImpactModel.sample_posterior_predictive`, :meth:`~aimz.ImpactModel.sample_prior_predictive`, and :meth:`~aimz.ImpactModel.log_likelihood`) accept their input either as arrays or as a *data loader*: any finite iterable that yields one batch at a time.
This guide describes the batch contract a data loader must satisfy, the built-in :class:`~aimz.utils.data.ArrayDataset` and :class:`~aimz.utils.data.ArrayLoader`, how loaders integrate with the high-level methods, and how to use an external loader such as a PyTorch ``DataLoader`` for inference or in a custom training loop.


The Batch Contract
------------------
A data loader is any finite iterable (a list, a generator, or an object defining ``__iter__``) whose items are mappings from kernel parameter names to NumPy or JAX arrays.
Each batch must satisfy the following:

* It contains the input field, named after :attr:`~aimz.ImpactModel.param_input` (``"X"`` by default).
* For :meth:`~aimz.ImpactModel.log_likelihood`, it also contains the output field, named after :attr:`~aimz.ImpactModel.param_output` (``"y"`` by default).
* Any additional array argument of the kernel is supplied as a field named after that parameter, not as a keyword argument alongside the loader.
  Non-array keyword arguments are passed to the method as usual.
* Every field has the batch's observations on its leading axis, and all fields share that axis size.
* Field names and the shapes beyond the leading axis stay the same from batch to batch.

Batches may differ in size, and the loader does not need to define ``__len__``.
Padding for multi-device sharding and device placement are handled by aimz one batch at a time, so a loader yields exactly its own rows.
Streamed results are written in the order the batches arrive, so a loader used for prediction or likelihood evaluation should yield the data in a fixed order.

A generator is the simplest data loader:

.. code-block:: python

    def batches(X, y, size):
        for start in range(0, len(X), size):
            yield {"X": X[start : start + size], "y": y[start : start + size]}


    dt = im.predict(batches(X, y, size=1000))
    ll = im.log_likelihood(batches(X, y, size=1000))

A generator is consumed once, so build a fresh one for each call.

.. note::

    Data loaders apply to the streaming methods under ``shard_axis="obs"`` (the default).
    ``shard_axis="draw"`` holds the whole input resident on every device, so it requires an in-memory array (see :doc:`sharding`).
    :meth:`~aimz.ImpactModel.fit` accepts arrays or an :class:`~aimz.utils.data.ArrayLoader`; for any other loader, write a training loop with :meth:`~aimz.ImpactModel.train_on_batch` (see :ref:`external-loaders`).
    The ``batch_size`` argument of the streaming methods is ignored for a loader, which controls its own batching.


Built-in Dataset & Loader
-------------------------
:class:`~aimz.utils.data.ArrayDataset`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It wraps one or more named arrays passed as keyword-only arguments.
All arrays must share the same leading-axis size (the observation axis).
By default arrays are stored as supplied; pass ``to_jax=True`` to convert to JAX arrays at construction.

.. code-block:: python

   from aimz.utils.data import ArrayDataset

   X, y = ...   # X and y are array-like
   dataset = ArrayDataset(X=X, y=y)
   len(dataset)         # total number of samples
   sample = dataset[0]  # {'X': X[0], 'y': y[0]} (dict of field -> element)


:class:`~aimz.utils.data.ArrayLoader`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It consumes an :class:`~aimz.utils.data.ArrayDataset` and yields one batch mapping (field name to array) at a time, satisfying the contract above.
Batches follow the dataset order, or a fresh permutation per epoch with ``shuffle=True``, seeded from ``rng_key``.
``batch_size`` must be a positive integer; the last batch is smaller when the dataset size is not a multiple of it.

.. code-block:: python

   from jax import random
   from aimz.utils.data import ArrayDataset, ArrayLoader

   loader = ArrayLoader(
       ArrayDataset(X=X, y=y),
       rng_key=random.key(0),
       batch_size=10,
       shuffle=True,
   )

   for batch in loader:
       # batch is a dict: {'X': ..., 'y': ...}
       ...


.. note::
    :class:`~aimz.utils.data.ArrayDataset` and :class:`~aimz.utils.data.ArrayLoader` are lightweight utilities for working with in-memory arrays.
    They are intentionally minimal and primarily used internally to enable batching and optional shuffling.
    The user can use them directly, but they are not meant to be a comprehensive data pipeline abstraction.
    For out-of-core datasets, write a generator that streams batches from disk or cloud storage and pass it directly (see :ref:`external-loaders`).


Storage and Device Transfer
---------------------------
When raw arrays are passed to high-level methods like :meth:`~aimz.ImpactModel.fit` or :meth:`~aimz.ImpactModel.predict`, aimz stores them as NumPy arrays on host memory and transfers one batch at a time to the device during iteration.
JAX arrays passed in are converted to NumPy at this stage; their original device placement is not preserved.
This allows datasets larger than device memory to be processed without modification.
The batches a data loader yields are treated the same way: NumPy batches stay on the host until their turn, and each batch is padded for sharding if needed and placed on the model's device or sharding.
You can keep arrays on device by constructing a loader explicitly with ``to_jax=True``:

.. code-block:: python

   loader = ArrayLoader(
       ArrayDataset(X=X, y=y, to_jax=True),
       rng_key=random.key(0),
       batch_size=batch_size,
   )
   im.predict(loader)


Integration with High-Level Methods
-----------------------------------
The streaming methods accept raw arrays (``X``, ``y``, etc.), an :class:`~aimz.utils.data.ArrayLoader`, or any data loader satisfying the contract above, while :meth:`~aimz.ImpactModel.fit` accepts raw arrays or an :class:`~aimz.utils.data.ArrayLoader`.
Passing a loader gives finer control over batch size, ordering, shuffling, and storage backend (see above).
If the user passes raw arrays instead, :meth:`~aimz.ImpactModel.fit` may internally construct a temporary loader with heuristic batching.

.. code-block:: python

    from numpyro.infer import SVI

    from aimz import ImpactModel

    # Set up variational inference strategy
    vi = SVI(model, ...)

    # Initialize ImpactModel with a model, random key, and inference object
    im = ImpactModel(model, rng_key=random.key(0), inference=vi)

    # Build one loader for both fit and predict. Loaders do not shuffle by default,
    # so prediction output stays aligned with the input order.
    loader = ArrayLoader(
        ArrayDataset(X=X, y=y),
        rng_key=random.key(0),
        batch_size=10,
    )

    # Explicit batching for fit
    im.fit(loader, epochs=10)

    # Predictions accept the same loader for consistent batching
    preds = im.predict(loader)

.. note::

    :class:`~aimz.utils.data.ArrayLoader` does not shuffle by default (``shuffle=False``), which is what prediction requires: streamed output is written in input order, so a ``shuffle=True`` loader would misalign the results with the input rows.
    Enable ``shuffle=True`` only for training/fit loops.


Custom Training Loops with :meth:`~aimz.ImpactModel.train_on_batch`
-------------------------------------------------------------------
For fine-grained control (e.g., custom scheduling, gradient accumulation, or early stopping), a custom training loop can be built with :meth:`~aimz.ImpactModel.train_on_batch`.

.. code-block:: python

    im = ImpactModel(...)

    for epoch in range(num_epochs):
        for batch in loader:
            # Perform one update step on this batch
            im.train_on_batch(**batch)
            ...

        # (Optional) validation, logging, early stop checks


.. _external-loaders:

Using Other Data Loader Implementations
---------------------------------------
You are not restricted to the built-in loader.
For the streaming methods, wrap an external loader in a generator that converts each batch into a mapping of NumPy or JAX arrays keyed by the kernel's parameter names:

.. code-block:: python

    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    im = ImpactModel(...)

    # PyTorch DataLoader example (CPU tensors -> NumPy conversion per batch)
    loader = DataLoader(TensorDataset(X, y), batch_size=1000)


    def batches():
        for X_batch, y_batch in loader:
            yield {"X": np.asarray(X_batch), "y": np.asarray(y_batch)}


    dt = im.predict(batches())
    ll = im.log_likelihood(batches())

For training, iterate the loader yourself and call :meth:`~aimz.ImpactModel.train_on_batch` on each batch:

.. code-block:: python

    losses = []
    for epoch in range(num_epochs):
        for X_batch, y_batch in loader:
            batch = {"X": jnp.asarray(X_batch), "y": jnp.asarray(y_batch)}
            _, loss = im.train_on_batch(**batch)
            losses.append(jax.device_get(loss))


After a manual training loop you can populate the model state so downstream calls (prediction, posterior predictive sampling) work the same as after :meth:`~aimz.ImpactModel.fit`:

1. Set :attr:`~aimz.ImpactModel.vi_result` to a structure containing the final parameters, the internal SVI state, and the loss history.
2. Draw posterior samples with :meth:`~aimz.ImpactModel.sample` (``return_datatree=False`` to get a raw dictionary instead of a :external:class:`~xarray.DataTree`).
3. Register the samples  via :meth:`~aimz.ImpactModel.set_posterior_sample`.

.. code-block:: python

    from numpyro.infer.svi import SVIRunResult

    # Store final VI parameters, the internal SVI state, and the collected loss trace
    im.vi_result = SVIRunResult(
        im.inference.get_params(im._vi_state),
        im._vi_state,
        losses,
    )

    # Obtain posterior samples
    posterior_sample = im.sample(return_datatree=False)

    # Register the samples so predictive methods can use them
    im.set_posterior_sample(posterior_sample)

The same wrapped loader then serves prediction and likelihood evaluation directly, as shown above.


See Also
--------
* `PyTorch DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__: Widely used reference implementation.
* `Grain <https://google-grain.readthedocs.io/>`__: JAX-native scalable input pipeline.
* `Dataloader for JAX <https://birkhoffg.github.io/jax-dataloader/>`__: Minimal NumPy/JAX DataLoader.
