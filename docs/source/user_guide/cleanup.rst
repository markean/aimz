Output Directory Cleanup
========================

Many high‑level methods of :class:`~aimz.ImpactModel` stream results to disk (e.g., posterior predictive samples, predictions) to support large datasets and memory efficiency.
Each of these methods accepts an ``output_dir`` parameter (see :doc:`disk_and_on_batch` for broader I/O behavior of them).
This page focuses on managing the temporary directory created when the user does not supply ``output_dir`` and on the :meth:`~aimz.ImpactModel.cleanup` method, as well as the :meth:`~aimz.ImpactModel.cleanup_models` class method, which removes temporary directories for all live model instances.


Creation Logic
--------------
When a disk‑writing method is called with ``output_dir=None`` (the default), the model creates a process‑scoped temporary root directory (via :class:`tempfile.TemporaryDirectory`) the first time such a call occurs.
Each invocation then writes to a timestamped subdirectory under that root, ensuring that earlier results are never overwritten.
Subdirectories follow the pattern ``<UTC-timestamp>_<caller_name>/``, where ``<caller_name>`` is the name of the method that triggered the write operation.
This root directory is stored in the :attr:`~aimz.ImpactModel.temp_dir` attribute and reused for subsequent calls until the user invokes :meth:`~aimz.ImpactModel.cleanup`.

Example Layout (implicit temp root)::

    /tmp/tmpz00u5kxk/       # model.temp_dir (root, reused until cleanup)
        20250926T185250223698Z_sample_prior_predictive/
        20250926T185359570134Z_log_likelihood/
        20250926T185419208087Z_predict/

If the user provides ``output_dir``, that directory becomes the root, and it will be created if it does not already exist.
The same timestamped subdirectory pattern is used there (e.g., ``my_runs/20250917T013040123456Z_predict``).
An explicit ``output_dir`` is **not** deleted by :meth:`~aimz.ImpactModel.cleanup`, since it is assumed that the user intends to manage its lifecycle manually.


Cleanup Behavior
----------------
Removes only the internally created **temporary** root directory, including all its timestamped subdirectories.
The path of the removed directory is logged for reference.
If no temporary directory exists, or if it has already been removed, the method is a no-op.

Additional guarantees:

* Safe to call multiple times; subsequent calls do nothing.
* Explicitly provided ``output_dir`` and its subdirectories are never deleted.
* Internal references are cleared so future implicit calls create a fresh temporary root.


Rationale for Explicit Calls
----------------------------
Although :class:`tempfile.TemporaryDirectory` *attempts* automatic removal upon garbage collection, the timing is nondeterministic—especially in notebooks or long-lived processes.
Large artifacts can accumulate quickly; calling :meth:`~aimz.ImpactModel.cleanup` ensures prompt reclamation of disk space.


Accessing Artifact Paths
------------------------
Every disk-writing method records the call's artifact path — the timestamped subdirectory holding the Zarr_ store with the results — in the ``artifact_path`` attribute, set on both the root tree and the group node (``tree.attrs["artifact_path"]`` and ``tree[<group>].attrs["artifact_path"]``).
The enclosing base directory is simply ``Path(artifact_path).parent``, and the temporary root (when no ``output_dir`` was given) is also available via :attr:`~aimz.ImpactModel.temp_dir`.
:meth:`~aimz.ImpactModel.estimate_effect` records the artifact paths of its two scenarios under ``artifact_path_baseline`` and ``artifact_path_intervention`` when the corresponding outputs were streamed to disk.
The output below shows an example :external:class:`xarray.DataTree` illustrating the artifact paths.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import logging

    import jax
    from aimz.model import ImpactModel
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from jax import random
    from numpyro import sample
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal
    from jax import jit

    logging.basicConfig(level=logging.INFO, force=True)


    def lm(X, y=None) -> None:
        """Linear regression model."""
        n_features = X.shape[1]

        # Priors for weights and bias
        w = sample("w", dist.Normal(jnp.zeros(n_features), jnp.ones(n_features)))
        b = sample("b", dist.Normal())

        # Likelihood
        mu = jnp.dot(X, w) + b
        sigma = sample("sigma", dist.Exponential())
        sample("y", dist.Normal(mu, sigma), obs=y)


    rng_key = random.key(42)
    key_w, key_b, key_x, key_e = random.split(rng_key, 4)

    w = random.normal(key_w, (2,))
    b = random.normal(key_b)

    X = random.normal(key_x, (100, 2))
    e = random.normal(key_e, (100,))
    y = jnp.dot(X, w) + b + e

    vi = SVI(
        lm,
        guide=AutoNormal(lm),
        optim=numpyro.optim.Adam(step_size=1e-3),
        loss=Trace_ELBO(),
    )

    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    im.fit_on_batch(X, y, progress=False)
    dt = im.predict(X)
    del dt["posterior"]

.. jupyter-execute::
    :hide-code:

    dt

.. note::

    A temporary directory is reclaimed when :meth:`~aimz.ImpactModel.cleanup` or :meth:`~aimz.ImpactModel.cleanup_models` is called, or when the model is garbage-collected.
    Afterwards the returned :external:class:`xarray.DataTree` and all its group entries remain accessible, but any arrays that were stored on disk read back with **all values set to zero**, since the underlying data files are gone.
    A temporary result is therefore valid only while its model is alive; pass an explicit ``output_dir`` to keep results beyond the model's lifetime.


Cleaning Multiple Models
------------------------
When a process creates multiple :class:`~aimz.ImpactModel` instances, it can be useful to clean up all their temporary directories in a single call.
The class method :meth:`~aimz.ImpactModel.cleanup_models` iterates over all live model instances and calls their :meth:`~aimz.ImpactModel.cleanup` method.
For example, this can be used as a pipeline hook after a run to clean up all temporary directories without tracking individual model instances.

Example::

    from aimz.model import ImpactModel

    # Create multiple instances and write to temporary directories
    im1 = ImpactModel(...)
    im1.fit(...)
    im1.predict(...)

    im2 = ImpactModel(...)
    im2.fit(...)
    im2.predict(...)

    # Clean temporary directories for all active instances
    ImpactModel.cleanup_models()

    print(im1.temp_dir)  # None
    print(im2.temp_dir)  # None


Typical Usage Pattern
---------------------
A typical workflow is to run these methods without specifying ``output_dir`` (using a temporary root), optionally access the results via the :attr:`~aimz.ImpactModel.temp_dir` attribute or the returned :external:class:`xarray.DataTree`, and then free disk space with :meth:`~aimz.ImpactModel.cleanup` or :meth:`~aimz.ImpactModel.cleanup_models`.

Tips for safe use:

* Use :meth:`~aimz.ImpactModel.cleanup` at the end of a notebook or in a ``finally`` block.
* Use :meth:`~aimz.ImpactModel.cleanup_models` to remove temporary directories for all live model instances at once.
* Copy any results you want to keep before :meth:`~aimz.ImpactModel.cleanup` or :meth:`~aimz.ImpactModel.cleanup_models`.
* In tests, check that temporary directories are removed to avoid disk bloat.
* Avoid leaving long sessions with un-cleaned temporary directories.
