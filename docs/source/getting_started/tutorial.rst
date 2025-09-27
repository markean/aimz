.. _NumPyro: https://num.pyro.ai/
.. _Zarr: https://zarr.readthedocs.io/

Tutorial
========
This guide demonstrates a minimal end-to-end workflow with :class:`~aimz.ImpactModel` for a binary outcome model using a Bayesian neural network.
The process involves simulating data, setting up a neural network, fitting the model via stochastic variational inference (SVI), and generating posterior predictive samples for downstream summaries or evaluation.


Synthetic Dataset
-----------------
A synthetic binary outcome dataset is generated.
A random weight vector ``beta`` defines the linear logit signal; applying the sigmoid to ``X @ beta`` yields the true success probabilities, and a Bernoulli draw produces the binary labels ``y``.

.. jupyter-execute::

  import jax.numpy as jnp
  import matplotlib.pyplot as plt
  import numpyro.distributions as dist
  from flax import nnx
  from jax import Array, default_backend, random
  from jax.typing import ArrayLike
  from numpyro import deterministic, plate, sample
  from numpyro.contrib.module import random_nnx_module
  from numpyro.infer import SVI, Trace_ELBO, init_to_uniform
  from numpyro.infer.autoguide import AutoNormal
  from optax import adam

  from aimz import ImpactModel

  %config InlineBackend.figure_format = "retina"

  n = 1_000_000   # dataset size
  d = 100         # dimensionality
  rng_key = random.key(0)
  rng_key, rng_subkey_X, rng_subkey_beta, rng_subkey_y = random.split(rng_key, 4)
  X = random.normal(rng_subkey_X, (n, d))
  beta = random.normal(rng_subkey_beta, (d,))
  logits = X @ beta
  p_true = nnx.sigmoid(logits)
  y = random.bernoulli(rng_subkey_y, p_true).astype(jnp.int32)


Model Specification
-------------------
Three fully connected layers with ReLU activations followed by a sigmoid define the neural network used as the kernel of the :class:`~aimz.ImpactModel`.
The ``plate("data", ...)`` construct enables efficient minibatch SVI via subsampling, scaling inference to large datasets.
A deterministic site ``p`` records the per‑observation probabilities so posterior predictive methods can return them directly without rerunning the forward pass.

.. jupyter-execute::

  class MLP(nnx.Module):
    dtype: jnp.dtype = jnp.bfloat16 if default_backend() == "gpu" else jnp.float32

    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs) -> None:
        self.linear1 = nnx.Linear(din, dmid, dtype=self.dtype, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dmid, dtype=self.dtype, rngs=rngs)
        self.linear3 = nnx.Linear(dmid, dout, dtype=self.dtype, rngs=rngs)

    def __call__(self, x: ArrayLike) -> Array:
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.relu(x)
        x = self.linear3(x)

        return nnx.sigmoid(x).squeeze()


  rng_key, rng_subkey = random.split(rng_key)
  p_nn_module = MLP(
      din=X.shape[1],
      dmid=32,
      dout=1,
      rngs=nnx.Rngs(params=rng_subkey),
  )


  def model(X: ArrayLike, *, y: ArrayLike | None = None) -> None:
      nn_p = random_nnx_module(
          "nn_p",
          nn_module=p_nn_module,
          scope_divider="_",
          prior=dist.Normal(),
      )
      with plate("data", size=n, subsample_size=len(X)):
          p = nn_p(X)
          deterministic("p", p)
          sample("y", dist.Bernoulli(p), obs=y)


\

The :class:`~aimz.ImpactModel` encapsulates the model and provides a unified API for fitting, sampling, posterior and posterior predictive generation, effect estimation, and diagnostics.
Initialization requires the model callable (the kernel), a JAX pseudo-random number generator key (``rng_key``), and an ``inference`` object (here an :external:class:`~numpyro.infer.svi.SVI` configured with a guide, optimizer, and ELBO loss).
During initialization, available accelerators (CPU/GPU) are automatically detected and registered, and the internal PRNG state is seeded for subsequent calls.

.. jupyter-execute::

  rng_key, rng_subkey = random.split(key=rng_key)
  im = ImpactModel(
      model,
      rng_key=rng_subkey,
      inference=SVI(
          model,
          guide=AutoNormal(model=model, init_loc_fn=init_to_uniform(radius=0.1)),
          optim=adam(learning_rate=1e-3),
          loss=Trace_ELBO(),
      ),
  )


Training
--------
The :meth:`~aimz.ImpactModel.fit` method performs mini‑batch SVI updates of the variational parameters.
After optimization it automatically draws posterior samples using the configured guide.
The SVI results are stored in the :attr:`~aimz.ImpactModel.vi_result` attribute of the :class:`~aimz.ImpactModel` instance, including the optimized variational parameters and ELBO loss history.

.. jupyter-execute::
  :hide-output:

  im.fit(X, y, num_samples=500, batch_size=2000, epochs=10, progress=False)


.. jupyter-execute::

  fig, ax = plt.subplots(figsize=(7.5, 6))
  ax.plot(im.vi_result.losses)
  ax.set(yscale="log")
  ax.set_xlabel("Iteration", fontsize=14)
  ax.set_ylabel("Loss (log scale)", fontsize=14)
  ax.set_title("ELBO Loss", fontsize=18);


Inference
---------
Prediction and posterior predictive sampling are performed with :meth:`~aimz.ImpactModel.predict`, which streams input batches to produce predictive draws.
Computation is JIT-compiled and automatically sharded across available devices, while sampling and disk writes run concurrently with results saved incrementally in Zarr_ format.

The ``batch_size`` parameter controls per‑step memory footprint and the chunk size of stored arrays.
The ``output_dir`` parameter specifies where timestamped subdirectories are created; if omitted, a model‑scoped temporary directory is allocated.
Interventions (do-operations) are passed via the ``intervention`` argument and applied with NumPyro_'s :external:class:`~numpyro.handlers.do` effect handler, enabling structural graph surgery without modifying the original kernel.
The return value is an :external:class:`~xarray.DataTree`.

.. jupyter-execute::

  dt = im.predict(X, batch_size=100_000)
  dt


\

When no explicit ``output_dir`` is provided, a temporary directory is created lazily and is automatically removed when the :class:`~aimz.ImpactModel` instance is finalized (object destruction).
Calling :meth:`~aimz.ImpactModel.cleanup` is optional but recommended for deterministic early release of disk resources, particularly in long-running kernels.
If an explicit ``output_dir`` was supplied or the directory has already been removed, the method performs no action.

.. jupyter-execute::

  im.cleanup()
