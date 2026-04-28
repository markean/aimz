"""Benchmark peak incremental RSS for NumPyro `Predictive`."""

import argparse
import contextlib
import gc
import logging
import os
import threading
import warnings
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from flax import nnx
from jax import (
    Array,
    default_backend,
    default_device,
    device_get,
    devices,
    local_device_count,
    random,
)
from numpyro import plate, sample
from numpyro.contrib.module import random_nnx_module
from numpyro.infer import SVI, Predictive, Trace_ELBO, init_to_uniform
from numpyro.infer.autoguide import AutoNormal
from optax import adam

if find_spec("flox") is None:
    warnings.warn(
        "The 'flox' package is required for efficient 'groupby' operations in Dask "
        "arrays.",
        stacklevel=2,
    )

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_num_cpu_devices", 4)

# ── Constants ─────────────────────────────────────────────────────────────────────────
NUM_FEATURES: int = 100
D_HIDDEN: int = 128
NUM_STEPS: int = 100  # Fit quality is irrelevant.
NUM_SAMPLES: int = 1_000
BATCH_SIZE: int = 10_000


# ── Model ─────────────────────────────────────────────────────────────────────────────
class MLP(nnx.Module):
    """Four-layer MLP with ReLU activations and sigmoid output."""

    dtype: jnp.dtype = jnp.bfloat16 if default_backend() == "gpu" else jnp.float32

    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs) -> None:
        """Initialize with input dim `din`, hidden dim `dmid`, and output dim `dout`."""
        self.linear1 = nnx.Linear(din, dmid, dtype=self.dtype, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dmid, dtype=self.dtype, rngs=rngs)
        self.linear3 = nnx.Linear(dmid, dmid, dtype=self.dtype, rngs=rngs)
        self.linear4 = nnx.Linear(dmid, dout, dtype=self.dtype, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """Forward pass returning sigmoid probabilities."""
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.relu(x)
        x = self.linear3(x)
        x = nnx.relu(x)
        x = self.linear4(x)

        return nnx.sigmoid(x).reshape(-1)


# ── RSS measurement ───────────────────────────────────────────────────────────────────
def _current_rss_mib() -> float:
    """Return current anonymous RSS in MiB."""
    with Path(f"/proc/{os.getpid()}/status").open() as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 1024.0
    msg = "RssAnon not found in /proc/self/status"
    raise RuntimeError(msg)


def _peak_rss_increment(fn: Callable) -> float:
    """Run *fn()* and return the peak RSS increment."""
    gc.collect()

    baseline = _current_rss_mib()
    peak = baseline
    stop = threading.Event()

    def _sampler() -> None:
        nonlocal peak
        while not stop.is_set():
            rss = _current_rss_mib()
            peak = max(peak, rss)
            stop.wait(0.005)

    t = threading.Thread(target=_sampler, daemon=True)
    t.start()
    result = fn()
    with contextlib.suppress(TypeError, ValueError):
        jax.block_until_ready(result)
    rss = _current_rss_mib()
    peak = max(peak, rss)
    stop.set()
    t.join()

    return peak - baseline


# ── Helpers ───────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(rng_key: Array, n: int, p: int) -> tuple[Array, Array]:
    """Generate synthetic data with a nonlinear data generating process.

    Data is generated as JAX arrays on CPU because NumPyro's ``Predictive`` requires
    JAX arrays. The arrays are later moved to the compute device via ``device_put``.
    """
    with default_device(devices("cpu")[0]):
        k1, k2, k3, k4 = random.split(rng_key, 4)
        X = random.normal(k1, (n, p))
        b1 = random.normal(k2, (p,))
        b2 = random.normal(k3, (p,))
        logits = jnp.sin(X @ b1) * jnp.tanh(X @ b2) + 0.5 * jnp.cos((X**2) @ b1)
        y = random.bernoulli(k4, nnx.sigmoid(logits)).astype(jnp.int32)

    return X, y


# ── Main ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


def main() -> None:
    """Benchmark NumPyro ``Predictive`` peak incremental RSS."""
    parser = argparse.ArgumentParser(
        description="Benchmark NumPyro Predictive memory.",
    )
    parser.add_argument("--n", type=int, required=True, help="Dataset size")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} | {levelname} | {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info(
        "n=%d, backend=%s, devices=%d",
        args.n,
        default_backend(),
        local_device_count(),
    )

    rng_key = random.key(0)
    rng_key, rng_subkey = random.split(rng_key)
    p_nn_module = MLP(
        din=NUM_FEATURES,
        dmid=D_HIDDEN,
        dout=1,
        rngs=nnx.Rngs(params=rng_subkey),
    )

    def model(X: Array, *, y: Array | None = None) -> None:
        nn_p = random_nnx_module(
            "nn_p",
            nn_module=p_nn_module,
            scope_divider="_",
            prior=dist.Normal(),
        )
        with plate("data", size=X.shape[0]):
            p = nn_p(X)
            sample("y", dist.Bernoulli(p), obs=y)

    rng_key, rng_subkey = random.split(rng_key)
    X_train, y_train = generate_synthetic_data(rng_subkey, n=BATCH_SIZE, p=NUM_FEATURES)

    logger.info("Fitting model...")
    guide = AutoNormal(model=model, init_loc_fn=init_to_uniform(radius=0.1))
    svi = SVI(model, guide=guide, optim=adam(learning_rate=1e-3), loss=Trace_ELBO())
    rng_key, rng_subkey = random.split(rng_key)
    svi_result = svi.run(
        rng_subkey,
        num_steps=NUM_STEPS,
        progress_bar=False,
        X=X_train,
        y=y_train,
    )
    guide_predictive = Predictive(
        guide,
        params=svi_result.params,
        num_samples=NUM_SAMPLES,
    )
    rng_key, rng_subkey = random.split(rng_key)
    posterior_samples = device_get(guide_predictive(rng_subkey))
    predictive = Predictive(model, posterior_samples=posterior_samples, parallel=True)

    logger.info("Dry run on training data...")
    rng_key, rng_subkey = random.split(rng_key)
    X_train = jax.device_put(X_train, devices(default_backend())[0])
    out = predictive(rng_subkey, X_train)
    jax.block_until_ready(out)

    logger.info("Generating synthetic data...")
    rng_key, rng_subkey = random.split(rng_key)
    X, _ = generate_synthetic_data(rng_subkey, n=args.n, p=NUM_FEATURES)
    # Predictive requires the full array on device — limits maximum n.
    X = jax.device_put(X, devices(default_backend())[0])

    logger.info("Measuring peak incremental RSS...")
    rng_key, rng_subkey = random.split(rng_key)
    delta = _peak_rss_increment(
        lambda: device_get(predictive(rng_subkey, X)),
    )
    logger.info("Peak incremental RSS: %.1f MiB", delta)


if __name__ == "__main__":
    main()
