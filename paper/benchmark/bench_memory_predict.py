"""Benchmark peak incremental RSS for aimz `predict`."""

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
import numpy as np
import numpy.typing as npt
import numpyro.distributions as dist
from flax import nnx
from jax import Array, default_backend, local_device_count, random
from numpyro import plate, sample
from numpyro.contrib.module import random_nnx_module
from numpyro.infer import SVI, Trace_ELBO, init_to_uniform
from numpyro.infer.autoguide import AutoNormal
from optax import adam

from aimz import ImpactModel
from aimz.utils.data import ArrayDataset, ArrayLoader

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
    with Path.open(f"/proc/{os.getpid()}/status") as f:
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
def generate_synthetic_data(
    rng: np.random.Generator,
    n: int,
    p: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Generate synthetic data with a nonlinear data generating process.

    Data is generated in NumPy to simulate a realistic workflow where input data resides
    in host memory. JAX conversion is deferred to `ArrayLoader` at batching time.
    """
    X = rng.standard_normal((n, p), dtype=np.float32)
    b1 = rng.standard_normal(p, dtype=np.float32)
    b2 = rng.standard_normal(p, dtype=np.float32)
    logits = np.sin(X @ b1) * np.tanh(X @ b2) + 0.5 * np.cos((X**2) @ b1)
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(np.int32)

    return X, y


# ── Main ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


def main() -> None:
    """Benchmark aimz ``predict`` peak incremental RSS."""
    parser = argparse.ArgumentParser(description="Benchmark aimz predict memory.")
    parser.add_argument("--n", type=int, required=True, help="Dataset size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Zarr output directory (default: temporary directory)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} | {levelname} | {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("aimz").setLevel(logging.WARNING)
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

    rng_np = np.random.default_rng(42)
    X_train, y_train = generate_synthetic_data(rng_np, n=BATCH_SIZE, p=NUM_FEATURES)

    rng_key, rng_subkey = random.split(rng_key)
    im = ImpactModel(
        model,
        rng_key=rng_subkey,
        inference=SVI(
            model,
            guide=AutoNormal(model, init_loc_fn=init_to_uniform(radius=0.1)),
            optim=adam(learning_rate=1e-3),
            loss=Trace_ELBO(),
        ),
    )

    logger.info("Fitting model...")
    im.fit_on_batch(
        X_train,
        y_train,
        num_steps=NUM_STEPS,
        num_samples=NUM_SAMPLES,
        progress=False,
    )

    logger.info("Dry run on training data...")
    im.predict(X_train, batch_size=BATCH_SIZE, progress=False)
    im.cleanup()

    logger.info("Generating synthetic data...")
    X, _ = generate_synthetic_data(rng_np, n=args.n, p=NUM_FEATURES)
    rng_key, rng_subkey = random.split(rng_key)
    loader = ArrayLoader(
        ArrayDataset(X=X, to_jax=False),
        rng_key=rng_subkey,
        batch_size=BATCH_SIZE,
    )

    logger.info("Measuring peak incremental RSS...")
    delta = _peak_rss_increment(
        lambda: im.predict(
            loader,
            batch_size=BATCH_SIZE,
            output_dir=args.output_dir,
            progress=False,
        ),
    )
    logger.info("Peak incremental RSS: %.1f MiB", delta)


if __name__ == "__main__":
    main()
