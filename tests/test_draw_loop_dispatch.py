# Copyright 2025 Eli Lilly and Company
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the backend-dependent draw loop (`lax.map` on CPU, `vmap` on GPU/TPU)."""

import numpy as np
import pytest
from jax import Array, jit, random

import aimz.sampling._forward as forward_mod
import aimz.utils._log_likelihood as loglik_mod
from tests.conftest import lm

NUM_DRAWS = 7


def _draws(n_feat: int = 3) -> dict:
    return {
        "w": random.normal(random.key(1), (NUM_DRAWS, n_feat)),
        "b": random.normal(random.key(2), (NUM_DRAWS,)),
        "sigma": np.abs(np.asarray(random.normal(random.key(3), (NUM_DRAWS,)))) + 0.5,
    }


def _run_forward(X: Array) -> np.ndarray:
    # A fresh jit wrapper per call: the backend dispatch happens at trace time, so a
    # shared wrapper would replay the first call's compiled program after the
    # monkeypatch.
    f = jit(
        lambda keys, samples: forward_mod._sample_forward(
            lm,
            rng_keys=keys,
            return_sites=("y",),
            samples=samples,
            model_kwargs={"X": X},
        ),
    )

    return np.asarray(f(random.split(random.key(0), NUM_DRAWS), _draws())["y"])


def _run_loglik(X: Array, y: np.ndarray) -> np.ndarray:
    f = jit(
        lambda samples: loglik_mod._log_likelihood(
            lm,
            samples=samples,
            model_kwargs={"X": X, "y": y},
        ),
    )

    return np.asarray(f(_draws())["y"])


def test_sample_forward_accelerator_dispatch_matches_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vectorized accelerator draw loop matches the CPU sequential loop."""
    X = random.normal(random.key(4), (20, 3))
    out_cpu = _run_forward(X)
    monkeypatch.setattr(forward_mod, "default_backend", lambda: "gpu")
    out_accel = _run_forward(X)
    np.testing.assert_allclose(out_accel, out_cpu, rtol=1e-5, atol=1e-5)


def test_log_likelihood_accelerator_dispatch_matches_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vectorized accelerator log-likelihood matches the CPU sequential loop."""
    X = random.normal(random.key(4), (20, 3))
    y = np.asarray(X @ np.ones(3))
    out_cpu = _run_loglik(X, y)
    monkeypatch.setattr(loglik_mod, "default_backend", lambda: "gpu")
    out_accel = _run_loglik(X, y)
    np.testing.assert_allclose(out_accel, out_cpu, rtol=1e-5, atol=1e-5)
