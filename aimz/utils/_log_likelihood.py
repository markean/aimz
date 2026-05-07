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

"""Module for computing log-likelihoods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax import Array, lax
from numpyro.handlers import substitute, trace

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def _log_likelihood(
    model: Callable,
    samples: dict[str, Array] | None,
    model_kwargs: Mapping[str, object] | None,
) -> dict[str, Array]:
    """Compute per-site log-likelihood at observed sites for each posterior draw.

    Args:
        model: A probabilistic model with NumPyro primitives.
        samples: A dictionary of posterior samples to substitute into the model, where
            each array has leading axis equal to the number of draws. ``None`` or an
            empty dict yields a single-draw result with a leading axis of size 1.
        model_kwargs: Arguments passed to the model. Must include the input and the
            output values keyed under the model's parameter names.

    Returns:
        A dictionary mapping each observed sample site to its log-probability array
        with leading axis equal to the number of posterior draws (or 1 when ``samples``
        is empty or ``None``).
    """

    def _loglik_one_sample(sample: dict[str, Array]) -> dict[str, Array]:
        substituted_model = substitute(model, sample) if sample else model
        model_trace = trace(substituted_model).get_trace(**(model_kwargs or {}))
        return {
            k: site["fn"].log_prob(site["value"])
            for k, site in model_trace.items()
            if site["type"] == "sample" and site["is_observed"]
        }

    if not samples:
        return {k: v[None, ...] for k, v in _loglik_one_sample({}).items()}

    return lax.map(_loglik_one_sample, xs=samples)
