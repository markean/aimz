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

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array, lax
from numpyro.handlers import substitute, trace

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpyro._typing import Message


def _pin_subsample_indices(msg: Message) -> Array | None:
    """Pin the indices of a subsampling plate to ``arange``, for a seed-free trace.

    Each batch is passed through the model's arguments, so the indices only set shapes.
    A kernel that gathers rows from a closed-over array through them is not supported:
    the pinned indices select the leading rows.

    Raises:
        ValueError: If the batch is larger than the plate size.
    """
    if msg["type"] != "plate":
        return None
    size, subsample_size = msg["args"]
    if subsample_size is None or subsample_size == size:
        return None
    if subsample_size > size:
        err_msg = (
            f"Plate site {msg['name']!r} declares size={size}, but the evaluated "
            f"batch has {subsample_size} observations. Evaluate at most `size` "
            "observations per batch, or declare a plate size covering the data."
        )
        raise ValueError(err_msg)

    return jnp.arange(subsample_size)


def _substitute_latent(msg: Message, sample: dict[str, Array]) -> Array | None:
    """Return the posterior value of a latent sample site, else ``None``.

    Deterministic sites are recomputed and observed sites score the data, so neither
    takes a value from the posterior.
    """
    if msg["type"] == "deterministic" or msg.get("is_observed", False):
        return None

    return sample.get(msg["name"])


def _log_likelihood(
    model: Callable,
    samples: dict[str, Array] | None,
    params: dict[str, Array] | None,
    model_kwargs: Mapping[str, object] | None,
) -> dict[str, Array]:
    """Compute the log-likelihood at the observed sites for each posterior draw.

    Without ``samples`` the result has a single draw. The plate indices are pinned by
    :func:`_pin_subsample_indices`.

    Returns:
        The log-probability of each observed site, with the draws on the leading axis.
    """
    if params:
        model = substitute(model, data=params)

    def _loglik_one_sample(sample: dict[str, Array]) -> dict[str, Array]:
        pinned_model = substitute(model, substitute_fn=_pin_subsample_indices)
        substituted_model = (
            substitute(
                pinned_model,
                substitute_fn=partial(_substitute_latent, sample=sample),
            )
            if sample
            else pinned_model
        )
        model_trace = trace(substituted_model).get_trace(**(model_kwargs or {}))

        return {
            k: site["fn"].log_prob(site["value"])
            for k, site in model_trace.items()
            if site["type"] == "sample" and site["is_observed"]
        }

    if not samples:
        return {k: v[None, ...] for k, v in _loglik_one_sample({}).items()}

    return lax.map(_loglik_one_sample, xs=samples)
