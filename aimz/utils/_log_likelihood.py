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

import jax.numpy as jnp
from jax import Array, lax
from numpyro.handlers import substitute, trace

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpyro._typing import Message


def _pin_subsample_indices(msg: Message) -> Array | None:
    """Provide deterministic indices for subsample plate sites.

    Log-likelihood evaluation passes each data batch to the model explicitly. For
    kernels that consume the batch through the model's arguments, the indices of a
    ``numpyro.plate`` site with ``subsample_size`` set smaller than ``size`` only
    determine broadcasting shapes. Drawing them randomly would require an rng key that a
    bare (unseeded) kernel does not have; pinning them to ``arange`` keeps tracing
    seed-free.

    Kernels that instead gather rows from a closed-over full-size array via the
    plate's indices (``with plate(...) as idx``, or ``numpyro.subsample``) are not
    supported by batched evaluation: the pinned indices always select the leading
    rows.

    Args:
        msg: A NumPyro effect-handler site message.

    Returns:
        Deterministic indices for a subsample plate site, or ``None`` to leave the site
        unchanged.

    Raises:
        ValueError: If the batch is larger than the plate size, which would require
            out-of-range indices.
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


def _log_likelihood(
    model: Callable,
    samples: dict[str, Array] | None,
    model_kwargs: Mapping[str, object] | None,
) -> dict[str, Array]:
    """Compute per-site log-likelihood at observed sites for each posterior draw.

    Kernels that subsample inside a ``numpyro.plate`` (``subsample_size``) get
    deterministic plate indices (see :func:`_pin_subsample_indices`): the batch passed
    through the model's arguments is scored as-is, and the trace stays free of
    subsampling randomness.

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
        pinned_model = substitute(model, substitute_fn=_pin_subsample_indices)
        substituted_model = substitute(pinned_model, sample) if sample else pinned_model
        model_trace = trace(substituted_model).get_trace(**(model_kwargs or {}))

        return {
            k: site["fn"].log_prob(site["value"])
            for k, site in model_trace.items()
            if site["type"] == "sample" and site["is_observed"]
        }

    if not samples:
        return {k: v[None, ...] for k, v in _loglik_one_sample({}).items()}

    return lax.map(_loglik_one_sample, xs=samples)
