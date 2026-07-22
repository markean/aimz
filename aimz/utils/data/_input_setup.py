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

"""Module for initializing inputs and preprocessing arguments for data pipelines."""

from __future__ import annotations

import logging
from os import cpu_count
from typing import TYPE_CHECKING
from warnings import warn

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from aimz.utils._kwargs import _group_kwargs
from aimz.utils._output import _WRITER_COUNT_MAX
from aimz.utils.data import ArrayDataset, ArrayLoader

if TYPE_CHECKING:
    from jax import Array
    from jax.sharding import Sharding

logger = logging.getLogger(__name__)

# Soft memory budget per batch or chunk, in bytes. The element cap is derived at call
# time by dividing this by the output dtype's item size, so the budget tracks precision:
# ~100 MB whether the predictive output is float32 (default, 25M elements) or float64
# (under `jax_enable_x64`, 12.5M elements). Helps control memory and disk usage.
MAX_BYTES = 100_000_000
# Automatic batching targets this many batches per writer thread, so the writer pool
# has enough independent work items to run fully occupied with headroom for load
# balancing and the pipelined producer.
_BATCHES_PER_WRITER = 4
# Floor on the output bytes a batch produces under automatic batching. Splitting below
# this trades real I/O parallelism for per-chunk file and dispatch overhead: outputs
# too small to reach it are not I/O-bound in the first place, so they stay whole.
_BATCH_BYTES_MIN = 4 * 1024 * 1024


def _resolve_batch_size(
    batch_size: int | None,
    axis_size: int,
    other_size: int,
    num_devices: int,
) -> int:
    """Resolve the per-step batch size along the chunked axis of a 2-axis output.

    Each streamed step produces an output of ``batch_size * other_size`` elements:
    data-parallel chunks the observation axis (``other_size`` draws per observation),
    while draw-parallel chunks the draw axis (``other_size`` resident observations).
    An explicit ``batch_size`` is used as given — it is the caller's contract, even
    when it yields a single batch.

    Automatic resolution balances three concerns: each batch stays within the
    :data:`MAX_BYTES` memory budget; the axis is split into enough batches to keep the
    writer-thread pool occupied (:data:`_BATCHES_PER_WRITER` per writer); and no batch
    produces less than :data:`_BATCH_BYTES_MIN` of output, so tiny workloads — which
    are not I/O-bound — stay whole instead of paying per-chunk overhead. The result is
    rounded down to a multiple of ``num_devices`` (floored at ``num_devices``) and
    clamped to ``axis_size``.

    Args:
        batch_size: The requested batch size, or ``None`` to resolve a default.
        axis_size: Length of the axis being chunked (observations or draws).
        other_size: Length of the axis held whole within each step.
        num_devices: Number of devices the chunked axis is sharded across.

    Returns:
        The resolved batch size.
    """
    if batch_size is not None:
        return batch_size

    # Output chunks hold predictive samples in JAX's default float precision; resolve
    # the element budgets against that dtype so they track precision.
    itemsize = jnp.result_type(float).itemsize
    max_elements = MAX_BYTES // itemsize
    other_size = max(1, other_size)

    # Memory ceiling and pool-occupancy target per batch, and the output-size floor.
    cap = max_elements // other_size
    target_batches = _BATCHES_PER_WRITER * min(cpu_count() or 1, _WRITER_COUNT_MAX)
    target = -(-axis_size // target_batches)
    floor = _BATCH_BYTES_MIN // itemsize // other_size

    resolved = max(min(cap, target), floor, num_devices)

    return min(max(resolved - resolved % num_devices, num_devices), axis_size)


def _fits_single_batch(axis_size: int, other_size: int) -> bool:
    """Return whether the whole axis fits the per-batch memory budget as one batch.

    Mirrors the memory ceiling used by :func:`_resolve_batch_size`, independent of its
    pool-occupancy splitting policy. Callers that cannot tolerate a split axis (e.g. an
    observation-aligned posterior) use this to decide between pinning a whole-input
    batch and falling back to draw-parallel streaming.

    Args:
        axis_size: Length of the axis that would be chunked.
        other_size: Length of the axis held whole within each step.

    Returns:
        ``True`` if a single batch covering the whole axis stays within the budget.
    """
    return axis_size * other_size < MAX_BYTES // jnp.result_type(float).itemsize


def _setup_inputs(
    *,
    X: ArrayLike | ArrayLoader,
    y: ArrayLike | None,
    param_input: str,
    param_output: str,
    rng_key: Array,
    batch_size: int | None,
    num_samples: int,
    shuffle: bool = False,
    device: Sharding | None = None,
    **kwargs: object,
) -> tuple[ArrayLoader, dict]:
    """Prepare an dataloader and grouped keyword arguments.

    Args:
        X (ArrayLike | ArrayLoader): Input data. If array-like, the leading axis is
            the observation axis. Alternatively, a data loader that holds all array-like
            objects and handles batching internally.
        y (ArrayLike | None): Output data. The leading axis is the observation axis.
            Must be ``None`` if ``X`` is a data loader.
        param_input: Dataset key for ``X``, matching the kernel's input parameter so
            each batch is keyed as the downstream lookup expects.
        param_output: Dataset key for ``y``, matching the kernel's output parameter.
        rng_key: A pseudo-random number generator key.
        batch_size: The size of batches for data loading.
        num_samples: Number of samples to draw, which affects the size of batches.
        shuffle: Whether to shuffle the dataset before batching.
        device: The device or sharding specification to which the data should be moved.
            By default, no device transfer is applied. If ``X`` is a data loader, it
            will override the device setting of the loader.
        **kwargs: Additional arguments passed to the model.

    Returns:
        - The data loader for batching.
        - Extra keyword arguments to be passed downstream.
    """
    kwargs_array, kwargs_extra = _group_kwargs(kwargs)

    if isinstance(X, ArrayLike):
        X = np.asarray(X)
        if X.ndim == 0:
            msg = "`X` must have at least 1 dimension."
            raise ValueError(msg)
        if y is not None:
            y = np.asarray(y)
            if y.ndim == 0:
                msg = "`y` must have at least 1 dimension."
                raise ValueError(msg)
        num_devices = device.num_devices if device else 1
        if batch_size is None:
            batch_size = _resolve_batch_size(
                None,
                axis_size=len(X),
                other_size=num_samples,
                num_devices=num_devices,
            )
            logger.debug("Resolved batch_size=%d automatically.", batch_size)
        if batch_size % num_devices != 0:
            msg = (
                f"The `batch_size` ({batch_size}) is not divisible by the number of "
                f"devices ({num_devices}). Use a multiple of {num_devices} "
                "for optimal performance."
            )
            warn(msg, category=UserWarning, stacklevel=2)
        # Key the dataset by the kernel's input/output parameter names (alongside the
        # array kwargs) so each batch is keyed as the downstream lookup expects.
        kwargs_array[param_input] = X
        kwargs_array[param_output] = y
        loader = ArrayLoader(
            ArrayDataset(**kwargs_array),
            rng_key=rng_key,
            batch_size=batch_size,
            shuffle=shuffle,
            device=device,
        )
    elif isinstance(X, ArrayLoader):
        if y is not None:
            msg = "`y` must be `None` when `X` is already a data loader."
            raise TypeError(msg)
        loader = X
        loader.device = device
    else:
        msg = f"`X` must be an array-like or a data loader, got {type(X).__name__!r}."
        raise TypeError(msg)

    return loader, kwargs_extra
