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
from collections.abc import Iterable, Mapping
from os import cpu_count
from typing import TYPE_CHECKING
from warnings import warn

import jax.numpy as jnp
import numpy as np
from jax import Array, device_put
from jax.typing import ArrayLike

from aimz._exceptions import _SKIP_FILE_PREFIXES, OutputWarning
from aimz.utils._kwargs import _group_kwargs
from aimz.utils._output import _WRITER_COUNT_MAX
from aimz.utils.data import ArrayDataset, ArrayLoader

if TYPE_CHECKING:
    from jax.sharding import Sharding

logger = logging.getLogger(__name__)


def _prepare_batch(
    batch: object,
    *,
    param_input: str,
    device: Sharding | None = None,
    size: int = 0,
) -> tuple[dict[str, Array | np.ndarray], int]:
    """Validate a batch of named arrays, and pad and place it for inference.

    Returns:
        The prepared mapping and the number of valid observations, padding excluded.

    Raises:
        TypeError: If the batch is not a mapping of named NumPy or JAX arrays.
        ValueError: If the input is missing, empty, scalar, or misaligned.
    """
    if not isinstance(batch, Mapping):
        msg = "Data loaders must yield mappings of parameter names to NumPy/JAX arrays."
        raise TypeError(msg)

    if param_input not in batch:
        msg = f"The data loader has no field named {param_input!r} for the model input."
        raise ValueError(msg)

    arrays: dict[str, Array | np.ndarray] = {}
    for name, arr in batch.items():
        if not isinstance(name, str) or not isinstance(arr, (Array, np.ndarray)):
            msg = "Batch fields must have string names and NumPy/JAX array values."
            raise TypeError(msg)
        if arr.ndim == 0 or arr.shape[0] == 0:
            msg = f"Batch field {name!r} must have a nonempty observation axis."
            raise ValueError(msg)
        arrays[name] = arr
    n_valid = arrays[param_input].shape[0]
    if any(arr.shape[0] != n_valid for arr in arrays.values()):
        msg = "All batch fields must have the same observation-axis size."
        raise ValueError(msg)

    # Rows up to `size`, then up to a multiple of the device count
    n_rows = max(n_valid, size)
    n_rows += -n_rows % (1 if device is None else device.num_devices)
    n_pad = n_rows - n_valid
    for name, value in arrays.items():
        arr = value
        if n_pad:
            pad = jnp.pad if isinstance(arr, Array) else np.pad
            arr = pad(arr, [(0, n_pad), *[(0, 0)] * (arr.ndim - 1)], mode="edge")
        arrays[name] = arr if device is None else device_put(arr, device)

    return arrays, n_valid


# Soft memory budget per batch or chunk; the element cap derives from it by the
# output's item size, so it tracks the float precision
MAX_BYTES = 100_000_000
# Batches per writer thread that automatic batching targets
_BATCHES_PER_WRITER = 4
# Floor on the output bytes of an automatic batch: a smaller output is not I/O-bound
# and stays whole
_BATCH_BYTES_MIN = 4 * 1024 * 1024


def _resolve_batch_size(
    batch_size: int | None,
    axis_size: int,
    other_size: int,
    num_devices: int,
    item_nbytes: int | None = None,
) -> int:
    """Resolve the batch size along the chunked axis of a two-axis output.

    An explicit ``batch_size`` is used as given. Otherwise each batch stays within
    :data:`MAX_BYTES`, the axis splits into enough batches to occupy the writer pool,
    and no batch is smaller than :data:`_BATCH_BYTES_MIN`, the memory budget taking
    precedence; the result is rounded down to a multiple of ``num_devices`` and
    clamped to the axis.

    Args:
        batch_size: The requested batch size, or ``None`` to resolve a default.
        axis_size: Length of the chunked axis, observations or draws.
        other_size: Length of the axis held whole within each step.
        num_devices: Number of devices the chunked axis is sharded across.
        item_nbytes: Output bytes per index along the chunked axis, or ``None`` for a
            single value per element.
    """
    if batch_size is not None:
        return batch_size

    # The budgets are resolved against JAX's default float type
    itemsize = jnp.result_type(float).itemsize
    other_size = max(1, other_size)

    cap = MAX_BYTES // max(1, item_nbytes or itemsize * other_size)
    target_batches = _BATCHES_PER_WRITER * min(cpu_count() or 1, _WRITER_COUNT_MAX)
    target = -(-axis_size // target_batches)
    floor = _BATCH_BYTES_MIN // itemsize // other_size

    resolved = max(min(max(target, floor), cap), num_devices)

    return min(max(resolved - resolved % num_devices, num_devices), axis_size)


def _fits_single_batch(axis_size: int, item_nbytes: int) -> bool:
    """Return whether the whole axis fits :data:`MAX_BYTES` as one batch."""
    return axis_size * item_nbytes < MAX_BYTES


def _setup_inputs(
    *,
    X: ArrayLike | ArrayLoader | Iterable[Mapping[str, Array | np.ndarray]],
    y: ArrayLike | None,
    param_input: str,
    param_output: str,
    rng_key: Array,
    batch_size: int | None,
    shuffle: bool = False,
    **kwargs: object,
) -> tuple[Iterable[Mapping[str, Array | np.ndarray]], dict]:
    """Return the data loader and the call constants of a streamed or fitting call.

    An array input is wrapped in an :class:`~aimz.utils.data.ArrayLoader` keyed by the
    kernel's parameter names; a data loader is used as is.

    Raises:
        TypeError: If ``y`` is passed with a data loader, or ``X`` is neither an array
            nor a data loader.
        ValueError: If ``X`` or ``y`` is 0-D.

    Warns:
        OutputWarning: If an :class:`~aimz.utils.data.ArrayLoader` that shuffles is
            passed where the data order matters.
    """
    kwargs_array, kwargs_extra = _group_kwargs(
        kwargs,
        n_obs=(
            np.shape(X)[0] if isinstance(X, ArrayLike) and np.ndim(X) >= 1 else None
        ),
        forbid=(param_input, param_output),
    )

    if isinstance(X, ArrayLike):
        X = np.asarray(X)
        if X.ndim == 0:
            msg = "`X` must have at least 1 dimension."
            raise ValueError(msg)
        y = np.asarray(y) if y is not None else None
        if y is not None and y.ndim == 0:
            msg = "`y` must have at least 1 dimension."
            raise ValueError(msg)
        if batch_size is None:
            batch_size = len(X)
        kwargs_array[param_input] = X
        kwargs_array[param_output] = y
        loader = ArrayLoader(
            ArrayDataset(**kwargs_array),
            rng_key=rng_key,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    elif isinstance(X, Iterable) and not isinstance(X, (str, bytes, Mapping)):
        if y is not None:
            msg = "`y` must be `None` when `X` is already a data loader."
            raise TypeError(msg)
        if isinstance(X, ArrayLoader) and X.shuffle and not shuffle:
            msg = (
                "The data loader shuffles, so results will not follow the data order. "
                "Create the ArrayLoader with `shuffle=False` to preserve it."
            )
            warn(msg, category=OutputWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
        loader = X
    else:
        msg = f"`X` must be an array-like or a data loader, got {type(X).__name__!r}."
        raise TypeError(msg)

    return loader, kwargs_extra
