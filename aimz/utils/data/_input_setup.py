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
from typing import TYPE_CHECKING, NamedTuple, cast
from warnings import warn

from jax.typing import ArrayLike
from sklearn.utils.validation import check_array, check_X_y

from aimz.utils._kwargs import _group_kwargs
from aimz.utils.data import ArrayDataset, ArrayLoader

if TYPE_CHECKING:
    from collections.abc import Sized

    from jax.sharding import Sharding

logger = logging.getLogger(__name__)

# Maximum number of elements allowed per batch or chunk. Each element is assumed to be
# 4 bytes (float32 or int32). `MAX_ELEMENTS = 25_000_000` corresponds to ~100 MB per
# chunk, helping to control disk and memory usage during data processing.
MAX_ELEMENTS = 25_000_000


def _setup_inputs(
    *,
    X: ArrayLike | ArrayLoader,
    y: ArrayLike | None,
    rng_key: ArrayLike,
    batch_size: int | None,
    num_samples: int,
    shuffle: bool = False,
    device: Sharding | None = None,
    **kwargs: object,
) -> tuple[ArrayLoader, NamedTuple]:
    """Prepare an dataloader and grouped keyword arguments.

    Args:
        X (ArrayLike | ArrayLoader): Input data, either an array-like of shape
            ``(n_samples, n_features)`` or a data loader that holds all array-like
            objects and handles batching internally; if a data loader is passed,
            ``rng_key``, ``batch_size`` and ``shuffle`` are ignored.
        y (ArrayLike | None): Output data with shape ``(n_samples_Y,)``. Must be
            ``None`` if ``X`` is a data loader.
        rng_key (ArrayLike): A pseudo-random number generator key.
        batch_size: The size of batches for data loading.
        num_samples: Number of samples to draw, which affects the size of batches.
        shuffle: Whether to shuffle the dataset before batching.
        device: The device or sharding specification to which the data should be moved.
            By default, no device transfer is applied. If ``X`` is a data loader, it
            will override the device setting of the loader.
        **kwargs: Additional arguments passed to the model. All array-like values are
            expected to be JAX arrays.

    Returns:
        - The data loader for batching.
        - Extra keyword arguments to be passed downstream.
    """
    kwargs_array, kwargs_extra = _group_kwargs(kwargs)

    if isinstance(X, ArrayLike):
        if y is None:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y, force_writeable=True, y_numeric=True)
        num_devices = device.num_devices if device else 1
        if batch_size is None:
            if len(cast("Sized", X)) * num_samples < MAX_ELEMENTS:
                batch_size = len(cast("Sized", X))
            else:
                batch_size = MAX_ELEMENTS // num_samples
                batch_size -= batch_size % num_devices
            logger.info(
                "Using batch_size=%d. Specify explicitly to better control memory "
                "usage.",
                batch_size,
            )
        if batch_size % num_devices != 0:
            msg = (
                f"The `batch_size` ({batch_size}) is not divisible by the number of "
                f"devices ({num_devices}). Use a multiple of {num_devices} "
                "for optimal performance."
            )
            warn(msg, category=UserWarning, stacklevel=2)
        loader = ArrayLoader(
            ArrayDataset(X=cast("ArrayLike", X), y=y, **kwargs_array._asdict()),
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

    return loader, kwargs_extra
