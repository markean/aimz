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

"""Module for :class:`~aimz.utils.data.ArrayLoader`."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING
from warnings import warn

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax, random

from aimz._exceptions import AimzWarning

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

    from aimz.utils.data.array_dataset import ArrayDataset

# Jitted, as an eager slice of a JAX array dispatches anew for every batch
_slice_rows = jit(lax.dynamic_slice_in_dim, static_argnums=2)


class ArrayLoader:
    """Data loader yielding mappings of named arrays.

    Shuffling advances ``rng_key`` on each iteration. Without shuffling, a NumPy batch
    is a view of the data, so writing into it changes the dataset. Model methods handle
    device placement and any padding required for sharding.
    """

    def __init__(
        self,
        dataset: ArrayDataset,
        rng_key: Array,
        *,
        batch_size: int = 32,
        shuffle: bool = False,
    ) -> None:
        """Initialize an ArrayLoader instance.

        Args:
            dataset: The dataset to load.
            rng_key: A typed key array from :external:func:`jax.random.key`, used only
                when ``shuffle`` is ``True``.
            batch_size: The number of samples per batch.
            shuffle: Whether to shuffle the dataset before batching.

        Raises:
            ValueError: If ``batch_size`` is not a positive integer.

        Warns:
            AimzWarning: If ``rng_key`` is a legacy ``uint32`` key, which is converted
                to a typed key array.
        """
        self.dataset = dataset
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            msg = f"`batch_size` should be a positive integer, but got {batch_size!r}."
            raise ValueError(msg)
        self.batch_size = batch_size
        self.shuffle = shuffle
        if isinstance(rng_key, Array) and rng_key.dtype == jnp.uint32:
            msg = "Legacy `uint32` PRNGKey detected; converting to a typed key array."
            # Not `skip_file_prefixes`: skipping the aimz frames would blame MLflow's
            # autolog wrapper, which then hides the warning
            warn(msg, category=AimzWarning, stacklevel=2)
            rng_key = random.wrap_key_data(rng_key)
        self.rng_key = rng_key

    def __iter__(self) -> Iterator[dict[str, Array | npt.NDArray]]:
        """Yield the batches of one pass over the dataset, shuffled if asked."""
        n = len(self.dataset)
        order = None
        if self.shuffle:
            self.rng_key, subkey = random.split(self.rng_key)
            seed = np.asarray(random.key_data(subkey))
            order = np.random.default_rng(seed).permutation(n)
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            if order is not None:
                rows = order[start:stop]
                yield {
                    k: jnp.take(arr, rows, axis=0)
                    if isinstance(arr, Array)
                    else arr[rows]
                    for k, arr in self.dataset.arrays.items()
                }
            else:
                yield {
                    k: _slice_rows(arr, start, stop - start)
                    if isinstance(arr, Array) and stop - start < n
                    else arr[start:stop]
                    for k, arr in self.dataset.arrays.items()
                }

    def __len__(self) -> int:
        """Return the number of batches."""
        return ceil(len(self.dataset) / self.batch_size)
