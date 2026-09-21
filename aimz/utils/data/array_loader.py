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
from jax import Array, random

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

    from aimz.utils.data.array_dataset import ArrayDataset


class ArrayLoader:
    """Data loader yielding mappings of named arrays.

    Shuffling advances ``rng_key`` on each iteration. Model methods handle device
    placement and any padding required for sharding.
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
            rng_key: A pseudo-random number generator key.
            batch_size: The number of samples per batch.
            shuffle: Whether to shuffle the dataset before batching.
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
        self.indices = np.arange(len(self.dataset))
        if isinstance(rng_key, Array) and rng_key.dtype == jnp.uint32:
            msg = "Legacy `uint32` PRNGKey detected; converting to a typed key array."
            warn(msg, category=UserWarning, stacklevel=2)
            rng_key = random.wrap_key_data(rng_key)
        self.rng_key = rng_key

    def __iter__(self) -> Iterator[dict[str, Array | npt.NDArray]]:
        """Iterate over the dataset in batches.

        Yields:
            A batch of arrays with data from the dataset.
        """
        indices = self.indices
        if self.shuffle:
            self.rng_key, subkey = random.split(self.rng_key)
            seed = np.asarray(random.key_data(subkey))
            indices = np.random.default_rng(seed).permutation(len(self.dataset))
        for start in range(0, len(self.dataset), self.batch_size):
            end = start + self.batch_size
            batch_idx = indices[start:end]
            yield {
                k: arr[batch_idx]
                for k, arr in self.dataset.arrays.items()
                if arr is not None
            }

    def __len__(self) -> int:
        """Return the number of batches.

        Returns:
            The total number of batches.
        """
        return ceil(len(self.dataset) / self.batch_size)
