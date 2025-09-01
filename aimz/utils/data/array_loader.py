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

"""Module for custom data loader with padding logic for JAX arrays.

This module defines a custom ArrayLoader that processes batches of data and applies
padding to ensure the batch size is compatible with sharding across multiple XLA
devices.
"""

from collections.abc import Iterator
from math import ceil
from typing import TYPE_CHECKING
from warnings import warn

import jax.numpy as jnp
from jax import Array, device_put, local_device_count, random
from jax.typing import ArrayLike

from aimz.utils.data.array_dataset import ArrayDataset

if TYPE_CHECKING:
    from jax.sharding import Sharding
    from numpy import ndarray


class ArrayLoader:
    """ArrayLoader class for JAX arrays."""

    def __init__(
        self,
        dataset: ArrayDataset,
        rng_key: ArrayLike,
        *,
        batch_size: int = 32,
        shuffle: bool = False,
        device: "Sharding | None" = None,
    ) -> None:
        """Initialize an ArrayLoader instance.

        Args:
            dataset (ArrayDataset): The dataset to load.
            rng_key (ArrayLike): A pseudo-random number generator key.
            batch_size (int): The number of samples per batch.
            shuffle (bool, optional): Whether to shuffle the dataset before batching.
            device (Sharding | None, optional): The device or sharding specification to
                which the data should be moved. By default, no device transfer is
                applied. When used as an input to a model, this will be overridden by
                the device setting of the model.
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
        self.indices = jnp.arange(len(self.dataset))
        if isinstance(rng_key, Array) and rng_key.dtype == jnp.uint32:
            msg = "Legacy `uint32` PRNGKey detected; converting to a typed key array."
            warn(msg, category=UserWarning, stacklevel=2)
            rng_key = random.wrap_key_data(rng_key)
        self.rng_key = rng_key
        self._num_devices = local_device_count()
        self.device = device

    def pad_array(self, x: "Array | ndarray", n_pad: int, axis: int = 0) -> Array:
        """Pad an array to ensure compatibility with sharding.

        Args:
            x (Array | ndarray): The input array to be padded.
            n_pad (int): The number of padding elements to add.
            axis (int): The axis along which to apply the padding.

        Returns:
            The padded array with padding applied along the specified axis.

        Raises:
            ValueError: If padding is requested along an unsupported axis for a 1D
                array.
        """
        if x.ndim == 1:
            if axis == 0:
                return jnp.pad(x, pad_width=(0, n_pad), mode="edge")
            msg = "Padding 1D arrays is only supported along axis 0."
            raise ValueError(msg)

        # Initialize all axes with no padding
        pad_width: list[tuple[int, int]] = [(0, 0)] * x.ndim
        # Apply padding to the specified axis
        pad_width[axis] = (0, n_pad)

        return jnp.pad(x, pad_width=pad_width, mode="edge")

    def __iter__(self) -> Iterator[tuple[dict[str, Array], int]]:
        """Iterate over the dataset in batches.

        Yields:
            A batch of arrays with data from the dataset.
            The number of padded samples added for sharding compatibility.
        """
        indices = self.indices
        if self.shuffle:
            self.rng_key, subkey = random.split(self.rng_key)
            indices = random.permutation(subkey, self.indices)
        for start in range(0, len(self.dataset), self.batch_size):
            end = start + self.batch_size
            batch_idx = indices[start:end]
            if self.device is not None:
                n_pad = (-len(batch_idx)) % self._num_devices
                batch = {
                    k: self.pad_array(arr[batch_idx], n_pad=n_pad)
                    for k, arr in self.dataset.arrays.items()
                }
                batch = {k: device_put(v, self.device) for k, v in batch.items()}
            else:
                n_pad = 0
                batch = {k: arr[batch_idx] for k, arr in self.dataset.arrays.items()}
            yield batch, n_pad

    def __len__(self) -> int:
        """Return the number of batches.

        Returns:
            The total number of batches.
        """
        return ceil(len(self.dataset) / self.batch_size)
