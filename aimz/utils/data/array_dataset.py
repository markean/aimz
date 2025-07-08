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

"""Module for custom dataset for JAX arrays."""


import numpy as np
from jax import tree
from jax.typing import ArrayLike
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    """Custom Dataset class for JAX arrays based on PyTorch's Dataset."""

    def __init__(self, *arrays: ArrayLike) -> None:
        """Initialize an ArrayDataset instance.

        Args:
            *arrays: One or more JAX arrays or compatible array-like objects.

        Raises:
            ValueError: If no arrays are provided or if the arrays do not have the same
                length.
        """
        if not arrays:
            msg = "At least one array must be provided."
            raise ValueError(msg)
        length = len(arrays[0])
        if any(len(arr) != length for arr in arrays):
            msg = "All arrays must have the same length."
            raise ValueError(msg)
        # Convert inputs to NumPy arrays for efficient CPU-based batching and slicing.
        # Keeping data as NumPy arrays until batch collation speeds up data loading
        # and reduces overhead from host-to-device data transfers.
        self.arrays = tuple(np.asarray(arr) for arr in arrays)

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            The number of samples, equal to the length of the first array.
        """
        # Since all arrays have the same length, return the length of the first one
        return len(self.arrays[0])

    def __getitem__(self, index: int) -> object:
        """Retrieve the elements at the specified index.

        Args:
            index: Index of the item to retrieve.

        Returns:
            A pytree containing the elements at the given index from each array.
        """
        return tree.map(lambda x: x[index], self.arrays)
