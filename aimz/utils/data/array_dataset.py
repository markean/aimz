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

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


class ArrayDataset:
    """ArrayDataset class for JAX arrays."""

    def __init__(self, *, to_jax: bool = True, **arrays: ArrayLike | None) -> None:
        """Initialize an ArrayDataset instance.

        Args:
            to_jax (bool): Whether to convert the input arrays to JAX arrays.
            **arrays (ArrayLike): One or more JAX arrays or compatible array-like
                objects.

        Raises:
            ValueError: If no arrays are provided or if the arrays do not have the same
                length.
        """
        arrays = {k: v for k, v in arrays.items() if v is not None}
        if not arrays:
            msg = "At least one array must be provided."
            raise ValueError(msg)
        lengths = {len(arr) for arr in arrays.values()}
        if len(lengths) != 1:
            msg = "All arrays must have the same length."
            raise ValueError(msg)
        (self.length,) = lengths
        if to_jax:
            self.arrays = {k: jnp.asarray(v) for k, v in arrays.items()}
        else:
            self.arrays = arrays

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            The number of samples.
        """
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Array]:
        """Retrieve the elements at the specified index.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            A tuple containing the elements from each array at the specified index.
        """
        return {k: v[idx] for k, v in self.arrays.items()}
