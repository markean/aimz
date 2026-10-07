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

"""Module for :class:`~aimz.utils.data.ArrayDataset`."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np
from jax import Array

if TYPE_CHECKING:
    from collections.abc import Sized

    import numpy.typing as npt


class ArrayDataset:
    """Dataset of named arrays, kept as given unless ``to_jax`` converts them."""

    def __init__(
        self,
        *,
        to_jax: bool = False,
        **arrays: npt.ArrayLike | None,
    ) -> None:
        """Initialize an ArrayDataset instance.

        Args:
            to_jax: Whether to convert the arrays to JAX arrays.
            **arrays: Named arrays, or ``None`` to leave a name out; an array-like
                other than a JAX or NumPy array becomes a NumPy array.

        Raises:
            ValueError: If no array is given, or the arrays differ in length.
        """
        self.arrays = {
            k: v if isinstance(v, (Array, np.ndarray)) else np.asarray(v)
            for k, v in arrays.items()
            if v is not None
        }
        if not self.arrays:
            msg = "At least one array must be provided."
            raise ValueError(msg)
        lengths = {len(cast("Sized", arr)) for arr in self.arrays.values()}
        if len(lengths) != 1:
            msg = "All arrays must have the same leading-axis size."
            raise ValueError(msg)
        (self.length,) = lengths
        if to_jax:
            self.arrays = {k: jnp.asarray(v) for k, v in self.arrays.items()}

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Array | npt.NDArray | np.generic]:
        """Return the element of each array at ``idx``, by name."""
        return {k: v[idx] for k, v in self.arrays.items()}
