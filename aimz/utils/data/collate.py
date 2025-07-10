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

"""Module for collate functions for batching data."""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def jax_collate(batch: list[tuple[ArrayLike]]) -> tuple[Array]:
    """Collate function that stacks and returns a tuple of JAX arrays.

    Args:
        batch (list[tuple[ArrayLike]]): A list of tuples, where each tuple contains
        elements that can be converted to JAX arrays (e.g., numpy arrays).

    """
    transposed = list(zip(*batch, strict=True))

    return tuple(jnp.stack(field) for field in transposed)
