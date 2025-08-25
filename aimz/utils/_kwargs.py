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

"""Module for processing keyword arguments for sharding."""

from typing import NamedTuple

from jax.typing import ArrayLike
from sklearn.utils.validation import _is_arraylike


def _group_kwargs(kwargs: dict) -> tuple[NamedTuple, NamedTuple]:
    """Separate keyword arguments into array-like and non-array-like groups.

    Args:
        kwargs (dict): A dictionary of keyword arguments where values could be
            array-like or non-array-like.

    Returns:
        tuple[KwargsArray, KwargsExtra]: A tuple containing two ``NamedTuple`` objects:
            - kwargs_array (KwargsArray): Contains the array-like arguments.
            - kwargs_extra (KwargsExtra): Contains the non-array-like arguments.
    """
    dict_kwargs_array = {k: v for k, v in kwargs.items() if _is_arraylike(v)}
    dict_kwargs_extra = {k: v for k, v in kwargs.items() if not _is_arraylike(v)}

    # Dynamically create NamedTuple classes
    KwargsArray = NamedTuple(
        "KwargsArray",
        [(k, ArrayLike) for k in dict_kwargs_array],
    )
    KwargsExtra = NamedTuple(
        "KwargsExtra",
        [(k, object) for k in dict_kwargs_extra],
    )

    return KwargsArray(**dict_kwargs_array), KwargsExtra(**dict_kwargs_extra)
