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

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from jax import Array, tree

from aimz.utils._validation import _is_arraylike

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from jax.typing import ArrayLike

# Prefix of the reserved batch-field names that carry per-observation intervention
# values alongside the input, so they are batched, padded, and sharded with it.
_INTERVENTION_PREFIX = "__do__"

# Placeholder for the traced (array) leaves in the static part of a partition.
_TRACED = object()


def _is_per_observation(value: object, n_obs: int | None) -> bool:
    """Return whether a value is an array aligned with the observation axis.

    Args:
        value: The value to check.
        n_obs: The number of observations in an array input, or ``None`` when the input
            is a data loader, whose batches carry the per-observation arrays.

    Returns:
        ``True`` if the value is an at least 1-D array-like with ``n_obs`` entries on
        its leading axis.
    """
    if n_obs is None or not _is_arraylike(value):
        return False
    arr = cast("ArrayLike", value)

    return np.ndim(arr) >= 1 and np.shape(arr)[0] == n_obs


def _group_kwargs(
    kwargs: dict,
    n_obs: int | None = None,
    forbid: tuple[str, ...] = (),
) -> tuple[dict, dict]:
    """Separate keyword arguments into per-observation arrays and call constants.

    Per-observation arrays are batched and sharded with the input. Every other value is
    a constant of the call, passed whole to every batch, including an array whose
    leading axis does not match the input.

    Args:
        kwargs: A dictionary of keyword arguments.
        n_obs: The number of observations in an array input, or ``None`` when the input
            is a data loader, whose batches carry the per-observation arrays.
        forbid: Names that must not appear in ``kwargs``. Reserved parameter names whose
            values are supplied through dedicated arguments instead.

    Returns:
        A tuple containing two dictionaries:
            - kwargs_array: Contains the per-observation arrays.
            - kwargs_extra: Contains the constants of the call.

    Raises:
        TypeError: If a forbidden name appears in ``kwargs``.
    """
    for name in forbid:
        if name in kwargs:
            msg = (
                f"{name!r} is a reserved kernel parameter and cannot be passed as a "
                "keyword argument."
            )
            raise TypeError(msg)
    kwargs_array = {k: v for k, v in kwargs.items() if _is_per_observation(v, n_obs)}
    kwargs_extra = {k: v for k, v in kwargs.items() if k not in kwargs_array}

    return kwargs_array, kwargs_extra


def _partition(kwargs: Mapping[str, object]) -> tuple[list, tuple]:
    """Split call constants into traced array leaves and a hashable static part.

    JAX and NumPy arrays, including NumPy scalars, are traced, and every other leaf is
    held static, so it can set shapes or drive Python control flow as it does in an
    uncompiled call.

    Args:
        kwargs: The call constants by name.

    Returns:
        The array leaves in order, and the static part that :func:`_combine` uses to
        rebuild the constants from them.

    Raises:
        TypeError: If a constant holds a leaf that is neither an array nor hashable.
    """
    leaves_traced, static = [], []
    for name, value in kwargs.items():
        leaves, treedef = tree.flatten(value)
        template = tuple(
            _TRACED if isinstance(leaf, (Array, np.ndarray, np.generic)) else leaf
            for leaf in leaves
        )
        try:
            hash(template)
        except TypeError:
            msg = (
                f"Keyword argument {name!r} must hold arrays or hashable values, such "
                "as numbers and strings."
            )
            raise TypeError(msg) from None
        leaves_traced.extend(
            leaf for leaf, slot in zip(leaves, template, strict=True) if slot is _TRACED
        )
        static.append((name, treedef, template))

    return leaves_traced, tuple(static)


def _combine(leaves: Iterable[object], static: tuple) -> dict:
    """Rebuild the call constants from their array leaves and static part.

    Args:
        leaves: The array leaves returned by :func:`_partition`, possibly traced.
        static: The static part returned by :func:`_partition`.

    Returns:
        The call constants by name.
    """
    leaves = iter(leaves)

    return {
        name: tree.unflatten(
            treedef,
            [next(leaves) if slot is _TRACED else slot for slot in template],
        )
        for name, treedef, template in static
    }


def _split_intervention(
    intervention: dict | None,
    n_obs: int | None,
) -> tuple[dict, dict]:
    """Separate per-observation intervention values from replicated constants.

    Args:
        intervention: A dictionary mapping sample site names to replacement values, or
            ``None``.
        n_obs: The number of observations in an array input, or ``None`` when the input
            is a data loader.

    Returns:
        A tuple containing two dictionaries:
            - constants: The replicated values keyed by site name.
            - fields: The per-observation values keyed by reserved batch-field name.
    """
    constants, fields = {}, {}
    for site, value in (intervention or {}).items():
        if _is_per_observation(value, n_obs):
            fields[_INTERVENTION_PREFIX + site] = value
        else:
            constants[site] = value

    return constants, fields


def _split_intervention_fields(
    kwargs: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Separate the reserved per-observation intervention fields from model kwargs.

    Args:
        kwargs: Keyword arguments bound by name, possibly including reserved fields.

    Returns:
        A tuple containing two dictionaries:
            - model_kwargs: The arguments passed to the model.
            - intervention: The per-observation values keyed by sample site name.
    """
    model_kwargs = {
        name: value
        for name, value in kwargs.items()
        if not name.startswith(_INTERVENTION_PREFIX)
    }
    intervention = {
        name.removeprefix(_INTERVENTION_PREFIX): value
        for name, value in kwargs.items()
        if name.startswith(_INTERVENTION_PREFIX)
    }

    return model_kwargs, intervention
