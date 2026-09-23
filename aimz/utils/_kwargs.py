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

from typing import TYPE_CHECKING

import numpy as np

from aimz.utils._validation import _is_arraylike

if TYPE_CHECKING:
    from collections.abc import Mapping

# Prefix of the reserved batch-field names that carry per-observation intervention
# values alongside the input, so they are batched, padded, and sharded with it.
_INTERVENTION_PREFIX = "__do__"


def _group_kwargs(
    kwargs: dict,
    forbid: tuple[str, ...] = (),
) -> tuple[dict, dict]:
    """Separate keyword arguments into array-like and non-array-like groups.

    Args:
        kwargs: A dictionary of keyword arguments where values could be array-like or
            non-array-like.
        forbid: Names that must not appear in ``kwargs``. Reserved parameter names whose
            values are supplied through dedicated arguments instead.

    Returns:
        A tuple containing two dictionaries:
            - kwargs_array: Contains the array-like arguments.
            - kwargs_extra: Contains the non-array-like arguments.

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
    kwargs_array = {k: v for k, v in kwargs.items() if _is_arraylike(v)}
    kwargs_extra = {k: v for k, v in kwargs.items() if not _is_arraylike(v)}

    return kwargs_array, kwargs_extra


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
        if np.ndim(value) >= 1 and np.shape(value)[0] == n_obs:
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
