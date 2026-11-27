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

"""Scenarios: editing the inputs of one and comparing two against the causal card."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from warnings import warn

import numpy as np
from jax import tree
from jax.typing import ArrayLike

from aimz._exceptions import _SKIP_FILE_PREFIXES, ScopeWarning
from aimz.utils._kwargs import _is_per_observation

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet

    from aimz.model.causal_card import CausalCard
    from aimz.model.kernel_spec import KernelSpec


def _build_scenario(
    args: Mapping[str, object],
    changes: Mapping[str, object],
    *,
    observations: ArrayLike | None,
    card: CausalCard | None,
    kernel_spec: KernelSpec | None,
    own: AbstractSet[str],
) -> dict[str, object]:
    """Build a scenario, as :meth:`~aimz.ImpactModel.build_scenario` documents.

    ``own`` names the parameters of the predictive methods themselves, which are no
    inputs of the kernel.
    """
    if not isinstance(args.get("X"), ArrayLike):
        msg = (
            "`build_scenario` edits an array passed under `X`, whatever the "
            "kernel names its input; a data loader has no observations."
        )
        raise TypeError(msg)
    X = np.asarray(args["X"])
    n_obs = X.shape[0]
    features = card.features if card is not None else ()
    if features and (X.ndim <= 1 or len(features) != X.shape[1]):
        msg = (
            f"The card names {len(features)} feature(s), but the input has shape "
            f"{X.shape}."
        )
        raise ValueError(msg)
    # The inputs by name: the input, its features, and the keyword arrays
    inputs: dict[str, tuple[str, int | None]] = {"X": ("X", None)}
    if X.ndim > 1:
        for j in range(X.shape[1]):
            inputs[features[j] if features else f"X[{j}]"] = ("X", j)
    inputs.update(
        (name, (name, None))
        for name, value in args.items()
        if name not in own and isinstance(value, ArrayLike) and np.ndim(value)
    )
    index = None
    if observations is not None:
        index = np.asarray(observations)
        fits = (
            index.shape == (n_obs,)
            if index.dtype == bool
            else index.ndim == 1
            and np.issubdtype(index.dtype, np.integer)
            and bool(((-n_obs <= index) & (index < n_obs)).all())
        )
        if not fits:
            msg = (
                f"`observations` must be a boolean mask over the {n_obs} "
                "observations or integer indices among them."
            )
            raise ValueError(msg)
    treatments = card.treatments if card is not None else {}
    spec = kernel_spec
    sites = set(spec.sample_sites) if spec is not None and spec.traced else set()
    edited: dict[str, np.ndarray] = {}
    for name, change in changes.items():
        for each in treatments.get(name, (name,)):
            if each not in inputs:
                # The treatments whose inputs can all be edited, then the inputs
                names = [t for t in treatments if {*treatments[t]} <= inputs.keys()]
                names += inputs
                msg = (
                    f"{each!r} is a sample site, which `intervention` sets."
                    if each in sites
                    else f"{each!r} is none of the inputs a scenario can edit, "
                    f"which hold an array in `args`: {', '.join(map(repr, names))}."
                )
                raise ValueError(msg)
            argument, column = inputs[each]
            if argument not in edited:
                edited[argument] = np.array(args[argument])
            array = edited[argument]
            current = array if column is None else array[:, column]
            value = np.asarray(change(current) if callable(change) else change)
            try:
                new = np.broadcast_to(value, current.shape)
            except ValueError:
                msg = (
                    f"The value for {each!r} has shape {value.shape}, which does "
                    f"not broadcast to {current.shape}."
                )
                raise ValueError(msg) from None
            if index is None:
                current[...] = new
            elif array.shape[0] != n_obs:
                msg = (
                    f"`observations` selects among the {n_obs} observations, but "
                    f"{argument!r} has no value per observation."
                )
                raise ValueError(msg)
            else:
                current[index] = new[index]

    return {**args, **edited}


def _check_scenario(
    args_baseline: Mapping[str, object],
    args_intervention: Mapping[str, object],
    *,
    by: Mapping[str, ArrayLike] | None,
    pattern: Iterable[str] | Mapping[str, Iterable[str]] | None,
    absent: Mapping[str, object] | None,
    card: CausalCard | None,
    own: AbstractSet[str],
) -> dict[str, object]:
    """Compare two scenarios, as :meth:`~aimz.ImpactModel.check_scenario` documents.

    ``own`` names the parameters of the predictive methods themselves, which are no
    inputs of the kernel.
    """
    if not all(
        isinstance(args.get("X"), ArrayLike)
        for args in (args_baseline, args_intervention)
    ):
        msg = (
            "`check_scenario` compares arrays passed under `X`, whatever the "
            "kernel names its input; a data loader has no observations."
        )
        raise TypeError(msg)
    X_b, X_i = np.asarray(args_baseline["X"]), np.asarray(args_intervention["X"])
    if X_b.shape != X_i.shape:
        msg = (
            f"The scenarios' inputs differ in shape, {X_b.shape} against "
            f"{X_i.shape}; a scenario compares the same observations."
        )
        raise ValueError(msg)
    n_obs = X_b.shape[0]
    features = card.features if card is not None else ()
    if features and (X_b.ndim <= 1 or len(features) != X_b.shape[1]):
        msg = (
            f"The card names {len(features)} feature(s), but the input has shape "
            f"{X_b.shape}."
        )
        raise ValueError(msg)

    # Per-observation pairs by name, in the order a call lets them stand
    stacked: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    if X_b.ndim > 1:
        for j in range(X_b.shape[1]):
            stacked[features[j] if features else f"X[{j}]"] = [
                (X_b[:, j].reshape(n_obs, -1), X_i[:, j].reshape(n_obs, -1)),
            ]
    else:
        stacked["X"] = [(X_b.reshape(n_obs, -1), X_i.reshape(n_obs, -1))]
    columns = tuple(stacked)
    pairs = [
        (name, args_baseline.get(name), args_intervention.get(name))
        for name in sorted({*args_baseline, *args_intervention} - own)
    ]
    for name in ("condition", "intervention"):
        mappings = [
            cast("Mapping[str, object]", args.get(name) or {})
            for args in (args_baseline, args_intervention)
        ]
        pairs.extend(
            (site, mappings[0].get(site), mappings[1].get(site))
            for site in sorted({*mappings[0], *mappings[1]})
        )
    whole: dict[str, bool] = {}
    for name, a, b in pairs:
        if a is None or b is None:
            differs = a is not b or (name in args_baseline) != (
                name in args_intervention
            )
        # A distribution's `shape` is a method, which no array's is
        elif (
            not callable(getattr(a, "shape", None))
            and not callable(getattr(b, "shape", None))
            and _is_per_observation(a, n_obs)
            and _is_per_observation(b, n_obs)
            and np.asarray(a).shape == np.asarray(b).shape
        ):
            stacked.setdefault(name, []).append(
                (
                    np.asarray(a).reshape(n_obs, -1),
                    np.asarray(b).reshape(n_obs, -1),
                ),
            )
            continue
        else:
            differs = a is not b and (
                tree.structure(a) != tree.structure(b)
                or not all(map(np.array_equal, tree.leaves(a), tree.leaves(b)))
            )
        # A change as a whole voids the pairs read before it, a feature's own apart
        whole[name] = whole.get(name, False) or differs
        if differs and name in stacked:
            stacked[name] = stacked[name][:1] if name in columns else []

    values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    changed: dict[str, float] = {}
    masks: dict[str, np.ndarray] = {}
    any_changed = np.zeros(n_obs, dtype=bool)
    for name, group in stacked.items():
        union = np.zeros(n_obs, dtype=bool)
        for a, b in group:
            neq = a != b
            if np.issubdtype(a.dtype, np.floating) and np.issubdtype(
                b.dtype,
                np.floating,
            ):
                neq &= ~(np.isnan(a) & np.isnan(b))
            rows = neq.any(axis=1)
            union |= rows
            # The last pair that differs stands for the name, else the last
            if rows.any() or name not in masks:
                values[name] = (a, b)
            if rows.any():
                masks[name] = rows
        if union.any():
            changed[name] = float(union.mean())
            any_changed |= union
    for name, differs in whole.items():
        if differs:
            changed[name] = 1.0
            any_changed[:] = True

    treatments = card.treatments if card is not None else {}
    spanned = {name for names in treatments.values() for name in names}
    undeclared = sorted(
        name
        for name in changed
        if spanned
        and name not in spanned
        and not ("X" in spanned and (name.startswith("X[") or name in features))
    )
    extrapolation: dict[str, float] = {}
    for name, rows in masks.items():
        a, b = values[name]
        if a.shape[1] != 1 or not np.issubdtype(a.dtype, np.number):
            continue
        new = b[rows, 0]
        extrapolation[name] = float(
            ((new < np.nanmin(a)) | (new > np.nanmax(a))).mean(),
        )

    per_obs = {name for name, (a, _) in values.items() if a.shape[1] == 1}
    if pattern is None:
        groups = {
            group: tuple(
                name
                for each in names
                for name in (columns if each == "X" else (each,))
                if name in per_obs
            )
            for group, names in treatments.items()
        }
        groups = {group: names for group, names in groups.items() if names} or {
            name: (name,) for name in changed if name in per_obs
        }
    elif hasattr(pattern, "items"):
        groups = {
            str(group): tuple(names)
            for group, names in cast(
                "Mapping[str, Iterable[str]]",
                pattern,
            ).items()
        }
    else:
        groups = {name: (name,) for name in cast("Iterable[str]", pattern)}
    zero = dict(absent or {})
    unknown = sorted(
        name
        for name in {*zero, *(name for names in groups.values() for name in names)}
        if name not in per_obs
    )
    if unknown:
        msg = (
            f"Name(s) {', '.join(map(repr, unknown))} in `pattern` or `absent` "
            "are not inputs with one value per observation in both scenarios: "
            f"{', '.join(map(repr, sorted(per_obs)))}."
        )
        raise ValueError(msg)
    # As a Python float, a float64 is compared in the input's precision
    zero = {
        name: float(array)
        if (array := np.asarray(value)).ndim == 0 and array.dtype == np.float64
        else value
        for name, value in zero.items()
    }
    support = None
    if groups and any_changed.any():
        packed = [
            np.packbits(
                np.stack(
                    [
                        np.any(
                            [
                                values[name][side][:, 0] != zero.get(name, 0)
                                for name in names
                            ],
                            axis=0,
                        )
                        for names in groups.values()
                    ],
                    axis=1,
                ),
                axis=1,
            )
            for side in (0, 1)
        ]
        void = np.dtype((np.void, packed[0].shape[1]))
        keys_b = np.ascontiguousarray(packed[0]).view(void).ravel()
        keys_i = np.ascontiguousarray(packed[1]).view(void).ravel()[any_changed]
        index = np.unique(np.concatenate([keys_b, keys_i]), return_inverse=True)[1]
        index_b, index_i = index[: len(keys_b)], index[len(keys_b) :]
        count_b = np.bincount(index_b, minlength=index.max() + 1)
        support = float((count_b[index_i] > 0).mean())

    post = sorted(
        name for name in (by or {}) if card is not None and name in card.post_treatment
    )
    if undeclared:
        msg = (
            f"Changed input(s) {', '.join(map(repr, undeclared))} are not among "
            f"the card's treatments ({', '.join(map(repr, treatments))}); the "
            "scenario is outside the declared causal scope."
        )
        warn(msg, category=ScopeWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
    if post:
        msg = (
            f"Grouping by {', '.join(map(repr, post))}, which the card declares "
            "post-treatment; a contrast within these groups is descriptive."
        )
        warn(msg, category=ScopeWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)

    return {
        "n_obs": int(n_obs),
        "changed": changed,
        "obs_changed": float(any_changed.mean()) if n_obs else 0.0,
        "undeclared": undeclared,
        "extrapolation": extrapolation,
        "pattern": {group: list(names) for group, names in groups.items()},
        "support": support,
        "post_treatment": post,
    }
