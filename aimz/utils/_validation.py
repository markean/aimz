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

"""Module for validating models and objects."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from inspect import Parameter, signature
from typing import TYPE_CHECKING
from warnings import warn

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from aimz._exceptions import (
    _SKIP_FILE_PREFIXES,
    KernelValidationError,
    NotFittedError,
    OutputWarning,
)

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable, Iterable
    from pathlib import Path

    import xarray as xr

    from aimz import ImpactModel
    from aimz.model.kernel_spec import KernelSpec
    from aimz.utils.data import ArrayLoader


def _is_arraylike(x: object) -> bool:
    """Returns whether the input is array-like."""
    if isinstance(x, (str, bytes, bytearray, Mapping)):
        return False

    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


def _check_is_fitted(model: ImpactModel) -> None:
    """Check if the model is fitted.

    Raises:
        NotFittedError: If the model has not been fitted.
    """
    if not model.is_fitted():
        msg = (
            f"This {type(model).__name__} instance is not fitted yet. Call `.fit()` "
            "or `.fit_on_batch()` with appropriate arguments before using the model."
        )
        raise NotFittedError(msg)


def _validate_group(dt_baseline: xr.DataTree, dt_intervention: xr.DataTree) -> str:
    """Return the predictive group the two scenarios share, checking they compare.

    The group is the first of ``predictions``, ``posterior_predictive``, and
    ``prior_predictive`` in ``dt_baseline``.

    Raises:
        ValueError: If the group is missing from either tree.

    Warns:
        OutputWarning: If the group's dimension sizes or coordinate labels differ
            between the scenarios, or the scenarios hold different posterior samples.
    """
    group = next(
        (
            name
            for name in ("predictions", "posterior_predictive", "prior_predictive")
            if name in dt_baseline.children
        ),
        "posterior_predictive",
    )

    if group not in dt_baseline.children:
        msg = (
            f"Group {group!r} not found in `dt_baseline`. Available "
            f"groups: {', '.join(map(repr, dt_baseline.children))}"
        )
        raise ValueError(msg)

    if group not in dt_intervention.children:
        msg = (
            f"Group {group!r} not found in `dt_intervention`. Available "
            f"groups: {', '.join(map(repr, dt_intervention.children))}"
        )
        raise ValueError(msg)

    if dict(dt_baseline[group].sizes) != dict(dt_intervention[group].sizes):
        msg = (
            f"Baseline and intervention have different dimension sizes in group "
            f"{group!r}: {dict(dt_baseline[group].sizes)} vs "
            f"{dict(dt_intervention[group].sizes)}."
        )
        warn(msg, category=OutputWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)
    elif unmatched := [
        dim
        for dim, index in dt_baseline[group].indexes.items()
        if dim in dt_intervention[group].indexes
        and not index.difference(dt_intervention[group].indexes[dim]).empty
    ]:
        msg = (
            f"Baseline and intervention have different coordinate labels along "
            f"{', '.join(map(repr, unmatched))} in group {group!r}; the effect "
            "covers only the labels present in both."
        )
        warn(msg, category=OutputWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)

    # Draws under the prior do not come from the posterior samples
    if (
        group != "prior_predictive"
        and "posterior" in dt_baseline.children
        and "posterior" in dt_intervention.children
        and not dt_baseline.children["posterior"].equals(
            dt_intervention.children["posterior"],
        )
    ):
        msg = "Baseline and intervention have different posterior samples."
        warn(msg, category=OutputWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)

    return group


def _validate_intervention(
    intervention: dict | None,
    kernel_spec: KernelSpec | None,
) -> None:
    """Raise if an intervened site is not a sample site of a traced kernel.

    Raises:
        ValueError: If ``intervention`` names a site that is not a sample site.
    """
    if intervention is None or kernel_spec is None or not kernel_spec.traced:
        return
    unknown = [site for site in intervention if site not in kernel_spec.sample_sites]
    if unknown:
        msg = (
            f"Intervention site(s) not among the kernel's sample sites: "
            f"{', '.join(map(repr, unknown))}. Sample sites: "
            f"{', '.join(map(repr, kernel_spec.sample_sites))}."
        )
        raise ValueError(msg)


def _warn_unreachable_intervention(
    intervention: dict | None,
    *,
    output: str,
    parents: Mapping[str, AbstractSet[str]],
    fixed: AbstractSet[str],
) -> None:
    """Warn for the intervened sites whose values cannot reach the output.

    Warns:
        OutputWarning: If an intervened site reaches the output only through sites
            whose values are fixed or intervened on.
    """
    if not intervention:
        return
    # Walk up from the output once through every site and once stopping at the fixed
    # and intervened sites; a site only the first walk meets is blocked
    reached = []
    for stop in (frozenset(), fixed | intervention.keys()):
        seen = {output}
        frontier = [output]
        edges = set()
        while frontier:
            for parent in parents.get(frontier.pop(), ()):
                edges.add(parent)
                if parent not in seen and parent not in stop:
                    seen.add(parent)
                    frontier.append(parent)
        reached.append(edges)
    if blocked := sorted(intervention.keys() & (reached[0] - reached[1])):
        msg = (
            f"Intervention site(s) {', '.join(map(repr, blocked))} reach the output "
            f"site {output!r} only through sites whose values are taken from the "
            "posterior or intervened on, so the draws do not respond to them."
        )
        warn(msg, category=OutputWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)


def _validate_shard_axis(
    shard_axis: str,
    X: ArrayLike | ArrayLoader | Iterable[Mapping[str, Array | np.ndarray]],
) -> None:
    """Raise for an unknown ``shard_axis``, or ``"draw"`` with a data loader.

    Raises:
        ValueError: If ``shard_axis`` is not ``"obs"`` or ``"draw"``.
        TypeError: If ``shard_axis="draw"`` is used with a data loader ``X``.
    """
    if shard_axis not in ("obs", "draw"):
        msg = f"`shard_axis` must be either 'obs' or 'draw', got {shard_axis!r}."
        raise ValueError(msg)
    if shard_axis == "draw" and not isinstance(X, ArrayLike):
        msg = (
            "`shard_axis='draw'` replicates the whole input across devices, "
            "so `X` must be an array, not a data loader."
        )
        raise TypeError(msg)


def _validate_store(store: str, output_dir: str | Path | None) -> None:
    """Raise for an unknown ``store``, or an ``output_dir`` with the memory store.

    Raises:
        ValueError: If ``store`` is not ``"persistent"`` or ``"memory"``, or an
            ``output_dir`` is passed with ``store="memory"``.
    """
    if store not in ("persistent", "memory"):
        msg = f"`store` must be either 'persistent' or 'memory', got {store!r}."
        raise ValueError(msg)
    if store == "memory" and output_dir is not None:
        msg = "`output_dir` must be `None` when `store='memory'`."
        raise ValueError(msg)


def _validate_batch_size(
    batch_size: int | None,
    X: ArrayLike | ArrayLoader | Iterable[Mapping[str, Array | np.ndarray]],
) -> None:
    """Raise for an explicit ``batch_size`` of an array input that is not positive.

    Raises:
        ValueError: If ``batch_size`` is not a positive integer.
    """
    if not isinstance(X, ArrayLike) or batch_size is None:
        return
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        msg = f"`batch_size` should be a positive integer, but got {batch_size!r}."
        raise ValueError(msg)


def _validate_aligned_inputs(
    X: ArrayLike | ArrayLoader | Iterable[Mapping[str, Array | np.ndarray]],
    y: ArrayLike | None,
) -> None:
    """Raise for an array ``X`` or ``y`` that is 0-D, empty, or misaligned.

    A data loader is skipped, and keyword arguments are not checked: an array whose
    leading axis differs is a constant of the call.

    Raises:
        ValueError: If ``X`` or ``y`` is 0-D, ``X`` is empty, or ``y`` does not share
            ``X``'s leading-axis size.
    """
    if not isinstance(X, ArrayLike):
        return
    inputs: dict[str, ArrayLike] = {"X": X}
    if y is not None:
        inputs["y"] = y

    sizes: dict[str, int] = {}
    for name, arr in inputs.items():
        if np.ndim(arr) == 0:
            msg = f"`{name}` must have at least 1 dimension."
            raise ValueError(msg)
        sizes[name] = np.shape(arr)[0]
    if sizes["X"] == 0:
        msg = "`X` must not be empty."
        raise ValueError(msg)
    if len(set(sizes.values())) > 1:
        detail = ", ".join(f"{name}={size}" for name, size in sizes.items())
        msg = f"All inputs must have the same leading-axis size; got {detail}."
        raise ValueError(msg)


def _validate_kernel_signature(
    kernel: Callable,
    param_input: str,
    param_output: str,
) -> None:
    """Raise for a kernel signature that aimz cannot call.

    Raises:
        KernelValidationError: If the kernel takes variable arguments, lacks the input
            or output parameter, gives the input a default, or gives the output a
            default other than ``None``.
    """
    params = signature(kernel).parameters
    if any(
        p.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
        for p in params.values()
    ):
        msg = "Kernel must not accept variable arguments (*args or **kwargs)."
        raise KernelValidationError(msg)

    param_main = [arg for arg in (param_input, param_output) if arg not in params]
    if param_main:
        sub = ", ".join(map(repr, param_main))
        msg = (
            f"Kernel must accept {sub} as argument(s). Modify the kernel signature or "
            "set `param_input` and `param_output` accordingly."
        )
        raise KernelValidationError(msg)

    if params[param_input].default is not Parameter.empty:
        sub = param_input
        msg = f"{sub!r} must not have a default value."
        raise KernelValidationError(msg)
    if params[param_output].default is not None:
        sub = param_output
        msg = f"{sub!r} must have a default value of `None`."
        raise KernelValidationError(msg)


def _validate_kernel_body(
    kernel: Callable,
    *,
    param_output: str,
    model_trace: OrderedDict[str, dict],
    with_output: bool,
) -> None:
    """Raise for a trace without a usable output site.

    Raises:
        KernelValidationError: If a site name holds ``/``, the output site is missing,
            is not an observed sample site when ``with_output`` is set, or a kernel
            parameter shares a site's name.
    """
    invalid_site = [site for site in model_trace if "/" in site]
    if invalid_site:
        msg = (
            f"Invalid site names containing '/': {invalid_site!r}. "
            "xarray.DataTree does not allow '/' in variable names."
        )
        raise KernelValidationError(msg)

    if param_output not in model_trace:
        msg = (
            f"Kernel must include a sample or deterministic site named "
            f"{param_output!r}."
        )
        raise KernelValidationError(msg)
    site = model_trace[param_output]
    if with_output:
        if site["type"] != "sample":
            msg = (
                f"Expected {param_output!r} to have type 'sample', got "
                f"{site['type']!r}."
            )
            raise KernelValidationError(msg)
        if not site.get("is_observed", False):
            msg = (
                f"{param_output!r} must be observed (i.e., defined with `obs=` in the "
                "kernel)."
            )
            raise KernelValidationError(msg)
    elif site["type"] not in ("sample", "deterministic"):
        msg = (
            f"Expected {param_output!r} to have type 'sample' or 'deterministic', "
            f"got {site['type']!r}."
        )
        raise KernelValidationError(msg)

    params = list(signature(kernel).parameters)
    params.remove(param_output)
    conflicts = set(params) & set(model_trace.keys())
    if conflicts:
        msg = (
            f"Kernel parameters conflict with model sites: "
            f"{', '.join(repr(k) for k in sorted(conflicts))}. "
            "Rename parameters or revise the model to avoid shadowing."
        )
        raise KernelValidationError(msg)


def _validate_X_y_to_jax(
    X: ArrayLike,
    y: ArrayLike | None = None,
) -> tuple[Array, Array] | Array:
    """Convert ``X`` and ``y`` to JAX arrays on their own devices, checking alignment.

    Returns:
        ``X`` alone when ``y`` is ``None``, else ``(X, y)``.

    Raises:
        ValueError: If ``X`` or ``y`` is 0-D, or ``y`` does not share ``X``'s
            leading-axis size.
    """
    device_x = X.device if isinstance(X, Array) and X.committed else None
    X = jnp.asarray(X, device=device_x)
    if X.ndim == 0:
        msg = "`X` must have at least 1 dimension."
        raise ValueError(msg)

    if y is None:
        return X

    device_y = y.device if isinstance(y, Array) and y.committed else None
    y = jnp.asarray(y, device=device_y)
    if y.ndim == 0:
        msg = "`y` must have at least 1 dimension."
        raise ValueError(msg)
    if len(X) != len(y):
        msg = "`X` and `y` must have the same leading-axis size."
        raise ValueError(msg)

    return X, y
