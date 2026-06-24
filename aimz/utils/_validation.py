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
from inspect import Parameter, getfullargspec, signature
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from aimz._exceptions import KernelValidationError, NotFittedError

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable

    import xarray as xr

    from aimz import ImpactModel
    from aimz.utils.data import ArrayLoader


def _is_arraylike(x: object) -> bool:
    """Returns whether the input is array-like."""
    if isinstance(x, (str, bytes, bytearray, Mapping)):
        return False

    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


def _check_is_fitted(model: ImpactModel, msg: str | None = None) -> None:
    """Check if the model is fitted.

    Raises:
        NotFittedError: If the model has not been fitted.
    """
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call ``.fit()`` with "
            "appropriate arguments before using the model."
        )
    if not _is_fitted(model):
        raise NotFittedError(msg % {"name": type(model).__name__})


def _is_fitted(model: ImpactModel) -> bool:
    if hasattr(model, "_is_fitted"):
        return model.is_fitted()

    return any(v.endswith("_") and not v.startswith("__") for v in vars(model))


def _validate_group(dt_baseline: xr.DataTree, dt_intervention: xr.DataTree) -> str:
    """Validate the groups in ``dt_baseline`` and ``dt_intervention``.

    Args:
        dt_baseline: Precomputed output for the baseline scenario.
        dt_intervention: Precomputed output for the intervention scenario.

    Returns:
        The group name (``predictions`` or ``posterior_predictive``).

    Raises:
        ValueError: If the chosen group is missing from ``dt_baseline`` or
            ``dt_intervention``.
    """
    group = (
        "predictions"
        if "predictions" in dt_baseline.children
        else "posterior_predictive"
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

    return group


def _validate_shard_axis(shard_axis: str, X: ArrayLike | ArrayLoader) -> None:
    """Validate a multi-device sharding strategy and its input compatibility.

    Checked before any ``shard_axis`` coercion so the contract holds regardless of
    posterior state: ``shard_axis="draw"`` replicates the whole input across devices,
    so a data loader (which batches internally) is rejected.

    Args:
        shard_axis: The sharding strategy to validate.
        X: Input data, used to enforce the draw-parallel array-only constraint.

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


def _validate_batch_size(batch_size: int | None, X: ArrayLike | ArrayLoader) -> None:
    """Validate an explicit ``batch_size`` for the streaming entry points.

    ``batch_size`` is ignored for a data loader (it batches internally) and ``None``
    means auto-resolve, so both are skipped. Otherwise it must be a positive integer.
    The draw path divides and steps by it; the data path requires the same via the
    ``ArrayLoader``.

    Args:
        batch_size: The requested batch size, ``None`` to auto-resolve.
        X: Input data.

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
    X: ArrayLike | ArrayLoader,
    y: ArrayLike | None,
    kwargs: dict,
) -> None:
    """Validate array inputs share one leading-axis size, for either parallel path.

    Called from the streaming entry points before any output directory is created. A
    data loader is skipped. For an array ``X``, then ``X``, ``y`` (if given), and every
    array-like kwarg must be at least 1-D and share ``X``'s leading-axis size.

    Args:
        X: Input data.
        y: Output data, or ``None``.
        kwargs: Additional arguments passed to the model; only array-like values are
            checked.

    Raises:
        ValueError: If any checked input is 0-D, or the inputs do not all share one
            leading-axis size.
    """
    if not isinstance(X, ArrayLike):
        return
    inputs: dict[str, ArrayLike] = {"X": X}
    if y is not None:
        inputs["y"] = y
    inputs.update({k: v for k, v in kwargs.items() if _is_arraylike(v)})

    sizes: dict[str, int] = {}
    for name, arr in inputs.items():
        if np.ndim(arr) == 0:
            msg = f"`{name}` must have at least 1 dimension."
            raise ValueError(msg)
        sizes[name] = np.shape(arr)[0]
    if len(set(sizes.values())) > 1:
        detail = ", ".join(f"{name}={size}" for name, size in sizes.items())
        msg = f"All inputs must have the same leading-axis size; got {detail}."
        raise ValueError(msg)


def _validate_kernel_signature(
    kernel: Callable,
    param_input: str,
    param_output: str,
) -> None:
    """Validate the signature of a kernel function.

    Args:
        kernel: The kernel function to validate.
        param_input: Name of the parameter in ``kernel`` corresponding to the input.
        param_output: Name of the parameter in ``kernel`` corresponding to the output.

    Raises:
        KernelValidationError: If the kernel signature does not meet the required
            constraints.
    """
    argspec = getfullargspec(kernel)
    if argspec.varargs is not None or argspec.varkw is not None:
        msg = "Kernel must not accept variable arguments (*args or **kwargs)."
        raise KernelValidationError(msg)

    param_main = [
        arg
        for arg in (param_input, param_output)
        if arg not in (argspec.args + argspec.kwonlyargs)
    ]
    if param_main:
        sub = ", ".join(map(repr, param_main))
        msg = (
            f"Kernel must accept {sub} as argument(s). Modify the kernel signature or "
            "set `param_input` and `param_output` accordingly."
        )
        raise KernelValidationError(msg)

    sig = signature(kernel)
    if sig.parameters[param_input].default is not Parameter.empty:
        sub = param_input
        msg = f"{sub!r} must not have a default value."
        raise KernelValidationError(msg)
    if sig.parameters[param_output].default:
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
    """Validate the body of a kernel function.

    Args:
        kernel: The kernel function to validate.
        param_output: Name of the parameter in ``kernel`` corresponding to the output.
        model_trace: The model trace containing the sites.
        with_output: Whether the kernel is expected to have observed output.

    Raises:
        KernelValidationError: If the kernel body does not meet the required
            constraints.
    """
    invalid_site = [site for site in model_trace if "/" in site]
    if invalid_site:
        msg = (
            f"Invalid site names containing '/': {invalid_site!r}. "
            "xarray.DataTree does not allow '/' in variable names."
        )
        raise KernelValidationError(msg)

    if param_output not in model_trace:
        msg = f"Kernel must include a sample site named {param_output!r}."
        raise KernelValidationError(msg)
    site = model_trace[param_output]
    if site["type"] != "sample":
        msg = f"Expected {param_output!r} to have type 'sample', got {site['type']!r}."
        raise KernelValidationError(msg)
    if with_output and not site.get("is_observed", False):
        msg = (
            f"{param_output!r} must be observed (i.e., defined with `obs=` in the "
            "kernel)."
        )
        raise KernelValidationError(msg)

    # Collect parameter names from the kernel signature, excluding the output parameter
    params = getfullargspec(kernel).args + getfullargspec(kernel).kwonlyargs
    params.remove(param_output)
    # Check for name conflicts between parameter names and model site names
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
    """Validate and convert data arrays to JAX arrays.

    Arrays are checked, converted, and placed on the same device as their originals
    when available.

    Args:
        X (ArrayLike): Input data. The leading axis is the observation axis.
        y (ArrayLike): Output data. The leading axis is the observation axis.

    Returns:
        Validated JAX arrays, returning ``X`` if only X is provided, or a tuple
        ``(X, y)`` otherwise.
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
