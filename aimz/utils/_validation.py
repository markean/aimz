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

from inspect import Parameter, getfullargspec, signature
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from sklearn.utils.validation import check_array, check_X_y

from aimz._exceptions import KernelValidationError, NotFittedError

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable

    import xarray as xr

    from aimz import ImpactModel


def _check_is_fitted(model: "ImpactModel", msg: str | None = None) -> None:
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


def _is_fitted(model: "ImpactModel") -> bool:
    if hasattr(model, "_is_fitted"):
        return model.is_fitted()

    return any(v.endswith("_") and not v.startswith("__") for v in vars(model))


def _validate_group(dt_baseline: "xr.DataTree", dt_intervention: "xr.DataTree") -> str:
    """Validate the groups in ``dt_baseline`` and ``dt_intervention``.

    Args:
        dt_baseline (xr.DataTree): Precomputed output for the baseline scenario.
        dt_intervention (xr.DataTree): Precomputed output for the intervention scenario.

    Returns:
        str: The group name (``predictions`` or ``posterior_predictive``).

    Raises:
        ValueError: If the group is not found in ``dt_intervention``.
    """
    group = (
        "predictions"
        if "predictions" in dt_baseline.children
        else "posterior_predictive"
    )

    if group not in dt_intervention.children:
        sub = group
        msg = (
            f"Group {sub!r} not found in `dt_intervention`. Available "
            f"groups: {', '.join(map(repr, dt_intervention.children))}"
        )
        raise ValueError(msg)

    return group


def _validate_kernel_signature(
    kernel: "Callable",
    param_input: str,
    param_output: str,
) -> None:
    """Validate the signature of a kernel function.

    Args:
        kernel (Callable): The kernel function to validate.
        param_input (str): Name of the parameter in ``kernel`` corresponding to the
            input.
        param_output (str): Name of the parameter in ``kernel`` corresponding to the
            output.

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
            "set `param_input` and param_output` accordingly."
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
    kernel: "Callable",
    param_output: str,
    model_trace: "OrderedDict[str, dict]",
) -> None:
    """Validate the body of a kernel function.

    Args:
        kernel (Callable): The kernel function to validate.
        param_output (str): Name of the parameter in ``kernel`` corresponding to the
            output.
        model_trace (OrderedDict[str, dict]): The model trace containing the sites.

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
    if not site.get("is_observed", False):
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
        X (ArrayLike): Input data with shape ``(n_samples_X, n_features)``.
        y (ArrayLike): Output data with shape ``(n_samples_Y,)``.

    Returns:
        Validated JAX arrays, returning ``X`` if only X is provided, or a tuple
        ``(X, y)`` otherwise.
    """
    if y is None:
        device_x = X.device if isinstance(X, Array) and X.committed else None

        return jnp.asarray(check_array(X), device=device_x)

    device_x, device_y = (
        arr.device if isinstance(arr, Array) and arr.committed else None
        for arr in (X, y)
    )
    X, y = check_X_y(X, y, force_writeable=True, y_numeric=True)

    return jnp.asarray(X, device=device_x), jnp.asarray(y, device=device_y)
