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

"""Execution planning: the sharding strategy and the batch size of a streamed call."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast
from warnings import warn

import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct, eval_shape
from jax.typing import ArrayLike
from numpyro.handlers import do, seed, substitute, trace
from numpyro.infer import init_to_sample

from aimz._exceptions import _SKIP_FILE_PREFIXES, OutputWarning, PerformanceWarning
from aimz.utils._kwargs import (
    _combine,
    _group_kwargs,
    _partition,
    _split_intervention,
    _split_intervention_fields,
    _to_jax,
)
from aimz.utils.data import ArrayLoader
from aimz.utils.data._input_setup import _fits_single_batch, _resolve_batch_size

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sized
    from collections.abc import Set as AbstractSet

    from jax import Array

    from aimz.model.kernel_spec import KernelSpec

logger = logging.getLogger(__name__)


def _output_nbytes(
    site_sizes: Mapping[str, tuple[int, int]],
    return_sites: tuple[str, ...],
) -> tuple[int, int]:
    """Return the output bytes per draw, per observation and for the other sites.

    Each value counts at the width of the default float type, since the trace holds
    the observed data, whose type the output need not share. A site the trace did
    not record counts as one value per observation.
    """
    itemsize = jnp.result_type(float).itemsize
    sizes = [site_sizes.get(site, (1, 0)) for site in return_sites]

    return itemsize * sum(n for n, _ in sizes), itemsize * sum(n for _, n in sizes)


def _plan_execution(
    X: ArrayLike | ArrayLoader | Iterable[Mapping[str, Array | np.ndarray]],
    *,
    shard_axis: Literal["obs", "draw"],
    batch_size: int | None,
    num_samples: int,
    nbytes: tuple[int, int],
    posterior: dict[str, Array] | None,
    return_sites: tuple[str, ...] = (),
    kwargs: Mapping[str, object] | None = None,
    intervention: dict | None = None,
    condition: dict | None = None,
    kernel: Callable,
    rng_key: Array,
    param_input: str,
    param_output: str,
    kernel_spec: KernelSpec | None,
    coupled_sites: AbstractSet[str],
    observed_sites: AbstractSet[str],
    num_devices: int,
    plan_cache: dict[tuple, tuple[frozenset[str], bool, dict[str, tuple[int, ...]]]],
) -> tuple[Literal["obs", "draw"], int | None]:
    """Choose the sharding strategy and the batch size of a streamed call.

    A posterior site shaped ``(num_samples, n_obs, ...)`` cannot be split with the
    observation axis, nor can a kernel that computes across observations, nor a
    ``condition`` value with one entry per observation, which is passed whole: only
    a whole-input batch on a single device keeps them intact, so the call otherwise
    runs draw-parallel, with a warning. A call with such a ``condition`` value runs
    draw-parallel on a whole-input batch too, without a warning, since under
    ``"obs"`` the streamer can take a site set from a whole value for one without
    the observation axis. Deterministic posterior sites are exempt, since the kernel
    recomputes them, but observed ones are not, since a call without their data
    takes their draws, and neither is a site no trace records, which the kernel may
    sample under other arguments.

    Args:
        X: Input array, or a data loader, which batches itself.
        shard_axis: The requested sharding strategy.
        batch_size: The requested batch size, or ``None`` to choose it.
        num_samples: Number of draws the call produces.
        nbytes: Output bytes per draw, as :func:`_output_nbytes` returns them.
        posterior: The posterior samples the call conditions on, if any.
        return_sites: The sites the call returns; one without the observation axis
            for the call's arguments makes it run draw-parallel.
        kwargs: The call's other kernel arguments.
        intervention: The call's intervention.
        condition: The call's condition.
        kernel: The model's kernel.
        rng_key: The model's key, which seeds the abstract traces.
        param_input: Name of the kernel's parameter for the input data.
        param_output: Name of the kernel's parameter for the output data.
        kernel_spec: The model's kernel spec.
        coupled_sites: The sites at which the kernel computes across observations.
        observed_sites: The sample sites that a trace observed.
        num_devices: Number of devices the call runs on.
        plan_cache: The outcomes of the abstract traces of earlier calls, by the
            call's shapes and constants; a new outcome is added to it.

    Returns:
        The sharding strategy and the batch size, ``None`` for a data loader.
    """
    if not isinstance(X, ArrayLike):
        # A data loader's batches split the rows, except one array loader batch on
        # one device, and no rerun can keep them whole
        if coupled_sites and not (
            num_devices == 1
            and isinstance(X, ArrayLoader)
            and X.batch_size >= len(X.dataset)
        ):
            msg = (
                "The kernel computes across observations at site(s) "
                f"{', '.join(map(repr, sorted(coupled_sites)))}, which the "
                "batches of a data loader split. Pass the input as an array, which "
                "then runs under `shard_axis='draw'`."
            )
            warn(
                msg,
                category=OutputWarning,
                skip_file_prefixes=_SKIP_FILE_PREFIXES,
            )
        return shard_axis, None

    n_obs = len(cast("Sized", X))
    row_nbytes, rest_nbytes = nbytes
    min_aligned_ndim = 2
    spec = cast("KernelSpec", kernel_spec)
    deterministic = set(spec.return_sites) - set(spec.sample_sites)
    conditioned = sorted(
        site for site, v in (condition or {}).items() if np.shape(v)[:1] == (n_obs,)
    )
    local = sorted(
        site
        for site, v in (posterior or {}).items()
        if site not in deterministic
        and v.ndim >= min_aligned_ndim
        and v.shape[1] == n_obs
    )
    # Under `obs`, several devices or batches split the rows
    splits = num_devices > 1 or (
        batch_size < n_obs
        if batch_size is not None
        else not _fits_single_batch(n_obs, item_nbytes=num_samples * row_nbytes)
    )
    if shard_axis == "obs" and (coupled_sites or conditioned or local):
        if splits:
            reason = (
                "The kernel computes across observations at site(s) "
                f"{', '.join(map(repr, sorted(coupled_sites)))}, which "
                "`shard_axis='obs'` would split"
                if coupled_sites
                else "The `condition` values of site(s) "
                f"{', '.join(map(repr, conditioned))} have one entry per "
                "observation and are passed whole, so `shard_axis='obs'` cannot "
                "split them"
                if conditioned
                else "The posterior site(s) "
                f"{', '.join(map(repr, local))} are as long as the number of "
                "observations in each draw, so `shard_axis='obs'` cannot split them"
            )
            msg = (
                f"{reason}; rerunning with `shard_axis='draw'`. Pass "
                "`shard_axis='draw'` to silence this warning."
            )
            warn(
                msg,
                category=PerformanceWarning,
                skip_file_prefixes=_SKIP_FILE_PREFIXES,
            )
            shard_axis, batch_size = "draw", None
        elif conditioned:
            # One batch keeps the rows whole, but under `obs` the streamer can
            # reject a site set from a whole `condition` value as having no
            # observation axis; `draw` keeps them whole and checks only the output
            shard_axis, batch_size = "draw", None
        elif batch_size is None:
            batch_size = n_obs
    # Posterior sites a call under `obs` replicates, deterministic and scalar ones
    # aside; their sizes matter only when the rows may be split or the batch is open
    latent = sorted(
        k
        for k, v in (posterior or {}).items()
        if k not in deterministic and v.ndim > 1 and (splits or batch_size is None)
    )
    if (return_sites or latent) and (
        shard_axis == "obs" or param_output in return_sites
    ):
        # A site without the observation axis, such as a global deterministic site,
        # has no rows to split, and a posterior site whose size follows the rows
        # without one value per row, such as a random walk's increments in a plate
        # of one row fewer, fits no device's rows. Abstract traces of this call, at
        # its rows and at one row fewer, as in a device batch, tell the sites that
        # follow the rows: an earlier trace may have had other arguments, and a
        # global site may match the row count. Under `draw`, which keeps the rows
        # whole and passes every argument whole, only the output is checked.
        n_split = n_obs if shard_axis == "obs" else None
        kwargs_rows, kwargs_const = _group_kwargs(kwargs or {}, n_obs=n_split)
        constants, fields = _split_intervention(intervention, n_split)
        leaves, static = _partition(kwargs_const)

        def values(rows: dict[str, object], leaves: list) -> dict[str, object]:
            model_kwargs, intervention_rows = _split_intervention_fields(rows)
            model = (
                do(
                    kernel,
                    data=cast("dict", {**_to_jax(constants), **intervention_rows}),
                )
                if intervention
                else kernel
            )
            model_trace = trace(
                substitute(
                    seed(model, rng_seed=rng_key),
                    # A site that already holds a value, as under `substitute`,
                    # keeps it
                    substitute_fn=lambda msg: (
                        None if msg["value"] is not None else init_to_sample(msg)
                    ),
                ),
            ).get_trace(**model_kwargs, **_combine(leaves, static))

            return {
                k: v["value"]
                for k, v in model_trace.items()
                if k in return_sites or k in latent
            }

        rows = {
            k: ShapeDtypeStruct(
                np.shape(v),
                jnp.result_type(v if hasattr(v, "dtype") else np.asarray(v)),
            )
            for k, v in {param_input: X, **kwargs_rows, **fields}.items()
        }

        def shapes_at(size: int) -> dict[str, ShapeDtypeStruct]:
            return eval_shape(
                values,
                {
                    k: ShapeDtypeStruct((size, *v.shape[1:]), v.dtype)
                    for k, v in rows.items()
                },
                leaves,
            )

        # Whether each device evaluates one row fewer than the input, where draws
        # that fit those rows would pass
        fewer_rows = not return_sites and -(
            -_resolve_batch_size(
                batch_size,
                axis_size=n_obs,
                other_size=num_samples,
                num_devices=num_devices,
                item_nbytes=num_samples * row_nbytes,
            )
            // num_devices
        ) == (n_obs - 1)
        # The traces depend only on the call's shapes and constants, and for a call
        # without return sites on the rows each device evaluates, so a repeated
        # call reuses their outcome
        key = (
            shard_axis,
            return_sites,
            tuple((k, v.shape, v.dtype) for k, v in rows.items()),
            tuple((np.shape(leaf), jnp.result_type(leaf)) for leaf in leaves),
            static,
            tuple(
                (
                    k,
                    np.shape(v),
                    jnp.result_type(v if hasattr(v, "dtype") else np.asarray(v)),
                )
                for k, v in sorted(constants.items())
            ),
            tuple(sorted(intervention or {})),
            posterior is None,
            tuple((k, cast("dict", posterior)[k].shape[1:]) for k in latent),
            fewer_rows,
        )
        if key not in plan_cache:
            # A call without return sites, as `log_likelihood`, traces its rows
            # only for a latent site whose draws do not fit one row fewer, or when
            # each device evaluates one row fewer, which would take such draws
            shapes = shapes_at(n_obs) if return_sites else {}
            per_row = {
                k
                for k, v in shapes.items()
                if k in return_sites and v.shape[:1] == (n_obs,)
            }
            follows, failed = {}, False
            try:
                if (shapes or not return_sites) and n_obs > 1 and shard_axis == "obs":
                    fewer = shapes_at(n_obs - 1)
                    if not return_sites and (
                        fewer_rows
                        or any(
                            fewer[k].shape != cast("dict", posterior)[k].shape[1:]
                            for k in latent
                            if k in fewer
                        )
                    ):
                        shapes = shapes_at(n_obs)
                    per_row &= {
                        k for k, v in fewer.items() if v.shape[:1] == (n_obs - 1,)
                    }
                    follows = {
                        k: shapes[k].shape
                        for k in latent
                        if k in shapes
                        and k in fewer
                        and fewer[k].shape != shapes[k].shape
                    }
                    # The prior draws its global sites from one row
                    if posterior is None:
                        shapes_at(1)
                # One row has no row fewer, so a second row tells a global site of
                # length one from a site with one value per row.
                elif shapes and shard_axis == "obs":
                    per_row &= {
                        k for k, v in shapes_at(2).items() if v.shape[:1] == (2,)
                    }
            # A kernel that does not run on fewer rows, such as one whose plate size
            # is a fixed count, cannot be split either; one row is never split.
            except Exception:
                failed = n_obs > 1
            plan_cache[key] = (
                frozenset(shapes.keys() & set(return_sites)) - per_row,
                failed,
                follows,
            )
        unsplit, failed, follows = plan_cache[key]
        # A posterior site whose draws do not fit the rows fails as on one batch,
        # though device batches with the fitted row count would take them.
        for k, shape in sorted(follows.items()):
            draws = (posterior or {})[k].shape[1:]
            if (
                k not in (intervention or {})
                and k not in observed_sites
                and len(draws) == len(shape)
                and draws != shape
            ):
                err_msg = (
                    f"Each posterior draw of site {k!r} has shape {draws}, which "
                    f"does not fit the shape {shape} the site has for the "
                    "observations being evaluated: a site with one value per "
                    "observation fits only the fitted observations, and a site "
                    "sized by a count read from the data fits only the fitted "
                    "count."
                )
                raise ValueError(err_msg)
        # A posterior site follows the rows only if its draws fit them all
        rowed = sorted(
            k for k, shape in follows.items() if (posterior or {})[k].shape[1:] == shape
        )
        msg, category = None, PerformanceWarning
        if failed and return_sites:
            msg = (
                "The kernel does not run on fewer observations than the input, so "
                "`shard_axis='obs'` cannot split it; rerunning with "
                "`shard_axis='draw'`. Pass `shard_axis='draw'` to silence this "
                "warning."
            )
        whole = sorted(unsplit)
        if param_output in whole:
            msg = (
                f"The output site {param_output!r} does not have one value "
                "per observation of the input"
            ) + (
                ", so `shard_axis='obs'` cannot split it; rerunning with "
                "`shard_axis='draw'`."
                if shard_axis == "obs"
                else "."
            )
            category = OutputWarning
        elif whole and msg is None and shard_axis == "obs":
            msg = (
                f"Return site(s) {', '.join(map(repr, whole))} have no "
                "observation axis, so `shard_axis='obs'` cannot split them; "
                "rerunning with `shard_axis='draw'`. Pass `shard_axis='draw'`, "
                "or leave the site(s) out of `return_sites`, to silence this "
                "warning."
            )
        elif rowed and msg is None and splits:
            msg = (
                f"The posterior site(s) {', '.join(map(repr, rowed))} have a size "
                "that follows the number of observations in each draw, so "
                "`shard_axis='obs'` cannot split them; rerunning with "
                "`shard_axis='draw'`. Pass `shard_axis='draw'` to silence this "
                "warning."
            )
        elif rowed and msg is None and batch_size is None:
            # One batch on one device keeps the rows whole
            batch_size = n_obs
        if msg is not None:
            warn(msg, category=category, skip_file_prefixes=_SKIP_FILE_PREFIXES)
            if shard_axis == "obs":
                shard_axis, batch_size = "draw", None
    if shard_axis == "obs" and batch_size is not None and batch_size % num_devices:
        msg = (
            f"The `batch_size` ({batch_size}) is not divisible by the number of "
            f"devices ({num_devices}). Use a multiple of {num_devices} "
            "for optimal performance."
        )
        warn(msg, category=PerformanceWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)

    resolved = (
        _resolve_batch_size(
            batch_size,
            axis_size=n_obs,
            other_size=num_samples,
            num_devices=num_devices,
            item_nbytes=num_samples * row_nbytes,
        )
        if shard_axis == "obs"
        else _resolve_batch_size(
            batch_size,
            axis_size=num_samples,
            other_size=n_obs,
            num_devices=num_devices,
            item_nbytes=n_obs * row_nbytes + rest_nbytes,
        )
    )
    if batch_size is None:
        logger.debug("Resolved batch_size=%d automatically.", resolved)

    return shard_axis, resolved
