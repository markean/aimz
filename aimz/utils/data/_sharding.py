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

"""Module for creating functions for sharding."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array, device_put, jit, lax, random, shard_map
from jax.sharding import PartitionSpec

from aimz.sampling._forward import _sample_forward
from aimz.utils._kwargs import _combine, _split_intervention_fields
from aimz.utils._log_likelihood import _log_likelihood
from aimz.utils._output import _validate_streamed_axis_size

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.sharding import Mesh, Sharding
    from jax.typing import ArrayLike


def _create_sharded_sampler(
    mesh: Mesh | None,
    n_kwargs_array: int,
    n_kwargs_const: int,
    shard_axis: Literal["obs", "draw"] = "obs",
) -> Callable:
    """Create a jitted predictive sampler, sharded over ``mesh`` if any.

    ``"obs"`` shards the input and the per-observation keyword arguments and
    replicates the posterior; ``"draw"`` shards the pre-split per-draw keys and the
    posterior and replicates the input. The function takes the kernel, the draw count
    (per device under ``"draw"``), the key or keys, the return sites, the samples, the
    params, the intervention, the input parameter name, the per-observation keyword
    names, the static constants, the input, and then the per-observation arrays and
    the array leaves of the constants.
    """
    draws = shard_axis == "draw"
    axis = None
    if mesh is not None:
        (axis,) = mesh.axis_names

    def f(
        kernel: Callable,
        num_samples: int,
        rng_key: Array,
        return_sites: tuple[str, ...],
        samples: dict[str, Array],
        params: dict,
        intervention: dict,
        param_input: str,
        kwargs_key: tuple[str, ...],
        kwargs_static: tuple,
        X: Array,
        *args: object,
    ) -> dict[str, Array]:
        # Under `obs` each device folds its index into the key before splitting it
        if draws:
            rng_keys = rng_key
        else:
            rng_keys = random.split(
                rng_key
                if axis is None
                else random.fold_in(rng_key, data=lax.axis_index(axis)),
                num=num_samples,
            )

        n = len(kwargs_key)
        model_kwargs, fields = _split_intervention_fields(
            {
                param_input: X,
                **dict(zip(kwargs_key, args[:n], strict=True)),
                **_combine(args[n:], kwargs_static),
            },
        )

        out = _sample_forward(
            kernel,
            rng_keys=rng_keys,
            return_sites=return_sites,
            samples=samples,
            params=params,
            intervention={**intervention, **fields},
            model_kwargs=model_kwargs,
        )
        # Checked at trace time, so every device count raises the same error
        if not draws:
            for site, value in out.items():
                _validate_streamed_axis_size(
                    value,
                    site=site,
                    axis=1,
                    chunk_size=X.shape[0],
                )

        return out

    if mesh is None:
        return partial(
            jit,
            static_argnames=[
                "kernel",
                "num_samples",
                "return_sites",
                "param_input",
                "kwargs_key",
                "kwargs_static",
            ],
        )(f)

    # Under `draw`, `out_spec` shards only the leading axis, so a rank-1 per-draw site
    # stays valid
    if draws:
        rng_spec = samples_spec = PartitionSpec(axis)
        x_spec = kw_spec = PartitionSpec()
        out_spec = PartitionSpec(axis)
    else:
        rng_spec = samples_spec = PartitionSpec()
        x_spec = kw_spec = PartitionSpec(axis)
        out_spec = PartitionSpec(None, axis)

    return partial(
        jit,
        static_argnames=[
            "kernel",
            "num_samples",
            "return_sites",
            "param_input",
            "kwargs_key",
            "kwargs_static",
        ],
    )(
        partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                None,  # kernel
                None,  # num_samples
                rng_spec,  # rng_key
                None,  # return_sites
                samples_spec,  # samples
                PartitionSpec(),  # params
                PartitionSpec(),  # intervention
                None,  # param_input
                None,  # kwargs_key
                None,  # kwargs_static
                x_spec,  # X
                *(
                    [kw_spec] * n_kwargs_array  # kwargs_array
                    + [PartitionSpec()] * n_kwargs_const  # kwargs_const
                ),
            ),
            out_specs=out_spec,
            check_vma=False,
        )(f),
    )


def _create_sharded_log_likelihood(
    mesh: Mesh | None,
    n_kwargs_array: int,
    n_kwargs_const: int,
    shard_axis: Literal["obs", "draw"] = "obs",
) -> Callable:
    """Create a jitted log-likelihood function, sharded over ``mesh`` if any.

    ``"obs"`` shards the input, the output, and the per-observation keyword arguments
    and replicates the posterior; ``"draw"`` the reverse. The function takes the
    kernel, the samples, the params, the input and output parameter names, the
    per-observation keyword names, the static constants, the input, the output, and
    then the per-observation arrays and the array leaves of the constants.
    """
    draws = shard_axis == "draw"

    def f(
        kernel: Callable,
        samples: dict[str, Array],
        params: dict,
        param_input: str,
        param_output: str,
        kwargs_key: tuple[str, ...],
        kwargs_static: tuple,
        X: Array,
        y: Array,
        *args: object,
    ) -> Array:
        n = len(kwargs_key)
        out = _log_likelihood(
            kernel,
            samples=samples,
            params=params,
            model_kwargs={
                param_input: X,
                param_output: y,
                **dict(zip(kwargs_key, args[:n], strict=True)),
                **_combine(args[n:], kwargs_static),
            },
        )[param_output]
        if not draws:
            _validate_streamed_axis_size(
                out,
                site=param_output,
                axis=1,
                chunk_size=X.shape[0],
            )

        return out

    if mesh is None:
        return partial(
            jit,
            static_argnames=[
                "kernel",
                "param_input",
                "param_output",
                "kwargs_key",
                "kwargs_static",
            ],
        )(f)

    (axis,) = mesh.axis_names
    # `out_spec` as in the sampler: the leading axis under `draw`, the observations
    # under `obs`
    if draws:
        samples_spec = PartitionSpec(axis)
        xy_spec = kw_spec = PartitionSpec()
        out_spec = PartitionSpec(axis)
    else:
        samples_spec = PartitionSpec()
        xy_spec = kw_spec = PartitionSpec(axis)
        out_spec = PartitionSpec(None, axis)

    return partial(
        jit,
        static_argnames=[
            "kernel",
            "param_input",
            "param_output",
            "kwargs_key",
            "kwargs_static",
        ],
    )(
        partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                None,  # kernel
                samples_spec,  # samples
                PartitionSpec(),  # params
                None,  # param_input
                None,  # param_output
                None,  # kwargs_key
                None,  # kwargs_static
                xy_spec,  # X
                xy_spec,  # y
                *(
                    [kw_spec] * n_kwargs_array  # kwargs_array
                    + [PartitionSpec()] * n_kwargs_const  # kwargs_const
                ),
            ),
            out_specs=out_spec,
            check_vma=False,
        )(f),
    )


def _replicate(arr: ArrayLike, sharding: Sharding | None) -> Array:
    """Place an array replicated across devices, or as is on a single device."""
    if sharding is None:
        return jnp.asarray(arr)

    return device_put(jnp.asarray(arr), device=sharding)


@partial(jit, static_argnames="size")
def _take_draws(arr: Array, start: int, *, size: int) -> Array:
    """Take ``size`` draws from ``start`` along the leading axis.

    ``start`` is traced, so every chunk reuses one program; indices past the last draw
    repeat it, which pads a shorter last chunk.
    """
    return arr[jnp.minimum(start + jnp.arange(size), arr.shape[0] - 1)]


def _prepare_draw_chunk(
    posterior: dict[str, Array],
    draw_keys: Array | None,
    start: int,
    stop: int,
    size: int,
    num_devices: int,
    sharding: Sharding | None,
) -> tuple[dict[str, Array], Array | None, int]:
    """Slice, pad, and shard the posterior and the keys of the draws ``[start:stop)``.

    The chunk is padded to ``size`` rounded up to a multiple of ``num_devices``, so a
    shorter last chunk keeps the shape of the others. A host-backed posterior is sliced
    on the host.

    Returns:
        The posterior chunk, the chunk keys or ``None``, and the per-device draw count.
    """
    clen = stop - start
    clen_pad = -(-max(clen, size) // num_devices) * num_devices
    per_device, n_pad = clen_pad // num_devices, clen_pad - clen
    chunk_samples = {}
    for k, v in posterior.items():
        if isinstance(v, Array):
            chunk_samples[k] = _take_draws(v, start, size=clen_pad)
            continue
        arr = v[start:stop]
        if n_pad:
            arr = np.pad(arr, [(0, n_pad), *[(0, 0)] * (arr.ndim - 1)], mode="edge")
        chunk_samples[k] = arr
    chunk_keys = (
        None if draw_keys is None else _take_draws(draw_keys, start, size=clen_pad)
    )
    if sharding is not None:
        if chunk_samples:
            chunk_samples = device_put(chunk_samples, device=sharding)
        if chunk_keys is not None:
            chunk_keys = device_put(chunk_keys, device=sharding)

    return chunk_samples, chunk_keys, per_device
