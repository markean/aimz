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
from jax import Array, device_put, jit, random, shard_map
from jax.sharding import PartitionSpec

from aimz.sampling._forward import _sample_forward
from aimz.utils._log_likelihood import _log_likelihood

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.sharding import Mesh, Sharding
    from jax.typing import ArrayLike


def _create_sharded_sampler(
    mesh: Mesh | None,
    n_kwargs_array: int,
    n_kwargs_extra: int,
    shard_axis: Literal["obs", "draw"] = "obs",
) -> Callable:
    """Create a sharded predictive sampling function.

    Args:
        mesh: The JAX mesh object defining the device mesh for sharding.
        n_kwargs_array: The number of arguments in the keyword arguments that are
            array-like.
        n_kwargs_extra: The number of extra keyword arguments that are not array-like
            (not sharded).
        shard_axis: ``"obs"`` (default) shards the observation axis of ``X`` and the
            array-kwargs and replicates the posterior on every device. ``"draw"``
            shards the posterior draws axis and replicates ``X`` and the array-kwargs;
            the caller then passes a pre-split ``(num_samples,)`` key array as
            ``rng_key`` and the per-device draw count as ``num_samples``.

    Returns:
        A sharded function that takes the following arguments:
            - kernel (Callable): A probabilistic model with NumPyro primitives.
            - num_samples (int): The number of samples to draw (per-device under
                ``"draw"`` sharding).
            - rng_key: A scalar PRNG key under ``"obs"`` sharding, or a pre-split
                ``(num_samples,)`` key array under ``"draw"`` sharding.
            - return_sites (tuple[str, ...]): Names of variables (sites) to return.
            - samples (dict): A dictionary of samples to condition on.
            - param_input (str): The name of the parameter in the ``kernel`` for the
                input data.
            - kwargs_key (tuple[str, ...]): A tuple of keyword argument names.
            - X (Array): Input data.
            - *args (tuple): Additional arguments constructed from the original keyword
                arguments (both array-like and non-array-like).
    """
    draws = shard_axis == "draw"

    def f(
        kernel: Callable,
        num_samples: int,
        rng_key: Array,
        return_sites: tuple[str, ...],
        samples: dict[str, Array],
        param_input: str,
        kwargs_key: tuple[str, ...],
        X: Array,
        *args: object,
    ) -> dict[str, Array]:
        # Under draws-sharding the device receives its slice of the pre-split per-draw
        # keys and forwards them directly; data-sharding splits the replicated scalar
        # key into ``num_samples`` per-draw keys.
        rng_keys = rng_key if draws else random.split(rng_key, num=num_samples)

        return _sample_forward(
            kernel,
            rng_keys=rng_keys,
            return_sites=return_sites,
            samples=samples,
            model_kwargs={
                param_input: X,
                **dict(zip(kwargs_key, args, strict=True)),
            },
        )

    if mesh is None:
        return partial(
            jit,
            static_argnames=[
                "kernel",
                "num_samples",
                "return_sites",
                "param_input",
                "kwargs_key",
            ],
        )(f)

    (axis,) = mesh.axis_names
    # Draw mode shards the posterior draw axis (rng keys + samples) and replicates the
    # whole input; data mode shards the observation axis of the input and replicates
    # the posterior. Under draw, ``out_spec`` shards only the leading axis: a
    # rank-agnostic single-axis spec keeps scalar-per-draw (rank-1) sites valid, whereas
    # ``PartitionSpec(axis, None)`` would require rank >= 2.
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
                None,  # param_input
                None,  # kwargs_key
                x_spec,  # X
                *(
                    [kw_spec] * n_kwargs_array  # kwargs_array
                    + [None] * n_kwargs_extra  # kwargs_extra
                ),
            ),
            out_specs=out_spec,
            check_vma=False,
        )(f),
    )


def _create_sharded_log_likelihood(
    mesh: Mesh | None,
    n_kwargs_array: int,
    n_kwargs_extra: int,
    shard_axis: Literal["obs", "draw"] = "obs",
) -> Callable:
    """Create a sharded log-likelihood function.

    Args:
        mesh: The JAX mesh object defining the device mesh for sharding.
        n_kwargs_array: The number of arguments in the keyword arguments that are
            array-like.
        n_kwargs_extra: The number of extra keyword arguments that are not array-like
            (not sharded).
        shard_axis: ``"obs"`` (default) shards the observation axis of ``X``/``y`` and
            the array-kwargs and replicates the posterior. ``"draw"`` shards the
            posterior draws axis and replicates ``X``/``y`` and the array-kwargs.

    Returns:
        A sharded function that takes the following arguments:
            - kernel (Callable): A probabilistic model with NumPyro primitives.
            - samples (dict): A dictionary of posterior samples to condition on.
            - param_input (str): The name of the parameter in the ``kernel`` for the
                input data.
            - param_output (str): The name of the parameter in the ``kernel`` for the
                output data.
            - kwargs_key (tuple[str, ...]): A tuple of keyword argument names.
            - X (Array): Input data.
            - y (Array): Output data.
            - *args (tuple): Additional arguments constructed from the original keyword
                arguments (both array-like and non-array-like).
    """
    draws = shard_axis == "draw"

    def f(
        kernel: Callable,
        samples: dict[str, Array],
        param_input: str,
        param_output: str,
        kwargs_key: tuple[str, ...],
        X: Array,
        y: Array,
        *args: object,
    ) -> Array:
        return _log_likelihood(
            kernel,
            samples=samples,
            model_kwargs={
                param_input: X,
                param_output: y,
                **dict(zip(kwargs_key, args, strict=True)),
            },
        )[param_output]

    if mesh is None:
        return partial(
            jit,
            static_argnames=[
                "kernel",
                "param_input",
                "param_output",
                "kwargs_key",
            ],
        )(f)

    (axis,) = mesh.axis_names
    # Draw mode shards the posterior draw axis (samples) and replicates the whole input
    # (X + y); data mode shards the observation axis of the input and replicates the
    # posterior. ``out_spec`` matches the sampler factory: a rank-agnostic single-axis
    # spec under draw, the observation axis under data.
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
        ],
    )(
        partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                None,  # kernel
                samples_spec,  # samples
                None,  # param_input
                None,  # param_output
                None,  # kwargs_key
                xy_spec,  # X
                xy_spec,  # y
                *(
                    [kw_spec] * n_kwargs_array  # kwargs_array
                    + [None] * n_kwargs_extra  # kwargs_extra
                ),
            ),
            out_specs=out_spec,
            check_vma=False,
        )(f),
    )


def _replicate(arr: ArrayLike, sharding: Sharding | None) -> Array:
    """Place a whole input array replicated across devices (single device: as-is).

    Draw-parallel holds the input resident and replicated while only the posterior draw
    axis is sharded, so every device sees the full input.

    Args:
        arr: The array to replicate.
        sharding: The replicated sharding, or ``None`` on a single device.

    Returns:
        The array placed on every device (or left as-is when ``sharding`` is ``None``).
    """
    if sharding is None:
        return jnp.asarray(arr)

    return device_put(jnp.asarray(arr), device=sharding)


def _prepare_draw_chunk(
    posterior: dict[str, Array],
    draw_keys: Array | None,
    start: int,
    stop: int,
    num_devices: int,
    sharding: Sharding | None,
) -> tuple[dict[str, Array], Array | None, int]:
    """Slice, pad, and shard one draw chunk for draw-parallel streaming.

    Returns ``(chunk_samples, chunk_keys, per_device)`` for draws ``[start:stop)``: the
    posterior slice and per-draw keys edge-padded so the chunk's draw count is a
    multiple of ``num_devices`` (so it splits evenly under draw-parallel sharding),
    with the per-device draw count. ``chunk_samples`` is empty for prior predictive;
    ``chunk_keys`` is ``None`` when no keys are used.

    Args:
        posterior: The whole posterior to slice (empty for prior predictive).
        draw_keys: The whole per-draw key array, or ``None`` when no keys are used.
        start: Start index of the chunk along the draw axis.
        stop: Stop index of the chunk along the draw axis.
        num_devices: Number of devices the draw axis is sharded across.
        sharding: The draw-sharding to place the chunk on, or ``None`` on a single
            device.

    Returns:
        The padded posterior chunk, the padded chunk keys (or ``None``), and the
        per-device draw count.
    """
    clen = stop - start
    d = num_devices
    if d <= 1:
        per_device, n_pad = clen, 0
    else:
        clen_pad = ((clen + d - 1) // d) * d
        per_device, n_pad = clen_pad // d, clen_pad - clen
    chunk_samples = {
        k: jnp.pad(
            v[start:stop],
            [(0, n_pad), *([(0, 0)] * (v.ndim - 1))],
            mode="edge",
        )
        for k, v in posterior.items()
    }
    chunk_keys = None
    if draw_keys is not None:
        chunk_keys = draw_keys[start:stop]
        if n_pad > 0:
            chunk_keys = jnp.concatenate(
                [chunk_keys, chunk_keys[jnp.zeros(n_pad, dtype=int)]],
            )
    if sharding is not None:
        if chunk_samples:
            chunk_samples = device_put(chunk_samples, device=sharding)
        if chunk_keys is not None:
            chunk_keys = device_put(chunk_keys, device=sharding)

    return chunk_samples, chunk_keys, per_device
