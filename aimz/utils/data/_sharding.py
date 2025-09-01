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

"""Module for creating functions for sharding.

NOTE: This module is experimental and subject to change. It utilizes JAX's `shard_map()`
to distribute computations across devices. Tested on CPU and GPU.
"""

from functools import partial
from typing import TYPE_CHECKING

from jax import Array, jit
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
from jax.typing import ArrayLike
from numpyro.infer import log_likelihood as log_lik

from aimz.sampling._forward import _sample_forward

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.sharding import Mesh


def _create_sharded_sampler(
    mesh: "Mesh | None",
    n_kwargs_array: int,
    n_kwargs_extra: int,
) -> "Callable":
    """Create a sharded predictive sampling function.

    Args:
        mesh (Mesh): The JAX mesh object defining the device mesh for sharding.
        n_kwargs_array (int): The number of arguments in the keyword arguments that are
            array-like (sharded).
        n_kwargs_extra (int): The number of extra keyword arguments that are not
            array-like (not sharded).

    Returns:
        Callable: A sharded function that takes the following arguments:
            - rng_key (Array): A pseudo-random number generator key.
            - kernel (Callable): A probabilistic model with Pyro primitives.
            - samples (dict): A dictionary of samples to condition on.
            - batch_shape (tuple[int]): The shape of the batch dimension, specifically
                ``(num_samples,)``.
            - param_input (str): The name of the parameter in the ``kernel`` for the
                input data.
            - kwargs_key (tuple[str]): A tuple of keyword argument names.
            - X (Array): Input data.
            - *args (tuple): Additional arguments constructed from the original keyword
                arguments (both sharded and non-sharded).

    """

    def f(
        kernel: "Callable",
        num_samples: int,
        rng_key: ArrayLike,
        return_sites: tuple[str],
        samples: dict[str, Array],
        param_input: str,
        kwargs_key: tuple[str],
        X: Array,
        *args: ArrayLike,
    ) -> dict[str, Array]:
        return _sample_forward(
            model=kernel,
            num_samples=num_samples,
            rng_key=rng_key,
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
                None,  # posterior_samples
                None,  # rng_key
                None,  # num_samples
                None,  # return_sites
                None,  # param_input
                None,  # kwargs_key
                PartitionSpec(axis),  # X
                *(
                    [PartitionSpec(axis)] * n_kwargs_array  # kwargs_array
                    + [None] * n_kwargs_extra  # kwargs_extra
                ),
            ),
            out_specs=PartitionSpec(None, axis),
            check_rep=False,
        )(f),
    )


def _create_sharded_log_likelihood(
    mesh: "Mesh | None",
    n_kwargs_array: int,
    n_kwargs_extra: int,
) -> "Callable":
    """Create a sharded log-likelihood function.

    Args:
        mesh (Mesh): The JAX mesh object defining the device mesh for sharding.
        n_kwargs_array (int): The number of arguments in the keyword arguments that are
            array-like (sharded).
        n_kwargs_extra (int): The number of extra keyword arguments that are not
            array-like (not sharded).

    Returns:
        Callable: A sharded function that takes the following arguments:
            - kernel (Callable): A probabilistic model with Pyro primitives optimized
                with variational inference.
            - posterior_samples (dict): A dictionary of posterior samples.
            - param_input (str): The name of the parameter in the ``kernel`` for the
                input data.
            - param_output (str): The name of the parameter in the ``kernel`` for the
                output data.
            - kwargs_key (tuple[str]): A tuple of keyword argument names.
            - X (Array): Input data.
            - y (Array): Output data.
            - *args (tuple): Additional arguments constructed from the original keyword
                arguments (both sharded and non-sharded).

    """

    def f(
        kernel: "Callable",
        posterior_samples: dict,
        param_input: str,
        param_output: str,
        kwargs_key: tuple[str],
        X: Array,
        y: Array,
        *args: tuple,
    ) -> Array:
        return log_lik(
            kernel,
            posterior_samples=posterior_samples,
            **{
                param_input: X,
                param_output: y,
                **dict(zip(kwargs_key, args, strict=True)),
            },
        ).get(param_output)

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
                None,  # posterior_samples
                None,  # param_input
                None,  # param_output
                None,  # kwargs_key
                PartitionSpec(axis),  # X
                PartitionSpec(axis),  # y
                *(
                    [PartitionSpec(axis)] * n_kwargs_array  # kwargs_array
                    + [None] * n_kwargs_extra  # kwargs_extra
                ),
            ),
            out_specs=PartitionSpec(None, axis),
            check_rep=False,
        )(f),
    )
