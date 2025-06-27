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

"""Module for custom data loader with padding logic for JAX arrays.

This module defines a custom `ArrayLoader` that processes batches of data and applies
padding to ensure the batch size is compatible with sharding across multiple XLA
devices.
"""

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import device_put
from jax_dataloader import ArrayDataset
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax
    from jax.sharding import NamedSharding
    from torch.utils.data import Sampler


class ArrayLoader(DataLoader):
    """A custom DataLoader class that extends PyTorch's DataLoader."""

    def __init__(
        self,
        dataset: ArrayDataset,
        *,
        batch_size: int = 1,
        sampler: "Sampler | None" = None,
        num_workers: int = 0,
        collate_fn: "Callable | None" = None,
        pin_memory: bool = False,
    ) -> None:
        """Initializes an ArrayLoader instance."""
        super().__init__(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    @staticmethod
    def calculate_padding(batch_size: int, num_devices: int) -> int:
        """Calculate the number of padding needed.

        Args:
            batch_size (int): The size of the batch.
            num_devices (int): The number of devices.

        Returns:
            int: The number of padding rows (or elements) needed to make the batch size
                divisible by the number of devices.
        """
        remainder = batch_size % num_devices
        return 0 if remainder == 0 else num_devices - remainder

    @staticmethod
    def pad_array(x: "jax.Array", n_pad: int, axis: int) -> "jax.Array":
        """Pad an array to ensure compatibility with sharding.

        Args:
            x (jax.Array): The input array to be padded.
            n_pad (int): The number of padding elements to add.
            axis (int): The axis along which to apply the padding.

        Returns:
            jax.Array: The padded array with padding applied along the specified axis.

        Raises:
            ValueError: If padding is requested along an unsupported axis for a 1D
                array.
        """
        if x.ndim == 1:
            if axis == 0:
                return jnp.pad(x, pad_width=(0, n_pad), mode="edge")
            msg = "Padding 1D arrays is only supported along axis 0."
            raise ValueError(msg)

        # Initialize all axes with no padding
        pad_width: list[tuple[int, int]] = [(0, 0)] * x.ndim
        # Apply padding to the specified axis
        pad_width[axis] = (0, n_pad)

        return jnp.pad(x, pad_width=pad_width, mode="edge")

    @staticmethod
    def collate_without_output(
        batch: list[tuple],
        device: "NamedSharding | None" = None,
    ) -> tuple:
        """Collate function to process batches with sharding and padding.

        This function unpacks the batch of data, converts it into JAX arrays, and
        applies padding to ensure the batch size is compatible with the number of
        devices, if sharding is necessary. When a device is provided, the data is
        automatically distributed across the available devices.

        Args:
            batch (list[tuple]): A list of tuples, where each tuple contains the input
                data, optional target data, and array-like keyword arguments.
            device (NamedSharding | None, optional): Sharding using named axes for
                parallel data distribution across devices. Defaults to `None`, meaning
                no sharding is applied.

        Returns:
            tuple: A tuple containing:
                - n_pad (int): The number of padding rows/elements added (0 if no
                    padding was required).
                - x_batch (jax.Array): The input batch with padding applied if
                    necessary.
                - kwargs_batch (list[jax.Array]): A list of keyword arguments with
                    padding applied if necessary.
        """
        x_batch, *kwargs_batch = map(jnp.asarray, zip(*batch, strict=True))

        n_pad = (
            ArrayLoader.calculate_padding(
                len(x_batch),
                num_devices=device.num_devices,
            )
            if device
            else 0
        )
        if n_pad:
            x_batch = ArrayLoader.pad_array(x_batch, n_pad=n_pad, axis=0)
            kwargs_batch = [
                ArrayLoader.pad_array(x, n_pad=n_pad, axis=0) for x in kwargs_batch
            ]

        if device:
            x_batch = device_put(x_batch, device=device)
            kwargs_batch = [device_put(x, device=device) for x in kwargs_batch]

        return n_pad, x_batch, *kwargs_batch

    @staticmethod
    def collate_with_sharding(
        batch: list[tuple],
        device: "NamedSharding | None" = None,
    ) -> tuple:
        """Collate function to process batches with sharding and padding.

        This function unpacks the batch of data, converts it into JAX arrays, and
        applies padding to ensure the batch size is compatible with the number of
        devices, if sharding is necessary. When a device is provided, the data is
        automatically distributed across the available devices.

        Args:
            batch (list[tuple]): A list of tuples, where each tuple contains the input
                data, optional target data, and array-like keyword arguments.
            device (NamedSharding | None, optional): Sharding using named axes for
                parallel data distribution across devices. Defaults to `None`, meaning
                no sharding is applied.

        Returns:
            tuple: A tuple containing:
                - n_pad (int): The number of padding rows/elements added (0 if no
                    padding was required).
                - x_batch (jax.Array): The input batch with padding applied if
                    necessary.
                - y_batch (jax.Array): The target batch with padding applied.
                - kwargs_batch (list[jax.Array]): A list of keyword arguments with
                    padding applied if necessary.
        """
        x_batch, y_batch, *kwargs_batch = map(jnp.asarray, zip(*batch, strict=True))

        n_pad = (
            ArrayLoader.calculate_padding(
                len(x_batch),
                num_devices=device.num_devices,
            )
            if device
            else 0
        )
        if n_pad:
            x_batch = ArrayLoader.pad_array(x_batch, n_pad=n_pad, axis=0)
            y_batch = ArrayLoader.pad_array(y_batch, n_pad=n_pad, axis=0)
            kwargs_batch = [
                ArrayLoader.pad_array(x, n_pad=n_pad, axis=0) for x in kwargs_batch
            ]

        if device:
            x_batch = device_put(x_batch, device=device)
            y_batch = device_put(y_batch, device=device)
            kwargs_batch = [device_put(x, device=device) for x in kwargs_batch]

        return n_pad, x_batch, y_batch, *kwargs_batch
