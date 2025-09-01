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

"""Module for formatting and handling model outputs."""

import datetime
from importlib import metadata

import numpy as np
import xarray as xr
from jax import Array


def _make_attrs() -> dict[str, str]:
    """Generate metadata attributes for the aimz library.

    Returns:
        dict[str, str]: Attributes including creation timestamp and library version.
    """
    return {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "aimz_version": metadata.version("aimz"),
    }


def _dict_to_datatree(data: dict[str, Array]) -> xr.DataTree:
    """Convert a dictionary of arrays to an xarray DataTree.

    Each key in the dictionary becomes a variable in the Dataset, and its associated
    array is wrapped as an xarray DataArray with a ``chain`` and ``draw`` dimension to
    support MCMC-style outputs. Additional dimensions are automatically named using the
    pattern ``<variable>_dim_<N>``.

    Args:
        data (dict[str, Array]): Dictionary mapping variable names to arrays. Each array
        should have shape ``(num_samples, dim_0, dim_1, ...)`` where the first dimension
        represents samples or draws.

    Returns:
        All variables with added ``chain`` and ``draw`` dimensions, along with
            coordinates for each array dimension.
    """
    return xr.DataTree(
        xr.Dataset(
            {
                site: xr.DataArray(
                    np.expand_dims(arr, axis=0),
                    coords={
                        "chain": np.arange(1),
                        "draw": np.arange(arr.shape[0]),
                        **{
                            f"{site}_dim_{i}": np.arange(arr.shape[i + 1])
                            for i in range(arr.ndim - 1)
                        },
                    },
                    dims=(
                        # Adding the 'chain' dimension to support MCMC-style structures.
                        "chain",
                        "draw",
                        # arr has shape (draw, dim_0, dim_1, ...), so arr.ndim includes
                        # 'draw' and we subtract 1
                        *[f"{site}_dim_{i}" for i in range(arr.ndim - 1)],
                    ),
                    name=site,
                )
                for site, arr in data.items()
            },
        ).assign_attrs(_make_attrs()),
    )
