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

from __future__ import annotations

import datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr
from xarray import open_zarr

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    from dask.array import Array as DaskArray
    from jax import Array


def _make_attrs() -> dict[str, str]:
    """Generate metadata attributes for the aimz library.

    Returns:
        Attributes including creation timestamp and library version.
    """
    return {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "aimz_version": version("aimz"),
    }


def _dict_to_datatree(
    data: Mapping[str, Array | npt.NDArray | DaskArray],
) -> xr.DataTree:
    """Convert a dictionary of arrays to an xarray DataTree.

    Each key in the dictionary becomes a variable in the Dataset, and its associated
    array is wrapped as an xarray DataArray with a ``chain`` and ``draw`` dimension to
    support MCMC-style outputs. Additional dimensions are automatically named using the
    pattern ``<variable>_dim_<N>``. Dask arrays pass through and keep the result lazy.

    Args:
        data: A dictionary mapping variable names to arrays. Each array should have
            shape ``(num_samples, dim_0, dim_1, ...)`` where the first dimension
            represents samples or draws.

    Returns:
        All variables with added ``chain`` and ``draw`` dimensions, along with
            coordinates for each array dimension.
    """
    return xr.DataTree(
        xr.Dataset(
            {
                site: xr.DataArray(
                    np.expand_dims(cast("npt.NDArray", arr), axis=0),
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


def _zarr_to_datatree(artifact_path: Path) -> xr.DataTree:
    """Load a Zarr group as an xarray DataTree.

    Reads the store with :external:func:`~xarray.open_zarr` and adds a ``chain``
    dimension along with coordinates for each dimension, matching the structure
    produced by :func:`_dict_to_datatree`.

    Args:
        artifact_path: Path holding the Zarr group.

    Returns:
        The loaded dataset with an added ``chain`` dimension, along with coordinates
            for each array dimension.
    """
    ds = open_zarr(artifact_path, consolidated=False).expand_dims(dim="chain", axis=0)
    ds = ds.assign_coords(
        {k: np.arange(ds.sizes[k]) for k in ds.sizes},
    ).assign_attrs(_make_attrs())

    return xr.DataTree(ds)


def _build_datatree(
    data: Path | Mapping[str, Array | npt.NDArray | DaskArray],
    group: str,
    posterior: Mapping[str, Array | npt.NDArray] | None = None,
) -> xr.DataTree:
    """Build the aimz output DataTree.

    The ``group`` node is loaded via :func:`_zarr_to_datatree` when ``data`` is a
    path, or built via :func:`_dict_to_datatree` when it is a mapping of in-memory
    arrays. Only the Zarr-backed tree records the path (as ``str``) in the
    ``artifact_path`` attribute on both the root tree and the ``group`` node; the
    in-memory tree has no artifact.

    Args:
        data: Source of the site data to attach under ``group``: a call-specific path
            holding a Zarr group, or a mapping of site arrays each with shape
            ``(num_samples, dim_0, ...)``.
        group: Group name to attach the site data under (e.g. ``"log_likelihood"``,
            ``"prior_predictive"``).
        posterior: Optional posterior samples; when provided, added as a ``"posterior"``
            subtree before ``group``.

    Returns:
        A DataTree rooted at ``"root"`` with the site data attached under ``group``
        and, optionally, a ``"posterior"`` subtree.
    """
    out = xr.DataTree(name="root")
    if posterior:
        out["posterior"] = _dict_to_datatree(posterior)
    if isinstance(data, Path):
        out[group] = _zarr_to_datatree(data)
        out[group].attrs["artifact_path"] = str(data)
        out.attrs["artifact_path"] = str(data)
    else:
        out[group] = _dict_to_datatree(data)

    return out
