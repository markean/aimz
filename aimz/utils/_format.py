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

import datetime as dt
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from warnings import warn

import numpy as np
import xarray as xr
from xarray import open_zarr
from zarr import config as zarr_config
from zarr import open_group

from aimz._exceptions import _SKIP_FILE_PREFIXES, OutputWarning

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt
    from dask.array import Array as DaskArray
    from jax import Array


def _make_attrs() -> dict[str, str]:
    """Return the creation time and the aimz version as attributes."""
    return {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "aimz_version": version("aimz"),
    }


def _site_dims(site: str, ndim: int, names: Sequence[str] = ()) -> list[str]:
    """Name the ``ndim`` dimensions of a site after its sample dimensions.

    The given names come first and the other dimensions take ``<site>_dim_<i>``,
    numbered among themselves. Names beyond the dimensions are dropped; names that
    repeat or are ``chain`` or ``draw`` give way to the defaults.
    """
    if len(set(names)) < len(names) or {"chain", "draw"} & set(names):
        names = ()

    return [*names[:ndim], *(f"{site}_dim_{i}" for i in range(ndim - len(names)))]


def _group_dims(
    shapes: Mapping[str, Sequence[int]],
    dims: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Name the dimensions of a group's sites, keeping the defaults where names clash.

    Warns:
        OutputWarning: If sites keep the default names because a name would have two
            lengths within the group.
    """
    names = {
        site: _site_dims(site, len(shape), dims.get(site, ()))
        for site, shape in shapes.items()
    }
    lengths: dict[str, int] = {}
    clashes = set()
    for site, shape in shapes.items():
        for name, length in zip(names[site], shape, strict=True):
            if lengths.setdefault(name, length) != length:
                clashes.add(name)
    if clashes:
        kept = [site for site in shapes if clashes & set(names[site])]
        msg = (
            f"The sites {', '.join(map(repr, kept))} keep the default dimension names: "
            f"{', '.join(map(repr, sorted(clashes)))} would have two lengths in one "
            "group."
        )
        warn(msg, category=OutputWarning, skip_file_prefixes=_SKIP_FILE_PREFIXES)

    return {
        site: _site_dims(site, len(shapes[site]))
        if clashes & set(names[site])
        else names[site]
        for site in shapes
    }


def _dict_to_datatree(
    data: Mapping[str, Array | npt.NDArray | DaskArray],
    num_chains: int,
    dims: Mapping[str, Sequence[str]] | None = None,
) -> xr.DataTree:
    """Convert arrays with the draws stacked chain by chain on their leading axis.

    The leading axis splits into ``chain`` and ``draw``; the other dimensions take the
    names in ``dims`` or ``<site>_dim_<i>``. Dask arrays keep the result lazy.
    """
    names = _group_dims({site: arr.shape[1:] for site, arr in data.items()}, dims or {})

    return xr.DataTree(
        xr.Dataset(
            {
                site: xr.DataArray(
                    np.expand_dims(cast("npt.NDArray", arr), axis=0).reshape(
                        num_chains,
                        arr.shape[0] // num_chains,
                        *arr.shape[1:],
                    ),
                    coords={
                        "chain": np.arange(num_chains),
                        "draw": np.arange(arr.shape[0] // num_chains),
                        **{
                            name: np.arange(arr.shape[i + 1])
                            for i, name in enumerate(names[site])
                        },
                    },
                    dims=("chain", "draw", *names[site]),
                    name=site,
                )
                for site, arr in data.items()
            },
        ).assign_attrs(_make_attrs()),
    )


def _zarr_to_datatree(artifact_path: Path) -> xr.DataTree:
    """Load a Zarr group, splitting its ``draw`` dimension by its ``num_chains``."""
    with zarr_config.set({"array.read_missing_chunks": False}):
        ds = open_zarr(artifact_path, consolidated=False)
    num_chains = ds.attrs.pop("num_chains", 1)
    ds = ds[sorted(ds.data_vars)]
    # An empty result (no return sites) has no draw axis to split into chains
    ds = (
        ds.coarsen(draw=ds.sizes["draw"] // num_chains).construct(
            draw=("chain", "draw"),
        )
        if num_chains > 1 and "draw" in ds.sizes
        else ds.expand_dims(dim="chain", axis=0)
    )
    ds = ds.assign_coords(
        {k: np.arange(ds.sizes[k]) for k in ds.sizes},
    ).assign_attrs(_make_attrs())

    return xr.DataTree(ds)


def _build_datatree(
    data: Path | Mapping[str, Array | npt.NDArray | DaskArray],
    group: str,
    posterior: Mapping[str, Array | npt.NDArray] | None = None,
    num_chains: int = 1,
    dims: Mapping[str, Sequence[str]] | None = None,
    attrs: Mapping[str, object] | None = None,
) -> xr.DataTree:
    """Build the output tree with the site data under ``group``.

    A Zarr group is first given ``attrs`` and its chain count as attributes, so the
    tree can be rebuilt from the files alone, and the tree records its path as the
    ``artifact_path`` attribute. A posterior is added as a ``posterior`` subtree.
    """
    out = xr.DataTree(name="root")
    if posterior:
        out["posterior"] = _dict_to_datatree(
            posterior,
            num_chains=num_chains,
            dims=dims,
        )
    group_chains = 1 if group == "prior_predictive" else num_chains
    if isinstance(data, Path):
        # String, integer and list values only, so the store keeps them as they are
        stored: dict[str, Any] = {
            **_make_attrs(),
            "num_chains": group_chains,
            **(attrs or {}),
        }
        open_group(data, mode="r+").attrs.update(stored)
        out[group] = _zarr_to_datatree(data)
        out[group].attrs["artifact_path"] = str(data)
        out.attrs["artifact_path"] = str(data)
    else:
        out[group] = _dict_to_datatree(data, num_chains=group_chains, dims=dims)
        out[group].attrs.update(attrs or {})

    return out
