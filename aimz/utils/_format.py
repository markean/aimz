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
    """Generate metadata attributes for the aimz library.

    Returns:
        Attributes including creation timestamp and library version.
    """
    return {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "aimz_version": version("aimz"),
    }


def _site_dims(site: str, ndim: int, names: Sequence[str] = ()) -> list[str]:
    """Name the dimensions of a site that follow its sample dimensions.

    The given names come first and every other dimension takes the default
    ``<site>_dim_<i>``, numbered from zero among the dimensions without a name. Names
    beyond the dimensions are dropped, as for a log-likelihood, which has no event
    dimensions. Names that cannot apply, being repeated or ``chain`` or ``draw``, give
    way to the default names.

    Args:
        site: The site name.
        ndim: Number of dimensions after the sample dimensions.
        names: The site's dimension names, outermost first.

    Returns:
        The dimension names.
    """
    if len(set(names)) < len(names) or {"chain", "draw"} & set(names):
        names = ()

    return [*names[:ndim], *(f"{site}_dim_{i}" for i in range(ndim - len(names)))]


def _group_dims(
    shapes: Mapping[str, Sequence[int]],
    dims: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Name the dimensions of the sites of one group after their sample dimensions.

    Each site is named by :func:`_site_dims`. Sites whose names would give a dimension
    two lengths within the group keep the default names instead, so naming never fails.

    Args:
        shapes: The shape of each site after its sample dimensions.
        dims: The dimension names by site.

    Returns:
        The dimension names by site.

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
    """Convert a dictionary of arrays to an xarray DataTree.

    Each key in the dictionary becomes a variable in the Dataset, and its associated
    array is wrapped as an xarray DataArray with a ``chain`` and ``draw`` dimension to
    support MCMC-style outputs. Additional dimensions take the names in ``dims``, or the
    pattern ``<variable>_dim_<N>``. Dask arrays pass through and keep the result lazy.

    Args:
        data: A dictionary mapping variable names to arrays. Each array should have
            shape ``(num_samples, dim_0, dim_1, ...)`` where the first dimension
            represents samples or draws, stacked chain by chain.
        num_chains: Number of chains the draws are stacked from. The first dimension is
            split into ``chain`` and ``draw``.
        dims: Names of the dimensions after ``draw``, by variable.

    Returns:
        All variables with added ``chain`` and ``draw`` dimensions, along with
            coordinates for each array dimension.
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
                    dims=(
                        # The stacked draws are split into 'chain' and 'draw'.
                        "chain",
                        "draw",
                        *names[site],
                    ),
                    name=site,
                )
                for site, arr in data.items()
            },
        ).assign_attrs(_make_attrs()),
    )


def _zarr_to_datatree(artifact_path: Path) -> xr.DataTree:
    """Load a Zarr group as an xarray DataTree.

    Reads the store with :external:func:`~xarray.open_zarr`, sorts its sites by name,
    and splits its ``draw`` dimension into ``chain`` and ``draw`` by the
    ``num_chains`` attribute of the group, along with coordinates for each dimension,
    matching the structure produced by :func:`_dict_to_datatree`. The group's other
    attributes are kept.

    Args:
        artifact_path: Path holding the Zarr group.

    Returns:
        The loaded dataset with ``chain`` and ``draw`` dimensions, along with
            coordinates for each array dimension.
    """
    with zarr_config.set({"array.read_missing_chunks": False}):
        ds = open_zarr(artifact_path, consolidated=False)
    # The tree shows the chains as a dimension
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
    """Build the aimz output DataTree.

    The ``group`` node is loaded via :func:`_zarr_to_datatree` when ``data`` is a
    path, or built via :func:`_dict_to_datatree` when it is a mapping of in-memory
    arrays. ``attrs`` become the attributes of the ``group`` node. A Zarr group is
    first given them as its attributes, with its chain count, so the tree can be
    rebuilt from the files alone. Only the Zarr-backed tree records the path (as
    ``str``) in the ``artifact_path`` attribute on both the root tree and the
    ``group`` node; the in-memory tree has no artifact.

    Args:
        data: Source of the site data to attach under ``group``: a call-specific path
            holding a Zarr group, or a mapping of site arrays each with shape
            ``(num_samples, dim_0, ...)``.
        group: Group name to attach the site data under (e.g. ``"log_likelihood"``,
            ``"prior_predictive"``).
        posterior: Optional posterior samples; when provided, added as a ``"posterior"``
            subtree before ``group``.
        num_chains: Number of chains the posterior draws are stacked from. The
            posterior subtree and ``group`` split their draws into ``chain`` and
            ``draw``, except a ``"prior_predictive"`` group, whose draws do not come
            from the posterior.
        dims: Names of the dimensions after ``draw``, by variable. A Zarr group already
            carries them.
        attrs: Attributes describing the call that produced the site data, added to
            the ``group`` node.

    Returns:
        A DataTree rooted at ``"root"`` with the site data attached under ``group``
        and, optionally, a ``"posterior"`` subtree.
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
