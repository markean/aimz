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

"""Kernel tracing: the sites of a trace, their dimension names, sizes and relations."""

from __future__ import annotations

import math
from contextlib import suppress
from typing import TYPE_CHECKING, cast
from warnings import warn

import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct, eval_shape, tree
from jax.core import eval_context
from numpyro import distributions, validation_enabled
from numpyro.distributions import (
    Distribution,
    ExpandedDistribution,
    Independent,
    MaskedDistribution,
)
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import init_to_sample
from numpyro.infer.inspect import get_dependencies
from numpyro.ops.provenance import eval_provenance

from aimz._exceptions import _SKIP_FILE_PREFIXES, OutputWarning
from aimz.model.kernel_spec import KernelSpec
from aimz.utils._kwargs import _combine, _partition, _to_jax
from aimz.utils._validation import _validate_kernel_body

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from jax import Array


def _trace_kernel(
    args_bound: Mapping[str, object],
    *,
    with_output: bool,
    rows_more: int | None = None,
    whole: Collection[str] = (),
    kernel: Callable,
    rng_key: Array,
    param_input: str,
    param_output: str,
    kernel_spec: KernelSpec | None,
    specs: dict[tuple, KernelSpec | None],
    dims: dict[str, tuple[str, ...]],
    site_sizes: dict[str, tuple[int, int]],
    observed_sites: set[str],
    fitted_sites: set[str],
    coupled_sites: set[str],
    site_parents: dict[str, set[str]],
    site_upstream: dict[str, set[str]],
    site_arguments: dict[str, set[str]],
    site_dists: dict[str, str],
) -> tuple[KernelSpec, KernelSpec | None]:
    """Trace the kernel once for each set of call arguments and record its sites.

    The sets differ by the arguments' names and values other than arrays, since a
    kernel may define sites only for some of them, as when data are supplied or a
    forecast is asked for. A new set merges its sites into the model's spec and
    keeps a spec of its own. The trace is abstract, unless the kernel reads a
    concrete value of an argument. A site marked for an event as long as the rows
    loses its mark when a trace at ``rows_more`` rows, twice the rows by default,
    gives it the same event length; the arguments named in ``whole`` keep their
    shape there, and a failed trace keeps every mark.

    For :meth:`~aimz.ImpactModel.describe`, the kernel then runs once more,
    abstractly, with every sample site and array argument replaced by a stand-in
    tagged with its name, so that each site carries the tags of the sites and
    arguments it is computed from. A site that control flow records has no
    stand-in and is left out, and a kernel that cannot run abstractly keeps the
    relations of an earlier trace.

    What the trace shows is written into the model's containers, which are passed in
    and changed in place: ``specs``, ``dims``, ``site_sizes``, ``observed_sites``,
    ``fitted_sites``, ``coupled_sites``, ``site_parents``, ``site_upstream``,
    ``site_arguments`` and ``site_dists``.

    Returns:
        The spec of the trace with these arguments, and the model's spec, with the
        trace's sites merged into it unless the arguments were traced before.
    """
    leaves, static = _partition(args_bound)
    if (spec := specs.get(static)) is not None:
        return spec, kernel_spec

    def sites_at(arrays: list) -> dict[str, dict]:
        sites: dict[str, dict] = {}

        def record(arrays: list) -> None:
            with validation_enabled(is_validate=False):
                model_trace = trace(
                    substitute(
                        seed(kernel, rng_seed=rng_key),
                        substitute_fn=init_to_sample,
                    ),
                ).get_trace(**_combine(arrays, static))
            # Each value keeps only its shape and type
            sites.update(
                {
                    k: {
                        **v,
                        "value": ShapeDtypeStruct(
                            v["value"].shape,
                            v["value"].dtype,
                        ),
                    }
                    if hasattr(v.get("value"), "shape")
                    else v
                    for k, v in model_trace.items()
                },
            )

        eval_shape(record, arrays)

        return sites

    # An abstract trace runs nothing; a kernel that reads a concrete value of an
    # argument, as through `int`, runs once instead, on JAX arrays that the
    # dependency trace below then keeps as given
    try:
        model_trace = sites_at(leaves)
    except Exception:
        args_bound = _to_jax(args_bound)
        leaves = _partition(args_bound)[0]
        with validation_enabled(is_validate=False):
            model_trace = trace(
                substitute(
                    seed(kernel, rng_seed=rng_key),
                    substitute_fn=init_to_sample,
                ),
            ).get_trace(**args_bound)
    _validate_kernel_body(
        param_output=param_output,
        model_trace=model_trace,
        with_output=with_output,
    )
    sample_sites = tuple(k for k, v in model_trace.items() if v["type"] == "sample")
    return_sites = (
        param_output,
        *tuple(
            k
            for k, v in model_trace.items()
            if v["type"] == "deterministic" and k != param_output
        ),
    )
    # Dimension names: the plates, outermost first, then the event dimensions named
    # in the site's `infer` dictionary, none of them a site's in this trace or an
    # earlier one. A site that `scan` records keeps the default names, as `scan`
    # stacks its values along a leading time axis and `cond` does not. Only sample
    # sites carry the mark of `scan`, so a deterministic site recorded by control
    # flow keeps them only in a trace with a marked sample site.
    rows = getattr(args_bound[param_input], "shape", ())[:1]
    prev = kernel_spec
    scanned = {
        k
        for k, v in model_trace.items()
        if "_scan_current_index" in (v.get("infer") or {})
    }
    taken = {"chain", "draw", *sample_sites, *return_sites}
    if prev is not None and prev.traced:
        taken |= {*prev.sample_sites, *prev.return_sites}
    events = []
    for k, v in model_trace.items():
        if v["type"] not in {"sample", "deterministic"}:
            continue
        frames = sorted(v.get("cond_indep_stack") or (), key=lambda f: f.dim)
        event_dims = (v.get("infer") or {}).get("event_dims", ())
        names = (*(f.name for f in frames), *event_dims)
        shape = getattr(v["value"], "shape", ())
        names_apply = (
            k not in scanned
            and not (
                scanned and v["type"] == "deterministic" and v.get("_control_flow_done")
            )
            and 0 < len(names) <= len(shape)
            and all(isinstance(name, str) for name in names)
            and len(set(names)) == len(names)
            and not taken & set(names)
            and shape[: len(frames)] == tuple(f.size for f in frames)
        )
        if names_apply:
            dims[k] = names
        else:
            # Names that an earlier trace applied give way too
            dims.pop(k, None)
            if event_dims:
                msg = (
                    f"The event dimension names {tuple(event_dims)!r} of site "
                    f"{k!r} cannot apply, so the site keeps the default dimension "
                    "names."
                )
                warn(
                    msg,
                    category=OutputWarning,
                    skip_file_prefixes=_SKIP_FILE_PREFIXES,
                )
        # Values per draw: per observation when the leading axis follows the input
        # rows, else whole
        if hasattr(v["value"], "shape"):
            site_sizes[k] = (
                (math.prod(shape[1:]), 0)
                if shape[:1] == rows
                else (0, math.prod(shape))
            )
        if v.get("is_observed"):
            observed_sites.add(k)
        if v["type"] == "sample":
            # The distribution's name, under the wrappers that only reshape it,
            # as its factory spells it where the class carries a suffix
            base = v["fn"]
            while isinstance(
                base, (ExpandedDistribution, Independent, MaskedDistribution)
            ):
                base = base.base_dist
            name = type(base).__name__
            short = name.removesuffix("Probs").removesuffix("Logits")
            site_dists[k] = short if hasattr(distributions, short) else name
        # A site recorded by control flow, which NumPyro marks alike for `scan` and
        # `cond`, or drawn as one event over the rows, counts as computing across
        # observations. Every trace adds its marks, which every call reads, so a
        # site marked only with some arguments, as in a forecast, stays marked. The
        # first trace and each trace with the output also decide anew for the
        # events whose leading axis, batch or event, is not the rows, so a trace
        # over all the rows clears a one-element event that a one-row trace took for
        # one over the rows.
        fn = v.get("fn")
        if v.get("_control_flow_done"):
            coupled_sites.add(k)
        elif (
            isinstance(fn, Distribution)
            and not isinstance(fn, Independent)
            and fn.batch_shape[:1] != rows
            and fn.event_shape[:1] == rows
        ):
            coupled_sites.add(k)
            events.append(k)
        elif (
            (prev is None or not prev.traced or with_output)
            and isinstance(fn, Distribution)
            and fn.event_shape
            and fn.shape()[:1] != rows
        ):
            coupled_sites.discard(k)
    # An event that keeps its length on more rows is global, as long as the rows
    # only by chance; a kernel that does not run on more rows keeps every mark
    if events:
        more = rows_more or 2 * math.prod(rows)
        try:
            grows = [
                name not in whole
                for name, value in args_bound.items()
                for _ in _partition({name: value})[0]
            ]
            wider = sites_at(
                [
                    ShapeDtypeStruct(
                        (more, *leaf.shape[1:])
                        if grow and leaf.shape[:1] == rows
                        else leaf.shape,
                        jnp.result_type(leaf),
                    )
                    for leaf, grow in zip(leaves, grows, strict=True)
                ],
            )
        except Exception:
            wider = {}
        coupled_sites.difference_update(
            k
            for k in events
            if (site := wider.get(k))
            and getattr(site.get("fn"), "event_shape", ())[:1] == rows
            # The value too, since a distribution may take its event size from an
            # axis that the wider trace keeps, as `scale_tril` does
            and getattr(site["value"], "shape", ())[-len(site["fn"].event_shape) :]
            == site["fn"].event_shape
        )
    # Abstract arrays stand in for NumPy arrays, uncopied, so that the kernel's
    # operations on them are recorded, not run, as on JAX arrays; JAX arrays and
    # scalars stay as given, as `_to_jax` keeps them. JAX compares the site scales
    # of NumPyro's abstract trace with those of an earlier one; an enclosing trace
    # would record that comparison instead of running it, so NumPyro's trace runs
    # in the eval context, and without NumPy arrays nothing encloses it
    deps: dict[str, object] = {"prior_dependencies": {}}
    with suppress(Exception):
        if not any(isinstance(leaf, np.ndarray) and leaf.ndim for leaf in leaves):
            deps = get_dependencies(kernel, model_kwargs=dict(args_bound))
        else:
            eval_shape(
                lambda arrays: deps.update(
                    eval_context()(get_dependencies)(
                        kernel,
                        model_kwargs=_combine(
                            (
                                a
                                if isinstance(leaf, np.ndarray) and leaf.ndim
                                else leaf
                                for a, leaf in zip(arrays, leaves, strict=True)
                            ),
                            static,
                        ),
                    ),
                ),
                leaves,
            )
    for k, v in cast("Mapping[str, Mapping]", deps["prior_dependencies"]).items():
        site_parents[k] = set(v) - {k}
    # The sites and arguments each site is computed from, for `describe`
    arguments = {}
    for name, value in args_bound.items():
        if leaves_of := _partition({name: value})[0]:
            arguments[f"argument:{name}"] = [
                ShapeDtypeStruct(leaf.shape, jnp.result_type(leaf))
                for leaf in leaves_of
            ]
    reported = {
        k
        for k, v in model_trace.items()
        if v["type"] in {"sample", "deterministic"} and not v.get("_control_flow_done")
    }
    sites = {
        f"site:{k}": model_trace[k]["value"]
        for k in reported
        if model_trace[k]["type"] == "sample"
        and hasattr(model_trace[k]["value"], "shape")
    }

    def read(**tagged: object) -> dict[str, list]:
        computed: dict[str, list] = {}

        def substitute_fn(msg: dict) -> object:
            if msg["type"] == "sample":
                computed[msg["name"]] = tree.leaves(msg["fn"])

                return tagged.get(f"site:{msg['name']}")
            if msg["type"] == "deterministic":
                computed[msg["name"]] = [msg["value"]]

            return None

        with validation_enabled(is_validate=False):
            substitute(
                seed(kernel, rng_seed=rng_key),
                substitute_fn=substitute_fn,
            )(
                **_combine(
                    (
                        leaf
                        for name in args_bound
                        for leaf in cast("list", tagged.get(f"argument:{name}", ()))
                    ),
                    static,
                ),
            )

        return {k: v for k, v in computed.items() if k in reported}

    with suppress(Exception):
        for k, tags in eval_provenance(read, **arguments, **sites).items():
            found = frozenset().union(*tags)
            site_upstream[k] = {
                tag.removeprefix("site:") for tag in found if tag.startswith("site:")
            }
            site_arguments[k] = {
                tag.removeprefix("argument:")
                for tag in found
                if tag.startswith("argument:")
            }
    spec = KernelSpec(
        traced=True,
        sample_sites=sample_sites,
        return_sites=return_sites,
        output_observed=bool(with_output),
    )
    # A posterior covers the sample sites of the same traces, not those only a later
    # prediction samples, such as a forecast
    if with_output or prev is None or not prev.traced:
        fitted_sites.update(sample_sites)
    specs[static] = spec
    kernel_spec = (
        KernelSpec(
            traced=True,
            sample_sites=tuple(dict.fromkeys(prev.sample_sites + sample_sites)),
            return_sites=tuple(dict.fromkeys(return_sites + prev.return_sites)),
            output_observed=prev.output_observed or bool(with_output),
        )
        if prev is not None and prev.traced
        else spec
    )

    return spec, kernel_spec
