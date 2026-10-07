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

"""Forward sampling implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jax import Array, lax, vmap
from jax.core import Tracer
from numpyro.handlers import do, mask, seed, substitute, trace

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable, Mapping


def _sample_forward(
    model: Callable,
    *,
    rng_keys: Array,
    return_sites: tuple[str, ...] | None,
    samples: dict[str, Array] | None,
    params: dict[str, Array] | None,
    intervention: dict | None,
    model_kwargs: Mapping[str, object] | None,
) -> dict[str, Array]:
    """Draw forward samples from a model, one trace per key in ``rng_keys``.

    Each draw conditions on its row of ``samples``, deterministic sites excepted. Under
    a trace the draws run as one :external:func:`jax.lax.map` loop; eagerly they are
    vectorized with :external:func:`jax.vmap`, since an eager ``lax.map`` would cache a
    program per call.

    Args:
        model: A probabilistic model with NumPyro primitives.
        rng_keys: Per-draw keys with shape ``(num_samples,)``.
        return_sites: Names of the sites to return; by default every sample site not
            conditioned on and every deterministic site.
        samples: Samples to condition on, with the draws on the leading axis.
        params: Values of the model's ``param`` sites and mutable state, shared by all
            draws.
        intervention: Replacement values by sample site name.
        model_kwargs: Arguments passed to the model.

    Returns:
        The traced values of each return site, with the draws on the leading axis.
    """
    if params:
        model = substitute(model, data=params)
    if intervention:
        model = do(model, data=intervention)

    def _trace_one_sample(
        sample_input: tuple[Array, dict[str, Array]],
    ) -> dict[str, Array]:
        rng_key, sample = sample_input

        def _exclude_deterministic(msg: OrderedDict[str, Any]) -> Array | None:
            return sample.get(msg["name"]) if msg["type"] != "deterministic" else None

        masked_model = mask(model, mask=False)
        substituted_model = substitute(
            masked_model,
            substitute_fn=_exclude_deterministic,
        )
        model_trace = trace(seed(substituted_model, rng_seed=rng_key)).get_trace(
            **(model_kwargs or {}),
        )

        if return_sites is None:
            sites = {
                k
                for k, site in model_trace.items()
                if (site["type"] == "sample" and k not in sample)
                or (site["type"] == "deterministic")
            }
        else:
            sites = set(return_sites)

        return {k: v["value"] for k, v in model_trace.items() if k in sites}

    if isinstance(rng_keys, Tracer):
        return lax.map(_trace_one_sample, xs=(rng_keys, samples or {}))

    return vmap(_trace_one_sample)((rng_keys, samples or {}))
