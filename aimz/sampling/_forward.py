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

from typing import TYPE_CHECKING, Any

from jax import Array, lax, random
from jax.typing import ArrayLike
from numpyro.handlers import mask, seed, substitute, trace

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable


def _sample_forward(
    model: "Callable",
    num_samples: int,
    rng_key: ArrayLike,
    return_sites: tuple[str] | None,
    samples: dict[str, Array] | None,
    model_kwargs: dict[str, Array] | None,
) -> dict[str, Array]:
    """Generates forward samples from a model conditioned on parameter draws.

    This function repeatedly traces the model with different random keys and sets of
    parameter values. Deterministic sites in the model are excluded from substitution.

    Args:
        model (Callable): A probabilistic model with Pyro primitives.
        num_samples (int): The number of samples to draw.
        rng_key (ArrayLike): A pseudo-random number generator key.
        return_sites (tuple[str] | None): Names of variables (sites) to return.
        samples (dict[str, Array]| None): A dictionary of samples to condition on,
            where each array has shape ``(num_samples, ...)``.
        model_kwargs (dict[str, Array] | None): Additional arguments passed to the
            model.

    Returns:
        dict[str, Array]: A dictionary mapping each return site to an array of traced
            values with shape ``(num_samples, ...)``.
    """

    def _trace_one_sample(
        sample_input: tuple[Array, dict[str, Array]],
    ) -> dict[str, Array]:
        rng_key, sample = sample_input

        def _exclude_deterministic(msg: "OrderedDict[str, Any]") -> Array | None:
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

    rng_keys = random.split(rng_key, num=num_samples).reshape((num_samples,))

    return lax.map(_trace_one_sample, xs=(rng_keys, samples or {}))
