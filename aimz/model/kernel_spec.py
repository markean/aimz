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

"""Dataclass for kernel metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelSpec:
    """A dataclass describing the kernel structure.

    Stores only the minimal information needed for downstream operations without
    re-tracing the NumPyro model.

    Attributes:
        traced:
            ``True`` once at least one successful trace has been performed.
        sample_sites:
            Names of latent (stochastic) sample sites encountered during tracing.
        return_sites:
            Names of default return sites. Always includes the output first, followed by
            any deterministic sites. Latent sample sites are excluded.
        output_observed:
            ``True`` if the output site was observed in the validating trace, used to
            distinguish prior-only from observed traces.

    Notes:
        Re-tracing only upgrades a prior-only spec (``output_observed`` is ``False``) to
        one that includes an observed output; the user kernel is assumed immutable after
        construction.
    """

    traced: bool
    sample_sites: tuple[str, ...]
    return_sites: tuple[str, ...]
    output_observed: bool
