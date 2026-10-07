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
    """The kernel structure that a trace recorded.

    A trace with an observed output upgrades a spec traced without one, merging the
    sites. Sites that appear only under specific arguments are not discovered and must
    be requested through ``return_sites``.

    Attributes:
        traced: Whether a trace has run.
        sample_sites: The sample sites, latent and observed, seen in traces.
        return_sites: The default return sites: the output, then the deterministic
            sites.
        output_observed: Whether the output site was observed in the validating trace.
    """

    traced: bool
    sample_sites: tuple[str, ...]
    return_sites: tuple[str, ...]
    output_observed: bool
