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
    """The kernel structure that its traces recorded.

    The training and predictive methods trace the kernel once for each set of call
    arguments, told apart by their names and values other than arrays, and each trace
    merges its sites into the spec, so a site that the kernel records only for some
    arguments is in it once such a call with them has run. A posterior predictive call
    with a data loader, which has nothing to trace before its batches, and, on a traced
    kernel, :meth:`~aimz.ImpactModel.log_likelihood` and a predictive call whose trace
    fails, as when the kernel names the output otherwise without data, use the sites
    traced so far.

    Attributes:
        traced: Whether a trace has run.
        sample_sites: The sample sites, latent and observed, seen in traces.
        return_sites: The default return sites: the output, then the deterministic
            sites.
        output_observed: Whether a trace observed the output site.
    """

    traced: bool
    sample_sites: tuple[str, ...]
    return_sites: tuple[str, ...]
    output_observed: bool
