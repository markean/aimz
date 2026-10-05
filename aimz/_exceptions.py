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

"""Custom warnings and errors."""

from os import sep
from pathlib import Path

__all__ = [
    "AimzWarning",
    "FitWarning",
    "KernelValidationError",
    "NotFittedError",
    "OutputWarning",
    "PerformanceWarning",
]

# Warnings skip the frames in this package to point at the caller's own line
_SKIP_FILE_PREFIXES = (f"{Path(__file__).parent}{sep}",)


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if model is used before fitting."""


class KernelValidationError(Exception):
    """Exception class to raise if kernel validation fails."""


class AimzWarning(UserWarning):
    """Base class for warnings issued by aimz."""


class FitWarning(AimzWarning, RuntimeWarning):
    """Warning class for a fit whose result may be unreliable."""


class OutputWarning(AimzWarning):
    """Warning class for an output that may differ from what was expected."""


class PerformanceWarning(AimzWarning):
    """Warning class for a call that runs correctly but by a fallback or slower path."""
