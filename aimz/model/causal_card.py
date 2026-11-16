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

"""Dataclass for the causal scope a modeler declares."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class CausalCard:
    """The causal scope a modeler declares for an :class:`~aimz.ImpactModel`.

    aimz derives none of this from the kernel. The card records what the modeler
    asserts and travels with the model. ``unit`` is checked with the names the
    treatments span, and the other fields are carried for the readers of a fitted
    model. Sequences passed as lists are stored as tuples, and ``treatments`` as a
    mapping of each treatment's name to the names it spans; a single name may be
    given as a string.

    Raises:
        ValueError: If a feature name is repeated or a treatment spans no name.

    Attributes:
        features: Names of the input's features, in order: the entries along its
            second axis. Each name must be unique, and differ from ``X`` and from the
            keyword arrays of a scenario, which are addressed by the same names.
        treatments: The inputs that interventions change, by treatment: a mapping of
            a treatment's name to the names it spans, for a treatment recorded in
            several features, or a sequence of names, each a treatment of its own.
            The names spanned are declared features, the input ``X``, keyword
            arguments of the kernel, or sample sites.
        outcome: Name of the site whose contrast is reported as the effect, such as a
            deterministic site holding the expected outcome; ``None`` leaves it to the
            output parameter.
        unit: Name of the feature or keyword argument that identifies repeated
            observations of one unit.
        post_treatment: Names of variables downstream of the treatments, such as
            labels an effect is grouped by; a contrast within such groups is
            descriptive.
        adjustment: Names the modeler claims adjust for confounding.
        extra: Anything else the modeler attaches, such as notes or evidence; stored
            and returned untouched. In the ``MLmodel`` record, a value JSON cannot
            hold is written as text, and a key it cannot hold, such as a tuple, makes
            :func:`~aimz.mlflow.save_model` fail.
    """

    features: tuple[str, ...] = ()
    treatments: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    outcome: str | None = None
    unit: str | None = None
    post_treatment: tuple[str, ...] = ()
    adjustment: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Store the sequences as tuples, treatments and ``extra`` as dictionaries."""
        for name in ("features", "post_treatment", "adjustment"):
            names = getattr(self, name)
            names = (names,) if isinstance(names, str) else tuple(names)
            object.__setattr__(self, name, names)
        if len(set(self.features)) != len(self.features):
            msg = f"The feature names must be unique, got {self.features}."
            raise ValueError(msg)
        treatments = self.treatments
        if isinstance(treatments, str):
            treatments = (treatments,)
        if not isinstance(treatments, Mapping):
            treatments = {name: (name,) for name in treatments}
        object.__setattr__(
            self,
            "treatments",
            {
                str(name): (names,) if isinstance(names, str) else tuple(names)
                for name, names in treatments.items()
            },
        )
        if empty := [name for name, names in self.treatments.items() if not names]:
            msg = f"Each treatment must span a name, got none for {empty}."
            raise ValueError(msg)
        object.__setattr__(self, "extra", dict(self.extra))

    def to_dict(self) -> dict[str, Any]:
        """Return the card as plain values, the form an ``MLmodel`` file records."""
        return {
            "features": list(self.features),
            "treatments": {
                name: list(names) for name, names in self.treatments.items()
            },
            "outcome": self.outcome,
            "unit": self.unit,
            "post_treatment": list(self.post_treatment),
            "adjustment": list(self.adjustment),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CausalCard:
        """Build a card from the plain values of :meth:`to_dict`."""
        return cls(**data)
