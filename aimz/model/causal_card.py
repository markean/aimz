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


@dataclass(frozen=True)
class CausalCard:
    """The causal scope a modeler declares for an :class:`~aimz.ImpactModel`.

    aimz derives none of this from the kernel. The card records what the modeler
    asserts, travels with the model, and lets :meth:`~aimz.ImpactModel.check_scenario`
    compare a scenario against it. Only ``features``, ``treatments`` and
    ``post_treatment`` change what aimz does; the other fields are carried for the
    readers of a fitted model. Sequences passed as lists are stored as tuples, and
    ``treatments`` as a mapping of each treatment's name to the names it spans.

    Attributes:
        features: Names of the input's features, in order, for a two-dimensional input.
        treatments: The inputs that interventions change, by treatment: a mapping of
            a treatment's name to the names it spans, for a treatment recorded in
            several features, or a sequence of names, each a treatment of its own.
            The names spanned are declared features, keyword arguments of the kernel,
            or sample sites.
        outcome: Name of the site a consumer reports; ``None`` leaves it to the output
            parameter.
        unit: Name of the feature or keyword argument that identifies repeated
            observations of one unit.
        post_treatment: Names of variables downstream of the treatments, such as
            labels an effect is grouped by; a contrast within such groups is
            descriptive.
        adjustment: Names the modeler claims adjust for confounding.
        graph: Directed edges ``(parent, child)`` of the causal graph the modeler
            declares.
        extra: Anything else the modeler attaches, such as evidence, thresholds or a
            design verdict; stored and returned untouched.
    """

    features: tuple[str, ...] = ()
    treatments: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    outcome: str | None = None
    unit: str | None = None
    post_treatment: tuple[str, ...] = ()
    adjustment: tuple[str, ...] = ()
    graph: tuple[tuple[str, str], ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Store the sequences as tuples, treatments and ``extra`` as dictionaries."""
        for name in ("features", "post_treatment", "adjustment"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
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
        object.__setattr__(self, "graph", tuple(tuple(edge) for edge in self.graph))
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
            "graph": [list(edge) for edge in self.graph],
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CausalCard:
        """Build a card from the plain values of :meth:`to_dict`."""
        return cls(**data)
