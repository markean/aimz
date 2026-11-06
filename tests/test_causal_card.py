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

"""Tests for `CausalCard` and the `card` property."""

import json

import pytest
from jax import Array, random

from aimz import CausalCard, ImpactModel
from tests.conftest import lm, lm_with_kwargs_array, make_svi


def test_causal_card_workflow(synthetic_data: tuple[Array, Array]) -> None:
    """A card is attached, described, serialized, and checked against the kernel."""
    X, y = synthetic_data
    features = tuple(f"x{j}" for j in range(X.shape[1]))
    card = CausalCard(
        features=list(features),
        treatments=["x0"],
        post_treatment=["decile"],
        graph=[["x0", "y"]],
        extra={"verdict": "within-unit"},
    )
    # Lists are stored as tuples, treatments as a mapping of each to the names it
    # spans, and the plain form round-trips through JSON
    assert card.features == features
    assert card.treatments == {"x0": ("x0",)}
    assert card.graph == (("x0", "y"),)
    assert CausalCard.from_dict(json.loads(json.dumps(card.to_dict()))) == card
    # A treatment that spans several inputs is given as a mapping, with a bare name
    # standing for one input
    grouped = CausalCard(features=features, treatments={"x": ["x0", "x1"], "x2": "x2"})
    assert grouped.treatments == {"x": ("x0", "x1"), "x2": ("x2",)}
    assert grouped.to_dict()["treatments"] == {"x": ["x0", "x1"], "x2": ["x2"]}
    assert CausalCard.from_dict(grouped.to_dict()) == grouped

    im = ImpactModel(lm, rng_key=random.key(0), inference=make_svi(lm), card=card)
    assert im.card is card
    described = im.describe()
    assert described["card"]["treatments"] == {"x0": ["x0"]}
    assert described["card"]["extra"] == {"verdict": "within-unit"}
    json.dumps(described)

    # Before a trace, a name that is neither a feature nor an argument may be a site
    im.card = CausalCard(features=features, treatments=("w",))
    im.fit_on_batch(X, y, num_steps=5, num_samples=10, progress=False)
    # Once traced, a sample site passes and an unknown name does not, also when a
    # treatment of a free name spans it
    im.card = CausalCard(features=features, treatments=("w",), unit="x1")
    with pytest.raises(ValueError, match="nope"):
        im.card = CausalCard(features=features, treatments=("nope",))
    with pytest.raises(ValueError, match="nope"):
        im.card = CausalCard(features=features, treatments={"group": ("x0", "nope")})
    assert im.card.treatments == {"w": ("w",)}

    # A keyword argument of the kernel is a valid treatment and unit; None clears
    im_kw = ImpactModel(
        lm_with_kwargs_array,
        rng_key=random.key(0),
        inference=make_svi(lm_with_kwargs_array),
        card=CausalCard(treatments=("c",), unit="c"),
    )
    assert im_kw.card is not None
    im_kw.card = None
    assert im_kw.describe()["card"] is None
