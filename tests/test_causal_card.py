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

import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import plate, sample

from aimz import CausalCard, ImpactModel
from tests.conftest import lm, lm_with_kwargs_array, make_svi


def lm_other_input(data: Array, y: Array | None = None) -> None:
    """Linear regression whose input parameter is not called `X`."""
    w = sample("w", dist.Normal(0.0, 1.0).expand([data.shape[1]]).to_event(1))
    with plate("obs", data.shape[0]):
        sample("y", dist.Normal(data @ w, 1.0), obs=y)


def test_causal_card_workflow(synthetic_data: tuple[Array, Array]) -> None:
    """A card is attached, described, serialized, and checked against the kernel."""
    X, y = synthetic_data
    features = tuple(f"x{j}" for j in range(X.shape[1]))
    card = CausalCard(
        features=list(features),
        treatments="x0",
        post_treatment="decile",
        extra={"verdict": "within-unit"},
    )
    # Lists are stored as tuples, a single string as one name, treatments as a
    # mapping of each to the names it spans, repeated feature names and a treatment
    # that spans nothing raise, and the plain form round-trips through JSON
    assert card.features == features
    assert card.treatments == {"x0": ("x0",)}
    assert card.post_treatment == ("decile",)
    with pytest.raises(ValueError, match="unique"):
        CausalCard(features=["x0", "x0"])
    with pytest.raises(ValueError, match="span"):
        CausalCard(treatments={"x": []})
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

    # The whole input is `X` in a card whatever the kernel calls it, and a treatment
    # that spans it spans the declared features
    im_in = ImpactModel(
        lm_other_input,
        rng_key=random.key(0),
        inference=make_svi(lm_other_input),
        param_input="data",
    )
    im_in.fit_on_batch(X, y, num_steps=5, num_samples=10, progress=False)
    with pytest.raises(ValueError, match="data"):
        im_in.card = CausalCard(treatments=["data"])
    im_in.card = CausalCard(features=features, treatments={"all": "X"})
    scenario = im_in.build_scenario({"X": X}, {"all": 0.0})
    assert im_in.check_scenario({"X": X}, scenario)["undeclared"] == []
