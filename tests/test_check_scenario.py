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

"""Tests for the `.check_scenario()` method."""

import json

import numpy as np
import pytest
from jax import Array, random

from aimz import CausalCard, ImpactModel, ScopeWarning
from tests.conftest import lm, lm_with_kwargs_array, make_svi


def test_check_scenario_facts(synthetic_data: tuple[Array, Array]) -> None:
    """The check reports changes, extrapolation and support as plain values."""
    X, _ = synthetic_data
    X_b = np.asarray(X)
    n_obs, n_cols = X_b.shape
    features = tuple(f"x{j}" for j in range(n_cols))
    card = CausalCard(features=features, treatments=("x0", "x1"))
    im = ImpactModel(lm, rng_key=random.key(0), inference=make_svi(lm), card=card)

    # Zero out a declared treatment on half the observations
    share = 0.5
    X_i = X_b.copy()
    X_i[: int(n_obs * share), 0] = 0.0
    out = im.check_scenario({"X": X_b}, {"X": X_i})
    assert out["n_obs"] == n_obs
    assert out["changed"] == {"x0": share}
    assert out["obs_changed"] == share
    assert out["undeclared"] == []
    # Zero lies inside the range of a standard normal feature
    assert out["extrapolation"] == {"x0": 0.0}
    # The pattern is the presence of the declared treatments; every baseline row has
    # both, so the zeroed rows show a pattern the baseline never holds
    assert out["pattern"] == {"x0": ["x0"], "x1": ["x1"]}
    assert out["support"] == 0.0
    assert out["crowded"] == 1.0
    assert out["post_treatment"] == []
    json.dumps(out)

    # A treatment that spans several features declares a change to any of them and
    # is one group of the default pattern; the group is present wherever one of its
    # features is, so zeroing `x0` alone leaves every pattern as in the baseline
    im.card = CausalCard(features=features, treatments={"x": ("x0", "x1"), "x2": "x2"})
    out_group = im.check_scenario({"X": X_b}, {"X": X_i})
    assert out_group["changed"] == {"x0": share}
    assert out_group["undeclared"] == []
    assert out_group["pattern"] == {"x": ["x0", "x1"], "x2": ["x2"]}
    assert out_group["support"] == 1.0
    assert out_group["crowded"] == 0.0

    # A keyword-array treatment, with the pattern given as a group and the absence
    # value moved; the method's own parameters in the dictionaries are not compared
    im_kw = ImpactModel(
        lm_with_kwargs_array,
        rng_key=random.key(0),
        inference=make_svi(lm_with_kwargs_array),
        card=CausalCard(treatments=("c",)),
    )
    c_b, c_i = np.ones(n_obs), np.zeros(n_obs)
    out_kw = im_kw.check_scenario(
        {"X": X_b, "c": c_b, "return_sites": "mu"},
        {"X": X_b, "c": c_i, "return_sites": "mu"},
        pattern={"exposed": ("c",)},
    )
    assert out_kw["changed"] == {"c": 1.0}
    assert out_kw["pattern"] == {"exposed": ["c"]}
    assert out_kw["support"] == 0.0
    # With one as the absence value, the counterfactual zeros are the present ones,
    # a pattern the baseline never holds either
    out_abs = im_kw.check_scenario(
        {"X": X_b, "c": c_b},
        {"X": X_b, "c": c_i},
        pattern=("c",),
        absent={"c": 1.0},
    )
    assert out_abs["support"] == 0.0
    # A changed scalar keyword argument changes every observation, and one the card
    # does not declare is outside the scope
    with pytest.warns(ScopeWarning, match="scale"):
        out_scalar = im_kw.check_scenario(
            {"X": X_b, "c": c_b, "scale": 1.0},
            {"X": X_b, "c": c_b, "scale": 2.0},
        )
    assert out_scalar["changed"] == {"scale": 1.0}
    assert out_scalar["obs_changed"] == 1.0

    # Without a card the features are positional and nothing is undeclared
    im_plain = ImpactModel(lm, rng_key=random.key(0), inference=make_svi(lm))
    out_plain = im_plain.check_scenario({"X": X_b}, {"X": X_i})
    assert out_plain["changed"] == {"X[0]": share}
    assert out_plain["undeclared"] == []
    assert out_plain["pattern"] == {"X[0]": ["X[0]"]}
    # An intervention mapping is compared by site
    out_site = im_plain.check_scenario(
        {"X": X_b, "intervention": {"b": 0.0}},
        {"X": X_b, "intervention": {"b": 1.0}},
    )
    assert out_site["changed"] == {"b": 1.0}


def test_check_scenario_scope_and_errors(synthetic_data: tuple[Array, Array]) -> None:
    """Changes outside the declared scope warn, and misfit inputs raise."""
    X, _ = synthetic_data
    X_b = np.asarray(X)
    n_obs, n_cols = X_b.shape
    features = tuple(f"x{j}" for j in range(n_cols))
    card = CausalCard(
        features=features,
        treatments=("x0", "x1"),
        post_treatment=("decile",),
    )
    im = ImpactModel(lm, rng_key=random.key(0), inference=make_svi(lm), card=card)
    X_i = X_b.copy()
    X_i[:, 0] = 0.0

    # A change the card does not declare warns, and here lies outside the range
    X_u = X_b.copy()
    X_u[:, 2] = X_b[:, 2] + 100.0
    with pytest.warns(ScopeWarning, match="x2"):
        out_u = im.check_scenario({"X": X_b}, {"X": X_u})
    assert out_u["undeclared"] == ["x2"]
    assert out_u["extrapolation"] == {"x2": 1.0}

    # Grouping by a post-treatment variable warns
    with pytest.warns(ScopeWarning, match="decile"):
        out_by = im.check_scenario(
            {"X": X_b},
            {"X": X_i},
            by={"decile": np.zeros(n_obs)},
        )
    assert out_by["post_treatment"] == ["decile"]

    # Unchanged scenarios: nothing changed, no pattern to check
    out_same = im.check_scenario({"X": X_b}, {"X": X_b})
    assert out_same["changed"] == {}
    assert out_same["support"] is None

    # Inputs of different shapes, a card whose features do not fit, a pattern name
    # that is not per observation, and a data loader all raise
    with pytest.raises(ValueError, match="shape"):
        im.check_scenario({"X": X_b}, {"X": X_b[:10]})
    with pytest.raises(ValueError, match="pattern"):
        im.check_scenario({"X": X_b}, {"X": X_i}, pattern=("nope",))
    with pytest.raises(TypeError, match="array"):
        im.check_scenario({"X": [X_b]}, {"X": [X_i]})
    im_bad = ImpactModel(
        lm,
        rng_key=random.key(0),
        inference=make_svi(lm),
        card=CausalCard(features=("a",), treatments=("a",)),
    )
    with pytest.raises(ValueError, match="feature"):
        im_bad.check_scenario({"X": X_b}, {"X": X_i})
