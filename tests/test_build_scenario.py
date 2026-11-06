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

"""Tests for the `.build_scenario()` method."""

import numpy as np
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import plate, sample

from aimz import CausalCard, ImpactModel, OutputWarning
from tests.conftest import lm, lm_with_kwargs_array, make_svi


def unused_argument(X: Array, u: Array, y: Array | None = None) -> None:
    """Linear regression with an argument the kernel never reads."""
    w = sample("w", dist.Normal(0.0, 1.0).expand([X.shape[1]]).to_event(1))
    with plate("obs", X.shape[0]):
        sample("y", dist.Normal(X @ w, 1.0), obs=y)


def test_build_scenario_workflow(synthetic_data: tuple[Array, Array]) -> None:
    """A scenario is built from the baseline by name, checked and estimated."""
    X, y = synthetic_data
    X_b = np.asarray(X)
    n_obs, n_cols = X_b.shape
    features = tuple(f"x{j}" for j in range(n_cols))
    im = ImpactModel(lm, rng_key=random.key(0), inference=make_svi(lm))
    args = {"X": X_b}

    # Without a card, a feature goes by `X[i]`, and the baseline is left as it is
    scenario = im.build_scenario(args, {"X[3]": 0.0})
    assert (scenario["X"][:, 3] == 0.0).all()
    assert np.array_equal(np.delete(scenario["X"], 3, axis=1), np.delete(X_b, 3, 1))
    assert (X_b[:, 3] != 0.0).any()

    # With a card, a treatment edits every feature it spans, a callable transforms
    # the current values, a per-observation value broadcasts over the features, and
    # `observations` selects the ones to change, as a mask or as indices; once the
    # kernel is traced,
    # an edit of the input warns of nothing
    im.card = CausalCard(features=features, treatments={"t": ("x0", "x1"), "x2": "x2"})
    im.fit_on_batch(X_b, y, num_steps=10, num_samples=5, progress=False)
    halved = im.build_scenario(args, {"t": lambda v: 0.5 * v, "x2": 1.0})
    assert np.allclose(halved["X"][:, :2], 0.5 * X_b[:, :2])
    assert (halved["X"][:, 2] == 1.0).all()
    assert np.array_equal(halved["X"][:, 3:], X_b[:, 3:])
    per_obs = im.build_scenario(args, {"t": np.arange(n_obs, dtype=np.float32)})
    assert np.array_equal(per_obs["X"][:, :2], np.tile(np.arange(n_obs), (2, 1)).T)
    mask = X_b[:, 5] > 0
    by_mask = im.build_scenario(args, {"t": 0.0}, observations=mask)
    by_index = im.build_scenario(
        args,
        {"t": 0.0},
        observations=np.flatnonzero(mask),
    )
    assert np.array_equal(by_mask["X"], by_index["X"])
    assert (by_mask["X"][mask, :2] == 0.0).all()
    assert np.array_equal(by_mask["X"][~mask], X_b[~mask])

    # The built scenario is what the check and the estimate take
    report = im.check_scenario(args, by_mask)
    assert report["changed"] == {"x0": mask.mean(), "x1": mask.mean()}
    effect = im.estimate_effect(
        args_baseline=args,
        args_intervention=by_mask,
        on_batch=True,
    )
    assert effect.posterior_predictive["y"].shape[-1] == n_obs

    # A keyword array is edited by its argument name, with `observations` along its
    # leading axis, and the other entries come back as given; a keyword array the kernel
    # computes no site from warns once the kernel is traced
    im_kw = ImpactModel(
        lm_with_kwargs_array,
        rng_key=random.key(0),
        inference=make_svi(lm_with_kwargs_array),
    )
    c = np.ones(n_obs)
    shifted = im_kw.build_scenario(
        {"X": X_b, "c": c, "store": "memory"},
        {"c": lambda v: v + 1.0},
        observations=mask,
    )
    assert np.array_equal(shifted["c"], np.where(mask, 2.0, 1.0))
    assert shifted["X"] is X_b
    im_unused = ImpactModel(
        unused_argument,
        rng_key=random.key(0),
        inference=make_svi(unused_argument),
    )
    im_unused.fit_on_batch(X_b, y, u=c, num_steps=10, num_samples=5, progress=False)
    with pytest.warns(OutputWarning, match="no site"):
        im_unused.build_scenario({"X": X_b, "u": c}, {"u": 0.0})

    # A name that is not an input, a value that does not broadcast, `observations`
    # that do not fit, a sample site, `observations` for an input without any, a card
    # whose features do not fit, and a data loader all raise
    for changes, kwargs, match in (
        ({"nope": 0.0}, {}, "none of the inputs"),
        ({"x0": np.zeros(3)}, {}, "broadcast"),
        ({"x0": 0.0}, {"observations": mask[:10]}, "`observations`"),
        ({"x0": 0.0}, {"observations": np.array([n_obs])}, "`observations`"),
    ):
        with pytest.raises(ValueError, match=match):
            im.build_scenario(args, changes, **kwargs)
    im.card = CausalCard(features=features, treatments={"w": "w"})
    with pytest.raises(ValueError, match="sample site"):
        im.build_scenario(args, {"w": 0.0})
    with pytest.raises(ValueError, match="per observation"):
        im_kw.build_scenario(
            {"X": X_b, "c": np.ones(3)},
            {"c": 0.0},
            observations=mask,
        )
    im.card = CausalCard(features=("a",), treatments=("a",))
    with pytest.raises(ValueError, match="feature"):
        im.build_scenario(args, {"a": 0.0})
    with pytest.raises(TypeError, match="array"):
        im.build_scenario({"X": [X_b]}, {"X": 0.0})
