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

"""Tests for the `.estimate_effect()` method."""

import warnings
from pathlib import Path

import jax.numpy as jnp
import pytest
from jax import Array, random

from aimz import ImpactModel, OutputWarning
from tests.conftest import latent_intervention_model, lm, make_svi


def test_estimate_effect_argument_validation(
    synthetic_data: tuple[Array, Array],
    im_latent_var_svi_fitted: ImpactModel,
) -> None:
    """Validate argument exclusivity and successful effect computation."""
    X, y = synthetic_data
    im = im_latent_var_svi_fitted

    msg = "Either `output_baseline` or `args_baseline` must be provided."
    with pytest.raises(ValueError, match=msg):
        im.estimate_effect(output_baseline=None, args_baseline=None)

    rng_key = random.key(0)
    dt_baseline = im.predict_on_batch(X, rng_key=rng_key)

    msg = "Either `output_intervention` or `args_intervention` must be provided."
    with pytest.raises(ValueError, match=msg):
        im.estimate_effect(output_baseline=dt_baseline)

    dt_intervention = im.predict_on_batch(
        X,
        intervention={"z": jnp.zeros_like(y)},
        rng_key=rng_key,
    )
    effect = im.estimate_effect(
        output_baseline=dt_baseline,
        output_intervention=dt_intervention,
    )

    assert effect.posterior_predictive["y"].mean(dim=["chain", "draw"]).shape == (
        len(y),
    )


def test_estimate_effect_lazy_args_record_artifact_paths(
    synthetic_data: tuple[Array, Array],
) -> None:
    """Lazily generated scenarios record their own subdirectories of the temp dir."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=make_svi(lm))
    im.fit_on_batch(X, y, num_steps=10, num_samples=10, progress=False)
    effect = im.estimate_effect(
        args_baseline={"X": X, "progress": False},
        args_intervention={"X": X, "intervention": {"sigma": 10.0}, "progress": False},
    )

    paths = [
        Path(effect.attrs[f"artifact_path_{scenario}"])
        for scenario in ("baseline", "intervention")
    ]
    assert paths[0] != paths[1]
    for path in paths:
        assert path.is_dir()
        assert path.parent == Path(im.temp_dir).resolve()
        # The subdir suffix is the outermost user-called method
        assert path.name.endswith("_estimate_effect")
    im.cleanup()


@pytest.mark.parametrize("in_sample", [True, False])
def test_estimate_effect_on_batch_dict(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    *,
    in_sample: bool,
) -> None:
    """`in_sample` picks the group of on-batch scenarios, even if dicts are asked."""
    X, _ = synthetic_data

    expected_group = "posterior_predictive" if in_sample else "predictions"

    effect = im_lm_svi_fitted.estimate_effect(
        args_baseline={"X": X, "return_datatree": False, "in_sample": in_sample},
        args_intervention={
            "X": X,
            "intervention": {"sigma": 10.0},
            "return_datatree": False,
            "in_sample": in_sample,
        },
        on_batch=True,
    )

    assert expected_group in effect.children


def test_estimate_effect_lazy_args_share_rng_key(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Lazily generated scenarios share one sampling key unless keys are given."""
    X, _ = synthetic_data
    im = im_lm_svi_fitted

    # Identical scenarios drawn with the shared key cancel draw by draw.
    effect = im.estimate_effect(
        args_baseline={"X": X},
        args_intervention={"X": X},
        on_batch=True,
    )
    assert (effect.posterior_predictive["y"] == 0).all()

    # Explicit keys are used as given, and the output is then not paired.
    with pytest.warns(OutputWarning, match="differ in 'rng_key'; their draws of 'y'"):
        effect = im.estimate_effect(
            args_baseline={"X": X, "rng_key": random.key(0)},
            args_intervention={"X": X, "rng_key": random.key(1)},
            on_batch=True,
        )
    expected = (
        im.predict_on_batch(X, rng_key=random.key(1)).posterior_predictive["y"]
        - im.predict_on_batch(X, rng_key=random.key(0)).posterior_predictive["y"]
    )
    assert effect.posterior_predictive["y"].equals(expected)


def test_estimate_effect_scenario_compatibility(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Scenarios must share a predictive group; differing sizes or labels warn."""
    X, y = synthetic_data
    im = im_lm_svi_fitted
    rng_key = random.key(0)
    base = im.predict_on_batch(X, rng_key=rng_key)

    # The baseline's group must be present in the intervention, and a tree without
    # any predictive group, such as a log-likelihood tree, has none to compare
    with pytest.raises(
        ValueError,
        match=r"Group 'posterior_predictive' not found in `dt_intervention`.",
    ):
        im.estimate_effect(
            output_baseline=base,
            output_intervention=im.predict_on_batch(X, in_sample=False),
        )
    with pytest.raises(
        ValueError,
        match=r"Group 'prior_predictive' not found in `dt_intervention`.",
    ):
        im.estimate_effect(
            output_baseline=im.sample_prior_predictive_on_batch(X),
            output_intervention=base,
        )
    with pytest.raises(
        ValueError,
        match=r"Group 'posterior_predictive' not found in `dt_baseline`.",
    ):
        im.estimate_effect(
            output_baseline=im.log_likelihood(X, y, store="memory", progress=False),
            output_intervention=base,
        )

    with pytest.warns(OutputWarning, match="different dimension sizes"):
        im.estimate_effect(
            output_baseline=base,
            output_intervention=im.predict_on_batch(X[:80], rng_key=rng_key),
        )
    with pytest.warns(OutputWarning, match="different coordinate labels"):
        im.estimate_effect(
            output_baseline=im.predict_on_batch(X[:50], rng_key=rng_key),
            output_intervention=base.isel(y_dim_0=slice(50, 100)),
        )


def test_estimate_effect_warns_on_stale_baseline(
    synthetic_data: tuple[Array, Array],
) -> None:
    """A baseline kept from before a refit warns; a deterministic site stays paired."""
    X, y = synthetic_data
    im = ImpactModel(
        latent_intervention_model,
        rng_key=random.key(42),
        inference=make_svi(latent_intervention_model),
    )
    im.fit_on_batch(X, y, num_steps=10, progress=False)
    baseline = im.predict_on_batch(X, rng_key=random.key(0), return_sites="mu")

    # A deterministic site is computed from the posterior samples alone, so scenarios
    # drawn with different keys are still paired on it.
    with warnings.catch_warnings():
        warnings.simplefilter("error", OutputWarning)
        im.estimate_effect(
            output_baseline=baseline,
            output_intervention=im.predict_on_batch(
                X,
                rng_key=random.key(1),
                return_sites="mu",
            ),
        )

    # A refit replaces the posterior samples that the baseline was computed from.
    im.fit_on_batch(X, y, num_steps=10, progress=False)
    with pytest.warns(OutputWarning, match="have different posterior samples"):
        im.estimate_effect(
            output_baseline=baseline,
            args_intervention={"X": X, "return_sites": "mu"},
            on_batch=True,
        )
