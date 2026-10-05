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
from numpyro.infer import SVI

from aimz import ImpactModel, OutputWarning, PerformanceWarning
from tests.conftest import _make_svi, latent_intervention_model, lm


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


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_estimate_effect_artifact_paths_lazy_args(
    synthetic_data: tuple[Array, Array],
    vi: SVI,
) -> None:
    """Ensure lazy (args_*) inputs work and both scenarios' artifact paths recorded."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    im.fit(X=X, y=y, batch_size=len(X))

    msg = (
        r"The `batch_size` \(\d+\) is not divisible by the number of devices"
        r" \(\d+\)\."
    )
    with pytest.warns(PerformanceWarning, match=msg):
        effect = im.estimate_effect(
            args_baseline={
                "X": X,
                "batch_size": len(X),
            },
            args_intervention={
                "X": X,
                "intervention": {"sigma": 10.0},
                "batch_size": len(X),
            },
        )

    # Each internally computed scenario records its own call-specific subdirectory.
    # The subdir suffix is the outermost user-called method: `estimate_effect`.
    path_baseline = Path(effect.attrs["artifact_path_baseline"])
    path_intervention = Path(effect.attrs["artifact_path_intervention"])
    assert path_baseline != path_intervention
    for path in (path_baseline, path_intervention):
        assert path.is_dir()
        assert path.parent == Path(im.temp_dir).resolve()
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


def test_estimate_effect_group_mismatch_raises(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """A predictive group missing from either side raises ``ValueError``."""
    X, y = synthetic_data

    # Group present in the baseline but missing from the intervention.
    with pytest.raises(
        ValueError,
        match=r"Group 'posterior_predictive' not found in `dt_intervention`.",
    ):
        im_lm_svi_fitted.estimate_effect(
            output_baseline=im_lm_svi_fitted.predict_on_batch(X),
            output_intervention=im_lm_svi_fitted.predict_on_batch(X, in_sample=False),
        )

    # A prior predictive baseline needs a prior predictive intervention.
    with pytest.raises(
        ValueError,
        match=r"Group 'prior_predictive' not found in `dt_intervention`.",
    ):
        im_lm_svi_fitted.estimate_effect(
            output_baseline=im_lm_svi_fitted.sample_prior_predictive_on_batch(X),
            output_intervention=im_lm_svi_fitted.predict_on_batch(X),
        )

    # A baseline without any predictive group, such as a log-likelihood tree
    with pytest.raises(
        ValueError,
        match=r"Group 'posterior_predictive' not found in `dt_baseline`.",
    ):
        im_lm_svi_fitted.estimate_effect(
            output_baseline=im_lm_svi_fitted.log_likelihood(X, y, store="memory"),
            output_intervention=im_lm_svi_fitted.predict_on_batch(X),
        )


def test_estimate_effect_warns_on_size_mismatch(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """A dimension-size mismatch between the scenarios warns."""
    X, _ = synthetic_data

    rng_key = random.key(0)
    base = im_lm_svi_fitted.predict_on_batch(X, rng_key=rng_key)
    intervention = im_lm_svi_fitted.predict_on_batch(X[:80], rng_key=rng_key)

    with pytest.warns(OutputWarning, match="different dimension sizes"):
        im_lm_svi_fitted.estimate_effect(
            output_baseline=base,
            output_intervention=intervention,
        )


def test_estimate_effect_warns_on_coordinate_mismatch(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """A coordinate-label mismatch between equally sized scenarios warns."""
    X, _ = synthetic_data

    rng_key = random.key(0)
    base = im_lm_svi_fitted.predict_on_batch(X[:50], rng_key=rng_key)
    intervention = im_lm_svi_fitted.predict_on_batch(X, rng_key=rng_key).isel(
        y_dim_0=slice(50, 100),
    )

    with pytest.warns(OutputWarning, match="different coordinate labels"):
        im_lm_svi_fitted.estimate_effect(
            output_baseline=base,
            output_intervention=intervention,
        )


def test_estimate_effect_warns_on_stale_baseline(
    synthetic_data: tuple[Array, Array],
) -> None:
    """A baseline kept from before a refit warns; a deterministic site stays paired."""
    X, y = synthetic_data
    im = ImpactModel(
        latent_intervention_model,
        rng_key=random.key(42),
        inference=_make_svi(latent_intervention_model),
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
