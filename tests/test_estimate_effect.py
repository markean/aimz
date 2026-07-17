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

from pathlib import Path

import jax.numpy as jnp
import pytest
from jax import Array, random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from aimz._exceptions import NotFittedError
from tests.conftest import lm


def test_model_not_fitted() -> None:
    """Calling `.estimate_effect()` on an unfitted model raises an error."""

    def kernel(X: Array, y: Array | None = None) -> None:
        pass

    im = ImpactModel(
        kernel,
        rng_key=random.key(42),
        inference=SVI(
            kernel,
            guide=AutoNormal(kernel),
            optim=Adam(step_size=1e-3),
            loss=Trace_ELBO(),
        ),
    )
    with pytest.raises(NotFittedError):
        im.estimate_effect()


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

    dt_baseline = im.predict_on_batch(X)

    msg = "Either `output_intervention` or `args_intervention` must be provided."
    with pytest.raises(ValueError, match=msg):
        im.estimate_effect(output_baseline=dt_baseline)

    dt_intervention = im.predict_on_batch(X, intervention={"z": jnp.zeros_like(y)})

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
    with pytest.warns(UserWarning, match=msg):
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


def test_estimate_effect_on_batch(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """Ensure `on_batch=True` uses `predict_on_batch` and produces valid results."""
    X, _ = synthetic_data

    effect = im_lm_svi_fitted.estimate_effect(
        args_baseline={"X": X},
        args_intervention={"X": X, "intervention": {"sigma": 10.0}},
        on_batch=True,
    )

    # `on_batch=True` should not create an `output_dir` attribute
    assert "output_dir" not in effect.attrs


@pytest.mark.parametrize("in_sample", [True, False])
def test_estimate_effect_on_batch_dict(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    *,
    in_sample: bool,
) -> None:
    """Dict results from `predict_on_batch` are wrapped in the correct group."""
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


def test_estimate_effect_group_mismatch_raises(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """A predictive group missing from either side raises ``ValueError``."""
    X, _ = synthetic_data

    # Group present in the baseline but missing from the intervention.
    with pytest.raises(
        ValueError,
        match=r"Group 'posterior_predictive' not found in `dt_intervention`.",
    ):
        im_lm_svi_fitted.estimate_effect(
            output_baseline=im_lm_svi_fitted.predict_on_batch(X),
            output_intervention=im_lm_svi_fitted.predict_on_batch(X, in_sample=False),
        )

    # Group missing from the baseline (prior-predictive has no posterior_predictive).
    with pytest.raises(
        ValueError,
        match=r"Group 'posterior_predictive' not found in `dt_baseline`.",
    ):
        im_lm_svi_fitted.estimate_effect(
            output_baseline=im_lm_svi_fitted.sample_prior_predictive_on_batch(X),
            output_intervention=im_lm_svi_fitted.predict_on_batch(X),
        )


def test_estimate_effect_warns_on_size_mismatch(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """A dimension-size mismatch between the scenarios warns."""
    X, _ = synthetic_data

    base = im_lm_svi_fitted.predict_on_batch(X)
    intervention = im_lm_svi_fitted.predict_on_batch(X[:80])

    with pytest.warns(UserWarning, match="different dimension sizes"):
        im_lm_svi_fitted.estimate_effect(
            output_baseline=base,
            output_intervention=intervention,
        )
