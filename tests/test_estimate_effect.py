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
from jax import random
from jax.typing import ArrayLike
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from aimz._exceptions import NotFittedError
from tests.conftest import latent_variable_model, lm


def test_model_not_fitted() -> None:
    """Calling `.estimate_effect()` on an unfitted model raises an error."""

    def kernel(X: ArrayLike, y: ArrayLike | None = None) -> None:
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


@pytest.mark.parametrize("vi", [latent_variable_model], indirect=True)
def test_estimate_effect_argument_validation(
    synthetic_data: tuple[ArrayLike, ArrayLike],
    vi: SVI,
) -> None:
    """Validate argument exclusivity and successful effect computation."""
    X, y = synthetic_data
    im = ImpactModel(latent_variable_model, rng_key=random.key(42), inference=vi)
    im.fit(X=X, y=y, batch_size=len(X))

    msg = "Either `output_baseline` or `args_baseline` must be provided."
    with pytest.raises(ValueError, match=msg):
        im.estimate_effect(output_baseline=None, args_baseline=None)

    dt_baseline = im.predict_on_batch(X)

    msg = "Either `output_intervention` or `args_intervention` must be provided."
    with pytest.raises(ValueError, match=msg):
        im.estimate_effect(output_baseline=dt_baseline)

    dt_intervention = im.predict_on_batch(X, intervention={"z": jnp.zeros_like(y)})

    impact = im.estimate_effect(
        output_baseline=dt_baseline,
        output_intervention=dt_intervention,
    )

    assert impact.posterior_predictive["y"].mean(dim=["chain", "draw"]).shape == (
        len(y),
    )


@pytest.mark.parametrize("vi", [lm], indirect=True)
def test_estimate_effect_output_dir_lazy_args(
    synthetic_data: tuple[ArrayLike, ArrayLike],
    vi: SVI,
) -> None:
    """Ensure lazy (args_*) inputs work and baseline `output_dir` is propagated."""
    X, y = synthetic_data
    im = ImpactModel(lm, rng_key=random.key(42), inference=vi)
    im.fit(X=X, y=y, batch_size=len(X))

    msg = (
        r"The `batch_size` \(\d+\) is not divisible by the number of devices"
        r" \(\d+\)\."
    )
    with pytest.warns(UserWarning, match=msg):
        impact = im.estimate_effect(
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

    # Confirm the temporary output directory propagated to effect result
    assert impact.attrs.get("output_dir") == str(Path(im.temp_dir).resolve())
    im.cleanup()
