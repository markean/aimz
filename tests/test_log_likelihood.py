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

"""Tests for the `.log_likelihood()` method."""

from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from aimz import ImpactModel
from aimz._exceptions import NotFittedError


def test_model_not_fitted() -> None:
    """Calling `.log_likelihood()` on an unfitted model raises an error."""

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
        im.log_likelihood(None, None)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_empty_posterior(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty posterior produces a single-draw log-likelihood without crashing."""
    X, y = synthetic_data
    monkeypatch.setattr(im_lm_svi_fitted, "_posterior", None)

    out = im_lm_svi_fitted.log_likelihood(X=X, y=y, progress=False)

    assert out.log_likelihood["y"].sizes["draw"] == 1


def test_subsampled_kernel(
    synthetic_data: tuple[Array, Array],
    im_lm_subsample_svi_fitted: ImpactModel,
) -> None:
    """A kernel subsampling inside a plate is evaluated without an rng key.

    Subsample indices are pinned deterministically rather than drawn at random, so
    the bare (unseeded) kernel traces cleanly and every passed observation is scored
    on both sharding paths.
    """
    X, y = synthetic_data
    im = im_lm_subsample_svi_fitted

    out = im.log_likelihood(X=X, y=y, batch_size=30, progress=False)
    out_draw = im.log_likelihood(
        X=X,
        y=y,
        shard_axis="draw",
        batch_size=250,
        progress=False,
    )

    assert out.log_likelihood["y"].shape == (1, 1000, len(X))
    assert np.isfinite(out.log_likelihood["y"].values).all()
    # Plate indices never gather here, so the sharding paths agree.
    assert np.allclose(
        out.log_likelihood["y"].values,
        out_draw.log_likelihood["y"].values,
        atol=1e-6,
    )


def test_subsampled_kernel_batch_exceeds_plate(
    synthetic_data: tuple[Array, Array],
    im_lm_subsample_svi_fitted: ImpactModel,
) -> None:
    """A batch larger than the declared plate size is rejected loudly."""
    X, y = synthetic_data
    X_big, y_big = jnp.tile(X, (6, 1)), jnp.tile(y, 6)

    with pytest.raises(ValueError, match="declares size=100"):
        im_lm_subsample_svi_fitted.log_likelihood(
            X=X_big,
            y=y_big,
            batch_size=600,
            progress=False,
        )


def test_log_likelihood_requires_y(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
) -> None:
    """`log_likelihood()` requires `y` for an array `X`."""
    X, _ = synthetic_data
    with pytest.raises(ValueError, match="`y` is required"):
        im_lm_svi_fitted.log_likelihood(X, progress=False)


def test_log_likelihood_cleans_subdir_on_write_failure(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure during the write phase reclaims the output subdirectory."""
    X, y = synthetic_data

    def boom(*args: object, **kwargs: object) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(im_lm_svi_fitted._streamer, "write_log_likelihood", boom)

    with TemporaryDirectory() as output_dir:
        with pytest.raises(RuntimeError, match="boom"):
            im_lm_svi_fitted.log_likelihood(
                X,
                y=y,
                output_dir=output_dir,
                batch_size=3,
                progress=False,
            )
        # The just-created timestamped subdir was reclaimed, not orphaned.
        assert not any(Path(output_dir).iterdir())


class TestBatchSize:
    """Test class related to batch size specification."""

    def test_default_batch_size(
        self,
        synthetic_data: tuple[Array, Array],
        im_lm_svi_fitted: ImpactModel,
    ) -> None:
        """Warns if `batch_size` is not explicitly set."""
        X, y = synthetic_data
        msg = (
            r"The `batch_size` \(\d+\) is not divisible by the number of devices "
            r"\(\d+\)\."
        )
        with pytest.warns(UserWarning, match=msg):
            im_lm_svi_fitted.log_likelihood(X=X, y=y)
