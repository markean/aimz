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


"""Tests for the guards shared by the streaming and on-batch entry points."""

from pathlib import Path

import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro import sample

from aimz import ImpactModel
from aimz._exceptions import NotFittedError
from tests.conftest import _make_svi, lm


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("predict", (None,)),
        ("predict_on_batch", (None,)),
        ("log_likelihood", (None, None)),
        ("estimate_effect", ()),
    ],
)
def test_not_fitted(method: str, args: tuple) -> None:
    """An unfitted model rejects the entry points that need a posterior."""
    im = ImpactModel(lm, rng_key=random.key(42), inference=_make_svi(lm))
    with pytest.raises(NotFittedError):
        getattr(im, method)(*args)


@pytest.mark.parametrize(
    "method",
    [
        "predict",
        "predict_on_batch",
        "sample_prior_predictive",
        "sample_prior_predictive_on_batch",
    ],
)
@pytest.mark.parametrize(
    "kwargs",
    [{"arg": True, "y": None}, {"arg": True, "extra": True}, {}],
    ids=["reserved", "extra", "missing"],
)
def test_kernel_argument_binding(
    synthetic_data: tuple[Array, Array],
    method: str,
    kwargs: dict,
) -> None:
    """A reserved, unknown, or missing kernel argument raises on every entry point."""
    X, y = synthetic_data

    def kernel(X: Array, arg: object, y: Array | None = None) -> None:
        b = sample("b", dist.Normal(0.0, 1.0))
        sample("y", dist.Normal(b, 1.0), obs=y)

    im = ImpactModel(kernel, rng_key=random.key(42), inference=_make_svi(kernel))
    im.fit_on_batch(X, y, arg=True, num_steps=1, num_samples=10, progress=False)
    with pytest.raises(TypeError):
        getattr(im, method)(X, **kwargs)


@pytest.mark.parametrize(
    ("method", "writer"),
    [
        ("predict", "write_predictive"),
        ("sample_prior_predictive", "write_predictive"),
        ("log_likelihood", "write_log_likelihood"),
    ],
)
def test_write_failure_reclaims_subdir(
    synthetic_data: tuple[Array, Array],
    im_lm_svi_fitted: ImpactModel,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    method: str,
    writer: str,
) -> None:
    """A failure during the write phase reclaims the output subdirectory."""
    X, y = synthetic_data

    def boom(*args: object, **kwargs: object) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(im_lm_svi_fitted._streamer, writer, boom)
    args = (X, y) if method == "log_likelihood" else (X,)
    with pytest.raises(RuntimeError, match="boom"):
        getattr(im_lm_svi_fitted, method)(
            *args,
            output_dir=tmp_path,
            batch_size=3,
            progress=False,
        )
    assert not any(tmp_path.iterdir())
