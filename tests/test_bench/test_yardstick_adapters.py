"""Record the actual public inference API calls on synthetic fev inputs."""

from dataclasses import replace

import numpy as np
import pytest
import torch

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.yardsticks.adapters import Chronos2Adapter, TimestralAdapter  # noqa: E402
from demand_forecast.bench.yardsticks.runner import Options, prepare_windows  # noqa: E402
from tests.helpers.fev_fixture import fixture_task, write_fev_fixture  # noqa: E402


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    import datasets.config
    import huggingface_hub.constants

    monkeypatch.setattr(datasets.config, "HF_DATASETS_OFFLINE", True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


class RecordingChronos:
    def __init__(self):
        self.calls = []

    def predict_quantiles(self, **kwargs):
        self.calls.append(kwargs)
        arrays = [
            torch.zeros(
                it["target"].shape[0], kwargs["prediction_length"], len(kwargs["quantile_levels"])
            )
            for it in kwargs["inputs"]
        ]
        return arrays, [a[..., 0] for a in arrays]


@pytest.mark.parametrize("off", [False, True])
def test_chronos_public_call_and_categories(tmp_path, off):
    path = write_fev_fixture(tmp_path, known=("promo", "event"), text_dynamic=("event",))
    task = fixture_task(path, known=("promo", "event"))
    options = Options(
        "chronos2",
        "unused",
        "reference",
        max_context=5,
        batch_size=1,
        cross_series="off" if off else "on",
    ).resolved()
    pipeline = RecordingChronos()
    adapter = Chronos2Adapter(options, task, pipeline=pipeline)
    outputs = adapter.predict(prepare_windows(task, options), task.horizon, task.quantile_levels)
    assert len(outputs) == 2
    for call in pipeline.calls:
        assert call["cross_learning"] is (not off)
        assert call["batch_size"] == 4  # one target + three dynamics
        assert call["context_length"] == 5
        assert call["limit_prediction_length"] is False
        assert call["quantile_levels"] == task.quantile_levels
        for item in call["inputs"]:
            assert item["target"].shape == (1, 5)
            assert item["past_covariates"]["event"].tolist()[-1] in ("0", "a", "b", None)
            assert item["past_covariates"]["promo"].dtype == np.float64
            assert item["future_covariates"]["event"].shape == (task.horizon,)


class RecordingTimestral:
    max_context = 32768
    quantile_levels = (0.1, 0.5, 0.9)

    def __init__(self):
        self.calls = []

    def forecast(self, context, horizon, **kwargs):
        self.calls.append((context, horizon, kwargs))

        class Forecast:
            quantiles = torch.as_tensor(
                np.broadcast_to(
                    np.asarray([1.0, 5.0, 9.0]), (*context.shape[:2], horizon, 3)
                ).copy()
            )

        return Forecast()


def test_timestral_published_batches_padding_categories_and_levels(tmp_path):
    task = fixture_task(
        write_fev_fixture(
            tmp_path,
            known=("promo", "event"),
            text_dynamic=("event",),
            lengths=(40, 38, 40, 28, 38, 40),
        ),
        known=("promo", "event"),
    )
    options = Options("timestral", "unused", "reference", batch_size=1024).resolved()
    forecaster = RecordingTimestral()
    adapter = TimestralAdapter(options, task, forecaster=forecaster)
    outputs = adapter.predict(prepare_windows(task, options), task.horizon, [0.1, 0.2, 0.5, 0.9])
    assert len(forecaster.calls) == 1  # all windows sorted and batched together
    ctx, horizon, kwargs = forecaster.calls[0]
    assert ctx.shape == (12, 1, 36) and np.isnan(ctx[0, 0, 0])
    assert kwargs["known_covariates"].shape == (12, 1, 40)
    assert kwargs["categorical_known_covariates"].shape == (12, 1, 40)
    assert kwargs["past_covariates"].shape == (12, 1, 36)
    assert kwargs["categorical_past_covariates"] is None
    assert kwargs["categorical_known_covariates"].dtype == np.float32
    # Symmetry, clipping and decoding defaults stay inside the public forecaster.
    assert "symmetric_averaging" not in kwargs
    np.testing.assert_allclose(outputs[0][0][0, 0], [1.0, 2.0, 5.0, 9.0])
    forecaster = RecordingTimestral()
    options = replace(options, max_context=5, covariates="none", batch_size=3)
    adapter = TimestralAdapter(options, task, forecaster=forecaster)
    adapter.predict(prepare_windows(task, options), task.horizon, [0.1, 0.5, 0.9])
    assert len(forecaster.calls) == 4
    assert all(
        c.shape == (3, 1, 5) and all(v is None for v in k.values()) for c, _, k in forecaster.calls
    )
    assert forecaster.max_context == 5
