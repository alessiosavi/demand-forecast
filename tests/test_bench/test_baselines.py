import subprocess
import sys

import numpy as np
import pytest

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.baselines import (  # noqa: E402
    SeasonalNaiveBaseline,
    baseline_summary,
)
from tests.helpers.fev_fixture import fixture_task, write_fev_fixture  # noqa: E402


def test_seasonal_naive_summary_on_fixture(tmp_path):
    summary = baseline_summary(fixture_task(write_fev_fixture(tmp_path)))
    assert summary["model_name"] == "seasonal_naive"
    for metric in ("SQL", "MASE", "WAPE", "WQL"):
        assert np.isfinite(summary[metric])
    assert summary["num_forecasts"] == 6 * 2
    assert summary["training_time_s"] >= 0
    assert summary["inference_time_s"] > 0


@pytest.mark.parametrize(
    "freq,seasonality,expected_period",
    [("W-SUN", 1, 1), ("W-WED", 3, 3), ("D", 1, 7), ("h", 1, 24), ("MS", 1, 12)],
)
def test_predictions_follow_seasonality_and_fev_item_order(
    tmp_path, freq, seasonality, expected_period
):
    task = fixture_task(write_fev_fixture(tmp_path, n_series=12, length=64, freq=freq))
    task.seasonality = seasonality
    model = SeasonalNaiveBaseline()
    assert isinstance(model, fev.ForecastingModel)
    predictions = model.fit_predict(task)
    assert len(predictions) == task.num_windows
    for window, prediction in zip(task.iter_windows(), predictions, strict=True):
        cleaned = task.clean_and_validate_predictions(prediction)
        past, _ = window.get_input_data()
        assert len(cleaned["target"]) == len(past) == 12
        assert set(cleaned["target"].column_names) == {
            "predictions",
            *(str(q) for q in task.quantile_levels),
        }
        for history, forecast in zip(past, cleaned["target"], strict=True):
            expected = np.resize(np.asarray(history["target"])[-expected_period:], task.horizon)
            np.testing.assert_allclose(forecast["predictions"], expected)
            np.testing.assert_allclose(forecast["0.5"], expected)
            quantiles = np.array([forecast[str(q)] for q in task.quantile_levels])
            assert quantiles.shape == (9, task.horizon)
            assert np.isfinite(quantiles).all()
            assert (np.diff(quantiles, axis=0) >= 0).all()


def test_predictions_use_requested_quantiles_and_reset_timing(tmp_path):
    task = fixture_task(write_fev_fixture(tmp_path, n_series=2))
    task.quantile_levels = [0.123, 0.5, 0.877]
    model = SeasonalNaiveBaseline()
    model.training_time = model.inference_time = float("inf")
    for prediction in model.fit_predict(task):
        cleaned = task.clean_and_validate_predictions(prediction)
        assert set(cleaned["target"].column_names) == {"predictions", "0.123", "0.5", "0.877"}
    assert np.isfinite(model.training_time) and model.training_time >= 0
    assert np.isfinite(model.inference_time) and model.inference_time > 0


@pytest.mark.parametrize("dependency", ["fev", "datasets", "statsforecast"])
def test_baseline_explains_missing_bench_dependencies(dependency):
    code = f"""
import sys
sys.modules[{dependency!r}] = None
import demand_forecast.bench
try:
    from demand_forecast.bench.baselines import SeasonalNaiveBaseline
    SeasonalNaiveBaseline().fit_predict(None)
except ImportError as exc:
    assert type(exc) is ImportError, type(exc)
    assert str(exc) == 'install demand-forecast[bench]', str(exc)
else:
    raise AssertionError('missing dependency did not raise ImportError')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
