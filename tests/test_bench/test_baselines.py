import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.baselines import (  # noqa: E402
    BASELINE_MODELS,
    SeasonalNaiveBaseline,
    baseline_summary,
    make_baseline,
)
from demand_forecast.bench.run import run_baseline  # noqa: E402
from tests.helpers.fev_fixture import fixture_task, write_fev_fixture  # noqa: E402
from tests.helpers.fev_spec import fixture_spec  # noqa: E402


@pytest.mark.parametrize("model_name", BASELINE_MODELS)
def test_baseline_summary_on_fixture(tmp_path, model_name):
    summary = baseline_summary(fixture_task(write_fev_fixture(tmp_path)), model=model_name)
    assert summary["model_name"] == model_name
    for metric in ("SQL", "MASE", "WAPE", "WQL"):
        assert np.isfinite(summary[metric])
    assert summary["num_forecasts"] == 6 * 2
    assert summary["training_time_s"] >= 0
    assert summary["inference_time_s"] > 0


def test_seasonal_naive_summary_metrics_are_unchanged(tmp_path):
    summary = baseline_summary(fixture_task(write_fev_fixture(tmp_path)))
    # Captured from the unmodified implementation at f681f68, statsforecast 2.0.1.
    expected = {
        "test_error": 0.9666070836020835,
        "SQL": 0.9666070836020835,
        "MASE": 1.186931297478433,
        "WAPE": 0.7036414177215173,
        "WQL": 0.5710514349460601,
    }
    assert {metric: summary[metric] for metric in expected} == expected


def quantile_array(task, prediction):
    cleaned = task.clean_and_validate_predictions(prediction)
    return np.stack([cleaned["target"][str(q)] for q in task.quantile_levels], axis=-1)


@pytest.mark.parametrize("model_name", BASELINE_MODELS)
@pytest.mark.parametrize("freq", ["W-SUN", "MS", "QS", "YS"])
def test_baselines_produce_finite_monotone_quantiles(tmp_path, model_name, freq):
    task = fixture_task(write_fev_fixture(tmp_path, n_series=2, freq=freq))
    model = make_baseline(model_name)
    assert isinstance(model, fev.ForecastingModel)
    predictions = model.fit_predict(task)
    assert len(predictions) == task.num_windows
    for prediction in predictions:
        quantiles = quantile_array(task, prediction)
        assert quantiles.shape == (2, task.horizon, len(task.quantile_levels))
        assert np.isfinite(quantiles).all()
        assert (np.diff(quantiles, axis=-1) >= 0).all()
        assert np.isfinite(
            task.clean_and_validate_predictions(prediction)["target"]["predictions"]
        ).all()


def test_median3_is_cellwise_median_of_component_quantiles(tmp_path):
    task = fixture_task(write_fev_fixture(tmp_path, n_series=12))
    predictions = {name: make_baseline(name).fit_predict(task) for name in BASELINE_MODELS}
    for index in range(task.num_windows):
        components = np.stack(
            [quantile_array(task, predictions[name][index]) for name in BASELINE_MODELS[:3]]
        )
        expected = np.sort(np.median(components, axis=0), axis=-1)
        assert expected.shape == (12, task.horizon, len(task.quantile_levels))
        np.testing.assert_array_equal(quantile_array(task, predictions["median3"][index]), expected)


@pytest.mark.parametrize("model_name", BASELINE_MODELS[1:])
def test_classical_model_errors_fall_back_to_seasonal_naive(tmp_path, monkeypatch, model_name):
    from statsforecast.models import AutoETS, AutoTheta

    task = fixture_task(write_fev_fixture(tmp_path, n_series=2, freq="MS"))
    expected = SeasonalNaiveBaseline().fit_predict(task)
    failed_calls = []

    def fail_forecast(self, *args, level=None, **kwargs):
        failed_calls.append(type(self).__name__)
        raise ValueError("forced classical model failure")

    monkeypatch.setattr(AutoETS, "forecast", fail_forecast)
    monkeypatch.setattr(AutoTheta, "forecast", fail_forecast)
    actual = make_baseline(model_name).fit_predict(task)
    assert failed_calls
    for reference, prediction in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(
            quantile_array(task, prediction), quantile_array(task, reference)
        )


def test_resume_is_keyed_by_task_and_model_and_preserves_rows(tmp_path, monkeypatch, capsys):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    out = tmp_path / "out"
    first = run_baseline([spec], out)
    previous = first.assign(model_name="other_model", task="other_task")
    pd.concat([first, previous], ignore_index=True).to_csv(out / "summaries.csv", index=False)
    for model_name in BASELINE_MODELS[1:]:
        result = run_baseline([spec, spec], out, model=model_name)
        assert f"Skipping completed task fixture for model {model_name}" in capsys.readouterr().out
        assert len(result.loc[result["model_name"] == model_name]) == 1
    assert result[["task", "model_name"]].values.tolist() == [
        ["fixture", "seasonal_naive"],
        ["other_task", "other_model"],
        *[["fixture", name] for name in BASELINE_MODELS[1:]],
    ]
    pd.testing.assert_frame_equal(result.iloc[:1], first)
    pd.testing.assert_frame_equal(result.iloc[1:2].reset_index(drop=True), previous)

    def fail_if_recomputed(*args):
        pytest.fail("completed task/model pair was recomputed")

    monkeypatch.setattr(type(spec), "to_task", fail_if_recomputed)
    for model_name in BASELINE_MODELS:
        resumed = run_baseline([spec], out, model=model_name)
        pd.testing.assert_frame_equal(resumed, result)


def test_unknown_baseline_is_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError, match="Unknown baseline model"):
        make_baseline("unknown")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Unknown baseline model"):
        run_baseline([], out, model="unknown")
    assert not out.exists()


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


@pytest.mark.parametrize("model_name", BASELINE_MODELS)
def test_predictions_use_requested_quantiles_and_reset_timing(tmp_path, model_name):
    task = fixture_task(write_fev_fixture(tmp_path, n_series=2))
    task.quantile_levels = [0.123, 0.5, 0.877]
    model = make_baseline(model_name)
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
