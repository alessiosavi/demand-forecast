"""Task seasonality controls scaling without optional benchmark imports."""

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from demand_forecast.bench.specs import FevTaskSpec
from demand_forecast.core.evaluation import prepare_validation
from demand_forecast.core.run import run_identity
from demand_forecast.core.validation import plan_inner_validation
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.loader_fev import panel_from_window
from demand_forecast.data.sampler import BatchBuilder
from demand_forecast.data.scaling import sample_frame
from tests.helpers.settings import tiny_settings


def window(freq="15min", length=200):
    dates = pd.date_range("2024-01-01", periods=length + 4, freq=freq)
    past = [{"id": "a", "timestamp": dates[:length], "target": np.arange(length) ** 2}]
    future = [{"id": "a", "timestamp": dates[length:]}]
    return past, future


@pytest.mark.parametrize("multi_target", [False, True])
@pytest.mark.parametrize(
    "freq,seasonality,expected",
    [("15T", 96, 96), ("15T", None, 1), ("D", 7, 7), ("D", None, 7), ("D", 1, 1)],
)
def test_panel_prefers_explicit_task_seasonality(freq, seasonality, expected, multi_target):
    spec = FevTaskSpec(name="local", dataset_path="unused", target_support="real")
    if seasonality is not None:
        spec = replace(spec, seasonality=seasonality)
    past, future = window("15min" if freq == "15T" else freq)
    if multi_target:
        spec = replace(spec, target=["target", "second"])
        past[0]["second"] = -past[0]["target"]
    panel = panel_from_window(past, future, spec, freq)
    assert panel.seasonal_period == expected
    assert len(panel.series) == (2 if multi_target else 1)


@pytest.mark.parametrize("policy", ["first_window", "per_window"])
@pytest.mark.parametrize(
    "freq,seasonality,changed",
    [("D", 7, False), ("D", None, False), ("D", 1, True), ("15T", 96, True), ("15T", None, False)],
)
def test_identity_changes_only_when_panel_period_changes(freq, seasonality, changed, policy):
    spec = FevTaskSpec(
        name="local", dataset_path="unused", target_support="real", seasonality=seasonality
    )
    # Exact definition before task seasonality controlled panel normalization.
    legacy = {
        "name": "local",
        "horizon": 1,
        "num_windows": 1,
        "initial_cutoff": None,
        "window_step_size": None,
        "min_context_length": 1,
        "seasonality": seasonality,
        "quantile_levels": [],
    }
    definition = spec.task_definition(freq)
    expected = {**legacy, **({"seasonal_period": seasonality} if changed else {})}
    assert json.dumps(definition).encode() == json.dumps(expected).encode()
    settings = tiny_settings()

    def identity(values):
        return run_identity(
            settings, settings.seed, "data", "version", "2024-01-01", values, policy=policy
        )

    assert (identity(definition) != identity(legacy)) is changed
    assert "seasonal_period_source" not in definition


def test_daily_task_matching_frequency_keeps_sample_frames_byte_identical():
    spec = FevTaskSpec(name="daily", dataset_path="unused", seasonality=7, target_support="real")
    past, future = window("D", length=40)
    past[0]["target"] = past[0]["target"].astype(float)
    past[0]["target"][[3, 12, 30]] = np.nan
    panel = panel_from_window(past, future, spec, "D")
    fallback = panel_from_window(past, future, replace(spec, seasonality=None), "D")
    assert panel.seasonal_period == fallback.seasonal_period == 7
    for origin in (16, 24, 40):
        values = panel.series[0].target[:origin]
        legacy = sample_frame(values[-8:], values, 7)
        for candidate in (panel, fallback):
            frame = sample_frame(values[-8:], values, candidate.seasonal_period)
            # Include all scalar values and flags, preserving their exact bytes.
            assert (
                np.array(list(asdict(frame).values()), dtype=np.float64).tobytes()
                == np.array(list(asdict(legacy).values()), dtype=np.float64).tobytes()
            )


@pytest.mark.parametrize("length", [32, 96])
def test_history_at_most_task_period_keeps_finite_fallback_and_no_validation(length):
    spec = FevTaskSpec(name="short", dataset_path="unused", seasonality=96, target_support="real")
    past, future = window(length=length)
    panel = panel_from_window(past, future, spec, "15min")
    assert panel.seasonal_period == 96
    values = panel.series[0].target
    frame = sample_frame(values[-8:], values, panel.seasonal_period)
    assert frame.has_context and not frame.eligible
    assert np.isnan(frame.scale_raw)
    assert frame.scale == 1.0 and np.isfinite(frame.loc)
    encoder = StaticEncoder().fit(panel, ["a"])
    batch = BatchBuilder(panel, 4, 8, encoder).build([(0, length)])
    assert not batch.eligible.any() and not batch.loss_eligible.any()
    assert batch.scale.tolist() == [1.0]
    assert np.isfinite(batch.target.numpy()).all()
    plan = plan_inner_validation(panel, future[0]["timestamp"][0], horizon=4)
    assert plan.exception and plan.val_items.size == 0 and plan.fit_ids == ["a"]


def test_task_period_larger_than_context_uses_full_past_for_validation():
    spec = FevTaskSpec(name="long", dataset_path="unused", seasonality=96, target_support="real")
    past, future = window()
    panel = panel_from_window(past, future, spec, "15min")
    plan = plan_inner_validation(panel, future[0]["timestamp"][0], horizon=4)
    assert not plan.exception and plan.val_items.tolist() == [0]
    encoder = StaticEncoder().fit(panel, plan.fit_ids)
    prepared = prepare_validation(panel, plan.val_items, plan.val_origin, 4, 8, encoder)
    assert prepared.seasonal_period == 96
    values = panel.series[0].target[:-4]
    expected = np.abs(values[96:] - values[:-96]).mean()
    assert prepared.batches[0].eligible.all()
    assert prepared.batches[0].scale.item() == expected
