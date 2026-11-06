"""Event conversion works without the optional fev/datasets dependencies."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data import loader_fev
from demand_forecast.data.loader_fev import panel_from_window
from tests.helpers.fev_spec import fixture_spec


def event_window():
    past = [
        {
            "id": "item-a",
            "timestamp": pd.date_range("2024-01-01", periods=7),
            "target": list(range(7)),
            "second": list(range(10, 17)),
            "promo": ["0", "a", None, np.nan, pd.NA, "b", 0],
            "oil": [False, True, False, True, False, True, False],
            "cat": "A",
        }
    ]
    future = [
        {
            "id": "item-a",
            "timestamp": pd.date_range("2024-01-08", periods=4),
            "promo": [None, "0", "unseen holiday", "a"],
            # Future past-only columns must never be read.
            "oil": ["ignored"] * 4,
        }
    ]
    return past, future


@pytest.mark.parametrize("multi_target", [False, True])
@pytest.mark.parametrize("past_event", [False, True])
def test_events_preserve_values_across_targets_and_window_halves(multi_target, past_event):
    past, future = event_window()
    events = {"promo": ["0"]}
    if past_event:
        past[0]["oil"] = [None, "none", "a", "none", "new", np.nan, pd.NA]
        events["oil"] = ["none"]
    spec = fixture_spec(Path("unused.parquet"), event_dynamics=events)
    if multi_target:
        spec = replace(spec, target=["target", "second"])
    panel = panel_from_window(past, future, spec, "D")
    assert len(panel.series) == (2 if multi_target else 1)
    assert panel.roles.known_dynamic == ("promo",)
    assert panel.roles.past_dynamic == ("oil",)
    for series in panel.series:
        np.testing.assert_array_equal(series.dynamic["promo"], [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(
            series.dynamic["oil"], [0, 0, 1, 0, 1, 0, 0] if past_event else [0, 1, 0, 1, 0, 1, 0]
        )
    # The conversion must not mutate reusable input rows.
    assert past[0]["promo"][1] == "a"
    assert future[0]["promo"][2] == "unseen holiday"


@pytest.mark.parametrize("part,column", [("past", "promo"), ("future", "promo"), ("past", "oil")])
@pytest.mark.parametrize("text", ["a", "None", np.str_("holiday"), object()])
def test_undeclared_text_names_task_column_and_profile_fix(part, column, text):
    past, future = event_window()
    past[0]["promo"] = [0.0] * 7
    future[0]["promo"] = [0.0] * 4
    (past if part == "past" else future)[0][column][0] = text
    spec = fixture_spec(Path("unused.parquet"))
    with pytest.raises(ValueError) as exc:
        panel_from_window(past, future, spec, "D")
    message = str(exc.value)
    assert spec.name in message and "item-a" in message and column in message
    assert "declare it under event_dynamics in the task profile" in message
    assert isinstance(exc.value.__cause__, (ValueError, TypeError))


@pytest.mark.parametrize("dtype", [np.float32, str])
def test_undeclared_numeric_columns_reach_frame_without_iteration(monkeypatch, dtype):
    class NoIteration(np.ndarray):
        def __iter__(self):
            raise AssertionError("numeric dynamic cells must not be iterated")

    past, future = event_window()
    expected_promo = np.arange(11) / 2
    past[0]["promo"] = np.array(expected_promo[:7], dtype=dtype).view(NoIteration)
    future[0]["promo"] = np.array(expected_promo[7:], dtype=dtype).view(NoIteration)
    past[0]["oil"] = np.array(np.arange(7) / 2, dtype=dtype).view(NoIteration)
    original_frame = loader_fev._float_frame
    seen = []

    def check_frame(values, index, row, spec):
        for name in ("promo", "oil"):
            if name in values:
                assert values[name] is row[name]
                seen.append(name)
        return original_frame(values, index, row, spec)

    monkeypatch.setattr(loader_fev, "_float_frame", check_frame)
    panel = panel_from_window(past, future, fixture_spec(Path("unused.parquet")), "D")
    assert seen == ["promo", "oil", "promo"]
    np.testing.assert_array_equal(panel.series[0].dynamic["promo"], expected_promo)
    np.testing.assert_array_equal(panel.series[0].dynamic["oil"], np.arange(7) / 2)


@pytest.mark.parametrize("bad_value", ["bad target", object()])
def test_target_cast_errors_propagate_unchanged(bad_value):
    past, future = event_window()
    past[0]["promo"] = [0.0] * 7
    future[0]["promo"] = [0.0] * 4
    past[0]["target"][0] = bad_value
    with pytest.raises((ValueError, TypeError)) as original:
        pd.DataFrame({"target": past[0]["target"]}, dtype=float)
    with pytest.raises(type(original.value)) as actual:
        panel_from_window(past, future, fixture_spec(Path("unused.parquet")), "D")
    assert str(actual.value) == str(original.value)
    assert actual.value.__cause__ is None
