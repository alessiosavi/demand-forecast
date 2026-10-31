import numpy as np
import pandas as pd
import pytest

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.data.frequency import GridError  # noqa: E402
from demand_forecast.data.loader_fev import item_ids, panel_from_window  # noqa: E402
from demand_forecast.data.panel import Roles  # noqa: E402
from tests.helpers.fev_fixture import write_fev_fixture  # noqa: E402
from tests.helpers.fev_spec import fixture_spec  # noqa: E402


def test_panel_preserves_order_history_end_and_future_covariates(tmp_path):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    task = spec.to_task()
    for window in task.iter_windows():
        past, future = window.get_input_data()
        past = past.select([2, 0, 1, 5, 3, 4])
        future = future.select(range(len(future) - 1, -1, -1))
        panel = panel_from_window(past, future, spec, task.freq)
        assert item_ids(panel) == list(past["id"])
        assert panel.freq.freqstr == "W-SUN"
        assert panel.seasonal_period == 1
        assert panel.target_support == "nonnegative"
        assert panel.roles == Roles("target", ("promo",), ("oil",), ("cat",), ())
        future_by_id = {row["id"]: row for row in future}
        for series, row in zip(panel.series, past, strict=True):
            assert series.length == len(row["timestamp"])
            assert len(series.dynamic["promo"]) == series.length + 4
            assert len(series.dynamic["oil"]) == series.length
            assert series.statics["cat"] == row["cat"]
            assert series.start == row["timestamp"][0]
            np.testing.assert_equal(series.target, row["target"])
            np.testing.assert_equal(series.dynamic["oil"], row["oil"])
            np.testing.assert_equal(
                series.dynamic["promo"],
                np.concatenate([row["promo"], future_by_id[row["id"]]["promo"]]),
            )


@pytest.mark.parametrize("all_missing", [False, True])
def test_missing_targets_do_not_shorten_history(tmp_path, all_missing):
    spec = fixture_spec(write_fev_fixture(tmp_path, n_series=1))
    task = spec.to_task()
    past, future = next(task.iter_windows()).get_input_data()
    length = len(past[0]["timestamp"])
    missing = length if all_missing else 2
    past = past.map(
        lambda row: {
            "target": list(row["target"][:-missing]) + [None] * missing,
            "oil": list(row["oil"][:-missing]) + [None] * missing,
        }
    )
    panel = panel_from_window(past, future, spec, task.freq)
    series = panel.series[0]
    assert series.length == length
    assert np.isnan(series.target[-missing:]).all()
    assert np.isnan(series.dynamic["oil"][-missing:]).all()
    assert panel.timestamps(0)[-1] == past[0]["timestamp"][-1]
    assert len(series.dynamic["promo"]) == length + spec.horizon


def test_gaps_and_future_values_are_aligned_by_timestamp(tmp_path):
    from datasets import Dataset

    spec = fixture_spec(tmp_path / "unused.parquet", target_support="real")
    past = Dataset.from_list(
        [
            {
                "id": "a",
                "timestamp": pd.to_datetime(["2023-01-01", "2023-01-15", "2023-01-22"]),
                "target": [-2.0, 4.0, None],
                "promo": [0.0, 1.0, 0.0],
                "oil": [3.0, 5.0, None],
                "cat": "A",
            }
        ]
    )
    future = Dataset.from_list(
        [
            {
                "id": "a",
                "timestamp": pd.to_datetime(["2023-02-12", "2023-02-05"]),
                "promo": [7.0, None],
                # Even if supplied, held-out targets/past covariates must be ignored.
                "target": [999.0, 999.0],
                "oil": [999.0, 999.0],
                "cat": "future category",
            }
        ]
    )
    before = past.to_dict(), future.to_dict()
    panel = panel_from_window(past, future, spec, "W-SUN")
    series = panel.series[0]
    assert series.length == 4
    assert panel.target_support == "real"
    np.testing.assert_equal(series.target, [-2.0, np.nan, 4.0, np.nan])
    np.testing.assert_equal(series.dynamic["oil"], [3.0, np.nan, 5.0, np.nan])
    np.testing.assert_equal(series.dynamic["promo"], [0, np.nan, 1, 0, np.nan, np.nan, 7])
    assert series.statics == {"cat": "A"}
    assert (past.to_dict(), future.to_dict()) == before


def test_statics_follow_declared_types_and_preserve_missing_values(tmp_path):
    spec = fixture_spec(write_fev_fixture(tmp_path, n_series=3), static_columns=["size", "cat"])
    task = fixture_spec(spec.dataset_path).to_task()
    past, future = next(task.iter_windows()).get_input_data()
    past = past.remove_columns("cat").add_column("cat", [17, None, 0])
    past = past.add_column("size", [20, None, 0])
    panel = panel_from_window(past, future, spec, task.freq)
    assert panel.roles.static_categorical == ("cat",)
    assert panel.roles.static_numeric == ("size",)
    assert [series.statics["cat"] for series in panel.series] == ["17", None, "0"]
    np.testing.assert_equal([series.statics["size"] for series in panel.series], [20, np.nan, 0])
    assert all(isinstance(series.statics["size"], float) for series in panel.series)


@pytest.mark.parametrize(
    "freq,start,period", [("D", "2023-01-01", 7), ("h", "2023-01-01", 24), ("MS", "2023-01-01", 12)]
)
def test_custom_column_names_no_covariates_and_frequency(tmp_path, freq, start, period):
    from datasets import Dataset

    parquet = write_fev_fixture(
        tmp_path, n_series=2, freq=freq, start=start, known=(), past=(), statics=()
    )
    dataset = Dataset.from_parquet(str(parquet)).rename_columns(
        {"id": "item", "timestamp": "date", "target": "sales"}
    )
    custom = tmp_path / "custom.parquet"
    dataset.to_parquet(custom)
    spec = fixture_spec(
        custom,
        id_column="item",
        timestamp_column="date",
        target="sales",
        known_dynamic=[],
        past_dynamic=[],
        static_columns=[],
        categorical_statics=[],
        seasonality=99,
    )
    task = spec.to_task()
    past, future = next(task.iter_windows()).get_input_data()
    panel = panel_from_window(past, future, spec, task.freq)
    assert panel.freq == pd.tseries.frequencies.to_offset(freq)
    assert panel.seasonal_period == period
    assert panel.roles == Roles("sales", (), (), (), ())
    assert item_ids(panel) == list(past["item"])
    for series, row in zip(panel.series, past, strict=True):
        assert series.length == len(row["date"])
        np.testing.assert_equal(series.target, row["sales"])
        assert series.dynamic == series.statics == {}


@pytest.mark.parametrize("part", ["past", "future"])
@pytest.mark.parametrize("shift", [pd.Timedelta(days=1), pd.Timedelta(hours=1)])
def test_off_grid_timestamps_are_rejected_per_series(tmp_path, part, shift):
    spec = fixture_spec(write_fev_fixture(tmp_path, n_series=2))
    task = spec.to_task()
    past, future = next(task.iter_windows()).get_input_data()

    def move_timestamp(row):
        if row["id"] == "1":
            row["timestamp"][-1] += shift
        return row

    if part == "past":
        past = past.map(move_timestamp)
    else:
        future = future.map(move_timestamp)
    with pytest.raises(GridError) as exc:
        panel_from_window(past, future, spec, task.freq)
    assert exc.value.series_id == "1"
    assert exc.value.timestamp == (past if part == "past" else future)[1]["timestamp"][-1]
