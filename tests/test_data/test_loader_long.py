import sys

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.frequency import GridError
from demand_forecast.data.loader_long import load_long, read_table, series_key
from demand_forecast.data.panel import Roles
from tests.helpers.long_fixture import long_cfg


def test_default_panel_metadata_and_sorted_rows(tmp_path):
    cfg = long_cfg(tmp_path)
    frame = pd.read_csv(cfg.path).iloc[::-1]
    frame.to_csv(cfg.path, index=False)
    panel = load_long(cfg)
    assert len(panel) == 2
    assert {s.series_id for s in panel.series} == {"p1|s1", "p2|s1"}
    assert panel.roles == Roles("y", ("promo",), (), (), ())
    assert panel.freq.freqstr == "W-SUN" and panel.seasonal_period == 1
    assert panel.target_support == "nonnegative"
    for s in panel.series:
        assert s.start == pd.Timestamp("2023-01-01") and s.history_length == 6
        np.testing.assert_equal(s.target, [1, 2, 3, 4, 5, 6])
        np.testing.assert_equal(s.dynamic["promo"], [0, 1, 0, 1, 0, 1])
        assert s.statics == {}


def test_duplicate_rows_error_or_sum_keeping_all_nan(tmp_path):
    cfg = long_cfg(
        tmp_path,
        rows=[
            ("p1", "s1", "2023-01-01", 2.0),
            ("p1", "s1", "2023-01-01", 3.0),
            ("p1", "s1", "2023-01-08", np.nan),
            ("p1", "s1", "2023-01-08", np.nan),
            ("p1", "s1", "2023-01-15", 1.0),
        ],
    )
    with pytest.raises(ValueError, match=r"p1\|s1.*2023-01-01"):
        load_long(cfg)
    s = load_long(cfg.model_copy(update={"duplicates": "sum", "missing_target": "nan"})).series[0]
    assert s.series_id == "p1|s1"
    np.testing.assert_equal(s.target, [5.0, np.nan, 1.0])
    np.testing.assert_equal(s.dynamic["promo"], [1, 1, 0])


def test_duplicate_sum_takes_last_covariate_even_when_missing(tmp_path):
    rows = pd.DataFrame(
        {
            "product_id": ["p1", "p1", "p1", "p2"],
            "store_id": ["s1"] * 4,
            "timestamp": ["2023-01-01", "2023-01-08", "2023-01-01", "2023-01-01"],
            "y": [2.0, 9.0, np.nan, 100.0],
            "promo": [1.0, 0.0, np.nan, 1.0],
            "oil": [4.0, 8.0, 6.0, 10.0],
        }
    )
    panel = load_long(
        long_cfg(tmp_path, rows=rows, duplicates="sum", past_dynamic=["oil"], freq="W-SUN")
    )
    s = panel.series[panel.index_of("p1|s1")]
    np.testing.assert_equal(s.target, [2, 9])
    np.testing.assert_equal(s.dynamic["promo"], [np.nan, 0])
    np.testing.assert_equal(s.dynamic["oil"], [6, 8])
    np.testing.assert_equal(panel.series[panel.index_of("p2|s1")].target, [100])


@pytest.mark.parametrize("ids", [("a|b", "c"), ("a", "b|c")])
def test_pipe_in_id_value_is_rejected(tmp_path, ids):
    with pytest.raises(ValueError, match="must not contain"):
        load_long(long_cfg(tmp_path, rows=[(*ids, "2023-01-01", 1.0)]))


@pytest.mark.parametrize("ids", [(None, "s1"), ("p1", None)])
def test_null_series_keys_are_rejected(tmp_path, ids):
    with pytest.raises(ValueError, match="null series key"):
        load_long(long_cfg(tmp_path, rows=[(*ids, "2023-01-01", 1.0)]))


def test_series_key_preserves_index_and_column_order():
    frame = pd.DataFrame({"product": [2, 1], "store": ["b", "a"]}, index=[4, 2])
    original = frame.copy(deep=True)
    keys = series_key(frame, ["store", "product"])
    assert keys.tolist() == ["b|2", "a|1"] and keys.index.equals(frame.index)
    assert series_key(frame, ["product"]).tolist() == ["2", "1"]
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="id_columns"):
        series_key(frame, [])


def test_static_join_rules(tmp_path):
    panel = load_long(long_cfg(tmp_path, statics={"p1|s1": {"cat": "A"}}))
    assert panel.series[panel.index_of("p1|s1")].statics == {"cat": "A"}
    assert panel.series[panel.index_of("p2|s1")].statics["cat"] is None
    with pytest.raises(ValueError, match="duplicate static key"):
        load_long(long_cfg(tmp_path, statics_duplicate=True))


def test_static_join_uses_full_key_and_declared_columns(tmp_path):
    cfg = long_cfg(
        tmp_path,
        statics={
            "p1|other": {"cat": "wrong", "size": 999, "unused": "ignore"},
            "p2|s1": {"cat": "B", "size": 2.5, "unused": "ignore"},
        },
        static_categorical=["cat", "store_id"],
        static_numeric=["size"],
    )
    panel = load_long(cfg)
    assert panel.roles.static_categorical == ("cat", "store_id")
    assert panel.roles.static_numeric == ("size",)
    assert panel.series[panel.index_of("p1|s1")].statics == {
        "cat": None,
        "store_id": None,
        "size": None,
    }
    assert panel.series[panel.index_of("p2|s1")].statics == {
        "cat": "B",
        "store_id": "s1",
        "size": 2.5,
    }


def test_missing_static_values_and_no_table_are_none(tmp_path):
    cfg = long_cfg(
        tmp_path,
        statics={"p1|s1": {"cat": None, "size": np.nan}},
        static_numeric=["size"],
    )
    for config in (cfg, cfg.model_copy(update={"static_path": None})):
        assert all(s.statics == {"cat": None, "size": None} for s in load_long(config).series)


@pytest.mark.parametrize("bad_id,message", [(None, "null series key"), ("p|1", "must not contain")])
def test_static_keys_are_validated(tmp_path, bad_id, message):
    cfg = long_cfg(tmp_path, statics={"p1|s1": {"cat": "A"}})
    frame = pd.read_csv(cfg.static_path)
    frame["product_id"] = bad_id
    frame.to_csv(cfg.static_path, index=False)
    with pytest.raises(ValueError, match=message):
        load_long(cfg)


def test_string_dynamic_column_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="indicator encoding"):
        load_long(long_cfg(tmp_path, promo_values=["a", "b"]))


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
@pytest.mark.parametrize("role", ["known_dynamic", "past_dynamic"])
def test_string_dynamic_dtypes_are_rejected_before_aggregation(tmp_path, monkeypatch, dtype, role):
    overrides = {"known_dynamic": [], role: ["promo"]}
    cfg = long_cfg(tmp_path, resample="W-SUN", **overrides)
    frame = pd.read_csv(cfg.path)
    frame["promo"] = pd.Series(["a", "b"] * 6, dtype=dtype)
    monkeypatch.setattr("demand_forecast.data.loader_long.read_table", lambda path: frame)
    with pytest.raises(ValueError, match="indicator encoding"):
        load_long(cfg)


@pytest.mark.parametrize("dtype", ["object", "string", "Float64"])
@pytest.mark.parametrize("resample", [None, "W-SUN"])
def test_all_null_columns_keep_missingness_across_dtypes(tmp_path, monkeypatch, dtype, resample):
    cfg = long_cfg(tmp_path, past_dynamic=["oil"], resample=resample)
    frame = pd.read_csv(cfg.path)
    frame["y"] = pd.Series([None] * len(frame), dtype=dtype)
    frame["promo"] = pd.Series([None] * len(frame), dtype=dtype)
    frame["oil"] = pd.Series([1, None, 3, None, None, None] * 2, dtype="Float64")
    original = frame.copy(deep=True)
    monkeypatch.setattr("demand_forecast.data.loader_long.read_table", lambda path: frame)
    for s in load_long(cfg).series:
        assert s.history_length == 3 and len(s.dynamic["promo"]) == 6
        assert np.isnan(s.target).all() and np.isnan(s.dynamic["promo"]).all()
        np.testing.assert_equal(s.dynamic["oil"], [1, np.nan, 3])
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("missing_target,absent", [("zero", 0.0), ("nan", np.nan)])
def test_gaps_explicit_nans_and_future_rows(tmp_path, missing_target, absent):
    cfg = long_cfg(
        tmp_path,
        rows=[
            ("p1", "s1", "2023-01-01", 1.0),
            ("p1", "s1", "2023-01-15", np.nan),
            ("p1", "s1", "2023-01-22", 4.0),
            ("p1", "s1", "2023-02-05", np.nan),
        ],
        freq="W-SUN",
        missing_target=missing_target,
        promo_values=[0, 1, 0, 1],
    )
    s = load_long(cfg).series[0]
    assert s.history_length == 4
    np.testing.assert_equal(s.target, [1, absent, np.nan, 4])
    np.testing.assert_equal(s.dynamic["promo"], [0, np.nan, 1, 0, np.nan, 1])


@pytest.mark.parametrize("has_past", [False, True])
def test_history_ends_at_last_target_or_past_covariate(tmp_path, has_past):
    cfg = long_cfg(tmp_path, past_dynamic=["oil"] if has_past else [])
    frame = pd.read_csv(cfg.path)
    frame["y"] = [1, 2, np.nan, np.nan, np.nan, np.nan] * 2
    frame["oil"] = [5, 6, np.nan, 0, np.nan, np.nan] * 2
    frame.to_csv(cfg.path, index=False)
    for s in load_long(cfg).series:
        assert s.history_length == (4 if has_past else 2)
        assert len(s.dynamic["promo"]) == 6
        if has_past:
            np.testing.assert_equal(s.target, [1, 2, np.nan, np.nan])
            np.testing.assert_equal(s.dynamic["oil"], [5, 6, np.nan, 0])


def test_all_nan_series_has_known_future_but_no_history(tmp_path):
    cfg = long_cfg(tmp_path)
    frame = pd.read_csv(cfg.path)
    frame["y"] = np.nan
    frame.to_csv(cfg.path, index=False)
    for s in load_long(cfg).series:
        assert s.history_length == s.target.size == 0
        assert len(s.dynamic["promo"]) == 6


def test_short_series_inherits_panel_frequency(tmp_path):
    cfg = long_cfg(tmp_path)
    frame = pd.read_csv(cfg.path).iloc[:7]
    frame.to_csv(cfg.path, index=False)
    panel = load_long(cfg)
    assert panel.freq.freqstr == "W-SUN"
    assert panel.series[panel.index_of("p2|s1")].history_length == 1


def test_uninferable_frequency_requires_pin(tmp_path):
    cfg = long_cfg(tmp_path, rows=[("p1", "s1", "2023-01-01", 1)])
    with pytest.raises(GridError, match="pin data.freq"):
        load_long(cfg)
    assert load_long(cfg.model_copy(update={"freq": "W-SUN"})).series[0].length == 1


def test_disagreeing_inferred_frequencies_are_rejected(tmp_path):
    cfg = long_cfg(tmp_path)
    frame = pd.read_csv(cfg.path)
    frame.loc[frame.product_id == "p2", "timestamp"] = pd.date_range(
        "2023-01-02", periods=6, freq="W-MON"
    ).strftime("%Y-%m-%d")
    frame.to_csv(cfg.path, index=False)
    with pytest.raises(GridError, match="disagree"):
        load_long(cfg)


@pytest.mark.parametrize("timestamp", ["2023-02-06", "2023-02-05 13:00"])
def test_grid_validation_includes_future_rows(tmp_path, timestamp):
    cfg = long_cfg(tmp_path, freq="W-SUN")
    frame = pd.read_csv(cfg.path)
    frame.loc[5, ["timestamp", "y"]] = [timestamp, np.nan]
    frame.to_csv(cfg.path, index=False)
    with pytest.raises(GridError) as exc:
        load_long(cfg)
    assert exc.value.series_id == "p1|s1"
    assert exc.value.timestamp == pd.Timestamp(timestamp)


@pytest.mark.parametrize("resample", [None, "W-SUN"])
def test_null_timestamp_is_not_silently_dropped(tmp_path, resample):
    cfg = long_cfg(tmp_path, freq="W-SUN", resample=resample)
    frame = pd.read_csv(cfg.path)
    frame.loc[1, "timestamp"] = None
    frame.to_csv(cfg.path, index=False)
    with pytest.raises(ValueError, match="timestamp"):
        load_long(cfg)


@pytest.mark.parametrize("missing_target,absent", [("zero", 0.0), ("nan", np.nan)])
def test_resample_sums_targets_means_covariates_and_preserves_missingness(
    tmp_path, missing_target, absent
):
    rows = pd.DataFrame(
        {
            "product_id": ["p1"] * 7,
            "store_id": ["s1"] * 7,
            "timestamp": [
                "2023-01-02 12:00",
                "2023-01-03 12:00",
                "2023-01-09 12:00",
                "2023-01-10 12:00",
                "2023-01-23 12:00",
                "2023-01-30 12:00",
                "2023-01-31 12:00",
            ],
            "y": [2, 3, np.nan, np.nan, 4, np.nan, np.nan],
            "promo": [0, 1, 1, 0, 1, 0, 1],
            "oil": [10, 20, np.nan, np.nan, 40, np.nan, np.nan],
        }
    )
    cfg = long_cfg(
        tmp_path, rows=rows, past_dynamic=["oil"], missing_target=missing_target, freq="W-SUN"
    )
    with pytest.raises(GridError):
        load_long(cfg.model_copy(update={"freq": "W-SUN"}))
    panel = load_long(cfg.model_copy(update={"resample": "W-SUN"}))
    assert panel.freq.freqstr == "W-SUN" and panel.seasonal_period == 1
    s = panel.series[0]
    assert s.start == pd.Timestamp("2023-01-08") and s.history_length == 4
    np.testing.assert_equal(s.target, [5, np.nan, absent, 4])
    np.testing.assert_equal(s.dynamic["promo"], [0.5, 0.5, np.nan, 1, 0.5])
    np.testing.assert_equal(s.dynamic["oil"], [15, np.nan, np.nan, 40])


def test_frequency_can_be_inferred_after_resampling(tmp_path):
    cfg = long_cfg(
        tmp_path,
        rows=[
            ("p1", "s1", "2023-01-02", 2),
            ("p1", "s1", "2023-01-10", 3),
            ("p1", "s1", "2023-01-16", 4),
        ],
        resample="W-SUN",
    )
    panel = load_long(cfg)
    assert panel.freq.freqstr == "W-SUN" and panel.series[0].start == pd.Timestamp("2023-01-08")
    np.testing.assert_equal(panel.series[0].target, [2, 3, 4])


def test_duplicates_are_resolved_before_resampling(tmp_path):
    cfg = long_cfg(
        tmp_path,
        rows=[
            ("p1", "s1", "2023-01-02", 1, 0),
            ("p1", "s1", "2023-01-02", 2, 1),
            ("p1", "s1", "2023-01-03", 3, 0),
        ],
        freq="W-SUN",
        resample="W-SUN",
    )
    with pytest.raises(ValueError, match=r"p1\|s1.*2023-01-02"):
        load_long(cfg)
    s = load_long(cfg.model_copy(update={"duplicates": "sum"})).series[0]
    np.testing.assert_equal(s.target, [6])
    np.testing.assert_equal(s.dynamic["promo"], [0.5])


def test_resampling_does_not_override_pinned_grid(tmp_path):
    with pytest.raises(GridError):
        load_long(long_cfg(tmp_path, freq="W-MON", resample="W-SUN"))


def test_custom_columns_without_covariates_and_real_support(tmp_path):
    rows = pd.DataFrame({"id": [1, 1], "date": ["2023-01-01", "2023-01-03"], "sales": [-2, 3]})
    cfg = long_cfg(
        tmp_path,
        rows=rows,
        id_columns=["id"],
        timestamp_column="date",
        target="sales",
        target_support="real",
        known_dynamic=[],
        freq="D",
    )
    panel = load_long(cfg)
    assert panel.roles == Roles("sales", (), (), (), ())
    assert panel.target_support == "real" and panel.seasonal_period == 7
    assert panel.series[0].series_id == "1" and panel.series[0].dynamic == {}
    np.testing.assert_equal(panel.series[0].target, [-2, 0, 3])


@pytest.mark.parametrize(
    "static,column", [(False, "timestamp"), (False, "y"), (False, "promo"), (True, "cat")]
)
def test_missing_configured_columns_raise_clear_error(tmp_path, static, column):
    cfg = long_cfg(tmp_path, statics={"p1|s1": {"cat": "A"}})
    path = cfg.static_path if static else cfg.path
    pd.read_csv(path).drop(columns=[column]).to_csv(path, index=False)
    with pytest.raises(ValueError, match=column):
        load_long(cfg)


def test_read_table_csv_and_unsupported_extension(tmp_path):
    cfg = long_cfg(tmp_path)
    pd.testing.assert_frame_equal(read_table(str(cfg.path)), pd.read_csv(cfg.path))
    with pytest.raises(ValueError, match="Unsupported"):
        read_table(tmp_path / "input.json")


def test_read_parquet_reports_missing_pyarrow(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(ImportError, match="pyarrow"):
        read_table(tmp_path / "input.parquet")


def test_parquet_long_and_statics_match_csv(tmp_path):
    pytest.importorskip("pyarrow")
    cfg = long_cfg(tmp_path, statics={"p1|s1": {"cat": "A", "size": 2.5}}, static_numeric=["size"])
    expected = load_long(cfg)
    updates = {}
    for field in ("path", "static_path"):
        csv_path = getattr(cfg, field)
        frame = pd.read_csv(csv_path)
        path = csv_path.with_suffix(".parquet")
        frame.to_parquet(path, index=False)
        pd.testing.assert_frame_equal(read_table(path), frame)
        updates[field] = path
    actual = load_long(cfg.model_copy(update=updates))
    assert actual.roles == expected.roles and actual.freq == expected.freq
    for s, e in zip(actual.series, expected.series):
        assert (s.series_id, s.start, s.history_length, s.statics) == (
            e.series_id,
            e.start,
            e.history_length,
            e.statics,
        )
        np.testing.assert_equal(s.target, e.target)
        np.testing.assert_equal(s.dynamic["promo"], e.dynamic["promo"])
