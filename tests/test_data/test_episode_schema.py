"""Contracts for the v5 input schema, cutoff kinds and affine scale."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.episode import Item, Panel
from demand_forecast.data.episode import schema as sc
from demand_forecast.data.episode.calendar import calendar_arrays
from demand_forecast.data.episode.kinds import (
    flag_codes,
    infer_kind,
    infer_static_kind,
    is_missing,
    label_of,
)
from demand_forecast.data.episode.scale import Scale, compress, fit_scale, to_scaled
from demand_forecast.data.episode.schema import KIND_CATEGORY, KIND_FLAG, KIND_NUMBER
from demand_forecast.data.frequency import CANONICAL_CALENDAR_SCHEMA
from demand_forecast.data.scaling import seasonal_error
from tests.helpers.episodes import perturb_after, toy_panel


def test_dynamic_kinds():
    assert infer_kind(np.array([0, 1, 1, 0, np.nan])) == KIND_FLAG
    assert infer_kind(np.array([3.5, 7.25, 3.5])) == KIND_FLAG
    assert infer_kind(np.array(["a", "b", "c", None], dtype=object)) == KIND_CATEGORY
    assert infer_kind(np.tile([1.0, 2.0, 3.0], 4)) == KIND_CATEGORY
    assert infer_kind(np.arange(10, 15, dtype=float)) == KIND_NUMBER
    assert infer_kind(np.array([0.5, 1.25, 3.0])) == KIND_NUMBER
    assert infer_kind(np.zeros(40)) == KIND_NUMBER


@pytest.mark.parametrize("nonfinite", [np.inf, -np.inf])
def test_nonfinite_dynamic_values_are_not_whole_number_categories(nonfinite):
    assert infer_kind(np.tile([nonfinite, 1.0, 2.0], 4)) == KIND_NUMBER


def test_static_kinds():
    assert infer_static_kind(["a", "b", "a", None]) == "category"
    assert infer_static_kind([["x"], ["x", "y"], []]) == "multilabel"
    assert infer_static_kind([1, 2, 1, 2, 1, 2]) == "category"
    assert infer_static_kind([10.5, 11.0, 12.25, 9.0]) == "number"


def test_missing_and_labels():
    assert all(is_missing(v) for v in (None, float("nan"), "", [], ()))
    assert not any(is_missing(v) for v in (0, 0.0, False, "0", ["a"]))
    assert label_of(3) == label_of("3") and label_of(True) == "True"


@pytest.mark.parametrize("missing", [pd.NA, pd.NaT])
def test_pandas_missing_scalars_are_missing(missing):
    assert is_missing(missing)
    values = np.array(["a", "b", "a", missing], dtype=object)
    assert infer_kind(values) == KIND_FLAG
    np.testing.assert_array_equal(flag_codes(values, 4), [0.0, 1.0, 0.0, np.nan])


@pytest.mark.parametrize("dtype", ["Int64", "boolean", "string"])
def test_nullable_pandas_columns_do_not_create_a_missing_label(dtype):
    cells = ["a", "b", "a", pd.NA] if dtype == "string" else [0, 1, 0, pd.NA]
    values = pd.array(cells, dtype=dtype).to_numpy(dtype=object)
    assert infer_kind(values) == KIND_FLAG
    np.testing.assert_array_equal(flag_codes(values, 4), [0.0, 1.0, 0.0, np.nan])


def test_list_and_tuple_cells_keep_scalar_missing_semantics():
    values = np.empty(8, dtype=object)
    values[:] = [[], (), ["a", "b"], ("a", "b"), 0, False, "0", [pd.NA]]
    assert [is_missing(value) for value in values] == [True, True, *([False] * 6)]
    assert infer_kind(values) == KIND_CATEGORY
    assert np.isnan(flag_codes(values, 8)[:2]).all()
    assert np.isfinite(flag_codes(values, 8)[2:]).all()


@pytest.mark.parametrize("nonfinite", [np.inf, -np.inf])
def test_nonfinite_static_values_are_numbers(nonfinite):
    assert infer_static_kind([1.0, nonfinite, 1.0, nonfinite]) == "number"


def test_flag_codes_do_not_depend_on_spelling():
    a = flag_codes(np.array(["a", "b", "a", "a", "c"], dtype=object), 4)
    z = flag_codes(np.array(["z", "b", "z", "z", "c"], dtype=object), 4)
    assert a.tolist() == z.tolist() == [0.0, 1.0, 0.0, 0.0, 1.0]
    assert flag_codes(np.array([7.25, 3.5, 3.5, 3.5]), 4).tolist() == [1.0, 0.0, 0.0, 0.0]


def test_long_lag():
    assert (
        toy_panel().long_lag == 364
        and toy_panel(dated=False).long_lag is None
        and toy_panel().season == 7
    )


def test_scale_reads_only_the_prefix():
    y = np.random.default_rng(0).normal(size=200)
    z = y.copy()
    z[120:] = 1e9
    assert fit_scale(y, 120, 7) == fit_scale(z, 120, 7)


def test_scale_floors():
    assert fit_scale(np.full(50, 3.0), 50, 7) == Scale(3.0, 1.0)
    assert fit_scale(np.full(50, np.nan), 50, 7) == Scale(0.0, 1.0)


def test_negative_scale_cutoff_has_an_empty_prefix():
    values = np.arange(1.0, 11.0)
    perturbed = values.copy()
    perturbed[-5:] = 1e9
    for array in (values, perturbed):
        assert fit_scale(array, -3, 1) == fit_scale(array, 0, 1) == Scale(0.0, 1.0)


def test_negative_flag_cutoff_has_an_empty_prefix():
    values = np.array([0.0, 1.0, 0.0, np.nan, 1.0, 0.0, 1.0, 0.0])
    perturbed = values.copy()
    perturbed[-5:] = 42.0
    for array in (values, perturbed):
        codes = flag_codes(array, -3)
        np.testing.assert_array_equal(codes, flag_codes(array, 0))
        assert (codes[np.isfinite(array)] == 1.0).all()


def test_panel_validate():
    panel = toy_panel()
    panel.validate()
    item = panel.items[0]
    bad_items = (
        replace(item, dynamic={**item.dynamic, "temp": item.dynamic["temp"][:-1]}),
        replace(item, targets={"y": item.targets["y"].astype(np.float32)}),
    )
    for bad_item, column in zip(bad_items, ("temp", "y")):
        with pytest.raises(ValueError, match=rf"{item.item_id}.*{column}"):
            replace(panel, items=(bad_item, *panel.items[1:])).validate()
    for roles in (
        {key: role for key, role in panel.roles.items() if key != "temp"},
        {**panel.roles, "temp": "future"},
    ):
        with pytest.raises(ValueError, match=rf"{item.item_id}.*temp"):
            replace(panel, roles=roles).validate()
    with pytest.raises(ValueError, match="epoch"):
        replace(panel, freq=None).validate()


@pytest.mark.parametrize("aligned", [False, True])
def test_frequency_without_dates_keeps_periods_without_calendar(aligned):
    panel = replace(toy_panel(), freq="h", epoch=None, aligned=aligned)
    panel.validate()
    assert panel.timestamps(-2, 10) is None
    assert panel.season == 24 and panel.long_lag == 168
    assert panel.freq_class == sc.FREQ_CLASSES.index("hourly")
    assert calendar_arrays(panel, 0, 10) is None


@pytest.mark.parametrize("freq", ["not-a-frequency", "-1h", "0h"])
def test_frequency_without_dates_still_rejects_invalid_frequencies(freq):
    with pytest.raises(ValueError, match="freq"):
        replace(toy_panel(), freq=freq, epoch=None).validate()


@pytest.mark.parametrize("epoch", [pd.NaT, pd.Timestamp("2024-01-02")])
def test_dated_panel_rejects_missing_or_off_grid_epoch(epoch):
    with pytest.raises(ValueError, match="epoch"):
        replace(toy_panel(), freq="MS", epoch=epoch).validate()


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_panel_target_names_must_match_the_first_item(change):
    panel = toy_panel(n_targets=2)
    item = panel.items[1]
    targets = item.targets.copy()
    if change == "missing":
        del targets["y1"]
    else:
        targets["extra"] = targets["y"].copy()
    changed = replace(item, targets=targets)
    with pytest.raises(
        ValueError, match=rf"{item.item_id}.*target.*{('y1' if change == 'missing' else 'extra')}"
    ):
        replace(panel, items=(panel.items[0], changed, *panel.items[2:])).validate()


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_panel_dynamic_names_must_match_the_first_item(change):
    panel = toy_panel()
    item = panel.items[1]
    dynamic = item.dynamic.copy()
    if change == "missing":
        del dynamic["temp"]
    else:
        dynamic["extra"] = dynamic["temp"].copy()
    changed = replace(item, dynamic=dynamic)
    with pytest.raises(
        ValueError,
        match=rf"{item.item_id}.*dynamic.*{('temp' if change == 'missing' else 'extra')}",
    ):
        replace(
            panel,
            roles={**panel.roles, "extra": "past"},
            items=(panel.items[0], changed, *panel.items[2:]),
        ).validate()


def test_panel_static_keys_may_differ_between_items():
    panel = toy_panel()
    item = replace(panel.items[1], static={"brand": "other", "extra": 3.5})
    replace(panel, items=(panel.items[0], item, *panel.items[2:])).validate()


def test_mixed_labels_preserve_missing_and_renaming():
    values = np.empty(10, dtype=object)
    values[:] = [3, "3", True, None, float("nan"), "", [], "True", "3", "new"]
    assert infer_kind(values[:8]) == KIND_FLAG
    codes = flag_codes(values, 8)
    np.testing.assert_array_equal(codes, [0, 0, 1, np.nan, np.nan, np.nan, np.nan, 1, 0, 1])
    renamed = values.copy()
    renamed[[0, 1, 8]] = "renamed number"
    renamed[[2, 7]] = "renamed bool"
    np.testing.assert_array_equal(flag_codes(renamed, 8), codes)
    assert infer_kind(renamed[:8]) == KIND_FLAG
    assert infer_static_kind(values.tolist()) == "multilabel"


@pytest.mark.parametrize("labels", [("a", "z"), ("z", "a"), (3, True)])
def test_flag_ties_use_first_occurrence_and_ignore_future_counts(labels):
    first, later = labels
    values = np.array([None, first, later, first, later, later, later], dtype=object)
    np.testing.assert_array_equal(flag_codes(values, 5), [np.nan, 0, 1, 0, 1, 1, 1])
    assert flag_codes(values, 5).dtype == np.float64


def test_flags_without_a_prefix_and_missing_only_columns():
    values = np.array([None, "", "a", "b"], dtype=object)
    np.testing.assert_array_equal(flag_codes(values, 2), [np.nan, np.nan, 1, 1])
    np.testing.assert_array_equal(flag_codes(values, 0), [np.nan, np.nan, 1, 1])
    assert flag_codes(np.array([], dtype=object), 0).shape == (0,)
    assert infer_kind(np.array([np.nan, np.nan])) == KIND_NUMBER
    assert infer_kind(np.array([None, ""], dtype=object)) == KIND_CATEGORY


@pytest.mark.parametrize(
    ("values", "kind"),
    [
        (np.tile([1, 2, 3], 3), KIND_NUMBER),
        (np.tile(np.arange(16), 4), KIND_CATEGORY),
        (np.tile(np.arange(17), 4), KIND_NUMBER),
        (np.array([True, True]), KIND_CATEGORY),
        (np.array([False, True]), KIND_FLAG),
        (np.array(["same", "same"]), KIND_CATEGORY),
    ],
)
def test_dynamic_kind_boundaries(values, kind):
    assert infer_kind(values) == kind


@pytest.mark.parametrize(
    ("values", "kind"),
    [
        ([], "number"),
        ([None, np.nan, ""], "number"),
        ([[], None], "multilabel"),
        ([("x",), None], "multilabel"),
        ([True, False], "category"),
        ([1, 2, 3], "number"),
        ([1, 1, 2, 2], "category"),
    ],
)
def test_static_kind_boundaries(values, kind):
    assert infer_static_kind(values) == kind


@pytest.mark.parametrize(
    ("freq", "epoch", "freq_class", "season", "long_lag"),
    [
        (None, None, "undated", 1, None),
        ("15min", "2024-01-01", "subhourly", 96, 672),
        ("H", "2024-01-01", "hourly", 24, 168),
        ("h", "2024-01-01", "hourly", 24, 168),
        ("120min", "2024-01-01", "hourly", 1, None),
        ("D", "2024-01-01", "daily", 7, 364),
        ("W-MON", "2024-01-01", "weekly", 1, 52),
        ("M", "2024-01-31", "monthly", 12, 24),
        ("ME", "2024-01-31", "monthly", 12, 24),
        ("MS", "2024-01-01", "monthly", 12, 24),
        ("QS-JAN", "2024-01-01", "quarterly", 4, 8),
        ("YS", "2024-01-01", "yearly", 1, None),
    ],
)
def test_panel_frequency_properties(freq, epoch, freq_class, season, long_lag):
    panel = replace(toy_panel(), freq=freq, epoch=None if epoch is None else pd.Timestamp(epoch))
    panel.validate()
    assert panel.freq_class == sc.FREQ_CLASSES.index(freq_class)
    assert panel.season == season
    assert panel.long_lag == long_lag


@pytest.mark.parametrize(
    ("freq", "epoch", "freq_class"),
    [
        ("BME", "2024-01-31", "monthly"),
        ("BMS", "2024-01-01", "monthly"),
        ("BQE", "2024-03-29", "quarterly"),
        ("BQS", "2024-01-01", "quarterly"),
        ("BYE", "2024-12-31", "yearly"),
        ("BYS", "2024-01-01", "yearly"),
        ("bh", "2024-01-01 09:00", "hourly"),
        ("B", "2024-01-01", "daily"),
    ],
)
def test_business_anchors_get_their_family_class(freq, epoch, freq_class):
    # Only the class is pinned: the season of a business grid follows the repository's
    # pretraining convention (pretrain/periods.py), which this package does not change.
    panel = replace(toy_panel(), freq=freq, epoch=pd.Timestamp(epoch))
    panel.validate()
    assert panel.freq_class == sc.FREQ_CLASSES.index(freq_class)


def test_timestamps_use_absolute_grid_and_calendar_offsets():
    panel = toy_panel()
    pd.testing.assert_index_equal(
        panel.timestamps(3, 5), pd.date_range(panel.epoch, periods=8, freq="D")[3:]
    )
    pd.testing.assert_index_equal(
        panel.timestamps(-2, 4), pd.date_range(panel.epoch - pd.Timedelta(days=2), periods=4)
    )
    monthly = replace(panel, freq="ME", epoch=pd.Timestamp("2024-01-31"))
    pd.testing.assert_index_equal(
        monthly.timestamps(1, 3), pd.date_range("2024-02-29", periods=3, freq="ME")
    )
    daily = replace(panel, epoch=pd.Timestamp("2024-03-30", tz="Europe/Rome"))
    pd.testing.assert_index_equal(
        daily.timestamps(2, 3), pd.date_range(daily.epoch, periods=5, freq="D")[2:]
    )
    assert panel.timestamps(0, 0).empty
    assert toy_panel(dated=False).timestamps(0, 10) is None


@pytest.mark.parametrize(
    ("freq", "epoch", "reference_start", "start", "reference_position"),
    [
        ("D", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"), 35, 35),
        ("ME", pd.Timestamp("2024-01-31"), pd.Timestamp("2024-01-31"), 35, 35),
        (
            "D",
            pd.Timestamp("2024-03-30", tz="Europe/Rome"),
            pd.Timestamp("2024-03-30", tz="Europe/Rome"),
            2,
            2,
        ),
        ("D", pd.Timestamp("2024-01-01"), pd.Timestamp("2023-12-01"), -2, 29),
    ],
)
def test_timestamps_match_a_long_grid_slice(
    freq, epoch, reference_start, start, reference_position
):
    panel = replace(toy_panel(), freq=freq, epoch=epoch)
    grid = pd.date_range(reference_start, periods=100, freq=freq)
    pd.testing.assert_index_equal(
        panel.timestamps(start, 5), grid[reference_position : reference_position + 5]
    )


def test_panel_validation_rejects_invalid_grid_and_shapes():
    panel = toy_panel()
    with pytest.raises(ValueError, match="aligned"):
        replace(panel, aligned=False).validate()
    replace(toy_panel(dated=False), aligned=False).validate()
    with pytest.raises(ValueError, match="epoch"):
        replace(panel, freq=None).validate()
    with pytest.raises(ValueError, match="freq"):
        replace(panel, freq="not-a-frequency").validate()
    item = panel.items[0]
    for column in (np.ones((400, 1)), np.ones(400, dtype=np.float32)):
        bad = replace(item, dynamic={**item.dynamic, "temp": column})
        with pytest.raises(ValueError, match=rf"{item.item_id}.*temp"):
            replace(panel, items=(bad,)).validate()


def test_schema_constants_match_plan():
    assert sc.PATCH == 16
    assert sc.MAX_CONTEXT == 2048
    assert sc.MAX_HORIZON == 288
    assert sc.BANK_WINDOW == 8192
    assert (
        sc.MAX_FORECAST_ROWS,
        sc.MAX_FEATURE_ROWS,
        sc.MAX_CONTEXT_ROWS,
        sc.MAX_ATTRIBUTE_TOKENS,
    ) == (16, 16, 8, 16)
    assert sc.MAX_STATIC_NUMBERS == 16
    assert sc.QUANTILES == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert (sc.KIND_NUMBER, sc.KIND_FLAG, sc.KIND_CATEGORY) == (0, 1, 2)
    assert (
        sc.ROLE_FORECAST,
        sc.ROLE_SIBLING,
        sc.ROLE_KNOWN,
        sc.ROLE_PAST,
        sc.ROLE_PEER,
        sc.ROLE_PANEL,
    ) == (0, 1, 2, 3, 4, 5)
    assert len(sc.STEP_FEATURES) == 15
    assert sc.STEP_FAMILIES == {
        "rolling": slice(0, 8),
        "lags": slice(8, 12),
        "zeros": slice(12, 15),
    }
    assert len(sc.SUMMARY_FEATURES) == 28
    assert sc.VALUE_STATS == 6
    assert sc.CH_VALUE == 0
    assert sc.CH_OBSERVED == 1
    assert sc.CH_SUPPORT == 2
    assert sc.CH_STEP == 3
    assert sc.CH_STEP_FLAG == 18
    assert sc.CH_VALUE_STATS == 33
    assert sc.N_CHANNELS == 39
    assert sc.N_SUMMARY == 28
    assert sc.N_VALUE_TOKEN == 57
    assert sc.CALENDAR_FIELDS == ("hour", "dow", "dom", "month", "woy", "quarter")
    assert sc.CALENDAR_SIZES == (24, 7, 31, 12, 53, 4)
    assert sc.N_PHASES == 16
    assert sc.FREQ_CLASSES == (
        "undated",
        "subhourly",
        "hourly",
        "daily",
        "weekly",
        "monthly",
        "quarterly",
        "yearly",
    )


def test_schema_dimensions_and_frozen_records():
    assert len(sc.STEP_FEATURES) == 15
    assert len(sc.SUMMARY_FEATURES) == sc.N_SUMMARY == 28
    assert sc.N_VALUE_TOKEN == 2 * sc.N_SUMMARY + 1
    assert sc.N_PHASES == len(CANONICAL_CALENDAR_SCHEMA) == 16
    assert sc.CH_STEP_FLAG == sc.CH_STEP + len(sc.STEP_FEATURES)
    assert sc.CH_VALUE_STATS == sc.CH_STEP_FLAG + len(sc.STEP_FEATURES)
    assert sc.N_CHANNELS == sc.CH_VALUE_STATS + sc.VALUE_STATS
    assert sc.PATCH == 16 and sc.MAX_CONTEXT == 2048 and sc.MAX_HORIZON == 288
    panel = toy_panel()
    assert isinstance(panel, Panel) and isinstance(panel.items[0], Item)
    with pytest.raises(FrozenInstanceError):
        panel.name = "changed"
    with pytest.raises(FrozenInstanceError):
        panel.items[0].start = 1


def test_scale_uses_seasonal_error_without_closing_gaps():
    values = np.array([1.0, np.nan, 5.0, 9.0, 20.0])
    before = values.copy()
    scale = fit_scale(values, 4, 2)
    assert scale.loc == 5.0
    assert scale.scale == seasonal_error(values[:4], 2) == 4.0
    np.testing.assert_array_equal(values, before)


def test_scale_deviation_fallbacks_and_empty_prefix():
    assert fit_scale(np.array([1.0, np.nan, 5.0]), 3, 1) == Scale(3.0, 2.0)
    assert fit_scale(np.array([1.0, 3.0, 1.0, 3.0]), 4, 2) == Scale(2.0, 1.0)
    assert fit_scale(np.array([9.0]), 0, 7) == Scale(0.0, 1.0)
    assert fit_scale(np.array([9.0]), 1, 7) == Scale(9.0, 1.0)
    assert fit_scale(np.array([np.inf, -np.inf, np.nan]), 3, 1) == Scale(0.0, 1.0)


def test_scaling_and_compression_are_separate_and_preserve_missing():
    values = np.array([0.0, 2.0, np.nan, 6.0])
    scaled = to_scaled(values, Scale(2.0, 2.0))
    np.testing.assert_array_equal(scaled, [-1.0, 0.0, np.nan, 2.0])
    np.testing.assert_array_equal(compress(scaled), np.arcsinh(scaled))
    assert scaled.dtype == np.float64
    assert values[0] == 0.0


def test_toy_panel_is_reproducible_ragged_and_typed():
    panel = toy_panel(n_targets=3)
    other = toy_panel(n_targets=3)
    assert panel.items[0].start == 0 and len(panel.items[0].targets["y"]) == 400
    assert len({item.start for item in panel.items}) > 1
    assert panel.roles["promo"] == "known" and panel.roles["temp"] == "past"
    for item, repeated in zip(panel.items, other.items):
        assert set(item.targets) == {"y", "y1", "y2"}
        assert set(item.dynamic) == {"promo", "temp", "event"}
        assert set(item.static) == {"brand", "tags", "size", "dept"}
        assert isinstance(item.static["tags"], list)
        assert infer_kind(item.dynamic["event"][: 300 - item.start]) == KIND_CATEGORY
        assert infer_kind(item.dynamic["promo"][: 300 - item.start]) == KIND_FLAG
        assert infer_kind(item.dynamic["temp"][: 300 - item.start]) == KIND_NUMBER
        event_prefix = item.dynamic["event"][: 200 - item.start]
        assert len({value for value in event_prefix if value is not None}) >= 3
        assert any(value is None for value in event_prefix)
        for column in item.dynamic:
            np.testing.assert_array_equal(item.dynamic[column], repeated.dynamic[column])
        for column in item.targets:
            assert item.targets[column].dtype == np.float64
            np.testing.assert_array_equal(item.targets[column], repeated.targets[column])
    panel.validate()
    assert toy_panel(dated=False).freq is None and toy_panel(dated=False).epoch is None


def test_toy_panel_tags_have_no_duplicates_and_share_labels():
    panel = toy_panel()
    for item in panel.items:
        assert len(item.static["tags"]) == len(set(item.static["tags"])) == 2
    assert set(panel.items[0].static["tags"]) & set(panel.items[2].static["tags"])


@pytest.mark.parametrize("keep_known", [False, True])
def test_perturb_after_uses_grid_indices_and_copies(keep_known):
    panel = toy_panel(n_targets=2)
    changed = perturb_after(panel, 5, value=-123.0, keep_known=keep_known, only_items=[1])
    for position, (old, new) in enumerate(zip(panel.items, changed.items)):
        for name, source in {**old.targets, **old.dynamic}.items():
            actual = {**new.targets, **new.dynamic}[name]
            expected = source.copy()
            if position == 1 and not (keep_known and panel.roles.get(name) == "known"):
                expected[np.arange(len(source)) + old.start >= 5] = -123.0
            np.testing.assert_array_equal(actual, expected)
            assert not np.shares_memory(actual, source)
    changed.items[0].static["tags"].append("copied")
    assert "copied" not in panel.items[0].static["tags"]
    changed.validate()


def test_perturb_after_defaults_to_all_items():
    panel = toy_panel()
    changed = perturb_after(panel, 0)
    for item in changed.items:
        for values in (*item.targets.values(), *item.dynamic.values()):
            assert (values == 1e6).all()
