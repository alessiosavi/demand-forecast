"""Exact scalar references for episode cell handling and calendar reuse."""

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.episode import (
    EpisodeBuilder,
    Item,
    Panel,
    builder,
    kinds,
    screening,
    value_stats,
)
from demand_forecast.data.episode.calendar import calendar_arrays
from demand_forecast.data.episode.schema import KIND_CATEGORY, KIND_FLAG, KIND_NUMBER


def _observed(values):
    return ~np.asarray(np.frompyfunc(kinds.is_missing, 1, 1)(values), dtype=bool)


def _labels(values):
    return np.asarray(np.frompyfunc(kinds.label_of, 1, 1)(values), dtype=object)


def _infer_kind(values):
    values = np.asarray(values)
    observed = values[_observed(values)]
    distinct = len(np.unique(_labels(observed)))
    if distinct == 2:
        return KIND_FLAG
    if values.dtype.kind in "OUSb":
        return KIND_CATEGORY
    if (
        3 <= distinct <= 16
        and len(observed) >= 4 * distinct
        and np.isfinite(observed).all()
        and np.all(observed == np.floor(observed))
    ):
        return KIND_CATEGORY
    return KIND_NUMBER


def _flag_codes(values, cutoff):
    values = np.asarray(values)
    cutoff = max(0, cutoff)
    observed = _observed(values)
    labels = _labels(values)
    prefix_codes, prefix_labels = pd.factorize(labels[:cutoff][observed[:cutoff]], sort=False)
    codes = np.full(values.shape, np.nan, dtype=np.float64)
    codes[observed] = 1.0
    if len(prefix_labels):
        majority = prefix_labels[np.argmax(np.bincount(prefix_codes))]
        codes[observed & (labels == majority)] = 0.0
    return codes


def _cell_labels(value):
    if kinds.is_missing(value):
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(sorted({kinds.label_of(v) for v in value if not kinds.is_missing(v)}))
    return (kinds.label_of(value),)


def _label_sets(values):
    return np.frompyfunc(_cell_labels, 1, 1)(values)


def _object_cells(*cells):
    result = np.empty(len(cells), dtype=object)
    result[:] = cells
    return result


def _samples():
    rng = np.random.default_rng(541)
    strings = ("", "a", "é", "🐈", "a\0", "a\0b", "\0", "\0a", "nan", "None")
    floats = (0.0, -0.0, 0.1, -1e-7, np.inf, -np.inf, np.nan)
    nan64 = np.array(
        [0x7FF8000000000001, 0xFFF8000000000002, 0x7FF0000000000001], dtype=np.uint64
    ).view(np.float64)
    nan32 = np.array([0x7FC00001, 0xFFC00002], dtype=np.uint32).view(np.float32)
    pools = (
        np.array([*floats, *nan64]),
        np.array([*floats, *nan32], dtype=np.float32),
        np.array([*floats, 2**-30], dtype=np.float16),
        np.array([*floats, 2**-60], dtype=np.longdouble),
        np.array([0, -1, 3, 2**53 + 1, -(2**63), 2**63 - 1], dtype=np.int64),
        np.array([0, 1, 2**64 - 1], dtype=np.uint64),
        np.array([False, True]),
        np.array(strings),
        np.array([b"", b"a", b"a\x00b", b"\xff"]),
        _object_cells(*strings),
        _object_cells(*strings, None, *floats),
        _object_cells(
            *strings,
            None,
            *floats,
            pd.NA,
            pd.NaT,
            [],
            [1],
            (),
            ("b", "a", "b", "", pd.NA),
            0,
            False,
            np.float32("nan"),
            np.float32(0.1),
            np.int64(4),
            np.bool_(False),
            b"",
            np.str_(""),
            np.datetime64("NaT", "ns"),
        ),
    )
    for pool in pools:
        yield pool
        for size in (0, 1, 200, *rng.integers(0, 201, size=24)):
            yield rng.choice(pool, size)
    yield np.array(floats, dtype=">f8")[::-1]
    yield np.resize(np.array([-0.0, 0.0]), 48)
    yield np.resize(np.array([0, 1, 2]), 48)


def test_observed_mask_and_labels_equal_the_per_cell_reference():
    shaped = (
        np.array(1.0),
        np.array(""),
        np.array(None, dtype=object),
        np.arange(12).reshape(3, 4).T,
    )
    for values in (*_samples(), *shaped):
        observed, labels = kinds.observed_mask(values), kinds.labels_of(values)
        assert isinstance(observed, np.ndarray) and isinstance(labels, np.ndarray)
        assert observed.shape == labels.shape == values.shape
        assert observed.dtype == np.dtype(bool)
        assert labels.dtype == np.dtype(object)
        assert all(type(label) is str for label in labels.flat)
        np.testing.assert_array_equal(observed, _observed(values))
        np.testing.assert_array_equal(labels, _labels(values))


def test_flag_codes_and_infer_kind_are_unchanged():
    for values in _samples():
        assert kinds.infer_kind(values) == _infer_kind(values)
        for cutoff in (-1, 0, len(values) // 2, len(values), len(values) + 1):
            codes = kinds.flag_codes(values, cutoff)
            assert codes.dtype == np.dtype(np.float64)
            assert np.array_equal(codes, _flag_codes(values, cutoff), equal_nan=True)


def test_screening_label_sets_equal_the_per_cell_reference():
    for values in _samples():
        actual = screening._label_sets(values)
        assert actual.shape == values.shape
        assert actual.dtype == np.dtype(object)
        np.testing.assert_array_equal(actual, _label_sets(values))


def _mixed_panel():
    items = []
    for i in range(8):
        t = np.arange(i % 3, 80)
        items.append(
            Item(
                f"item_{i}",
                i % 3,
                {"y": (3 + i + t % 9).astype(float), "z": (2 + t % 7).astype(float)},
                {
                    "number": np.sin(t / 5),
                    "flag": np.where(t % 11 == 0, np.nan, (t % 3 == 0).astype(float)),
                    "category": np.array(["a", "b", "c"], dtype=object)[(i + t) % 3],
                    "gaps": np.array(["d", None, "", "e", float("nan"), "f"], dtype=object)[t % 6],
                },
                {"tags": ["shared", f"tag{i % 2}"]},
            )
        )
    return Panel(
        "mixed",
        tuple(items),
        {"number": "past", "flag": "known", "category": "known", "gaps": "past"},
        "D",
        pd.Timestamp("2020-02-20"),
        True,
    )


def test_calendar_cache_returns_equal_arrays_and_cannot_be_corrupted(monkeypatch):
    panel = _mixed_panel()
    start, length = -2, 48
    expected = calendar_arrays(panel, start, length)
    calls = []

    def count_calendar(panel, start, length):
        calls.append((start, length))
        return calendar_arrays(panel, start, length)

    monkeypatch.setattr(builder, "calendar_arrays", count_calendar)
    episode_builder = EpisodeBuilder(panel)
    monkeypatch.setattr(episode_builder, "_CALENDAR_CACHE_SIZE", 2)
    first = episode_builder._calendar_arrays(start, length)
    second = episode_builder._calendar_arrays(start, length)
    for result in (first, second):
        for cached, uncached in zip(result, expected, strict=True):
            assert cached.dtype == uncached.dtype
            np.testing.assert_array_equal(cached, uncached)
    for array in first:
        try:
            array[0, 0] = 42
        except ValueError:
            with pytest.raises(ValueError):
                array.setflags(write=True)
    for cached, uncached in zip(
        episode_builder._calendar_arrays(start, length), expected, strict=True
    ):
        np.testing.assert_array_equal(cached, uncached)
    for other_start, other_length in ((start + 1, length), (start, length - 1)):
        actual = episode_builder._calendar_arrays(other_start, other_length)
        expected = calendar_arrays(panel, other_start, other_length)
        for cached, uncached in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(cached, uncached)
    assert calls == [(start, length), (start + 1, length), (start, length - 1)]
    episode_builder._calendar_arrays(start, length)
    assert len(calls) == 4 and calls[-1] == (start, length)


def test_episodes_are_byte_equal_before_and_after(monkeypatch):
    panel = _mixed_panel()

    def build_all():
        episode_builder = EpisodeBuilder(panel)
        plans = [
            episode_builder.plan_requests(i, (32, 48), 12, context=48, drop=drop)
            for i in range(len(panel.items))
            for drop in (frozenset(), frozenset({"value_stats"}))
        ]
        return plans, [episode_builder.build(request) for plan in plans for request in plan]

    plans, episodes = build_all()
    assert all(len(plan) == 2 for plan in plans)
    assert {"number", "flag", "category", "gaps"} <= set(episodes[0].row_names)
    assert {"category~effect", "gaps~effect"} <= set(episodes[2].row_names)
    assert {"tags=shared", "tags=tag0"} <= set(episodes[0].attr_names)

    for module in (kinds, builder, screening, value_stats):
        monkeypatch.setattr(module, "observed_mask", _observed)
        monkeypatch.setattr(module, "labels_of", _labels)
    monkeypatch.setattr(screening, "infer_kind", _infer_kind)
    monkeypatch.setattr(screening, "_label_sets", _label_sets)
    for module in (builder, screening):
        monkeypatch.setattr(module, "flag_codes", _flag_codes)
    monkeypatch.setattr(
        EpisodeBuilder,
        "_calendar_arrays",
        lambda self, start, length: calendar_arrays(self.panel, start, length),
    )
    slow_plans, slow_episodes = build_all()
    assert plans == slow_plans
    for actual, expected in zip(episodes, slow_episodes, strict=True):
        for field in fields(actual):
            left, right = getattr(actual, field.name), getattr(expected, field.name)
            if isinstance(left, np.ndarray):
                assert left.shape == right.shape and left.dtype == right.dtype, field.name
                assert np.array_equal(left, right, equal_nan=True), field.name
                assert left.tobytes() == right.tobytes(), field.name
            else:
                assert left == right, field.name
