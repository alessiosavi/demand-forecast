"""Convert one benchmark window into dated and unaligned episode panels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from demand_forecast.data.episode.kinds import infer_kind, infer_static_kind, is_missing, label_of
from demand_forecast.data.episode.schema import KIND_CATEGORY, MAX_HORIZON, PATCH, Item, Panel
from demand_forecast.data.frequency import (
    GridError,
    _requires_midnight,
    steps_between,
    to_offset,
)

if TYPE_CHECKING:
    from demand_forecast.bench.yardsticks.runner import WindowInput

REMOVABLE = frozenset(
    {
        "features",
        "known",
        "past",
        "dynamic_categories",
        "statics",
        "static_categories",
        "static_numbers",
    }
)
_OVER_ORIGIN_BUDGET = "over_origin_budget"
_MISSING_FREQUENCIES = frozenset({"missing_frequency", "invalid_frequency"})
ParsedFrequency = tuple[pd.offsets.BaseOffset | None, str | None]


@dataclass(frozen=True)
class TimeAxis:
    first: pd.Timestamp | None
    on_grid: bool
    reason: str | None
    missing_timestamps: bool = False


def _positive_offset(freq: str | pd.offsets.BaseOffset | None) -> ParsedFrequency:
    if freq is None:
        return None, "missing_frequency"
    try:
        offset = freq if isinstance(freq, pd.offsets.BaseOffset) else to_offset(freq)
    except (TypeError, ValueError):
        return None, "invalid_frequency"
    if offset.n <= 0:
        return None, "non_positive_frequency"
    return offset, None


def time_axis(
    history: np.ndarray,
    freq: str | pd.offsets.BaseOffset | None,
    *,
    future: np.ndarray | None = None,
    horizon: int | None = None,
) -> TimeAxis:
    """Check the history's own cadence and the exact continuation into the horizon."""
    return _prove_time_axis(history, _positive_offset(freq), future=future, horizon=horizon)


def _is_history_on_grid(dates: pd.DatetimeIndex, offset: pd.offsets.BaseOffset) -> bool:
    """Distinguish a gap or duplicate from an off-grid timestamp after a mismatch."""
    if isinstance(offset, (pd.offsets.Day, pd.offsets.Week)):
        local = dates.tz_localize(None)
        days = offset.n * (7 if isinstance(offset, pd.offsets.Week) else 1)
        return bool(((local - local[0]) % pd.Timedelta(days, "D") == pd.Timedelta(0)).all())
    if isinstance(offset, pd.offsets.Tick):
        step = pd.Timedelta(offset.nanos, "ns")
        return bool(((dates - dates[0]) % step == pd.Timedelta(0)).all())

    month_start = (pd.offsets.MonthBegin, pd.offsets.QuarterBegin, pd.offsets.YearBegin)
    month_end = (pd.offsets.MonthEnd, pd.offsets.QuarterEnd, pd.offsets.YearEnd)
    if isinstance(offset, month_start + month_end):
        anchor_day = 1 if isinstance(offset, month_start) else dates.days_in_month
        if isinstance(offset, (pd.offsets.YearBegin, pd.offsets.YearEnd)):
            months_per_step = 12
        elif isinstance(offset, (pd.offsets.QuarterBegin, pd.offsets.QuarterEnd)):
            months_per_step = 3
        else:
            months_per_step = 1
        months = dates.year * 12 + dates.month
        whole_steps = (months - months[0]) % (months_per_step * offset.n) == 0
        return bool(((dates.day == anchor_day) & whole_steps).all())

    # Other pandas calendars, such as business calendars, need a membership range.
    grid = pd.date_range(dates.min(), dates.max(), freq=offset)
    return bool(dates.isin(grid).all())


def _prove_time_axis(
    history: np.ndarray,
    parsed_frequency: ParsedFrequency,
    *,
    future: np.ndarray | None = None,
    horizon: int | None = None,
) -> TimeAxis:
    try:
        dates = pd.DatetimeIndex(history)
    except (TypeError, ValueError, OverflowError):
        return TimeAxis(None, False, "invalid_history_timestamps", missing_timestamps=True)
    # DatetimeIndex reports one dimension even if its backing array has more.
    if dates.array.ndim != 1:
        return TimeAxis(None, False, "invalid_history_timestamps", missing_timestamps=True)
    if not len(dates):
        return TimeAxis(None, False, "empty_history", missing_timestamps=True)
    first = None if pd.isna(dates[0]) else dates[0]
    offset, reason = parsed_frequency
    if offset is None:
        return TimeAxis(first, False, reason, missing_timestamps=dates.hasnans)

    future_dates = None
    future_reason = None
    missing_timestamps = dates.hasnans
    if future is not None:
        try:
            future_dates = pd.DatetimeIndex(future)
        except (TypeError, ValueError, OverflowError):
            future_dates = None
        if future_dates is None or future_dates.array.ndim != 1:
            future_reason = "invalid_future_timestamps"
            missing_timestamps = True
        else:
            missing_timestamps |= future_dates.hasnans
            if horizon is not None and len(future_dates) != horizon:
                future_reason = "future_length"

    if first is None or dates.hasnans or not offset.is_on_offset(first):
        return TimeAxis(first, False, "history_off_grid", missing_timestamps)
    if _requires_midnight(offset):
        local = dates.tz_localize(None) if dates.tz is not None else dates
        if not (local == local.normalize()).all():
            return TimeAxis(first, False, "history_off_grid", missing_timestamps)

    length = len(dates)
    future_steps = len(future_dates) if future_dates is not None and future_reason is None else 0
    try:
        expected = pd.date_range(first, periods=length + future_steps, freq=offset)
    except (ValueError, OverflowError):
        if not future_steps:
            return TimeAxis(first, False, "history_off_grid", missing_timestamps)
        # Preserve history error precedence if generating the future overflowed.
        try:
            expected = pd.date_range(first, periods=length, freq=offset)
        except (ValueError, OverflowError):
            return TimeAxis(first, False, "history_off_grid", missing_timestamps)
        future_reason = "future_off_grid"

    if not (dates == expected[:length]).all():
        try:
            reason = "history_gap" if _is_history_on_grid(dates, offset) else "history_off_grid"
        except (ValueError, OverflowError):
            reason = "history_off_grid"
        return TimeAxis(first, False, reason, missing_timestamps)
    if future_reason is None and future_dates is not None:
        if not (future_dates == expected[length:]).all():
            future_reason = "future_off_grid"
    return TimeAxis(first, future_reason is None, future_reason, missing_timestamps)


@dataclass(frozen=True)
class Part:
    panel: Panel
    positions: tuple[int, ...]
    origins: tuple[int, ...]
    alone: bool


@dataclass(frozen=True)
class WindowPanels:
    parts: tuple[Part, ...]
    reasons: dict[int, str]
    distinct_origins: int = 0
    mixed_kind_columns: int = 0
    items_without_time_axis: int = 0


def _cell(value: object, kind: str | None, labels: tuple[str, ...], name: str) -> object:
    if isinstance(value, np.generic) or isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_cell(element, kind, labels, name) for element in value)
    if kind == "class_label":
        if isinstance(value, str):
            if value in labels:
                return value
        elif is_missing(value):
            return None
        elif not isinstance(value, bool) and (
            isinstance(value, int) or isinstance(value, (float, np.floating)) and value.is_integer()
        ):
            # Nullable arrays can promote codes to floats. ClassLabel uses -1 for a missing label.
            code = int(value)
            if code == -1:
                return None
            if 0 <= code < len(labels):
                return labels[code]
        raise ValueError(f"Column {name!r}: invalid class label {value!r}")
    if is_missing(value):
        return None
    if kind == "string":
        return label_of(value)
    return value


def _dynamic_values(
    name: str,
    history: np.ndarray,
    future: np.ndarray | None,
    length: int,
    horizon: int,
    kind: str | None,
    labels: tuple[str, ...],
) -> np.ndarray:
    arrays = (history,) if future is None else (history, future)
    for values, size in zip(arrays, (length, horizon)):
        if values.ndim != 1 or (
            values.dtype.kind == "O"
            and any(isinstance(v, (list, tuple, np.ndarray)) for v in values)
        ):
            raise ValueError(f"Dynamic column {name!r}: expected scalar cells")
        if len(values) != size:
            raise ValueError(f"Dynamic column {name!r}: expected {size} values, got {len(values)}")

    numeric = kind in ("number", "boolean") or (
        kind is None
        and all(
            values.dtype.kind in "biuf"
            or (
                values.dtype.kind == "O"
                and all(is_missing(v) or isinstance(v, (Real, np.bool_)) for v in values)
            )
            for values in arrays
        )
    )
    result = np.full(
        length + horizon, np.nan if numeric else None, dtype=np.float64 if numeric else object
    )
    result_kind = kind if numeric or kind is not None else "string"
    for values, start in zip(arrays, (0, length)):
        cells = (
            values
            if numeric and values.dtype.kind in "biuf"
            else [_cell(value, result_kind, labels, name) for value in values]
        )
        result[start : start + len(values)] = cells
    return result


def _items(
    window: WindowInput, roles: dict[str, str], horizon: int
) -> tuple[list[Item], int, dict[str, set[int]]]:
    items = []
    kinds: dict[str, set[int]] = {}
    declared, labels = window.declared or {}, window.class_labels or {}
    for series in window.items:
        item_id = window.ids[series.item_index]
        if not series.target_indices:
            raise ValueError(f"Item {item_id!r} has no targets")
        names = [window.targets[index] for index in series.target_indices]
        if len(set(names)) != len(names):
            raise ValueError(f"Item {item_id!r} has duplicate target names")
        if series.target.ndim != 2 or len(series.target) != len(names):
            raise ValueError(f"Item {item_id!r}: target rows must match target indices")
        length = series.target.shape[-1]
        target = np.full((len(names), length + horizon), np.nan, dtype=np.float64)
        target[:, :length] = series.target
        targets = dict(zip(names, target, strict=True))

        missing_roles = (series.past.keys() | series.future.keys()) - roles.keys()
        if missing_roles:
            raise ValueError(f"Dynamic columns without a role: {sorted(missing_roles)!r}")
        if missing_history := series.future.keys() - series.past.keys():
            raise ValueError(f"Dynamic columns without history: {sorted(missing_history)!r}")
        dynamic = {}
        for name, history in series.past.items():
            future = series.future.get(name) if roles[name] == "known" else None
            if roles[name] == "known" and future is None:
                raise ValueError(f"Known dynamic column {name!r} has no future")
            values = _dynamic_values(
                name, history, future, length, horizon, declared.get(name), labels.get(name, ())
            )
            dynamic[name] = values
            kinds.setdefault(name, set()).add(infer_kind(values[:length]))
        static = {
            name: _cell(value, declared.get(name), labels.get(name, ()), name)
            for name, value in (series.static or {}).items()
        }
        items.append(Item(item_id, 0, targets, dynamic, static))
    return items, sum(len(observed) > 1 for observed in kinds.values()), kinds


def _remove(
    items: list[Item], roles: dict[str, str], kinds: dict[str, set[int]], without: frozenset[str]
) -> tuple[list[Item], dict[str, str]]:
    kept_roles = {
        name: role
        for name, role in roles.items()
        if "features" not in without
        and role not in without
        and not ("dynamic_categories" in without and KIND_CATEGORY in kinds[name])
    }
    statics = {name for item in items for name in item.static}
    if "statics" in without:
        statics.clear()
    elif without & {"static_numbers", "static_categories"}:
        statics = {
            name
            for name in statics
            if (
                "static_numbers"
                if infer_static_kind([item.static.get(name) for item in items]) == "number"
                else "static_categories"
            )
            not in without
        }
    return [
        replace(
            item,
            dynamic={name: value for name, value in item.dynamic.items() if name in kept_roles},
            static={name: value for name, value in item.static.items() if name in statics},
        )
        for item in items
    ], kept_roles


def window_panels(
    window: WindowInput,
    *,
    name: str,
    horizon: int,
    known: Sequence[str],
    past: Sequence[str],
    without: frozenset[str] = frozenset(),
    max_origins: int = 16,
) -> WindowPanels:
    """Group proven grid members, retaining runner positions and per-item origins."""
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be in [1, {MAX_HORIZON}]")
    if len(set(window.targets)) != len(window.targets):
        raise ValueError("Window has duplicate target names")
    if unknown := without - REMOVABLE:
        raise ValueError(f"Unknown removals: {sorted(unknown)!r}")

    roles = dict.fromkeys(past, "past") | dict.fromkeys(known, "known")
    items, mixed_kind_columns, kinds = _items(window, roles, horizon)
    roles = {column: roles[column] for column in kinds}
    if without:
        items, roles = _remove(items, roles, kinds, without)

    offset, freq_reason = window.parsed_frequency or _positive_offset(window.freq)
    dates = {}
    reasons = {}
    items_without_time_axis = 0
    for position, series in enumerate(window.items):
        axis = series.axis
        if axis is None:
            reasons[position] = "missing_axis"
        elif not axis.on_grid or axis.first is None:
            reasons[position] = axis.reason or "history_off_grid"
        elif offset is None:
            reasons[position] = freq_reason
        else:
            dates[position] = axis.first
            continue
        if (
            axis is None
            or axis.missing_timestamps
            or reasons[position] in _MISSING_FREQUENCIES
            or freq_reason in _MISSING_FREQUENCIES
        ):
            items_without_time_axis += 1

    epoch = min(dates.values(), default=None)
    starts = {}
    for position, first in dates.items():
        try:
            starts[position] = steps_between(epoch, first, offset)
        except GridError:
            reasons[position] = "off_common_grid"

    lengths = [series.target.shape[-1] for series in window.items]
    origins = {position: start + lengths[position] for position, start in starts.items()}
    distinct_origins = len(set(origins.values()))
    if distinct_origins > max_origins:
        reasons.update(dict.fromkeys(starts, _OVER_ORIGIN_BUDGET))
        starts.clear()

    parts = []
    if starts:
        panel = Panel(
            name,
            tuple(replace(items[position], start=start) for position, start in starts.items()),
            roles,
            window.freq,
            epoch,
            True,
        )
        parts.append(Part(panel, tuple(starts), tuple(origins.values()), False))
    for position, item in enumerate(items):
        if position in reasons:
            panel = Panel(
                name,
                (replace(item, static={}),),
                roles,
                window.freq if offset is not None else None,
                None,
                False,
            )
            parts.append(Part(panel, (position,), (lengths[position],), True))

    for part in parts:
        part.panel.validate()
    return WindowPanels(
        tuple(parts), reasons, distinct_origins, mixed_kind_columns, items_without_time_axis
    )


def describe(panels: Sequence[WindowPanels]) -> dict[str, int]:
    """Aggregate fallback counts and pre-fallback origin diagnostics.

    v5_items_alone counts every item forecast in its own panel, including origin
    budget fallbacks. v5_items_off_grid counts the items alone for other reasons,
    including unusable timestamps or frequency, a gap, a future that does not
    continue the history, or a mismatch with the common grid.
    v5_items_without_time_axis counts items with missing or unreadable timestamps
    or a frequency that is missing or does not parse.
    """
    return {
        # Every item forecast in a panel of its own, including origin-budget fallbacks.
        "v5_items_alone": sum(len(window.reasons) for window in panels),
        # Alone for any reason other than exceeding the origin budget.
        "v5_items_off_grid": sum(
            reason != _OVER_ORIGIN_BUDGET for window in panels for reason in window.reasons.values()
        ),
        # Missing or unreadable timestamps, or a missing or unparseable frequency.
        "v5_items_without_time_axis": sum(window.items_without_time_axis for window in panels),
        "v5_short_items": sum(
            origin - item.start < PATCH
            for window in panels
            for part in window.parts
            for origin, item in zip(part.origins, part.panel.items, strict=True)
        ),
        "v5_max_distinct_origins": max((window.distinct_origins for window in panels), default=0),
        "v5_windows_over_origin_budget": sum(
            _OVER_ORIGIN_BUDGET in window.reasons.values() for window in panels
        ),
        "v5_mixed_kind_columns": sum(window.mixed_kind_columns for window in panels),
    }
