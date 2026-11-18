"""Cutoff-bounded admission of the generator's named information paths."""

from __future__ import annotations

import numpy as np

from demand_forecast.data.episode.kinds import is_missing
from demand_forecast.data.episode.screening import ScreeningConfig
from demand_forecast.synthetic.typed import GeneratedPanel, necessary_items
from demand_forecast.synthetic.typed.draft import INPUT_MECHANISMS, STATIC_MECHANISMS
from demand_forecast.synthetic.typed.recipe import MECHANISMS

_STATIC = ("attributes", "peers", "value_stats")
_COLUMN = ("features", "value_stats")
PROTECT: dict[str, tuple[str, ...]] = {
    m: tuple(
        dict.fromkeys(
            (
                *(_STATIC if m in STATIC_MECHANISMS else ()),
                *(
                    _COLUMN
                    if m in INPUT_MECHANISMS
                    and m not in STATIC_MECHANISMS
                    or m in ("event_response", "label_lag")
                    else ()
                ),
                *(("static_numbers",) if m == "static_number" else ()),
                *(("calendar",) if m == "calendar" else ()),
                *(("siblings",) if m == "multi_target" else ()),
                *(
                    ("rolling", "lags", "zeros")
                    if m in ("seasonal_lag", "intermittent", "level_shift")
                    else ()
                ),
            )
        )
    )
    for m in MECHANISMS
}
_MIN = ScreeningConfig()


def _slice(item, values, left, right):
    return values[max(0, left - item.start) : max(0, right - item.start)]


def _available(values):
    return ~np.frompyfunc(is_missing, 1, 1)(values).astype(bool)


def _target(item):
    return item.targets[sorted(item.targets)[0]]


def _static_support(generated, mechanism, item, cutoff):
    if item.item_id not in necessary_items(generated, mechanism):
        return False
    records = tuple(r for r in generated.inputs_of(mechanism) if r.where == "static")
    record = records[1] if mechanism == "hierarchy" else records[0]
    value = item.static.get(record.name)
    if is_missing(value):
        return False
    qualified = []
    for peer in generated.panel.items:
        other = peer.static.get(record.name)
        prefix = _slice(peer, _target(peer), peer.start, cutoff)
        if (
            peer.item_id != item.item_id
            and not is_missing(other)
            and (np.isfinite(prefix).sum() >= _MIN.min_steps)
        ):
            qualified.append(other)
    if mechanism == "static_number":
        return len(qualified) >= _MIN.min_items
    if mechanism == "multi_label":
        return any(bool(set(value) & set(other)) for other in qualified)
    return any(value == other for other in qualified)


def supported(
    generated: GeneratedPanel,
    mechanism: str,
    item_id: str,
    origin: int,
    *,
    context: int,
    horizon: int,
    cutoff: int,
) -> bool:
    """Use exposed availability and recorded truth, never inferred input kinds."""
    item = next((item for item in generated.panel.items if item.item_id == item_id), None)
    if item is None:
        raise ValueError(f"unknown item: {item_id!r}")
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism: {mechanism}")
    if mechanism in STATIC_MECHANISMS:
        if not _static_support(generated, mechanism, item, cutoff):
            return False
        if mechanism not in ("event_response", "label_lag"):
            return True
    records = tuple(r for r in generated.inputs_of(mechanism) if r.where == "dynamic")
    if records:
        record = records[0]
        column = item.dynamic[record.name]
        prefix = _slice(item, column, item.start, cutoff)
        target = _slice(item, _target(item), item.start, cutoff)
        if (_available(prefix) & np.isfinite(target)).sum() < _MIN.min_steps:
            return False
        history = _slice(item, column, origin - context, origin)
        future = _slice(item, column, origin, origin + horizon)
        if mechanism == "known_driver":
            return bool(_available(history).sum() >= _MIN.min_steps and _available(future).any())
        if mechanism in ("planned_event", "event_response"):
            truth = next(i for i in generated.complete.items if i.item_id == item_id)
            events = _slice(truth, truth.dynamic[record.name], origin, origin + horizon)
            return bool((_available(future) & (events == 1)).any())
        if mechanism.startswith("dynamic_category_"):
            right = origin if record.role == "past" else origin + horizon
            values = _slice(item, column, origin - context, right)
            present = _available(values)
            return any(
                a and b and x != y
                for a, b, x, y in zip(present[:-1], present[1:], values[:-1], values[1:])
            )
        if mechanism in ("delayed_past_driver", "label_lag"):
            lag = generated.params[mechanism]["lag"]
            if mechanism == "label_lag":
                truth = next(i for i in generated.complete.items if i.item_id == item_id)
                attribute = next(
                    r.name for r in generated.inputs_of(mechanism) if r.where == "static"
                )
                lag = lag[truth.static[attribute]]
            return bool(
                origin - item.start >= lag + _MIN.min_steps
                and _available(history).sum() >= _MIN.min_steps
            )
    if mechanism == "seasonal_lag":
        lag = generated.panel.long_lag
        return lag is not None and origin - item.start >= lag + _MIN.min_steps
    if mechanism == "multi_target":
        return len(item.targets) >= 2
    if mechanism == "missing_vs_zero":
        values = _slice(item, _target(item), origin - context, origin)
        return bool(np.isnan(values).any() and (values == 0).any())
    return mechanism in ("calendar", "intermittent", "level_shift")
