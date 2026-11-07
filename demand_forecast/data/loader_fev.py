"""Adapt fev list-column windows to the shared data core."""

import json
from dataclasses import replace

import pandas as pd

from demand_forecast.bench.specs import FevTaskSpec
from demand_forecast.data.frequency import to_offset
from demand_forecast.data.panel import Panel, Roles, build_series


def _dynamic_values(row, name: str, spec: FevTaskSpec):
    """Encode declared events without learning a vocabulary from any window."""
    values = row[name]
    if name in spec.event_dynamics:
        no_event = set(spec.event_dynamics[name])
        return [0.0 if pd.isna(value) or str(value) in no_event else 1.0 for value in values]
    return values


def _float_frame(values, index, row, spec: FevTaskSpec):
    """Keep the normal float cast fast; diagnose dynamic columns only on failure."""
    try:
        return pd.DataFrame(values, index=index, dtype=float)
    except (ValueError, TypeError) as exc:
        for name in (*spec.known_dynamic, *spec.past_dynamic):
            if name not in values or name in spec.event_dynamics:
                continue
            try:
                pd.Series(values[name], dtype=float)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Task {spec.name!r}, series {row[spec.id_column]!r}: dynamic column "
                    f"{name!r} cannot be cast to float; "
                    "declare it under event_dynamics in the task profile"
                ) from exc
        raise


def panel_from_window(past_data, future_data, spec: FevTaskSpec, freq: str) -> Panel:
    """Build a Panel from the datasets returned by ``window.get_input_data()``.

    Preserve past row order, join future rows by item id, and align values by
    timestamp on the supplied task frequency. Only known covariates are read
    from future rows; each item's last past timestamp fixes its history end.
    Statics come from the past and follow the spec's categorical/numeric roles.
    Use the task's explicit seasonality, falling back to the frequency only
    when the spec omits it.

    Use ``missing_target="nan"`` because fev already represents missing values
    with NaN; the trainer applies the configured missing-target policy later.
    No optional bench imports are needed to consume these datasets.
    """
    if spec.event_dynamics:
        # Read nullable strings as Python cells rather than formatter-dependent
        # numpy strings, which can turn a null into literal text.
        if hasattr(past_data, "with_format"):
            past_data = past_data.with_format(None)
        if hasattr(future_data, "with_format"):
            future_data = future_data.with_format(None)
    if not isinstance(spec.target, str):
        if {"__fev_target__", "__fev_nonnegative__"}.intersection(spec.static_columns):
            raise ValueError("__fev_target__ and __fev_nonnegative__ are reserved")
        # Keep item order within each target; fev sorts target names itself.
        panels = []
        for target in spec.targets:
            support = (
                spec.target_support[target]
                if isinstance(spec.target_support, dict)
                else spec.target_support
            )
            single = replace(spec, target=target, target_support=support)
            columns = list(
                dict.fromkeys(
                    [
                        spec.id_column,
                        spec.timestamp_column,
                        target,
                        *spec.known_dynamic,
                        *spec.past_dynamic,
                        *spec.static_columns,
                    ]
                )
            )
            # Avoid decoding every other target for each univariate series.
            selected = (
                past_data.select_columns(columns)
                if hasattr(past_data, "select_columns")
                else [{column: row[column] for column in columns} for row in past_data]
            )
            panel = panel_from_window(selected, future_data, single, freq)
            for series in panel.series:
                series.series_id = target_series_id(series.series_id, target)
                series.statics["__fev_target__"] = target
            panels.append(panel)
        roles = replace(
            panels[0].roles,
            target="__fev_value__",
            static_categorical=(*panels[0].roles.static_categorical, "__fev_target__"),
        )
        return Panel(
            [series for panel in panels for series in panel.series],
            roles,
            panels[0].freq,
            panels[0].seasonal_period,
            spec.target_support,
        )
    roles = Roles(
        target=spec.target,
        known_dynamic=tuple(spec.known_dynamic),
        past_dynamic=tuple(spec.past_dynamic),
        static_categorical=tuple(spec.categorical_statics),
        static_numeric=tuple(
            name for name in spec.static_columns if name not in spec.categorical_statics
        ),
    )
    offset = to_offset(freq)
    future_by_id = {str(row[spec.id_column]): row for row in future_data}
    if len(future_by_id) != len(future_data):
        raise ValueError("duplicate future series id")

    series = []
    for row in past_data:
        key = str(row[spec.id_column])
        if "|" in key:
            raise ValueError("id values must not contain '|'")
        if key not in future_by_id:
            raise ValueError(f"Series {key!r}: missing future row or duplicate past id")
        future = future_by_id.pop(key)
        past_frame = _float_frame(
            {
                roles.target: row[roles.target],
                **{
                    name: _dynamic_values(row, name, spec)
                    for name in (*roles.known_dynamic, *roles.past_dynamic)
                },
            },
            index=pd.DatetimeIndex(row[spec.timestamp_column]),
            row=row,
            spec=spec,
        )
        if past_frame.empty:
            raise ValueError(f"Series {key!r}: cannot build from an empty history")
        future_frame = _float_frame(
            {name: _dynamic_values(future, name, spec) for name in roles.known_dynamic},
            index=pd.DatetimeIndex(future[spec.timestamp_column]),
            row=future,
            spec=spec,
        )
        statics = {}
        for name in roles.static_categorical:
            value = row.get(name)
            statics[name] = None if pd.isna(value) else str(value)
        for name in roles.static_numeric:
            value = row.get(name)
            statics[name] = float("nan") if pd.isna(value) else float(value)
        series.append(
            build_series(
                key,
                pd.concat([past_frame, future_frame]),
                freq=offset,
                roles=roles,
                missing_target="nan",
                statics=statics,
                history_end=past_frame.index[-1],
            )
        )
    if future_by_id:
        raise ValueError(f"Future rows without past series: {list(future_by_id)}")
    support = (
        spec.target_support[spec.target]
        if isinstance(spec.target_support, dict)
        else spec.target_support
    )
    return Panel(series, roles, offset, spec.effective_seasonal_period(offset), support)


def item_ids(panel: Panel) -> list[str]:
    """Return fev item ids in the panel's prediction order."""
    return [series.series_id for series in panel.series]


def target_series_id(item: str, target: str) -> str:
    """Collision-free item/target identity, independent of separator characters."""
    return json.dumps([str(item), target], ensure_ascii=True, separators=(",", ":"))
