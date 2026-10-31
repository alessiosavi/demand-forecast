"""Adapt fev list-column windows to the shared data core."""

import pandas as pd

from demand_forecast.bench.specs import FevTaskSpec
from demand_forecast.data.frequency import seasonal_period, to_offset
from demand_forecast.data.panel import Panel, Roles, build_series


def panel_from_window(past_data, future_data, spec: FevTaskSpec, freq: str) -> Panel:
    """Build a Panel from the datasets returned by ``window.get_input_data()``.

    Preserve past row order, join future rows by item id, and align values by
    timestamp on the supplied task frequency. Only known covariates are read
    from future rows; each item's last past timestamp fixes its history end.
    Statics come from the past and follow the spec's categorical/numeric roles.

    Use ``missing_target="nan"`` because fev already represents missing values
    with NaN; the trainer applies the configured missing-target policy later.
    No optional bench imports are needed to consume these datasets.
    """
    if not isinstance(spec.target, str):
        raise ValueError("Panel requires a single target column")
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
        past_frame = pd.DataFrame(
            {name: row[name] for name in [roles.target, *roles.known_dynamic, *roles.past_dynamic]},
            index=pd.DatetimeIndex(row[spec.timestamp_column]),
            dtype=float,
        )
        if past_frame.empty:
            raise ValueError(f"Series {key!r}: cannot build from an empty history")
        future_frame = pd.DataFrame(
            {name: future[name] for name in roles.known_dynamic},
            index=pd.DatetimeIndex(future[spec.timestamp_column]),
            dtype=float,
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
    return Panel(series, roles, offset, seasonal_period(offset), spec.target_support)


def item_ids(panel: Panel) -> list[str]:
    """Return fev item ids in the panel's prediction order."""
    return [series.series_id for series in panel.series]
