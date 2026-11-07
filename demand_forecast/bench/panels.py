"""Task-local arrays with the same information boundaries as fev windows."""

from dataclasses import replace

import numpy as np
import pandas as pd

from demand_forecast.bench.specs import FevTaskSpec
from demand_forecast.data.frequency import (
    grid_range,
    steps_between,
    to_offset,
    validate_on_grid,
)
from demand_forecast.data.loader_fev import _dynamic_values, _float_frame, target_series_id
from demand_forecast.data.panel import Panel, Roles, Series, build_series


class TaskPanelCache:
    """Decode full histories once; expose only each window's permitted arrays.

    fev 0.10.0's Task.get_window shares load_full_dataset() across windows.
    Its past_future_split cuts by observation count (integer cutoff) or by
    timestamp <= cutoff, then filters items by context and horizon lengths.
    Use the actual past item ids for that filtering and positional ordering.
    Full targets and past covariates stay private to this task-local cache.
    """

    def __init__(self, full_data, spec: FevTaskSpec, freq: str) -> None:
        self.spec = spec
        self.timestamps = {}
        self.positions = {}
        offset = to_offset(freq)
        multiple = not isinstance(spec.target, str)
        if multiple and {"__fev_target__", "__fev_nonnegative__"}.intersection(spec.static_columns):
            raise ValueError("__fev_target__ and __fev_nonnegative__ are reserved")
        roles = Roles(
            target="__fev_value__" if multiple else spec.target,
            known_dynamic=tuple(spec.known_dynamic),
            past_dynamic=tuple(spec.past_dynamic),
            static_categorical=tuple(spec.categorical_statics),
            static_numeric=tuple(
                name for name in spec.static_columns if name not in spec.categorical_statics
            ),
        )
        # Match the loader's nullable event handling, including numeric precision.
        if spec.event_dynamics:
            full_data = full_data.with_format(None)
        columns = list(
            dict.fromkeys(
                [
                    spec.id_column,
                    spec.timestamp_column,
                    *spec.targets,
                    *spec.known_dynamic,
                    *spec.past_dynamic,
                    *spec.static_columns,
                ]
            )
        )
        by_target = {target: [] for target in spec.targets}
        for row in full_data.select_columns(columns):
            key = str(row[spec.id_column])
            if "|" in key:
                raise ValueError("id values must not contain '|'")
            index = pd.DatetimeIndex(row[spec.timestamp_column])
            self.timestamps[key] = index
            frame = _float_frame(
                {
                    **{target: row[target] for target in spec.targets},
                    **{
                        name: _dynamic_values(row, name, spec)
                        for name in (*roles.known_dynamic, *roles.past_dynamic)
                    },
                },
                index=index,
                row=row,
                spec=spec,
            )
            statics = {
                name: None if pd.isna(row.get(name)) else str(row[name])
                for name in roles.static_categorical
            }
            statics.update(
                {
                    name: float("nan") if pd.isna(row.get(name)) else float(row[name])
                    for name in roles.static_numeric
                }
            )
            validate_on_grid(index[:1], offset, key)
            # Regular fev histories need no Python work per timestamp. Comparing
            # the whole grid also catches a different frequency on another item.
            contiguous = index.equals(grid_range(index[0], len(index), offset))
            self.positions[key] = (
                None
                if contiguous
                else np.asarray([steps_between(index.min(), stamp, offset) for stamp in index])
            )
            dynamic = {
                name: frame[name].to_numpy(dtype=float)
                for name in (*roles.known_dynamic, *roles.past_dynamic)
            }
            for target in spec.targets:
                if contiguous:
                    series = Series(
                        key,
                        index[0],
                        len(index),
                        frame[target].to_numpy(dtype=float),
                        dynamic,
                        dict(statics),
                    )
                else:
                    # Retain the shared loader's gap filling and grid errors for
                    # histories that are regular on a coarser grid than the task.
                    series = build_series(
                        key,
                        frame,
                        freq=offset,
                        roles=replace(roles, target=target),
                        missing_target="nan",
                        statics=statics,
                        history_end=index[-1],
                    )
                if multiple:
                    series.series_id = target_series_id(key, target)
                    series.statics["__fev_target__"] = target
                by_target[target].append(series)
        if multiple:
            roles = replace(roles, static_categorical=(*roles.static_categorical, "__fev_target__"))
        support = spec.target_support
        if isinstance(support, dict) and not multiple:
            support = support[spec.target]
        self.panel = Panel(
            [series for target in spec.targets for series in by_target[target]],
            roles,
            offset,
            spec.effective_seasonal_period(offset),
            support,
        )

    def for_window(self, window, past) -> Panel:
        """Slice target/past at the cutoff and known covariates at horizon end."""
        spec = self.spec
        items = [str(row[spec.id_column]) for row in past.select_columns([spec.id_column])]
        bounds = {}
        for item in items:
            timestamps = self.timestamps[item]
            end = (
                int(timestamps.searchsorted(pd.Timestamp(window.cutoff), side="right"))
                if isinstance(window.cutoff, str)
                else window.cutoff
                if window.cutoff >= 0
                else len(timestamps) + window.cutoff
            )
            begin = (
                0 if window.max_context_length is None else max(0, end - window.max_context_length)
            )
            positions = self.positions[item]
            bounds[item] = (
                begin if positions is None else int(positions[begin]),
                end if positions is None else int(positions[end - 1]) + 1,
                end + window.horizon
                if positions is None
                else int(positions[end + window.horizon - 1]) + 1,
                timestamps[begin],
            )
        series = []
        for target in spec.targets:
            for item in items:
                key = item if isinstance(spec.target, str) else target_series_id(item, target)
                full = self.panel.series[self.panel.index_of(key)]
                begin, end, future_end, start = bounds[item]
                series.append(
                    Series(
                        key,
                        start,
                        end - begin,
                        full.target[begin:end],
                        {
                            name: values[begin : future_end if name in spec.known_dynamic else end]
                            for name, values in full.dynamic.items()
                        },
                        dict(full.statics),
                    )
                )
        return replace(self.panel, series=series)
