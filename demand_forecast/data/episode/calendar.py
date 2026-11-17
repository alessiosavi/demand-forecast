"""Canonical calendar phases and fixed-vocabulary category ids for v5."""

import numpy as np

from demand_forecast.data.frequency import (
    CANONICAL_CALENDAR_SCHEMA,
    calendar_feature_mask,
    calendar_features,
)

from .schema import CALENDAR_FIELDS, CALENDAR_SIZES, Panel


def calendar_arrays(panel: Panel, start: int, length: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Encode dated grid steps; inactive category fields carry the sentinel -1."""
    index = panel.timestamps(start, length)
    if index is None:
        return None
    phases = calendar_features(index, panel.freq, schema="canonical")
    active = calendar_feature_mask(panel.freq, "canonical")
    fields = (
        index.hour,
        index.day_of_week,
        index.day - 1,
        index.month - 1,
        index.isocalendar().week - 1,
        index.quarter - 1,
    )
    ids = np.full((length, len(CALENDAR_FIELDS)), -1, dtype=np.int16)
    for column, (field, size, values) in enumerate(zip(CALENDAR_FIELDS, CALENDAR_SIZES, fields)):
        if active[CANONICAL_CALENDAR_SCHEMA.index(f"{field}_sin")]:
            values = values.to_numpy(dtype=np.int16)
            if not ((values >= 0) & (values < size)).all():
                raise ValueError(f"{field} ids outside [0, {size})")
            ids[:, column] = values
    return phases, ids
