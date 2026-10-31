"""Small deterministic panels shared by data-core tests, without bench extras."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from demand_forecast.data.panel import Panel, Roles, Series


def weekly_panel(lengths: Sequence[int]) -> Panel:
    """Build weekly histories with seasonal demand, promo, oil and two statics."""
    roles = Roles("y", ("promo",), ("oil",), ("cat",), ("size",))
    series = []
    for i, length in enumerate(lengths):
        steps = np.arange(length, dtype=np.float32)
        promo = ((steps + i) % 4 == 0).astype(np.float32)
        target = (10 + i + 3 * np.sin(2 * np.pi * steps / 13) + 5 * promo).astype(np.float32)
        series.append(
            Series(
                series_id=chr(ord("a") + i),
                start=pd.Timestamp("2023-01-01"),
                history_length=length,
                target=target,
                dynamic={"promo": promo, "oil": (50 + i + 0.1 * steps).astype(np.float32)},
                statics={"cat": "A" if i % 2 == 0 else "B", "size": float(10 * (i + 1))},
            )
        )
    return Panel(series, roles, freq="W-SUN", seasonal_period=1, target_support="nonnegative")


def series_with(panel: Panel, **overrides) -> Series:
    """Copy the first series with a new id and/or static values, leaving it intact."""
    source = panel.series[0]
    series_id = overrides.pop("series_id", source.series_id)
    return replace(source, series_id=series_id, statics={**source.statics, **overrides})
