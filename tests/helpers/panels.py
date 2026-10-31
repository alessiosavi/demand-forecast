"""Small deterministic panels shared by data-core tests, without bench extras."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from demand_forecast.config.settings import DatasetConfig
from demand_forecast.data.frequency import seasonal_period, to_offset
from demand_forecast.data.loader_long import load_long
from demand_forecast.data.panel import Panel, Roles, Series
from demand_forecast.synthetic.generator import generate_long_panel


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


def synthetic_panel(
    n_series: int, length: int, freq: str, seed: int, *, signal: str | None = None
) -> Panel:
    """Build uniform targets directly, or load a seasonal/promo panel through CSV.

    With ``seasonal_promo``, the promo effect is made dominant so a linear toy
    backbone can learn it (``promo_uplift=5.0``).
    """
    if signal == "seasonal_promo":
        long, static = generate_long_panel(
            n_series,
            length,
            freq,
            start="2023-01-01",
            known=("promo",),
            past=("oil",),
            statics=("cat",),
            seed=seed,
            promo_uplift=5.0,
        )
        with TemporaryDirectory(prefix="dfc-seasonal-promo-") as directory:
            path = Path(directory) / "long.csv"
            static_path = Path(directory) / "static.csv"
            long.to_csv(path, index=False)
            static.to_csv(static_path, index=False)
            return load_long(
                DatasetConfig(
                    path=path,
                    id_columns=["id"],
                    target="target",
                    target_support="nonnegative",
                    known_dynamic=["promo"],
                    past_dynamic=["oil"],
                    static_path=static_path,
                    static_categorical=["cat"],
                    freq=freq,
                )
            )
    if signal is not None:
        raise ValueError(f"Unknown synthetic signal: {signal!r}")
    rng = np.random.default_rng(seed)
    start = to_offset(freq).rollforward(pd.Timestamp("2023-01-01"))
    targets = rng.uniform(1, 20, size=(n_series, length)).astype(np.float32)
    series = [Series(str(i), start, length, targets[i], {}, {}) for i in range(n_series)]
    return Panel(series, Roles("y", (), (), (), ()), freq, seasonal_period(freq), "nonnegative")
