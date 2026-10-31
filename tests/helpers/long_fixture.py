"""Small CSV panels for long-format loader tests."""

from itertools import cycle, islice
from pathlib import Path

import pandas as pd

from demand_forecast.config.settings import DatasetConfig


def long_cfg(
    tmp_path: Path,
    *,
    rows=None,
    statics=None,
    statics_duplicate: bool = False,
    promo_values=None,
    freq: str | None = None,
    **cfg_overrides,
) -> DatasetConfig:
    """Write a two-series weekly fixture, or supplied rows, and its configuration.

    Tuple rows contain product, store, timestamp, target and optionally promo.
    DataFrames or dict rows can supply additional columns. Promo values cycle
    to fill the frame. Static mappings use joined ids but CSVs use separate ids;
    static columns default to categorical unless explicitly declared numeric.
    """
    if rows is None:
        rows = [
            (product, "s1", timestamp, float(i + 1))
            for product in ("p1", "p2")
            for i, timestamp in enumerate(pd.date_range("2023-01-01", periods=6, freq="W-SUN"))
        ]
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
    elif rows and isinstance(rows[0], dict):
        frame = pd.DataFrame(rows)
    else:
        columns = ["product_id", "store_id", "timestamp", "y", "promo"]
        frame = pd.DataFrame(rows, columns=columns[: len(rows[0]) if rows else 4])
    if promo_values is not None or "promo" not in frame:
        values = [0, 1] if promo_values is None else list(promo_values)
        if not values:
            raise ValueError("promo_values must not be empty")
        frame["promo"] = list(islice(cycle(values), len(frame)))
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "long.csv"
    frame.to_csv(path, index=False)

    static_path = None
    static_columns = []
    if statics_duplicate and statics is None:
        statics = {"p1|s1": {"cat": "A"}, "p2|s1": {"cat": "B"}}
    if statics is not None:
        static_columns = list(dict.fromkeys(name for values in statics.values() for name in values))
        static_rows = [
            {"product_id": key.split("|")[0], "store_id": key.split("|")[1], **values}
            for key, values in statics.items()
        ]
        if statics_duplicate:
            static_rows.append(static_rows[0].copy())
        static_path = tmp_path / "static.csv"
        pd.DataFrame(
            static_rows, columns=list(dict.fromkeys(["product_id", "store_id", *static_columns]))
        ).to_csv(static_path, index=False)

    config = {
        "source": "long",
        "path": path,
        "id_columns": ["product_id", "store_id"],
        "timestamp_column": "timestamp",
        "target": "y",
        "target_support": "nonnegative",
        "known_dynamic": ["promo"],
        "static_path": static_path,
        "static_categorical": [
            name for name in static_columns if name not in cfg_overrides.get("static_numeric", [])
        ],
        "freq": freq,
    }
    config.update(cfg_overrides)
    return DatasetConfig(**config)
