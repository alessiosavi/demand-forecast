"""Load declared long-format data onto the shared Panel frequency grid."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from demand_forecast.config.settings import DatasetConfig
from demand_forecast.data.frequency import (
    GridError,
    resolve_freq,
    seasonal_period,
    to_offset,
    validate_on_grid,
)
from demand_forecast.data.panel import Panel, Roles, build_series


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or Parquet, requiring pyarrow only for Parquet input."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow  # noqa: F401 -- check before entering pandas' optional backend.
        except ImportError as exc:
            raise ImportError("Reading parquet requires pyarrow; install pyarrow") from exc
        return pd.read_parquet(path, engine="pyarrow")
    raise ValueError(f"Unsupported table format: {path.suffix}; expected .csv or .parquet")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def series_key(frame: pd.DataFrame, id_columns: Sequence[str]) -> pd.Series:
    """Join non-null ids without permitting ambiguous embedded separators."""
    if not id_columns:
        raise ValueError("id_columns must contain at least one column")
    _require_columns(frame, id_columns)
    ids = frame[list(id_columns)]
    if ids.isna().any().any():
        raise ValueError("null series key")
    ids = ids.astype(str)
    for name in id_columns:
        if ids[name].str.contains("|", regex=False).any():
            raise ValueError("id values must not contain '|'")
    key = ids.iloc[:, 0].copy()
    for i in range(1, len(id_columns)):
        key = key + "|" + ids.iloc[:, i]
    return key


def _load_statics(cfg: DatasetConfig, roles: Roles) -> dict[str, dict[str, object]]:
    if cfg.static_path is None:
        return {}
    frame = read_table(cfg.static_path)
    keys = series_key(frame, cfg.id_columns)
    if keys.duplicated().any():
        raise ValueError("duplicate static key")
    names = [*roles.static_categorical, *roles.static_numeric]
    _require_columns(frame, names)
    # Keep id columns available as statics too (e.g. categorical store_id).
    frame = frame[names].copy()
    frame.index = keys
    return {
        key: {name: None if pd.isna(value) else value for name, value in values.items()}
        for key, values in frame.to_dict(orient="index").items()
    }


def load_long(cfg: DatasetConfig) -> Panel:
    """Aggregate explicitly, validate the panel grid and build ragged series.

    Duplicate covariates use the last input row, including missing values.
    Resampling uses target sums and covariate means; bins with no source rows
    remain absent so ``missing_target`` still applies when placing the grid.
    Only target or past-covariate observations extend the historical span;
    known covariates may continue beyond it.
    """
    if cfg.path is None:
        raise ValueError("data.path is required for long-format input")
    roles = Roles(
        cfg.target,
        tuple(cfg.known_dynamic),
        tuple(cfg.past_dynamic),
        tuple(cfg.static_categorical),
        tuple(cfg.static_numeric),
    )
    dynamic = [*roles.known_dynamic, *roles.past_dynamic]
    frame = read_table(cfg.path).copy()
    keys = series_key(frame, cfg.id_columns)
    _require_columns(frame, [cfg.timestamp_column, roles.target, *dynamic])
    for name in [roles.target, *dynamic]:
        if is_numeric_dtype(frame[name].dtype):
            continue
        if frame[name].notna().any():
            if name in dynamic:
                raise ValueError(
                    f"Dynamic column {name!r} must be numeric; use indicator encoding for categories"
                )
            raise ValueError(f"Target column {name!r} must be numeric")
        # An entirely missing column has no observed type; make it aggregatable
        # even when Parquet/pandas represents it as object or nullable string.
        frame[name] = float("nan")
    frame[cfg.timestamp_column] = pd.to_datetime(frame[cfg.timestamp_column], format="mixed")
    statics = _load_statics(cfg, roles)
    missing_statics = dict.fromkeys([*roles.static_categorical, *roles.static_numeric])

    frames = {}
    for key, group in frame.groupby(keys, sort=False):
        group = group.set_index(cfg.timestamp_column)[[roles.target, *dynamic]]
        group = group.sort_index(kind="stable")
        if group.index.hasnans:
            raise GridError(f"Series {key!r}: null timestamp", series_id=key, timestamp=pd.NaT)
        duplicate = group.index.duplicated()
        if duplicate.any():
            if cfg.duplicates == "error":
                raise ValueError(f"Series {key!r}: duplicate timestamp {group.index[duplicate][0]}")
            totals = group[roles.target].groupby(level=0).sum(min_count=1)
            group = group.loc[~group.index.duplicated(keep="last")].copy()
            group[roles.target] = totals
        if cfg.resample is not None:
            resampler = group.resample(to_offset(cfg.resample))
            group = resampler[[roles.target]].sum(min_count=1)
            if dynamic:
                group[dynamic] = resampler[dynamic].mean()
            group = group.loc[resampler.size() > 0]
        frames[key] = group

    freq = resolve_freq([group.index for group in frames.values()], cfg.freq)
    series = []
    for key, group in frames.items():
        validate_on_grid(group.index, freq, key)
        series.append(
            build_series(
                key,
                group,
                freq=freq,
                roles=roles,
                missing_target=cfg.missing_target,
                statics=statics.get(key, missing_statics),
            )
        )
    return Panel(series, roles, freq, seasonal_period(freq), cfg.target_support)
