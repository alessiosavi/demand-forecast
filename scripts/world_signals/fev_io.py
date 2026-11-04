"""Explicit Arrow list-column I/O, with a long pandas frame at both boundaries."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_fev_parquet(frame, roles, path):
    """Write sorted series; roles has static, known_dynamic and past_dynamic lists.

    IDs/statics are scalar strings, times are naive ns lists, and numeric lists
    retain IEEE NaNs (not Arrow nulls). Reject undeclared columns and varying
    statics instead of silently losing data when collapsing to one row per ID.
    """
    statics = list(roles.get("static", []))
    dynamics = ["target", *roles.get("known_dynamic", []), *roles.get("past_dynamic", [])]
    columns = ["id", "timestamp", *dynamics, *statics]
    if len(set(columns)) != len(columns) or set(frame.columns) != set(columns):
        raise ValueError("Columns must match unique declared roles plus id/timestamp/target")
    if frame.empty or frame[["id", *statics]].isna().any().any():
        raise ValueError("Nonempty data and nonmissing id/statics required")
    data = frame.copy()
    data["id"] = data.id.astype(str)
    data["timestamp"] = pd.to_datetime(data.timestamp)
    if data.timestamp.dt.tz is not None or data.timestamp.isna().any():
        raise ValueError("Timestamps must be nonmissing and timezone-naive")
    if not data.timestamp.eq(data.timestamp.dt.normalize()).all():
        raise ValueError("Timestamps must be midnight labels")
    if data.duplicated(["id", "timestamp"]).any():
        raise ValueError("Duplicate id/timestamp")
    data = data.sort_values(["id", "timestamp"])
    groups = list(data.groupby("id", sort=True))
    arrays = {"id": pa.array([asset for asset, _ in groups], type=pa.string())}
    for column in statics:
        if any(group[column].nunique(dropna=False) != 1 for _, group in groups):
            raise ValueError(f"Static column changes within an id: {column}")
        arrays[column] = pa.array([str(group[column].iloc[0]) for _, group in groups], pa.string())
    for _, group in groups:
        if not group.timestamp.diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError("Every series must use a consecutive daily grid")
    arrays["timestamp"] = pa.array(
        [group.timestamp.to_numpy(dtype="datetime64[ns]") for _, group in groups],
        type=pa.list_(pa.timestamp("ns")),
    )
    for column in dynamics:
        values = [group[column].to_numpy(dtype=np.float64, na_value=np.nan) for _, group in groups]
        if any(np.isinf(value).any() for value in values):
            raise ValueError(f"Infinite numeric values in {column}")
        arrays[column] = pa.array(values, type=pa.list_(pa.float64()), from_pandas=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), path)


def read_fev_parquet(path):
    """Expand list columns back to a long frame; scalar statics repeat by day."""
    table = pq.read_table(path)
    if table.schema.field("timestamp").type != pa.list_(pa.timestamp("ns")):
        raise ValueError("Expected timestamp list<timestamp[ns]> without timezone")
    lists = [field.name for field in table.schema if pa.types.is_list(field.type)]
    for field in table.schema:
        expected = pa.list_(pa.float64()) if field.name in lists else pa.string()
        if field.name != "timestamp" and field.type != expected:
            raise ValueError(f"Unexpected type for {field.name}: {field.type}")
    parts = []
    ids = table["id"].to_pylist()
    if ids != sorted(set(ids)):
        raise ValueError("IDs must be unique and sorted")
    for row in table.to_pylist():
        if len({len(row[name]) for name in lists}) != 1:
            raise ValueError("Unequal list lengths within an id")
        parts.append(pd.DataFrame(row))
    if not parts:
        raise ValueError("Empty panel")
    frame = pd.concat(parts, ignore_index=True)
    frame["timestamp"] = frame.timestamp.astype("datetime64[ns]")
    frame[[name for name in lists if name != "timestamp"]] = frame[
        [name for name in lists if name != "timestamp"]
    ].astype(np.float64)
    return frame
