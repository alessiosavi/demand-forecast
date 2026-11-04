import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.world_signals.fev_io import read_fev_parquet, write_fev_parquet


def test_fev_parquet_roundtrip_preserves_lists_and_nans(tmp_path):
    frame = pd.DataFrame(
        {
            "id": ["GOLD", "GOLD", "BTC", "BTC", "BTC"],
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "target": [np.nan, 0.2, np.nan, -0.1, 0.3],
            "asset_class": ["metals", "metals", "crypto", "crypto", "crypto"],
            "known": [0, 1, 1, 0, 1],
            "past": [np.nan, np.nan, 3, np.nan, 0],
        }
    )
    roles = {"static": ["asset_class"], "known_dynamic": ["known"], "past_dynamic": ["past"]}
    path = tmp_path / "panel.parquet"
    write_fev_parquet(frame.sample(frac=1, random_state=42), roles, path)
    table = pq.read_table(path)
    assert table["id"].to_pylist() == ["BTC", "GOLD"]
    assert table.schema.field("id").type == pa.string()
    assert table.schema.field("asset_class").type == pa.string()
    assert table.schema.field("timestamp").type == pa.list_(pa.timestamp("ns"))
    for name in ("target", "known", "past"):
        assert table.schema.field(name).type == pa.list_(pa.float64())
        assert table[name].combine_chunks().values.null_count == 0  # IEEE NaN, not Arrow null.
    for row in table.to_pylist():
        assert len({len(row[name]) for name in ("timestamp", "target", "known", "past")}) == 1
    restored = read_fev_parquet(path)
    expected = frame.sort_values(["id", "timestamp"]).reset_index(drop=True)
    expected[["target", "known", "past"]] = expected[["target", "known", "past"]].astype(float)
    pd.testing.assert_frame_equal(restored[expected.columns], expected)


@pytest.mark.parametrize("problem", ["duplicate", "static_changes", "timezone", "undeclared"])
def test_write_rejects_invalid_panel(tmp_path, problem):
    frame = pd.DataFrame(
        {
            "id": ["BTC"] * 3,
            "timestamp": pd.date_range("2024-01-01", periods=3),
            "target": [np.nan, 0.1, 0.2],
            "asset_class": ["crypto"] * 3,
        }
    )
    if problem == "duplicate":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    elif problem == "static_changes":
        frame.loc[1, "asset_class"] = "metals"
    elif problem == "timezone":
        frame["timestamp"] = frame.timestamp.dt.tz_localize("UTC")
    else:
        frame["oops"] = 1.0
    with pytest.raises(ValueError):
        write_fev_parquet(frame, {"static": ["asset_class"]}, tmp_path / "bad.parquet")
