"""Corpus schema, synthetic time grids, and portable shard identities."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.bench

from demand_forecast.data.panel import Roles  # noqa: E402
from demand_forecast.pretrain.corpus import (  # noqa: E402
    corpus_manifest,
    read_manifest,
    read_shard,
    shard_panel,
    write_manifest,
)
from tests.helpers.corpus_fixture import write_corpus_fixture  # noqa: E402


def test_read_shard_builds_series_with_dummy_weekly_grid(tmp_path):
    import pyarrow.parquet as pq

    path = write_corpus_fixture(tmp_path)
    rows = pq.read_table(path).to_pylist()
    series = read_shard(path)
    panel = shard_panel(series)

    assert len(panel) == 12
    assert panel.roles == Roles("target", (), (), (), ())
    assert panel.freq == "W-SUN"
    assert panel.seasonal_period == 1
    assert panel.target_support == "real"
    assert [item.series_id for item in series] == [row["id"] for row in rows]
    for i, (item, row) in enumerate(zip(series, rows, strict=True)):
        assert item.start == pd.Timestamp("2000-01-02")
        assert item.start != row["timestamp"][0]
        assert item.length == item.history_length == len(row["target"])
        assert item.target.dtype == np.float32
        np.testing.assert_array_equal(item.target, np.array(row["target"], dtype=np.float32))
        assert item.dynamic == item.statics == {}
        pd.testing.assert_index_equal(
            panel.timestamps(i), pd.date_range("2000-01-02", periods=item.length, freq="W-SUN")
        )


@pytest.mark.parametrize("constant,all_nan", [(True, False), (False, True), (True, True)])
def test_read_shard_preserves_constant_and_missing_histories(tmp_path, constant, all_nan):
    path = write_corpus_fixture(tmp_path, lengths=(70, 70), constant=constant, all_nan=all_nan)
    for item in read_shard(path, dtype=np.float64):
        assert item.history_length == 70
        assert item.target.dtype == np.float64
        np.testing.assert_array_equal(item.target, np.full(70, np.nan if all_nan else 1.0))


def test_read_shard_preserves_signed_values_and_internal_and_trailing_nans(tmp_path):
    import pyarrow.parquet as pq

    path = write_corpus_fixture(tmp_path, n_series=1, lengths=(4, 4))
    table = pq.read_table(path)
    target = pyarrow.array([[-2.5, np.nan, 0.25, np.nan]], type=pyarrow.list_(pyarrow.float64()))
    pq.write_table(table.set_column(2, "target", target), path)

    item = read_shard(path)[0]
    assert item.history_length == 4
    np.testing.assert_array_equal(item.target, [-2.5, np.nan, 0.25, np.nan])


def test_manifest_round_trip_is_root_relative(tmp_path, monkeypatch):
    root = tmp_path / "original"
    paths = [
        write_corpus_fixture(root, kind="tsmixup", index=7),
        write_corpus_fixture(root, kind="tsmixup", index=2),
        write_corpus_fixture(root, kind="kernel_synth", index=3),
    ]
    # Other files and similarly named shards outside the corpus are not inputs.
    (root / "train-00000-of-00200.parquet").write_bytes(b"unrelated")
    (paths[0].parent / "train-00000-of-00020.parquet").write_bytes(b"wrong shard count")
    (paths[0].parent / "train-backup-of-00200.parquet").write_bytes(b"not an index")

    monkeypatch.chdir(tmp_path)
    refs = corpus_manifest(Path("original"))
    assert [ref.path for ref in refs] == sorted(paths)
    assert [(ref.kind, ref.index) for ref in refs] == [
        ("kernel_synth", 3),
        ("tsmixup", 2),
        ("tsmixup", 7),
    ]
    for ref in refs:
        assert ref.sha256 == hashlib.sha256(ref.path.read_bytes()).hexdigest()
        assert ref.size == ref.path.stat().st_size

    manifest = root / "manifest.json"
    write_manifest(refs, manifest)
    payload = json.loads(manifest.read_text())
    assert {entry["path"] for entry in payload} == {
        "training_corpus/tsmixup_10m/train-00007-of-00200.parquet",
        "training_corpus/tsmixup_10m/train-00002-of-00200.parquet",
        "training_corpus/kernel_synth_1m/train-00003-of-00020.parquet",
    }
    assert all(not Path(entry["path"]).is_absolute() for entry in payload)
    assert str(root) not in manifest.read_text()
    assert read_manifest(manifest) == refs

    relocated = tmp_path / "ec2"
    shutil.move(root, relocated)
    loaded = read_manifest(relocated / "manifest.json")
    assert loaded == corpus_manifest(relocated)
    assert all(ref.path.is_file() for ref in loaded)
    assert [ref.sha256 for ref in loaded] == [ref.sha256 for ref in refs]


def test_empty_corpus_manifest_round_trips(tmp_path):
    assert corpus_manifest(tmp_path) == []
    manifest = tmp_path / "manifest.json"
    write_manifest([], manifest)
    assert read_manifest(manifest) == []


@pytest.mark.parametrize("stored_path", ["/tmp/shard.parquet", "../shard.parquet"])
def test_manifest_rejects_paths_outside_root(tmp_path, stored_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [{"path": stored_path, "kind": "tsmixup", "index": 0, "sha256": "a" * 64, "size": 1}]
        )
    )
    with pytest.raises(ValueError, match="root-relative"):
        read_manifest(manifest)


@pytest.mark.parametrize("kind,count", [("tsmixup", 200), ("kernel_synth", 20)])
def test_fixture_schema_matches_corpus(tmp_path, kind, count):
    import pyarrow.parquet as pq

    path = write_corpus_fixture(tmp_path, kind=kind, index=3)
    table = pq.read_table(path)
    assert path.name == f"train-00003-of-{count:05d}.parquet"
    assert table.schema == pyarrow.schema(
        [
            ("id", pyarrow.string()),
            ("timestamp", pyarrow.list_(pyarrow.timestamp("ms"))),
            ("target", pyarrow.list_(pyarrow.float64())),
        ]
    )
    assert table.num_rows == 12
    assert len(set(table["id"].to_pylist())) == 12
    lengths = []
    for row in table.to_pylist():
        assert 70 <= len(row["target"]) <= 90
        assert len(row["timestamp"]) == len(row["target"])
        assert set(row["timestamp"]) == {pd.Timestamp("1970-01-01")}
        lengths.append(len(row["target"]))
    assert len(set(lengths)) > 1
    same = write_corpus_fixture(tmp_path / "same", kind=kind, index=3)
    different = write_corpus_fixture(tmp_path / "different", kind=kind, index=3, seed=1)
    assert table.equals(pq.read_table(same))
    assert not table.equals(pq.read_table(different))
