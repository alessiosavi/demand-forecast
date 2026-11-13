"""Offline allowlist, integrity and native-schema contracts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from scripts.corpus_v4.fetch_native import HubFile, fetch, measure_folder  # noqa: E402
from scripts.corpus_v4.freeze_manifest import REPO, REVISION, freeze  # noqa: E402


def native(path, *, cov=False, length=64, freq="h", item="a@0", target=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(length, dtype=float).tolist() if target is None else target
    row = dict(item_id=item, source="fixture", freq=freq, target=x if cov else [x])
    if cov:
        row["covariates"] = [x, x]
    pq.write_table(pa.Table.from_pylist([row]), path)
    return path


def manifest(folder="univariate/demo", status="allowed"):
    return dict(
        repo_id=REPO,
        revision=REVISION,
        folders=[
            dict(
                folder=folder,
                status=status,
                rule_f="no_identified_overlap",
                rule_l="no_identified_overlap",
                domain="unknown",
                lineage="fixture",
            )
        ],
    )


class FakeHub:
    def __init__(self, path, folder="univariate/demo", *, sha=None, revision=REVISION):
        self.path, self.folder, self.revision = path, folder, revision
        self.sha = sha or hashlib.sha256(path.read_bytes()).hexdigest()
        self.calls = 0
        self.interrupt = False

    def inventory(self, repo_id, revision):
        assert repo_id == REPO and revision == REVISION
        return self.revision, [
            HubFile(
                f"{self.folder}/train-000.parquet", self.path.stat().st_size, self.sha, "sha256"
            )
        ]

    def download(self, repo_id, revision, file, destination):
        self.calls += 1
        destination.write_bytes(self.path.read_bytes())
        if self.interrupt:
            raise InterruptedError("spot interruption")


def test_frozen_counts_and_rules():
    data = yaml.safe_load(Path("configs/corpus-v4.yaml").read_text())
    assert len(data["folders"]) == 121
    assert sum(x["status"] == "allowed" for x in data["folders"]) == 97
    assert sum(x["status"] == "quarantined" for x in data["folders"]) == 24
    assert data["revision"] == REVISION
    assert data["folders"] == sorted(data["folders"], key=lambda x: x["folder"])
    assert all(
        x["rule_f"] == x["rule_l"] == "no_identified_overlap"
        for x in data["folders"]
        if x["status"] == "allowed"
    )


def test_freeze_deterministic(tmp_path):
    entry = dict(
        repo_id=REPO,
        revision=REVISION,
        config="univariate/a",
        rule_f=dict(status="no_identified_overlap"),
        rule_l=dict(status="no_identified_overlap"),
        licence={},
        structure={},
        lineage=dict(original_dataset="a"),
        reason="fixture",
    )
    assert freeze(dict(sources=[entry])) == freeze(dict(sources=[entry]))
    entry["revision"] = "other"
    with pytest.raises(ValueError, match="revision"):
        freeze(dict(sources=[entry]))


def test_fetch_recheck_and_measure(tmp_path):
    hub = FakeHub(native(tmp_path / "remote.parquet"))
    root = tmp_path / "root"
    fetch(manifest(), root, hub, state=tmp_path / "state")
    fetch(manifest(), root, hub, state=tmp_path / "state")
    assert hub.calls == 1
    stats = json.loads((root / "univariate/demo/stats.json").read_text())
    assert stats["rows"] == 1 and stats["observed_target_values"] == 64
    assert stats["first_shard_representative"]["representative"]
    (root / "univariate/demo/train-000.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="(hash|size)"):
        fetch(manifest(), root, hub, state=tmp_path / "state")


@pytest.mark.parametrize("fault", ["folder", "revision", "hash", "rule"])
def test_fetch_refuses_unknown_revision_hash_or_rule(tmp_path, fault):
    hub = FakeHub(native(tmp_path / "remote.parquet"))
    data = manifest()
    if fault == "revision":
        hub.revision = "other"
    if fault == "folder":
        hub.folder = "univariate/not-allowed"
    if fault == "hash":
        hub.sha = "0" * 64
    if fault == "rule":
        data["folders"][0]["rule_l"] = "unresolved"
    with pytest.raises(ValueError):
        fetch(data, tmp_path / "root", hub, state=tmp_path / "state")


def test_quarantine_interrupted_restart_never_trainable(tmp_path):
    folder = "covariates/demo"
    hub = FakeHub(native(tmp_path / "remote.parquet", cov=True), folder)
    root = tmp_path / "root"
    hub.interrupt = True
    with pytest.raises(InterruptedError):
        fetch(manifest(folder, "quarantined"), root, hub, state=tmp_path / "state")
    assert not (root / "covariates").exists()
    hub.interrupt = False
    fetch(manifest(folder, "quarantined"), root, hub, state=tmp_path / "state")
    assert (root / "quarantine" / folder / "train-000.parquet").exists()
    assert not (root / "covariates").exists()
    fetch(manifest(folder, "quarantined"), root, hub, state=tmp_path / "state")
    assert not (root / "covariates").exists()


def test_stats_covariates_long_rows_and_nonrepresentative_first(tmp_path):
    native(tmp_path / "train-000.parquet", cov=True, length=64)
    native(tmp_path / "train-001.parquet", cov=True, length=16384, freq="D")
    stats = measure_folder(tmp_path, "covariates")
    assert stats["rows_ge_16384"] == 1
    assert stats["covariate_channel_counts"] == {"2": 2}
    assert not stats["first_shard_representative"]["representative"]


def test_ragged_schema_refused(tmp_path):
    path = native(tmp_path / "train-000.parquet")
    row = pq.read_table(path).to_pylist()[0]
    row["target"].append([1.0, 2.0])
    pq.write_table(pa.Table.from_pylist([row]), path)
    with pytest.raises(ValueError, match="(aligned|ragged|shape)"):
        measure_folder(tmp_path, "univariate")


def test_zero_width_covariate_channels_fail_alignment(tmp_path):
    path = native(tmp_path / "train-000.parquet", cov=True)
    table = pq.read_table(path)
    row = table.to_pylist()[0]
    row["covariates"] = [[], []]
    pq.write_table(pa.Table.from_pylist([row], schema=table.schema), path)
    with pytest.raises(ValueError, match="covariates must be aligned"):
        measure_folder(tmp_path, "covariates")


def test_empty_covariate_list_means_zero_channels(tmp_path):
    path = native(tmp_path / "train-000.parquet", cov=True)
    table = pq.read_table(path)
    row = table.to_pylist()[0]
    row["covariates"] = []
    pq.write_table(pa.Table.from_pylist([row], schema=table.schema), path)
    stats = measure_folder(tmp_path, "covariates")
    assert stats["covariate_channel_counts"] == {"0": 1}
    assert stats["covariate_values"] == 0
    assert stats["observed_target_values"] == 64


def test_target_string_schema_refused(tmp_path):
    path = native(tmp_path / "train-000.parquet")
    row = pq.read_table(path).to_pylist()[0]
    row["target"] = [["1", "2", "3"]]
    pq.write_table(pa.Table.from_pylist([row]), path)
    with pytest.raises(ValueError, match="numeric"):
        measure_folder(tmp_path, "univariate")


def test_quarantine_symlink_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "quarantine").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    hub = FakeHub(native(tmp_path / "remote.parquet"))
    with pytest.raises(ValueError, match="symlink"):
        fetch(manifest(status="quarantined"), root, hub, state=tmp_path / "state")


@pytest.mark.parametrize("lost", ["receipt.json", "train-000.parquet", "stats.json"])
def test_partial_state_restore_recovers_quarantine(tmp_path, lost):
    folder = "covariates/demo"
    hub = FakeHub(native(tmp_path / "remote.parquet", cov=True), folder)
    data = manifest(folder, "quarantined")
    root, state = tmp_path / "root", tmp_path / "state"
    fetch(data, root, hub, state=state)
    (root / "quarantine" / folder / lost).unlink()
    fetch(data, root, hub, state=state)
    assert (root / "quarantine" / folder / "receipt.json").exists()
    assert (root / "quarantine" / folder / "train-000.parquet").exists()
    assert not (root / "covariates").exists()


def test_empty_first_training_shard_is_reported_unrepresentative(tmp_path):
    path = native(tmp_path / "train-001.parquet")
    pq.write_table(pq.read_table(path).slice(0, 0), tmp_path / "train-000.parquet")
    stats = measure_folder(tmp_path, "univariate")
    assert not stats["first_shard_representative"]["representative"]


def test_native_values_must_survive_trainer_float32(tmp_path):
    native(tmp_path / "train-000.parquet", target=[1e40] * 64)
    with pytest.raises(ValueError, match="float32"):
        measure_folder(tmp_path, "univariate")
