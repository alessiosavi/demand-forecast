"""Pinned release metadata only; all file payloads used by these tests are synthetic."""

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.corpus_v4.audit_fev_overlap import BenchmarkSeries, run_audit
from scripts.corpus_v4.fetch_native import PublicHub, fetch, inventory
from scripts.corpus_v4.freeze_manifest import REPO, REVISION, read_manifest
from scripts.corpus_v4.native import digest
from tests.test_corpus_v4.test_manifest_fetch import manifest, native

FIXTURE = Path(__file__).with_name("fixtures") / "simons-5f17ab8-tree.json"


def recorded_hub(monkeypatch, siblings):
    """Exercise the production DatasetInfo parser without contacting the Hub."""
    from huggingface_hub import DatasetInfo, HfApi

    def dataset_info(self, repo_id, *, revision, files_metadata):
        assert repo_id == REPO and revision == REVISION and files_metadata
        assert self.token is False
        return DatasetInfo(id=REPO, sha=REVISION, siblings=siblings)

    monkeypatch.setattr(HfApi, "dataset_info", dataset_info)
    return PublicHub()


def test_pinned_release_inventory_matches_every_manifest_folder(monkeypatch):
    siblings = json.loads(FIXTURE.read_text())
    data = read_manifest()
    folders = inventory(data, recorded_hub(monkeypatch, siblings))
    assert len(siblings) == 172
    assert set(folders) == {entry["folder"] for entry in data["folders"]}
    assert len(folders) == 121
    assert Counter(entry["status"] for entry in data["folders"]) == {
        "allowed": 97,
        "quarantined": 24,
    }
    assert Counter(folder.split("/")[0] for folder in folders) == {
        "univariate": 62,
        "covariates": 59,
    }
    files = [file for group in folders.values() for file in group]
    assert Counter(Path(file.path).suffix for file in files) == {
        ".parquet": 150,
        ".md": 10,
        ".csv": 3,
    }
    assert Counter(file.algorithm for file in files) == {"sha256": 150, "git-sha1": 13}
    assert all(
        file.size > 0 and len(file.hash) == (64 if file.algorithm == "sha256" else 40)
        for file in files
    )
    # A dataset can span many shards; inventory must retain every one.
    assert Counter(
        sum(file.path.endswith(".parquet") for file in group) for group in folders.values()
    ) == {1: 115, 2: 2, 3: 1, 4: 1, 11: 1, 13: 1}
    licence = next(
        file for file in files if file.path == "covariates/bcb_focus_inflation/LICENSE.md"
    )
    assert (licence.size, licence.hash) == (457, "32e0db25ef6d741312f7ff305395689ab9dd33e3")


def test_inventory_reports_all_unexpected_paths_before_transfer(monkeypatch, tmp_path):
    paths = [
        "univariate/demo/unknown.csv",
        "covariates/demo/LICENSE.txt",
        "univariate/demo/nested/LICENSE.md",
        "univariate/demo/../LICENSE.md",
        "univariate/demo/test-00000.parquet",
        "univariate/unlisted/train-00000.parquet",
    ]
    siblings = [dict(rfilename=path, size=1, blobId="a" * 40) for path in paths]
    hub = recorded_hub(monkeypatch, siblings)
    with pytest.raises(ValueError) as error:
        fetch(manifest(), tmp_path / "root", hub, state=tmp_path / "state")
    assert all(path in str(error.value) for path in paths)
    assert not (tmp_path / "root").exists()


def test_side_files_alone_cannot_satisfy_missing_training_data(monkeypatch):
    hub = recorded_hub(
        monkeypatch,
        [
            dict(
                rfilename="univariate/demo/LICENSE.md",
                size=1,
                blobId="a" * 40,
            )
        ],
    )
    with pytest.raises(ValueError, match="(missing|train)"):
        inventory(manifest(), hub)


@pytest.mark.parametrize("status", ["allowed", "quarantined"])
def test_fetch_side_files_receipt_restart_and_measurement(tmp_path, monkeypatch, status):
    folder = "covariates/demo"
    remote = tmp_path / "remote"
    native(remote / "train-00000.parquet", cov=True, length=64)
    (remote / "LICENSE.md").write_text("Synthetic licence record\n")
    (remote / "articles.csv").write_text("item_id,article\na,synthetic attribution\n")
    siblings = [
        dict(
            rfilename=f"{folder}/{path.name}",
            size=path.stat().st_size,
            blobId=digest(path, "git-sha1"),
        )
        for path in sorted(remote.iterdir())
    ]
    hub = recorded_hub(monkeypatch, siblings)

    def download(repo_id, revision, file, destination):
        assert repo_id == REPO and revision == REVISION
        destination.write_bytes((remote / Path(file.path).name).read_bytes())

    monkeypatch.setattr(hub, "download", download)
    root, state = tmp_path / "corpus", tmp_path / "state"
    data = manifest(folder, status)
    fetch(data, root, hub, state=state)
    local = root / ("quarantine" if status == "quarantined" else "") / folder
    receipt = json.loads((local / "receipt.json").read_text())
    assert {Path(file["path"]).name for file in receipt["files"]} == {
        "train-00000.parquet",
        "LICENSE.md",
        "articles.csv",
    }
    for file in receipt["files"]:
        assert file["hash"] == digest(local / Path(file["path"]).name, file["algorithm"])
    assert (local / "articles.csv").read_bytes() == (remote / "articles.csv").read_bytes()
    stats = json.loads((local / "stats.json").read_text())
    assert stats["rows"] == 1 and stats["target_values"] == 64
    assert stats["covariate_values"] == 128
    report, code = run_audit(
        root,
        tmp_path / "audit",
        data,
        [BenchmarkSeries("fixture", "a", "target", "h", [1.0] * 64)],
        state=state,
        control_runner=lambda *args: {"ok": True},
    )
    assert code == 0 and report["valid"]
    assert report["input_errors"] == []
    assert report["folders"][0]["verdict"] == "clean"
    # Missing attribution must be restored along with an otherwise complete folder.
    (local / "articles.csv").unlink()
    fetch(data, root, hub, state=state)
    assert (local / "articles.csv").read_bytes() == (remote / "articles.csv").read_bytes()
    assert json.loads((local / "stats.json").read_text()) == stats
    # A changed side file must fail revalidation even when its size is unchanged.
    licence = local / "LICENSE.md"
    licence.write_bytes(b"x" * licence.stat().st_size)
    with pytest.raises(ValueError, match="hash mismatch"):
        fetch(data, root, hub, state=state)
    if status == "quarantined":
        assert not (root / "covariates").exists()
