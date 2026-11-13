"""Audit-only verification never relies on Hub data access or claimed receipts."""

import json
import subprocess
from dataclasses import asdict

import pytest

from scripts.corpus_v4.fetch_native import fetch
from scripts.corpus_v4.make_corpus_job import render
from tests.test_corpus_v4.test_manifest_fetch import FakeHub, manifest, native


def published(tmp_path):
    hub = FakeHub(native(tmp_path / "remote.parquet"))
    data = manifest()
    root = tmp_path / "corpus"
    fetch(data, root, hub, state=tmp_path / "fetch-state")
    files = [asdict(f) for f in hub.inventory(data["repo_id"], data["revision"])[1]]
    return root, data, files


@pytest.mark.parametrize(
    "damage", [None, "bytes", "missing", "stats", "revision", "omitted", "extra", "marker"]
)
def test_published_receipts_checked_against_pinned_inventory(tmp_path, damage):
    from scripts.corpus_v4.verify_published import verify_published

    root, data, files = published(tmp_path)
    folder = root / "univariate/demo"
    receipt = folder / "receipt.json"
    if damage == "bytes":
        p = folder / "train-000.parquet"
        p.write_bytes(b"x" * p.stat().st_size)
    elif damage == "missing":
        (folder / "train-000.parquet").unlink()
    elif damage == "stats":
        (folder / "stats.json").write_text("{}")
    elif damage == "revision":
        r = json.loads(receipt.read_text())
        r["revision"] = "a" * 40
        receipt.write_text(json.dumps(r))
    elif damage == "omitted":
        r = json.loads(receipt.read_text())
        r["files"] = []
        receipt.write_text(json.dumps(r))
    elif damage == "extra":
        (folder / "train-extra.parquet").write_bytes(b"extra")
    elif damage == "marker":
        (root / "corpus-identity.json").unlink()
    if damage:
        with pytest.raises((ValueError, OSError)):
            verify_published(root, data, pinned_files=files)
    else:
        result = verify_published(root, data, pinned_files=files)
        assert result["revision"] == data["revision"]
        assert result["folders"] == 1 and result["files"] == 1


def test_audit_only_job_is_parseable_and_has_no_fetch_phase(tmp_path):
    script = render(
        "a" * 40, "gs://research/corpus-v4", audit_only=True, workers=90, memory_budget_gib=256
    )
    path = tmp_path / "audit-only.sh"
    path.write_text(script)
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    assert "python -m scripts.corpus_v4.fetch_native" not in script
    assert "python -m scripts.corpus_v4.verify_published" in script
    assert "CORPUS_FETCH_ATTEMPT_ID" in script
    assert '--workers "$WORKERS"' in script


def test_checked_in_inventory_is_complete_and_pinned():
    from scripts.corpus_v4.fetch_native import inventory
    from scripts.corpus_v4.freeze_manifest import read_manifest
    from scripts.corpus_v4.verify_published import pinned_inventory

    hub = pinned_inventory()
    groups = inventory(read_manifest(), hub)
    assert len(groups) == 121
    assert sum(f.path.endswith(".parquet") for group in groups.values() for f in group) == 150
