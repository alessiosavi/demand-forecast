"""A bad folder cannot hide failures/successes in the rest of an offline attempt."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts.corpus_v4 import audit_fev_overlap as audit
from scripts.corpus_v4 import fetch_native as fetcher
from scripts.corpus_v4.native import digest
from tests.test_corpus_v4.test_manifest_fetch import manifest, native


class MultiHub:
    def __init__(self, sources):
        self.sources = sources
        self.calls = Counter()
        self.corrupt = set()

    def inventory(self, repo_id, revision):
        return revision, [
            fetcher.HubFile(
                f"{folder}/train-000.parquet", path.stat().st_size, digest(path), "sha256"
            )
            for folder, path in self.sources.items()
        ]

    def download(self, repo_id, revision, file, destination):
        self.calls[file.path] += 1
        destination.write_bytes(
            b"broken parquet"
            if file.path in self.corrupt
            else self.sources[str(Path(file.path).parent)].read_bytes()
        )


def test_collect_all_failures_audit_completed_and_retry_only_failure(tmp_path, monkeypatch):
    folders = ["univariate/a", "univariate/b", "univariate/c"]
    sources = {
        folder: native(tmp_path / f"remote/{i}.parquet", target=[0.0] * 96)
        for i, folder in enumerate(folders)
    }
    hub = MultiHub(sources)
    hub.corrupt.add("univariate/b/train-000.parquet")
    data = manifest()
    data["folders"] = [dict(manifest(folder)["folders"][0]) for folder in folders]
    root, state = tmp_path / "state/corpus", tmp_path / "state"
    with pytest.raises(ValueError, match="univariate/b"):
        fetcher.fetch(data, root, hub, state=state)
    summary = json.loads((state / "fetch-summary.json").read_text())
    assert summary["failed_folders"] == ["univariate/b"]
    assert all((root / folder / "receipt.json").exists() for folder in (folders[0], folders[2]))
    benchmarks = [audit.BenchmarkSeries("b", "b", "target", "h", np.zeros(96))]
    report, code = audit.run_audit(
        root,
        tmp_path / "out",
        data,
        benchmarks,
        state=state,
        control_runner=lambda *a: {"ok": True},
    )
    assert code != 0 and report["failed_folders"] == ["univariate/b"]
    assert [r["verdict"] for r in report["folders"]] == ["clean", "inconclusive", "clean"]
    # Retry the interrupted/corrupt transfer; the pinned remote identity stays fixed.
    hub.corrupt.clear()
    measured = []
    original = fetcher.measure_folder

    def measure(path, *args, **kwargs):
        measured.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(fetcher, "measure_folder", measure)
    fetcher.fetch(data, root, hub, state=state)
    assert measured == ["b"]
    assert hub.calls == Counter(
        {f"{f}/train-000.parquet": (2 if f == folders[1] else 1) for f in folders}
    )
    report, code = audit.run_audit(
        root,
        tmp_path / "out",
        data,
        benchmarks,
        state=state,
        control_runner=lambda *a: {"ok": True},
    )
    assert code == 0 and not report["failed_folders"]
    assert [r["resumed"] for r in report["folders"]] == [True, False, True]
    assert not any(p.is_symlink() for p in state.rglob("*"))
    assert len(list(state.rglob("*.parquet"))) == 3


def test_public_hub_uses_local_download_without_blob_cache(tmp_path, monkeypatch):
    import huggingface_hub

    source = native(tmp_path / "remote.parquet")
    calls = []

    def download(repo_id, filename, **kwargs):
        calls.append(kwargs)
        assert "cache_dir" not in kwargs
        local = Path(kwargs["local_dir"]) / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(source.read_bytes())
        return str(local)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    public = fetcher.PublicHub()
    fake = MultiHub({"univariate/demo": source})
    monkeypatch.setattr(public, "inventory", fake.inventory)
    state = tmp_path / "state"
    fetcher.fetch(manifest(), state / "corpus", public, state=state)
    assert calls and not any(p.is_symlink() for p in state.rglob("*"))
    blobs = [p for p in state.rglob("*") if p.is_file() and digest(p) == digest(source)]
    assert len(blobs) == 1


def test_collects_multiple_corrupt_shards_and_quarantine_errors(tmp_path):
    folders = ["univariate/a", "univariate/b", "univariate/c"]
    sources = {f: native(tmp_path / f"remote/{i}.parquet") for i, f in enumerate(folders)}
    for folder in (folders[0], folders[2]):
        sources[folder].write_bytes(b"corrupt parquet")
    data = manifest()
    data["folders"] = [
        manifest(f, "quarantined" if i == 2 else "allowed")["folders"][0]
        for i, f in enumerate(folders)
    ]
    state = tmp_path / "state"
    with pytest.raises(ValueError) as exc:
        fetcher.fetch(data, state / "corpus", MultiHub(sources), state=state)
    assert "univariate/a" in str(exc.value) and "univariate/c" in str(exc.value)
    assert (state / "corpus/univariate/b/stats.json").exists()
    report, code = audit.run_audit(
        state / "corpus",
        tmp_path / "out",
        data,
        [audit.BenchmarkSeries("b", "b", "target", "h", np.zeros(96))],
        state=state,
        control_runner=lambda *a: {"ok": True},
    )
    assert code != 0
    assert report["failed_folders"] == [folders[0], folders[2]]


def test_transport_error_is_a_folder_error_and_does_not_stop_other_folders(tmp_path):
    folders = ["univariate/a", "univariate/b"]
    hub = MultiHub({f: native(tmp_path / f"remote/{i}.parquet") for i, f in enumerate(folders)})
    original = hub.download

    def download(repo_id, revision, file, destination):
        if file.path.startswith(folders[0] + "/"):
            raise RuntimeError("synthetic transport failure")
        return original(repo_id, revision, file, destination)

    hub.download = download
    data = manifest()
    data["folders"] = [manifest(f)["folders"][0] for f in folders]
    with pytest.raises(ValueError, match="synthetic transport failure"):
        fetcher.fetch(data, tmp_path / "root", hub, state=tmp_path / "state")
    assert (tmp_path / "root/univariate/b/receipt.json").exists()
