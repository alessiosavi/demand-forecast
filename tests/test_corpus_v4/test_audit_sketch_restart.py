"""Real disposable-worker death, immutable inputs, and checkpoint revalidation."""

import json
import subprocess
import sys

import numpy as np
import pytest

from scripts.corpus_v4 import audit_sketch as sketch
from scripts.corpus_v4.audit_fev_overlap import BenchmarkSeries
from tests.test_corpus_v4.test_manifest_fetch import manifest, native


def fixture(tmp_path):
    values = np.random.default_rng(9).normal(30, 8, 512)
    root, state, out = (tmp_path / p for p in ("corpus", "state", "out"))
    native(root / "univariate/demo/train-000.parquet", target=values.tolist())
    return root, state, out, [BenchmarkSeries("fixture", "b", "target", "h", values)]


def test_killed_worker_resume_matches_uninterrupted_report(tmp_path, monkeypatch):
    root, state, out, benchmarks = fixture(tmp_path)
    execute = sketch._execute
    killed = []

    def interrupt(request):
        if request.get("control"):
            return execute(request)
        # Kill a real folder worker after it has computed its first sketch and
        # opened the temporary evidence journal, before completion publication.
        program = """
import json, os, signal, sys
from scripts.corpus_v4 import audit_sketch as s
original = s.sketch
def die(values):
    original(values)
    os.kill(os.getpid(), signal.SIGKILL)
s.sketch = die
s.scan_folder(json.load(sys.stdin))
"""
        result = subprocess.run(
            [sys.executable, "-c", program],
            input=json.dumps(request),
            text=True,
            capture_output=True,
        )
        assert result.returncode == -9
        killed.append(True)
        raise RuntimeError("sketch worker exit -9")

    monkeypatch.setattr(sketch, "_execute", interrupt)
    failed, code = sketch.run_sketch(root, out, manifest(), benchmarks, state=state, workers=2)
    assert killed and code == 2 and not failed["execution_ok"]
    assert not (out / "evidence/univariate--demo.json").exists()
    monkeypatch.setattr(sketch, "_execute", execute)
    resumed, code = sketch.run_sketch(root, out, manifest(), state=state, workers=4)
    assert code == 0
    clean, code = sketch.run_sketch(
        root, tmp_path / "clean", manifest(), benchmarks, state=tmp_path / "clean-state", workers=1
    )
    assert code == 0 and resumed == clean


def test_resume_reassembles_corrupt_journal_and_reuses_verified_controls(tmp_path, monkeypatch):
    root, state, out, benchmarks = fixture(tmp_path)
    before, _ = sketch.run_sketch(root, out, manifest(), benchmarks, state=state)
    journal = out / "evidence/univariate--demo.jsonl"
    journal.write_text("corrupt")
    execute = sketch._execute
    calls = []

    def observe(request):
        calls.append(request["entry"]["folder"])
        return execute(request)

    monkeypatch.setattr(sketch, "_execute", observe)
    after, code = sketch.run_sketch(root, out, manifest(), state=state)
    assert code == 0 and before == after
    assert calls == []
    calls.clear()
    same, code = sketch.run_sketch(root, out, manifest(), state=state)
    assert same == before and code == 0
    assert calls == []


def test_spool_requires_revision_and_byte_identity(tmp_path, monkeypatch):
    root, state, out, benchmarks = fixture(tmp_path)
    path, before = sketch.prepare_benchmarks(state, benchmarks)
    monkeypatch.setattr(sketch, "load_benchmarks", lambda cache: iter(benchmarks))
    _, reused = sketch.prepare_benchmarks(state)
    assert before == reused
    with path.with_suffix(".values").open("r+b") as stream:
        stream.write(b"damaged!")
    _, rebuilt = sketch.prepare_benchmarks(state)
    assert rebuilt == dict(before, origin="pinned_loader")
    marker = state / "benchmark.sketch-identity.json"
    record = json.loads(marker.read_text())
    record["contract"]["revision"] = "wrong-revision"
    marker.write_text(json.dumps(record))
    _, rebuilt = sketch.prepare_benchmarks(state)
    assert rebuilt == dict(before, origin="pinned_loader")


def test_production_cli_verifies_published_before_scanning(tmp_path, monkeypatch):
    def refuse(*args):
        raise ValueError("published bytes changed")

    monkeypatch.setattr(sketch, "verify_published", refuse)
    monkeypatch.setattr(sketch, "run_sketch", lambda *a, **k: pytest.fail("scan before verify"))
    with pytest.raises(ValueError, match="published bytes changed"):
        sketch.main([str(tmp_path / "corpus"), str(tmp_path / "out")])


def test_packed_common_key_cap_is_explicit(tmp_path):
    values = np.ones(512)
    benchmarks = [
        BenchmarkSeries("same", str(i), "target", "h", values)
        for i in range(sketch.POSTING_CAP + 5)
    ]
    spool, identity = sketch.prepare_benchmarks(tmp_path, benchmarks)
    directory = tmp_path / "index"
    sketch.build_index(spool, directory, identity, memory_budget_bytes=2 * 1024**3)
    index = sketch.SketchIndex(directory, spool)
    try:
        pairs, summaries, heavy = index.lookup(sketch.sketch(values))
        assert heavy and all(h["reason"] == "posting_cap" for h in heavy)
        assert all(h["postings"] == len(benchmarks) for h in heavy)
        assert len(summaries) <= sketch.POSTING_CAP
    finally:
        index.close()


def test_corrupt_completion_marker_is_replayed(tmp_path):
    root, state, out, benchmarks = fixture(tmp_path)
    before, _ = sketch.run_sketch(root, out, manifest(), benchmarks, state=state)
    marker = out / "evidence/univariate--demo.json"
    record = json.loads(marker.read_text())
    record["result"]["counts"]["channels"] = 0
    marker.write_text(json.dumps(record))
    after, code = sketch.run_sketch(root, out, manifest(), state=state)
    assert code == 0 and after == before
    marker.write_text("interrupted JSON")
    after, code = sketch.run_sketch(root, out, manifest(), state=state)
    assert code == 0 and after == before


def test_production_controls_fail_if_verifier_is_disabled(tmp_path, monkeypatch):
    root, state, out, benchmarks = fixture(tmp_path)
    monkeypatch.setattr(sketch, "_execute", sketch.scan_folder)
    monkeypatch.setattr(sketch.old, "fit_segment", lambda *args, **kwargs: None)
    report, code = sketch.run_sketch(root, out, manifest(), benchmarks, state=state)
    assert code == 2 and not report["controls"]["ok"]
    assert all(not c["ok"] for c in report["controls"]["cases"] if c["expected"])


def test_full_crowded_index_controls_and_metadata_routing(tmp_path):
    root, state, out, benchmarks = fixture(tmp_path)
    for i in range(200):
        benchmarks.append(
            BenchmarkSeries(
                "fixture", str(i), "target", "h", np.random.default_rng(i).normal(10, 3, 512)
            )
        )
    policy = manifest()
    policy["folders"][0]["domain"] = "Energy"
    benchmarks[0].domain = "Energy"
    report, code = sketch.run_sketch(root, out, policy, benchmarks, state=state, workers=4)
    assert code == 0 and report["controls"]["ok"]
    assert report["benchmark_series"] == 201
    route = report["folders"][0]["routing"][0]
    assert route["shared_frequency"] and route["shared_domain"] and route["shared_source"]


def test_production_never_reuses_injected_synthetic_spool(tmp_path, monkeypatch):
    root, state, out, benchmarks = fixture(tmp_path)
    sketch.prepare_benchmarks(state, benchmarks)
    calls = []

    def pinned(cache):
        calls.append(cache)
        raise ValueError("pinned loader required")

    monkeypatch.setattr(sketch, "load_benchmarks", pinned)
    with pytest.raises(ValueError, match="pinned loader required"):
        sketch.prepare_benchmarks(state, require_pinned=True)
    assert calls


def test_sidecars_and_reports_cannot_enter_corpus_tree(tmp_path):
    root, state, out, benchmarks = fixture(tmp_path)
    with pytest.raises(ValueError, match="outside the corpus"):
        sketch.run_sketch(root, root / "univariate/reports", manifest(), benchmarks, state=state)
    with pytest.raises(ValueError, match="outside the corpus"):
        sketch.run_sketch(root, out, manifest(), benchmarks, state=root / "controls")


def test_short_identifying_benchmarks_can_supply_all_control_classes(tmp_path):
    values = np.random.default_rng(39).normal(30, 8, 128)
    root, state, out = (tmp_path / p for p in ("corpus", "state", "out"))
    native(root / "univariate/demo/train-000.parquet", target=values.tolist())
    report, code = sketch.run_sketch(
        root, out, manifest(), [BenchmarkSeries("short", "b", "target", "h", values)], state=state
    )
    assert code == 0 and report["controls"]["ok"]
    assert not report["controls"]["unplantable_datasets"]


@pytest.mark.parametrize("source", ["audit_sketch.py", "audit_match.py", "audit_info.py"])
def test_source_only_change_replays_old_numerical_decisions(tmp_path, monkeypatch, source):
    root, state, out, benchmarks = fixture(tmp_path)
    before, _ = sketch.run_sketch(root, out, manifest(), benchmarks, state=state)
    index_files = json.loads((state / "sketch-index/index.json").read_text())["files"]
    actual_digest, execute = sketch.digest, sketch._execute
    calls = []

    def revised(path, *args, **kwargs):
        return (
            "source-only-revision" if path.name == source else actual_digest(path, *args, **kwargs)
        )

    def observe(request):
        calls.append(request["entry"]["folder"])
        return execute(request)

    monkeypatch.setattr(sketch, "digest", revised)
    monkeypatch.setattr(sketch, "_execute", observe)
    monkeypatch.setattr(sketch, "load_benchmarks", lambda cache: iter(benchmarks))
    after, code = sketch.run_sketch(root, out, manifest(), state=state)
    assert code == 0
    assert index_files == json.loads((state / "sketch-index/index.json").read_text())["files"]
    assert calls == ["univariate/controls", "univariate/demo"]
    assert after["input_fingerprint"] != before["input_fingerprint"]


def test_calendar_only_column_cannot_hide_identifying_control_source(tmp_path):
    import pandas as pd

    stamps = pd.date_range("1990", periods=512, freq="MS")
    days = stamps.days_in_month.to_numpy(dtype=float)
    values = np.random.default_rng(39).normal(30, 8, 512)
    root, state, out = (tmp_path / p for p in ("corpus", "state", "out"))
    native(root / "univariate/demo/train-000.parquet", target=values.tolist(), freq="MS")
    benchmarks = [
        BenchmarkSeries("monthly", "b", "days_in_month", "MS", days, stamps),
        BenchmarkSeries("monthly", "b", "target", "MS", values, stamps),
    ]
    report, code = sketch.run_sketch(root, out, manifest(), benchmarks, state=state)
    assert code == 0 and report["controls"]["ok"]
    assert {c["benchmark"]["column"] for c in report["controls"]["cases"] if c["expected"]} == {
        "target"
    }


def test_recommended_memory_budget_leaves_the_documented_parent_reserve(tmp_path, monkeypatch):
    root, state, out, benchmarks = fixture(tmp_path)
    execute = sketch._execute
    quotas = []

    def observe(request):
        quotas.append(request["budget"])
        return execute(request)

    monkeypatch.setattr(sketch, "_execute", observe)
    report, code = sketch.run_sketch(
        root,
        out,
        manifest(),
        benchmarks,
        state=state,
        workers=16,
        memory_budget_bytes=104 * 1024**3,
    )
    assert code == 0 and report["execution_ok"]
    assert quotas and max(quotas) <= 6 * 1024**3


def test_production_reuse_requires_all_fev100_datasets(tmp_path, monkeypatch):
    root, state, out, benchmarks = fixture(tmp_path)
    monkeypatch.setattr(sketch, "load_benchmarks", lambda cache: iter(benchmarks))
    sketch.prepare_benchmarks(state)
    with pytest.raises(ValueError, match="every fev100"):
        sketch.prepare_benchmarks(state, require_pinned=True)
