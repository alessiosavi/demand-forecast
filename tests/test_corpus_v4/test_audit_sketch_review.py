"""Offline regressions for spool identity, periodic controls and missed plants."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.corpus_v4 import audit_sketch as s
from scripts.corpus_v4.audit_fev_overlap import BenchmarkSeries
from scripts.corpus_v4.make_sketch_job import render
from tests.test_corpus_v4.test_audit_sketch_restart import fixture
from tests.test_corpus_v4.test_manifest_fetch import manifest, native

SPOOL_SOURCES = (
    "scripts/corpus_v4/audit_sketch.py",
    "scripts/corpus_v4/audit_fev_overlap.py",
    "scripts/corpus_v4/native.py",
    "scripts/corpus_v4/matching.py",
    "demand_forecast/bench/specs.py",
    "demand_forecast/pretrain/audit.py",
    "demand_forecast/pretrain/audit_match.py",
    "demand_forecast/pretrain/audit_info.py",
)


@pytest.mark.parametrize("source", SPOOL_SOURCES)
def test_spool_source_bytes_force_rebuild_and_rescan(tmp_path, monkeypatch, source):
    root, state, out, benchmarks = fixture(tmp_path)
    repository = Path(s.__file__).resolve().parents[2]
    mirror = tmp_path / "source"
    for name in SPOOL_SOURCES:
        target = mirror / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, target)
    monkeypatch.setattr(s, "__file__", str(mirror / SPOOL_SOURCES[0]))
    loads, calls = [], []
    execute = s._execute

    def load(cache):
        loads.append(cache)
        return iter(benchmarks)

    def observe(request):
        calls.append(request["entry"]["folder"])
        return execute(request)

    monkeypatch.setattr(s, "load_benchmarks", load)
    monkeypatch.setattr(s, "_execute", observe)
    before, code = s.run_sketch(root, out, manifest(), state=state)
    assert code == 0
    marker = state / "benchmark.sketch-identity.json"
    initial = json.loads(marker.read_text())
    assert set(initial["contract"]["sources"]) == set(SPOOL_SOURCES)
    loads.clear()
    calls.clear()
    reused, code = s.run_sketch(root, out, manifest(), state=state)
    assert code == 0 and reused == before and not loads and not calls
    with (mirror / source).open("a") as stream:
        stream.write("\n# source-byte regression\n")
    after, code = s.run_sketch(root, out, manifest(), state=state)
    rebuilt = json.loads(marker.read_text())
    assert code == 0 and len(loads) == 1
    assert calls == ["univariate/controls", "univariate/demo"]
    assert rebuilt["files"] == initial["files"]  # Identical values still require replay.
    assert rebuilt["contract"]["sources"][source] != initial["contract"]["sources"][source]
    assert after["input_fingerprint"] != before["input_fingerprint"]
    loads.clear()
    calls.clear()
    reused, code = s.run_sketch(root, out, manifest(), state=state)
    assert code == 0 and reused == after and not loads and not calls


def test_periodic_parent_does_not_supply_a_false_interior_negative(tmp_path):
    values = np.tile(np.random.default_rng(8).normal(30, 8, 33), 3)
    root, state, out = (tmp_path / name for name in ("corpus", "state", "out"))
    native(root / "univariate/demo/train-000.parquet", target=values.tolist())
    report, code = s.run_sketch(
        root,
        out,
        manifest(),
        [BenchmarkSeries("periodic", "b", "target", "h", values)],
        state=state,
    )
    assert code == 0 and report["controls"]["ok"]
    assert all(c["class"] != "interior_miss" for c in report["controls"]["cases"])
    skipped = report["controls"]["skipped_cases"]
    assert len(skipped) == 1 and skipped[0]["class"] == "interior_miss"
    assert skipped[0]["benchmark"]["dataset"] == "periodic"
    assert skipped[0]["reason"] == "interior_endpoint_keys_match_parent"


def test_capped_prefix_miss_names_case_and_retrieval_reason(tmp_path, monkeypatch):
    # All prefixes share the planted keys; the true parent is beyond the cap.
    rng = np.random.default_rng(19)
    values = rng.normal(30, 8, 512)
    other = np.r_[values[:256], rng.normal(30, 8, 256)]
    assert s.sketch(other).keys[-1][0] == s.sketch(values[:340]).keys[-1][0]
    root, state, out = (tmp_path / name for name in ("corpus", "state", "out"))
    native(root / "univariate/demo/train-000.parquet", target=values.tolist())
    monkeypatch.setattr(s, "POSTING_CAP", 1)
    monkeypatch.setattr(s, "_execute", s.scan_folder)
    report, code = s.run_sketch(
        root,
        out,
        manifest(),
        [
            BenchmarkSeries("fixture", "decoy", "target", "h", other),
            BenchmarkSeries("wanted", "parent", "target", "h", values),
        ],
        state=state,
    )
    assert code == 2
    missed = next(c for c in report["controls"]["failures"] if c["item_id"] == "plant-2-prefix")
    assert missed["benchmark"]["item_id"] == "parent" and missed["class"] == "prefix"
    assert missed["reason"] == "expected_benchmark_not_proposed"
    assert "posting_cap" in missed["retrieval_reasons"]
    markdown = (out / "sketch-report.md").read_text()
    assert "plant-2-prefix" in markdown and "expected_benchmark_not_proposed" in markdown


def test_control_miss_reports_verifier_rejection(tmp_path, monkeypatch):
    root, state, out, benchmarks = fixture(tmp_path)
    monkeypatch.setattr(s, "_execute", s.scan_folder)
    monkeypatch.setattr(s.old, "fit_segment", lambda *a, **kw: None)
    report, code = s.run_sketch(root, out, manifest(), benchmarks, state=state)
    assert code == 2
    missed = next(c for c in report["controls"]["failures"] if c["class"] == "prefix")
    assert missed["reason"] == "numerical_fit_rejected"
    assert missed["rejection_reasons"]["numerical_fit_rejected"] > 0


@pytest.mark.parametrize(
    "benchmark", ["gs://bucket/state", "gs://bucket/state/bench", "gs://bucket"]
)
def test_benchmark_state_cannot_overlap_writable_destinations(benchmark):
    with pytest.raises(ValueError, match="overlap|separate"):
        render(
            "a" * 40,
            "gs://bucket/corpus",
            "gs://bucket/reports",
            state_prefix="gs://bucket/state",
            benchmark_state_prefix=benchmark,
        )


@pytest.mark.parametrize(
    "benchmark", ["gs://bucket/state", "gs://bucket/state/bench", "gs://bucket"]
)
def test_guest_checks_benchmark_state_against_runtime_state(benchmark):
    source = render(
        "a" * 40, "gs://other/corpus", "gs://other/reports", benchmark_state_prefix=benchmark
    )
    contract = source.split("<<'SKETCH_DESTINATIONS'\n", 1)[1].split("\nSKETCH_DESTINATIONS", 1)[0]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            contract,
            "gs://other/corpus",
            "gs://other/reports",
            "gs://bucket/state",
            "",
            benchmark,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "overlap" in result.stderr
