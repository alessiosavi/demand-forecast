"""Synthetic equivalence, durable units, setup reuse and progress contracts."""

import json

import numpy as np
import pytest

# Collected in the environment without the bench extra too: skip there instead of failing collection.
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from scripts.corpus_v4 import audit_sketch as s  # noqa: E402
from scripts.corpus_v4.audit_fev_overlap import BenchmarkSeries  # noqa: E402
from tests.test_corpus_v4.test_manifest_fetch import manifest  # noqa: E402


def fixture(tmp_path):
    root = tmp_path / "corpus"
    path = root / "univariate/demo/train-000.parquet"
    path.parent.mkdir(parents=True)
    rng = np.random.default_rng(42)
    identifying = rng.normal(30, 8, 512)
    rows = [
        dict(item_id=str(i), source="fixture", freq="h", target=[v.tolist()])
        for i, v in enumerate(
            [identifying, np.ones(512), np.tile([0.0, 1.0], 256), identifying * 3 + 4]
        )
    ]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
    benchmarks = [
        BenchmarkSeries("fixture", "b", "target", "h", identifying),
        BenchmarkSeries("constant", "c", "target", "h", np.ones(512)),
    ]
    return root, benchmarks


def test_shortcut_preserves_matches_controls_and_proposal_counts(tmp_path):
    root, benchmarks = fixture(tmp_path)
    results = []
    for shortcut in (False, True):
        report, code = s.run_sketch(
            root, tmp_path / str(shortcut), manifest(), benchmarks, nonidentifying_shortcut=shortcut
        )
        assert code == 0
        results.append(report)
    plain, fast = results
    assert plain["controls"] == fast["controls"]
    assert plain["folders"][0]["matches"] == fast["folders"][0]["matches"]
    for key in (
        "channels",
        "values",
        "proposed_pairs",
        "verified_matches",
        "unverified_candidates",
    ):
        assert plain["folders"][0]["counts"].get(key, 0) == fast["folders"][0]["counts"].get(key, 0)
    assert fast["folders"][0]["counts"]["nonidentifying_pairs_skipped"] > 0


def test_shortcut_uses_joint_mask_and_overlap_not_whole_channel():
    # Globally identifying signal becomes constant under the joint mask.
    x = np.random.default_rng(4).normal(size=512)
    y = x.copy()
    x[:256] = 1
    y[:256] = 1
    y[256:] = np.nan
    proposal = s.Candidate(0, "fixture")
    fast = s.verify_candidate(x, y, proposal, opportunities=1, nonidentifying_shortcut=True)
    plain = s.verify_candidate(x, y, proposal, opportunities=1)
    assert not fast["verified"] and not plain["verified"]
    assert fast["shortcut"] and fast["fit"] is None
    # A non-identifying whole channel may contain an identifying crop.
    x = np.r_[np.zeros(10000), np.random.default_rng(7).normal(30, 8, 512), 1e10]
    y = x[10000:-1]
    fast = s.verify_candidate(
        x, y, s.Candidate(10000, "crop"), opportunities=1, nonidentifying_shortcut=True
    )
    assert fast["verified"]


def test_units_resume_reuses_setup_and_completed_units(tmp_path, monkeypatch):
    root, benchmarks = fixture(tmp_path)
    out, state = tmp_path / "out", tmp_path / "state"
    execute = s._execute
    seen = []

    def fail(request):
        if not request.get("control"):
            seen.append(request["unit"]["row_group"])
            if request["unit"]["row_group"] == 1:
                raise RuntimeError("killed worker")
        return execute(request)

    monkeypatch.setattr(s, "_execute", fail)
    failed, code = s.run_sketch(root, out, manifest(), benchmarks, state=state, workers=1)
    assert code == 2 and not failed["execution_ok"]
    assert len(list((out / "evidence/units").glob("*-rg??????.json"))) == 3
    seen.clear()

    def observe(request):
        assert not request.get("control"), "resume must reuse verified controls"
        seen.append(request["unit"]["row_group"])
        return execute(request)

    monkeypatch.setattr(s, "_execute", observe)
    monkeypatch.setattr(s, "_control_shards", lambda *a: pytest.fail("controls rebuilt"))
    resumed, code = s.run_sketch(root, out, manifest(), state=state, workers=3)
    assert code == 0 and seen == [1]
    progress = json.loads((state / "sketch-progress.json").read_text())
    assert progress["units_done"] == progress["units_total"] == 4
    assert progress["channels_done"] == 4 and progress["values_done"] == 2048
    assert progress["folders"]["univariate/demo"]["units_done"] == 4
    assert progress["estimated_seconds_left"] == 0
    assert progress["values_per_minute"] >= 0
    assert not list(state.rglob("*.tmp")) and not list(out.rglob("*.tmp"))


def test_units_summary_is_identical_across_workers(tmp_path):
    root, benchmarks = fixture(tmp_path)
    reports = [
        s.run_sketch(root, tmp_path / f"out{w}", manifest(), benchmarks, workers=w)[0]
        for w in (1, 3)
    ]
    assert reports[0] == reports[1]
    assert len(reports[0]["folders"][0]["evidence"]["units"]) == 4


def test_only_folders_requires_valid_finished_identity(tmp_path, monkeypatch):
    root, benchmarks = fixture(tmp_path)
    policy = manifest()
    second = root / "univariate/other"
    second.mkdir()
    (second / "train-000.parquet").write_bytes(
        (root / "univariate/demo/train-000.parquet").read_bytes()
    )
    policy["folders"] += manifest("univariate/other")["folders"]
    out, state = tmp_path / "out", tmp_path / "state"
    before, _ = s.run_sketch(root, out, policy, benchmarks, state=state)
    execute = s._execute
    calls = []

    def observe(request):
        calls.append(request["entry"]["folder"])
        return execute(request)

    monkeypatch.setattr(s, "_execute", observe)
    after, code = s.run_sketch(
        root, out, policy, state=state, only_folders=["univariate/demo"], skip_finished=True
    )
    assert code == 0 and before == after and not calls
    marker = out / "evidence/univariate--other.json"
    record = json.loads(marker.read_text())
    record["identity"] = "wrong"
    marker.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unselected.*identity"):
        s.run_sketch(
            root, out, policy, state=state, only_folders=["univariate/demo"], skip_finished=True
        )


def test_legacy_summary_is_not_reused(tmp_path):
    root, benchmarks = fixture(tmp_path)
    out, state = tmp_path / "out", tmp_path / "state"
    _, code = s.run_sketch(root, out, manifest(), benchmarks, state=state)
    assert code == 0
    marker = out / "evidence/univariate--demo.json"
    record = json.loads(marker.read_text())
    identity = json.loads((state / "sketch-input.json").read_text())
    assert "compatible_legacy_fingerprint" not in identity
    record["identity"] = "legacy-sketch1-fingerprint"
    record["result"]["evidence"].pop("units")
    record["result_digest"] = s._fingerprint(record["result"])
    marker.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unselected.*identity"):
        s.run_sketch(root, out, manifest(), state=state, only_folders=[], skip_finished=True)


def test_killed_group_keeps_completed_bounded_fragments(tmp_path, monkeypatch):
    import subprocess
    import sys

    root, benchmarks = fixture(tmp_path)
    path = root / "univariate/demo/train-000.parquet"
    row = pq.read_table(path).to_pylist()[0]
    rows = [dict(row, item_id=str(i)) for i in range(6)]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=6)
    out, state = tmp_path / "out", tmp_path / "state"
    execute = s._execute

    def kill(request):
        if request.get("control"):
            return execute(request)
        program = """
import json, os, signal, sys
from scripts.corpus_v4 import audit_sketch as s
s.CHECKPOINT_ROWS = 2
original = s.sketch
count = 0
def die(values):
    global count
    count += 1
    if count == 3:
        os.kill(os.getpid(), signal.SIGKILL)
    return original(values)
s.sketch = die
s.scan_folder(json.load(sys.stdin))
"""
        result = subprocess.run(
            [sys.executable, "-c", program],
            input=json.dumps(request),
            text=True,
            capture_output=True,
        )
        assert result.returncode == -9, result.stderr
        # Two rows are durably closed; the third row never enters the state tree.
        partials = list((out / "evidence/units").glob("*part*.json"))
        assert len(partials) == 1
        assert json.loads(partials[0].read_text())["result"]["counts"]["channels"] == 2
        assert not list(state.rglob("*.tmp")) and not list(out.rglob("*.tmp"))
        raise RuntimeError("worker killed after durable fragment")

    monkeypatch.setattr(s, "_execute", kill)
    _, code = s.run_sketch(root, out, manifest(), benchmarks, state=state)
    assert code == 2
    program = """
import json, sys
from scripts.corpus_v4 import audit_sketch as s
s.CHECKPOINT_ROWS = 2
original = s.sketch
count = 0
def observe(values):
    global count
    count += 1
    return original(values)
s.sketch = observe
result = s.scan_folder(json.load(sys.stdin))
assert count == 4, count
print(json.dumps(result))
"""

    def resume(request):
        result = subprocess.run(
            [sys.executable, "-c", program],
            input=json.dumps(request),
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    monkeypatch.setattr(s, "_execute", resume)
    resumed, code = s.run_sketch(root, out, manifest(), state=state)
    assert code == 0 and resumed["folders"][0]["counts"]["channels"] == 6
    progress = json.loads((state / "sketch-progress.json").read_text())
    assert progress["values_reused"] == 1024

    def clean_execute(request):
        clean_program = "import json, sys; from scripts.corpus_v4 import audit_sketch as s; s.CHECKPOINT_ROWS=2; print(json.dumps(s.scan_folder(json.load(sys.stdin))))"
        result = subprocess.run(
            [sys.executable, "-c", clean_program],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    monkeypatch.setattr(s, "_execute", clean_execute)
    clean, code = s.run_sketch(root, tmp_path / "clean", manifest(), benchmarks)
    assert code == 0
    # Physical fragment boundaries can differ; numerical folder output cannot.
    assert resumed["folders"] == clean["folders"]


def test_complete_unit_journal_is_compressed_and_inspectable(tmp_path):
    import gzip

    root, benchmarks = fixture(tmp_path)
    out = tmp_path / "out"
    report, code = s.run_sketch(root, out, manifest(), benchmarks)
    assert code == 0
    references = report["folders"][0]["evidence"]["units"]
    records = []
    for reference in references:
        path = out / "evidence" / reference["file"]
        assert path.suffix == ".gz"
        with gzip.open(path, "rt") as stream:
            records.extend(json.loads(line) for line in stream)
    verified = [r for r in records if r.get("verified") and not r.get("deferred_summary")]
    assert len(verified) == report["folders"][0]["counts"]["verified_matches"]


def test_missing_identity_is_refused(tmp_path):
    root, benchmarks = fixture(tmp_path)
    out = tmp_path / "out"
    _, code = s.run_sketch(root, out, manifest(), benchmarks)
    assert code == 0
    marker = out / "evidence/univariate--demo.json"
    record = json.loads(marker.read_text())
    record.pop("identity")
    marker.write_text(json.dumps(record))
    assert s._saved_result(marker, "current-identity", out / "evidence") is None


def test_large_group_splits_into_bounded_ranges_without_duplicate_rows(tmp_path, monkeypatch):
    root, benchmarks = fixture(tmp_path)
    path = root / "univariate/demo/train-000.parquet"
    pq.write_table(pq.read_table(path), path, row_group_size=4)
    monkeypatch.setattr(s, "UNIT_VALUES", 1024)
    reports = [
        s.run_sketch(root, tmp_path / f"ranges{workers}", manifest(), benchmarks, workers=workers)[
            0
        ]
        for workers in (1, 3)
    ]
    assert reports[0] == reports[1]
    folder = reports[0]["folders"][0]
    assert folder["counts"]["channels"] == 4 and folder["counts"]["values"] == 2048
    assert len(folder["evidence"]["units"]) == 2
    assert {r["corpus"]["row"] for r in folder["matches"]} == {0, 3}


def test_completed_range_fragments_do_not_resume_past_range_end(tmp_path, monkeypatch):
    root, benchmarks = fixture(tmp_path)
    path = root / "univariate/demo/train-000.parquet"
    pq.write_table(pq.read_table(path), path, row_group_size=4)
    monkeypatch.setattr(s, "UNIT_VALUES", 1024)
    out = tmp_path / "out"
    before, code = s.run_sketch(root, out, manifest(), benchmarks)
    assert code == 0
    first = before["folders"][0]["evidence"]["units"][0]["file"]
    (out / "evidence" / first).write_bytes(b"corrupt final journal; fragments remain intact")
    after, code = s.run_sketch(root, out, manifest())
    assert code == 0 and after == before


def test_corrupt_fragment_values_are_counted_as_new_work(tmp_path, monkeypatch):
    root, benchmarks = fixture(tmp_path)
    path = root / "univariate/demo/train-000.parquet"
    pq.write_table(pq.read_table(path), path, row_group_size=4)
    monkeypatch.setattr(s, "UNIT_VALUES", 1024)
    out = tmp_path / "out"
    before, _ = s.run_sketch(root, out, manifest(), benchmarks)
    journal = out / "evidence" / before["folders"][0]["evidence"]["units"][0]["file"]
    fragment = journal.with_name(journal.name.removesuffix(".jsonl.gz") + "-part000000.jsonl.gz")
    fragment.write_bytes(b"corrupt fragment")
    journal.write_bytes(b"corrupt compiled unit")
    after, code = s.run_sketch(root, out, manifest())
    assert code == 0 and after == before
    progress = json.loads((out / "state/sketch-progress.json").read_text())
    assert progress["values_reused"] == 1024
    assert progress["values_per_minute"] > 0
