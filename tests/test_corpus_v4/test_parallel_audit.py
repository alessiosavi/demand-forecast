"""Compare execution changes against the frozen 32511ee audit, offline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.corpus_v4 import audit_fev_overlap as audit
from scripts.corpus_v4.native import iter_rows
from tests.test_corpus_v4 import baseline_audit_32511ee as baseline
from tests.test_corpus_v4 import test_audit as fixtures
from tests.test_corpus_v4.test_manifest_fetch import manifest, native


def oracle(*args, **kwargs):
    # Run the original scanner directly: its CLI worker name belongs to the new code.
    baseline._scan_in_subprocess = baseline._scan_folder
    return baseline.run_audit(*args, **kwargs)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_mixed_folders_frequencies_batches_match_oracle(tmp_path, monkeypatch, workers):
    x = fixtures.signal(96)
    data = manifest()
    data["folders"] = [manifest(f)["folders"][0] for f in ("univariate/a", "covariates/b")]
    native(tmp_path / "corpus/univariate/a/train-000.parquet", target=x.tolist())
    path = native(tmp_path / "corpus/covariates/b/train-000.parquet", cov=True, target=x.tolist())
    rows = pq.read_table(path).to_pylist()
    rows += [dict(rows[0], freq="2h", item_id="coarse@32", target=(x * 3 + 2).tolist())]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
    benches = [
        audit.BenchmarkSeries("h", "a", "target", "h", x),
        audit.BenchmarkSeries("h-other", "b", "target", "h", fixtures.signal(96, seed=4)),
        audit.BenchmarkSeries("coarse", "c", "target", "2h", x * 3 + 2),
        audit.BenchmarkSeries("monthly", "m", "target", "MS", fixtures.signal(96, seed=5)),
        audit.BenchmarkSeries("unknown", "u", "target", None, x),
        audit.BenchmarkSeries(
            "calendar",
            "d",
            "target",
            "D",
            fixtures.signal(96),
            pd.date_range("2000", periods=96, freq="D"),
        ),
    ]
    monkeypatch.setattr(audit, "MAX_INDEX_POINTS", 128)
    monkeypatch.setattr(baseline, "MAX_INDEX_POINTS", 128)
    args = (tmp_path / "corpus", tmp_path / "new", data, benches)
    actual, code = audit.run_audit(*args, state=tmp_path / "state", workers=workers)
    expected, old_code = oracle(*args, state=tmp_path / "old-state")
    assert (actual, code) == (expected, old_code)
    assert actual["folders"][0]["verdict"] == "rule_f_match"
    assert any(f["matches"] for f in actual["folders"])
    progress = json.loads((tmp_path / "state/audit-progress.json").read_text())
    assert progress["units_total"] == progress["units_done"] == 3
    assert progress["units_failed"] == 0
    assert progress["values_scanned"] > 0
    assert progress["native_values_decoded"] == 672
    assert progress["elapsed_seconds"] >= 0
    assert progress["estimated_remaining_seconds"] == 0
    assert set(progress["folders"]) == {"univariate/a", "covariates/b"}
    checkpoints = list((tmp_path / "state/audit-units").glob("*/*.json"))
    assert checkpoints
    assert all(
        json.loads(p.read_text())["memory_budget_bytes"] == 32 * 1024**3 // workers
        for p in checkpoints
    )


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize(
    "scenario",
    [
        "test_float32_scaled_cropped_inside_multivariate_segment",
        "test_hourly_parent_matches_daily_but_seasonal_twin_does_not",
        "test_control_failure_invalidates_clean_audit",
        "test_resume_revalidates_inputs_and_reruns_controls",
        "test_calendar_unknown_origin_is_inconclusive_and_quarantine_stays_quarantined",
        "test_calendar_benchmark_parent_aggregates_to_monthly_corpus",
        "test_corrupt_parquet_still_writes_inconclusive_audit",
        "test_unlisted_corpus_folder_invalidates_audit",
        "test_fiscal_quarter_parent_not_silently_calendar_quarter",
        "test_multi_month_unknown_phase_is_not_clean",
        "test_constant_daily_parent_month_lengths_do_not_establish_lineage",
        "test_calendar_only_cropped_match_from_parent_with_excursions_is_rejected",
    ],
)
def test_existing_fixtures_field_equivalence(tmp_path, monkeypatch, workers, scenario):
    def compare(root, out, data, benchmarks=None, **kwargs):
        actual, code = audit.run_audit(root, out, data, benchmarks, workers=workers, **kwargs)
        old_kwargs = dict(kwargs, state=Path(kwargs["state"]).with_name("oracle-state"))
        expected, old_code = oracle(
            root, Path(out).with_name("oracle-out"), data, benchmarks, **old_kwargs
        )
        # Compare the published JSON contract: legacy controls contain tuples,
        # whereas resumed controls are necessarily JSON arrays. Ignore no fields.
        assert json.loads(json.dumps(actual)) == json.loads(json.dumps(expected))
        assert code == old_code
        return actual, code

    monkeypatch.setattr(fixtures, "run_audit", compare)
    getattr(fixtures, scenario)(tmp_path)


def test_row_group_selection_keeps_original_row_numbers(tmp_path):
    path = native(tmp_path / "train-000.parquet")
    rows = pq.read_table(path).to_pylist() * 5
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    selected = list(iter_rows(tmp_path, "univariate", shard=path.name, row_group=1))
    assert [r["row_index"] for r in selected] == [2, 3]


def test_pool_restart_keeps_completed_units_and_controls_once(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    path = native(tmp_path / "corpus/univariate/demo/train-000.parquet", target=[0.0] * 96)
    rows = pq.read_table(path).to_pylist() * 20
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
    program = tmp_path / "restart.py"
    program.write_text("""
import json, sys
from pathlib import Path
import numpy as np
from scripts.corpus_v4.audit_fev_overlap import run_audit, BenchmarkSeries
from tests.test_corpus_v4.test_manifest_fetch import manifest
p = Path(sys.argv[1])
def checks(*args):
    with (p / "control-calls").open("a") as f:
        f.write("once\\n")
    return {"ok": True}
r, code = run_audit(p / "corpus", p / "out", manifest(),
    [BenchmarkSeries("bench", "b", "target", "h", np.random.default_rng(42).normal(size=96))],
    state=p / "state", workers=int(sys.argv[2]), control_runner=checks)
sys.exit(code)
""")
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), OPENBLAS_NUM_THREADS="1")
    log = (tmp_path / "process.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, str(program), str(tmp_path), "2"], stdout=log, stderr=log, env=env
    )
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            progress = tmp_path / "state/audit-progress.json"
            if progress.exists():
                r = json.loads(progress.read_text())
                if 0 < r["units_done"] < r["units_total"]:
                    break
            assert proc.poll() is None
            time.sleep(0.05)
        else:
            pytest.fail("pool did not checkpoint a unit before timeout")
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) != 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        log.close()
    completed = {p: p.stat().st_mtime_ns for p in (tmp_path / "state/audit-units").glob("*/*.json")}
    assert completed
    resumed = subprocess.run(
        [sys.executable, str(program), str(tmp_path), "4"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert (tmp_path / "control-calls").read_text() == "once\n"
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in completed.items())
    actual = json.loads((tmp_path / "out/audit.json").read_text())
    expected, _ = oracle(
        tmp_path / "corpus",
        tmp_path / "old-out",
        manifest(),
        [
            audit.BenchmarkSeries(
                "bench", "b", "target", "h", np.random.default_rng(42).normal(size=96)
            )
        ],
        state=tmp_path / "oracle-state",
        control_runner=lambda *args: {"ok": True},
    )
    assert actual == expected


def test_journal_reducer_keeps_cap_query_counters_and_legacy_order(tmp_path):
    from scripts.corpus_v4.audit_parallel import reduce_folder

    pairs = [
        dict(
            benchmark="bench",
            benchmark_frequency="h",
            corpus_frequency="h",
            domains="all pairs (conservative)",
            rule="F",
            direction="same_frequency",
        )
    ]
    plan = dict(ca="h", ba="h", pairs=pairs, failure=None, direction="same_frequency", rule="F")
    paths = []
    for unit, count in enumerate([6000, 6000]):
        path = tmp_path / f"unit-{unit}.jsonl"
        record = dict(rule="F", benchmark=dict(dataset="bench"), corpus=dict(row=unit))
        delta = dict(
            coverage=dict(low_information=unit + 1),
            comparisons=dict(bench=unit + 1),
            aggregations=dict(bench=['{"method": "identity"}']),
            reasons=[],
            skipped=None,
            matches=[record] * count,
        )
        path.write_text(json.dumps([0, 0, delta]) + "\n")
        paths.append(path)
    report = reduce_folder(manifest()["folders"][0], [plan], paths)
    assert len(report["matches"]) == 10000
    assert [r["corpus"]["row"] for r in report["matches"]] == [0] * 6000 + [1] * 4000
    assert report["coverage"] == dict(low_information=3)
    assert report["searched_pairs"][0]["comparisons"] == 3
    assert report["searched_pairs"][0]["matched"] == 10000
    assert report["reason"] == ["evidence record limit reached"]


@pytest.mark.parametrize("missing", [False, True])
def test_reused_query_hashes_preserve_full_matcher_result(missing):
    from demand_forecast.pretrain.audit_match import Index
    from scripts.corpus_v4.matching import QueryHashes, find_index_matches

    x = fixtures.signal(160)
    if missing:
        x[::2] = float("nan")
    prepared = QueryHashes(x)
    for seed in (23, 42):
        index = Index([fixtures.signal(240, seed=seed)], [str(seed)])
        actual = find_index_matches(x, index, query_hashes=prepared)
        expected = baseline.find_index_matches(x, index)
        assert actual == expected


def test_strided_benchmark_indices_are_reused_without_changing_results():
    from demand_forecast.pretrain.audit_match import Index
    from scripts.corpus_v4.matching import find_index_matches

    y = fixtures.signal(160)
    y[::5] = float("nan")
    index = Index([y], ["missing"])
    stride_indices = {}
    for seed in (23, 42):
        x = fixtures.signal(160, seed=seed)
        actual = find_index_matches(x, index, stride_indices=stride_indices)
        expected = baseline.find_index_matches(x, index)
        assert actual == expected
    assert len(stride_indices) == 9


def test_checkpoint_publisher_orders_journal_before_marker(tmp_path, monkeypatch):
    import os
    import sys

    from scripts.corpus_v4.audit_parallel import publish_checkpoints

    state = tmp_path / "state"
    state.mkdir()
    files = [state / "unit.jsonl", state / "unit.json"]
    for p in files:
        p.write_text(p.name)
    binary = tmp_path / "bin"
    binary.mkdir()
    tool = binary / "gcloud"
    tool.write_text(
        f"#!{sys.executable}\n"
        + """
import os, pathlib, sys
with open(os.environ["PUBLISH_LOG"], "a") as f:
    f.write(pathlib.Path(sys.argv[-2]).name + "\\n")
sys.exit(int(os.environ.get("PUBLISH_FAIL", "0")))
"""
    )
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary}:{os.environ['PATH']}")
    monkeypatch.setenv("PUBLISH_LOG", str(tmp_path / "publish.log"))
    monkeypatch.setenv("CORPUS_AUDIT_STATE_URI", "gs://synthetic/state")
    publish_checkpoints(state, files)
    assert (tmp_path / "publish.log").read_text() == "unit.jsonl\nunit.json\n"
    monkeypatch.setenv("PUBLISH_FAIL", "9")
    with pytest.raises(ValueError, match="checkpoint upload failed"):
        publish_checkpoints(state, files)
    assert (tmp_path / "publish.log").read_text() == "unit.jsonl\nunit.json\nunit.jsonl\n"


def test_unit_stops_after_local_evidence_cap_with_identical_global_report(tmp_path, monkeypatch):
    from scripts.corpus_v4 import audit_parallel as parallel

    root = tmp_path / "corpus"
    native(root / "univariate/demo/train-000.parquet", target=fixtures.signal(96).tolist())
    benches = [
        audit.BenchmarkSeries("bench", str(i), "target", "h", fixtures.signal(96, seed=i))
        for i in range(21)
    ]
    store = audit.BenchmarkStore(tmp_path / "benchmark.sqlite", benches)
    calls = []

    def many_records(*args, **kwargs):
        calls.append(1)
        return [dict(index_series_index=0) for _ in range(1000)], {"low_information": 1}

    monkeypatch.setattr(parallel, "find_index_matches", many_records)
    monkeypatch.setattr(baseline, "find_index_matches", many_records)
    monkeypatch.setattr(baseline, "MAX_INDEX_POINTS", 128)
    monkeypatch.setattr(audit, "MAX_INDEX_POINTS", 128)
    entry = manifest()["folders"][0]
    journal = tmp_path / "unit.jsonl"
    try:
        parallel.scan_unit(
            dict(
                root=str(root),
                entry=entry,
                store=str(store.path),
                memory_budget_bytes=32 * 1024**3,
                max_index_points=128,
                scratch=str(tmp_path / "scratch"),
                aliases=["h"],
                opportunities=1 << 60,
                unit=dict(shard="train-000.parquet", group=0, order=0),
            ),
            journal,
            tmp_path / "progress.json",
        )
        # Co-probing can finish the resident group, but must skip all later groups.
        assert len(calls) == 12
        actual = parallel.reduce_folder(entry, list(parallel.pair_plans(store, ["h"])), [journal])
        expected = baseline._scan_folder(root, entry, store, 1 << 60, 32 * 1024**3)
        assert actual == expected
        assert len(actual["matches"]) == 10000
    finally:
        store.close()


def test_reducer_memory_does_not_multiply_positive_payloads_by_units(tmp_path):
    import tracemalloc

    from scripts.corpus_v4.audit_parallel import reduce_folder

    pairs = [
        dict(
            benchmark="bench",
            benchmark_frequency="h",
            corpus_frequency="h",
            domains="all pairs (conservative)",
            rule="F",
            direction="same_frequency",
        )
    ]
    plan = dict(ca="h", ba="h", pairs=pairs, failure=None, direction="same_frequency", rule="F")
    paths = []
    for unit in range(8):
        record = dict(rule="F", benchmark=dict(dataset="bench"), corpus=dict(row=unit))
        delta = dict(
            coverage={},
            comparisons=dict(bench=1),
            aggregations=dict(bench=[]),
            reasons=[],
            skipped=None,
            matches=[record] * 10000,
        )
        path = tmp_path / f"{unit}.jsonl"
        path.write_text(json.dumps([0, 0, delta]) + "\n")
        paths.append(path)
    tracemalloc.start()
    try:
        report = reduce_folder(manifest()["folders"][0], [plan], paths)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(report["matches"]) == 10000
    assert peak < 24 * 1024**2, f"eager per-unit payloads retained {peak} bytes"


@pytest.mark.parametrize("row_group_size", [1, 2])
def test_speculative_failure_after_legacy_cap_preserves_report(
    tmp_path, monkeypatch, row_group_size
):
    from scripts.corpus_v4 import audit_parallel as parallel

    path = native(
        tmp_path / "corpus/quarantine/univariate/demo/train-000.parquet",
        target=fixtures.signal(96).tolist(),
    )
    rows = pq.read_table(path).to_pylist()
    rows.append(dict(rows[0], item_id="bad", target=[]))
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)

    def cap_records(*args, **kwargs):
        return [dict(index_series_index=0) for _ in range(10000)], {}

    monkeypatch.setattr(baseline, "find_index_matches", cap_records)
    calls = []

    def original_order(root, entry, store, opportunities, budget):
        calls.append(budget)
        return baseline._scan_folder(root, entry, store, opportunities, budget)

    monkeypatch.setattr(parallel, "_serial_scan", original_order, raising=False)
    data = manifest(status="quarantined")
    benches = [audit.BenchmarkSeries("bench", "b", "target", "h", fixtures.signal(96))]
    actual, code = audit.run_audit(
        tmp_path / "corpus",
        tmp_path / "out",
        data,
        benches,
        state=tmp_path / "state",
        workers=2,
        control_runner=lambda *args: {"ok": True},
    )
    expected, old_code = oracle(
        tmp_path / "corpus",
        tmp_path / "old-out",
        data,
        benches,
        state=tmp_path / "old-state",
        control_runner=lambda *args: {"ok": True},
    )
    assert (actual, code) == (expected, old_code)
    assert code == 0 and actual["folders"][0]["verdict"] == "rule_f_match"
    assert calls == [16 * 1024**3]


def test_real_serial_fallback_matches_original_scanner(tmp_path):
    from scripts.corpus_v4 import audit_parallel as parallel

    native(
        tmp_path / "corpus/univariate/demo/train-000.parquet", target=fixtures.signal(96).tolist()
    )
    store = audit.BenchmarkStore(
        tmp_path / "benchmark.sqlite",
        [audit.BenchmarkSeries("b", "b", "target", "h", fixtures.signal(96))],
    )
    entry = manifest()["folders"][0]
    try:
        actual = parallel._serial_scan(tmp_path / "corpus", entry, store, 1 << 60, 32 * 1024**3)
        expected = baseline._scan_folder(tmp_path / "corpus", entry, store, 1 << 60, 32 * 1024**3)
        assert actual == expected
    finally:
        store.close()
