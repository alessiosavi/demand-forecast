"""Content matching controls use synthetic series and native Parquet only."""

import json

import numpy as np
import pytest

pytest.importorskip("pyarrow")
from scripts.corpus_v4.audit_fev_overlap import BenchmarkSeries, run_audit
from scripts.corpus_v4.matching import aggregate_fixed, find_matches
from tests.test_corpus_v4.test_manifest_fetch import manifest, native


def signal(n=320, seed=23):
    rng = np.random.default_rng(seed)
    return rng.normal(30, 8, n) + np.sin(np.arange(n) / 7) * 5


@pytest.fixture
def benchmark_loader(monkeypatch):
    from demand_forecast.bench.specs import FevDatasetTask
    from scripts.corpus_v4 import audit_fev_overlap as audit

    def configure(rows):
        dataset = pytest.importorskip("datasets").Dataset.from_list(rows)
        tasks = [FevDatasetTask(f"bench-{i}", "offline") for i in range(100)]
        monkeypatch.setattr(audit, "fev_bench_tasks", lambda selector: tasks)
        monkeypatch.setattr(
            audit,
            "load_task_dataset",
            lambda task, cache: dataset if task == tasks[0] else dataset.select([]),
        )
        return audit

    return configure


def test_frequency_none_is_unsupported():
    from scripts.corpus_v4.matching import frequency

    assert frequency(None) is None


@pytest.mark.parametrize("length", [0, 1])
def test_short_benchmark_row_borrows_uniform_dataset_alias(tmp_path, benchmark_loader, length):
    import pandas as pd

    audit = benchmark_loader(
        [
            dict(
                id="short",
                timestamp=pd.date_range("2000", periods=length, freq="h"),
                target=signal(length),
            ),
            dict(
                id="long",
                timestamp=pd.date_range("2000", periods=160, freq="h"),
                target=signal(160),
            ),
        ]
    )
    series = list(audit.load_benchmarks(tmp_path / "cache"))
    assert [(item.item_id, item.freq) for item in series] == [("short", "h"), ("long", "h")]
    assert len(series[0].values) == length


@pytest.mark.parametrize("known_aliases", [[], ["h", "D"]], ids=["all_unknown", "mixed"])
def test_unknown_benchmark_frequency_is_reported_in_coverage(
    tmp_path, benchmark_loader, known_aliases
):
    import pandas as pd

    rows = [dict(id="short", timestamp=pd.date_range("2000", periods=1, freq="h"), target=[1.0])]
    rows += [
        dict(id=alias, timestamp=pd.date_range("2000", periods=160, freq=alias), target=signal(160))
        for alias in known_aliases
    ]
    audit = benchmark_loader(rows)
    native(tmp_path / "corpus/univariate/demo/train-000.parquet", target=signal(seed=2).tolist())
    report, code = run_audit(
        tmp_path / "corpus",
        tmp_path / "out",
        manifest(),
        state=tmp_path / "state",
        control_runner=lambda *a: {"ok": True},
    )
    assert code == 2
    assert report["benchmark_coverage"]["unknown_frequency_rows"] == [
        dict(dataset="bench-0", item_id="short", column="target", length=1)
    ]
    folder = report["folders"][0]
    assert folder["verdict"] == "inconclusive"
    assert any(
        pair["benchmark"] == "bench-0"
        and pair["benchmark_frequency"] is None
        and pair["reason"] == "unsupported frequency"
        for pair in folder["skipped_pairs"]
    )
    assert json.loads((tmp_path / "out/audit.json").read_text()) == report
    series = list(audit.load_benchmarks(tmp_path / "cache"))
    assert series[0].freq is None


def test_float32_scaled_cropped_inside_multivariate_segment(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    original = signal()
    planted = (original[47:247] * 3.71 - 8.32).astype("float32")
    planted[30:36] = np.nan
    folder = tmp_path / "corpus/univariate/demo"
    path = native(folder / "train-000.parquet", item="entity@512", target=planted.tolist())
    row = pq.read_table(path).to_pylist()[0]
    row["target"].insert(0, signal(200, seed=74).tolist())
    pq.write_table(pa.Table.from_pylist([row]), path)
    result, code = run_audit(
        tmp_path / "corpus",
        tmp_path / "out",
        manifest(),
        [BenchmarkSeries("bench", "b", "target", "h", original)],
        state=tmp_path / "state",
    )
    assert code != 0
    report = result["folders"][0]
    assert report["verdict"] == "rule_f_match"
    evidence = report["matches"][0]
    assert evidence["corpus"]["channel"] == 1
    assert evidence["corpus"]["segment_offset"] == 512
    assert evidence["benchmark"]["dataset"] == "bench"
    assert evidence["segment"] >= 24
    assert abs(evidence["coef"][0] - 1 / 3.71) < 1e-4


def test_hourly_parent_matches_daily_but_seasonal_twin_does_not(tmp_path):
    daily = signal(160)
    rng = np.random.default_rng(88)
    noise = rng.normal(0, 0.5, (len(daily), 24))
    noise -= noise.mean(axis=1)[:, None]
    hourly = (daily[:, None] / 24 + noise).ravel()
    # Shift the boundary: no absolute start metadata is needed for fixed aggregation.
    hourly = np.r_[rng.normal(size=7), hourly, rng.normal(size=3)]
    native(tmp_path / "corpus/univariate/demo/train-000.parquet", target=hourly.tolist())
    benchmarks = [
        BenchmarkSeries("daily", "d", "target", "D", daily),
        BenchmarkSeries("seasonal-twin", "t", "target", "D", signal(160, seed=99)),
    ]
    report, code = run_audit(
        tmp_path / "corpus", tmp_path / "out", manifest(), benchmarks, state=tmp_path / "state"
    )
    assert code != 0
    folder = report["folders"][0]
    assert folder["verdict"] == "rule_l_match"
    assert {m["benchmark"]["dataset"] for m in folder["matches"]} == {"daily"}
    assert any(m["aggregation"]["phase"] == 7 for m in folder["matches"])


@pytest.mark.parametrize(
    "x", [np.zeros(300), np.ones(300), np.tile([0, 0, 0, 2.0], 75), np.r_[np.zeros(297), [2, 3, 4]]]
)
def test_zero_inflated_constants_and_short_motifs_do_not_match(x):
    matches, coverage = find_matches(x, x * 4 + 3)
    assert not matches


def test_missing_values_and_unrelated_seasonality():
    x = signal()
    y = x.astype("float32") * -2.4 + 7
    y[::2] = np.nan
    assert find_matches(y, x)[0]
    assert not find_matches(signal(seed=11), x)[0]


def test_control_failure_invalidates_clean_audit(tmp_path):
    native(tmp_path / "corpus/univariate/demo/train-000.parquet", target=signal().tolist())
    result, code = run_audit(
        tmp_path / "corpus",
        tmp_path / "out",
        manifest(),
        [BenchmarkSeries("bench", "b", "target", "h", signal(seed=54))],
        state=tmp_path / "state",
        control_runner=lambda *a: {"ok": False, "reason": "planted failure"},
    )
    assert code != 0 and not result["valid"]
    assert json.loads((tmp_path / "out/audit.json").read_text())["controls"]["ok"] is False


def test_resume_revalidates_inputs_and_reruns_controls(tmp_path):
    corpus = tmp_path / "corpus"
    path = native(corpus / "univariate/demo/train-000.parquet", target=signal(seed=2).tolist())
    args = (
        corpus,
        tmp_path / "out",
        manifest(),
        [BenchmarkSeries("b", "b", "target", "h", signal())],
    )
    first, _ = run_audit(*args, state=tmp_path / "state")
    second, _ = run_audit(*args, state=tmp_path / "state")
    assert second["folders"][0]["resumed"]
    native(path, target=signal().tolist())
    third, code = run_audit(*args, state=tmp_path / "state")
    assert code != 0 and not third["folders"][0]["resumed"]
    assert first["input_fingerprint"] != third["input_fingerprint"]


def test_benchmark_cache_and_checkpoints_are_scoped_by_revision(tmp_path, monkeypatch):
    from scripts.corpus_v4 import audit_fev_overlap as audit

    native(tmp_path / "corpus/univariate/demo/train-000.parquet", target=signal(seed=2).tolist())
    caches = []

    def benchmarks(cache):
        caches.append(cache)
        return [BenchmarkSeries("bench", "b", "target", "h", signal())]

    monkeypatch.setattr(audit, "load_benchmarks", benchmarks)
    state = tmp_path / "state"
    args = (tmp_path / "corpus", tmp_path / "out", manifest())
    kwargs = dict(state=state, control_runner=lambda *a: {"ok": True})
    first, _ = run_audit(*args, **kwargs)
    second, _ = run_audit(*args, **kwargs)
    assert second["folders"][0]["resumed"]
    original_revision = audit.FEV_DATASETS_REVISION
    monkeypatch.setattr(audit, "FEV_DATASETS_REVISION", "another-revision")
    third, _ = run_audit(*args, **kwargs)
    assert caches == [
        state / "benchmark-cache" / original_revision,
        state / "benchmark-cache" / original_revision,
        state / "benchmark-cache" / "another-revision",
    ]
    assert third["benchmark_revision"] == "another-revision"
    assert not third["folders"][0]["resumed"]
    assert first["input_fingerprint"] != third["input_fingerprint"]


def test_aggregation_does_not_invent_observed_values():
    x = np.arange(96, dtype=float)
    x[2] = np.nan
    a = aggregate_fixed(x, 24, 0, "sum")
    assert np.isnan(a[0]) and a[1] == sum(x[24:48])


def test_calendar_unknown_origin_is_inconclusive_and_quarantine_stays_quarantined(tmp_path):
    native(
        tmp_path / "corpus/quarantine/univariate/demo/train-000.parquet",
        freq="D",
        target=signal(800).tolist(),
    )
    data = manifest(status="quarantined")
    report, code = run_audit(
        tmp_path / "corpus",
        tmp_path / "out",
        data,
        [BenchmarkSeries("monthly", "b", "target", "MS", signal(160))],
        state=tmp_path / "state",
    )
    assert code == 0
    assert report["folders"][0]["verdict"] == "inconclusive"
    assert report["folders"][0]["status"] == "quarantined"


def test_calendar_benchmark_parent_aggregates_to_monthly_corpus(tmp_path):
    import pandas as pd

    daily = signal(365 * 12)
    stamps = pd.date_range("2000-01-01", periods=len(daily), freq="D")
    monthly = pd.Series(daily, index=stamps).resample("MS").sum().iloc[1:-1].to_numpy()
    native(
        tmp_path / "corpus/univariate/demo/train-000.parquet", freq="MS", target=monthly.tolist()
    )
    report, _ = run_audit(
        tmp_path / "corpus",
        tmp_path / "out",
        manifest(),
        [BenchmarkSeries("daily", "b", "target", "D", daily, stamps)],
        state=tmp_path / "state",
    )
    assert report["folders"][0]["verdict"] == "rule_l_match"


def test_matching_gap_at_chunk_boundary_and_benchmark_crop():
    from scripts.corpus_v4.matching import chunks

    original = signal(70000)
    crop = original[65000:66000].astype("float32") * 3 + 5
    found = [r for _, part in chunks(original) for r in find_matches(crop, part)[0]]
    assert found


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("H", "60min"),
        ("1H", "h"),
        ("15T", "15min"),
        ("3M", "3ME"),
        ("A", "YE"),
        ("2A-JUN", "2YE-JUN"),
        ("2Q-FEB", "2QE-FEB"),
        ("2Y-MAR", "2YE-MAR"),
        ("D", "24h"),
        ("W-MON", "W-SUN"),
        ("MS", "ME"),
    ],
)
def test_same_resolution_frequency_aliases(alias, canonical):
    from scripts.corpus_v4.matching import frequency

    assert frequency(alias) is not None
    assert frequency(alias) == frequency(canonical)


def test_business_days_are_not_daily():
    from scripts.corpus_v4.matching import frequency

    assert frequency("B") != frequency("D")


def test_corrupt_parquet_still_writes_inconclusive_audit(tmp_path):
    path = native(tmp_path / "corpus/univariate/demo/train-000.parquet")
    path.write_bytes(b"bad parquet payload")
    report, code = run_audit(
        tmp_path / "corpus",
        tmp_path / "out",
        manifest(),
        [BenchmarkSeries("b", "b", "target", "h", signal())],
        state=tmp_path / "state",
    )
    assert code == 2
    assert report["folders"][0]["verdict"] == "inconclusive"
    assert (tmp_path / "out/audit.json").exists()


def test_unlisted_corpus_folder_invalidates_audit(tmp_path):
    native(
        tmp_path / "corpus/univariate/demo/train-000.parquet", target=signal(160, seed=5).tolist()
    )
    native(tmp_path / "corpus/univariate/unknown/train-000.parquet", target=signal(160).tolist())
    report, code = run_audit(
        tmp_path / "corpus",
        tmp_path / "out",
        manifest(),
        [BenchmarkSeries("b", "b", "target", "h", signal())],
        state=tmp_path / "state",
    )
    assert code == 2 and not report["valid"]
    assert "unlisted" in str(report["input_errors"])


def test_fiscal_quarter_parent_not_silently_calendar_quarter(tmp_path):
    import pandas as pd

    daily = signal(365 * 15)
    stamps = pd.date_range("2000-01-01", periods=len(daily), freq="D")
    fiscal = pd.Series(daily, index=stamps).resample("QS-FEB").sum().iloc[1:-1].to_numpy()
    native(
        tmp_path / "corpus/univariate/demo/train-000.parquet", freq="QS-FEB", target=fiscal.tolist()
    )
    report, _ = run_audit(
        tmp_path / "corpus",
        tmp_path / "out",
        manifest(),
        [BenchmarkSeries("daily", "b", "target", "D", daily, stamps)],
        state=tmp_path / "state",
    )
    assert report["folders"][0]["verdict"] == "rule_l_match"


def test_multi_month_unknown_phase_is_not_clean(tmp_path):
    import pandas as pd

    daily = signal(365 * 15)
    stamps = pd.date_range("2000-01-01", periods=len(daily), freq="D")
    two_month = pd.Series(daily, index=stamps).resample("2MS").sum().iloc[1:-1].to_numpy()
    native(
        tmp_path / "corpus/univariate/demo/train-000.parquet", freq="2MS", target=two_month.tolist()
    )
    report, _ = run_audit(
        tmp_path / "corpus",
        tmp_path / "out",
        manifest(),
        [BenchmarkSeries("daily", "b", "target", "D", daily, stamps)],
        state=tmp_path / "state",
    )
    assert report["folders"][0]["verdict"] != "clean"


def test_combined_index_reports_correct_benchmark_owner_and_masks():
    from demand_forecast.pretrain.audit_match import Index
    from scripts.corpus_v4.matching import find_index_matches

    wrong, right = signal(seed=2), signal(seed=5)
    query = (right[50:250] * 3 + 7).astype("float32")
    query[::2] = np.nan
    found, _ = find_index_matches(query, Index([wrong, right], ["wrong", "right"]))
    assert found and {x["index_series_index"] for x in found} == {1}


def test_posting_budget_checked_before_old_matcher_expansion(monkeypatch):
    from demand_forecast.pretrain import audit_match as old
    from scripts.corpus_v4.matching import find_index_matches

    rng = np.random.default_rng(23)
    motif = np.tile(rng.normal(size=8), 8192)
    motif[-200:] = rng.normal(size=200)  # whole series is not periodic
    index = old.Index([motif], ["repetitive-local"])

    def forbid_expansion(*args, **kwargs):
        raise AssertionError("unbounded expansion entered")

    monkeypatch.setattr(old, "scan_group", forbid_expansion)
    found, coverage = find_index_matches(motif, index)
    assert not found and coverage["unresolved_rows"] > 0


def test_constant_daily_parent_month_lengths_do_not_establish_lineage(tmp_path):
    import pandas as pd

    daily = np.ones(365 * 15)
    stamps = pd.date_range("2000-01-01", periods=len(daily), freq="D")
    months = pd.Series(daily, index=stamps).resample("MS").sum().iloc[1:-1].to_numpy()
    native(tmp_path / "corpus/univariate/demo/train-000.parquet", freq="MS", target=months.tolist())
    report, _ = run_audit(
        tmp_path / "corpus",
        tmp_path / "out",
        manifest(),
        [
            BenchmarkSeries("daily", "b", "target", "D", daily, stamps),
            BenchmarkSeries("control-source", "c", "target", "MS", signal(160)),
        ],
        state=tmp_path / "state",
    )
    assert report["folders"][0]["verdict"] == "clean"


def test_calendar_only_cropped_match_from_parent_with_excursions_is_rejected(tmp_path):
    import pandas as pd

    daily = np.ones(365 * 15)
    stamps = pd.date_range("2000-01-01", periods=len(daily), freq="D")
    months = pd.Series(daily, index=stamps).resample("MS").sum().iloc[1:-1].to_numpy()
    daily[-6:] += np.arange(1, 7)
    native(tmp_path / "corpus/univariate/demo/train-000.parquet", freq="MS", target=months.tolist())
    report, _ = run_audit(
        tmp_path / "corpus",
        tmp_path / "out",
        manifest(),
        [
            BenchmarkSeries("daily", "b", "target", "D", daily, stamps),
            BenchmarkSeries("control-source", "c", "target", "MS", signal(160)),
        ],
        state=tmp_path / "state",
    )
    assert not report["folders"][0]["matches"]
