"""Offline endpoint triage contracts: synthetic values, never dataset downloads."""

import json
from dataclasses import replace

import numpy as np
import pytest

from scripts.corpus_v4 import audit_sketch as sketch
from scripts.corpus_v4.audit_fev_overlap import BenchmarkSeries
from tests.test_corpus_v4.test_manifest_fetch import manifest, native


def signal(n=512, seed=41):
    return np.random.default_rng(seed).normal(30, 8, n)


@pytest.mark.parametrize(
    "transform",
    [
        lambda x: x,
        lambda x: 3.7 * x + 8,
        lambda x: -2.5 * x - 7,
        lambda x: x + 19,
        lambda x: x.astype("float32").astype("float64"),
        lambda x: x[:320],
        lambda x: x[-320:],
    ],
    ids=["whole", "positive", "negative", "shift", "float32", "prefix", "suffix"],
)
def test_endpoint_detection_and_original_verifier(transform, monkeypatch):
    x = signal()
    y = transform(x)
    actual = sketch.old.fit_segment
    calls = []

    def observed(a, b, *args, **kwargs):
        calls.append((a, b))
        return actual(a, b, *args, **kwargs)

    monkeypatch.setattr(sketch.old, "fit_segment", observed)
    candidates = sketch.compare_sketches(sketch.sketch(y), sketch.sketch(x))
    evidence = [sketch.verify_candidate(y, x, c, opportunities=1 << 80) for c in candidates]
    assert calls, "retrieval must call the original numerical verifier"
    assert any(e["verified"] for e in evidence)
    assert all(np.array_equal(a, y, equal_nan=True) for a, b in calls)
    monkeypatch.setattr(sketch.old, "fit_segment", lambda *a, **k: None)
    assert not any(
        sketch.verify_candidate(y, x, c, opportunities=1 << 80)["verified"] for c in candidates
    )


def test_interior_crop_is_deliberate_miss_and_masks_are_positional():
    x = signal()
    assert not sketch.compare_sketches(sketch.sketch(x[80:380]), sketch.sketch(x))
    x[::13] = np.nan
    assert sketch.compare_sketches(sketch.sketch(x * -2), sketch.sketch(x))
    moved = np.roll(x, 1)
    assert sketch.sketch(x).finite_digest != sketch.sketch(moved).finite_digest
    assert not sketch.compare_sketches(sketch.sketch(moved), sketch.sketch(x))


@pytest.mark.parametrize("x", [np.ones(512), np.arange(512.0), np.tile([28.0, 30.0, 31.0], 171)])
def test_nonidentifying_patterns_are_never_verified(x):
    for candidate in sketch.compare_sketches(sketch.sketch(x), sketch.sketch(x)):
        assert not sketch.verify_candidate(x, x, candidate, opportunities=1 << 80)["verified"]


def test_calendar_gate_on_original_values():
    import pandas as pd

    stamps = pd.date_range("1990", periods=512, freq="MS")
    days = stamps.days_in_month.to_numpy(dtype=float)
    candidate = sketch.Candidate(0, "raw", 0.0)
    result = sketch.verify_candidate(days, days, candidate, opportunities=1, dimensions=days)
    assert not result["verified"]
    assert result["reason"] == "calendar_pattern"


def test_hash_collision_only_proposes_and_never_declares_copy():
    a, b = sketch.sketch(signal()), sketch.sketch(signal(seed=99))
    collision = replace(b, keys=a.keys)
    candidates = sketch.compare_sketches(a, collision)
    assert candidates
    assert not any(
        sketch.verify_candidate(signal(), signal(seed=99), c, opportunities=1)["verified"]
        for c in candidates
    )


def test_full_pass_controls_separation_and_determinism(tmp_path):
    root = tmp_path / "corpus"
    x = signal()
    native(root / "univariate/demo/train-000.parquet", target=x.tolist())
    native(
        root / "quarantine/covariates/demo/train-000.parquet",
        cov=True,
        target=signal(seed=81).tolist(),
    )
    policy = manifest()
    policy["folders"] += manifest("covariates/demo", "quarantined")["folders"]
    benches = [BenchmarkSeries("fixture", "b", "target", "h", x)]
    reports = []
    for workers in (1, 2, 4):
        report, code = sketch.run_sketch(
            root,
            tmp_path / f"out{workers}",
            policy,
            benches,
            state=tmp_path / f"state{workers}",
            workers=workers,
        )
        assert code == 0
        assert report["provisional"] is True
        assert report["release_eligible"] is False
        assert report["artifact_type"] == "corpus-sketch-triage"
        assert "valid" not in report and "exit_code" not in report
        assert all(f.get("verdict") != "clean" for f in report["folders"])
        assert report["controls"]["ok"]
        assert {c["class"] for c in report["controls"]["cases"]} >= {
            "whole",
            "affine_positive",
            "affine_negative",
            "shift",
            "float32",
            "prefix",
            "suffix",
            "interior_miss",
        }
        assert report["totals"]["allowed"]["channels"] == 1
        assert report["totals"]["quarantined"]["channels"] == 3
        assert report["folders"][0]["routing"][0]["shared_source"]
        assert not list(root.rglob("*control*"))
        assert not (tmp_path / f"out{workers}/audit.json").exists()
        assert json.loads((tmp_path / f"out{workers}/sketch-report.json").read_text()) == report
        reports.append(report)
    assert reports[0] == reports[1] == reports[2]
