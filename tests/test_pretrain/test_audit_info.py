"""Score-C reference equivalence and scan-level identification contracts."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from demand_forecast.pretrain import audit_info as info
from demand_forecast.pretrain import audit_match as match
from demand_forecast.pretrain.audit_controls import control_cases

FIXTURE = Path(__file__).parents[1] / "fixtures/audit_info/segments.jsonl"
SEGMENTS = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
LIMITED = {
    "indexed_1100_times_shift+10_fallback": 48,
    "three_levels_seven_runs_dof7_2x+1": 48,
    "four_levels_eight_runs_1.1x+0.5": 64,
    "five_level_staircase_five_runs_x2": 61,
    "intermittent_4_nonzero_of_26_2x+1": 55,
}


@pytest.mark.parametrize("row", SEGMENTS, ids=lambda row: str(row["record_id"]))
def test_reference_score_exact(row):
    score = info.info_bits(row["values"])
    assert score.bits == row["bits_c"]
    assert score.strict_ramp == row["ramp"]
    assert (score.n, score.levels, score.runs) == (row["n"], row["k"], row["runs"])


def test_fixture_inventory():
    assert len(SEGMENTS) >= 300 and FIXTURE.stat().st_size <= 300_000
    assert len({r["task"] for r in SEGMENTS}) == 17
    assert len({(r["task"], r["cls"]) for r in SEGMENTS}) == 37
    assert {r["cls"] for r in SEGMENTS} == {
        "linear_ramp",
        "near_linear_ramp",
        "zero_sparse_counts",
        "two_level",
        "periodic",
        "plateau_few_excursions",
        "few_level",
        "other",
    }
    boomlet = [r for r in SEGMENTS if r["task"] == "boomlet_1975"]
    assert len(boomlet) == 11 and sum(r["ramp"] for r in boomlet) == 7
    assert sum(r["candidate"] for r in SEGMENTS) >= 40
    assert max(r["n"] for r in SEGMENTS) == 96
    assert max(r["bits_c"] for r in SEGMENTS) == 95


CASES, _ = control_cases()


@pytest.mark.parametrize("name", CASES)
def test_control_identification_and_detection_limit(name):
    spec = CASES[name]
    target, planted, expected = spec[:3]
    index = match.Index(planted, list(range(len(planted))))
    dims = [spec[3]] if len(spec) > 3 and spec[3] is not None else None
    pairs, rejections, scan_info = [], [], {}
    counts = match.scan(
        "control", name, [target], index, {}, pairs, rejections, dims=dims, scan_info=scan_info
    )
    copies = [p for p in pairs if p["is_copy"]]
    assert bool(copies) == expected
    if len(spec) > 4:
        assert any(spec[4] in r["degenerate_reason"] for r in rejections)
    assert scan_info["search_bits"] == (
        math.log2(max(1, sum(counts["windows"].values())))
        + math.log2(max(1, sum(index.cov["windows"].values())))
    )
    if expected:
        assert any(p["identification"] == "identifying" for p in copies)
        bits = max(min(p["info_bits_target"], p["info_bits_index"]) for p in copies)
        assert (bits < 56.84675 + info.INFO_MARGIN_BITS) == (name in LIMITED)
        if name in LIMITED:
            assert bits == LIMITED[name]
        # Exercise the actual labeller with exactly the specified production B.
        # Fractional counts here are a test input solely to fix log2(T) exactly.
        info.finish_identification({"windows": {"exact": 2**56.84675}}, copies, {})
        expected_status = "low_information" if name in LIMITED else "identifying"
        assert any(p["identification"] == expected_status for p in copies)
        if name in LIMITED:
            assert {p["identification"] for p in copies} == {expected_status}


def _record(bits, *, ramp=False, index_bits=None):
    return {
        "is_copy": True,
        "info_bits_target": bits,
        "info_bits_index": bits if index_bits is None else index_bits,
        "info_strict_ramp": ramp,
        "_info_values": ([1.0], [2.0]),
    }


def test_threshold_equality_minimum_and_values_selection():
    counts = {"windows": {"exact": 8, "affine": 8}}
    records = [
        _record(29),
        _record(28, index_bits=29),
        _record(100, index_bits=27),
        _record(100, ramp=True),
        {"is_copy": False},
    ]
    got = info.finish_identification(counts, records, {"exact": 16})
    assert got == {
        "version": 1,
        "margin_bits": 20,
        "target_windows": 16,
        "index_windows": 16,
        "search_bits": 8.0,
    }
    assert [p.get("identification") for p in records] == [
        "identifying",
        "identifying",
        "low_information",
        "interpolation_ramp",
        None,
    ]
    assert all(p["info_threshold"] == 28 for p in records[:4])
    assert [counts[s + "_copies"] for s in info.STATUSES] == [2, 1, 1]
    assert all("values_target" in p for p in records[:3])
    assert "values_target" not in records[3]
    assert all("_info_values" not in p for p in records)


def test_low_information_top20_uses_bits_and_stable_record_order():
    records = [_record(5) for _ in range(25)] + [_record(6), _record(20), _record(0, ramp=True)]
    info.finish_identification({"windows": {}}, records, {})
    kept = [i for i, p in enumerate(records) if "values_target" in p]
    assert kept == [*range(19), 25, 26]  # 20 low-information plus every identifying
    assert all("_info_values" not in p for p in records)


def test_zero_window_scan():
    counts = {"windows": dict.fromkeys(match.FAMILIES, 0)}
    got = info.finish_identification(counts, [], dict.fromkeys(match.FAMILIES, 0))
    assert got["search_bits"] == got["target_windows"] == got["index_windows"] == 0
    assert all(counts[s + "_copies"] == 0 for s in info.STATUSES)


def test_strict_ramp_uses_original_joint_positions():
    target = np.arange(48, dtype=float)
    corpus = 2 * target + 1
    target[[12, 14, 30]] = np.nan
    corpus[[20, 35]] = np.nan
    record = match.fit_segment(target, corpus, 0, 48, 0)
    assert record["is_copy"] and record["info_strict_ramp"]
    assert record["info_bits_target"] == record["info_bits_index"] == 0
    both = np.isfinite(target) & np.isfinite(corpus)
    assert record["_info_values"] == (target[both].tolist(), corpus[both].tolist())
    assert not info.info_bits(target[both]).strict_ramp
    # Conversely, a ramp only after compression is not an original-position ramp.
    compressed = target.copy()
    compressed[both] = np.arange(both.sum())
    compressed[~both] = np.nan
    rec = match.fit_segment(compressed, compressed.copy(), 0, 48, 0)
    assert rec["is_copy"] and not rec["info_strict_ramp"]
    assert info.info_bits(compressed[both]).strict_ramp


def test_strict_ramp_requires_both_precision_and_copy_tolerance():
    ramp = np.linspace(0, 1, 32)
    assert info.info_bits(ramp.astype(np.float32)).strict_ramp
    ramp[10] += 1e-4  # within copy tolerance, beyond eight float32 ULPs
    assert not info.info_bits(ramp).strict_ramp
    large = 1e6 + np.arange(32, dtype=float)
    large[10] += 0.2  # below eight ULPs, above range-relative copy tolerance
    assert not info.info_bits(large).strict_ramp


def test_quantisation_uses_matcher_constants(monkeypatch):
    v = np.random.default_rng(3).normal(size=32)
    assert info.info_bits(v).bits > 0
    monkeypatch.setattr(match, "COPY_ATOL", 100)
    assert info.info_bits(v).bits == 0


def test_values_bounded_but_score_uses_complete_segment():
    v = np.random.default_rng(42).normal(size=700)
    record = match.fit_segment(v, 2 * v + 1, 0, len(v), 0)
    assert record["segment"] == record["info_n"] == 700
    assert record["info_bits_target"] == info.info_bits(v).bits
    original = copy.deepcopy(record)
    info.finish_identification({"windows": {}}, [record], {})
    assert record["identification"] == "identifying"
    assert record["values_target"] == v[:512].tolist()
    assert record["values_index"] == (2 * v[:512] + 1).tolist()
    assert record["is_copy"] == original["is_copy"]
    assert len(record["values_target"]) == 512
