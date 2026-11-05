"""Score-C reference equivalence and scan-level identification contracts."""

import copy
import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from demand_forecast.pretrain import audit_info as info
from demand_forecast.pretrain import audit_match as match
from demand_forecast.pretrain.audit_controls import control_cases

FIXTURE = Path(__file__).parents[1] / "fixtures/audit_info/segments.jsonl"
# Wall-clock bound for the 100,000-point cost tests. Local two-side times are 0.05-0.55 s; shared CI runners measured
# about twice that. The bound only has to catch a super-linear regression, which costs minutes at this length.
COST_BOUND_S = 10.0
SEGMENTS = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
LIMITED = {
    "indexed_1100_times_shift+10_fallback": 48,
    "three_levels_seven_runs_dof7_2x+1": 48,
    "four_levels_eight_runs_1.1x+0.5": 64,
    "five_level_staircase_five_runs_x2": 61,
    "intermittent_4_nonzero_of_26_2x+1": 55,
}


@pytest.mark.parametrize("row", SEGMENTS, ids=lambda row: str(row["record_id"]))
def test_reference_score_exact(row, monkeypatch):
    score = info.info_bits(row["values"])
    assert score == _full_score(row["values"], monkeypatch)
    assert type(score.bits) is int
    assert score.bits == row["bits_c"]
    assert score.code == row["code"]
    assert score.strict_ramp == row["ramp"]
    assert (score.n, score.levels, score.runs) == (row["n"], row["k"], row["runs"])


def _full_score(values, monkeypatch):
    # Disabling only the pruning proof evaluates all six codes, as at 71e193d.
    with monkeypatch.context() as full:
        full.setattr(info, "_kt_exceeds_bound", lambda transitions, limit: False)
        return info.info_bits(values)


def test_lazy_kt_matches_full_evaluation_random_short_sequences(monkeypatch):
    rng = np.random.default_rng(20261106)
    for i in range(2400):
        n = int(rng.integers(3, 130))
        k = int(rng.integers(2, 9))
        values = (
            rng.normal(size=n),
            rng.integers(0, k, n),
            np.sort(rng.integers(0, k, n)),
            np.resize(rng.integers(0, k, 7), n),
            np.repeat(rng.integers(0, k, n // 3 + 1), 3)[:n],
            rng.normal(size=n).cumsum(),
            np.rint(np.linspace(0, 200, n)) + rng.integers(-1, 2, n),
            1e6 + rng.normal(size=n),
        )[i % 8]
        assert info.info_bits(values) == _full_score(values, monkeypatch), (i, values)


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
        # Integer comparison opportunities nearest the specified display budget.
        info.finish_identification({"windows": {"exact": round(2**56.84675)}}, copies, {})
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
        "_info_values": (np.array([1.0]), np.array([2.0])),
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
    assert "_info_values" not in record
    assert not info.info_bits(target[both]).strict_ramp
    # Conversely, a ramp only after compression is not an original-position ramp.
    compressed = target.copy()
    compressed[both] = np.arange(both.sum())
    compressed[~both] = np.nan
    rec = match.fit_segment(compressed, compressed.copy(), 0, 48, 0)
    assert rec["is_copy"] and not rec["info_strict_ramp"]
    for values in rec["_info_values"]:
        np.testing.assert_array_equal(values, compressed[both])
        assert values.flags.owndata and values.dtype == np.float64
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


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_values_bounded_but_score_uses_complete_segment(dtype):
    v = np.random.default_rng(42).normal(size=700).astype(dtype)
    record = match.fit_segment(v, 2 * v + 1, 0, len(v), 0)
    assert record["segment"] == record["info_n"] == 700
    assert record["info_bits_target"] == info.info_bits(v).bits
    for values in record["_info_values"]:
        assert values.dtype == np.float64 and values.shape == (512,)
        assert values.flags.owndata and values.base is None
    original = copy.deepcopy(record)
    info.finish_identification({"windows": {}}, [record], {})
    assert record["identification"] == "identifying"
    assert record["values_target"] == v[:512].tolist()
    assert record["values_index"] == (2 * v[:512] + 1).tolist()
    assert record["is_copy"] == original["is_copy"]
    assert len(record["values_target"]) == 512


@pytest.mark.parametrize("n, expected", [(131074, 62), (65538, 60)])
def test_long_piecewise_integer_boundaries(n, expected, monkeypatch):
    # tau=1, q=0,...,0,1,...,200: exactly one nonzero second difference.
    # Piecewise = selector 3 + ceil(log2(n-1)) + ceil(log2 C(n-2,1))
    #             + (1+2)*ceil(log2(201)).
    # n=131074: 3+18+17+24=62; n=65538: 3+17+16+24=60.
    values = np.r_[np.zeros(n - 200), np.arange(1, 201)]
    score = info.info_bits(values)
    assert score == _full_score(values, monkeypatch)
    assert (score.bits, score.code) == (expected, "piecewise_linear")
    record = _record(score.bits)
    info.finish_identification({"windows": {"exact": 2**17}}, [record], {"exact": 2**24})
    assert record["identification"] == ("identifying" if expected >= 61 else "low_information")


def test_scoring_never_calls_floating_logs(monkeypatch):
    def forbidden(*args):
        pytest.fail("floating log on the code-length path")

    with monkeypatch.context() as no_logs:
        no_logs.setattr(math, "lgamma", forbidden)
        no_logs.setattr(math, "log2", forbidden)
        no_logs.setattr(math, "log", forbidden)
        for row in SEGMENTS:
            assert info.info_bits(row["values"]).bits == row["bits_c"]
    # Only the two human-readable search logarithms may use math.log2.
    original = math.log2
    calls = []

    def display_log(n):
        calls.append(n)
        return original(n)

    monkeypatch.setattr(math, "log2", display_log)
    info.finish_identification({"windows": {"exact": 3}}, [_record(22)], {"exact": 5})
    assert calls == [3, 5]


def test_exact_status_beyond_float_precision():
    # All three products round to 2**60 as floats, but only the first two pass.
    for windows, expected in (
        (2**60 - 1, "identifying"),
        (2**60, "identifying"),
        (2**60 + 1, "low_information"),
    ):
        record = _record(80)
        info.finish_identification({"windows": {"exact": windows}}, [record], {"exact": 1})
        assert record["identification"] == expected
    record = _record(19)
    info.finish_identification({"windows": {}}, [record], {})
    assert record["identification"] == "low_information"


def test_exact_kt_against_independent_fraction_products():
    rng = np.random.default_rng(30)
    for k in (2, 3, 4, 5):
        for _ in range(10):
            transitions = rng.integers(0, 8, (k, k))
            ratio = Fraction(k)
            for row in transitions:
                for i in range(int(row.sum())):
                    ratio *= Fraction(k, 2) + i
                for count in row:
                    for i in range(int(count)):
                        ratio /= Fraction(1, 2) + i
            # Independent oracle: walk successive powers, with no logarithms.
            ceiling, power = 0, 1
            while power < ratio:
                power *= 2
                ceiling += 1
            assert info._kt_bits(transitions) == ceiling
            for limit in (-1, 0, ceiling - 1, ceiling, ceiling + 1):
                proved = info._kt_exceeds_bound(transitions, limit)
                if proved:
                    assert ceiling > limit
                if transitions.sum() <= info._KT_BOUND_FACTORS:
                    assert proved == (ceiling > limit)
    assert info._ratio_bits(8, 2) == 2
    assert info._ratio_bits(9, 2) == 3


def test_kt_bound_strict_power_of_two_and_factor_cap(monkeypatch):
    transitions = np.array([[1, 0], [0, 0]])  # inverse probability = 2 * 2 = 4
    assert info._kt_exceeds_bound(transitions, 1)
    assert not info._kt_exceeds_bound(transitions, 2)
    # The same proof cannot conclude after zero factors: use the exact fallback.
    monkeypatch.setattr(info, "_KT_BOUND_FACTORS", 0)
    assert not info._kt_exceeds_bound(transitions, 1)
    values = [0, 1, 1, 1, 1, 1]
    assert info.info_bits(values) == _full_score(values, monkeypatch)


def _brute_period(q):
    return next((p for p in range(1, len(q) // 2 + 1) if np.array_equal(q[p:], q[:-p])), None)


def test_period_exhaustive_binary():
    for n in range(1, 13):
        for values in product((0, 1), repeat=n):
            q = np.asarray(values)
            assert info._smallest_period(q) == _brute_period(q)


def test_period_random_long_and_partial_motifs():
    rng = np.random.default_rng(72)
    for n in (1001, 10000, 100000):
        random = rng.integers(0, 200, n)
        partial = np.resize([0, 3, 1, 8, 2, 7, 4], n)
        almost = partial.copy()
        almost[-1] = 199
        for q in (random, partial, almost):
            assert info._smallest_period(q) == _brute_period(q)
        assert info._smallest_period(partial) == 7  # no n % p requirement


def test_long_aperiodic_both_sides_cost():
    values = np.random.default_rng(0).normal(size=100000)
    started = perf_counter()
    scores = [info.info_bits(v) for v in (values, 2 * values + 1)]
    elapsed = perf_counter() - started
    assert all(s.code == "enumerative" and s.bits > 0 for s in scores)
    assert elapsed < COST_BOUND_S, f"two 100,000-point scores took {elapsed:.3f}s"


@pytest.mark.parametrize("levels, bits", [(200, 108), (201, 61)])
def test_long_skewed_both_sides_cost(levels, bits, monkeypatch):
    values = np.r_[np.zeros(100001 - levels), np.arange(1, levels)]

    def forbidden(transitions):
        pytest.fail("provably losing KT code reached the factorial path")

    monkeypatch.setattr(info, "_kt_bits", forbidden)
    started = perf_counter()
    # Both actual sides of the accepted identity copy from the review.
    scores = [info.info_bits(v) for v in (values, values.copy())]
    elapsed = perf_counter() - started
    assert all((s.bits, s.code, s.levels) == (bits, "piecewise_linear", levels) for s in scores)
    assert elapsed < COST_BOUND_S, f"two {levels}-level 100,000-point scores took {elapsed:.3f}s"
    assert match.fit_segment(values, values, 0, len(values), 0)["is_copy"]


@pytest.mark.parametrize(
    "values, bits, code",
    [
        ([0, 1, 0], 8, "enumerative"),  # enumerative ties run
        ([0, 1, 0, 1], 8, "run"),  # run ties periodic_motif
    ],
)
def test_code_tie_order(values, bits, code):
    score = info.info_bits(values)
    assert (score.bits, score.code) == (bits, code)


@pytest.mark.parametrize(
    "values, bits, code, tied_code",
    [
        ([0, 1, 1, 1, 1, 1], 11, "kt_markov", None),
        ([0, 1, 1, 1, 1], 11, "enumerative", "kt_markov"),
        ([0, 0, 0.02, 0.01, 0.01, 0.02, 0.01, 0.02], 21, "kt_markov", "digital_line"),
    ],
)
def test_lazy_kt_winner_and_ties_use_full_evaluation(values, bits, code, tied_code, monkeypatch):
    original = info._kt_bits
    calls = []

    def full_kt(transitions):
        calls.append(True)
        return original(transitions)

    monkeypatch.setattr(info, "_kt_bits", full_kt)
    score = info.info_bits(values)
    assert calls == [True]  # Neither a KT win nor a tie may be pruned.
    assert (score.bits, score.code) == (bits, code)
    assert score == _full_score(values, monkeypatch)
    if tied_code is not None:
        # Removing the preferred code exposes its equal-cost competitor.
        monkeypatch.setattr(info, "_CODE_ORDER", tuple(c for c in info._CODE_ORDER if c != code))
        tied = info.info_bits(values)
        assert (tied.bits, tied.code) == (bits, tied_code)


def test_documented_six_code_tie_order():
    assert info._CODE_ORDER == (
        "enumerative",
        "run",
        "piecewise_linear",
        "kt_markov",
        "periodic_motif",
        "digital_line",
    )


def test_exact_digital_rounding():
    for denominator in range(1, 30):
        for numerator in range(-100, 101):
            assert info._round_ratio(numerator, denominator) == round(
                Fraction(numerator, denominator)
            )
