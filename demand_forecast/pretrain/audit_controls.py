"""Offline run-11 controls. The former M4 inputs use synthetic surrogates.

The base real-valued series is deterministic and the T002210 surrogate has its
119-point length, three levels and four-run topology, not its original values.
All other original constructions and ten rule expectations are preserved.
"""

import numpy as np
import pandas as pd

from .audit_match import (
    Index,
    calendar_pattern,
    days_in_month_of,
    monthly_aggs,
    nondegenerate,
    scan,
    weekly_sums,
)


def control_cases():
    base = np.random.default_rng(472).normal(80, 15, 200).cumsum()
    # Seven runs have seven information degrees of freedom but no runs8 window.
    # This preserves the 1,100-duplicate control's masked fallback path with
    # synthetic inputs (an arbitrary real-valued surrogate can get runs8 hits).
    fallback_base = np.repeat([3.0, 5.0, 3.0, 7.0, 3.0, 5.0, 3.0], [4, 4, 4, 4, 4, 3, 3])
    # Synthetic 119-point surrogate: three levels, four runs, four degrees of freedom.
    t002210 = np.repeat([3.0, 5.0, 9.0, 3.0], [30, 30, 30, 29])
    rng = np.random.default_rng(0)
    daily = np.abs(rng.standard_normal(1400)).cumsum() + rng.standard_normal(1400) * 3
    d0 = pd.Timestamp("2011-03-02")
    x200 = rng.normal(10, 3, 200)
    x16 = np.concatenate(
        [
            rng.normal(50, 5, 16),
            np.repeat(rng.choice([3.98, 4.48, 5.98, 6.28, 7.5, 8.1], 6, replace=False), 10),
        ]
    )  # 16 varying + 6 constant runs = 76
    binary = np.concatenate([np.full(n, v) for v, n in zip([0, 1] * 20, rng.integers(1, 5, 40))])[
        :120
    ].astype(np.float64)
    offset_series = rng.normal(0, 3, 100) + 1e6
    dim26 = days_in_month_of(pd.date_range("2011-02-28", periods=26, freq="ME"))
    bin30 = np.repeat(np.arange(30) % 2, rng.integers(2, 7, 30)).astype(
        np.float64
    )  # two-level, 30 runs (reviewer's indexed-65 case)
    # (target, [planted ...], expect_copy)
    controls = {
        "verbatim": (base, [base], True),
        "shift+10": (base, [base + 10.0], True),
        "scale_x2": (base, [base * 2.0], True),
        "negate": (base, [-base], True),
        "affine_3x-50": (base, [3.0 * base - 50.0], True),
        "float32_offset_1e4": (base, [np.asarray(np.float32(base + 1e4), np.float64)], True),
        "prefix70_then_divergent": (
            base,
            [
                np.concatenate(
                    [base[:70] * 1.5 + 3, rng.standard_normal(130) * base.std() + base.mean()]
                )
            ],
            True,
        ),
        "truncated_shifted_slice": (base, [base[40:160] + 7.0], True),
        "piecewise_constant_price": (
            (lambda p: (p, [p * 1.1 + 0.5], True))(
                np.repeat(rng.choice([3.98, 4.48, 5.98, 6.28], 40), rng.integers(3, 12, 40))[:200]
            )
        ),
        "weekly_sum_anchor3_vs_daily": (
            weekly_sums(daily, d0, 3),
            [weekly_sums(daily, d0, 3) / 7.0],
            True,
        ),
        "monthly_sum_shift+5": (
            monthly_aggs(daily, d0)[0],
            [monthly_aggs(daily, d0)[0] + 5.0],
            True,
            monthly_aggs(daily, d0)[2],
        ),  # calendar rule active: sums of a VARYING daily series are still a copy
        "monthly_mean_affine_2.5x+1": (
            monthly_aggs(daily, d0)[1],
            [monthly_aggs(daily, d0)[1] * 2.5 + 1.0],
            True,
        ),
        # round-4 reviewer cases
        # a real 119-day M4 series with 3 levels and 4 runs: the shifted copy is FOUND (candidate, exact fit) but carries 4 degrees of freedom -> non-identifying by policy
        "m4_daily_T002210_shift+10_found_low_dof": (
            t002210 + 10.0,
            [t002210],
            False,
            None,
            "low_dof",
        ),
        "varying16_then_6_constant_runs_1.5x+3": (x16, [x16 * 1.5 + 3.0], True),
        "splice_two_maps_70+10_then_2x+20": (
            x200,
            [np.concatenate([x200[:70] + 10.0, 2.0 * x200[70:] + 20.0])],
            True,
        ),
        "three_segments_60copy_60noise_80x3": (
            x200,
            [np.concatenate([x200[:60] + 1.0, rng.normal(10, 3, 60), x200[120:] * 3.0])],
            True,
        ),
        "indexed_65_times_shift+10": (base + 10.0, [base] * 65, True),
        "indexed_1100_times_shift+10_fallback": (
            fallback_base + 10.0,
            [fallback_base] * 1100,
            True,
        ),  # above CAP -> masked -> fallback path
        "offset_1e6_std3_shift+5": (offset_series, [offset_series + 5.0], True),
        "binary_irregular_runs_100x+7": (binary, [binary * 100.0 + 7.0], True),
        "NEG_float32_constant_vs_two_level": (
            (
                lambda noise: (
                    np.asarray(np.float32(134.4) + np.float32(noise), np.float64),
                    [np.where(noise > 0, 500.0, 300.0)],
                    False,
                )
            )(rng.choice([0.0, 1.5e-5], 60))
        ),
        "NEG_two_level_single_step": (
            np.concatenate([np.zeros(12), np.ones(12)]),
            [np.concatenate([np.full(12, 300.0), np.full(12, 500.0)])],
            False,
        ),  # no window in any family: a coverage fact, no candidate
        # run-7 negative controls modelled on the eight run-5 chance matches (constant segments with <= 3 excursions) and a staircase
        "NEG_three_levels_three_excursions_then_constant": (
            np.array([1.0, 2.0, 1.0] + [0.0] * 23),
            [np.array([9800.0, 9700.0, 9800.0] + [9900.0] * 23)],
            False,
            None,
            "off_modal_lt4",
        ),
        "NEG_two_level_two_spikes_4runs": (
            np.array([1.63] * 23 + [1.594, 1.63, 1.594]),
            [np.array([1040.0] * 23 + [1050.0, 1040.0, 1050.0])],
            False,
        ),
        "NEG_binary_two_sparse_spikes": (
            np.array([1.0] + [0.0] * 14 + [1.0] + [0.0] * 11),
            [np.array([5100.0] + [5000.0] * 14 + [5100.0] + [5000.0] * 11)],
            False,
        ),
        "NEG_staircase_three_levels_three_runs_verbatim": (
            np.array([3.0] * 12 + [5.0] * 12 + [9.0] * 8),
            [np.array([3.0] * 12 + [5.0] * 12 + [9.0] * 8)],
            False,
            None,
            "low_dof",
        ),  # exact windows match; 3 degrees of freedom
        # run-8 negative controls: the two run-6 cases that passed the run-7 rules, plus boundary positives
        "NEG_calendar_constant_price_x_days_in_month": (
            2.94 * dim26,
            [100.0 * dim26 - 300.0],
            False,
            dim26,
            "calendar_pattern",
        ),
        "NEG_four_level_staircase_four_runs": (
            np.array([3.44] + [3.68] * 5 + [3.838] + [3.96] * 21),
            [np.array([173.0] + [179.0] * 5 + [183.0] + [186.0] * 21)],
            False,
            None,
            "low_dof",
        ),
        "NEG_three_levels_six_runs_dof6": (
            (lambda s: (s, [2.0 * s + 1.0], False, None, "low_dof"))(
                np.repeat([3.0, 5.0, 3.0, 7.0, 3.0, 5.0], [5, 4, 4, 4, 5, 4]).astype(np.float64)
            )
        ),
        "three_levels_seven_runs_dof7_2x+1": (
            (lambda s: (s, [2.0 * s + 1.0], True))(
                np.repeat([3.0, 5.0, 3.0, 7.0, 3.0, 5.0, 3.0], [4, 4, 4, 4, 4, 3, 3]).astype(
                    np.float64
                )
            )
        ),
        "four_levels_eight_runs_1.1x+0.5": (
            (lambda s: (s, [1.1 * s + 0.5], True))(
                np.repeat(
                    [3.98, 4.48, 3.98, 5.98, 4.48, 6.28, 3.98, 5.98], [4, 3, 3, 4, 3, 3, 3, 3]
                ).astype(np.float64)
            )
        ),
        "five_level_staircase_five_runs_x2": (
            (lambda s: (s, [2.0 * s], True))(
                np.repeat([3.0, 5.0, 9.0, 12.0, 20.0], [6, 6, 5, 5, 6]).astype(np.float64)
            )
        ),
        "NEG_monthly_price_with_one_change_x_days": (
            (lambda s: (s, [10.0 * s + 7.0], False, dim26, "calendar_pattern"))(
                np.concatenate([2.94 * dim26[:12], 3.48 * dim26[12:]])
            )
        ),  # one step beyond the calendar
        "NEG_calendar_rate_step_in_last_month_T032919_shape": (
            np.append(2.94 * dim26[:23], [94.08, 96.04, 106.33]),
            [np.append(100.0 * dim26[:23] - 300.0, [2900.0, 2600.0, 3300.0])],
            False,
            dim26,
            "calendar_pattern",
        ),
        "monthly_rate_three_levels_eight_runs_x_days_10x+7": (
            (lambda s: (s, [10.0 * s + 7.0], True, dim26))(
                np.repeat(
                    [2.94, 3.43, 2.94, 3.98, 2.94, 3.43, 3.98, 2.94], [4, 3, 3, 4, 3, 3, 3, 3]
                )
                * dim26
            )
        ),
        # round-5 reviewer cases
        "splice_x+10_then_1.05x+10_same_coarse_key": (
            x200,
            [np.concatenate([x200[:70] + 10.0, 1.05 * x200[70:] + 10.0])],
            True,
        ),
        "binary_30runs_100x+7_indexed_65": (bin30, [bin30 * 100.0 + 7.0] * 65, True),
        "NEG_single_step_with_float32_jitter": (
            np.asarray(
                np.float32(
                    np.concatenate([np.zeros(12), np.ones(12)]) + rng.choice([0.0, 1e-5], 24)
                ),
                np.float64,
            ),
            [np.concatenate([np.full(12, 300.0), np.full(12, 500.0)])],
            False,
        ),
        "NEG_monthly_price_one_change_x_days_plus7": (
            (lambda s: (s, [10.0 * s + 7.0], False, dim26, "calendar_pattern"))(
                np.concatenate([2.94 * dim26[:12], 3.48 * dim26[12:]]) + 7.0
            )
        ),
        # boundary positive: an intermittent series with exactly 4 points off the modal level (and >= 4 runs) is still a copy
        "intermittent_4_nonzero_of_26_2x+1": (
            (lambda s: (s, [2.0 * s + 1.0], True))(
                np.array(
                    [0.0] * 6 + [3.0] + [0.0] * 5 + [1.0] + [0.0] * 7 + [2.0] + [0.0] * 4 + [5.0]
                )
            )
        ),
    }
    jit_step = np.asarray(
        np.float32(np.concatenate([np.zeros(12), np.ones(12)]) + rng.choice([0.0, 1e-5], 24)),
        np.float64,
    )
    rule_tests = {
        "rule_single_step_float32_jitter_is_low_dof": (nondegenerate(jit_step), (False, "low_dof")),
        "rule_single_step_is_low_dof": (
            nondegenerate(np.concatenate([np.zeros(12), np.ones(12)])),
            (False, "low_dof"),
        ),
        "rule_T002210_is_low_dof": (nondegenerate(t002210), (False, "low_dof")),
        "rule_three_excursions_is_off_modal": (
            nondegenerate(np.array([1.0, 2.0, 1.0] + [0.0] * 23)),
            (False, "off_modal_lt4"),
        ),
        "rule_intermittent_4_nonzero_passes": (
            nondegenerate(
                np.array(
                    [0.0] * 6 + [3.0] + [0.0] * 5 + [1.0] + [0.0] * 7 + [2.0] + [0.0] * 4 + [5.0]
                )
            ),
            (True, None),
        ),
        "rule_calendar_constant_rate": (calendar_pattern(2.94 * dim26, dim26), True),
        "rule_calendar_one_change_plus_offset_7": (
            calendar_pattern(np.concatenate([2.94 * dim26[:12], 3.48 * dim26[12:]]) + 7.0, dim26),
            True,
        ),
        "rule_calendar_T032919_shape": (
            calendar_pattern(np.append(2.94 * dim26[:23], [94.08, 96.04, 106.33]), dim26),
            True,
        ),
        "rule_calendar_three_rates_eight_runs_is_informative": (
            calendar_pattern(
                np.repeat(
                    [2.94, 3.43, 2.94, 3.98, 2.94, 3.43, 3.98, 2.94], [4, 3, 3, 4, 3, 3, 3, 3]
                )
                * dim26,
                dim26,
            ),
            False,
        ),
        "rule_calendar_varying_daily_sums_is_informative": (
            calendar_pattern(
                monthly_aggs(daily, d0)[0][np.isfinite(monthly_aggs(daily, d0)[0])],
                monthly_aggs(daily, d0)[2][np.isfinite(monthly_aggs(daily, d0)[0])],
            ),
            False,
        ),
    }
    return controls, rule_tests


def run_control(name, spec):
    target, planted, expect = spec[:3]
    dims = [spec[3]] if len(spec) > 3 and spec[3] is not None else None
    reason = spec[4] if len(spec) > 4 else None
    idx = Index(
        [np.asarray(p, np.float64) for p in planted], [f"planted{i}" for i in range(len(planted))]
    )
    pairs, records = [], []
    counts = scan(
        "control", name, [np.asarray(target, np.float64)], idx, {}, pairs, records, dims=dims
    )
    detected = any(p["is_copy"] for p in pairs)
    reason_ok = reason is None or any(reason in r["degenerate_reason"] for r in records)
    return {
        "expected_copy": expect,
        "detected_copy": detected,
        "expected_reason": reason,
        "reason_ok": reason_ok,
        "ok": detected == expect and reason_ok,
        "counts": counts,
    }


def run_controls():
    cases, rules = control_cases()
    controls = {name: run_control(name, spec) for name, spec in cases.items()}
    self_tests = {
        name: {"got": got, "expected": want, "ok": got == want}
        for name, (got, want) in rules.items()
    }
    return {
        "controls": controls,
        "self_tests": self_tests,
        "positive_passed": sum(c["ok"] for c in controls.values() if c["expected_copy"]),
        "negative_passed": sum(c["ok"] for c in controls.values() if not c["expected_copy"]),
        "self_tests_passed": sum(c["ok"] for c in self_tests.values()),
    }
