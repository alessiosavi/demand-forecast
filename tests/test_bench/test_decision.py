"""Decision arithmetic and offline CLI contracts on invented inputs."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from demand_forecast.bench import decision

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/fev_decision.py"


def test_relative_clips_both_bounds():
    tasks = ["small", "normal", "large"]
    baseline = dict.fromkeys(tasks, 2.0)
    # 0.002 / 2 = 0.001 -> 0.01; 1 / 2 = 0.5; 400 / 2 = 200 -> 100.
    assert decision.relative(dict(zip(tasks, (0.002, 1.0, 400.0))), baseline, tasks) == {
        "small": 0.01,
        "normal": 0.5,
        "large": 100.0,
    }


@pytest.mark.parametrize("error", [0.0, -1.0, float("inf"), float("nan"), None])
def test_baseline_validated_even_for_unscored_tasks(error):
    baseline = {"scored": 1.0}
    if error is not None:
        baseline["unscored"] = error
    # An invalid or missing denominator fails even where the numerator is missing.
    with pytest.raises(ValueError, match="baseline.*unscored"):
        decision.relative({"scored": 0.5}, baseline, ["scored", "unscored"])


def test_empty_cohort_is_invalid():
    with pytest.raises(ValueError, match="task"):
        decision.relative({}, {}, [])


def test_win_rate_counts_clipped_ties_and_unscored_losses():
    tasks = ["small", "large", "absent", "invalid"]
    baseline = dict.fromkeys(tasks, 1.0)
    panel = [
        dict(zip(tasks, (0.009, 150.0, 0.1, 0.1))),
        dict.fromkeys(tasks, 1.0),
    ]
    system = {"small": 0.001, "large": 200.0, "invalid": float("inf")}
    # Small: 0.5 + 1. Large: 0.5 + 0. Absent and invalid: four losses.
    # (0.5 + 1 + 0.5) / (4 tasks * 2 members) = 0.25.
    assert decision.win_rate(system, panel, baseline, tasks) == 0.25


def test_skill_uses_only_finite_scores():
    tasks = ["a", "b", "absent", "invalid"]
    system = {"a": 1.0, "b": 4.0, "invalid": float("nan")}
    # 1 - sqrt((1 / 8) * (4 / 8)) = 1 - 1/4 = 0.75.
    assert decision.skill(system, dict.fromkeys(tasks, 8.0), tasks) == pytest.approx(0.75)


def test_skill_requires_at_least_one_score():
    with pytest.raises(ValueError, match="scored"):
        decision.skill({"a": float("-inf")}, {"a": 1.0}, ["a"])


def test_firsts_requires_strict_win_against_every_member():
    tasks = ["clear", "tie", "split", "missing"]
    system = dict.fromkeys(tasks[:3], 1.0)
    panel = [dict.fromkeys(tasks, 2.0), dict(zip(tasks, (2.0, 1.0, 0.5, 2.0)))]
    # Clear wins twice. Tie and split each beat one member. Missing wins zero.
    assert decision.firsts(system, panel, dict.fromkeys(tasks, 1.0), tasks) == 1


@pytest.mark.parametrize("member", [{}, {"a": float("nan")}])
@pytest.mark.parametrize("metric", ["win_rate", "firsts"])
def test_panel_needs_a_finite_error_even_when_system_is_unscored(member, metric):
    with pytest.raises(ValueError, match="panel.*a"):
        getattr(decision, metric)({}, [member], {"a": 1.0}, ["a"])


@pytest.mark.parametrize(
    ("w", "s", "f", "scored", "expected"),
    [
        # Skill 0.397 < 0.4 - 0.002, even with both gains.
        pytest.param(0.51, 0.397, 3, "ab", False, id="skill-loss"),
        # The two scores are on a and c, so b is lost despite equal counts.
        pytest.param(0.51, 0.4, 3, "ac", False, id="same-count-different-tasks"),
        # W fell by 0.001, so the extra first cannot qualify.
        pytest.param(0.499, 0.4, 3, "ab", False, id="w-loss"),
        # W rose by 0.004 and F by zero, both below the required gains.
        pytest.param(0.504, 0.4, 2, "ab", False, id="no-required-gain"),
        # S is exactly 0.4 - 0.002, W gains 0.005, and an extra task is fine.
        pytest.param(0.505, 0.398, 2, "abc", True, id="w-gain-at-skill-floor"),
        # W stays at 0.5 and F rises from 2 to 3.
        pytest.param(0.5, 0.4, 3, "ab", True, id="one-more-first"),
    ],
)
def test_eligibility_clauses(w, s, f, scored, expected):
    incumbent = decision.Figures(0.5, 0.4, 2, frozenset("ab"))
    candidate = decision.Figures(w, s, f, frozenset(scored))
    assert decision.eligible(candidate, incumbent) is expected


def test_exact_win_gain_qualifies_at_fractional_rates():
    scored = frozenset("ab")
    incumbent = decision.Figures(40 / 200, 0.4, 2, scored)
    candidate = decision.Figures(41 / 200, 0.4, 2, scored)
    # (41 - 40) / 200 = 0.005, exactly the required gain.
    assert decision.eligible(candidate, incumbent)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        # Only 0.520 is within 0.005 of 0.520. More F outside the window cannot win.
        pytest.param({"more": (0.514, 0.4, 9), "best": (0.520, 0.4, 2)}, "best", id="w-window"),
        # The lowest score inside the window is 0.520 - 0.005 = 0.515. Three F beats two.
        pytest.param({"best": (0.520, 0.4, 2), "more": (0.515, 0.4, 3)}, "more", id="most-firsts"),
        # Three F each, so S 0.42 beats 0.41 even with lower W inside the window.
        pytest.param(
            {"best": (0.520, 0.41, 3), "skilled": (0.516, 0.42, 3)}, "skilled", id="skill"
        ),
        # Exactly 0.005 better W qualifies. Given equal F and S the incumbent stays.
        pytest.param({"a": (0.505, 0.4, 2)}, "incumbent", id="incumbent"),
        # Equal F and S, with no incumbent in the window. "a" sorts before "z".
        pytest.param({"z": (0.521, 0.4, 3), "a": (0.518, 0.4, 3)}, "a", id="name"),
        # S 0.3 is below 0.398, so W 0.9 cannot set the eligible W window.
        pytest.param({"bad": (0.9, 0.3, 9), "ok": (0.51, 0.4, 3)}, "ok", id="eligible-window"),
    ],
)
def test_choose_order(rows, expected):
    scored = frozenset("ab")
    incumbent = ("incumbent", decision.Figures(0.5, 0.4, 2, scored))
    candidates = {name: decision.Figures(*figures, scored) for name, figures in rows.items()}
    assert decision.choose(incumbent, candidates) == expected


def test_win_window_keeps_boundary_at_fractional_rates():
    scored = frozenset("ab")
    incumbent = ("incumbent", decision.Figures(4 / 200, 0.4, 2, scored))
    candidates = {
        "best_w": decision.Figures(7 / 200, 0.4, 2, scored),
        "most_firsts": decision.Figures(6 / 200, 0.4, 3, scored),
    }
    # (7 - 6) / 200 = 0.005. Both qualify and three F beats two.
    assert decision.choose(incumbent, candidates) == "most_firsts"


def test_family_merges_prefixes_and_strips_variant_suffix():
    # Both names start with the merged prefix. The last name loses /variant and _12W-MON.
    assert decision.family("solar_east_1D") == decision.family("solar_west_2H") == "solar"
    assert decision.family("invented_12W-MON/variant") == "invented"


def test_paired_draws_use_sorted_families_with_task_weighting():
    tasks = ["c", "a_1D", "a_1W", "b"]
    baseline = dict.fromkeys(tasks, 10.0)
    a = dict(zip(tasks, (80.0, 10.0, 40.0, 20.0)))
    draws = [[0, 0, 0], [0, 0, 2], [0, 1, 2], [2, 2, 2]]
    result = decision.paired_ratio(a, baseline, baseline, tasks, draws=draws)
    # Sorted families: a has ratios (1, 4), b has 2, c has 8.
    # Full ratio: (1 * 4 * 2 * 8) ** (1/4) = 2 ** 1.5.
    # Draw ratios: 2, (1 * 4 * 1 * 4 * 8) ** (1/5) = 2 ** (7/5), 2 ** 1.5, 8.
    # Four sorted draws give percentile positions 0.05 * 3 = 0.15 and 0.95 * 3 = 2.85.
    assert result.families == 3
    assert result.ratio == pytest.approx(2**1.5)
    assert result.low == pytest.approx(2 + 0.15 * (2 ** (7 / 5) - 2))
    assert result.high == pytest.approx(8 - 0.15 * (8 - 2**1.5))


def test_paired_ratio_clips_each_relative_error():
    # A/B is 0.05 raw. Relative A is 0.001 -> 0.01 and relative B is 0.02.
    # The clipped ratio is therefore 0.01 / 0.02 = 0.5.
    result = decision.paired_ratio({"a": 1.0}, {"a": 20.0}, {"a": 1000.0}, ["a"])
    assert result.ratio == pytest.approx(0.5)


def test_paired_ratio_is_reproducible():
    tasks = ["x", "y", "z"]
    baseline = dict.fromkeys(tasks, 1.0)
    a = dict(zip(tasks, (1.0, 2.0, 4.0)))
    first = decision.paired_ratio(a, baseline, baseline, tasks, resamples=31, seed=17)
    second = decision.paired_ratio(a, baseline, baseline, tasks, resamples=31, seed=17)
    # (1 * 2 * 4) ** (1/3) = 2. Identical seeds must give identical draws and endpoints.
    assert first.ratio == pytest.approx(2.0)
    assert first.low is not None and first.high is not None
    assert first == second


def test_paired_ratio_uses_only_shared_scores_and_needs_three_families_for_interval():
    tasks = ["a", "b", "c", "d"]
    a = {"a": 2.0, "b": 8.0, "c": 4.0, "d": float("nan")}
    b = {"a": 1.0, "b": 1.0, "d": 1.0}
    result = decision.paired_ratio(a, b, dict.fromkeys(tasks, 1.0), tasks)
    # Only a and b are shared and finite. sqrt(2 * 8) = 4, across two families.
    assert result.ratio == pytest.approx(4.0)
    assert (result.families, result.low, result.high) == (2, None, None)


def test_paired_ratio_rejects_disjoint_scores():
    with pytest.raises(ValueError, match="shared"):
        decision.paired_ratio({"a": 1.0}, {"b": 1.0}, {"a": 1.0, "b": 1.0}, ["a", "b"])


@pytest.mark.parametrize("draws", [[[0, 1]], [[0, 1, -1]], [[0.0, 1.0, 2.0]]])
def test_paired_ratio_rejects_invalid_family_indices(draws):
    # Three draws per row, using integer indices 0 through 2 only.
    baseline = dict.fromkeys(["a", "b", "c"], 1.0)
    with pytest.raises(ValueError, match="draws"):
        decision.paired_ratio(baseline, baseline, baseline, list(baseline), draws=draws)


def test_paired_ratio_checks_draws_even_when_too_few_families_give_no_interval():
    # Two families (a and b): no interval is computed, but a malformed matrix is still an error.
    baseline = dict.fromkeys(["a", "b"], 1.0)
    with pytest.raises(ValueError, match="draws"):
        decision.paired_ratio(baseline, baseline, baseline, list(baseline), draws=[[0, 5]])


def test_slices_accept_a_single_target_name_and_no_targets():
    def task(name, targets):
        return SimpleNamespace(
            task_name=name,
            target_columns=targets,
            known_dynamic_columns=(),
            past_dynamic_columns=(),
            static_columns=(),
        )

    # A bare string is one target, not one target per character; None is no target.
    result = decision.slices(
        [task("one", "unit_sales"), task("none", None), task("two", ["a", "b"])]
    )
    assert result["multi_target"] == ["two"]
    assert result["all"] == ["one", "none", "two"]


def test_slices_overlap_on_six_tasks():
    def task(name, *, known=(), past=(), static=(), targets=("sales",)):
        return SimpleNamespace(
            task_name=name,
            target_columns=targets,
            known_dynamic_columns=known,
            past_dynamic_columns=past,
            static_columns=static,
        )

    tasks = [
        task("plain"),
        task("multi", targets=("sales", "orders")),
        task("known", known=("promo",)),
        task("past", past=("oil",)),
        task("static", static=("store",)),
        task(
            "mixed", known=("promo",), past=("oil",), static=("store",), targets=("sales", "orders")
        ),
    ]
    # Features are the union of known, past, and static columns. Targets are independent.
    assert decision.slices(tasks) == {
        "all": ["plain", "multi", "known", "past", "static", "mixed"],
        "without_features": ["plain", "multi"],
        "with_features": ["known", "past", "static", "mixed"],
        "known_future": ["known", "mixed"],
        "past_only": ["past", "mixed"],
        "static": ["static", "mixed"],
        "multi_target": ["multi", "mixed"],
    }


@pytest.fixture
def cli(tmp_path):
    pytest.importorskip("fev")
    files = {
        "names": "walk\nroll\n",
        "tasks.yaml": """tasks:
- name: walk
  dataset_path: ./unused.parquet
  horizon: 1
- task_name: roll
  dataset_path: ./unused.parquet
  horizon: 1
  known_dynamic_columns: [promo]
- task_name: outside_cohort
  dataset_path: ./unused.parquet
  horizon: 1
""",
        "baseline.csv": "task,test_error\nwalk,8\nroll,8\n",
        "panel.csv": "task_name,test_error\nwalk,4\nroll,4\n",
        "incumbent.csv": "task,test_error\nwalk,4\nroll,4\n",
        "candidate.csv": "task_name,test_error,model_name\nwalk,2,fresh\nroll,2,fresh\n",
    }
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    def run(*extra):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "names",
                "baseline.csv",
                "tasks.yaml",
                "--panel",
                "reference=panel.csv",
                "--incumbent",
                "incumbent=incumbent.csv",
                "--candidate",
                "candidate=candidate.csv",
                *extra,
            ],
            cwd=tmp_path,
            env={
                **os.environ,
                "PYTHONPATH": str(SCRIPT.parents[1]),
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
            capture_output=True,
            text=True,
            check=False,
        )

    return run


@pytest.mark.bench
def test_cli_writes_figures_and_empty_slices(cli, tmp_path):
    result = cli("--against", "reference=panel.csv", "--require-complete", "--out", "out")
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "out/decision.json").read_text(encoding="utf-8"))
    all_slice = report["slices"]["all"]
    # The incumbent ties both. It has W=(0.5+0.5)/2, S=1-4/8, and no firsts.
    assert all_slice["figures"]["incumbent"] == {
        "w": 0.5,
        "s": 0.5,
        "f": 0,
        "scored": ["roll", "walk"],
    }
    # Candidate wins both, S=1-2/8, and both tasks are strict firsts.
    assert all_slice["figures"]["candidate"] == {
        "w": 1.0,
        "s": 0.75,
        "f": 2,
        "scored": ["roll", "walk"],
    }
    assert report["eligibility"] == {"candidate": True}
    assert report["choice"] == "candidate"
    # Both paired ratios are (2/8)/(4/8) = 0.5. Two families cannot form an interval.
    assert all_slice["paired_ratios"]["candidate"] == {
        "ratio": 0.5,
        "low": None,
        "high": None,
        "families": 2,
    }
    # Only roll has known columns. Neither selected task has static columns.
    assert report["slices"]["known_future"]["tasks"] == ["roll"]
    assert report["slices"]["static"]["tasks"] == []
    assert "static (empty)" in result.stdout
    assert "runtime criterion is unavailable" in result.stdout
    assert (tmp_path / "out/decision.md").read_text(encoding="utf-8") == result.stdout


@pytest.mark.bench
def test_cli_requires_label_for_ambiguous_rows(cli, tmp_path):
    with (tmp_path / "candidate.csv").open("a", encoding="utf-8") as handle:
        handle.write("walk,12,old\nroll,12,old\n")
    result = cli()
    assert result.returncode == 2
    assert "candidate.csv" in result.stderr and "--model-name" in result.stderr
    # A label that matches no row is an error too, not an empty system.
    missing = cli("--model-name", "candidate=nope")
    assert missing.returncode == 2
    assert "candidate.csv" in missing.stderr and "nope" in missing.stderr
    # Selecting fresh gives 2 < 4 twice. Without filtering, the old rows would give 12 > 4 twice.
    chosen = cli("--model-name", "candidate=fresh", "--out", "out")
    assert chosen.returncode == 0, chosen.stderr
    report = json.loads((tmp_path / "out/decision.json").read_text(encoding="utf-8"))
    assert report["slices"]["all"]["figures"]["candidate"]["w"] == 1.0


@pytest.mark.bench
@pytest.mark.parametrize(
    ("name", "rows"),
    [
        ("candidate", "task_name,test_error\nwalk,2\nroll,nan\n"),
        ("incumbent", "task,test_error\nwalk,2\n"),
    ],
)
def test_cli_require_complete_flags_missing_or_nonfinite_row(cli, tmp_path, name, rows):
    (tmp_path / f"{name}.csv").write_text(rows, encoding="utf-8")
    # Neither a non-finite nor an absent roll row supplies a score.
    result = cli("--require-complete")
    assert result.returncode == 2
    assert name in result.stderr and "roll" in result.stderr


@pytest.mark.bench
def test_cli_reports_partial_slices_without_scores(cli, tmp_path):
    (tmp_path / "candidate.csv").write_text("task,test_error\nwalk,2\n", encoding="utf-8")
    result = cli("--against", "reference=panel.csv", "--out", "out")
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "out/decision.json").read_text(encoding="utf-8"))
    # Candidate W = (1+0)/2, S = 1-2/8, F = 1, but it loses roll and is ineligible.
    assert report["slices"]["all"]["figures"]["candidate"] == {
        "w": 0.5,
        "s": 0.75,
        "f": 1,
        "scored": ["walk"],
    }
    assert report["eligibility"] == {"candidate": False}
    assert report["choice"] == "incumbent"
    # The feature slice contains only roll. Candidate has zero wins and firsts,
    # with no score for its geometric mean and no overlap for its paired ratio.
    featured = report["slices"]["with_features"]
    assert featured["figures"]["candidate"] == {"w": 0.0, "s": None, "f": 0, "scored": []}
    assert featured["paired_ratios"]["candidate"] is None
