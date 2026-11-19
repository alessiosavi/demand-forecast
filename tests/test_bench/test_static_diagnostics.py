"""Stage D contracts on invented panels and fev's real history splits."""

import csv
import json
from collections import Counter
from dataclasses import replace
from zlib import crc32

import numpy as np
import pandas as pd
import pytest

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from datasets import ClassLabel, Dataset, Sequence, Value  # noqa: E402

from demand_forecast.bench.v5_panels import Part, WindowPanels, describe, window_panels  # noqa: E402
from demand_forecast.bench.yardsticks.runner import Options, prepare_windows  # noqa: E402
from demand_forecast.data.episode.panel_stats import PanelStatsProvider  # noqa: E402
from demand_forecast.data.episode.schema import Item, Panel  # noqa: E402
from demand_forecast.data.episode.screening import ScreeningConfig, screen  # noqa: E402
from tests.helpers.v5_sampler import paired_static_panel  # noqa: E402


@pytest.fixture
def diagnostics():
    from demand_forecast.bench import static_diagnostics

    return static_diagnostics


def make_part(statics, *, lengths=None, values=None, starts=None, dynamic=False):
    size = len(statics)
    lengths = lengths or [24] * size
    starts = starts or [0] * size
    values = values if values is not None else range(size)
    items = []
    for i, (static, length, level, start) in enumerate(
        zip(statics, lengths, values, starts, strict=True)
    ):
        y = np.arange(length, dtype=float) + level
        columns = {"plan": np.r_[y, np.nan], "empty": np.full(length + 1, np.nan)}
        items.append(
            Item(str(i), start, {"target": np.r_[y, np.nan]}, columns if dynamic else {}, static)
        )
    roles = {"plan": "known", "empty": "past"} if dynamic else {}
    panel = Panel("toy", tuple(items), roles, "D", pd.Timestamp("2024-01-01"), True)
    return Part(panel, tuple(range(size)), tuple(np.add(starts, lengths).tolist()), False)


def measured(diagnostics, monkeypatch, part, *, windows=None):
    windows = (
        windows if windows is not None else [(0, WindowPanels((part,), {}, len(set(part.origins))))]
    )
    monkeypatch.setattr(diagnostics, "_history_panels", lambda _: ((i, w, {}) for i, w in windows))
    task = fev.Task(dataset_path="in-memory", task_name="toy", num_windows=len(windows))
    return diagnostics.diagnose_task(task)["windows"][0]


def test_window_items_target_counts_and_history_lengths(diagnostics, monkeypatch):
    part = make_part([{}, {}, {}], lengths=[3, 16, 21], starts=[0, 8, 1])
    items = tuple(
        replace(item, targets={**item.targets, "other": item.targets["target"]})
        for item in part.panel.items
    )
    part = replace(part, panel=replace(part.panel, items=items))
    row = measured(diagnostics, monkeypatch, part)
    assert (row["items"], row["targets_per_item"], row["short_items"]) == (3, 2, 1)
    assert (row["history_min"], row["history_max"]) == (3, 21)
    assert row["history_median"] == pytest.approx(16)


def test_grid_reasons_and_origin_budget(diagnostics, monkeypatch):
    part = make_part([{}])
    solo = replace(part, alone=True, panel=replace(part.panel, epoch=None, aligned=False))
    windows = [
        (
            0,
            WindowPanels(
                (solo, replace(solo, positions=(1,))),
                {0: "history_gap", 1: "over_origin_budget"},
                17,
            ),
        )
    ]
    row = measured(diagnostics, monkeypatch, part, windows=windows)
    assert (row["on_grid"], row["off_grid"], row["distinct_origins"]) == (1, 1, 17)
    assert row["over_origin_budget"] is True
    assert row["grid_reasons"] == {"history_gap": 1, "over_origin_budget": 1}


def test_static_kinds_missingness_group_sizes_and_support(diagnostics, monkeypatch):
    cells = [
        {"x": ["a", "b", "a"], "numeric": 0.1},
        {"x": ["b"], "numeric": 1.1},
        {"x": ["a"], "numeric": 2.1},
        {"x": ["a"], "numeric": 3.1},
        {"x": ["a"], "numeric": None},
        {"x": [None], "numeric": 4.1},
        {"x": ["c"], "numeric": 5.1},
        {"x": None, "numeric": 6.1},
    ]
    row = measured(diagnostics, monkeypatch, make_part(cells))["attributes"]
    x = row["x"]
    assert (x["kind"], x["labels"], x["label_size_min"], x["label_size_max"]) == (
        "multilabel",
        3,
        1,
        4,
    )
    assert x["missing_share"] == pytest.approx(2 / 8)
    assert x["label_size_median"] == pytest.approx(2)
    assert x["share_at_least_2"] == pytest.approx(5 / 8)
    assert x["share_at_least_4"] == pytest.approx(4 / 8)
    for size in (8, 16, 32):
        assert x[f"share_at_least_{size}"] == pytest.approx(0)
    assert row["numeric"]["kind"] == "number"
    assert row["numeric"]["missing_share"] == pytest.approx(1 / 8)


def test_static_relationships_and_builder_removals(diagnostics, monkeypatch):
    cells = [
        {
            "a_group": ["a", "a", "b", "b"][i],
            "b_copy": ["a", "a", "b", "b"][i],
            "c_parent": "same",
            "d_identifier": str(i),
            "e_unshared": [str(i), str(i + 4)] if i else None,
            "f_missing": [None],
        }
        for i in range(4)
    ]
    result = measured(diagnostics, monkeypatch, make_part(cells))
    attributes = result["attributes"]
    assert attributes["c_parent"]["constant"] is True
    assert attributes["d_identifier"]["identifier"] is True
    assert attributes["e_unshared"]["unshared"] is True
    assert attributes["a_group"]["copied_from"] == ["b_copy"]
    assert "a_group" in attributes["c_parent"]["determined_by"]
    assert "c_parent" not in attributes["a_group"]["determined_by"]
    assert {row["attribute"]: row["removed"] for row in result["scores"]} == {
        "a_group": None,
        "b_copy": "duplicate",
        "c_parent": None,
        "d_identifier": "entity",
        "e_unshared": "unshared",
        "f_missing": "unshared",
    }


def test_groups_of_equal_statics_and_equal_target_histories(diagnostics, monkeypatch):
    part = make_part([{"area": "a"}, {"area": "a"}, {"area": "b"}], values=[1, 2, 1])
    row = measured(diagnostics, monkeypatch, part)
    assert row["equal_static_groups"] == [["0", "1"]]
    assert row["equal_target_groups"] == [["0", "2"]]


def test_dynamic_column_role_kind_missingness_and_relevance(diagnostics, monkeypatch):
    row = measured(diagnostics, monkeypatch, make_part([{}] * 3, dynamic=True))
    assert row["dynamic"]["plan"] == {
        "role": "known",
        "kinds": ["number"],
        "missing_share": pytest.approx(0),
        "score": pytest.approx(1),
    }
    assert row["dynamic"]["empty"] == {
        "role": "past",
        "kinds": ["number"],
        "missing_share": pytest.approx(1),
        "score": pytest.approx(0),
    }


def test_cutoff_ties_prefer_later_origins(diagnostics):
    part = make_part([{"g": "group"}] * 7, lengths=[20, 21, 22, 23, 20, 21, 25])
    rows = diagnostics.attribute_scores(part, name="toy", shuffles=0)
    assert [r["cutoff"] for r in rows] == [20, 21, 25]


def test_shuffled_scores_are_seeded_and_leave_other_inputs(diagnostics):
    size = 48
    nuisance = np.random.default_rng(675).permutation(size)
    cells = [
        {"informative": None if i == 0 else f"group{i // 24}", "noise": nuisance[i] + 0.5}
        for i in range(size)
    ]
    part = make_part(cells, values=[i // 24 * 100 for i in range(size)], dynamic=True)
    kwargs = {"name": "toy-ñ", "window": 2, "seed": 19}
    rows = diagnostics.attribute_scores(part, **kwargs)
    assert rows == diagnostics.attribute_scores(part, **kwargs)
    info, noise = rows
    assert info["score"] == pytest.approx(1)
    assert info["score"] > max(info["shuffled_scores"])
    assert noise["score"] <= max(noise["shuffled_scores"])
    assert len(info["shuffled_scores"]) == 20
    order = np.random.default_rng([19, crc32("toy-ñ".encode()), 2, crc32(b"informative"), 0])
    permuted = order.permutation(size)
    items = tuple(
        replace(item, static={**item.static, "informative": cells[j]["informative"]})
        for item, j in zip(part.panel.items, permuted, strict=True)
    )
    panel = replace(part.panel, items=items)
    stats = PanelStatsProvider(panel).get("target", 24)
    index = next(i for i, item in enumerate(items) if item.static["informative"] is not None)
    expected = next(
        c.score
        for c in screen(
            panel, stats, index, ("target",), 24, ScreeningConfig(fair_static_spread=True)
        )
        if c.name == "informative"
    )
    assert info["shuffled_scores"][0] == pytest.approx(expected)
    assert part.panel.items[0].static["informative"] is None
    np.testing.assert_array_equal(part.panel.items[0].dynamic["plan"], np.r_[np.arange(24), np.nan])


def test_diagnostics_calibrate_observed_and_shuffled_paired_categories(diagnostics):
    panel = replace(paired_static_panel(), freq="D", epoch=pd.Timestamp("2024-01-01"))
    part = Part(panel, tuple(range(len(panel.items))), (48,) * len(panel.items), False)
    stats = PanelStatsProvider(panel).get("y0", 48)
    legacy = next(c.score for c in screen(panel, stats, 0, ("y0",), 48) if c.name == "pair")
    (row,) = diagnostics.attribute_scores(part, name="pairs", shuffles=4)
    assert legacy == pytest.approx(1.0)
    assert row["score"] < legacy
    assert len(row["shuffled_scores"]) == 4
    assert max(row["shuffled_scores"]) < legacy


def test_static_score_uses_largest_of_first_sixteen_sorted_targets(diagnostics):
    part = make_part([{"g": str(i // 6)} for i in range(12)], values=[0] * 12)
    items = []
    for i, item in enumerate(part.panel.items):
        base = item.targets["target"]
        targets = {
            f"t{j:02}": base + (100 * (i // 6) if j == 16 else 0) for j in reversed(range(17))
        }
        items.append(replace(item, targets=targets))
    part = replace(part, panel=replace(part.panel, items=tuple(items)))
    assert diagnostics.attribute_scores(part, name="toy", shuffles=0)[0]["score"] == pytest.approx(
        0
    )
    items = [replace(item, targets={**item.targets, "t15": item.targets["t16"]}) for item in items]
    part = replace(part, panel=replace(part.panel, items=tuple(items)))
    assert diagnostics.attribute_scores(part, name="toy", shuffles=0)[0]["score"] == pytest.approx(
        1
    )


def test_no_evidence_level(diagnostics):
    def row(scores, *, window_items=8, panel_items=8, removed=None):
        return {
            "shuffled_scores": scores,
            "window_items": window_items,
            "panel_items": panel_items,
            "removed": removed,
        }

    assert diagnostics.no_evidence_level([row([1] * 199)]) == pytest.approx(0)
    # Index 189.05 mixes 0.121 and 0.921 into 0.161.
    rows = [
        row([0.121] * 190 + [0.921] * 10),
        row([1] * 200, window_items=7),
        row([1] * 200, window_items=10, panel_items=2),
        row([1] * 200, removed="duplicate"),
    ]
    assert diagnostics.no_evidence_level(iter(rows)) == pytest.approx(0.17)
    assert diagnostics.no_evidence_level([row([0.995] * 200)]) == pytest.approx(0.99)


@pytest.mark.parametrize(("panel_items", "pooled_scores"), [(2, 0), (8, 20)])
def test_no_evidence_level_uses_dated_part(
    diagnostics, monkeypatch, tmp_path, panel_items, pooled_scores
):
    dated = make_part([{"group": "same"}] * panel_items)
    solo = make_part([{}])
    solos = tuple(
        replace(
            solo,
            panel=replace(solo.panel, items=(item,), epoch=None, aligned=False),
            positions=(position,),
            alone=True,
        )
        for position, item in enumerate(make_part([{}] * 10).panel.items[panel_items:], panel_items)
    )
    panels = WindowPanels(
        (dated, *solos), {position: "history_gap" for position in range(panel_items, 10)}, 1
    )
    monkeypatch.setattr(diagnostics, "_history_panels", lambda _: iter([(0, panels, {})]))
    task = fev.Task(dataset_path="in-memory", task_name="toy", num_windows=1)
    diagnostics.write_run([task], tmp_path)
    with (tmp_path / "attributes.csv").open(newline="") as stream:
        (row,) = csv.DictReader(stream)
    assert (row["window_items"], row["panel_items"], row["removed"]) == ("10", str(panel_items), "")
    level = json.loads((tmp_path / "level.json").read_text())
    assert level["pooled_scores"] == pooled_scores
    assert level["rule"]["min_window_items"] == level["rule"]["min_panel_items"] == 8


@pytest.mark.parametrize(("score", "expected"), [(0.07, 0.07), (0.29, 0.29), (0.0700001, 0.08)])
def test_no_evidence_level_rounding(diagnostics, score, expected):
    rows = [
        {"shuffled_scores": [score] * 200, "window_items": 8, "panel_items": 8, "removed": None}
    ]
    assert diagnostics.no_evidence_level(rows) == pytest.approx(expected)


def in_memory_task(*, cutoff=24, windows=1, changed_after=None, entrant=False, name="toy"):
    series = []
    for i in range(8 + int(entrant)):
        dates = pd.date_range("2024-01-22" if i == 8 else "2024-01-01", periods=40, freq="D")
        values = np.arange(40, dtype=float) + i * 15
        if changed_after is not None:
            values[changed_after:] = -3 * values[changed_after:] - 91
        series.append(
            dict(
                id=str(i),
                timestamp=list(dates),
                target=values.tolist(),
                sales=(values * 2).tolist(),
                plan=(values * 3).tolist(),
                group=i // 4,
            )
        )
    data = Dataset.from_list(series).cast_column("group", ClassLabel(names=["a", "b", "c"]))
    data = data.cast_column("plan", Sequence(Value("float64"))).with_format("numpy")
    task = fev.Task(
        dataset_path="in-memory",
        task_name=name,
        initial_cutoff=cutoff,
        horizon=2,
        num_windows=windows,
        window_step_size=4,
        min_context_length=4,
        static_columns=["group"],
        known_dynamic_columns=["plan"],
        past_dynamic_columns=["sales"],
    )
    task._full_dataset = data
    task._freq = pd.DatetimeIndex(series[0]["timestamp"], freq="infer").freqstr
    return task


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("regular", {}),
        ("irregular_first_series", {"missing_frequency": 2}),
        ("horizon_gap", {"future_off_grid": 1}),
        ("late_horizon", {"future_off_grid": 1}),
        ("two_daily_items_two_steps", {}),
        ("over_origin_budget", {"over_origin_budget": 17}),
    ],
)
def test_grid_columns_equal_what_scoring_sees(diagnostics, case, reason):
    cutoff = 2 if case == "two_daily_items_two_steps" else 24
    days = pd.date_range("2024-01-01", periods=60, freq="D")
    dates = [days[:40], days[:40]]
    if case == "irregular_first_series":
        dates[0] = dates[0].delete(35)
    elif case == "horizon_gap":
        dates[1] = dates[1].delete(cutoff + 1)
    elif case == "late_horizon":
        dates[1] = dates[1].delete(cutoff)
    elif case == "over_origin_budget":
        dates = [days[start : start + 40] for start in range(17)]
    data = Dataset.from_list(
        [
            {"id": f"{i:02}", "timestamp": list(index), "target": list(range(len(index)))}
            for i, index in enumerate(dates)
        ]
    ).cast_column("timestamp", Sequence(Value("timestamp[s]")))
    task = fev.Task(
        dataset_path="in-memory",
        task_name="toy",
        initial_cutoff=cutoff,
        horizon=2,
        num_windows=3,
        window_step_size=4,
        min_context_length=1,
    )
    task._full_dataset = data.with_format("numpy")
    # Supplying a cached dataset bypasses fev's inference on the first full series.
    task._freq = pd.DatetimeIndex(dates[0], freq="infer").freqstr
    if case == "irregular_first_series":
        with pytest.raises(ValueError):
            _ = task.freq
    else:
        assert task.freq == "D"
    scoring = prepare_windows(task, Options(model="v5", weights="", model_name="v5"))
    results = diagnostics.diagnose_task(task)["windows"]
    for index, (window, row) in enumerate(zip(scoring, results, strict=True)):
        panels = window_panels(window, name=task.task_name, horizon=task.horizon, known=[], past=[])
        counts = describe([panels])
        reasons = dict(Counter(panels.reasons.values()))
        if index == 0:
            assert reasons == reason
        expected = {
            "on_grid": len(window.ids) - counts["v5_items_off_grid"],
            "off_grid": counts["v5_items_off_grid"],
            "items_alone": counts["v5_items_alone"],
            "items_without_time_axis": counts["v5_items_without_time_axis"],
            "grid_reasons": reasons,
            "distinct_origins": panels.distinct_origins,
            "over_origin_budget": bool(counts["v5_windows_over_origin_budget"]),
        }
        assert {key: row[key] for key in expected} == expected


def test_no_output_depends_on_values_at_or_after_a_cutoff(diagnostics, monkeypatch, tmp_path):
    original = in_memory_task(windows=3)
    changed = in_memory_task(windows=3)
    # Changes after an early cutoff would be legal history in a later window.
    # Give each window its own independently modified copy of the full series.
    variants = [in_memory_task(windows=3, changed_after=cutoff) for cutoff in original.cutoffs]
    monkeypatch.setattr(
        changed, "get_window", lambda i, **kwargs: variants[i].get_window(i, **kwargs)
    )
    split = fev.EvaluationWindow.get_input_data
    read_cutoffs = set()

    class TimestampOnlyFuture:
        def __init__(self, window, future):
            self.window = window
            self.future = future

        def __getitem__(self, column):
            assert column == self.window.timestamp_column
            read_cutoffs.add(self.window.cutoff)
            return self.future[column]

        def __iter__(self):
            pytest.fail("diagnostics read a future row")

    def guarded_split(window):
        past, future = split(window)
        return past, TimestampOnlyFuture(window, future)

    monkeypatch.setattr(fev.EvaluationWindow, "get_input_data", guarded_split)
    a, b = tmp_path / "original", tmp_path / "changed"
    for task, path in ((original, a), (changed, b)):
        for (index, panels), cutoff in zip(
            diagnostics.past_panels(task), task.cutoffs, strict=True
        ):
            assert task.cutoffs[index] == cutoff
            item = panels.parts[0].panel.items[0]
            assert item.static["group"] == "a"
            for values in (item.targets["target"], item.dynamic["plan"], item.dynamic["sales"]):
                assert values.shape == (cutoff + 1,)
                assert np.isnan(values[-1])
        diagnostics.write_run([task], path)
    assert read_cutoffs == set(original.cutoffs)
    assert {p.name for p in a.iterdir()} == {
        "toy.json",
        "attributes.csv",
        "tasks.csv",
        "level.json",
    }
    assert {p.name: p.read_bytes() for p in a.iterdir()} == {
        p.name: p.read_bytes() for p in b.iterdir()
    }


def test_a_later_entrant_is_counted_in_its_window(diagnostics):
    task = in_memory_task(windows=3, entrant=True)
    task.initial_cutoff = "2024-01-24"
    task.window_step_size = "4D"
    rows = diagnostics.diagnose_task(task)["windows"]
    assert [r["items"] for r in rows] == [8, 9, 9]
    assert [r["new_items"] for r in rows] == [0, 1, 0]
    assert [len(r["scores"][0]["shuffled_scores"]) for r in rows] == [20, 0, 20]


def test_command_line_only_opens_selected_tasks(diagnostics, monkeypatch, tmp_path):
    task_by_name = {name: in_memory_task(name=name) for name in ("toy", "other")}
    constructed = []

    def constructor(**values):
        name = values["task_name"]
        constructed.append(name)
        return task_by_name[name]

    # Keep real window splitting while replacing task construction at the YAML boundary.
    monkeypatch.setattr(fev, "Task", constructor)
    yaml_file = tmp_path / "tasks.yaml"
    yaml_file.write_text(
        "tasks:\n- name: other\n  dataset_path: fake\n- name: toy\n  dataset_path: fake\n"
    )
    names = tmp_path / "names.txt"
    names.write_text("toy\n")
    out = tmp_path / "selected"
    diagnostics.main(["--tasks", str(yaml_file), "--names", f"@{names}", "--out", str(out)])
    assert constructed == ["toy"]
    with (out / "attributes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(r["task"], r["window"], r["cutoff"], r["attribute"]) for r in rows] == [
        ("toy", "0", "24", "group")
    ]
    level = json.loads((out / "level.json").read_text())
    assert (level["pooled_scores"], level["tasks"], level["windows"], level["attributes"]) == (
        20,
        1,
        1,
        1,
    )
    assert level["screen_no_evidence"] == pytest.approx(0)
    constructed.clear()
    out = tmp_path / "all"
    diagnostics.main(["--tasks", str(yaml_file), "--out", str(out)])
    assert sorted(constructed) == ["other", "toy"]
    with (out / "tasks.csv").open(newline="") as stream:
        assert [r["task"] for r in csv.DictReader(stream)] == ["other", "toy"]


def test_unnamed_relative_task_matches_runner_name(diagnostics, monkeypatch, tmp_path):
    config_dir = tmp_path / "families" / "toy"
    config_dir.mkdir(parents=True)
    yaml_file = config_dir / "tasks.yaml"
    yaml_file.write_text("tasks:\n- dataset_path: ./series.parquet\n")
    task = in_memory_task()
    calls = []

    def constructor(**values):
        calls.append(values)
        return task

    monkeypatch.setattr(fev, "Task", constructor)
    assert list(diagnostics._load_tasks(yaml_file, ["families/toy"])) == [task]
    assert calls == [
        {
            "dataset_path": str(config_dir / "series.parquet"),
            "dataset_config": None,
            "task_name": "families/toy",
        }
    ]


def test_no_evidence_level_reads_csv_rows(diagnostics):
    rows = csv.DictReader(
        [
            "window_items,panel_items,removed,shuffled_scores\n",
            f'8,8,,"{json.dumps([0.07] * 200)}"\n',
        ]
    )
    assert diagnostics.no_evidence_level(rows) == pytest.approx(0.07)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "/toy",
        "toy/",
        "a//b",
        "a/./b",
        "a/../b",
        *(f"a{char}b" for char in "\\*?[]#\r\n"),
        "level",
        "attributes.csv/toy",
        "tasks.csv/toy",
        "level.json/toy",
        "tasks.yaml/toy",
        "names.txt/toy",
        "inputs",
        "completed",
    ],
)
def test_refuses_unsafe_report_name_before_opening_any_task(
    diagnostics, monkeypatch, tmp_path, name
):
    def unexpected_task(**_):
        pytest.fail("constructed a task before rejecting an unsafe report path")

    monkeypatch.setattr(fev, "Task", unexpected_task)
    yaml_file = tmp_path / "tasks.yaml"
    yaml_file.write_text(
        json.dumps({"tasks": [{"name": n, "dataset_path": "fake"} for n in ("a-safe-task", name)]})
    )
    with pytest.raises(ValueError, match="unsafe|reserved"):
        diagnostics.main(["--tasks", str(yaml_file), "--out", str(tmp_path / "reports")])


@pytest.mark.parametrize("names", [None, ["other"]])
def test_refuses_conflicting_task_names(diagnostics, tmp_path, names):
    yaml_file = tmp_path / "tasks.yaml"
    yaml_file.write_text("tasks:\n- name: toy\n  task_name: other\n  dataset_path: fake\n")
    with pytest.raises(ValueError, match="conflicting name and task_name"):
        list(diagnostics._load_tasks(yaml_file, names))
