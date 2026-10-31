"""Merge per-task evidence without rounding metrics or mutating source runs."""

import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tests.test_bench.test_fev_report import report_module, report_template  # noqa: F401

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/fev_merge_runs.py"


def merge_module():
    spec = importlib.util.spec_from_file_location("fev_merge_runs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def split_run(tmp_path, report_template):  # noqa: F811 -- imported pytest fixture
    specs, source, tasks, profiles = report_template
    root = tmp_path / "split"
    frame = pd.read_csv(source / "summaries.csv", float_precision="round_trip")
    for spec in specs:
        directory = root / spec.name
        shutil.copytree(source / spec.name, directory / spec.name)
        frame.loc[frame.task == spec.name].to_csv(directory / "summaries.csv", index=False)
    return root, specs, source, tasks, profiles


def invoke(run, *extra):
    root, _, _, tasks, profiles = run
    return merge_module().main(
        ["--run-root", str(root), "--tasks", str(tasks), "--profiles", str(profiles), *extra]
    )


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_merge_matches_combined_dir(split_run):
    root, specs, source, _, _ = split_run
    before = snapshot(root)
    assert invoke(split_run) == 0
    assert {k: v for k, v in snapshot(root).items() if not k.startswith("merged/")} == before
    merged = root / "merged"
    assert report_module().build_report(merged, specs) == report_module().build_report(
        source, specs
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(merged / "summaries.csv", float_precision="round_trip")
        .sort_values(["task", "model_name"])
        .reset_index(drop=True),
        pd.read_csv(source / "summaries.csv", float_precision="round_trip")
        .sort_values(["task", "model_name"])
        .reset_index(drop=True),
    )
    for spec in specs:
        assert (merged / spec.name).readlink() == Path("..") / spec.name / spec.name


def test_merge_refuses_duplicate_rows(split_run):
    root = split_run[0]
    path = root / "first/summaries.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    before = snapshot(root)
    assert invoke(split_run) == 2
    assert snapshot(root) == before and not (root / "merged").exists()


def test_merge_refuses_foreign_task_rows(split_run):
    root = split_run[0]
    path = root / "first/summaries.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, "task"] = "second"
    frame.to_csv(path, index=False)
    assert invoke(split_run) == 2
    assert not (root / "merged").exists()


def test_merge_refuses_missing_task(split_run):
    root = split_run[0]
    shutil.rmtree(root / "second")
    assert invoke(split_run) == 2
    assert not (root / "merged").exists()


def test_merge_refuses_nonempty_out(split_run):
    root = split_run[0]
    (root / "merged").mkdir()
    (root / "merged/keep").write_text("keep")
    before = snapshot(root)
    assert invoke(split_run) == 2
    assert snapshot(root) == before


def test_merge_allow_partial(split_run):
    root = split_run[0]
    shutil.rmtree(root / "second")
    assert invoke(split_run, "--allow-partial") == 0
    assert set(pd.read_csv(root / "merged/summaries.csv").task) == {"first"}
    assert not (root / "merged/second").is_symlink()


def test_merge_empty_out_and_round_trip_float(split_run):
    root = split_run[0]
    (root / "merged").mkdir()
    value = 0.12345678901234566  # pandas' default parser does not preserve this value.
    path = root / "first/summaries.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame["SQL"] = value
    frame.to_csv(path, index=False)
    assert invoke(split_run) == 0
    merged = pd.read_csv(root / "merged/summaries.csv", float_precision="round_trip")
    assert (merged.loc[merged.task == "first", "SQL"] == value).all()


def test_merge_refuses_symlink_out(split_run, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (split_run[0] / "merged").symlink_to(outside, target_is_directory=True)
    assert invoke(split_run) == 2
    assert not list(outside.iterdir())


@pytest.mark.parametrize("content", ["", "task,model_name\n", "bad,columns\n1,2\n"])
def test_merge_refuses_invalid_summary_even_with_allow_partial(split_run, content):
    (split_run[0] / "first/summaries.csv").write_text(content)
    assert invoke(split_run, "--allow-partial") == 2
