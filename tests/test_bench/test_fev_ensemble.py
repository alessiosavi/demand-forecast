"""Synthetic saved-yardstick ensembles, without inference packages or downloads."""

import hashlib
import json
import shutil
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.yardsticks import run_yardstick, runner  # noqa: E402
from tests.test_bench.test_yardsticks import (  # noqa: E402
    FakeAdapter,
)
from tests.test_bench.test_yardsticks import (  # noqa: E402
    offline_only as offline_only,
)
from tests.test_bench.test_yardsticks import (  # noqa: E402
    scenario as scenario,
)


class OffsetAdapter(FakeAdapter):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def predict(self, windows, horizon, levels):
        return [
            [array + self.offset for array in window]
            for window in super().predict(windows, horizon, levels)
        ]


@pytest.fixture
def make_member(scenario, tmp_path):
    tasks, options, _ = scenario

    def make(index, offset=0.0, *, selected=None, **changes):
        weights = tmp_path / f"weights-{index}"
        weights.mkdir()
        (weights / "config.json").write_text(json.dumps({"seed": index}))
        (weights / "model.safetensors").write_bytes(f"synthetic seed {index}".encode())
        directory = tmp_path / f"member-{index}"
        resolved = replace(options, weights=str(weights), model_name=f"seed-{index}", **changes)
        run_yardstick(
            tasks if selected is None else selected,
            resolved,
            directory,
            adapter_factory=lambda _: OffsetAdapter(offset),
        )
        return directory

    return make


def ensemble(tasks, members, out, **kwargs):
    from demand_forecast.bench.yardsticks.ensemble import run_ensemble

    return run_ensemble(tasks, members, "average", out, **kwargs)


def rewrite_window(member, task, index=0, **changes):
    """Keep hashes consistent so validation reaches the altered window content."""
    directory = member / task.task_name
    path = directory / f"window_{index}.npz"
    with np.load(path, allow_pickle=False) as saved:
        payload = {key: saved[key] for key in saved.files}
    payload.update(changes)
    np.savez(path, **payload)
    record_path = directory / "run.json"
    record = json.loads(record_path.read_text())
    record["window_sha256"][path.name] = runner.file_digest(path)
    record_path.write_text(json.dumps(record))


def snapshot(directory):
    return {
        str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_identical_members_reproduce_all_scores_and_runner_columns(scenario, make_member, tmp_path):
    tasks, _, _ = scenario
    first = make_member(0)
    second = tmp_path / "identical"
    shutil.copytree(first, second)
    frame = ensemble(tasks, [first, second], tmp_path / "average", allow_identical_members=True)
    original = pd.read_csv(first / "summaries.csv", float_precision="round_trip")
    assert set(frame) == set(original)
    metrics = ["test_error", "SQL", "MASE", "WQL", "WAPE"] + [
        column for column in original if column.startswith(("SQL[", "WQL["))
    ]
    pd.testing.assert_frame_equal(frame[metrics], original[metrics])
    assert frame.inference_time_s.iloc[0] == 2 * original.inference_time_s.iloc[0]
    assert frame.training_time_s.iloc[0] == 0


def test_mean_matches_hand_computed_arrays_and_independently_scored_runner(
    scenario, make_member, tmp_path
):
    tasks, _, _ = scenario
    first, second, expected = make_member(0), make_member(1, 2), make_member(2, 1)
    out = tmp_path / "average"
    frame = ensemble(tasks, [first, second], out)
    reference = pd.read_csv(expected / "summaries.csv", float_precision="round_trip")
    assert frame.test_error.iloc[0] == reference.test_error.iloc[0]
    record = json.loads((out / tasks[0].task_name / "run.json").read_text())
    assert record["weights"] == [0.5, 0.5]
    assert [entry["directory"] for entry in record["members"]] == [str(first), str(second)]
    for member, evidence in zip([first, second], record["members"], strict=True):
        source = json.loads((member / tasks[0].task_name / "run.json").read_text())
        assert evidence["run_identity"] == source["run_identity"]
        assert evidence["window_sha256"] == source["window_sha256"]
    windows = runner.prepare_windows(tasks[0], runner.Options(**source["configuration"]["options"]))
    for index, window in enumerate(windows):
        actual, _ = runner.read_window(
            out / tasks[0].task_name / f"window_{index}.npz",
            window,
            record["levels"],
            record["run_identity"],
            tasks[0].horizon,
        )
        with np.load(first / tasks[0].task_name / f"window_{index}.npz") as saved:
            # q + 2 rounds once before averaging; q + 1 can differ by one ulp.
            np.testing.assert_allclose(actual, saved["arrays"] + 1, rtol=0, atol=1e-14)
        assert (np.diff(actual, axis=-1) >= 0).all()


def test_three_members_have_equal_one_third_weights(scenario, make_member, tmp_path):
    tasks, _, _ = scenario
    members = [make_member(index, 3 * index) for index in range(3)]
    out = tmp_path / "three"
    ensemble(tasks, members, out)
    record = json.loads((out / tasks[0].task_name / "run.json").read_text())
    assert record["weights"] == [1 / 3] * 3
    with (
        np.load(out / tasks[0].task_name / "window_0.npz") as averaged,
        np.load(members[0] / tasks[0].task_name / "window_0.npz") as source,
    ):
        np.testing.assert_allclose(averaged["arrays"], source["arrays"] + 3, atol=1e-14)


@pytest.mark.parametrize(
    "change,match",
    [
        ("ids", "ids|input identity"),
        ("targets", "targets|input identity"),
        ("quantiles", "quantile|input identity"),
        ("shape", "shape|saved forecasts"),
        ("nonfinite", "finite|saved forecasts"),
        ("window_count", "window"),
        ("missing_task", "missing|run.json"),
        ("incomplete", "complet|summary"),
        ("hash", "sha256|hash"),
    ],
)
def test_rejects_misaligned_or_incomplete_members(scenario, make_member, tmp_path, change, match):
    tasks, _, _ = scenario
    task = tasks[0]
    first, second = make_member(0), make_member(1)
    directory = second / task.task_name
    with np.load(directory / "window_0.npz") as saved:
        arrays = saved["arrays"].copy()
        ids, targets, quantiles = saved["ids"], saved["targets"], saved["quantiles"]
    if change == "ids":
        rewrite_window(second, task, ids=ids[::-1])
    elif change == "targets":
        rewrite_window(second, task, targets=targets[::-1])
    elif change == "quantiles":
        rewrite_window(second, task, quantiles=quantiles + 0.01)
    elif change == "shape":
        rewrite_window(second, task, arrays=arrays[:, :, :-1])
    elif change == "nonfinite":
        arrays[0, 0, 0, 0] = np.nan
        rewrite_window(second, task, arrays=arrays)
    elif change == "window_count":
        (directory / "window_1.npz").unlink()
    elif change == "missing_task":
        shutil.rmtree(directory)
    elif change == "incomplete":
        record_path = directory / "run.json"
        record = json.loads(record_path.read_text())
        del record["summary"]
        record_path.write_text(json.dumps(record))
    elif change == "hash":
        with (directory / "window_0.npz").open("ab") as stream:
            stream.write(b"changed")
    out = tmp_path / "rejected"
    with pytest.raises((ValueError, OSError), match=match):
        ensemble(tasks, [first, second], out)
    failure = json.loads((out / "failures.json").read_text())[task.task_name]
    assert failure["message"] and failure["exception_type"] and failure["traceback"]
    assert not (out / task.task_name / "run.json").exists()


@pytest.mark.parametrize("changes", [{"covariates": "none"}, {"max_context": 5}, {"batch_size": 7}])
def test_rejects_evaluation_option_changes(scenario, make_member, tmp_path, changes):
    tasks, _, _ = scenario
    first, second = make_member(0), make_member(1, **changes)
    with pytest.raises(ValueError, match="evaluation.*identity"):
        ensemble(tasks, [first, second], tmp_path / "mismatch")


def test_rejects_changed_requested_task_definition(scenario, make_member, tmp_path):
    tasks, _, _ = scenario
    members = [make_member(0), make_member(1)]
    changed = fev.Task(**{**tasks[0].to_dict(), "horizon": 3})
    with pytest.raises(ValueError, match="task.*definition"):
        ensemble([changed], members, tmp_path / "mismatch")


def test_rejects_identical_weights_by_default(scenario, make_member, tmp_path):
    tasks, _, _ = scenario
    first = make_member(0)
    second = tmp_path / "identical"
    shutil.copytree(first, second)
    with pytest.raises(ValueError, match="identical.*weights|weights.*identical"):
        ensemble(tasks, [first, second], tmp_path / "rejected")


def test_rerun_is_noop_and_changed_member_set_is_refused(
    scenario, make_member, tmp_path, monkeypatch
):
    tasks, _, _ = scenario
    first, second, third = make_member(0), make_member(1), make_member(2)
    out = tmp_path / "average"
    ensemble(tasks, [first, second], out)
    before = snapshot(out)
    monkeypatch.setattr(fev.Task, "evaluation_summary", lambda *a, **kw: pytest.fail("rescored"))
    ensemble(tasks, [first, second], out)
    assert snapshot(out) == before
    ensemble(tasks, [second, first], out)
    assert snapshot(out) == before
    with pytest.raises(ValueError, match="identity|members"):
        ensemble(tasks, [first, third], out)
    assert snapshot(out) == before


def test_changed_member_content_is_refused_without_altering_complete_output(
    scenario, make_member, tmp_path
):
    tasks, _, _ = scenario
    first, second = make_member(0), make_member(1)
    out = tmp_path / "average"
    ensemble(tasks, [first, second], out)
    before = snapshot(out)
    path = second / tasks[0].task_name / "window_0.npz"
    with np.load(path) as saved:
        arrays = saved["arrays"] + 1
    rewrite_window(second, tasks[0], arrays=arrays)
    with pytest.raises(ValueError, match="identity mismatch"):
        ensemble(tasks, [first, second], out)
    assert snapshot(out) == before


@pytest.mark.parametrize("key", ["published_adapter_source", "packages", "future_switch"])
def test_rejects_implementation_or_unknown_identity_changes(
    scenario, make_member, tmp_path, monkeypatch, key
):
    tasks, _, _ = scenario
    first = make_member(0)
    runtime_identity = runner.runtime_identity

    def changed(options):
        identity = runtime_identity(options)
        identity[key] = {"fev": "other-version"} if key == "packages" else "changed"
        return identity

    monkeypatch.setattr(runner, "runtime_identity", changed)
    second = make_member(1)
    with pytest.raises(ValueError, match="evaluation.*identity"):
        ensemble(tasks, [first, second], tmp_path / "mismatch")


@pytest.mark.parametrize("key", ["covariates", "packages", "published_adapter_source"])
def test_incomplete_evaluation_identity_cannot_fall_back_to_defaults(
    scenario, make_member, tmp_path, monkeypatch, key
):
    tasks, _, _ = scenario
    runtime_identity = runner.runtime_identity

    def incomplete(options):
        identity = runtime_identity(options)
        (identity["options"] if key == "covariates" else identity).pop(key)
        return identity

    monkeypatch.setattr(runner, "runtime_identity", incomplete)
    members = [make_member(0), make_member(1)]
    with pytest.raises(ValueError, match="incomplete evaluation identity"):
        ensemble(tasks, members, tmp_path / "mismatch")


def test_does_not_reopen_checkpoints_or_load_inference_models(
    scenario, make_member, tmp_path, monkeypatch
):
    tasks, _, _ = scenario
    first, second = make_member(0), make_member(1)
    for index in range(2):
        shutil.rmtree(tmp_path / f"weights-{index}")

    def forbidden(*args, **kwargs):
        pytest.fail("ensemble attempted checkpoint resolution or inference")

    monkeypatch.setattr(runner, "resolve_weights", forbidden)
    monkeypatch.setattr(runner, "_adapter", forbidden)
    frame = ensemble(tasks, [first, second], tmp_path / "average")
    assert len(frame) == 1 and np.isfinite(frame.test_error.iloc[0])


@pytest.mark.parametrize("processing", ["raw", "published"])
def test_simons_processing_identity_and_summary_columns(
    scenario, make_member, tmp_path, processing
):
    tasks, _, _ = scenario
    first = make_member(0, model="simons", processing=processing)
    second = make_member(1, model="simons", processing=processing)
    out = tmp_path / "average"
    frame = ensemble(tasks, [first, second], out)
    source = pd.read_csv(first / "summaries.csv")
    assert set(frame) == set(source)
    assert frame.processing.iloc[0] == processing
    assert frame.dropped_past_only_covariates.iloc[0] == 6
    third = make_member(2, model="simons", processing="published" if processing == "raw" else "raw")
    with pytest.raises(ValueError, match="evaluation.*identity"):
        ensemble(tasks, [first, third], tmp_path / "mixed")


def test_identical_members_reproduce_multiple_tasks_and_internal_median(
    scenario, make_member, tmp_path
):
    tasks, _, _ = scenario
    task = fev.Task(
        **{
            **tasks[0].to_dict(),
            "task_name": "single",
            "target": "target",
            "quantile_levels": [0.1, 0.9],
        }
    )
    selected = [*tasks, task]
    first = make_member(0, selected=selected)
    frame = ensemble(selected, [first, first], tmp_path / "average", allow_identical_members=True)
    source = pd.read_csv(first / "summaries.csv", float_precision="round_trip")
    pd.testing.assert_series_equal(frame.test_error, source.test_error)
    record = json.loads((tmp_path / "average" / "single" / "run.json").read_text())
    assert record["levels"] == [0.1, 0.5, 0.9]


@pytest.mark.parametrize("keep_going", [False, True])
def test_cli_records_failures_and_exits_nonzero(
    scenario, make_member, tmp_path, keep_going, capsys
):
    from scripts.fev_ensemble import main

    tasks, _, _ = scenario
    good = tasks[0]
    missing = fev.Task(**{**good.to_dict(), "task_name": "missing"})
    task_file = tmp_path / "selected.yaml"
    task_file.write_text(yaml.safe_dump({"tasks": [missing.to_dict(), good.to_dict()]}))
    first, second = make_member(0), make_member(1)
    out = tmp_path / "cli"
    args = [
        "--members",
        str(first),
        str(second),
        "--tasks",
        str(task_file),
        "--model-name",
        "average",
        "--out-dir",
        str(out),
    ]
    assert main(args + (["--keep-going"] if keep_going else [])) != 0
    failures = json.loads((out / "failures.json").read_text())
    assert "missing" in failures
    assert (out / good.task_name / "run.json").exists() is keep_going
    assert "missing" in (
        capsys.readouterr().err if not keep_going else failures["missing"]["message"]
    )


@pytest.mark.parametrize("names_form", ["csv", "file"])
def test_cli_names_select_tasks_and_allow_identical_members(
    scenario, make_member, tmp_path, names_form
):
    from scripts.fev_ensemble import main

    tasks, _, _ = scenario
    first = make_member(0)
    task_file = tmp_path / "selection.yaml"
    task_file.write_text(
        yaml.safe_dump(
            {"tasks": [tasks[0].to_dict(), {**tasks[0].to_dict(), "task_name": "unselected"}]}
        )
    )
    names = tasks[0].task_name
    if names_form == "file":
        path = tmp_path / "names.txt"
        path.write_text(names + "\n")
        names = f"@{path}"
    assert (
        main(
            [
                "--members",
                str(first),
                str(first),
                "--allow-identical-members",
                "--tasks",
                str(task_file),
                "--names",
                names,
                "--model-name",
                "average",
                "--out-dir",
                str(tmp_path / "selected"),
            ]
        )
        == 0
    )


@pytest.mark.parametrize("keep_going", [False, True])
@pytest.mark.parametrize("select_extra", [False, True])
@pytest.mark.parametrize("change", ["weights", "processing", "content", "identity"])
def test_mixed_member_configurations_are_rejected_before_any_task(
    scenario, make_member, tmp_path, monkeypatch, keep_going, select_extra, change
):
    from scripts.fev_ensemble import main

    tasks, _, _ = scenario
    extra = fev.Task(**{**tasks[0].to_dict(), "task_name": "extra"})
    selected = [*tasks, extra]
    changes = {"model": "simons", "processing": "raw"} if change == "processing" else {}
    first = make_member(0, selected=selected, **changes)
    second = make_member(1, selected=selected, **changes)
    if change in {"weights", "processing"}:
        replacement_changes = {**changes, "processing": "published"} if changes else {}
        replacement = make_member(2, selected=selected, **replacement_changes)
        shutil.rmtree(second / extra.task_name)
        shutil.copytree(replacement / extra.task_name, second / extra.task_name)
        frame = pd.read_csv(second / "summaries.csv", float_precision="round_trip")
        other = pd.read_csv(replacement / "summaries.csv", float_precision="round_trip")
        pd.concat(
            [frame[frame.task != extra.task_name], other[other.task == extra.task_name]]
        ).to_csv(second / "summaries.csv", index=False)
    else:
        path = second / extra.task_name / "run.json"
        record = json.loads(path.read_text())
        if change == "content":
            record["configuration"]["options"]["model_name"] = "other"
        else:
            record["configuration_identity"] = "other"
        path.write_text(json.dumps(record))
    # Every task must match, including ones outside the requested selection.
    task_file = tmp_path / "selection.yaml"
    requested = selected if select_extra else tasks
    task_file.write_text(yaml.safe_dump({"tasks": [task.to_dict() for task in requested]}))
    out = tmp_path / "rejected"
    monkeypatch.setattr(fev.Task, "evaluation_summary", lambda *a, **kw: pytest.fail("scored"))
    assert (
        main(
            [
                "--members",
                str(first),
                str(second),
                "--tasks",
                str(task_file),
                "--model-name",
                "average",
                "--out-dir",
                str(out),
            ]
            + (["--keep-going"] if keep_going else [])
        )
        != 0
    )
    assert not out.exists() or not list(out.rglob("*"))


@pytest.mark.parametrize("keep_going", [False, True])
@pytest.mark.parametrize("change", ["missing", "malformed", "digest", "incomplete"])
def test_invalid_root_manifest_is_rejected_before_any_task(
    scenario, make_member, tmp_path, monkeypatch, keep_going, change
):
    from scripts.fev_ensemble import main

    tasks, _, _ = scenario
    first, second = make_member(0), make_member(1)
    path = second / "yardstick.json"
    if change == "missing":
        path.unlink()
    elif change == "malformed":
        path.write_text("{")
    else:
        manifest = json.loads(path.read_text())
        if change == "digest":
            manifest["configuration_identity"] = "invalid"
        else:
            del manifest["packages"]
        path.write_text(json.dumps(manifest))
    task_file = tmp_path / "selection.yaml"
    task_file.write_text(yaml.safe_dump({"tasks": [task.to_dict() for task in tasks]}))
    out = tmp_path / "rejected"
    monkeypatch.setattr(fev.Task, "evaluation_summary", lambda *a, **kw: pytest.fail("scored"))
    assert (
        main(
            [
                "--members",
                str(first),
                str(second),
                "--tasks",
                str(task_file),
                "--model-name",
                "average",
                "--out-dir",
                str(out),
            ]
            + (["--keep-going"] if keep_going else [])
        )
        != 0
    )
    assert not out.exists() or not list(out.rglob("*"))


def test_replacement_after_member_validation_fails_task(
    scenario, make_member, tmp_path, monkeypatch
):
    from demand_forecast.bench.yardsticks import ensemble as implementation

    tasks, _, _ = scenario
    members = [make_member(0), make_member(1)]
    prepare = implementation.prepare_windows

    def replace_before_loading(*args, **kwargs):
        path = members[0] / tasks[0].task_name / "window_0.npz"
        with np.load(path) as saved:
            payload = {key: saved[key] for key in saved.files}
        payload["arrays"] += 100
        np.savez(path, **payload)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(implementation, "prepare_windows", replace_before_loading)
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="sha256"):
        ensemble(tasks, members, out)
    assert not (out / tasks[0].task_name / "run.json").exists()
    assert tasks[0].task_name in json.loads((out / "failures.json").read_text())


def test_successful_provenance_hashes_the_bytes_averaged(
    scenario, make_member, tmp_path, monkeypatch
):
    from demand_forecast.bench.yardsticks import ensemble as implementation

    tasks, _, _ = scenario
    task = tasks[0]
    members = [make_member(0), make_member(1, 2)]
    originals = {
        (member, index): (member / task.task_name / f"window_{index}.npz").read_bytes()
        for member in members
        for index in range(task.num_windows)
    }
    read_counts = {
        member / task.task_name / f"window_{index}.npz": 0 for member, index in originals
    }
    open_path = Path.open

    def count_reads(path, mode="r", *args, **kwargs):
        if path in read_counts and mode == "rb":
            read_counts[path] += 1
        return open_path(path, mode, *args, **kwargs)

    load = implementation.read_window
    loaded = []

    def replace_after_snapshot(source, *args):
        index, member_index = divmod(len(loaded), len(members))
        path = members[member_index] / task.task_name / f"window_{index}.npz"
        with np.load(BytesIO(originals[members[member_index], index])) as saved:
            payload = {key: saved[key] for key in saved.files}
        payload["arrays"] += 100
        np.savez(path, **payload)
        array, duration = load(source, *args)
        loaded.append(array.copy())
        return array, duration

    monkeypatch.setattr(Path, "open", count_reads)
    monkeypatch.setattr(implementation, "read_window", replace_after_snapshot)
    out = tmp_path / "average"
    ensemble(tasks, members, out)
    assert all(count == 1 for count in read_counts.values())
    record = json.loads((out / task.task_name / "run.json").read_text())
    for index in range(task.num_windows):
        arrays = []
        for member_index, member in enumerate(members):
            raw = originals[member, index]
            evidence = record["members"][member_index]["window_sha256"]
            assert evidence[f"window_{index}.npz"] == hashlib.sha256(raw).hexdigest()
            with np.load(BytesIO(raw)) as source:
                arrays.append(source["arrays"])
            np.testing.assert_array_equal(loaded[index * len(members) + member_index], arrays[-1])
        with np.load(out / task.task_name / f"window_{index}.npz") as averaged:
            np.testing.assert_array_equal(averaged["arrays"], arrays[0] / 2 + arrays[1] / 2)


@pytest.mark.parametrize("keep_going", [False, True])
@pytest.mark.parametrize("change", ["hash", "missing_task", "summary"])
def test_failed_member_reread_preserves_completed_output(
    scenario, make_member, tmp_path, keep_going, change
):
    from scripts.fev_ensemble import main

    tasks, _, _ = scenario
    first, second = make_member(0), make_member(1)
    out = tmp_path / "average"
    ensemble(tasks, [first, second], out)
    before = snapshot(out)
    directory = second / tasks[0].task_name
    if change == "hash":
        with (directory / "window_0.npz").open("ab") as stream:
            stream.write(b"changed")
    elif change == "missing_task":
        shutil.rmtree(directory)
    else:
        (second / "summaries.csv").unlink()
    task_file = tmp_path / "selection.yaml"
    task_file.write_text(yaml.safe_dump({"tasks": [task.to_dict() for task in tasks]}))
    assert (
        main(
            [
                "--members",
                str(first),
                str(second),
                "--tasks",
                str(task_file),
                "--model-name",
                "average",
                "--out-dir",
                str(out),
            ]
            + (["--keep-going"] if keep_going else [])
        )
        != 0
    )
    assert {key: value for key, value in snapshot(out).items() if key != "failures.json"} == before
    assert tasks[0].task_name in json.loads((out / "failures.json").read_text())


def test_root_configuration_binding_is_persisted_and_checked_on_resume(
    scenario, make_member, tmp_path, monkeypatch
):
    tasks, _, _ = scenario
    members = [make_member(0), make_member(1)]
    out = tmp_path / "average"
    ensemble(tasks, members, out)
    manifest = json.loads((out / "yardstick.json").read_text())
    assert manifest["member_configurations"] == {
        str(member): json.loads((member / "yardstick.json").read_text())["configuration_identity"]
        for member in members
    }
    replacement = make_member(2)
    shutil.rmtree(members[1])
    shutil.copytree(replacement, members[1])
    before = snapshot(out)
    monkeypatch.setattr(fev.Task, "evaluation_summary", lambda *a, **kw: pytest.fail("rescored"))
    with pytest.raises(ValueError, match="ensemble.*identity|members.*identity"):
        ensemble(tasks, members, out)
    assert snapshot(out) == before
