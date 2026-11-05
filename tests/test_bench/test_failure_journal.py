"""A failure journal records outcomes without controlling artifact reuse."""

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("fev")

from demand_forecast.bench import run as bench  # noqa: E402
from demand_forecast.core import run as core  # noqa: E402
from demand_forecast.core.training import InterruptedRun, Trainer  # noqa: E402
from tests.helpers.fev_fixture import write_fev_fixture  # noqa: E402
from tests.helpers.fev_spec import fixture_spec  # noqa: E402
from tests.helpers.settings import tiny_settings  # noqa: E402

POLICIES = ["first_window", "per_window"]
RECORD_FIELDS = {"task", "exception_type", "message", "traceback", "utc_time", "run_identity"}


def file_states(directory):
    return {
        path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


def training_dir(out, spec, policy):
    directory = out / spec.name
    return directory if policy == "first_window" else directory / "window_0"


@pytest.mark.parametrize("policy", POLICIES)
def test_parent_retry_preserves_completed_child(tmp_path, monkeypatch, policy):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    child = replace(spec, name="fixture/child")
    settings = tiny_settings(base_steps=1, d_model=8)
    out = tmp_path / "out"
    bench.run_fev([child], settings, out, policy=policy)
    before = file_states(out / child.name)
    assert before

    def fail(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    with monkeypatch.context() as patch:
        patch.setattr(core, "train_panel", fail)
        failed = bench.run_fev([spec], settings, out, policy=policy, keep_going=True)
    assert failed.attrs["failed_tasks"] == [spec.name]
    result = bench.run_fev([spec], settings, out, policy=policy, resume=True, keep_going=True)
    assert not result.attrs["failed_tasks"]
    assert file_states(out / child.name) == before
    assert set(result.task) == {spec.name, child.name}
    assert json.loads((out / "failures.json").read_text()) == {}


@pytest.mark.parametrize("policy", POLICIES)
def test_transient_fingerprint_failure_preserves_checkpoint_resume(tmp_path, monkeypatch, policy):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=3, d_model=8)
    out = tmp_path / "out"
    with pytest.raises(InterruptedRun):
        bench.run_fev([spec], settings, out, policy=policy, stop_after=1)
    before = file_states(out / spec.name)
    assert before

    def transient(task):
        raise ConnectionError("temporary data-source failure")

    with monkeypatch.context() as patch:
        patch.setattr(bench, "dataset_fingerprint", transient)
        failed = bench.run_fev([spec], settings, out, policy=policy, resume=True, keep_going=True)
    assert failed.attrs["failed_tasks"] == [spec.name]
    record = json.loads((out / "failures.json").read_text())[spec.name]
    assert set(record) == RECORD_FIELDS
    assert record["run_identity"] is None
    assert record["exception_type"] == "ConnectionError"
    assert file_states(out / spec.name) == before
    result = bench.run_fev([spec], settings, out, policy=policy, resume=True, keep_going=True)
    assert not result.attrs["failed_tasks"]
    report = json.loads((training_dir(out, spec, policy) / "run.json").read_text())
    assert report["resumed_from_step"] == 1
    assert json.loads((out / "failures.json").read_text()) == {}


@pytest.mark.parametrize("policy", POLICIES)
def test_completed_task_reused_after_transient_bundle_failure(tmp_path, monkeypatch, policy):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=1, d_model=8)
    out = tmp_path / "out"
    bench.run_fev([spec], settings, out, policy=policy)
    before = file_states(out / spec.name)
    summary = (out / "summaries.csv").read_bytes()
    assert before

    def transient(*args, **kwargs):
        raise RuntimeError("out of memory loading completed bundle")

    def forbidden(*args, **kwargs):
        pytest.fail("completed task must not enter training")

    monkeypatch.setattr(Trainer, "fit", forbidden)
    with monkeypatch.context() as patch:
        patch.setattr(core, "load_bundle", transient)
        failed = bench.run_fev([spec], settings, out, policy=policy, resume=True, keep_going=True)
    assert failed.attrs["failed_tasks"] == [spec.name]
    record = json.loads((out / "failures.json").read_text())[spec.name]
    assert set(record) == RECORD_FIELDS
    assert record["run_identity"]
    assert record["exception_type"] == "RuntimeError"
    assert file_states(out / spec.name) == before
    result = bench.run_fev([spec], settings, out, policy=policy, resume=True, keep_going=True)
    assert not result.attrs["failed_tasks"]
    assert file_states(out / spec.name) == before
    assert (out / "summaries.csv").read_bytes() == summary
    assert json.loads((out / "failures.json").read_text()) == {}


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("artifact", ["complete", "checkpoint"])
def test_journal_cannot_bypass_identity(tmp_path, policy, artifact):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=3, d_model=8)
    out = tmp_path / "out"
    if artifact == "checkpoint":
        with pytest.raises(InterruptedRun):
            bench.run_fev([spec], settings, out, policy=policy, stop_after=1)
    else:
        bench.run_fev([spec], settings, out, policy=policy)
    before = file_states(out / spec.name)
    settings.seed += 1
    for _ in range(2):
        result = bench.run_fev([spec], settings, out, policy=policy, resume=True, keep_going=True)
        assert result.attrs["failed_tasks"] == [spec.name]
        failure = json.loads((out / "failures.json").read_text())[spec.name]
        assert set(failure) == RECORD_FIELDS
        assert failure["run_identity"]
        assert failure["exception_type"] == "ValueError"
        assert failure["message"] == "run identity mismatch"
        assert file_states(out / spec.name) == before
    with pytest.raises(ValueError, match="^run identity mismatch$"):
        bench.run_fev([spec], settings, out, policy=policy, resume=True)
    assert file_states(out / spec.name) == before


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("keep_going", [False, True])
def test_oom_and_interrupted_retry_resume_existing_checkpoints(
    tmp_path, monkeypatch, policy, keep_going
):
    import torch

    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=4, d_model=8)
    out = tmp_path / "out"
    original = core.train_panel

    def fail_after_checkpoint(*args, **kwargs):
        kwargs["stop_after"] = 1
        try:
            original(*args, **kwargs)
        except InterruptedRun:
            raise RuntimeError("CUDA out of memory") from None

    with monkeypatch.context() as patch:
        patch.setattr(core, "train_panel", fail_after_checkpoint)
        bench.run_fev([spec], settings, out, policy=policy, keep_going=True)
    journal = out / "failures.json"
    record = json.loads(journal.read_text())[spec.name]
    assert set(record) == RECORD_FIELDS
    assert record["run_identity"]
    directory = training_dir(out, spec, policy)
    seen_steps = []

    def observe(*args, **kwargs):
        if args[3] == directory:
            assert kwargs["resume"] is True
            seen_steps.append(
                torch.load(directory / "checkpoints/latest.pt", weights_only=False)["step"]
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "train_panel", observe)
    with pytest.raises(InterruptedRun):
        bench.run_fev(
            [spec], settings, out, policy=policy, keep_going=keep_going, resume=True, stop_after=2
        )
    assert seen_steps == [1]
    assert json.loads(journal.read_text())[spec.name] == record
    result = bench.run_fev([spec], settings, out, policy=policy, keep_going=keep_going, resume=True)
    assert not result.attrs["failed_tasks"]
    # stop_after counts steps in this invocation: 1 saved + 2 new = 3.
    assert seen_steps == [1, 3]
    report = json.loads((directory / "run.json").read_text())
    assert report["resumed_from_step"] == 3
    assert json.loads(journal.read_text()) == {}


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("keep_going", [False, True])
def test_legacy_journal_never_changes_caller_resume(tmp_path, monkeypatch, resume, keep_going):
    spec = fixture_spec(tmp_path / "unused")
    # Old reset controls and an unrelated identity must have no operational effect.
    journal = tmp_path / "failures.json"
    journal.write_text(
        json.dumps(
            {
                spec.name: {
                    "run_identity": "another-experiment",
                    "restartable": False,
                    "state": "retry_started",
                    "reset_pending": True,
                }
            }
        )
    )
    sentinel = tmp_path / spec.name / "sentinel"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"preserve")
    before = file_states(sentinel.parent)
    seen = []

    def observe(*args, **kwargs):
        seen.append(kwargs["resume"])

    monkeypatch.setattr(bench, "_run_fev_tasks", observe)
    bench.run_fev([spec], tiny_settings(), tmp_path, resume=resume, keep_going=keep_going)
    assert seen == [resume]
    assert file_states(sentinel.parent) == before
    assert json.loads(journal.read_text()) == {}


@pytest.mark.parametrize("panel_check", [False, True])
def test_journal_replaces_failure_atomically_and_clears_on_success(
    tmp_path, monkeypatch, panel_check
):
    spec = fixture_spec(tmp_path / "unused")
    journal = tmp_path / ("panel_failures.json" if panel_check else "failures.json")
    other_journal = tmp_path / ("failures.json" if panel_check else "panel_failures.json")
    other_journal.write_text('{"unrelated": {"message": "preserve"}}')
    other_before = other_journal.read_bytes()
    errors = [ConnectionError("first"), RuntimeError("second")]
    published = []
    original_replace = Path.replace

    def observe_replace(source, target):
        if target == journal:
            prior = json.loads(journal.read_text()) if journal.exists() else {}
            incoming = json.loads(source.read_text())
            published.append((prior, incoming))
        return original_replace(source, target)

    def attempt(*args, **kwargs):
        if errors:
            raise errors.pop(0)

    monkeypatch.setattr(Path, "replace", observe_replace)
    monkeypatch.setattr(bench, "check_panel" if panel_check else "_run_fev_tasks", attempt)
    for message in ("first", "second"):
        result = bench.run_fev(
            [spec], tiny_settings(), tmp_path, keep_going=True, panel_check=panel_check
        )
        assert result.attrs["failed_tasks"] == [spec.name]
        record = json.loads(journal.read_text())[spec.name]
        assert set(record) == RECORD_FIELDS
        assert record["task"] == spec.name
        assert record["message"] == message
        assert record["utc_time"].endswith("+00:00")
        assert 0 < len(record["traceback"]) <= 20
        assert message in record["traceback"][-1]
    bench.run_fev([spec], tiny_settings(), tmp_path, keep_going=True, panel_check=panel_check)
    assert json.loads(journal.read_text()) == {}
    assert len(published) == 3
    assert published[0][0] == {}
    assert published[1][0] == published[0][1]
    assert published[2] == (published[1][1], {})
    assert other_journal.read_bytes() == other_before


def test_runner_has_only_temporary_file_cleanup():
    """The runner deletes nothing but the temporary files of its three atomic writers.

    Pinned as literals rather than compared with an earlier revision: CI checks
    out a single commit, so no history is available to read main's source from.
    """
    source = (Path(__file__).resolve().parents[2] / "demand_forecast/bench/run.py").read_text()
    for removed in ("rmtree", "before_task", "_verify_retry_artifacts", "ArtifactMismatchError"):
        assert removed not in source
    destructive = {"unlink", "rmtree", "rmdir", "remove", "removedirs"}
    calls = {
        node.name: [
            ast.unparse(call)
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in destructive
        ]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    }
    assert {name: found for name, found in calls.items() if found} == {
        "_save_predictions": ["temporary_path.unlink(missing_ok=True)"],
        "_save_summaries": ["temporary_path.unlink(missing_ok=True)"],
        "_save_failures": ["temporary.unlink(missing_ok=True)"],
    }
    assert "shutil" not in source
