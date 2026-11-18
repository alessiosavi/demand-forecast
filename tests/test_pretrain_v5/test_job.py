"""P7 units publish only after real, resumable work and verified artifacts."""

import copy
import importlib
import json
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from demand_forecast.pretrain.v5 import checkpoint, loop, measure, t1
from demand_forecast.pretrain.v5.checkpoint import file_digest, latest_checkpoint, read_manifest
from demand_forecast.pretrain.v5.config import V5TrainConfig
from tests.helpers.v5_loop import config, training_equal

CODE = "a" * 40
ENV = "b" * 64
JOB = "c" * 64


@pytest.fixture
def job():
    return importlib.import_module("demand_forecast.pretrain.v5.job")


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / "config.yaml"
    config(steps=6).to_yaml(path)
    return path


def args(tmp_path, action="train", **values):
    options = dict(run_dir=tmp_path, code=CODE, job_sha256=JOB, env_sha256=ENV, **values)
    return [action, *(s for k, v in options.items() for s in ("--" + k.replace("_", "-"), str(v)))]


def test_train_stop_resume_does_not_repeat_batches(job, settings, tmp_path):
    command = args(tmp_path, config=settings, name="leg")
    unit = tmp_path / "state/train/leg"
    assert job.main([*command, "--stop-after", "4"]) == 75
    assert not (unit / "unit.json").exists()
    assert not (unit / "train.json").exists()
    assert latest_checkpoint(unit)[1]["counters"]["batch"] == 4
    assert job.main(command) == 0
    manifest = read_manifest(unit / "unit.json")
    assert manifest is not None
    assert set(manifest["files"]) == {"checkpoint-final.pt", "train.json"}
    assert manifest["code"] == CODE
    assert manifest["job_sha256"] == JOB
    assert manifest["env_sha256"] == ENV
    reference = tmp_path / "reference"
    loop.train(config(steps=6), reference, code=CODE, env=ENV)
    training_equal(
        loop._read_checkpoint(unit / "checkpoint-final.pt"),
        loop._read_checkpoint(reference / "checkpoint-final.pt"),
    )
    before = file_digest(unit / "checkpoint-final.pt")
    assert job.main(command) == 0
    assert file_digest(unit / "checkpoint-final.pt") == before
    (unit / "train.json").write_text("broken")
    assert read_manifest(unit / "unit.json") is None


def test_wrong_init_digest_fails_with_reason(job, settings, tmp_path, capsys):
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"wrong contents")
    assert (
        job.main(args(tmp_path, config=settings, name="leg", init_file=initial, init_sha256=ENV))
        == 1
    )
    failure = json.loads((tmp_path / "state/train/leg/failure.json").read_text())
    assert "init" in failure["reason"].lower()
    assert not (tmp_path / "state/train/leg/unit.json").exists()
    # The place of the failure is kept: in the record and in the job's log.
    assert failure["traceback"].startswith("Traceback (most recent call last)")
    assert failure["reason"].split(":", 1)[0] in failure["traceback"]
    assert failure["traceback"] in capsys.readouterr().err


@pytest.mark.parametrize("finished", [False, True])
def test_changed_environment_rejects_existing_identity(job, settings, tmp_path, finished):
    command = args(tmp_path, config=settings, name="leg")
    assert job.main(command if finished else [*command, "--stop-after", "4"]) == (
        0 if finished else 75
    )
    command[command.index(ENV)] = "d" * 64
    assert job.main(command) == 1
    unit = tmp_path / "state/train/leg"
    assert "identity" in json.loads((unit / "failure.json").read_text())["reason"].lower()
    assert not (unit / "unit.json").exists()


def test_nonfinite_is_failure_not_completion(job, settings, tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "NONFINITE_STOP_AFTER", 1)
    real_loss = loop.v5_loss

    def nonfinite(*args):
        loss, parts = real_loss(*args)
        return loss * float("nan"), parts

    monkeypatch.setattr(loop, "v5_loss", nonfinite)
    assert job.main(args(tmp_path, config=settings, name="leg")) == 1
    unit = tmp_path / "state/train/leg"
    assert "nonfinite" in (unit / "failure.json").read_text()
    assert not (unit / "unit.json").exists()


@pytest.fixture
def small_t1(monkeypatch, tmp_path):
    base = config(steps=1)
    base = replace(
        base, sampler=replace(base.sampler, designate_target=True, contexts=(64,), horizon=(16, 24))
    )
    monkeypatch.setattr(t1, "T1_BASE", base)
    paths = copy.deepcopy(t1.PATHS)
    for entry in paths.values():
        entry["both"].setdefault("source", {}).update(max_items=8, length=(128, 160))
    monkeypatch.setattr(t1, "PATHS", paths)
    path = tmp_path / "base.yaml"
    base.to_yaml(path)
    return path


def t1_args(tmp_path, base, arm="true", path="static_category", episodes=2):
    return args(
        tmp_path, "t1", base=base, path=path, arm=arm, seed=7, episodes=episodes, max_panels=40
    )


def test_t1_interrupt_after_training_resumes_evaluation_only(job, small_t1, tmp_path, monkeypatch):
    command = t1_args(tmp_path, small_t1)
    root = tmp_path / "state/t1/static_category-true-s7"
    real_evaluate = t1.evaluate

    def interrupt(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(t1, "evaluate", interrupt)
    assert job.main(command) == 75
    assert read_manifest(root / "train/train.json") is not None
    assert not (root / "eval-2-40/unit.json").exists()
    digest = file_digest(root / "train/checkpoint-final.pt")
    monkeypatch.setattr(t1, "evaluate", real_evaluate)
    assert job.main(command) == 0
    assert read_manifest(root / "eval-2-40/unit.json") is not None
    assert file_digest(root / "train/checkpoint-final.pt") == digest
    assert job.main(t1_args(tmp_path, small_t1, episodes=3)) == 0
    assert file_digest(root / "train/checkpoint-final.pt") == digest
    first = read_manifest(root / "eval-2-40/unit.json")
    second = read_manifest(root / "eval-3-40/unit.json")
    assert first["identity"] == second["identity"]
    assert first["eval_digest"] != second["eval_digest"]
    assert first["base_sha256"] == file_digest(small_t1)
    assert (first["path"], first["arm"], first["seed"], first["episodes"], first["max_panels"]) == (
        "static_category",
        "true",
        7,
        2,
        40,
    )


def killable_t1(boundary, command):
    """Child process barriers preserve real training, checkpointing and evaluation."""
    import torch

    from demand_forecast.pretrain.v5 import job

    torch.set_num_threads(1)
    run = Path(command[command.index("--run-dir") + 1])
    root = run / "state/t1/static_category-true-s7"
    paths = copy.deepcopy(t1.PATHS)
    for entry in paths.values():
        entry["both"].setdefault("source", {}).update(max_items=8, length=(128, 160))
    t1.PATHS = paths

    def wait_for_release():
        (run / "ready").touch()
        deadline = time.monotonic() + 30
        while not (run / "release").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("parent did not release child")
            time.sleep(0.01)

    real_loss, real_evaluate = loop.v5_loss, t1.evaluate

    def loss(*a, **kw):
        if boundary == "training":
            saved = latest_checkpoint(root / "train")
            if saved is not None and saved[1]["counters"]["batch"] > 0:
                wait_for_release()
        with (run / "loss-calls").open("a") as stream:
            stream.write("update\n")
        return real_loss(*a, **kw)

    def evaluate(*a, **kw):
        if boundary == "evaluation":
            wait_for_release()
        return real_evaluate(*a, **kw)

    loop.v5_loss, t1.evaluate = loss, evaluate
    raise SystemExit(job.main(command))


@pytest.mark.parametrize("boundary", ["training", "evaluation"])
def test_t1_sigkill_resumes_from_published_training_or_slot(small_t1, tmp_path, boundary):
    base = replace(
        config(steps=4),
        sampler=replace(config().sampler, designate_target=True, contexts=(64,), horizon=(16, 24)),
    )
    base = replace(base, optim=replace(base.optim, checkpoint_minutes=1e-9))
    base.to_yaml(small_t1)
    command = t1_args(tmp_path, small_t1)
    child_command = [
        sys.executable,
        "-c",
        "import sys; from tests.test_pretrain_v5.test_job import killable_t1; "
        "killable_t1(sys.argv[1], sys.argv[2:])",
        boundary,
        *command,
    ]
    root = tmp_path / "state/t1/static_category-true-s7"
    unit = root / "eval-2-40/unit.json"
    with (tmp_path / "child.log").open("w+") as log:
        process = subprocess.Popen(child_command, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            manifest = root / "train/train.json"
            while not (tmp_path / "ready").exists() or (
                boundary == "evaluation" and not manifest.exists()
            ):
                assert process.poll() is None, (tmp_path / "child.log").read_text()
                assert time.monotonic() < deadline, (tmp_path / "child.log").read_text()
                time.sleep(0.01)
            assert not unit.exists()
            if boundary == "evaluation":
                assert read_manifest(manifest) is not None
                final_bytes = (root / "train/checkpoint-final.pt").read_bytes()
            else:
                saved = latest_checkpoint(root / "train")
                assert saved is not None
                assert saved[1]["counters"]["batch"] == 1
                assert not manifest.exists()
            process.send_signal(signal.SIGKILL)
            assert process.wait(timeout=10) == -signal.SIGKILL
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert not unit.exists()
    assert not (unit.parent / "failure.json").exists()
    calls_before = (tmp_path / "loss-calls").read_text().splitlines()
    (tmp_path / "release").touch()
    result = subprocess.run(child_command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert read_manifest(unit) is not None
    assert len(json.loads((unit.parent / "result.json").read_text())["loss"]) == 2
    calls_after = (tmp_path / "loss-calls").read_text().splitlines()
    assert len(calls_after) == 4  # Retraining would repeat the completed updates.
    if boundary == "evaluation":
        assert calls_after == calls_before
        assert (root / "train/checkpoint-final.pt").read_bytes() == final_bytes
    else:
        assert len(calls_after) - len(calls_before) == 3
        final = loop._read_checkpoint(root / "train/checkpoint-final.pt")
        assert final["counters"]["batch"] == 4


@pytest.mark.parametrize(
    "path,arms",
    [("static_category", ("true", "null")), ("selection", ("true", "null", "gates_one"))],
)
def test_t1_all_arms_publish_paired_evaluations(job, small_t1, tmp_path, path, arms, monkeypatch):
    base = V5TrainConfig.from_yaml(small_t1)
    eval_set = t1.eval_set
    observed = []

    def observe(*args, **kwargs):
        observed.append(kwargs.get("base"))
        return eval_set(*args, **kwargs)

    monkeypatch.setattr(t1, "eval_set", observe)
    digests = set()
    for arm in arms:
        assert job.main(t1_args(tmp_path, small_t1, arm=arm, path=path)) == 0
        unit = tmp_path / f"state/t1/{path}-{arm}-s7/eval-2-40"
        record = read_manifest(unit / "unit.json")
        assert record is not None
        assert set(record["files"]) == {"checkpoint-final.pt", "train.json", "result.json"}
        result = json.loads((unit / "result.json").read_text())
        assert len(result["loss"]) == 2
        assert ("forced" in result) == (path == "selection" and arm == "true")
        assert result["eval_digest"] == record["eval_digest"]
        digests.add(record["eval_digest"])
    assert len(digests) == 1
    assert observed == [base] * len(arms)


def test_qualify_publishes_real_measurements(job, settings, tmp_path, monkeypatch):
    original = measure.qualify
    atomic = checkpoint._atomic_write

    def outside_state(destination, write, *, staging=None):
        temporary_directory = destination.parent if staging is None else staging
        if destination.suffix == ".pt":
            assert not temporary_directory.is_relative_to(tmp_path / "state"), (
                "guest could copy an unfinished qualification checkpoint"
            )
        return atomic(destination, write, staging=staging)

    monkeypatch.setattr(checkpoint, "_atomic_write", outside_state)

    def tiny(cfg, out):
        return original(
            cfg,
            out,
            models={"tiny": cfg.model},
            updates=1,
            workers=(0,),
            panel_items=2,
            panel_length=64,
        )

    monkeypatch.setattr(measure, "qualify", tiny)
    command = args(tmp_path, "qualify", config=settings, name="cpu")
    assert job.main(command) == 0
    unit = tmp_path / "state/qualify/cpu"
    assert read_manifest(unit / "unit.json") is not None
    result = json.loads((unit / "qualify.json").read_text())
    assert result["models"]["tiny"]["parameters"]["total"] > 0
    assert job.main(command) == 0


def test_qualify_term_during_training_is_unfinished(job, settings, tmp_path, monkeypatch):
    original = measure.qualify
    real_loss = loop.v5_loss

    def terminate(*args):
        signal.raise_signal(signal.SIGTERM)
        return real_loss(*args)

    def tiny(cfg, out):
        return original(
            cfg,
            out,
            models={"tiny": cfg.model},
            updates=2,
            workers=(0,),
            panel_items=2,
            panel_length=64,
        )

    monkeypatch.setattr(loop, "v5_loss", terminate)
    monkeypatch.setattr(measure, "qualify", tiny)
    assert job.main(args(tmp_path, "qualify", config=settings, name="term")) == 75
    unit = tmp_path / "state/qualify/term"
    assert not (unit / "unit.json").exists()
    assert not (unit / "failure.json").exists()


def test_module_cli(job, settings, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            job.__name__,
            *args(tmp_path, config=settings, name="cli", stop_after=1),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 75, result.stderr


def test_rejects_unsafe_unit_name(job, settings, tmp_path):
    with pytest.raises(SystemExit) as error:
        job.main(args(tmp_path, config=settings, name="../escape"))
    assert error.value.code == 2
