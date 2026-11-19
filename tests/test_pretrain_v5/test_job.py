"""P7 units publish only after real, resumable work and verified artifacts."""

import copy
import hashlib
import importlib
import inspect
import json
import signal
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from demand_forecast.models.v5 import V5Model
from demand_forecast.pretrain.v5 import checkpoint, loop, measure, t1
from demand_forecast.pretrain.v5.checkpoint import file_digest, latest_checkpoint, read_manifest
from demand_forecast.pretrain.v5.config import T1_BASE, SourceConfig, V5TrainConfig, identity
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


@pytest.fixture
def export_api():
    return importlib.import_module("demand_forecast.models.v5.export")


@pytest.fixture
def export_checkpoint(tmp_path):
    training = config(steps=1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        model = V5Model(training.model)
    path, _ = checkpoint.save_checkpoint(
        tmp_path / "input",
        {
            "model": model.state_dict(),
            "export": "model",
            "config": asdict(training),
            "identity": "seeded-checkpoint",
            "counters": {"batch": 1, "updates": 1, "saves": 1},
        },
        counter=1,
    )
    return path


def export_args(run, source, digest):
    return args(run, "export", name="model", checkpoint_file=source, checkpoint_sha256=digest)


def test_export_publishes_verified_unit_and_reuses_it(
    job, export_api, export_checkpoint, tmp_path, monkeypatch
):
    run = tmp_path / "run"
    digest = file_digest(export_checkpoint)
    command = export_args(run, export_checkpoint, digest)
    assert job.main(command) == 0

    unit = run / "state/export/model"
    manifest = read_manifest(unit / "unit.json")
    assert manifest is not None
    files = {"config.json", "model.safetensors", "parity.npz"}
    assert set(manifest["files"]) == files
    assert {p.name for p in unit.iterdir()} == files | {"unit.json"}
    assert manifest["code"] == CODE
    assert manifest["job_sha256"] == JOB
    assert manifest["env_sha256"] == ENV
    canonical = (
        f'{{"action":"export","checkpoint_sha256":"{digest}","code":"{CODE}",'
        f'"env_sha256":"{ENV}","name":"model"}}'
    )
    assert manifest["identity"] == hashlib.sha256(canonical.encode()).hexdigest()
    _, metadata = export_api.load_export(unit)
    assert metadata["source"]["sha256"] == digest
    export_api.require_parity(unit)
    assert list((run / "staging").iterdir()) == []

    before = {p.name: (p.stat().st_ino, p.read_bytes()) for p in unit.iterdir()}

    def unexpected_export(*_args):
        pytest.fail("a valid unit should skip export")

    monkeypatch.setattr(export_api, "write_export", unexpected_export)
    assert job.main(command) == 0
    assert {p.name: (p.stat().st_ino, p.read_bytes()) for p in unit.iterdir()} == before


def test_export_rejects_wrong_digest_before_writing(job, export_checkpoint, tmp_path, capsys):
    run = tmp_path / "run"
    assert job.main(export_args(run, export_checkpoint, "0" * 64)) == 1
    assert "checkpoint SHA-256 mismatch" in capsys.readouterr().err
    assert not run.exists()


def test_export_rebuilds_same_name_for_another_checkpoint(
    job, export_api, export_checkpoint, tmp_path
):
    run = tmp_path / "run"
    first_digest = file_digest(export_checkpoint)
    assert job.main(export_args(run, export_checkpoint, first_digest)) == 0
    unit = run / "state/export/model"
    before = read_manifest(unit / "unit.json")
    assert before is not None

    payload = checkpoint.read_payload(export_checkpoint)
    second, _ = checkpoint.save_checkpoint(
        tmp_path / "second", {**payload, "identity": "second-checkpoint"}, counter=1
    )
    second_digest = file_digest(second)
    assert second_digest != first_digest
    assert job.main(export_args(run, second, second_digest)) == 0
    after = read_manifest(unit / "unit.json")
    assert after is not None
    assert before["identity"] != after["identity"]
    _, metadata = export_api.load_export(unit)
    assert metadata["source"]["sha256"] == second_digest


def test_export_failure_during_promotion_publishes_no_unit(
    job, export_checkpoint, tmp_path, monkeypatch
):
    run = tmp_path / "run"
    command = export_args(run, export_checkpoint, file_digest(export_checkpoint))
    manifest = run / "state/export/model/unit.json"
    original, promoted = job._copy, []

    def interrupted(source, destination, staging):
        promoted.append(destination.name)
        if len(promoted) == 2:
            raise OSError("simulated crash")
        original(source, destination, staging)

    monkeypatch.setattr(job, "_copy", interrupted)
    assert job.main(command) == 1
    assert len(promoted) == 2
    assert read_manifest(manifest) is None
    monkeypatch.setattr(job, "_copy", original)
    assert job.main(command) == 0
    files = {"config.json", "model.safetensors", "parity.npz"}
    assert set(read_manifest(manifest)["files"]) == files


@pytest.mark.parametrize("action", ["train", "export"])
def test_invalid_unit_manifest_is_removed_before_retry(
    job, export_api, export_checkpoint, tmp_path, monkeypatch, action
):
    run = tmp_path / "run"
    if action == "train":
        settings = tmp_path / "config.yaml"
        config(steps=1).to_yaml(settings)
        command = args(run, config=settings, name="model")
        module, compute = loop, "train"
    else:
        command = export_args(run, export_checkpoint, file_digest(export_checkpoint))
        module, compute = export_api, "write_export"
    assert job.main(command) == 0
    manifest = run / "state" / action / "model/unit.json"
    manifest.write_text("{")
    assert read_manifest(manifest) is None
    original = getattr(module, compute)
    retried = False

    def retry(*values, **options):
        nonlocal retried
        assert not manifest.exists()
        retried = True
        return original(*values, **options)

    monkeypatch.setattr(module, compute, retry)
    assert job.main(command) == 0
    assert retried
    assert read_manifest(manifest) is not None


def test_train_stop_resume_does_not_repeat_batches(job, settings, tmp_path):
    command = args(tmp_path, config=settings, name="leg")
    unit = tmp_path / "state/train/leg"
    assert job.main([*command, "--stop-after", "4"]) == 75
    assert not (unit / "unit.json").exists()
    assert not (unit / "train.json").exists()
    assert not (unit / "probe.json").exists()
    assert latest_checkpoint(unit)[1]["counters"]["batch"] == 4
    assert job.main(command) == 0
    manifest = read_manifest(unit / "unit.json")
    assert manifest is not None
    assert set(manifest["files"]) == {"checkpoint-final.pt", "train.json", "probe.json"}
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


@pytest.fixture(params=["train", "t1"])
def probe_unit(request, job, settings, small_t1, tmp_path):
    if request.param == "train":
        unit = tmp_path / "state/train/leg"
        command = args(tmp_path, config=settings, name="leg")
    else:
        unit = tmp_path / "state/t1/static_category-true-s7/eval-2-40"
        command = t1_args(tmp_path, small_t1)
    assert job.main(command) == 0
    return command, unit


def test_train_and_t1_units_publish_probe_json(job, probe_unit, capsys):
    _, unit = probe_unit
    manifest = read_manifest(unit / "unit.json")
    assert "probe.json" in manifest["files"]
    published = json.loads((unit / "probe.json").read_text())
    assert set(published) == {"report", "gate"}
    assert type(published["gate"]["passed"]) is bool
    assert job.main(["probe", "--checkpoint", str(unit / "checkpoint-final.pt")]) == 0
    assert json.loads(capsys.readouterr().out) == published


def test_a_unit_without_probe_json_is_incomplete(job, probe_unit):
    command, unit = probe_unit
    unit_path = unit / "unit.json"
    manifest = read_manifest(unit_path)
    manifest["files"].pop("probe.json", None)
    checkpoint.write_manifest(unit_path, manifest)
    assert read_manifest(unit_path) is not None
    existing_weights = file_digest(unit / "checkpoint-final.pt")
    assert job.main(command) == 0
    assert "probe.json" in read_manifest(unit_path)["files"]
    assert file_digest(unit / "checkpoint-final.pt") == existing_weights


def test_probe_cli_agrees_for_checkpoint_and_export(
    job, export_api, export_checkpoint, tmp_path, capsys
):
    payload = checkpoint.read_payload(export_checkpoint)
    payload["config"]["sampler"].update(
        derive=["naive_rows"], screen_no_evidence=0.125, screen_fair_spread=True
    )
    payload["ema"] = {name: value.clone() for name, value in payload["model"].items()}
    for value in payload["model"].values():
        if value.is_floating_point():
            value.zero_()
    payload["export"] = "ema"
    selected, _ = checkpoint.save_checkpoint(tmp_path / "selected", payload, counter=1)
    directory = tmp_path / "export"
    export_api.write_export(selected, directory)
    assert job.main(["probe", "--checkpoint", str(selected)]) == 0
    saved = json.loads(capsys.readouterr().out)
    assert job.main(["probe", "--export", str(directory)]) == 0
    assert saved == json.loads(capsys.readouterr().out)
    assert all("zeroed_naive_rows" in family for family in saved["report"]["families"].values())


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
    [
        ("static_category", ("true", "null")),
        ("selection", ("true", "null", "gates_one")),
        ("past_driver", ("true", "null")),
    ],
)
def test_t1_all_arms_publish_paired_evaluations(job, small_t1, tmp_path, path, arms, monkeypatch):
    base = V5TrainConfig.from_yaml(small_t1)
    episode_count = 6 if path == "past_driver" else 2
    eval_set = t1.eval_set
    observed = []

    def observe(*args, **kwargs):
        observed.append(kwargs.get("base"))
        return eval_set(*args, **kwargs)

    monkeypatch.setattr(t1, "eval_set", observe)
    digests = set()
    directories = {}
    results = {}
    removal = None
    for arm in arms:
        assert (
            job.main(t1_args(tmp_path, small_t1, arm=arm, path=path, episodes=episode_count)) == 0
        )
        unit = tmp_path / f"state/t1/{path}-{arm}-s7/eval-{episode_count}-40"
        record = read_manifest(unit / "unit.json")
        assert record is not None
        files = {"checkpoint-final.pt", "train.json", "result.json", "probe.json"}
        off = unit / "result-derive-off.json"
        publishes_removal = path == "past_driver" and arm == "true"
        assert off.is_file() == publishes_removal
        if publishes_removal:
            files.add(off.name)
            removal = json.loads(off.read_text())
            assert removal["eval_digest"] == record["eval_digest"]
            assert removal["derived"] == [[] for _ in range(episode_count)]
            assert removal["derive"] == []
        assert set(record["files"]) == files
        result = json.loads((unit / "result.json").read_text())
        assert len(result["loss"]) == episode_count
        if publishes_removal:
            assert sum(map(bool, result["derived"])) >= 2
        assert ("forced" in result) == (path == "selection" and arm == "true")
        assert result["eval_digest"] == record["eval_digest"]
        digests.add(record["eval_digest"])
        directories[arm] = unit
        results[arm] = result
    assert len(digests) == 1
    assert observed == [base] * len(arms)
    if removal is not None:
        out = tmp_path / "compare"
        command = [
            "t1-compare",
            "--base",
            str(small_t1),
            "--code",
            CODE,
            "--path",
            path,
            "--seed",
            "7",
            "--episodes",
            str(episode_count),
            "--max-panels",
            "40",
            "--out",
            str(out),
            *(
                value
                for arm, directory in directories.items()
                for value in ("--arm", f"{arm}={directory}")
            ),
        ]
        assert job.main(command) == 0
        report = json.loads((out / "compare.json").read_text())
        off_loss, on_loss = sum(removal["loss"]), sum(results["true"]["loss"])
        assert report["removal"] == {
            "n": episode_count,
            "low": None,
            "high": None,
            "gain": pytest.approx((off_loss - on_loss) / off_loss),
        }

        true_dir = directories["true"]
        manifest = read_manifest(true_dir / "unit.json")
        del manifest["files"]["result-derive-off.json"]
        (true_dir / "result-derive-off.json").unlink()
        checkpoint.write_manifest(true_dir / "unit.json", manifest)
        assert job.main(command) == 0
        without = json.loads((out / "compare.json").read_text())
        assert "removal" not in without
        assert report["comparisons"] == without["comparisons"]
        assert report["outcome"] == without["outcome"]

        assert job.main(t1_args(tmp_path, small_t1, path=path, episodes=episode_count)) == 0
        assert "result-derive-off.json" in read_manifest(true_dir / "unit.json")["files"]


@pytest.mark.parametrize(
    "corpus,expected",
    [
        (None, "f8261cb26b1f807b80ead541b3c3e70985f1ea0450fd31a2fd0dfbc4ffa9cd10"),
        ("d" * 64, "32810d2b80fd1c05bacfac63093a96f73ff135d3616dd1cc0063ccd2703ce9d5"),
    ],
)
def test_qualification_keeps_base_training_config_identity(corpus, expected):
    cfg = T1_BASE
    if corpus:
        cfg = replace(cfg, sources=(SourceConfig("typed", 0.5), SourceConfig("corpus", 0.5)))
    inputs = dict(code=CODE, corpus=corpus, init=None, env=ENV)
    assert identity(cfg, **inputs) == expected
    assert identity(replace(cfg, seed=cfg.seed + 1), **inputs) != expected


@pytest.mark.parametrize(
    "explicit",
    [
        ["--models", "both"],
        ["--updates", "200"],
        ["--workers", "8,16,32,46"],
        ["--models", "both", "--updates", "200", "--workers", "8,16,32,46"],
    ],
)
def test_explicit_default_options_are_not_a_scope(job, settings, tmp_path, explicit):
    base = args(tmp_path, "qualify", config=settings, name="unit")
    parser = job._parser()
    assert job._qualification_scope(parser.parse_args(base)) is None
    assert job._qualification_scope(parser.parse_args([*base, *explicit])) is None
    assert job._qualification_scope(parser.parse_args([*base, "--updates", "3"])) == {"updates": 3}
    assert job._qualification_scope(parser.parse_args([*base, "--workers", "16"])) == {
        "workers": [16]
    }
    signature = inspect.signature(measure.qualify).parameters
    assert job._QUALIFY_DEFAULTS["updates"] == signature["updates"].default
    assert job._QUALIFY_DEFAULTS["workers"] == list(signature["workers"].default)


def test_qualify_options_restrict_what_runs(job, settings, tmp_path, monkeypatch):
    architecture = config().model
    t1_architecture = replace(architecture, d_model=24)
    calls = []

    def trial(cfg, directory, device):
        calls.append((cfg.model, cfg.workers, cfg.optim.max_steps))
        return {"updates": cfg.optim.max_steps}, directory / "unused.pt"

    monkeypatch.setattr(measure, "T1_MODEL", t1_architecture)
    monkeypatch.setattr(measure, "loader", lambda *a: iter([None]))
    monkeypatch.setattr(measure, "_trial", trial)
    monkeypatch.setattr(measure, "_checkpoint_measure", lambda *a: {})
    monkeypatch.setattr(measure, "load_model", lambda _: (torch.nn.Identity(), None))
    monkeypatch.setattr(measure, "_worst_measure", lambda *a: {})
    monkeypatch.setattr(measure, "_inference_panel", lambda **kw: pytest.fail("inference ran"))
    command = args(
        tmp_path,
        "qualify",
        config=settings,
        name="scoped",
        workers="0,2",
        updates=3,
    )
    command.append("--no-inference")
    scoped = [*command, "--models", "t1"]
    assert job.main(scoped) == 0
    unit = tmp_path / "state/qualify/scoped"
    manifest = read_manifest(unit / "unit.json")
    assert calls == [(t1_architecture, 0, 3), (t1_architecture, 2, 3)]
    report = json.loads((unit / "qualify.json").read_text())
    assert report["models"].keys() == {"t1"}
    assert "inference" not in report["models"]["t1"]
    assert job.main(scoped) == 0
    assert job.main([*scoped, "--inference-workers", "7"]) == 0
    assert len(calls) == 2

    assert job.main([*command, "--models", "full"]) == 0
    assert calls[2:] == [(architecture, 0, 3), (architecture, 2, 3)]
    assert json.loads((unit / "qualify.json").read_text())["models"].keys() == {"full"}
    assert read_manifest(unit / "unit.json")["identity"] == manifest["identity"]


@pytest.mark.parametrize("successful_origins", [1, 2], ids=["origin-trial", "worker-trial"])
def test_qualify_retries_units_without_any_successful_training_trial(
    job, settings, tmp_path, monkeypatch, successful_origins
):
    cfg = config()
    replace(cfg, sampler=replace(cfg.sampler, origins=2)).to_yaml(settings)
    t1_architecture = replace(cfg.model, d_model=24)
    monkeypatch.setattr(measure, "T1_MODEL", t1_architecture)
    monkeypatch.setattr(measure, "usable_workers", lambda _: (0, None))

    def oom(part):
        def fail(*args, **kwargs):
            raise torch.OutOfMemoryError(f"{part} out of memory")

        return fail

    for probe in (
        "_corpus_measure",
        "_first_batch_seconds",
        "_worst_measure",
        "_checkpoint_measure",
        "load_model",
        "_inference_measure",
    ):
        monkeypatch.setattr(measure, probe, oom(probe))
    calls = []
    allow_success = False

    def trial(c, directory, device):
        calls.append((c.model, c.workers, c.sampler.origins))
        if allow_success and calls[-1] == (t1_architecture, 2, successful_origins):
            return {"updates": 1}, directory / "model.pt"
        raise torch.OutOfMemoryError(f"training out of memory at {c.workers} workers")

    monkeypatch.setattr(measure, "_trial", trial)
    command = args(
        tmp_path, "qualify", config=settings, name="retry", models="both", workers="2,0", updates=1
    )
    unit = tmp_path / "state/qualify/retry"
    for run in (1, 2):
        assert job.main(command) == 1
        assert len(calls) == 6 * run
        assert not (unit / "unit.json").exists()
        assert not (unit / "qualify.json").exists()
        failure = json.loads((unit / "failure.json").read_text())
        assert "_corpus_measure out of memory" in failure["reason"]
        assert "_first_batch_seconds out of memory" not in failure["reason"]

    allow_success = True
    assert job.main(command) == 0
    assert len(calls) == 18
    assert not (unit / "failure.json").exists()
    assert read_manifest(unit / "unit.json") is not None
    report = json.loads((unit / "qualify.json").read_text())
    assert report["models"]["full"]["workers"]["2"]["status"] == "oom"
    t1_report = report["models"]["t1"]
    successful = t1_report["origins"]["1"] if successful_origins == 1 else t1_report["workers"]["2"]
    assert successful["status"] == "ok"
    assert t1_report["workers"]["0"]["status"] == t1_report["inference"]["status"] == "oom"


@pytest.mark.parametrize("workers", ["16,16", "16,016"])
def test_qualify_refuses_duplicate_workers(job, settings, tmp_path, monkeypatch, capsys, workers):
    monkeypatch.setattr(job, "_run", lambda *a: pytest.fail("duplicate workers reached the action"))
    command = args(tmp_path, "qualify", config=settings, name="duplicate", workers=workers)
    with pytest.raises(SystemExit) as error:
        job.main(command)
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "--workers" in message and "duplicate" in message


def test_qualify_inference_workers_option_reaches_measurement(job, settings, tmp_path, monkeypatch):
    recorded = {}

    def qualify(cfg, out, **options):
        recorded.update(options)
        report = {
            "cold_start": {"status": "ok"},
            "models": {"full": {"workers": {"0": {"status": "ok"}}, "origins": {}}},
        }
        (out / "qualify.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(measure, "qualify", qualify)
    command = args(tmp_path, "qualify", config=settings, name="inference", inference_workers=3)
    assert job.main(command) == 0
    assert recorded["inference_workers"] == 3


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
