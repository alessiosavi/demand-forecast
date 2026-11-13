"""Offline generator and upstream integration contracts for preemptible jobs."""

import json
import subprocess
from pathlib import Path

import pytest

PIN = "8e338e915d26c8960713889ef12ab1cf5c08d591"
SHA = "5591bb0" * 5 + "abcde"
LEGS = ("stage1", "stage2", "stage3", "stage4", "stage4-long", "qualify", "export")


@pytest.mark.parametrize("size", ["25m", "12m"])
@pytest.mark.parametrize("leg", LEGS)
def test_generated_shell(jobs, tmp_path, size, leg):
    script = jobs.generate(
        size=size,
        leg=leg,
        seed=11,
        micro_batches=2,
        checkpoint_minutes=10,
        corpus_uri="gs://offline/corpus",
        repo_sha=SHA,
        upstream_pin=PIN,
    )
    path = tmp_path / "job.sh"
    path.write_text(script)
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert PIN in script and SHA in script
    assert "HF_HUB_OFFLINE=1" in script
    assert "--system-site-packages" in script
    assert '"$scratch/uv-bootstrap/bin/uv" venv' in script
    assert 'metadata.requires("simons")' in script
    assert "torch.cuda.is_available()" in script
    assert "pathlib.Path(spot_job.__file__).resolve()" in script


def test_corpus_required_and_pin_checked(jobs):
    args = dict(
        size="25m",
        leg="stage3",
        seed=11,
        micro_batches=1,
        checkpoint_minutes=10,
        repo_sha=SHA,
        upstream_pin=PIN,
    )
    with pytest.raises(ValueError, match="corpus"):
        jobs.generate(**args)
    args.update(leg="stage1", upstream_pin="0" * 40)
    with pytest.raises(ValueError, match="pin"):
        jobs.generate(**args)


@pytest.mark.parametrize("name", ["-x", "--help", "_probe"])
def test_generator_rejects_name_without_alphanumeric_start(jobs, name):
    with pytest.raises(ValueError, match="qualification name.*start"):
        jobs.generate(
            size="25m",
            leg="qualify",
            seed=11,
            micro_batches=1,
            checkpoint_minutes=10,
            repo_sha=SHA,
            upstream_pin=PIN,
            name=name,
        )


@pytest.mark.parametrize(
    "name,valid",
    [
        ("-x", False),
        ("--help", False),
        ("_probe", False),
        ("a100-stage1", True),
        ("A_1", True),
        ("0", True),
    ],
)
def test_spot_job_name_requires_alphanumeric_start(
    upstream, tmp_path, monkeypatch, capsys, name, valid
):
    import sys

    from simons.training import spot_job

    calls = []
    monkeypatch.setattr(spot_job, "qualify", lambda args: calls.append(args.name) or 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spot_job",
            "--leg=qualify",
            "--size=25m",
            f"--root={upstream}",
            f"--run-dir={tmp_path / 'run'}",
            "--seed=11",
            f"--repo-sha={SHA}",
            f"--upstream-pin={PIN}",
            "--bundle-sha256=test",
            f"--name={name}",
        ],
    )
    if valid:
        assert spot_job.main() == 0
        assert calls == [name]
    else:
        with pytest.raises(SystemExit) as error:
            spot_job.main()
        assert error.value.code == 2
        assert "qualification name must start" in capsys.readouterr().err
        assert not calls
        assert not (tmp_path / "run").exists()


def test_role_phase_origin_and_schedule(upstream):
    from simons.training import config, schedule
    from simons.training.covariates.roles import Hook
    from simons.training.reliability import role_position

    _, settings = config.load(upstream / "configs/simons-25m-stage2.yaml")
    spec = config.covariate_roles(settings)
    spec["effect_workers"] = 0
    uninterrupted = Hook(spec, 11, 0, "cpu", 12000, 64)
    resumed = Hook(spec, 11, 4000, "cpu", 12000, 64)
    for absolute in (4000, 4001, 6359, 6360, 11999):
        current, pool = role_position(absolute, 0)
        uninterrupted.planner.step = pool
        resumed.planner.step = pool
        assert resumed.planner.p_role() == uninterrupted.planner.p_role()
        assert resumed.p_eff(current) == uninterrupted.p_eff(current)
        assert resumed.p_eff(current) == (0.02 if absolute < 6360 else 0.04)
        assert schedule.multiplier(absolute, settings, 0, True) == schedule.multiplier(
            absolute, settings, 0, True
        )
    assert 6360 - 4000 == 2360  # Remaining updates in the second phase.


def test_resume_streams_and_next_batches(upstream, tmp_path):
    import random

    import numpy as np
    import torch
    from simons.synth import mech
    from simons.training import config
    from simons.training.reliability import reseed
    from simons.training.trainer import make_mixer, read_checkpoint, write_checkpoint

    path = tmp_path / "checkpoint.pt"
    write_checkpoint(path, {"format": "simons-train-1", "step": 37})
    _, settings = config.load(upstream / "configs/simons-25m-stage1.yaml")
    settings.update(
        synthetic_families="step",
        synthetic_family_weights=None,
        synthetic_pool_steps=1,
        synthetic_multi_fraction=0.5,
        synthetic_augment="",
        synthetic_observe="",
        mech_bank=64,
    )

    def next_batch(seed, rank=0):
        step = read_checkpoint(path)["step"]
        stream = reseed(seed, rank, step)
        mech.configure(64, 1024, 0, 1)
        mixer = make_mixer(settings, "cpu", stream, step, 1, 1.0)
        return mixer.take(32, 1024, "cpu"), (random.random(), np.random.random(), torch.rand(3))

    a, global_a = next_batch(11)
    b, global_b = next_batch(11)
    c, _ = next_batch(12)
    d, _ = next_batch(11, 1)
    torch.testing.assert_close(a, b, equal_nan=True, rtol=0, atol=0)
    assert not torch.allclose(a, c, equal_nan=True)
    assert not torch.allclose(a, d, equal_nan=True)
    assert global_a[:2] == global_b[:2]
    torch.testing.assert_close(global_a[2], global_b[2], rtol=0, atol=0)


def test_partial_checkpoint_is_never_selected(upstream, tmp_path, monkeypatch):
    from simons.training.reliability import latest_checkpoint
    from simons.training.trainer import write_checkpoint

    old = tmp_path / "checkpoint-slot-0.pt"
    write_checkpoint(old, checkpoint_payload(1))
    real_replace = Path.replace

    def killed(path, target):
        if str(target).endswith("checkpoint-slot-1.pt"):
            raise RuntimeError("interrupted before rename")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", killed)
    with pytest.raises(RuntimeError, match="interrupted"):
        write_checkpoint(tmp_path / "checkpoint-slot-1.pt", checkpoint_payload(2))
    assert latest_checkpoint(tmp_path) == old
    assert list(tmp_path.glob("*.tmp"))
    old.unlink()
    assert latest_checkpoint(tmp_path) is None


def test_leg_wiring_and_missing_predecessor(upstream, tmp_path):
    from simons.training.spot_job import LEGS, predecessor

    assert LEGS == {
        "stage1": None,
        "stage2": "stage1",
        "stage3": "stage2",
        "stage4": "stage3",
        "stage4-long": "stage3",
    }
    for leg, previous in LEGS.items():
        if previous is None:
            assert predecessor(tmp_path, leg) is None
        else:
            with pytest.raises(FileNotFoundError, match=previous):
                predecessor(tmp_path, leg)
    for leg in ("stage1", "stage2", "stage3"):
        directory = tmp_path / leg
        directory.mkdir()
        (directory / "completed.json").write_text(json.dumps({"checkpoint": "checkpoint-final.pt"}))
        from simons.training.trainer import write_checkpoint

        write_checkpoint(directory / "checkpoint-final.pt", checkpoint_payload(1))
    for leg, previous in LEGS.items():
        if previous:
            assert predecessor(tmp_path, leg).parent.name == previous


def test_microbatch_loss_and_gradient_parity(upstream):
    import copy

    import numpy as np
    import torch
    from simons.config import SimonsConfig
    from simons.model import Simons
    from simons.training.objective import packed_loss

    torch.manual_seed(17)
    model = Simons(
        SimonsConfig(
            width=24,
            heads=2,
            layers=1,
            patch_length=4,
            context_length=64,
            prediction_length=8,
            near_context=32,
            stats_window=32,
            period_slots=0,
        )
    )
    split = copy.deepcopy(model)
    x = torch.randn(3, 32, 40)
    x[0, 0, 12:16] = float("nan")
    ids = np.arange(96).reshape(3, 32)

    def slots():
        return ids, np.zeros_like(ids)

    torch.manual_seed(83)
    loss, count = packed_loss(model, x, slots, 0.4, 3)
    loss.backward()
    torch.manual_seed(83)
    small_loss, small_count = packed_loss(split, x, slots, 0.4, 3, micro_batches=2)
    torch.testing.assert_close(small_loss, loss, rtol=1e-6, atol=1e-7)
    assert count == small_count
    for full, small in zip(model.parameters(), split.parameters()):
        if full.grad is None:
            assert small.grad is None
        else:
            torch.testing.assert_close(small.grad, full.grad, rtol=3e-5, atol=2e-7)


def test_pin_mismatch_and_drift_fail_before_any_apply(tmp_path):
    import importlib.util
    import os
    import shutil

    source = os.environ.get("SIMONS_UPSTREAM")
    if not source or not (Path(source) / "src/simons").is_dir():
        pytest.skip("set SIMONS_UPSTREAM to the pinned source clone for patch checks")
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("prepare", root / "third_party/simons/prepare.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    clone = tmp_path / "upstream"
    for name in ("src", "configs", ".git"):
        shutil.copytree(Path(source) / name, clone / name)
    head = clone / ".git/HEAD"
    head.write_text("0" * 40 + "\n")
    original = (clone / "src/simons/training/trainer.py").read_text()
    with pytest.raises(ValueError, match="pin mismatch"):
        installer.prepare(clone, root / "third_party/simons")
    assert (clone / "src/simons/training/trainer.py").read_text() == original
    head.write_text(PIN + "\n")
    (clone / "src/simons/training/trainer.py").write_text("drift\n")
    with pytest.raises(subprocess.CalledProcessError):
        installer.prepare(clone, root / "third_party/simons")
    assert not (clone / "src/simons/training/reliability.py").exists()


def tiny_recipe(tmp_path, seed=19):
    import yaml

    raw = dict(
        model=dict(
            width=24,
            heads=2,
            layers=2,
            patch_length=32,
            context_length=384,
            prediction_length=32,
            near_context=128,
            stats_window=128,
            period_slots=0,
            covariates=True,
            binary_rows=True,
            read_after=0,
        ),
        training=dict(
            stage="stage1",
            steps=8,
            seed=seed,
            device="cpu",
            checkpoint_every=2,
            log_every=1,
            synthetic_rows=2,
            synthetic_context=128,
            synthetic_families="step",
            synthetic_family_weights=None,
            synthetic_pool_steps=1,
            synthetic_augment="",
            synthetic_observe="",
            synthetic_noise=0,
            synthetic_pad=0,
            synthetic_multi_fraction=0.5,
            role_phases="0.25,0.5",
            role_share="0.1,0.2,0.4",
            mech_bank=64,
            mech_length=256,
        ),
    )
    path = tmp_path / f"tiny-{seed}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_actual_training_resume_schedule_batches_and_rotation(upstream, tmp_path, monkeypatch):
    import torch
    from simons.training import objective
    from simons.training.covariates.roles import Hook
    from simons.training.trainer import read_checkpoint, train

    path = tiny_recipe(tmp_path)
    observed, batches = [], []
    actual_prepare = Hook.prepare
    actual_loss = objective.packed_loss

    def prepare(hook, step, *args, **kwargs):
        observed.append((step, hook.planner.p_role(), hook.p_eff(step), hook.stage_steps))
        return actual_prepare(hook, step, *args, **kwargs)

    def loss(model, sequence, *args, **kwargs):
        batches.append(sequence.clone())
        return actual_loss(model, sequence, *args, **kwargs)

    monkeypatch.setattr(Hook, "prepare", prepare)
    monkeypatch.setattr(objective, "packed_loss", loss)
    train(path, tmp_path / "uninterrupted", max_steps=5)
    original_schedule = observed.copy()
    observed.clear()
    state = tmp_path / "state"
    checkpoint = train(
        path, tmp_path / "interrupted", max_steps=3, checkpoint_dir=state, checkpoint_minutes=1e-12
    )
    assert read_checkpoint(checkpoint)["step"] == 3
    assert [p.name for p in sorted(state.glob("*.pt"))] == [
        "checkpoint-slot-0.pt",
        "checkpoint-slot-1.pt",
    ]
    observed.clear()
    batches.clear()
    a = train(path, tmp_path / "resume-a", resume=checkpoint, max_steps=2)
    schedule_a, batches_a = observed.copy(), batches.copy()
    assert schedule_a == original_schedule[3:5]
    observed.clear()
    batches.clear()
    b = train(path, tmp_path / "resume-b", resume=checkpoint, max_steps=2)
    assert observed == schedule_a
    for left, right in zip(batches_a, batches):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    weights_a, weights_b = read_checkpoint(a)["model"], read_checkpoint(b)["model"]
    for key in weights_a:
        torch.testing.assert_close(weights_a[key], weights_b[key], rtol=0, atol=0)
    batches.clear()
    train(tiny_recipe(tmp_path, seed=20), tmp_path / "other-seed", resume=checkpoint, max_steps=1)
    assert not torch.allclose(batches_a[0], batches[0], equal_nan=True)


def test_recipe_width_only_and_predecessor_paths(upstream, tmp_path, monkeypatch):
    import yaml
    from simons.training import spot_job

    def previous(state, leg):
        return Path("/weights") / spot_job.LEGS[leg]

    monkeypatch.setattr(spot_job, "predecessor", previous)
    for name in ("univariate", "covariates"):
        (tmp_path / name).mkdir()
    for leg in spot_job.LEGS:
        original = yaml.safe_load((upstream / f"configs/simons-25m-{leg}.yaml").read_text())
        raw = spot_job.recipe(upstream, tmp_path, tmp_path, "12m", leg, 11, 4)
        assert raw["model"] == {**original["model"], "width": 264}
        expected = dict(
            original["training"],
            seed=11,
            micro_batches=4,
            warm_start=None if leg == "stage1" else str(previous(tmp_path, leg)),
        )
        if leg in ("stage3", "stage4", "stage4-long"):
            expected.update(
                data=str(tmp_path / "univariate"), covariate_data=str(tmp_path / "covariates")
            )
        assert raw["training"] == expected


def test_qualification_measurement_counts_exactly_500(upstream, monkeypatch):
    import torch
    from simons.training import qualification

    clock = iter(float(i) for i in range(1600))
    monkeypatch.setattr(qualification.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(qualification, "host_rss", lambda: 1024)
    measured = qualification.Measurement(warmup=2)
    for step in range(502):
        measured.begin(torch.device("cpu"))
        measured.generated(torch.device("cpu"))
        measured.end(torch.device("cpu"), step + 1, 100, auxiliary_wait=0.1)
    report = measured.result()
    assert report["measured_updates"] == 500
    assert report["supervised_values"] == 50000
    assert report["first_step"] == 2 and report["last_step"] == 502
    assert report["updates_per_second"] == 0.5
    assert report["loader_generator_wait_share"] == pytest.approx(0.55)


def test_qualification_oom_retries_and_limit(upstream, tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from simons.training import spot_job

    args = SimpleNamespace(run_dir=tmp_path, name="probe", micro_batches=2, max_micro_batches=8)
    monkeypatch.setattr(sys, "argv", ["spot_job", "--leg", "qualify"])
    calls = []

    def run(command, **kwargs):
        calls.append(int(command[command.index("--micro-batches") + 1]))
        return SimpleNamespace(returncode=42)

    monkeypatch.setattr(spot_job.subprocess, "run", run)
    assert spot_job.qualify(args) == 1
    assert calls == [2, 4, 8]
    report = json.loads((tmp_path / "state/qualify/probe.json").read_text())
    assert report["status"] == "oom" and len(report["attempts"]) == 3


def test_completed_leg_noop_and_final_checkpoint_recovers_marker(upstream, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from simons.training import spot_job

    state = tmp_path / "state/stage2"
    state.mkdir(parents=True)
    from simons.training.trainer import write_checkpoint

    checkpoint = state / "checkpoint-final.pt"
    write_checkpoint(checkpoint, checkpoint_payload(8))
    (state / "completed.json").write_text(
        json.dumps({"checkpoint": checkpoint.name, "bytes": checkpoint.stat().st_size})
    )
    assert spot_job.run_leg(SimpleNamespace(run_dir=tmp_path, leg="stage2")) == 0


def test_completed_shell_exits_before_install(jobs, tmp_path):
    import os

    script = jobs.generate(
        size="25m",
        leg="stage2",
        seed=11,
        micro_batches=1,
        checkpoint_minutes=10,
        repo_sha=SHA,
        upstream_pin=PIN,
    )
    marker = tmp_path / "state/stage2/completed.json"
    marker.parent.mkdir(parents=True)
    checkpoint = marker.parent / "checkpoint-final.pt"
    marker.write_text(json.dumps(shell_checkpoint_record(checkpoint)))
    path = tmp_path / "job.sh"
    path.write_text(script)
    result = subprocess.run(
        ["bash", str(path)],
        capture_output=True,
        text=True,
        env={**os.environ, "RUN_DIR": str(tmp_path), "STATE_URI": "gs://offline"},
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.glob("simons-install-*"))


def test_export_averages_stage4_and_long(upstream, tmp_path):
    from types import SimpleNamespace

    import torch
    from simons.config import SimonsConfig
    from simons.model import Simons
    from simons.training.spot_job import run_export
    from simons.training.trainer import write_checkpoint

    model = Simons(
        SimonsConfig(
            width=24,
            heads=2,
            layers=1,
            patch_length=4,
            context_length=64,
            prediction_length=8,
            near_context=32,
            stats_window=32,
            period_slots=0,
        )
    )
    original = {k: v.clone() for k, v in model.state_dict().items()}
    for leg, offset in (("stage4", 0), ("stage4-long", 2)):
        directory = tmp_path / "state" / leg
        directory.mkdir(parents=True)
        payload = dict(
            format="simons-train-1",
            step=8,
            training={},
            model_config=model.config.to_dict(),
            model={k: v + offset for k, v in original.items()},
        )
        write_checkpoint(directory / "checkpoint-final.pt", payload)
        (directory / "completed.json").write_text(
            json.dumps(dict(checkpoint="checkpoint-final.pt"))
        )
    assert run_export(SimpleNamespace(run_dir=tmp_path)) == 0
    from safetensors.torch import load_file

    weights = load_file(tmp_path / "state/export/model.safetensors")
    for name, weight in weights.items():
        torch.testing.assert_close(weight, original[name] + 1)
    assert (tmp_path / "state/export/config.json").is_file()


def test_wall_clock_checkpoint_occurs_before_final_save(upstream, tmp_path, monkeypatch):
    from simons.training import trainer

    saved = []
    actual = trainer.save_checkpoint

    def save(path, *args):
        saved.append(path.name)
        return actual(path, *args)

    monkeypatch.setattr(trainer, "save_checkpoint", save)
    trainer.train(
        tiny_recipe(tmp_path),
        tmp_path / "output",
        max_steps=3,
        checkpoint_dir=tmp_path / "state",
        checkpoint_minutes=1e-12,
    )
    assert saved == [
        "checkpoint-slot-0.pt",
        "checkpoint-slot-1.pt",
        "checkpoint-slot-0.pt",
    ]


def test_resume_log_and_finished_checkpoint_marker(upstream, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import yaml
    from simons.training import spot_job
    from simons.training.reliability import resume_record
    from simons.training.trainer import write_checkpoint

    raw = yaml.safe_load(tiny_recipe(tmp_path).read_text())
    args = SimpleNamespace(
        run_dir=tmp_path,
        leg="stage1",
        size="25m",
        seed=19,
        root=upstream,
        corpus=None,
        corpus_uri=None,
        micro_batches=1,
        checkpoint_minutes=10,
        repo_sha=SHA,
        upstream_pin=PIN,
        bundle_sha256="test",
    )
    state = tmp_path / "state/stage1"
    state.mkdir(parents=True)
    spot_job.atomic_json(state / "identity.json", spot_job.identity(args, raw))
    write_checkpoint(state / "checkpoint-final.pt", checkpoint_payload(8))
    assert spot_job.run_leg(args) == 0
    assert json.loads((state / "completed.json").read_text())["step"] == 8
    resume_record(state, 8)
    records = [json.loads(line) for line in (state / "resume-log.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert all(r["step"] == 8 and r["utc"] and r["host"] for r in records)


def test_invalid_newest_checkpoint_falls_back(upstream, tmp_path):
    from simons.training.reliability import latest_checkpoint
    from simons.training.trainer import write_checkpoint

    old = tmp_path / "checkpoint-slot-0.pt"
    write_checkpoint(old, checkpoint_payload(1))
    broken = tmp_path / "checkpoint-slot-1.pt"
    broken.write_bytes(b"not a checkpoint")
    assert latest_checkpoint(tmp_path) == old
    write_checkpoint(broken, {"format": "simons-train-1", "step": 2})
    assert latest_checkpoint(tmp_path) == old


def test_qualification_absolute_phase_selection(upstream):
    from simons.training import config
    from simons.training.spot_job import origin, probe_start

    model, settings = config.load(upstream / "configs/simons-25m-stage3.yaml")
    assert origin(model, settings) == 26392
    assert probe_start(model, settings, 41000, 520) == 41000
    assert probe_start(model, settings, None, 520) == 26392
    with pytest.raises(ValueError, match="origin"):
        probe_start(model, settings, 0, 520)
    with pytest.raises(ValueError, match="500"):
        probe_start(model, settings, 57500, 520)


def checkpoint_payload(step):
    return dict(
        format="simons-train-1",
        step=step,
        model={},
        optimizer={},
        model_config={},
        training={},
        elapsed_seconds=0.0,
    )


def test_probe_execution_keeps_original_role_clock(upstream, tmp_path):
    from simons.training.qualification import Measurement
    from simons.training.trainer import read_checkpoint, train

    measurement = Measurement(warmup=0, updates=1)
    checkpoint = train(
        tiny_recipe(tmp_path), tmp_path / "probe", max_steps=1, observer=measurement, probe_step=3
    )
    assert read_checkpoint(checkpoint)["step"] == 4
    assert measurement.result()["first_step"] == 3


def test_effect_worker_stream_rebuilt_deterministically(upstream, monkeypatch):
    import torch
    from simons.synth import augment
    from simons.training.covariates import roles
    from simons.training.covariates.roles import EffectRows, settings
    from simons.training.reliability import reseed, seed_worker

    augment.configure("", 0, None)
    spec = settings(
        dict(
            base_families="step",
            base_weights="1",
            synth_calendar=0,
            effect_mix={"generic": 1},
            effect_horizons=[1],
            effect_pad=0,
        )
    )

    def draw(seed):
        stream = reseed(seed, 0, 300)
        from types import SimpleNamespace

        monkeypatch.setattr(roles, "get_worker_info", lambda: SimpleNamespace(id=1))
        torch.manual_seed(stream + 1)
        seed_worker(1)
        return next(iter(EffectRows(spec, stream, 300, 1024)))["values"]

    left, right, other = draw(17), draw(17), draw(18)
    torch.testing.assert_close(left, right, equal_nan=True, rtol=0, atol=0)
    assert not torch.allclose(left, other, equal_nan=True)


def test_pin_check_includes_staged_source_drift(tmp_path):
    import difflib
    import importlib.util
    import os
    import shutil

    source = os.environ.get("SIMONS_UPSTREAM")
    if not source or not (Path(source) / "src/simons").is_dir():
        pytest.skip("set SIMONS_UPSTREAM to the pinned source clone for staged-drift checks")
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "prepare_staged", root / "third_party/simons/prepare.py"
    )
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    clone = tmp_path / "upstream"
    for name in ("src", "configs", ".git"):
        shutil.copytree(Path(source) / name, clone / name)
    relative = "src/simons/training/optim.py"
    path = clone / relative
    original = path.read_text()
    changed = original + "\n# Staged drift outside the reviewed patches.\n"
    patch = "".join(
        difflib.unified_diff(
            original.splitlines(True),
            changed.splitlines(True),
            fromfile="a/" + relative,
            tofile="b/" + relative,
        )
    )
    # Only this disposable upstream copy's index is mutated, never our branch's index.
    subprocess.run(
        ["git", "-C", str(clone), "apply", "--cached", "-"],
        input=patch,
        text=True,
        check=True,
        capture_output=True,
    )
    path.write_text(changed)
    subprocess.run(
        ["git", "-C", str(clone), "diff", "--exit-code", "--", "src", "configs"],
        check=True,
        capture_output=True,
    )
    with pytest.raises(subprocess.CalledProcessError):
        installer.prepare(clone, root / "third_party/simons")
    assert not (clone / "src/simons/training/reliability.py").exists()


def test_warm_origin_and_learning_rate_survive_resume(upstream, tmp_path, monkeypatch):
    import yaml
    from simons.training import schedule
    from simons.training.trainer import train

    path = tiny_recipe(tmp_path)
    initial = train(path, tmp_path / "initial", max_steps=1)
    raw = yaml.safe_load(path.read_text())
    raw["training"].update(warm_start=str(initial), warm_start_step=2, rewarm_steps=6)
    path.write_text(yaml.safe_dump(raw))
    calls = []
    actual = schedule.multiplier

    def rate(step, settings, first, warm):
        value = actual(step, settings, first, warm)
        calls.append((step, first, warm, value))
        return value

    monkeypatch.setattr(schedule, "multiplier", rate)
    train(path, tmp_path / "warm-full", max_steps=4)
    full = calls.copy()
    calls.clear()
    checkpoint = train(path, tmp_path / "warm-partial", max_steps=2)
    calls.clear()
    train(path, tmp_path / "warm-resume", resume=checkpoint, max_steps=2)
    assert calls == full[2:]
    assert [(step, first, warm) for step, first, warm, _ in calls] == [(4, 2, True), (5, 2, True)]


def test_embedded_bundle_extracts_without_repo_or_network(jobs, tmp_path):
    import os

    script = jobs.generate(
        size="12m",
        leg="stage1",
        seed=11,
        micro_batches=1,
        checkpoint_minutes=10,
        repo_sha=SHA,
        upstream_pin=PIN,
    )
    prefix = script.split('upstream="$scratch/Simons"', 1)[0]
    path = tmp_path / "extract.sh"
    path.write_text(prefix)
    subprocess.run(
        ["bash", str(path)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "RUN_DIR": str(tmp_path), "STATE_URI": "gs://offline"},
    )
    extracted = next(tmp_path.glob("simons-install-*/bundle"))
    bundle = Path(__file__).resolve().parents[2] / "third_party/simons"
    for relative in [
        "PINNED",
        "NOTICE",
        "prepare.py",
        *[str(p.relative_to(bundle)) for p in sorted((bundle / "patches").glob("*.patch"))],
    ]:
        assert (extracted / relative).read_bytes() == (bundle / relative).read_bytes()


def test_checkpoint_recovers_identity_missing_from_restore(upstream, tmp_path):
    from types import SimpleNamespace

    import yaml
    from simons.training import spot_job
    from simons.training.trainer import read_checkpoint, train

    path = tiny_recipe(tmp_path)
    args = SimpleNamespace(
        run_dir=tmp_path,
        leg="stage1",
        size="25m",
        seed=19,
        root=upstream,
        corpus=None,
        corpus_uri=None,
        micro_batches=1,
        checkpoint_minutes=10,
        repo_sha=SHA,
        upstream_pin=PIN,
        bundle_sha256="test",
    )
    identity = spot_job.identity(args, yaml.safe_load(path.read_text()))
    state = tmp_path / "state/stage1"
    checkpoint = train(
        path,
        tmp_path / "first-attempt",
        max_steps=2,
        checkpoint_dir=state,
        checkpoint_metadata=identity,
    )
    assert read_checkpoint(checkpoint)["job_identity"] == identity
    assert not (state / "identity.json").exists()
    (state / "completed.json").write_text(
        json.dumps({"checkpoint": "checkpoint-final.pt", "bytes": 4096})
    )
    assert spot_job.run_leg(args) == 0
    assert json.loads((state / "identity.json").read_text()) == identity
    assert json.loads((state / "completed.json").read_text())["step"] == 8


def test_incomplete_uploaded_marker_does_not_skip_job(jobs, tmp_path):
    import os

    script = jobs.generate(
        size="25m",
        leg="stage2",
        seed=11,
        micro_batches=1,
        checkpoint_minutes=10,
        repo_sha=SHA,
        upstream_pin=PIN,
    )
    prefix = script.split("scratch=$(mktemp", 1)[0]
    marker = tmp_path / "state/stage2/completed.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"checkpoint": "checkpoint-final.pt", "bytes": 4096}))
    path = tmp_path / "preflight.sh"
    path.write_text(prefix + "printf 'continue\\n'\n")
    result = subprocess.run(
        ["bash", str(path)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "RUN_DIR": str(tmp_path), "STATE_URI": "gs://offline"},
    )
    assert result.stdout == "continue\n"


def test_snapshot_slots_stay_bounded_across_uploads(upstream, tmp_path, monkeypatch):
    import shutil

    from simons.training import trainer
    from simons.training.reliability import latest_checkpoint

    state = tmp_path / "state/stage1"
    remote = tmp_path / "bucket/stage1"
    remote.mkdir(parents=True)
    identity = {"seed": 19, "leg": "stage1", "recipe": "tiny"}
    steps = []
    actual = trainer.save_checkpoint

    def upload(path, *args):
        actual(path, *args)
        saved = trainer.read_checkpoint(path)
        steps.append(saved["step"])
        assert saved["training"]["seed"] == 19
        assert saved["job_identity"] == identity
        # Model the guest's copy: old remote objects are never deleted.
        for source in state.glob("*.pt*"):
            shutil.copy2(source, remote / source.name)
        assert {p.name for p in remote.iterdir()} <= {
            "checkpoint-slot-0.pt",
            "checkpoint-slot-1.pt",
            "checkpoint-final.pt",
        }
        assert trainer.read_checkpoint(latest_checkpoint(remote))["step"] == saved["step"]

    monkeypatch.setattr(trainer, "save_checkpoint", upload)
    final = trainer.train(
        tiny_recipe(tmp_path),
        tmp_path / "output",
        checkpoint_dir=state,
        checkpoint_minutes=1e-12,
        checkpoint_metadata=identity,
    )
    assert steps == list(range(1, 9))
    assert {p.name for p in remote.iterdir()} == {
        "checkpoint-slot-0.pt",
        "checkpoint-slot-1.pt",
        "checkpoint-final.pt",
    }
    assert final.name == "checkpoint-final.pt"
    assert list(state.glob("*.pt")) == [final]


def test_slot_selection_uses_embedded_step_not_name_or_time(upstream, tmp_path):
    import os

    from simons.training.reliability import latest_checkpoint
    from simons.training.trainer import write_checkpoint

    newer = tmp_path / "checkpoint-slot-0.pt"
    stale = tmp_path / "checkpoint-slot-1.pt"
    write_checkpoint(newer, checkpoint_payload(12))
    write_checkpoint(stale, checkpoint_payload(3))
    os.utime(newer, (1, 1))
    os.utime(stale, (10000, 10000))
    assert latest_checkpoint(tmp_path) == newer
    # A temporary slot copied mid-write is never a candidate.
    write_checkpoint(tmp_path / "checkpoint-slot-1.pt.tmp", checkpoint_payload(99))
    assert latest_checkpoint(tmp_path) == newer


@pytest.mark.parametrize("damage", ["truncate", "hash"])
def test_damaged_slot_falls_back_to_valid_snapshot(upstream, tmp_path, damage):
    from simons.training.reliability import latest_checkpoint
    from simons.training.trainer import read_checkpoint, write_checkpoint

    valid = tmp_path / "checkpoint-slot-0.pt"
    broken = tmp_path / "checkpoint-slot-1.pt"
    write_checkpoint(valid, checkpoint_payload(12))
    write_checkpoint(broken, checkpoint_payload(15))
    assert latest_checkpoint(tmp_path) == broken
    data = broken.read_bytes()
    if damage == "truncate":
        broken.write_bytes(data[: len(data) // 2])
    else:
        # Change only the stored digest: torch serialization remains readable.
        broken.write_bytes(data[:-1] + (b"0" if data[-1:] != b"0" else b"1"))
    with pytest.raises((ValueError, RuntimeError, EOFError)):
        read_checkpoint(broken)
    assert latest_checkpoint(tmp_path) == valid


@pytest.mark.parametrize("leg", ["stage3", "stage4", "stage4-long"])
@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("missing", ["univariate", "covariates"])
def test_missing_corpus_fails_before_training(
    upstream, tmp_path, monkeypatch, leg, resumed, missing
):
    from types import SimpleNamespace

    import yaml
    from simons.training import spot_job
    from simons.training.trainer import write_checkpoint

    corpus = tmp_path / "corpus"
    (corpus / ({"univariate", "covariates"} - {missing}).pop()).mkdir(parents=True)
    args = SimpleNamespace(
        run_dir=tmp_path,
        root=upstream,
        leg=leg,
        size="25m",
        seed=19,
        corpus=corpus,
        corpus_uri="gs://offline/corpus",
        micro_batches=1,
        checkpoint_minutes=10,
        repo_sha=SHA,
        upstream_pin=PIN,
        bundle_sha256="test",
    )
    monkeypatch.setattr(spot_job, "predecessor", lambda *_: Path("/weights/previous.pt"))
    if resumed:
        raw = yaml.safe_load(tiny_recipe(tmp_path).read_text())
        identity = spot_job.identity(args, raw)
        state = tmp_path / "state" / leg
        state.mkdir(parents=True)
        payload = checkpoint_payload(2)
        payload["job_identity"] = identity
        write_checkpoint(state / "checkpoint-slot-0.pt", payload)
    monkeypatch.setattr(spot_job, "train", lambda *a, **kw: pytest.fail("training started"))
    with pytest.raises(FileNotFoundError, match=f"missing corpus directory .*{missing}"):
        spot_job.run_leg(args)


def shell_checkpoint_record(path):
    import hashlib

    payload = b"committed checkpoint stand-in"
    digest = hashlib.sha256(payload).hexdigest()
    path.write_bytes(payload + b"\nSIMONS-SHA256:" + digest.encode())
    return {"checkpoint": path.name, "bytes": path.stat().st_size, "sha256": digest}


def test_corrupt_final_does_not_skip_job(jobs, tmp_path):
    import os

    script = jobs.generate(
        size="25m",
        leg="stage2",
        seed=11,
        micro_batches=1,
        checkpoint_minutes=10,
        repo_sha=SHA,
        upstream_pin=PIN,
    )
    marker = tmp_path / "state/stage2/completed.json"
    marker.parent.mkdir(parents=True)
    checkpoint = marker.parent / "checkpoint-final.pt"
    marker.write_text(json.dumps(shell_checkpoint_record(checkpoint)))
    payload = checkpoint.read_bytes()
    checkpoint.write_bytes(b"X" + payload[1:])  # Same size, incorrect content hash.
    path = tmp_path / "preflight.sh"
    path.write_text(script.split("scratch=$(mktemp", 1)[0] + "printf 'continue\\n'\n")
    result = subprocess.run(
        ["bash", str(path)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "RUN_DIR": str(tmp_path), "STATE_URI": "gs://offline"},
    )
    assert result.stdout == "continue\n"


def test_resumed_save_preserves_newest_slot_until_publish(upstream, tmp_path, monkeypatch):
    from simons.training.reliability import latest_checkpoint
    from simons.training.trainer import read_checkpoint, train

    recipe = tiny_recipe(tmp_path)
    state = tmp_path / "state/stage1"
    first = train(
        recipe,
        tmp_path / "first",
        max_steps=3,
        checkpoint_dir=state,
        checkpoint_minutes=1e-12,
    )
    assert first.name == "checkpoint-slot-0.pt"
    replace = Path.replace

    def killed(path, target):
        if path.name.endswith(".pt.tmp"):
            assert target.name == "checkpoint-slot-1.pt"
            raise RuntimeError("interrupted before slot publication")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", killed)
    with pytest.raises(RuntimeError, match="slot publication"):
        train(recipe, tmp_path / "interrupted", resume=first, max_steps=1, checkpoint_dir=state)
    assert latest_checkpoint(state) == first
    assert read_checkpoint(first)["step"] == 3
    monkeypatch.setattr(Path, "replace", replace)
    resumed = train(recipe, tmp_path / "resumed", resume=first, max_steps=1, checkpoint_dir=state)
    assert resumed.name == "checkpoint-slot-1.pt"
    assert read_checkpoint(latest_checkpoint(state))["step"] == 4


@pytest.mark.parametrize("leg", ["stage3", "stage4", "stage4-long"])
def test_resumed_leg_uses_public_corpus_layout(upstream, tmp_path, monkeypatch, leg):
    from types import SimpleNamespace

    import yaml
    from simons.training import spot_job
    from simons.training.trainer import write_checkpoint

    corpus = tmp_path / "corpus"
    for name in ("univariate", "covariates"):
        (corpus / name / "dataset").mkdir(parents=True)
    args = SimpleNamespace(
        run_dir=tmp_path,
        root=upstream,
        leg=leg,
        size="25m",
        seed=19,
        corpus=corpus,
        corpus_uri="gs://offline/corpus",
        micro_batches=1,
        checkpoint_minutes=10,
        repo_sha=SHA,
        upstream_pin=PIN,
        bundle_sha256="test",
    )
    raw = yaml.safe_load(tiny_recipe(tmp_path).read_text())
    identity = spot_job.identity(args, raw)
    state = tmp_path / "state" / leg
    state.mkdir(parents=True)
    payload = checkpoint_payload(2)
    payload["job_identity"] = identity
    write_checkpoint(state / "checkpoint-slot-0.pt", payload)

    def train(path, output, **kwargs):
        recipe = yaml.safe_load(path.read_text())["training"]
        assert recipe["data"] == str(corpus / "univariate")
        assert recipe["covariate_data"] == str(corpus / "covariates")
        final = kwargs["checkpoint_dir"] / "checkpoint-final.pt"
        payload["step"] = 8
        write_checkpoint(final, payload)
        return final

    monkeypatch.setattr(spot_job, "train", train)
    assert spot_job.run_leg(args) == 0
    marker = json.loads((state / "completed.json").read_text())
    assert marker["checkpoint"] == "checkpoint-final.pt"
    assert marker["step"] == 8 and len(marker["sha256"]) == 64
    assert [p.name for p in state.glob("*.pt")] == ["checkpoint-final.pt"]
