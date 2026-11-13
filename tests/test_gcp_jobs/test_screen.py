"""Offline acceptance for the registered short chain, audit gate and exposure lineage."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_jobs import PIN, SHA, tiny_recipe


def release(corpus):
    corpus.mkdir(exist_ok=True)
    for name in ("univariate", "covariates"):
        (corpus / name).mkdir(exist_ok=True)
    status = dict(attempt_id="attempt-1", complete=True, exit_code=0)
    audit = dict(
        attempt_id="attempt-1",
        valid=True,
        exit_code=0,
        folders=[dict(folder="tiny", status="allowed", verdict="clean")],
    )
    (corpus / "attempt-status.json").write_text(json.dumps(status))
    (corpus / "audit.json").write_text(json.dumps(audit))
    return status, audit


@pytest.mark.parametrize("size", ["12m", "25m"])
@pytest.mark.parametrize("budget", [25000, 30000])
def test_screen_clocks(upstream, tmp_path, monkeypatch, size, budget):
    from simons.training import config, schedule, spot_job

    monkeypatch.setattr(spot_job, "predecessor", lambda state, leg: tmp_path / spot_job.LEGS[leg])
    release(tmp_path)
    lengths, recipes = [], []
    for leg in spot_job.LEGS:
        raw = spot_job.recipe(
            upstream,
            tmp_path,
            tmp_path,
            size,
            leg,
            11,
            2,
            recipe_name="screen",
            screen_updates=budget,
        )
        recipes.append(raw)
        path = spot_job.write_recipe(raw, tmp_path)
        model, s = config.load(path)
        lengths.append(s["steps"] - spot_job.origin(model, s))
        if leg in ("stage4", "stage4-long"):
            assert s["warm_start"] == str(tmp_path / "stage3")
        original = __import__("yaml").safe_load(
            (upstream / f"configs/simons-25m-{leg}.yaml").read_text()
        )
        for key in (
            "mech_refresh",
            "mech_bank",
            "synthetic_pool_steps",
            "synthetic_pool_steps_long",
        ):
            assert s[key] == original["training"].get(key, config.DEFAULTS[key])
    assert sum(lengths) == budget

    def scale(n):
        return round(n * budget / 120208)

    assert [lengths[i] for i in (0, 1, 3, 4)] == [scale(n) for n in (65000, 12000, 4000, 8000)]
    assert recipes[2]["training"]["steps"] == scale(57600)
    assert recipes[2]["training"]["warm_start_step"] == scale(57600) - lengths[2]
    assert recipes[1]["training"]["role_phases"] == "0.27,0.53"
    if budget == 25000:
        assert lengths == [13518, 2496, 6490, 832, 1664]
        assert recipes[0]["training"]["warmup_steps"] == 270
        assert recipes[0]["training"]["decay_steps"] == 2028
        assert recipes[1]["training"]["rewarm_steps"] == 125
        assert recipes[1]["training"]["decay_steps"] == 374
        for raw in recipes[2:]:
            s = config.load(spot_job.write_recipe(raw, tmp_path))[1]
            assert s["steps"] == 11979
            assert s["rewarm_steps"] == 62
            assert s["synthetic_ramp_steps"] == 417
            phases = schedule.parse_schedule(s["context_schedule"], s["steps"], 32768, 32)
            assert phases == [(8192, 0, 5489), (7680, 5489, 8310), (32768, 8310, 11979)]
            assert s["warmup_fraction"] == 0.05 and s["decay_fraction"] == 0.5418


@pytest.mark.parametrize(
    "failure", ["missing", "incomplete", "exit", "mismatch", "invalid", "dirty", "empty"]
)
def test_audit_gate(upstream, tmp_path, failure):
    from simons.training.screen import corpus_release

    status, audit = release(tmp_path)
    if failure == "missing":
        (tmp_path / "attempt-status.json").unlink()
    else:
        if failure == "incomplete":
            status["complete"] = False
        if failure == "exit":
            status["exit_code"] = 2
        if failure == "mismatch":
            audit["attempt_id"] = "old"
        if failure == "invalid":
            audit["valid"] = False
        if failure == "dirty":
            audit["folders"][0]["verdict"] = "inconclusive"
        if failure == "empty":
            audit["folders"] = []
        (tmp_path / "attempt-status.json").write_text(json.dumps(status))
        (tmp_path / "audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="corpus audit gate"):
        corpus_release(tmp_path)
    assert corpus_release(tmp_path, qualification_only=True)["qualification_only"] is True


def test_audit_receipt(upstream, tmp_path):
    from simons.training.screen import corpus_release

    release(tmp_path)
    assert corpus_release(tmp_path) == dict(
        attempt_id="attempt-1",
        audit_sha256=hashlib.sha256((tmp_path / "audit.json").read_bytes()).hexdigest(),
    )


def test_predecessor_identity_legacy_and_screen(upstream):
    from simons.training.screen import check_identity

    legacy = dict(size="25m", seed=11, leg="stage1", upstream_pin=PIN, recipe={})
    args = SimpleNamespace(
        size="25m", seed=11, recipe="full", screen_updates=None, upstream_pin=PIN
    )
    check_identity(legacy, args)
    args.recipe, args.screen_updates = "screen", 25000
    with pytest.raises(ValueError, match="identity"):
        check_identity(legacy, args)
    screen = {**legacy, "recipe_name": "screen", "screen_updates": 25000}
    check_identity(screen, args)
    args.recipe = "full"
    with pytest.raises(ValueError, match="identity"):
        check_identity(screen, args)
    with pytest.raises(ValueError, match="qualification"):
        check_identity({**legacy, "qualification_only": True}, args)


@pytest.mark.parametrize(
    "size,recipe",
    [("12m", "full"), ("25m", "full"), ("60m", "full"), ("12m", "screen"), ("25m", "screen")],
)
def test_all_shells(jobs, tmp_path, size, recipe):
    import subprocess

    for leg in jobs.LEGS:
        script = jobs.generate(
            size=size,
            leg=leg,
            seed=11,
            micro_batches=2,
            checkpoint_minutes=10,
            repo_sha=SHA,
            upstream_pin=PIN,
            corpus_uri="gs://offline/corpus",
            recipe=recipe,
        )
        path = tmp_path / "job.sh"
        path.write_text(script)
        assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


@pytest.mark.parametrize(
    "leg", ["stage1", "stage2", "stage3", "stage4", "stage4-long", "qualify", "export"]
)
def test_generator_refuses_unregistered_60m_screen(jobs, leg):
    with pytest.raises(ValueError, match="60M short schedule needs a registered protocol"):
        jobs.generate(
            size="60m",
            leg=leg,
            seed=11,
            micro_batches=2,
            checkpoint_minutes=10,
            repo_sha=SHA,
            upstream_pin=PIN,
            corpus_uri="gs://offline/corpus",
            recipe="screen",
        )


@pytest.mark.parametrize(
    "leg", ["stage1", "stage2", "stage3", "stage4", "stage4-long", "qualify", "export"]
)
def test_guest_refuses_unregistered_60m_screen(upstream, tmp_path, monkeypatch, capsys, leg):
    import sys

    from simons.training import spot_job

    def forbidden(*args):
        pytest.fail("unsupported schedule reached execution")

    for name in ("run_leg", "qualify", "run_export"):
        monkeypatch.setattr(spot_job, name, forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spot_job",
            f"--leg={leg}",
            "--size=60m",
            "--recipe=screen",
            f"--root={upstream}",
            f"--run-dir={tmp_path / 'run'}",
            "--seed=11",
            f"--repo-sha={SHA}",
            f"--upstream-pin={PIN}",
            "--bundle-sha256=test",
        ],
    )
    with pytest.raises(SystemExit) as error:
        spot_job.main()
    assert error.value.code == 2
    assert "60M short schedule needs a registered protocol" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("leg", ["stage1", "stage2", "stage3", "stage4", "stage4-long"])
def test_recipe_refuses_unregistered_60m_screen(upstream, tmp_path, leg):
    from simons.training import spot_job

    with pytest.raises(ValueError, match="60M short schedule needs a registered protocol"):
        spot_job.recipe(upstream, tmp_path, None, "60m", leg, 11, 2, recipe_name="screen")


def test_override_is_only_for_qualification(jobs):
    with pytest.raises(ValueError, match="qualification"):
        jobs.generate(
            size="25m",
            leg="stage3",
            seed=11,
            micro_batches=2,
            checkpoint_minutes=10,
            repo_sha=SHA,
            upstream_pin=PIN,
            corpus_uri="gs://offline/corpus",
            unsafe_qualification_allow_unaudited_corpus=True,
        )


@pytest.mark.parametrize("synthetic_rows", [1, 2])
def test_exposure_masks_and_microbatch_invariance(upstream, synthetic_rows):
    import numpy as np
    import torch
    from simons.config import SimonsConfig
    from simons.model import Simons
    from simons.training.exposure import Counts
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
    sequence = torch.ones(2, 32, 16)
    sequence[0, 0, 0] = float("nan")
    ids = np.arange(64).reshape(2, 32)

    def slots():
        return ids, np.zeros_like(ids)

    totals = []
    for micro in (1, 2):
        counts = Counts()
        packed_loss(
            model,
            sequence,
            slots,
            0,
            1,
            micro_batches=micro,
            exposure=counts,
            synthetic_rows=synthetic_rows,
        )
        totals.append(counts.snapshot())
    assert totals[0] == totals[1]
    assert totals[0]["E_real"] == (511 if synthetic_rows == 1 else 0)
    assert totals[0]["E_synthetic"] == (512 if synthetic_rows == 1 else 1023)
    assert totals[0]["S_real"] == (384 if synthetic_rows == 1 else 0)
    assert totals[0]["S_synthetic"] == 384 * synthetic_rows


def test_role_exposure_separates_covariates_and_procedural_rows(upstream):
    import numpy as np
    import torch
    from simons.config import SimonsConfig
    from simons.model import Simons
    from simons.synth.covariates.families import FAMILIES
    from simons.training.covariates.roles import Hook
    from simons.training.exposure import Counts
    from simons.training.objective import packed_loss

    torch.manual_seed(17)
    model = Simons(
        SimonsConfig(
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
        )
    )
    sequence = torch.arange(128, dtype=torch.float32).expand(3, 32, -1).clone()
    ids = np.arange(96).reshape(3, 32)
    synthetic_meta = torch.zeros(1, 32, 3, dtype=torch.long)
    synthetic_meta[0, :4, 0] = 1
    effect_meta = torch.zeros(32, 3, dtype=torch.long)
    effect_meta[:8, 0] = 1
    hook = Hook({}, 11, 0, "cpu", 100, 128)
    # Deterministic effect contents exercise the real prepare/tagging path.
    hook.effects = iter(
        [
            dict(
                values=sequence[0],
                meta=effect_meta,
                group=torch.arange(32),
                first=torch.zeros(32, dtype=torch.long),
                families=torch.zeros(len(FAMILIES), dtype=torch.long),
            )
        ]
    )
    hook.p_eff = lambda step: 1 / 3
    mixer = SimpleNamespace(fraction=1 / 3, last_meta=lambda: synthetic_meta)

    def slots():
        return ids.copy(), np.zeros_like(ids)

    sequence, _, slots, roles = hook.prepare(0, mixer, sequence, torch.from_numpy(ids), slots)
    assert roles.role.sum().item() == 12
    assert roles.exposure_synthetic[0].all()
    assert not roles.exposure_synthetic[1:].any()
    totals = []
    for micro in (1, 3):
        counts = Counts()
        packed_loss(
            model,
            sequence,
            slots,
            0,
            1,
            roles=roles,
            micro_batches=micro,
            exposure=counts,
            synthetic_rows=1,
        )
        totals.append(counts.snapshot())
    assert totals[0] == totals[1]
    # Row 0 is a procedural effect, row 1 real, row 2 the synthetic mixer tail.
    # Covariates shift left one patch and never supply target E or supervised S.
    assert totals[0]["E_real"] == 32 * 128
    assert totals[0]["E_synthetic"] == (24 + 28) * 128
    assert totals[0]["S_real"] == 32 * 96
    assert totals[0]["S_synthetic"] == (24 + 28) * 96
    assert totals[0]["covariate_values"] == (8 + 4) * 96
    assert totals[0]["channel_windows"] == 32 + 24 + 28


def test_auxiliary_real_derived_and_positive_weights(upstream):
    import torch
    from simons.training.exposure import Counts, auxiliary_counts

    batch = dict(
        values=torch.ones(3, 1, 96),
        covs=torch.ones(3, 2, 96),
        future=torch.zeros(3, 96, dtype=torch.bool),
        kinds={"synthetic": 1, "sibling": 1, "real": 1},
    )
    batch["future"][:, 64:] = True
    # Auxiliary 'synthetic' uses corpus bases: all three kinds have real ancestry.
    counts = Counts()
    auxiliary_counts(counts, batch, 0.1, 0.025)
    assert counts.snapshot()["E_real"] == 288
    assert counts.snapshot()["S_real"] == 192
    assert counts.snapshot()["covariate_values"] == 576
    assert counts.snapshot()["E_synthetic"] == 0
    zero = Counts()
    auxiliary_counts(zero, batch, 0.1, 0)
    assert zero.snapshot()["E_real"] == 0
    future_only = Counts()
    auxiliary_counts(future_only, batch, 0, 0.025)
    assert future_only.snapshot()["S_real"] == 96


def test_counter_resume_continuity(upstream, tmp_path):
    from simons.training.trainer import read_checkpoint, train

    path = tiny_recipe(tmp_path)
    checkpoint = train(path, tmp_path / "first", max_steps=2, checkpoint_dir=tmp_path / "state")
    saved = read_checkpoint(checkpoint)
    assert saved["gpu_hours"] == 0
    assert len(saved["config_sha256"]) == 64
    initial = copy.deepcopy(saved["exposure"])
    resumed = train(path, tmp_path / "second", max_steps=2, resume=checkpoint)
    final = read_checkpoint(resumed)["exposure"]
    assert initial["optimizer_updates"] == 2 and final["optimizer_updates"] == 4
    assert final["E_synthetic"] > initial["E_synthetic"] > 0
    records = [json.loads(s) for s in (tmp_path / "second/training.jsonl").read_text().splitlines()]
    assert records[-1]["exposure"] == final
    assert records[-1]["config_sha256"] == saved["config_sha256"]
    assert records[-1]["gpu_hours"] == 0
    assert final["replay"]["lost_telemetry"] == "unknown"


@pytest.mark.parametrize("size", ["12m", "25m"])
def test_full_shell_matches_e5959bb_except_listed_changes(jobs, size):
    """Golden hashes measured from main e5959bb; normalize only explicit exceptions."""
    import re

    golden = json.loads((Path(__file__).parent / "full-e5959bb-sha256.json").read_text())
    for leg in jobs.LEGS:
        text = jobs.generate(
            size=size,
            leg=leg,
            seed=11,
            micro_batches=2,
            checkpoint_minutes=10,
            repo_sha=SHA,
            upstream_pin=PIN,
            corpus_uri="gs://offline/corpus",
            recipe="full",
        )
        # New patch/installer bytes necessarily change embedded payload and bundle digest.
        text = re.sub(r'base64.b64decode\("[^"\n]+"\)', 'base64.b64decode("PAYLOAD")', text)
        text = re.sub(r"\b[0-9a-f]{64}\b", "BUNDLE_DIGEST", text)
        # Explicit cache defaults fix the measured guest startup warning.
        cache = (
            'export HOME="${HOME:-$RUN_DIR/home}"\n'
            'export PYTORCH_KERNEL_CACHE_PATH="$RUN_DIR/torch-kernel-cache"\n'
            'mkdir -p "$HOME" "$PYTORCH_KERNEL_CACHE_PATH"\n'
        )
        assert cache in text
        text = text.replace(cache, "")
        # Existing fast completion now checks identity, so full/screen cannot mix.
        if leg != "qualify":
            start = text.index('    identity_path = root / "identity.json"')
            end = text.index('    if sys.argv[2] == "export":', start)
            text = text[:start] + text[end:]
        # A repeated local run must not retain receipts missing at the remote prefix.
        text = text.replace(
            "# Remove stale release receipts before syncing a possibly incomplete prefix.\n"
            'rm -f "$RUN_DIR/corpus/attempt-status.json" "$RUN_DIR/corpus/audit.json"\n',
            "",
        )
        assert hashlib.sha256(text.encode()).hexdigest() == golden[f"{size}-{leg}"]


def job_args(tmp_path, upstream, **changes):
    return SimpleNamespace(
        **(
            dict(
                run_dir=tmp_path,
                root=upstream,
                corpus=tmp_path / "corpus",
                corpus_uri="gs://offline/corpus",
                leg="stage3",
                size="25m",
                seed=11,
                micro_batches=1,
                checkpoint_minutes=10,
                repo_sha=SHA,
                upstream_pin=PIN,
                bundle_sha256="new-bundle",
                recipe="full",
                screen_updates=None,
            )
            | changes
        )
    )


def test_gate_is_on_training_and_qualification_paths(upstream, tmp_path, monkeypatch):
    from simons.training import spot_job

    release(tmp_path / "corpus")
    (tmp_path / "corpus/attempt-status.json").unlink()
    monkeypatch.setattr(spot_job, "train", lambda *a, **kw: pytest.fail("training began"))
    args = job_args(tmp_path, upstream, qualify_leg="stage3")
    with pytest.raises(ValueError, match="corpus audit gate"):
        spot_job.run_leg(args)
    with pytest.raises(ValueError, match="corpus audit gate"):
        spot_job.qualify_once(args)


def test_unsafe_qualification_executes_but_checkpoint_is_unusable(upstream, tmp_path, monkeypatch):
    import torch
    import yaml
    from simons.training import qualification, spot_job
    from simons.training.trainer import write_checkpoint
    from test_jobs import checkpoint_payload

    corpus = tmp_path / "corpus"
    release(corpus)
    (corpus / "audit.json").unlink()
    raw = yaml.safe_load(tiny_recipe(tmp_path).read_text())
    raw["training"].update(steps=1000, warm_start=None)
    monkeypatch.setattr(spot_job, "recipe", lambda *a, **kw: copy.deepcopy(raw))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "offline fake GPU")
    monkeypatch.setattr(qualification, "Measurement", lambda *a: SimpleNamespace(result=lambda: {}))
    called = []

    def train(path, output, **kwargs):
        called.append(kwargs)
        output.mkdir()
        final = output / "checkpoint.pt"
        payload = checkpoint_payload(520)
        payload["job_identity"] = kwargs["checkpoint_metadata"]
        write_checkpoint(final, payload)
        return final

    monkeypatch.setattr(spot_job, "train", train)
    args = job_args(
        tmp_path,
        upstream,
        leg="qualify",
        qualify_leg="stage3",
        warmup=20,
        qualify_step=None,
        result=tmp_path / "result.json",
        unsafe_qualification_allow_unaudited_corpus=True,
    )
    assert spot_job.qualify_once(args) == 0
    assert called and called[0]["checkpoint_metadata"]["qualification_only"] is True
    final = next(tmp_path.glob("qualify-attempt-*/output/checkpoint.pt"))
    final.rename(final.with_name("checkpoint-final.pt"))
    (final.parent / "completed.json").write_text(json.dumps({"checkpoint": "checkpoint-final.pt"}))
    with pytest.raises(ValueError, match="qualification-only"):
        spot_job.completed_checkpoint(final.parent)
    report = json.loads(args.result.read_text())
    assert report["qualification_only"] and report["corpus_release"]["qualification_only"]


@pytest.mark.parametrize(
    "pre_recipe,next_recipe,accepted",
    [("full", "full", True), ("full", "screen", False), ("screen", "full", False)],
)
@pytest.mark.parametrize("provenance", ["e5959bb", "f86878c", "b606a50"])
def test_actual_legacy_predecessor_path(
    upstream, tmp_path, monkeypatch, pre_recipe, next_recipe, accepted, provenance
):
    from simons.training import spot_job
    from simons.training.trainer import write_checkpoint
    from test_jobs import checkpoint_payload

    directory = tmp_path / "state/stage1"
    directory.mkdir(parents=True)
    saved = dict(
        size="25m",
        seed=11,
        leg="stage1",
        repo_sha=provenance,
        upstream_pin=PIN,
        bundle_sha256="legacy-bundle",
        corpus_uri=None,
        recipe={},
    )
    if pre_recipe == "screen":
        saved.update(recipe_name="screen", screen_updates=25000)
    payload = checkpoint_payload(65000)
    payload["job_identity"] = saved
    write_checkpoint(directory / "checkpoint-final.pt", payload)
    (directory / "completed.json").write_text(json.dumps({"checkpoint": "checkpoint-final.pt"}))
    called = []

    def train(path, output, **kwargs):
        called.append(kwargs)
        final = kwargs["checkpoint_dir"] / "checkpoint-final.pt"
        payload = checkpoint_payload(12000)
        payload["job_identity"] = kwargs["checkpoint_metadata"]
        write_checkpoint(final, payload)
        return final

    monkeypatch.setattr(spot_job, "train", train)
    args = job_args(
        tmp_path, upstream, leg="stage2", corpus=None, corpus_uri=None, recipe=next_recipe
    )
    if accepted:
        assert spot_job.run_leg(args) == 0 and called
    else:
        with pytest.raises(ValueError, match="identity"):
            spot_job.run_leg(args)
        assert not called


def test_replayed_work_is_separate_from_lineage(upstream, tmp_path):
    from simons.training.trainer import read_checkpoint, train

    path = tiny_recipe(tmp_path)
    state = tmp_path / "state"
    checkpoint = train(path, tmp_path / "first", max_steps=2, checkpoint_dir=state)
    original = read_checkpoint(checkpoint)["exposure"]
    log = state / "attempts/old/output/training.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"step": 4}) + "\n{partial")
    final = train(path, tmp_path / "second", max_steps=2, resume=checkpoint, checkpoint_dir=state)
    counts = read_checkpoint(final)["exposure"]
    assert counts["optimizer_updates"] == 4
    assert counts["replay"]["known_updates"] == 2
    assert counts["replay"]["E_synthetic"] == counts["E_synthetic"] - original["E_synthetic"]
    assert counts["replay"]["S_synthetic"] == counts["S_synthetic"] - original["S_synthetic"]
    assert counts["replay"]["lost_telemetry"] == "unknown"


@pytest.mark.parametrize(
    "recipe,member_releases,with_identity,accepted",
    [
        pytest.param("full", (None, None), True, True, id="legacy-legacy"),
        pytest.param("full", (None, "a"), True, True, id="legacy-audited"),
        pytest.param("full", ("a", None), True, True, id="audited-legacy"),
        pytest.param("full", ("a", "a"), True, True, id="same-release"),
        pytest.param("full", ("a", "new-attempt"), True, False, id="changed-attempt"),
        pytest.param("full", ("new-attempt", "a"), True, False, id="changed-attempt-reversed"),
        pytest.param("full", ("a", "new-hash"), True, False, id="changed-audit"),
        pytest.param("full", ("new-hash", "a"), True, False, id="changed-audit-reversed"),
        pytest.param("screen", ("a", "a"), True, True, id="screen-same-release"),
        pytest.param("screen", ("a", "new-attempt"), True, False, id="screen-changed-attempt"),
        pytest.param("screen", ("a", "new-hash"), True, False, id="screen-changed-audit"),
        pytest.param("screen", ("a", "a"), False, False, id="screen-missing-identity"),
    ],
)
def test_export_identity_and_corpus_free_invocation(
    upstream, tmp_path, recipe, member_releases, with_identity, accepted
):
    from simons.config import SimonsConfig
    from simons.model import Simons
    from simons.training import spot_job
    from simons.training.trainer import write_checkpoint
    from test_jobs import checkpoint_payload

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
    releases = {
        None: None,
        "a": dict(attempt_id="tiny", audit_sha256="hash"),
        "new-attempt": dict(attempt_id="other", audit_sha256="hash"),
        "new-hash": dict(attempt_id="tiny", audit_sha256="other-hash"),
    }
    receipts = [releases[name] for name in member_releases]
    for leg, receipt in zip(("stage4", "stage4-long"), receipts):
        directory = tmp_path / "state" / leg
        directory.mkdir(parents=True)
        payload = checkpoint_payload(8)
        payload.update(model_config=model.config.to_dict(), model=model.state_dict())
        if with_identity:
            payload["job_identity"] = dict(
                size="25m",
                seed=11,
                leg=leg,
                upstream_pin=PIN,
                corpus_uri="gs://offline/corpus",
                recipe={},
            )
            if receipt is not None:
                payload["job_identity"]["corpus_release"] = receipt
            if recipe == "screen":
                payload["job_identity"].update(recipe_name="screen", screen_updates=25000)
        write_checkpoint(directory / "checkpoint-final.pt", payload)
        (directory / "completed.json").write_text(json.dumps({"checkpoint": "checkpoint-final.pt"}))
    args = job_args(tmp_path, upstream, leg="export", recipe=recipe, corpus_uri=None)
    if accepted:
        assert spot_job.run_export(args) == 0
        assert (tmp_path / "state/export/model.safetensors").stat().st_size > 0
        marker = json.loads((tmp_path / "state/export/completed.json").read_text())
        assert marker["member_corpus_releases"] == receipts
        assert [Path(path).parent.name for path in marker["members"]] == [
            "stage4",
            "stage4-long",
        ]
        assert marker["corpus_release"] == (receipts[0] or receipts[1])
        assert not args.corpus.exists()
    else:
        error = "different corpus audit identities" if with_identity else "identity"
        with pytest.raises(ValueError, match=error):
            spot_job.run_export(args)
        assert not (tmp_path / "state/export/completed.json").exists()


def test_legacy_receipt_migration_can_retry_before_new_checkpoint(upstream, tmp_path, monkeypatch):
    import yaml
    from simons.training import spot_job
    from simons.training.trainer import train

    release(tmp_path / "corpus")
    path = tiny_recipe(tmp_path)
    raw = yaml.safe_load(path.read_text())
    args = job_args(tmp_path, upstream, leg="stage3", seed=19)
    legacy = spot_job.identity(args, raw)
    state = tmp_path / "state/stage3"
    train(path, tmp_path / "first", max_steps=2, checkpoint_dir=state, checkpoint_metadata=legacy)

    def interrupted(*a, **kw):
        raise RuntimeError("interrupted before checkpoint")

    monkeypatch.setattr(spot_job, "train", interrupted)
    with pytest.raises(RuntimeError, match="interrupted before checkpoint"):
        spot_job.run_leg(args)
    assert "corpus_release" in json.loads((state / "identity.json").read_text())
    monkeypatch.setattr(spot_job, "train", train)
    assert spot_job.run_leg(args) == 0
    marker = json.loads((state / "completed.json").read_text())
    assert (
        marker["corpus_release"]
        == json.loads((state / "identity.json").read_text())["corpus_release"]
    )


@pytest.mark.parametrize("provenance", ["e5959bb", "f86878c", "b606a50"])
@pytest.mark.parametrize("checkpoint_name", ["checkpoint-slot-0.pt", "checkpoint-final.pt"])
def test_final_legacy_checkpoint_recovers_accepted_receipt(
    upstream, tmp_path, monkeypatch, provenance, checkpoint_name
):
    import yaml
    from simons.training import spot_job
    from simons.training.screen import corpus_release
    from simons.training.trainer import read_checkpoint, write_checkpoint
    from test_jobs import checkpoint_payload

    release(tmp_path / "corpus")
    args = job_args(tmp_path, upstream, seed=19)
    raw = yaml.safe_load(tiny_recipe(tmp_path).read_text())
    receipt = corpus_release(args.corpus)
    if provenance != "e5959bb":
        args.corpus_release = receipt
    saved = spot_job.identity(args, raw)
    saved.update(repo_sha=provenance, bundle_sha256="previous-bundle")
    payload = {**checkpoint_payload(8), "job_identity": saved}
    directory = tmp_path / "state/stage3"
    directory.mkdir(parents=True)
    checkpoint = directory / checkpoint_name
    write_checkpoint(checkpoint, payload)
    original = checkpoint.read_bytes()
    monkeypatch.setattr(spot_job, "train", lambda *a, **kw: pytest.fail("training began"))

    assert spot_job.run_leg(args) == 0
    marker = json.loads((directory / "completed.json").read_text())
    manifest = json.loads((directory / "identity.json").read_text())
    assert marker["corpus_release"] == receipt
    assert marker["job_identity"] == manifest == {**saved, "corpus_release": receipt}
    assert marker["step"] == 8
    final = directory / "checkpoint-final.pt"
    assert final.read_bytes() == original
    assert read_checkpoint(final)["job_identity"] == saved


def test_native_60m_recipe_is_unchanged(upstream, tmp_path, monkeypatch):
    import yaml
    from simons.training import config, spot_job

    release(tmp_path)
    monkeypatch.setattr(spot_job, "predecessor", lambda state, leg: tmp_path / spot_job.LEGS[leg])
    lengths = []
    for leg in spot_job.LEGS:
        native = yaml.safe_load((upstream / f"configs/simons-60m-{leg}.yaml").read_text())
        expected = copy.deepcopy(native)
        expected["training"].update(
            seed=11,
            micro_batches=2,
            warm_start=None if leg == "stage1" else str(tmp_path / spot_job.LEGS[leg]),
        )
        if leg in ("stage3", "stage4", "stage4-long"):
            expected["training"].update(
                data=str(tmp_path / "univariate"), covariate_data=str(tmp_path / "covariates")
            )
        actual = spot_job.recipe(upstream, tmp_path, tmp_path, "60m", leg, 11, 2)
        assert actual == expected
        model, s = config.load(spot_job.write_recipe(actual, tmp_path))
        lengths.append(s["steps"] - spot_job.origin(model, s))
    assert lengths == [55000, 12000, 9341, 1200, 2400]
    assert sum(lengths) == 79941


def test_changed_audit_refuses_resume(upstream, tmp_path, monkeypatch):
    import yaml
    from simons.training import spot_job
    from simons.training.screen import corpus_release
    from simons.training.trainer import train

    _, audit = release(tmp_path / "corpus")
    path = tiny_recipe(tmp_path)
    args = job_args(tmp_path, upstream, seed=19)
    args.corpus_release = corpus_release(args.corpus)
    identity = spot_job.identity(args, yaml.safe_load(path.read_text()))
    train(
        path,
        tmp_path / "first",
        max_steps=2,
        checkpoint_dir=tmp_path / "state/stage3",
        checkpoint_metadata=identity,
    )
    audit["extra"] = "different valid bytes"
    (args.corpus / "audit.json").write_text(json.dumps(audit))
    monkeypatch.setattr(spot_job, "train", lambda *a, **kw: pytest.fail("training began"))
    with pytest.raises(ValueError, match="identity changed"):
        spot_job.run_leg(args)


@pytest.mark.parametrize("budget", [0, -1, 99, True, 1.5])
def test_invalid_screen_budget(jobs, budget):
    with pytest.raises(ValueError, match="screen updates"):
        jobs.generate(
            size="25m",
            leg="stage1",
            seed=11,
            micro_batches=1,
            checkpoint_minutes=10,
            repo_sha=SHA,
            upstream_pin=PIN,
            recipe="screen",
            screen_updates=budget,
        )
