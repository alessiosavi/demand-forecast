"""W12: fork identity transitions, recipe effects, and fail-closed recovery."""

import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from test_jobs import PIN, SHA, checkpoint_payload, shell_checkpoint_record
from test_screen import job_args
from test_screen import release as screen_release

CONDITION = {"name": "cov-w2", "overrides": {"covariate_weight": "0.05"}}
URI = "gs://offline/corpus"
PROVISIONAL = {
    "provisional": True,
    "warning": "UNAUDITED_CORPUS_PROVISIONAL",
    "attempt_id": "tiny",
    "corpus_uri": URI,
    "attempt_complete": True,
    "attempt_exit_code": 0,
}
AUDITED = {"attempt_id": "tiny", "audit_sha256": "a" * 64}
FIXTURES = Path(__file__).parent / "fixtures"


def release(corpus):
    status, audit = screen_release(corpus)
    status["attempt_id"] = audit["attempt_id"] = "tiny"
    (corpus / "attempt-status.json").write_text(json.dumps(status))
    (corpus / "audit.json").write_text(json.dumps(audit))


def options(**changes):
    return (
        dict(
            size="25m",
            leg="stage3",
            seed=11,
            micro_batches=1,
            checkpoint_minutes=10,
            repo_sha=SHA,
            upstream_pin=PIN,
            corpus_uri=URI,
            condition="cov-w2",
            override=["covariate_weight=.050"],
            provisional_unaudited_corpus_attempt="tiny",
        )
        | changes
    )


def parent(leg="stage3", receipt=PROVISIONAL, condition=CONDITION):
    saved = json.loads(
        (
            FIXTURES / f"control-{'stage2' if leg == 'stage2' else 'stage3'}-identity.json"
        ).read_text()
    )
    saved.update(leg=leg)
    if leg != "stage2":
        saved["corpus_uri"] = URI
    if receipt is not None:
        saved["corpus_release"] = copy.deepcopy(receipt)
    if condition is not None:
        saved["condition"] = copy.deepcopy(condition)
    return saved


def write_stage(root, leg, saved, *, step=8, completed=True, manifest=True, model=False):
    from simons.training import spot_job
    from simons.training.trainer import write_checkpoint

    directory = root / "state" / leg
    directory.mkdir(parents=True, exist_ok=True)
    payload = {**checkpoint_payload(step), "job_identity": saved}
    if model:
        from simons.config import SimonsConfig
        from simons.model import Simons

        tiny = Simons(
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
        payload.update(model_config=tiny.config.to_dict(), model=tiny.state_dict())
    path = directory / ("checkpoint-final.pt" if completed else "checkpoint-slot-0.pt")
    write_checkpoint(path, payload)
    if manifest:
        spot_job.atomic_json(directory / "identity.json", saved)
    if completed:
        spot_job.complete_leg(directory, path, step)
    return path


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("leg", ["stage3", "stage4", "stage4-long", "export"])
def test_generator_canonical_arguments(jobs, leg):
    text = jobs.generate(
        **options(
            leg=leg, override=["covariate_weight=.050", "covariate_mix=0,0.25,.75", "long_weight=1"]
        )
    )
    args = shlex.split(text.split('exec "$scratch/venv/bin/python"', 1)[1].replace("\\\n", " "))
    assert args[args.index("--condition") + 1] == "cov-w2"
    assert [args[i + 1] for i, x in enumerate(args) if x == "--override"] == [
        "covariate_mix=0.0,0.25,0.75",
        "covariate_weight=0.05",
        "long_weight=1.0",
    ]
    assert args[args.index("--provisional-unaudited-corpus-attempt") + 1] == "tiny"


@pytest.mark.parametrize(
    "changes",
    [
        {"condition": None},
        {"override": []},
        {"condition": "Upper"},
        {"condition": "a" * 33},
        {"override": ["no_such_key=1"]},
        {"override": ["covariate_weight=.1", "covariate_weight=.2"]},
        {"override": ["covariate_weight=0"]},
        {"override": ["covariate_weight=1.01"]},
        {"override": ["covariate_weight=nan"]},
        {"override": ["long_weight=inf"]},
        {"override": ["covariate_mix=.1,.2,.3"]},
        {"override": ["covariate_mix=nan,0,1"]},
        {"override": ["covariate_mix=-.1,.1,1"]},
        {"override": ["covariate_mix=0,1"]},
        {"provisional_unaudited_corpus_attempt": " "},
        {"unsafe_qualification_allow_unaudited_corpus": True},
    ],
)
def test_generator_rejects_invalid_fork(jobs, changes):
    with pytest.raises(ValueError):
        jobs.generate(**options(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"size": "12m"},
        {"size": "60m"},
        {"recipe": "screen"},
        {"leg": "stage1"},
        {"leg": "stage2"},
        {"leg": "qualify"},
    ],
)
@pytest.mark.parametrize("flag", ["condition", "provisional"])
def test_flags_restricted_in_generator_and_guest(
    jobs, upstream, tmp_path, monkeypatch, changes, flag
):
    from simons.training import spot_job

    opts = options(**changes)
    if flag == "condition":
        opts["provisional_unaudited_corpus_attempt"] = None
    else:
        opts.update(condition=None, override=[])
    with pytest.raises(ValueError):
        jobs.generate(**opts)
    argv = [
        "spot_job",
        f"--root={upstream}",
        f"--run-dir={tmp_path / 'run'}",
        "--bundle-sha256=test",
    ]
    for key in (
        "size",
        "leg",
        "seed",
        "repo_sha",
        "upstream_pin",
        "corpus_uri",
        "recipe",
        "condition",
        "provisional_unaudited_corpus_attempt",
    ):
        if opts.get(key) is not None:
            argv.append(f"--{key.replace('_', '-')}={opts[key]}")
    argv.extend(f"--override={value}" for value in opts["override"])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        spot_job.main()
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("leg", ["stage3", "stage4", "stage4-long"])
def test_recipe_and_runtime_passes(upstream, tmp_path, monkeypatch, leg):
    from types import SimpleNamespace

    import yaml
    from simons.training import config, spot_job, trainer
    from simons.training.covariates import stream

    release(tmp_path)
    monkeypatch.setattr(spot_job, "predecessor", lambda state, leg: Path("/weights") / leg)
    baseline = spot_job.recipe(upstream, tmp_path, tmp_path, "25m", leg, 11, 1)
    source = yaml.safe_load((upstream / f"configs/simons-25m-{leg}.yaml").read_text())
    assert baseline["training"] == source["training"] | dict(
        seed=11,
        micro_batches=1,
        warm_start=f"/weights/{leg}",
        data=str(tmp_path / "univariate"),
        covariate_data=str(tmp_path / "covariates"),
    )
    condition = {
        "name": "mix",
        "overrides": {
            "covariate_weight": "0.05",
            "covariate_mix": "0.0,0.25,0.75",
            "long_weight": "0.2",
        },
    }
    raw = spot_job.recipe(upstream, tmp_path, tmp_path, "25m", leg, 11, 1, condition=condition)
    expected = copy.deepcopy(baseline)
    expected["training"].update(covariate_weight=0.05, covariate_mix="0.0,0.25,0.75")
    if leg != "stage3":
        expected["training"]["long_weight"] = 0.2
    assert raw == expected
    assert ("long_weight" in raw["training"]) == (leg != "stage3")
    model, settings = config.load(spot_job.write_recipe(raw, tmp_path))
    monkeypatch.setattr(stream, "CovariateStream", lambda spec, *a: spec)
    passes = dict(trainer.covariate_passes(settings, model, SimpleNamespace(world=1), 11, 0, "cpu"))
    assert passes["covariate"]["weight"] == 0.05
    assert passes["covariate"]["mix"] == {"synthetic": 0.0, "sibling": 0.25, "real": 0.75}
    if leg != "stage3":
        assert passes["long"]["weight"] == 0.2
        assert passes["long"]["mix"] == {"real": 1.0}


def test_missing_yaml_key_refused(upstream, tmp_path, monkeypatch):
    import yaml
    from simons.training import spot_job

    release(tmp_path)
    (tmp_path / "configs").mkdir()
    raw = yaml.safe_load((upstream / "configs/simons-25m-stage3.yaml").read_text())
    del raw["training"]["covariate_weight"]
    (tmp_path / "configs/simons-25m-stage3.yaml").write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(spot_job, "predecessor", lambda *a: Path("/weights"))
    with pytest.raises(ValueError, match="covariate_weight"):
        spot_job.recipe(tmp_path, tmp_path, tmp_path, "25m", "stage3", 11, 1, condition=CONDITION)


@pytest.mark.parametrize(
    "parent_leg,receipt,condition,flag,next_leg,accepted",
    [
        pytest.param("stage2", None, None, False, "stage3", True, id="row1-audited-stage2"),
        pytest.param("stage2", None, None, True, "stage3", True, id="row2-provisional-stage2"),
        pytest.param(
            "stage3", PROVISIONAL, CONDITION, True, "stage4", True, id="row3-provisional-later"
        ),
        pytest.param(
            "stage3", PROVISIONAL, CONDITION, False, "stage4", False, id="row4-dropped-flag"
        ),
        pytest.param(
            "stage3", AUDITED, CONDITION, True, "stage4", False, id="row5-audited-to-provisional"
        ),
        pytest.param("stage3", None, None, True, "stage4", False, id="row6-legacy-to-provisional"),
        pytest.param("stage3", None, None, False, "stage4", True, id="row7-legacy-audited"),
    ],
)
def test_transition_rows(
    upstream, tmp_path, monkeypatch, parent_leg, receipt, condition, flag, next_leg, accepted
):
    from simons.training import spot_job
    from simons.training.trainer import read_checkpoint, write_checkpoint

    release(tmp_path / "corpus")
    saved = parent(parent_leg, receipt, condition)
    write_stage(tmp_path, parent_leg, saved)
    if flag:
        (tmp_path / "corpus/audit.json").unlink()
    conditioned = flag or condition is not None
    args = job_args(
        tmp_path,
        upstream,
        leg=next_leg,
        condition="cov-w2" if conditioned else None,
        override=["covariate_weight=.05"] if conditioned else [],
        provisional_unaudited_corpus_attempt="tiny" if flag else None,
    )
    called = []

    def train(path, output, **kwargs):
        called.append(kwargs)
        final = kwargs["checkpoint_dir"] / "checkpoint-final.pt"
        write_checkpoint(
            final, {**checkpoint_payload(57600), "job_identity": kwargs["checkpoint_metadata"]}
        )
        return final

    monkeypatch.setattr(spot_job, "train", train)
    before = snapshot(tmp_path / "state")
    if accepted:
        assert spot_job.run_leg(args) == 0
        identity = read_checkpoint(tmp_path / "state" / next_leg / "checkpoint-final.pt")[
            "job_identity"
        ]
        assert identity.get("condition") == (CONDITION if conditioned else None)
        if flag:
            assert identity["corpus_release"] == PROVISIONAL
            assert (
                json.loads((tmp_path / "state" / next_leg / "identity.json").read_text())
                == identity
            )
    else:
        with pytest.raises(ValueError, match="condition|receipt|provisional"):
            spot_job.run_leg(args)
        assert not called
        assert snapshot(tmp_path / "state") == before


@pytest.mark.parametrize(
    "status",
    [
        dict(attempt_id="other", complete=True, exit_code=0),
        dict(attempt_id="", complete=True, exit_code=0),
        dict(attempt_id=" ", complete=True, exit_code=0),
        dict(attempt_id=17, complete=True, exit_code=0),
        dict(complete=True, exit_code=0),
        dict(attempt_id="tiny", exit_code=0),
        dict(attempt_id="tiny", complete=True),
        [],
        "not JSON",
        None,
    ],
)
def test_provisional_refuses_malformed_or_mismatched_status(upstream, tmp_path, status):
    from simons.training import spot_job

    release(tmp_path / "corpus")
    path = tmp_path / "corpus/attempt-status.json"
    if status is None:
        path.unlink()
    else:
        path.write_text(status if isinstance(status, str) else json.dumps(status))
    args = job_args(tmp_path, upstream, provisional_unaudited_corpus_attempt="tiny")
    with pytest.raises(ValueError, match="corpus.*gate"):
        spot_job.run_leg(args)
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("complete,exit_code", [(False, -1), (True, 0)])
def test_provisional_stage3_accepts_real_corpus_status(
    upstream, tmp_path, monkeypatch, complete, exit_code
):
    from simons.training import spot_job
    from simons.training.trainer import read_checkpoint, write_checkpoint

    release(tmp_path / "corpus")
    status = json.loads((FIXTURES / "control-corpus-attempt-status.json").read_text())
    status.update(complete=complete, exit_code=exit_code)
    (tmp_path / "corpus/attempt-status.json").write_text(json.dumps(status))
    (tmp_path / "corpus/audit.json").write_bytes(b"not JSON")
    write_stage(tmp_path, "stage2", parent("stage2", None, None))
    args = job_args(
        tmp_path,
        upstream,
        corpus_uri=URI + "/",
        condition="cov-w2",
        override=["covariate_weight=.05"],
        provisional_unaudited_corpus_attempt="20261113T114417Z-1990",
    )

    def train(path, output, **kwargs):
        final = kwargs["checkpoint_dir"] / "checkpoint-final.pt"
        write_checkpoint(
            final, {**checkpoint_payload(57600), "job_identity": kwargs["checkpoint_metadata"]}
        )
        return final

    monkeypatch.setattr(spot_job, "train", train)
    assert spot_job.run_leg(args) == 0
    expected = {
        "provisional": True,
        "warning": "UNAUDITED_CORPUS_PROVISIONAL",
        "attempt_id": "20261113T114417Z-1990",
        "corpus_uri": URI,
        "attempt_complete": complete,
        "attempt_exit_code": exit_code,
    }
    directory = tmp_path / "state/stage3"
    identity = json.loads((directory / "identity.json").read_text())
    marker = json.loads((directory / "completed.json").read_text())
    assert identity["corpus_release"] == expected
    assert identity["corpus_release"]["attempt_complete"] is complete
    assert type(identity["corpus_release"]["attempt_exit_code"]) is int
    assert marker["corpus_release"] == marker["job_identity"]["corpus_release"] == expected
    assert read_checkpoint(directory / "checkpoint-final.pt")["job_identity"] == identity


def test_real_corpus_status_refused_without_provisional_flag(upstream, tmp_path, monkeypatch):
    from simons.training import spot_job

    release(tmp_path / "corpus")
    (tmp_path / "corpus/attempt-status.json").write_bytes(
        (FIXTURES / "control-corpus-attempt-status.json").read_bytes()
    )
    write_stage(tmp_path, "stage2", parent("stage2", None, None))
    before = snapshot(tmp_path / "state")
    monkeypatch.setattr(spot_job, "train", lambda *a, **kw: pytest.fail("training began"))
    with pytest.raises(ValueError, match="corpus audit gate: attempt is incomplete or failed"):
        spot_job.run_leg(job_args(tmp_path, upstream))
    assert snapshot(tmp_path / "state") == before


@pytest.mark.parametrize("leg", ["stage3", "stage4", "stage4-long"])
@pytest.mark.parametrize("field,value", [("complete", True), ("exit_code", 0)])
@pytest.mark.parametrize("manifest", [False, True])
@pytest.mark.parametrize("conditioned", [False, True])
def test_changed_provisional_status_refuses_resume_and_later_legs(
    upstream, tmp_path, monkeypatch, leg, field, value, manifest, conditioned
):
    from simons.training import spot_job

    release(tmp_path / "corpus")
    status = json.loads((FIXTURES / "control-corpus-attempt-status.json").read_text())
    receipt = {
        **PROVISIONAL,
        "attempt_id": status["attempt_id"],
        "attempt_complete": False,
        "attempt_exit_code": -1,
    }
    write_stage(
        tmp_path,
        "stage3",
        parent(receipt=receipt, condition=CONDITION if conditioned else None),
        step=35000 if leg == "stage3" else 57600,
        completed=leg != "stage3",
        manifest=manifest,
    )
    status[field] = value
    (tmp_path / "corpus/attempt-status.json").write_text(json.dumps(status))
    args = job_args(
        tmp_path,
        upstream,
        leg=leg,
        condition="cov-w2" if conditioned else None,
        override=["covariate_weight=.05"] if conditioned else [],
        provisional_unaudited_corpus_attempt=status["attempt_id"],
    )
    monkeypatch.setattr(spot_job, "train", lambda *a, **kw: pytest.fail("training began"))
    before = snapshot(tmp_path / "state")
    with pytest.raises(ValueError, match="receipt|corpus audit identity"):
        spot_job.run_leg(args)
    assert snapshot(tmp_path / "state") == before


@pytest.mark.parametrize("conditioned,flag", [(True, False), (False, True)])
def test_real_legacy_stage3_cannot_be_adopted(upstream, tmp_path, conditioned, flag):
    from simons.training import spot_job

    release(tmp_path / "corpus")
    saved = json.loads((FIXTURES / "control-stage3-identity.json").read_text())
    write_stage(tmp_path, "stage3", saved)
    args = job_args(
        tmp_path,
        upstream,
        leg="stage4",
        corpus_uri=saved["corpus_uri"],
        condition="cov-w2" if conditioned else None,
        override=["covariate_weight=.05"] if conditioned else [],
        provisional_unaudited_corpus_attempt="tiny" if flag else None,
    )
    before = snapshot(tmp_path / "state")
    with pytest.raises(ValueError, match="condition|receipt|provisional"):
        spot_job.run_leg(args)
    assert snapshot(tmp_path / "state") == before


@pytest.mark.parametrize(
    "completed,manifest,change",
    [
        (completed, manifest, change)
        for completed in (True, False)
        for manifest in (True, False)
        for change in (
            "drop-condition",
            "change-condition",
            "drop-flag",
            "change-attempt",
            "change-uri",
            "stale-manifest",
            "stale-marker",
        )
        if (change != "stale-marker" or completed) and (change != "stale-manifest" or manifest)
    ],
)
def test_resume_and_completion_refuse_without_writes(
    upstream, tmp_path, completed, manifest, change
):
    from simons.training import spot_job

    release(tmp_path / "corpus")
    args = job_args(
        tmp_path,
        upstream,
        condition="cov-w2",
        override=["covariate_weight=.05"],
        provisional_unaudited_corpus_attempt="tiny",
    )
    saved = parent()
    write_stage(tmp_path, "stage3", saved, completed=completed, manifest=manifest)
    if change == "drop-condition":
        args.condition, args.override = None, []
    elif change == "change-condition":
        args.override = ["covariate_weight=.1"]
    elif change == "drop-flag":
        args.provisional_unaudited_corpus_attempt = None
    elif change == "change-attempt":
        args.provisional_unaudited_corpus_attempt = "other"
    elif change == "change-uri":
        args.corpus_uri = "gs://other/corpus"
    else:
        path = (
            tmp_path
            / "state/stage3"
            / ("identity.json" if change == "stale-manifest" else "completed.json")
        )
        record = json.loads(path.read_text())
        target = record if change == "stale-manifest" else record["job_identity"]
        target["condition"] = {"name": "other", "overrides": {"covariate_weight": "0.05"}}
        path.write_text(json.dumps(record))
    before = snapshot(tmp_path / "state")
    with pytest.raises(ValueError, match="condition|receipt|provisional"):
        spot_job.run_leg(args)
    assert snapshot(tmp_path / "state") == before


@pytest.mark.parametrize("completed", [True, False])
@pytest.mark.parametrize("manifest", [True, False])
def test_matching_resume_and_marker_recovery(upstream, tmp_path, completed, manifest):
    from simons.training import spot_job

    release(tmp_path / "corpus")
    args = job_args(
        tmp_path,
        upstream,
        condition="cov-w2",
        override=["covariate_weight=.05"],
        provisional_unaudited_corpus_attempt="tiny",
    )
    saved = parent()
    saved["recipe"]["training"]["covariate_weight"] = 0.05
    write_stage(tmp_path, "stage3", saved, step=57600, completed=completed, manifest=manifest)
    assert spot_job.run_leg(args) == 0
    marker = json.loads((tmp_path / "state/stage3/completed.json").read_text())
    assert marker["job_identity"]["condition"] == CONDITION
    assert marker["corpus_release"] == PROVISIONAL


@pytest.mark.parametrize("leg", ["stage3", "export"])
@pytest.mark.parametrize("manifest", [True, False])
@pytest.mark.parametrize(
    "change", ["drop-condition", "drop-flag", "change-condition", "stale-marker"]
)
def test_shell_shortcut_checks_all_records(jobs, tmp_path, leg, manifest, change):
    opts = options(leg=leg)
    saved = parent(leg)
    directory = tmp_path / "state" / leg
    directory.mkdir(parents=True)
    if leg == "export":
        for name in ("config.json", "model.safetensors"):
            (directory / name).write_bytes(b"artifact")
        marker = {
            "artifacts": {"config.json": 8, "model.safetensors": 8},
            "corpus_release": PROVISIONAL,
        }
    else:
        marker = shell_checkpoint_record(directory / "checkpoint-final.pt")
        marker["corpus_release"] = PROVISIONAL
    marker["job_identity"] = copy.deepcopy(saved)
    if manifest:
        (directory / "identity.json").write_text(json.dumps(saved))
    if change == "drop-condition":
        opts.update(condition=None, override=[])
    elif change == "drop-flag":
        opts["provisional_unaudited_corpus_attempt"] = None
    elif change == "change-condition":
        opts["override"] = ["covariate_weight=.1"]
    else:
        marker["job_identity"]["condition"]["name"] = "other"
    (directory / "completed.json").write_text(json.dumps(marker))
    script = jobs.generate(**opts).split("scratch=$(mktemp", 1)[0] + "echo continue\n"
    before = snapshot(directory)
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        env={**os.environ, "RUN_DIR": str(tmp_path), "STATE_URI": "gs://offline"},
    )
    assert result.returncode != 0 or result.stdout == "continue\n"
    assert snapshot(directory) == before


@pytest.mark.parametrize("complete,exit_code", [(True, 0), (False, -1)])
@pytest.mark.parametrize(
    "difference",
    [None, "condition", "receipt", "uri", "legacy", "audited", "complete", "exit-code"],
)
def test_export_pair_and_record(upstream, tmp_path, difference, complete, exit_code):
    from simons.training import spot_job

    receipt = {**PROVISIONAL, "attempt_complete": complete, "attempt_exit_code": exit_code}
    first, second = parent("stage4", receipt), parent("stage4-long", receipt)
    if difference == "condition":
        second["condition"]["name"] = "other"
    elif difference == "receipt":
        second["corpus_release"]["attempt_id"] = "other"
    elif difference == "uri":
        second["corpus_release"]["corpus_uri"] = "gs://other/corpus"
    elif difference == "legacy":
        del second["corpus_release"]
    elif difference == "audited":
        second["corpus_release"] = AUDITED
    elif difference == "complete":
        second["corpus_release"]["attempt_complete"] = not complete
    elif difference == "exit-code":
        second["corpus_release"]["attempt_exit_code"] = exit_code + 1
    for leg, saved in (("stage4", first), ("stage4-long", second)):
        write_stage(tmp_path, leg, saved, model=True)
    args = job_args(
        tmp_path,
        upstream,
        leg="export",
        condition="cov-w2",
        override=["covariate_weight=.05"],
        provisional_unaudited_corpus_attempt="tiny",
    )
    before = snapshot(tmp_path / "state")
    if difference:
        with pytest.raises(ValueError, match="condition|receipt|provisional"):
            spot_job.run_export(args)
        assert snapshot(tmp_path / "state") == before
    else:
        assert spot_job.run_export(args) == 0
        marker = json.loads((tmp_path / "state/export/completed.json").read_text())
        assert marker["condition"] == marker["job_identity"]["condition"] == CONDITION
        assert marker["corpus_release"] == receipt
        assert marker["member_corpus_releases"] == [receipt, receipt]
        assert all(m["condition"] == CONDITION for m in marker["member_job_identities"])


@pytest.mark.parametrize("artifacts", [True, False])
@pytest.mark.parametrize(
    "change",
    [
        "drop-condition",
        "drop-flag",
        "changed-condition",
        "nested-condition",
        "top-receipt",
        "member-receipt",
        "top-complete",
        "member-exit-code",
    ],
)
def test_incompatible_completed_export_never_rebuilt(upstream, tmp_path, artifacts, change):
    from simons.training import spot_job

    directory = tmp_path / "state/export"
    directory.mkdir(parents=True)
    marker = {
        "job_identity": parent("stage4"),
        "condition": copy.deepcopy(CONDITION),
        "corpus_release": copy.deepcopy(PROVISIONAL),
        "member_corpus_releases": [copy.deepcopy(PROVISIONAL), copy.deepcopy(PROVISIONAL)],
        "artifacts": {"config.json": 8, "model.safetensors": 8},
    }
    if artifacts:
        for name in marker["artifacts"]:
            (directory / name).write_bytes(b"artifact")
    args = job_args(
        tmp_path,
        upstream,
        leg="export",
        condition="cov-w2",
        override=["covariate_weight=.05"],
        provisional_unaudited_corpus_attempt="tiny",
    )
    if change == "drop-condition":
        args.condition, args.override = None, []
    elif change == "drop-flag":
        args.provisional_unaudited_corpus_attempt = None
    elif change == "changed-condition":
        args.override = ["covariate_weight=.1"]
    elif change == "nested-condition":
        marker["job_identity"]["condition"]["name"] = "other"
    elif change == "top-receipt":
        marker["corpus_release"]["attempt_id"] = "other"
    elif change == "top-complete":
        marker["corpus_release"]["attempt_complete"] = False
    elif change == "member-exit-code":
        marker["member_corpus_releases"][1]["attempt_exit_code"] = -1
    else:
        marker["member_corpus_releases"][1]["corpus_uri"] = "gs://other/corpus"
    (directory / "completed.json").write_text(json.dumps(marker))
    before = snapshot(directory)
    with pytest.raises(ValueError, match="condition|receipt|provisional"):
        spot_job.run_export(args)
    assert snapshot(directory) == before


@pytest.mark.parametrize("completed", [True, False])
@pytest.mark.parametrize("receipt", [None, AUDITED])
def test_same_stage_cannot_adopt_provisional(upstream, tmp_path, completed, receipt):
    from simons.training import spot_job

    release(tmp_path / "corpus")
    write_stage(tmp_path, "stage3", parent(receipt=receipt, condition=None), completed=completed)
    args = job_args(tmp_path, upstream, provisional_unaudited_corpus_attempt="tiny")
    before = snapshot(tmp_path / "state")
    with pytest.raises(ValueError, match="receipt|provisional"):
        spot_job.run_leg(args)
    assert snapshot(tmp_path / "state") == before


@pytest.mark.parametrize("new_condition", [False, True])
@pytest.mark.parametrize("provisional", [False, True])
def test_stage2_fork_flags_independent(upstream, tmp_path, monkeypatch, new_condition, provisional):
    from simons.training import spot_job

    release(tmp_path / "corpus")
    write_stage(tmp_path, "stage2", parent("stage2", None, None))
    args = job_args(
        tmp_path,
        upstream,
        corpus_uri=URI + "/",
        condition="cov-w2" if new_condition else None,
        override=["covariate_weight=.05"] if new_condition else [],
        provisional_unaudited_corpus_attempt="tiny" if provisional else None,
    )
    if provisional:
        # Invalid audit bytes prove the provisional gate never parses audit.json.
        (tmp_path / "corpus/audit.json").write_bytes(b"not JSON")

    def interrupted(*args, **kwargs):
        raise RuntimeError("stop after identity publication")

    monkeypatch.setattr(spot_job, "train", interrupted)
    with pytest.raises(RuntimeError, match="stop after identity"):
        spot_job.run_leg(args)
    identity = json.loads((tmp_path / "state/stage3/identity.json").read_text())
    assert identity.get("condition") == (CONDITION if new_condition else None)
    if provisional:
        assert identity["corpus_release"] == PROVISIONAL
    else:
        assert "audit_sha256" in identity["corpus_release"]


def test_shell_checks_disagreeing_audited_records(jobs, tmp_path):
    directory = tmp_path / "state/stage3"
    directory.mkdir(parents=True)
    saved = parent(receipt=AUDITED, condition=None)
    marker = shell_checkpoint_record(directory / "checkpoint-final.pt")
    marker.update(job_identity=saved, corpus_release={**AUDITED, "audit_sha256": "b" * 64})
    (directory / "identity.json").write_text(json.dumps(saved))
    (directory / "completed.json").write_text(json.dumps(marker))
    script = (
        jobs.generate(
            **options(condition=None, override=[], provisional_unaudited_corpus_attempt=None)
        ).split("scratch=$(mktemp", 1)[0]
        + "echo continue\n"
    )
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        env={**os.environ, "RUN_DIR": str(tmp_path), "STATE_URI": "gs://offline"},
    )
    assert result.returncode != 0 or result.stdout == "continue\n"


@pytest.mark.parametrize("leg", ["stage3", "export"])
@pytest.mark.parametrize("manifest", [False, True])
def test_matching_provisional_completion_defers_to_guest(
    jobs, upstream, tmp_path, monkeypatch, leg, manifest
):
    from simons.training import spot_job

    receipt = {**PROVISIONAL, "attempt_complete": False, "attempt_exit_code": -1}
    args = job_args(
        tmp_path,
        upstream,
        leg=leg,
        condition="cov-w2",
        override=["covariate_weight=.05"],
        provisional_unaudited_corpus_attempt="tiny",
    )
    if leg == "export":
        for member in ("stage4", "stage4-long"):
            write_stage(tmp_path, member, parent(member, receipt), model=True, manifest=manifest)
        assert spot_job.run_export(args) == 0
    else:
        write_stage(tmp_path, leg, parent(leg, receipt), manifest=manifest)
    before = snapshot(tmp_path / "state")
    script = jobs.generate(**options(leg=leg)).split("scratch=$(mktemp", 1)[0] + "echo continue\n"
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        env={**os.environ, "RUN_DIR": str(tmp_path), "STATE_URI": "gs://offline"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "continue\n"
    monkeypatch.setattr(spot_job, "train", lambda *a, **kw: pytest.fail("training began"))
    monkeypatch.setattr(spot_job, "export", lambda *a: pytest.fail("export was rebuilt"))
    assert (spot_job.run_export if leg == "export" else spot_job.run_leg)(args) == 0
    assert snapshot(tmp_path / "state") == before


@pytest.mark.parametrize("change", ["condition", "complete", "exit-code"])
def test_shell_defers_fork_checkpoint_identity_verification(jobs, upstream, tmp_path, change):
    from simons.training import spot_job
    from simons.training.reliability import checkpoint_hash
    from simons.training.trainer import read_checkpoint, write_checkpoint

    path = write_stage(tmp_path, "stage3", parent())
    payload = read_checkpoint(path)
    if change == "condition":
        payload["job_identity"]["condition"]["name"] = "other"
    elif change == "complete":
        payload["job_identity"]["corpus_release"]["attempt_complete"] = False
    else:
        payload["job_identity"]["corpus_release"]["attempt_exit_code"] = -1
    write_checkpoint(path, payload)
    marker_path = path.parent / "completed.json"
    marker = json.loads(marker_path.read_text())
    marker.update(bytes=path.stat().st_size, sha256=checkpoint_hash(path))
    marker_path.write_text(json.dumps(marker))
    before = snapshot(tmp_path / "state")
    script = jobs.generate(**options()).split("scratch=$(mktemp", 1)[0] + "echo continue\n"
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        env={**os.environ, "RUN_DIR": str(tmp_path), "STATE_URI": "gs://offline"},
    )
    assert result.returncode != 0 or result.stdout == "continue\n"
    args = job_args(
        tmp_path,
        upstream,
        condition="cov-w2",
        override=["covariate_weight=.05"],
        provisional_unaudited_corpus_attempt="tiny",
    )
    with pytest.raises(ValueError, match="condition|receipt"):
        spot_job.run_leg(args)
    assert snapshot(tmp_path / "state") == before


@pytest.mark.parametrize("change", ["condition", "audited-receipt"])
def test_completed_export_checks_available_member_records(jobs, upstream, tmp_path, change):
    from simons.training import spot_job

    receipt = PROVISIONAL if change == "condition" else AUDITED
    attempt = "tiny" if change == "condition" else None
    for leg in ("stage4", "stage4-long"):
        write_stage(tmp_path, leg, parent(leg, receipt), model=True)
    args = job_args(
        tmp_path,
        upstream,
        leg="export",
        condition="cov-w2",
        override=["covariate_weight=.05"],
        provisional_unaudited_corpus_attempt=attempt,
    )
    assert spot_job.run_export(args) == 0
    path = tmp_path / "state/stage4-long/identity.json"
    saved = json.loads(path.read_text())
    if change == "condition":
        saved["condition"]["name"] = "other"
        path.write_text(json.dumps(saved))
    else:
        # Each member is internally consistent; only the pair/export disagrees.
        saved["corpus_release"]["audit_sha256"] = "b" * 64
        write_stage(tmp_path, "stage4-long", saved, model=True)
    before = snapshot(tmp_path / "state")
    script = (
        jobs.generate(**options(leg="export", provisional_unaudited_corpus_attempt=attempt)).split(
            "scratch=$(mktemp", 1
        )[0]
        + "echo continue\n"
    )
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        env={**os.environ, "RUN_DIR": str(tmp_path), "STATE_URI": "gs://offline"},
    )
    assert result.returncode != 0 or result.stdout == "continue\n"
    with pytest.raises(ValueError, match="condition|receipt"):
        spot_job.run_export(args)
    assert snapshot(tmp_path / "state") == before


@pytest.mark.parametrize("entrypoint", ["shell", "guest"])
@pytest.mark.parametrize("manifest", [True, False])
@pytest.mark.parametrize(
    "change",
    [
        None,
        "condition",
        "provisional",
        "both",
        "audited-receipt",
        "legacy-audited",
        "marker-condition",
        "marker-receipt",
    ],
)
def test_unflagged_completed_export_checks_members(
    jobs, upstream, tmp_path, monkeypatch, entrypoint, manifest, change
):
    from simons.training import spot_job

    for leg in ("stage4", "stage4-long"):
        write_stage(tmp_path, leg, parent(leg, AUDITED, None), model=True, manifest=manifest)
    args = job_args(tmp_path, upstream, leg="export")
    assert spot_job.run_export(args) == 0
    if change in ("marker-condition", "marker-receipt"):
        path = tmp_path / "state/stage4-long/completed.json"
        marker = json.loads(path.read_text())
        if change == "marker-condition":
            marker["condition"] = CONDITION
        else:
            marker["corpus_release"] = PROVISIONAL
        path.write_text(json.dumps(marker))
    elif change is not None:
        receipt = PROVISIONAL if change in ("provisional", "both") else AUDITED
        condition = CONDITION if change in ("condition", "both") else None
        if change == "audited-receipt":
            receipt = {**AUDITED, "audit_sha256": "b" * 64}
        elif change == "legacy-audited":
            receipt = None
        write_stage(
            tmp_path,
            "stage4-long",
            parent("stage4-long", receipt, condition),
            model=True,
            manifest=manifest,
        )
    before = snapshot(tmp_path / "state")
    incompatible = change not in (None, "legacy-audited")
    if entrypoint == "shell":
        script = (
            jobs.generate(
                **options(
                    leg="export",
                    condition=None,
                    override=[],
                    provisional_unaudited_corpus_attempt=None,
                )
            ).split("scratch=$(mktemp", 1)[0]
            + "echo continue\n"
        )
        result = subprocess.run(
            ["bash"],
            input=script,
            text=True,
            capture_output=True,
            env={**os.environ, "RUN_DIR": str(tmp_path), "STATE_URI": "gs://offline"},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == ("continue\n" if incompatible else "")
    else:
        monkeypatch.setattr(spot_job, "export", lambda *a: pytest.fail("export was rebuilt"))
        if incompatible:
            with pytest.raises(ValueError, match="condition|receipt"):
                spot_job.run_export(args)
        else:
            assert spot_job.run_export(args) == 0
    assert snapshot(tmp_path / "state") == before


@pytest.mark.parametrize("saved_uri,next_uri", [(URI + "/", URI), (URI, URI + "/")])
@pytest.mark.parametrize("leg", ["stage4", "stage4-long"])
def test_provisional_continuation_accepts_equivalent_corpus_uri(
    upstream, tmp_path, monkeypatch, saved_uri, next_uri, leg
):
    from simons.training import spot_job
    from simons.training.trainer import write_checkpoint

    release(tmp_path / "corpus")
    (tmp_path / "corpus/audit.json").unlink()
    saved = parent()
    saved["corpus_uri"] = saved_uri
    saved["recipe"]["training"]["covariate_weight"] = 0.05
    write_stage(tmp_path, "stage3", saved)
    before = snapshot(tmp_path / "state/stage3")
    args = job_args(
        tmp_path,
        upstream,
        leg=leg,
        corpus_uri=next_uri,
        condition="cov-w2",
        override=["covariate_weight=.05"],
        provisional_unaudited_corpus_attempt="tiny",
    )
    called = []

    def train(path, output, **kwargs):
        called.append(kwargs)
        assert kwargs["resume"] is None
        final = kwargs["checkpoint_dir"] / "checkpoint-final.pt"
        write_checkpoint(
            final, {**checkpoint_payload(57600), "job_identity": kwargs["checkpoint_metadata"]}
        )
        return final

    monkeypatch.setattr(spot_job, "train", train)
    assert spot_job.run_leg(args) == 0
    assert len(called) == 1
    identity = json.loads((tmp_path / "state" / leg / "identity.json").read_text())
    assert identity["corpus_uri"] == URI
    assert identity["condition"] == CONDITION
    assert identity["corpus_release"] == PROVISIONAL
    assert snapshot(tmp_path / "state/stage3") == before


@pytest.mark.parametrize(
    "saved_uri,next_uri",
    [(URI, URI), (URI + "/", URI + "/"), (URI + "/", URI), (URI, URI + "/")],
)
@pytest.mark.parametrize("manifest", [True, False])
@pytest.mark.parametrize("complete,exit_code", [(True, 0), (False, -1)])
def test_partial_resume_trains_with_unchanged_fork_metadata(
    upstream, tmp_path, monkeypatch, saved_uri, next_uri, manifest, complete, exit_code
):
    import yaml
    from simons.training import spot_job
    from simons.training.trainer import read_checkpoint, write_checkpoint

    release(tmp_path / "corpus")
    (tmp_path / "corpus/attempt-status.json").write_text(
        json.dumps(dict(attempt_id="tiny", complete=complete, exit_code=exit_code))
    )
    (tmp_path / "corpus/audit.json").unlink()
    saved = parent()
    saved["corpus_release"].update(attempt_complete=complete, attempt_exit_code=exit_code)
    saved["corpus_uri"] = saved_uri
    saved["recipe"]["training"]["covariate_weight"] = 0.05
    checkpoint = write_stage(
        tmp_path, "stage3", saved, step=35000, completed=False, manifest=manifest
    )
    args = job_args(
        tmp_path,
        upstream,
        corpus_uri=next_uri,
        condition="cov-w2",
        override=["covariate_weight=.05"],
        provisional_unaudited_corpus_attempt="tiny",
    )
    expected_raw = copy.deepcopy(saved["recipe"])
    expected_raw["training"].update(
        micro_batches=1,
        warm_start=str(tmp_path / "state/stage2"),
        data=str(tmp_path / "corpus/univariate"),
        covariate_data=str(tmp_path / "corpus/covariates"),
    )
    expected_identity = {**saved, "corpus_uri": URI}
    called = []

    def train(path, output, **kwargs):
        called.append(kwargs)
        assert kwargs["resume"] == checkpoint
        assert read_checkpoint(kwargs["resume"])["step"] == 35000
        assert yaml.safe_load(path.read_text()) == expected_raw
        assert kwargs["checkpoint_metadata"] == expected_identity
        final = kwargs["checkpoint_dir"] / "checkpoint-final.pt"
        write_checkpoint(
            final, {**checkpoint_payload(57600), "job_identity": kwargs["checkpoint_metadata"]}
        )
        return final

    monkeypatch.setattr(spot_job, "train", train)
    assert spot_job.run_leg(args) == 0
    assert len(called) == 1
    directory = tmp_path / "state/stage3"
    assert json.loads((directory / "identity.json").read_text()) == expected_identity
    marker = json.loads((directory / "completed.json").read_text())
    assert marker["job_identity"] == expected_identity
    assert marker["condition"] == CONDITION
    assert marker["corpus_release"] == saved["corpus_release"]
    assert read_checkpoint(directory / "checkpoint-final.pt")["job_identity"] == expected_identity


@pytest.mark.parametrize("manifest", [True, False])
@pytest.mark.parametrize(
    "change", ["drop-condition", "change-condition", "drop-flag", "change-attempt"]
)
def test_partial_resume_refuses_changed_fork(upstream, tmp_path, monkeypatch, manifest, change):
    from simons.training import spot_job

    release(tmp_path / "corpus")
    saved = parent()
    saved["recipe"]["training"]["covariate_weight"] = 0.05
    write_stage(tmp_path, "stage3", saved, step=35000, completed=False, manifest=manifest)
    args = job_args(
        tmp_path,
        upstream,
        condition="cov-w2",
        override=["covariate_weight=.05"],
        provisional_unaudited_corpus_attempt="tiny",
    )
    if change == "drop-condition":
        args.condition, args.override = None, []
    elif change == "change-condition":
        args.override = ["covariate_weight=.1"]
    elif change == "drop-flag":
        args.provisional_unaudited_corpus_attempt = None
    else:
        args.provisional_unaudited_corpus_attempt = "other"
    monkeypatch.setattr(spot_job, "train", lambda *a, **kw: pytest.fail("training began"))
    before = snapshot(tmp_path / "state")
    with pytest.raises(ValueError, match="condition|receipt|provisional"):
        spot_job.run_leg(args)
    assert snapshot(tmp_path / "state") == before


@pytest.mark.parametrize("saved_uri,next_uri", [(URI + "/", URI), (URI, URI + "/")])
def test_provisional_resume_comparison_normalizes_corpus_uri(upstream, saved_uri, next_uri):
    from simons.training.screen import compatible_resume

    saved = {**parent(), "corpus_uri": saved_uri}
    expected = {**copy.deepcopy(saved), "corpus_uri": next_uri}
    before = copy.deepcopy((saved, expected))
    assert compatible_resume(saved, expected) == {**saved, "corpus_uri": URI}
    assert (saved, expected) == before


def test_reference_corpus_uri_comparisons_remain_literal(upstream, tmp_path):
    from simons.training.screen import check_identity, compatible_resume

    saved = {**parent(receipt=AUDITED, condition=None), "corpus_uri": URI + "/"}
    with pytest.raises(ValueError, match="corpus identity changed"):
        check_identity(saved, job_args(tmp_path, upstream))
    with pytest.raises(ValueError, match="experiment identity changed"):
        compatible_resume(saved, {**saved, "corpus_uri": URI})
