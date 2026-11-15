"""Offline guest execution and artifact identity contracts."""

import base64
import gzip
import hashlib
import json
import os
import subprocess
import sys

import pytest

from scripts.corpus_v4 import make_exposure_job as jobs


def arguments(**changes):
    return dict(
        name="probe",
        repo_sha="a" * 40,
        upstream_pin="b" * 40,
        corpus_uri="gs://offline/corpus",
        batches=2,
        seeds=[11, 22],
        **changes,
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(name="../x"),
        dict(repo_sha="main"),
        dict(upstream_pin="x"),
        dict(corpus_uri="gs://offline/../x"),
        dict(batches=0),
        dict(seeds=[]),
        dict(seeds=[-1]),
        dict(overrides=["unknown=1"]),
        dict(overrides=["covariate_mix=0,0,0"]),
    ],
)
def test_validation(change):
    options = arguments()
    options.update(change)
    with pytest.raises(ValueError):
        jobs.generate(**options)


def test_cli_syntax_and_embedded_guest_hash(tmp_path):
    path = tmp_path / "job.sh"
    args = []
    for key, value in arguments().items():
        if key == "seeds":
            for seed in value:
                args += ["--seed", str(seed)]
        else:
            args += ["--" + key.replace("_", "-"), str(value)]
    assert jobs.main([*args, "--output", str(path)]) == 0
    assert os.access(path, os.X_OK)
    source = path.read_text()
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    embedded = source.split("# guest-gzip-base64: ")[1].splitlines()[0]
    guest = gzip.decompress(base64.b64decode(embedded))
    assert guest.decode() == jobs.GUEST
    assert hashlib.sha256(guest).hexdigest() in source
    assert "prepare.py" in source


@pytest.fixture
def machine(tmp_path, monkeypatch):
    bins = tmp_path / "bin"
    bins.mkdir()
    events = tmp_path / "events.jsonl"
    fake = (
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json, os, pathlib, shutil, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["EVENTS"], "a") as file:
    file.write(json.dumps([name, *args]) + "\n")
if name == "git":
    if args[0] == "clone":
        target = pathlib.Path(args[-1]); target.mkdir(parents=True)
        if "Simons" not in args[-2]:
            (target / "third_party/simons").mkdir(parents=True)
    elif args[:2] == ["rev-parse", "HEAD"]:
        print("b" * 40 if "upstream" in os.getcwd() else "a" * 40)
elif name == "gcloud":
    if os.environ.get("FAIL_UPLOAD") and args[-1].startswith("gs://"):
        sys.exit(23)
    if args[:3] == ["storage", "cp", "--recursive"] and args[-2].startswith("gs://"):
        (pathlib.Path(args[-1]) / "covariates").mkdir(parents=True)
elif args[:2] == ["-m", "venv"]:
    target = pathlib.Path(args[-1]) / "bin"; target.mkdir(parents=True)
    (target / "python").symlink_to(sys.argv[0])
elif args[:2] == ["-m", "pip"]:
    if args[2] == "freeze": print("torch==2.5.0\nnumpy==2.0.0")
elif args and args[0].endswith("prepare.py"):
    pass
elif args[:2] == ["-m", "scripts.corpus_v4.covariate_exposure"]:
    if os.environ.get("FAIL_AUDIT"): sys.exit(19)
    target = pathlib.Path(args[args.index("--output") + 1]); target.mkdir(parents=True, exist_ok=True)
    leg = args[args.index("--pass") + 1]
    (target / (leg + ".json")).write_text('{"replay":"real-only statistical"}')
    (target / (leg + ".csv")).write_text("replay\nreal-only statistical\n")
else:
    os.execv(os.environ["REAL_PYTHON"], [os.environ["REAL_PYTHON"], *args])
"""
    )
    for name in ("python3", "git", "gcloud"):
        path = bins / name
        path.write_text(fake)
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(bins) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("EVENTS", str(events))
    monkeypatch.setenv("REAL_PYTHON", sys.executable)
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("STATE_URI", "gs://offline/state")
    job = tmp_path / "job.sh"
    job.write_text(jobs.generate(**arguments()))

    def invoke(**env):
        return subprocess.run(
            ["bash", str(job)], capture_output=True, text=True, env={**os.environ, **env}
        )

    def calls():
        return [json.loads(line) for line in events.read_text().splitlines()]

    return invoke, tmp_path / "run/state/exposure/probe", calls


def test_state_hashes_metrics_covariates_only_and_preflight(machine):
    invoke, root, calls = machine
    result = invoke()
    assert result.returncode == 0, result.stderr
    marker = json.loads((root / "completed.json").read_text())
    assert marker["replay"] == marker["inputs"]["replay"] == "real-only statistical"
    assert len(marker["runs"]) == 8
    assert all(r["peak_rss_bytes"] > 0 and r["wall_seconds"] >= 0 for r in marker["runs"].values())
    for name, digest in marker["artifacts"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert len(list(root.glob("stage*.csv"))) == len(list(root.glob("stage*.json"))) == 8
    copies = [c for c in calls() if c[0] == "gcloud"]
    downloads = [c for c in copies if c[-2].startswith("gs://")]
    assert len(downloads) == 1
    assert downloads[0][-2] == "gs://offline/corpus/covariates"
    assert all(not c[-1].startswith("gs://offline/corpus") for c in copies)
    before = len(calls())
    assert invoke().returncode == 0
    assert not any(c[0] in ("git", "gcloud") or c[1:3] == ["-m", "pip"] for c in calls()[before:])


@pytest.mark.parametrize("failure", ["FAIL_AUDIT", "FAIL_UPLOAD"])
def test_failure_has_no_completion_marker(machine, failure):
    invoke, root, _ = machine
    assert invoke(**{failure: "1"}).returncode != 0
    assert not (root / "completed.json").exists()


def test_state_destination_overlap_refused_before_external_calls(machine):
    invoke, _, calls = machine
    result = invoke(STATE_URI="gs://offline/corpus/run")
    assert result.returncode != 0 and "overlap" in result.stderr
    assert not any(c[0] in ("git", "gcloud") for c in calls())


def test_completion_records_replay_in_environment_and_rejects_changed_inputs(machine):
    invoke, root, calls = machine
    assert invoke().returncode == 0
    assert "real-only statistical" in (root / "environment.txt").read_text()
    inputs = json.loads((root / "inputs.json").read_text())
    inputs["seed"] = "different"
    (root / "inputs.json").write_text(json.dumps(inputs))
    before = len(calls())
    result = invoke()
    assert result.returncode != 0 and "different inputs" in result.stderr
    assert not any(c[0] in ("git", "gcloud") for c in calls()[before:])


@pytest.fixture
def completion_state(tmp_path):
    namespace = {"__name__": "exposure_guest_test"}
    exec(compile(jobs.GUEST, "exposure-guest.py", "exec"), namespace)
    config = dict(
        replay="real-only statistical",
        seeds=[11, 22],
        passes=[
            ["stage3", "main"],
            ["stage4", "main"],
            ["stage4-long", "main"],
            ["stage4-long", "long"],
        ],
    )
    names = (
        "stage3-main-seed11",
        "stage4-main-seed11",
        "stage4-long-main-seed11",
        "stage4-long-long-seed11",
        "stage3-main-seed22",
        "stage4-main-seed22",
        "stage4-long-main-seed22",
        "stage4-long-long-seed22",
    )
    files = {"inputs.json": json.dumps(config), "environment.txt": "offline"}
    files.update({name + ext: "evidence" for name in names for ext in (".json", ".csv")})
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    record = dict(
        schema=1,
        replay=config["replay"],
        inputs=config,
        runs={name: dict(exit_code=0, wall_seconds=1.0, peak_rss_bytes=1024) for name in names},
        artifacts={name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()},
    )
    return namespace["preflight"], config, tmp_path, record


def test_preflight_accepts_complete_successful_runs_and_zero_metrics(completion_state):
    preflight, config, root, record = completion_state
    record["runs"]["stage3-main-seed11"].update(wall_seconds=0, peak_rss_bytes=0)
    (root / "completed.json").write_text(json.dumps(record))
    assert preflight(config, root) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", None),
        ("schema", 999),
        *[("schema", value) for value in (True, False, 1.0, 0.0)],
        ("runs", None),
        ("runs", {}),
        ("runs", []),
        ("runs.stage3-main-seed11", None),
        ("runs.stage3-main-seed11", {}),
        ("runs.unexpected-run", dict(exit_code=0, wall_seconds=1, peak_rss_bytes=1024)),
        ("runs.stage3-main-seed11.exit_code", None),
        ("runs.stage3-main-seed11.exit_code", 19),
        *[("runs.stage3-main-seed11.exit_code", value) for value in (True, False, 1.0, 0.0)],
        ("runs.stage3-main-seed11.wall_seconds", None),
        ("runs.stage3-main-seed11.peak_rss_bytes", None),
        *[
            ("runs.stage3-main-seed11." + metric, value)
            for metric in ("wall_seconds", "peak_rss_bytes")
            for value in (-1, float("nan"), float("inf"), -float("inf"), "1", True)
        ],
    ],
)
def test_preflight_refuses_malformed_completion_records(completion_state, field, value):
    preflight, config, root, record = completion_state
    parts = field.split(".")
    target = record
    for part in parts[:-1]:
        target = target[part]
    if value is None:
        target.pop(parts[-1])
    else:
        target[parts[-1]] = value
    (root / "completed.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="completion"):
        preflight(config, root)
