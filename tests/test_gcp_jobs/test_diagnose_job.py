"""Exercise the generated CPU script with local stand-ins for external tools."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.bench

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/gcp/jobs/make_diagnose_job.py"
TASKS = b"tasks:\n- name: family/toy\n  dataset_path: fake\n"
NAMES = b"family/toy\n"
SHA = "a" * 40


def arguments(**changes):
    return {
        "name": "toy-job",
        "repo_sha": SHA,
        "tasks_uri": "gs://offline/tasks.yaml",
        "tasks_sha256": hashlib.sha256(TASKS).hexdigest(),
        "names_uri": "gs://offline/names.txt",
        "names_sha256": hashlib.sha256(NAMES).hexdigest(),
        **changes,
    }


@pytest.fixture
def jobs():
    spec = importlib.util.spec_from_file_location("make_diagnose_job", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_executable_valid_bash(tmp_path):
    out = tmp_path / "job.sh"
    command = [sys.executable, str(GENERATOR), "--output", str(out)]
    for key, value in arguments().items():
        command.extend(("--" + key.replace("_", "-"), value))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert os.access(out, os.X_OK)
    assert subprocess.run(["bash", "-n", str(out)], check=False).returncode == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"repo_sha": "main"},
        {"tasks_sha256": "a" * 63},
        {"name": "../toy"},
        {"tasks_uri": "gs://offline/../toy"},
        {"names_uri": None},
    ],
)
def test_generator_refuses_invalid_inputs(jobs, changes):
    with pytest.raises(ValueError):
        jobs.generate(**arguments(**changes))


@pytest.fixture
def machine(jobs, tmp_path):
    cloud, bins = tmp_path / "bucket", tmp_path / "bin"
    cloud.mkdir()
    bins.mkdir()
    (cloud / "tasks.yaml").write_bytes(TASKS)
    (cloud / "names.txt").write_bytes(NAMES)
    stub = bins / "python3"
    stub.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
        import csv, json, os, pathlib, shutil, sys
        tool = pathlib.Path(sys.argv[0]).name
        args = sys.argv[1:]
        if tool == "python3" and args[:1] == ["-"] and len(args) == 2:
            os.execv(sys.executable, [sys.executable, *args])
        with open(os.environ["EVENTS"], "a") as log:
            log.write(json.dumps([tool, *args]) + "\\n")
        if tool == "gcloud":
            assert args[:2] == ["storage", "cp"]
            src, dst = args[2:]
            cloud = pathlib.Path(os.environ["CLOUD"])
            if src.startswith("gs://offline/"):
                shutil.copyfile(cloud / src.removeprefix("gs://offline/"), dst)
            else:
                if pathlib.Path(src).name == os.environ.get("FAIL_UPLOAD"):
                    sys.exit(17)
                target = cloud / dst.removeprefix("gs://offline/")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
        elif tool == "git":
            if args[0] == "init":
                pathlib.Path(args[1]).mkdir(parents=True)
            if "rev-parse" in args:
                print("a" * 40)
            if "checkout" in args:
                shutil.copyfile(pathlib.Path(os.environ["ROOT"]) / "requirements-yardsticks.txt",
                                pathlib.Path(args[1]) / "requirements-yardsticks.txt")
        elif tool == "uv":
            if args[0] == "venv":
                target = pathlib.Path(args[-1]) / "bin"
                target.mkdir(parents=True)
                (target / "python").symlink_to(os.environ["STUB"])
            elif args[:2] == ["pip", "install"] and "-r" in args:
                pins = pathlib.Path(args[args.index("-r") + 1]).read_text()
                with open(os.environ["EVENTS"], "a") as log:
                    log.write(json.dumps(["pins", pins]) + "\\n")
        elif args[:2] == ["-m", "demand_forecast.bench.static_diagnostics"]:
            assert pathlib.Path(args[args.index("--tasks") + 1]).read_bytes() == b"tasks:\\n- name: family/toy\\n  dataset_path: fake\\n"
            if "--names" in args:
                assert pathlib.Path(args[args.index("--names") + 1][1:]).read_bytes() == b"family/toy\\n"
            out = pathlib.Path(args[args.index("--out") + 1])
            name = os.environ.get("REPORT_TASK", "family/toy")
            out.mkdir()
            if name == "family/toy":
                (out / "family").mkdir()
                (out / "family/toy.json").write_text("{}\\n")
            (out / "level.json").write_text("{}\\n")
            with (out / "tasks.csv").open("w", newline="") as stream:
                csv.writer(stream).writerows([["task", "window"], [name, 0]])
            (out / "attributes.csv").write_text("task,window,cutoff,attribute\\n")
        elif args[:1] == ["-c"]:
            pass  # Runtime dependency check, no packages are installed here.
        else:
            os.execv(sys.executable, [sys.executable, *args])
        """)
    )
    stub.chmod(0o755)
    for tool in ("git", "gcloud", "uv"):
        (bins / tool).symlink_to(stub)
    events_path = tmp_path / "events.jsonl"
    run_dir = tmp_path / "run"
    env = {
        **os.environ,
        "PATH": str(bins) + os.pathsep + os.environ["PATH"],
        "RUN_DIR": str(run_dir),
        "STATE_URI": "gs://offline/state",
        "CLOUD": str(cloud),
        "STUB": str(stub),
        "EVENTS": str(events_path),
        "ROOT": str(ROOT),
    }
    executable = tmp_path / "job.sh"

    def run(options=None, **extra):
        executable.write_text(jobs.generate(**arguments(**(options or {}))))
        return subprocess.run(
            ["bash", str(executable)], capture_output=True, text=True, env={**env, **extra}
        )

    def events():
        return (
            [json.loads(line) for line in events_path.read_text().splitlines()]
            if events_path.exists()
            else []
        )

    return run, run_dir / "state/diagnose/toy-job", cloud, events


def test_first_run_publishes_completion_last_and_second_is_noop(machine):
    run, state, cloud, events = machine
    result = run()
    assert result.returncode == 0, result.stderr
    remote = cloud / "state/diagnose/toy-job"
    marker = json.loads((state / "completed.json").read_text())
    assert {"attributes.csv", "tasks.csv", "level.json", "family/toy.json"} <= marker[
        "artifacts"
    ].keys()
    for name, digest in marker["artifacts"].items():
        content = (state / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == digest
        assert (remote / name).read_bytes() == content
    first = events()
    uploads = [e[-1] for e in first if e[0] == "gcloud" and e[-1].startswith("gs://")]
    assert uploads[-1] == "gs://offline/state/diagnose/toy-job/completed.json"
    invoke = next(e for e in first if "demand_forecast.bench.static_diagnostics" in e)
    assert "--names" in invoke
    assert any(e[:4] == ["uv", "venv", "--python", "3.12"] for e in first)
    assert any(
        "torch==2.8.0+cpu" in e and "https://download.pytorch.org/whl/cpu" in e for e in first
    )
    pins = next(e[1] for e in first if e[0] == "pins")
    assert "numpy==2.2.6\n" in pins
    assert "torch==2.8.0+cpu\n" in pins
    assert "timestral" not in pins
    assert "chronos-forecasting" not in pins
    result = run()
    assert result.returncode == 0, result.stderr
    assert not [e for e in events()[len(first) :] if e[0] in {"git", "gcloud", "uv"} or "-m" in e]


@pytest.mark.parametrize("damage", ["attributes.csv", "family/toy.json", "completed.json"])
def test_damaged_completed_unit_is_refused(machine, damage):
    run, state, _, events = machine
    assert run().returncode == 0
    path = state / damage
    if damage == "family/toy.json":
        path.unlink()
    else:
        path.write_text("damaged\n")
    before = events()
    result = run()
    assert result.returncode != 0
    assert "damaged" in result.stderr.lower()
    assert not [e for e in events()[len(before) :] if e[0] in {"git", "gcloud", "uv"} or "-m" in e]


@pytest.mark.parametrize("filename", ["tasks.yaml", "names.txt"])
def test_input_hash_mismatch_stops_before_diagnostics(machine, filename):
    run, state, cloud, events = machine
    (cloud / filename).write_text("wrong\n")
    result = run()
    assert result.returncode != 0
    assert "sha256" in result.stderr.lower()
    assert not (state / "completed.json").exists()
    assert all("demand_forecast.bench.static_diagnostics" not in e for e in events())


def test_failed_publication_can_retry_without_names(machine):
    run, state, _, events = machine
    options = {"names_uri": None, "names_sha256": None}
    assert run(options, FAIL_UPLOAD="completed.json").returncode != 0
    assert not (state / "completed.json").exists()
    result = run(options)
    assert result.returncode == 0, result.stderr
    assert (state / "completed.json").is_file()
    commands = [e for e in events() if "demand_forecast.bench.static_diagnostics" in e]
    assert len(commands) == 2
    assert all("--names" not in e for e in commands)


@pytest.mark.parametrize(
    "name",
    [
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
def test_report_cannot_replace_job_state(machine, name):
    run, state, cloud, _ = machine
    result = run(REPORT_TASK=name)
    assert result.returncode != 0
    assert "reserved" in result.stderr
    assert not (state / "completed.json").exists()
    assert not (cloud / "state/diagnose/toy-job/completed.json").exists()


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
    ],
)
def test_guest_refuses_unsafe_report_name(machine, name):
    run, state, cloud, _ = machine
    result = run(REPORT_TASK=name)
    assert result.returncode != 0
    assert "unsafe diagnostic task path" in result.stderr
    assert not (state / "completed.json").exists()
    assert not (cloud / "state/diagnose/toy-job/completed.json").exists()
