"""Offline execution of the generated scoring job; external installation is faked."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PIN = "8e338e915d26c8960713889ef12ab1cf5c08d591"
SHA = "a" * 40
TASKS = b"tasks:\n  - name: alpha\n    dataset_path: fake/data\n  - name: beta/new\n    dataset_path: fake/data\n"
NAMES = b"alpha\nbeta/new\n"


@pytest.fixture
def score_jobs():
    path = ROOT / "scripts/gcp/jobs/make_score_job.py"
    assert path.exists(), "the scoring generator is missing"
    spec = importlib.util.spec_from_file_location("make_score_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(**overrides):
    return {
        "name": "probe_1",
        "repo_sha": SHA,
        "upstream_pin": PIN,
        "checkpoint_uri": "gs://offline/export",
        "tasks_uri": "gs://offline/tasks.yaml",
        "tasks_sha256": hashlib.sha256(TASKS).hexdigest(),
        **overrides,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"repo_sha": "main"},
        {"upstream_pin": "b" * 40},
        {"tasks_sha256": "bad"},
        {"checkpoint_uri": None},
        {"hub_weights": "abel/model", "hub_revision": "b" * 40},
        {"checkpoint_uri": None, "hub_weights": "abel/model"},
        {"checkpoint_uri": None, "hub_weights": "abel/model", "hub_revision": "main"},
        {"hub_revision": "b" * 40},
        {"checkpoint_uri": "https://bucket/export"},
        {"tasks_uri": "gs://bucket/a*"},
        {"tasks_uri": "gs://bucket/x\ny"},
        {"tasks_uri": "gs://bucket/../x"},
        {"names_uri": "gs://offline/names"},
        {"names_sha256": "c" * 64},
        {"names_uri": "gs://offline/names", "names_sha256": "bad"},
        {"name": "../escape"},
        {"name": "-flag"},
        {"name": "a/b"},
        {"name": "_probe"},
        {"processing": "other"},
        {"covariates": "other"},
        {"max_context": 0},
        {"max_context": True},
        {"max_context": "1; exit 0"},
    ],
)
def test_invalid_inputs_fail_before_generating(score_jobs, overrides):
    with pytest.raises(ValueError):
        score_jobs.generate(**arguments(**overrides))


@pytest.mark.parametrize("processing", ["raw", "published", "both"])
def test_generated_shell_is_valid(score_jobs, tmp_path, processing):
    path = tmp_path / "job.sh"
    path.write_text(score_jobs.generate(**arguments(processing=processing)))
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("covariates", [None, "all", "none", "other"])
def test_covariates_cli(tmp_path, covariates):
    output = tmp_path / "job.sh"
    command = [sys.executable, str(ROOT / "scripts/gcp/jobs/make_score_job.py")]
    for name, value in arguments().items():
        command += ["--" + name.replace("_", "-"), value]
    command += ["--output", str(output)]
    if covariates is not None:
        command += ["--covariates", covariates]
    result = subprocess.run(command, capture_output=True, text=True)
    if covariates == "other":
        assert result.returncode == 2 and "invalid choice" in result.stderr
        assert not output.exists()
        return
    assert result.returncode == 0, result.stderr
    assert os.access(output, os.X_OK)
    snippet = output.read_text().split("<<'SCORE_BUNDLE'\n", 1)[1].split("\nSCORE_BUNDLE\n", 1)[0]
    result = subprocess.run(
        [sys.executable, "-", str(tmp_path)], input=snippet, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / "inputs.json").read_text())
    assert config["covariates"] == (covariates or "all")


@pytest.fixture
def machine(score_jobs, tmp_path):
    score_name = "probe_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    bins = tmp_path / "bin"
    bins.mkdir()
    cloud = tmp_path / "cloud"
    (cloud / "export").mkdir(parents=True)
    (cloud / "export/config.json").write_text('{"width": 24}')
    (cloud / "export/model.safetensors").write_bytes(b"offline weights")
    (cloud / "tasks.yaml").write_bytes(TASKS)
    (cloud / "names.txt").write_bytes(NAMES)
    packages = tmp_path / "packages"
    (packages / "demand_forecast/bench/yardsticks").mkdir(parents=True)
    for path in ["demand_forecast", "demand_forecast/bench", "demand_forecast/bench/yardsticks"]:
        (packages / path / "__init__.py").write_text("")
    (packages / "demand_forecast/bench/yardsticks/runner.py").write_text(
        "import types\n"
        "def read_names(value):\n"
        "    return open(value[1:]).read().splitlines() if value else None\n"
        "def load_tasks(path, names=None):\n"
        "    return [types.SimpleNamespace(task_name=n) for n in (names or ['alpha', 'beta/new'])]\n"
    )
    (packages / "torch.py").write_text(
        "import os, types\n__version__ = '2.8.0+cu128'\n"
        "version = types.SimpleNamespace(cuda='12.8')\n"
        "cuda = types.SimpleNamespace(is_available=lambda: os.environ.get('NO_CUDA') != '1')\n"
    )
    for name, version in [
        ("fev", "0.10.0"),
        ("datasets", "3.6.0"),
        ("huggingface_hub", "0.36.0"),
        ("simons", "0.1.1"),
    ]:
        info = packages / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        (info / "direct_url.json").write_text(json.dumps({"vcs_info": {"commit_id": PIN}}))
    (packages / "huggingface_hub.py").write_text(
        "import os, pathlib, shutil\n"
        "def hf_hub_download(repo_id, filename, *, revision, token, local_dir):\n"
        "    assert repo_id == 'abel/model' and revision == '" + "b" * 40 + "' and token is False\n"
        "    target = pathlib.Path(local_dir) / filename\n"
        "    shutil.copy2(pathlib.Path(os.environ['CLOUD']) / 'export' / filename, target)\n"
        "    return str(target)\n"
    )
    scorer = tmp_path / "scorer.py"
    scorer.write_text(
        textwrap.dedent("""\
        import csv, hashlib, json, os, pathlib, sys
        args = sys.argv[1:]
        def value(flag): return args[args.index(flag) + 1]
        assert value('--model') == 'simons'
        assert '--resume' in args and '--keep-going' in args
        assert pathlib.Path(value('--checkpoint'), 'model.safetensors').is_file()
        assert pathlib.Path(value('--tasks')).read_bytes() == bytes.fromhex(os.environ['TASK_HEX'])
        if '--names' in args:
            assert pathlib.Path(value('--names')[1:]).read_text() == 'alpha\\nbeta/new\\n'
        assert not pathlib.Path(os.environ['HF_DATASETS_CACHE']).is_relative_to(pathlib.Path(os.environ['RUN_DIR']) / 'state')
        mode = value('--processing')
        out = pathlib.Path(value('--out-dir'))
        out.mkdir(parents=True, exist_ok=True)
        with open(os.environ['EVENTS'], 'a') as log:
            log.write(json.dumps(['score', mode, args, os.environ['HF_DATASETS_CACHE']]) + '\\n')
        if os.environ.get('CORRUPT_RAW') == '1' and mode == 'raw':
            (out / 'failures.json').write_text('{')
            print('partial restored failures.json')
            sys.exit(2)
        rows, failures = [], {}
        for name in ['alpha', 'beta/new']:
            if os.environ.get('FAIL_TASK') == '1' and name == 'alpha':
                failures[name] = {'message': 'inference broke\\nsecond line', 'exception_type': 'RuntimeError'}
                continue
            task = out / name
            task.mkdir(parents=True, exist_ok=True)
            window = task / 'window_0.npz'
            if window.exists():
                assert window.read_bytes() == b'resume sentinel'
            else:
                window.write_bytes(b'resume sentinel')
            row = {'task': name, 'model_name': value('--model-name'), 'run_identity': 'identity'}
            (task / 'run.json').write_text(json.dumps({'run_identity': 'identity', 'summary': row, 'window_sha256': {'window_0.npz': hashlib.sha256(window.read_bytes()).hexdigest()}}))
            rows.append(row)
        with (out / 'summaries.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['task', 'model_name', 'run_identity'])
            writer.writeheader()
            writer.writerows(rows)
        (out / 'failures.json').write_text(json.dumps(failures))
        sys.exit(bool(failures))
        """)
    )
    wrapper = bins / "python3"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
        import json, os, pathlib, runpy, shutil, sys
        args = sys.argv[1:]
        with open(os.environ['EVENTS'], 'a') as log: log.write(json.dumps(['python', *args]) + '\\n')
        if args[:2] == ['-m', 'venv'] or args[:1] == ['--fake-venv']:
            if args[:2] == ['-m', 'venv'] and os.environ.get('FAIL_VENV', '1') == '1':
                sys.exit('ensurepip is not available')
            target = pathlib.Path(args[-1]) / 'bin'
            target.mkdir(parents=True, exist_ok=True)
            (target / 'python').unlink(missing_ok=True)
            (target / 'python').symlink_to(os.environ['FAKE_PYTHON'])
            sys.exit(0)
        if args[:3] == ['-m', 'pip', 'install']:
            if 'uv' in args:
                target = pathlib.Path(args[args.index('--target') + 1]) / 'bin'
                target.mkdir(parents=True)
                shutil.copy2(pathlib.Path(os.environ['FAKE_BIN']) / 'uv', target / 'uv')
            sys.exit(0)
        if args[:3] == ['-m', 'pip', 'freeze']:
            print('fev==0.10.0\\ndatasets==3.6.0\\nhuggingface-hub==0.36.0\\ntorch==2.8.0+cu128')
            sys.exit(0)
        if args[:3] == ['-m', 'pip', 'check']: sys.exit(0)
        if args[:1] == ['-c'] and 'version_info' in args[1]:
            print('3.10')
            sys.exit(0)
        if args and args[0].endswith('/scripts/fev_yardstick.py'):
            os.execv(sys.executable, [sys.executable, os.environ['SCORER'], *args[1:]])
        os.execv(sys.executable, [sys.executable, *args])
        """)
    )
    (bins / "uv").write_text(
        "#!/bin/bash\n"
        'printf \'["uv", "%s"]\\n\' "$*" >> "$EVENTS"\n'
        'if [[ $1 == venv ]]; then exec "$FAKE_PYTHON" --fake-venv "${@: -1}"; fi\n'
    )
    (bins / "git").write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
        import json, os, pathlib, shutil, sys
        args = sys.argv[1:]
        with open(os.environ['EVENTS'], 'a') as log: log.write(json.dumps(['git', *args]) + '\\n')
        if args[0] == 'init':
            repo = pathlib.Path(args[-1])
            repo.mkdir(parents=True, exist_ok=True)
            shutil.copy2(os.environ['REQUIREMENTS'], repo / 'requirements-yardsticks.txt')
        if 'rev-parse' in args: print(os.environ.get('FETCHED_SHA', 'a' * 40))
        """)
    )
    (bins / "gcloud").write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
        import json, os, pathlib, shutil, sys
        args = sys.argv[1:]
        with open(os.environ['EVENTS'], 'a') as log: log.write(json.dumps(['gcloud', *args]) + '\\n')
        assert args[:2] == ['storage', 'cp']
        source, destination = args[2:4]
        shutil.copy2(pathlib.Path(os.environ['CLOUD']) / source.removeprefix('gs://offline/'), destination)
        """)
    )
    (bins / "nvidia-smi").write_text("#!/bin/bash\necho 580.105.08\n")
    for executable in bins.iterdir():
        executable.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(bins) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(packages),
        "RUN_DIR": str(tmp_path / "run"),
        "STATE_URI": "gs://offline/state",
        "CLOUD": str(cloud),
        "FAKE_PYTHON": str(wrapper),
        "FAKE_BIN": str(bins),
        "SCORER": str(scorer),
        "REQUIREMENTS": str(ROOT / "requirements-yardsticks.txt"),
        "EVENTS": str(tmp_path / "events.jsonl"),
        "TASK_HEX": TASKS.hex(),
    }
    env.pop("HOME", None)
    script = tmp_path / "job.sh"

    def run(options=None, **overrides):
        script.write_text(
            score_jobs.generate(**arguments(**{"name": score_name, **(options or {})}))
        )
        return subprocess.run(
            ["bash", str(script)], env={**env, **overrides}, capture_output=True, text=True
        )

    def events():
        return [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]

    yield run, Path(env["RUN_DIR"]) / "state/score" / score_name, events, cloud
    # The checkpoint path must remain stable across guest labels for the adapter's identity.
    for event in events():
        if event[0] == "score":
            checkpoint = Path(event[2][event[2].index("--checkpoint") + 1])
            if checkpoint.parent.name.startswith("demand-forecast-score-"):
                shutil.rmtree(checkpoint.parent, ignore_errors=True)


def test_both_modes_completion_early_exit_and_changed_inputs(machine):
    run, state, events, _ = machine
    options = {
        "names_uri": "gs://offline/names.txt",
        "names_sha256": hashlib.sha256(NAMES).hexdigest(),
        "max_context": 2048,
    }
    result = run(options)
    assert result.returncode == 0, result.stderr + result.stdout
    calls = [e for e in events() if e[0] == "score"]
    assert [e[1] for e in calls] == ["raw", "published"]
    steps = events()
    fetch = next(i for i, e in enumerate(steps) if e[:1] == ["git"] and "fetch" in e)
    verify = next(i for i, e in enumerate(steps) if e[:1] == ["git"] and "rev-parse" in e)
    installs = [
        i
        for i, e in enumerate(steps)
        if e[:4] == ["python", "-m", "pip", "install"] and "uv" not in e
    ]
    cuda = next(i for i, e in enumerate(steps) if e[0] == "python" and "environment" in e)
    downloads = [i for i, e in enumerate(steps) if e[0] == "gcloud"]
    scoring = [i for i, e in enumerate(steps) if e[0] == "score"]
    assert (
        fetch
        < verify
        < min(installs)
        <= max(installs)
        < cuda
        < min(downloads)
        <= max(downloads)
        < min(scoring)
    )
    assert any(e[0] == "uv" and "--no-python-downloads" in e[1] for e in steps)
    assert any(e[0] == "uv" and "--python 3.12" in e[1] for e in steps)
    assert "fev==0.10.0" in (state / "environment.txt").read_text()
    assert "580.105.08" in (state / "environment.txt").read_text()
    assert all("--names" in e[2] and e[2][-2:] == ["--max-context", "2048"] for e in calls)
    marker = json.loads((state / "completed.json").read_text())
    assert marker["inputs"]["repo_sha"] == SHA
    assert marker["checkpoint_sha256"] == {
        "config.json": hashlib.sha256(b'{"width": 24}').hexdigest(),
        "model.safetensors": hashlib.sha256(b"offline weights").hexdigest(),
    }
    for mode in ["raw", "published"]:
        report = marker["modes"][mode]
        assert report["requested"] == report["completed"] == ["alpha", "beta/new"]
        assert report["failed"] == {} and report["wall_seconds"] >= 0
    before = events()
    assert run(options).returncode == 0
    assert not any(e[0] in ["git", "gcloud", "score", "uv"] for e in events()[len(before) :])
    saved = (state / "completed.json").read_bytes()
    result = run({**options, "max_context": 1024})
    assert result.returncode != 0 and "different inputs" in result.stderr
    assert (state / "completed.json").read_bytes() == saved


@pytest.mark.parametrize(
    ("options", "expected"),
    [({}, "all"), ({"covariates": "all"}, "all"), ({"covariates": "none"}, "none")],
)
def test_covariates_recorded_and_forwarded_to_scorer(machine, options, expected):
    run, state, events, _ = machine
    result = run(options)
    assert result.returncode == 0, result.stderr
    assert json.loads((state / "inputs.json").read_text())["covariates"] == expected
    marker = json.loads((state / "completed.json").read_text())
    assert marker["inputs"]["covariates"] == expected
    calls = [event for event in events() if event[0] == "score"]
    assert [event[1] for event in calls] == ["raw", "published"]
    for _, _, command, _ in calls:
        if expected == "none":
            index = command.index("--covariates")
            assert command[index : index + 2] == ["--covariates", "none"]
        else:
            assert "--covariates" not in command


@pytest.mark.parametrize(("recorded", "resumed"), [("all", "none"), ("none", "all")])
@pytest.mark.parametrize("with_marker", [False, True])
def test_changed_covariates_refused_on_resume(machine, recorded, resumed, with_marker):
    run, state, events, _ = machine
    result = run({"covariates": recorded})
    assert result.returncode == 0, result.stderr
    marker = state / "completed.json"
    if not with_marker:
        marker.unlink()
    saved = {path: path.read_bytes() for path in state.rglob("*") if path.is_file()}
    before = len(events())
    result = run({"covariates": resumed})
    assert result.returncode != 0 and "different inputs" in result.stderr
    assert {path: path.read_bytes() for path in state.rglob("*") if path.is_file()} == saved
    assert not any(event[0] in ["score", "git", "gcloud", "uv"] for event in events()[before:])


def test_failed_task_attempts_both_modes_and_resumes(machine):
    run, state, events, _ = machine
    result = run(FAIL_TASK="1")
    assert result.returncode != 0
    marker = json.loads((state / "completed.json").read_text())
    assert [e[1] for e in events() if e[0] == "score"] == ["raw", "published"]
    for mode in ["raw", "published"]:
        assert marker["modes"][mode]["failed"] == {"alpha": "inference broke"}
        assert marker["modes"][mode]["completed"] == ["beta/new"]
    assert run().returncode == 0
    assert len([e for e in events() if e[0] == "score"]) == 4


@pytest.mark.parametrize("changed", ["tasks", "names", "weights", "repo", "cuda"])
def test_input_or_environment_errors_stop_before_scoring(machine, changed):
    run, _, events, cloud = machine
    options, overrides = {}, {}
    if changed == "tasks":
        (cloud / "tasks.yaml").write_bytes(b"changed")
    if changed == "names":
        options = {"names_uri": "gs://offline/names.txt", "names_sha256": "f" * 64}
    if changed == "weights":
        (cloud / "export/model.safetensors").unlink()
    if changed == "repo":
        overrides = {"FETCHED_SHA": "f" * 40}
    if changed == "cuda":
        overrides = {"NO_CUDA": "1"}
    assert run(options, **overrides).returncode != 0
    assert not any(e[0] == "score" for e in events())


def test_hub_source_is_anonymous_and_single_mode(machine):
    run, state, events, _ = machine
    result = run(
        {
            "checkpoint_uri": None,
            "hub_weights": "abel/model",
            "hub_revision": "b" * 40,
            "processing": "published",
        }
    )
    assert result.returncode == 0, result.stderr
    assert [e[1] for e in events() if e[0] == "score"] == ["published"]
    assert not any(e[0] == "gcloud" and "/export/" in e[3] for e in events())
    assert json.loads((state / "completed.json").read_text())["inputs"]["hub_revision"] == "b" * 40


def test_corrupted_evidence_prevents_early_exit(machine):
    run, state, events, _ = machine
    assert run().returncode == 0
    (state / "raw/alpha/window_0.npz").unlink()
    assert run().returncode == 0
    assert len([e for e in events() if e[0] == "score"]) == 4


def test_symlink_state_cannot_escape(machine, tmp_path):
    run, state, _, _ = machine
    state.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    state.symlink_to(outside, target_is_directory=True)
    result = run()
    assert result.returncode != 0
    assert list(outside.iterdir()) == []


def test_restored_state_resumes_with_new_guest_label(machine, tmp_path):
    run, state, events, _ = machine
    assert run(FAIL_TASK="1").returncode != 0
    old_window = state / "raw/beta/new/window_0.npz"
    saved_time = old_window.stat().st_mtime_ns
    new_run = tmp_path / "new-label"
    shutil.copytree(state.parent.parent, new_run / "state")
    result = run(RUN_DIR=str(new_run))
    assert result.returncode == 0, result.stderr
    calls = [e for e in events() if e[0] == "score"]
    checkpoints = [e[2][e[2].index("--checkpoint") + 1] for e in calls]
    assert len(set(checkpoints)) == 1
    assert (
        new_run / "state/score" / state.name / "raw/beta/new/window_0.npz"
    ).stat().st_mtime_ns == saved_time


def test_changed_checkpoint_prefix_is_refused_on_resume(machine):
    run, state, events, cloud = machine
    assert run(FAIL_TASK="1").returncode != 0
    hashes = (state / "checkpoint-sha256.json").read_bytes()
    (cloud / "export/model.safetensors").write_bytes(b"different checkpoint")
    result = run()
    assert result.returncode != 0 and "different checkpoint contents" in result.stderr
    assert (state / "checkpoint-sha256.json").read_bytes() == hashes
    assert len([e for e in events() if e[0] == "score"]) == 2


def test_no_ensurepip_failure_uses_stdlib_venv_then_upgrades_interpreter(machine):
    run, _, events, _ = machine
    result = run(FAIL_VENV="0")
    assert result.returncode == 0, result.stderr
    assert not any(e[0] == "uv" and "--no-python-downloads" in e[1] for e in events())
    assert any(e[0] == "uv" and "--python 3.12" in e[1] for e in events())


def test_damaged_mode_report_does_not_prevent_other_mode(machine):
    run, state, events, _ = machine
    result = run(CORRUPT_RAW="1")
    assert result.returncode != 0
    assert [e[1] for e in events() if e[0] == "score"] == ["raw", "published"]
    marker = json.loads((state / "completed.json").read_text())
    assert set(marker["modes"]["raw"]["failed"]) == {"alpha", "beta/new"}
    assert marker["modes"]["published"]["failed"] == {}


def test_completion_repairs_identity_missing_from_partial_restore(machine):
    run, state, events, _ = machine
    assert run().returncode == 0
    (state / "inputs.json").unlink()
    before = len(events())
    result = run()
    assert result.returncode == 0, result.stderr
    assert (state / "inputs.json").is_file()
    assert not any(e[0] in ["score", "git", "gcloud"] for e in events()[before:])


@pytest.mark.parametrize("damage", ["{", "null", "{}"])
@pytest.mark.parametrize("marker_state", ["matching", "missing", "different"])
def test_damaged_identity_requires_matching_completion_marker(machine, damage, marker_state):
    run, state, events, _ = machine
    assert run().returncode == 0
    identity = state / "inputs.json"
    original = identity.read_bytes()
    identity.write_text(damage)
    marker_path = state / "completed.json"
    if marker_state == "missing":
        marker_path.unlink()
    elif marker_state == "different":
        marker = json.loads(marker_path.read_text())
        marker["inputs"]["max_context"] = 1024
        marker_path.write_text(json.dumps(marker))
    before = len(events())
    result = run()
    if marker_state == "matching":
        assert result.returncode == 0, result.stderr
        assert identity.read_bytes() == original
    else:
        assert result.returncode != 0
        assert (
            "input identity" if marker_state == "missing" else "different inputs"
        ) in result.stderr
        assert identity.read_text() == damage
    assert not any(e[0] in ["score", "git", "gcloud"] for e in events()[before:])


@pytest.mark.parametrize("damage", ["{", "null", "{}"])
@pytest.mark.parametrize("with_marker", [False, True])
@pytest.mark.parametrize("changed", [False, True])
def test_damaged_checkpoint_hashes_require_independent_evidence(
    machine, damage, with_marker, changed
):
    run, state, events, cloud = machine
    assert run().returncode == 0
    prior = state / "checkpoint-sha256.json"
    original = prior.read_bytes()
    prior.write_text(damage)
    if not with_marker:
        (state / "completed.json").unlink()
    if changed:
        (cloud / "export/model.safetensors").write_bytes(b"different checkpoint")
    before = len(events())
    result = run()
    calls = [e for e in events()[before:] if e[0] == "score"]
    if with_marker and not changed:
        assert result.returncode == 0, result.stderr
        assert prior.read_bytes() == original
        assert [e[1] for e in calls] == ["raw", "published"]
    else:
        assert result.returncode != 0
        expected = "different checkpoint contents" if with_marker else "checkpoint hash evidence"
        assert expected in result.stderr
        if not with_marker:
            assert "different checkpoint contents" not in result.stderr
        assert prior.read_text() == damage
        assert not calls


@pytest.mark.parametrize("changed", [False, True])
def test_missing_checkpoint_hashes_use_completion_evidence(machine, changed):
    run, state, events, cloud = machine
    assert run().returncode == 0
    (state / "checkpoint-sha256.json").unlink()
    if changed:
        (cloud / "export/model.safetensors").write_bytes(b"different checkpoint")
    before = len(events())
    result = run()
    calls = [e for e in events()[before:] if e[0] == "score"]
    if changed:
        assert result.returncode != 0 and "different checkpoint contents" in result.stderr
        assert not (state / "checkpoint-sha256.json").exists()
        assert not calls
    else:
        assert result.returncode == 0, result.stderr
        assert (state / "checkpoint-sha256.json").is_file()
        assert [e[1] for e in calls] == ["raw", "published"]


@pytest.mark.parametrize("damage", ["{}", "null"])
def test_missing_hashes_and_damaged_marker_refuse_without_independent_evidence(machine, damage):
    run, state, events, _ = machine
    assert run().returncode == 0
    (state / "checkpoint-sha256.json").unlink()
    (state / "completed.json").write_text(damage)
    before = len(events())
    result = run()
    assert result.returncode != 0 and "checkpoint hash evidence" in result.stderr
    assert not (state / "checkpoint-sha256.json").exists()
    assert not any(e[0] == "score" for e in events()[before:])


@pytest.mark.parametrize("changed", [False, True])
def test_completion_hashes_decide_when_local_hashes_conflict(machine, changed):
    run, state, events, cloud = machine
    assert run().returncode == 0
    prior = state / "checkpoint-sha256.json"
    original = prior.read_bytes()
    local_hashes = json.loads(original)
    local_hashes["model.safetensors"] = hashlib.sha256(b"different checkpoint").hexdigest()
    prior.write_text(json.dumps(local_hashes))
    damaged = prior.read_bytes()
    if changed:
        (cloud / "export/model.safetensors").write_bytes(b"different checkpoint")
    before = len(events())
    result = run()
    calls = [e for e in events()[before:] if e[0] == "score"]
    if changed:
        assert result.returncode != 0 and "different checkpoint contents" in result.stderr
        assert prior.read_bytes() == damaged
        assert not calls
    else:
        assert result.returncode == 0, result.stderr
        assert prior.read_bytes() == original
        assert [e[1] for e in calls] == ["raw", "published"]


@pytest.mark.parametrize("damage", ["{", "null", "{}", '{"inputs": null}'])
def test_damaged_completion_marker_resumes_with_durable_identity(machine, damage):
    run, state, events, _ = machine
    assert run().returncode == 0
    (state / "completed.json").write_text(damage)
    result = run()
    assert result.returncode == 0, result.stderr
    assert len([e for e in events() if e[0] == "score"]) == 4


@pytest.mark.parametrize("windows", [["window_0.npz"], "window_0.npz"])
def test_damaged_window_manifest_is_incomplete_evidence(score_jobs, tmp_path, windows):
    guest = {"__name__": "score_job_test"}
    exec(score_jobs.GUEST, guest)
    task = tmp_path / "raw/alpha"
    task.mkdir(parents=True)
    (task.parent / "summaries.csv").write_text(
        "task,model_name,run_identity\nalpha,probe-raw,identity\n"
    )
    (task / "run.json").write_text(
        json.dumps({"run_identity": "identity", "summary": {}, "window_sha256": windows})
    )
    assert guest["evidence"](tmp_path, "raw", ["alpha"], "probe-raw") == []


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("identity_damage", [None, "{", "null"])
def test_139f3f7_state_keeps_its_identity_and_checkpoint_path(machine, failed, identity_damage):
    run, state, events, _ = machine
    assert run(FAIL_TASK="1" if failed else "0").returncode == int(failed)
    identity = state / "inputs.json"
    config = json.loads(identity.read_text())
    config["guest_sha256"] = "a726e1e81f4ad804683b5164aebbf38aefb2107a7983a0b4faad8b1de200916f"
    identity.write_text(json.dumps(config, sort_keys=True) + "\n")
    marker_path = state / "completed.json"
    marker = json.loads(marker_path.read_text())
    marker["inputs"] = config
    marker["artifacts"]["inputs.json"] = hashlib.sha256(identity.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker))
    if identity_damage is not None:
        identity.write_text(identity_damage)
    before = len(events())
    result = run()
    assert result.returncode == 0, result.stderr
    assert json.loads(identity.read_text()) == config
    calls = [e for e in events()[before:] if e[0] == "score"]
    if failed:
        expected = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        assert len(calls) == 2
        assert all(
            Path(e[2][e[2].index("--checkpoint") + 1])
            == Path("/tmp") / ("demand-forecast-score-" + expected) / "checkpoint"
            for e in calls
        )
    else:
        assert not any(e[0] in ["score", "git", "gcloud"] for e in events()[before:])


def test_unknown_guest_identity_is_refused(machine):
    run, state, events, _ = machine
    assert run().returncode == 0
    identity = state / "inputs.json"
    config = json.loads(identity.read_text())
    config["guest_sha256"] = "f" * 64
    identity.write_text(json.dumps(config))
    before = len(events())
    result = run()
    assert result.returncode != 0 and "different inputs" in result.stderr
    assert json.loads(identity.read_text()) == config
    assert not any(e[0] in ["score", "git", "gcloud"] for e in events()[before:])
