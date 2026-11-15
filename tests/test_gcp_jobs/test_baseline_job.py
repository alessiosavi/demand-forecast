"""Offline contracts for the pinned classical-baseline guest job."""

import ast
import csv
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
SHA = "a" * 40
HISTORY_ERROR = "All time series in the dataset are too short for the chosen cutoff, horizon and min_context_length"
TASKS = b"tasks:\n- name: alpha\n  dataset_path: fake/data\n- task_name: beta/new\n  dataset_path: fake/data\n"
NAMES = b"beta/new\nalpha\n"


@pytest.fixture
def jobs():
    path = ROOT / "scripts/gcp/jobs/make_baseline_job.py"
    assert path.exists(), "baseline generator is missing"
    spec = importlib.util.spec_from_file_location("make_baseline_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(**changes):
    return {
        "name": "probe_1",
        "repo_sha": SHA,
        "tasks_uri": "gs://offline/tasks.yaml",
        "tasks_sha256": hashlib.sha256(TASKS).hexdigest(),
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"repo_sha": "main"},
        {"repo_sha": "A" * 40},
        {"tasks_sha256": "bad"},
        {"tasks_uri": "https://offline/tasks.yaml"},
        {"tasks_uri": "gs://offline/a*"},
        {"tasks_uri": "gs://offline/../x"},
        {"tasks_uri": "gs://offline/tasks.yaml/"},
        {"tasks_uri": "gs://offline/x\ny"},
        {"name": "../x"},
        {"name": "-flag"},
        {"name": "a/b"},
        {"name": "_name"},
        {"model": "unknown"},
        {"names_uri": "gs://offline/names.txt"},
        {"names_sha256": "b" * 64},
        {"names_uri": "gs://offline/names.txt", "names_sha256": "bad"},
    ],
)
def test_validation(jobs, changes):
    with pytest.raises(ValueError):
        jobs.generate(**arguments(**changes))


@pytest.mark.parametrize("model", [None, "seasonal_naive", "auto_ets", "auto_theta", "median3"])
def test_cli_bash_and_embedded_guest_hash(jobs, tmp_path, model):
    output = tmp_path / "job.sh"
    command = [sys.executable, str(ROOT / "scripts/gcp/jobs/make_baseline_job.py")]
    for key, value in arguments().items():
        command += ["--" + key.replace("_", "-"), value]
    if model:
        command += ["--model", model]
    result = subprocess.run(command + ["--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert os.access(output, os.X_OK)
    assert subprocess.run(["bash", "-n", str(output)]).returncode == 0
    source = output.read_text()
    snippet = source.split("<<'BASELINE_BUNDLE'\n", 1)[1].split("\nBASELINE_BUNDLE\n", 1)[0]
    unpack = subprocess.run(
        [sys.executable, "-", str(tmp_path)], input=snippet, capture_output=True, text=True
    )
    assert unpack.returncode == 0, unpack.stderr
    guest = (tmp_path / "baseline.py").read_bytes()
    assert guest.decode() == jobs.GUEST
    config = json.loads((tmp_path / "inputs.json").read_text())
    assert config["guest_sha256"] == hashlib.sha256(guest).hexdigest()
    assert config["model"] == (model or "seasonal_naive")
    ast.parse(guest, feature_version=(3, 10))


@pytest.fixture
def machine(jobs, tmp_path):
    bins, cloud, packages = (tmp_path / name for name in ("bin", "cloud", "packages"))
    for path in (bins, cloud, packages / "demand_forecast/bench"):
        path.mkdir(parents=True)
    (cloud / "tasks.yaml").write_bytes(TASKS)
    (cloud / "names.txt").write_bytes(NAMES)
    for path in (packages / "demand_forecast", packages / "demand_forecast/bench"):
        (path / "__init__.py").write_text("")
    (packages / "torch.py").write_text(
        "import os, types\n__version__ = '2.8.0+cpu'\n"
        "version = types.SimpleNamespace(cuda='12.8' if os.environ.get('GPU_TORCH') else None)\n"
    )
    (packages / "fev.py").write_text(
        textwrap.dedent(f"""\
        import json, os, types
        def Task(**values):
            name = values['task_name']
            with open(os.environ['EVENTS'], 'a') as log:
                log.write(json.dumps(['task', name, values]) + '\\n')
            if name == os.environ.get('BUILD_FAIL'):
                raise ValueError('invalid task constructor')
            if name == os.environ.get('SHORT_BUILD'):
                raise ValueError({HISTORY_ERROR!r})
            return types.SimpleNamespace(**values)
        """)
    )
    (packages / "demand_forecast/bench/run.py").write_text(
        textwrap.dedent(f"""\
        import csv, json, os, pathlib
        def run_baseline(specs, out_dir, model='seasonal_naive'):
            path = pathlib.Path(out_dir) / 'summaries.csv'
            with open(os.environ['EVENTS'], 'a') as log:
                log.write(json.dumps(['baseline', [s.name for s in specs], model]) + '\\n')
            rows = list(csv.DictReader(path.open())) if path.exists() else []
            def save():
                with path.open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['task', 'model_name', 'MASE', 'quantile_levels'])
                    writer.writeheader()
                    writer.writerows(rows)
            for spec in specs:
                if any(r['task'] == spec.name and r['model_name'] == model for r in rows):
                    continue
                task = spec.to_task()
                if task.task_name == os.environ.get('SHORT_TASK'):
                    raise ValueError({HISTORY_ERROR!r})
                if task.task_name == os.environ.get('FAIL_TASK'):
                    raise RuntimeError('inference broke\\nsecond line')
                rows.append({{'task': spec.name, 'model_name': model, 'MASE': 0.75, 'quantile_levels': '[0.1, 0.5, 0.9]'}})
                save()
                if spec.name == os.environ.get('CRASH_AFTER_ROW'):
                    os._exit(19)
            if not path.exists():
                save()
            return rows
        """)
    )
    for name, version in (
        ("fev", "0.10.0"),
        ("datasets", "3.6.0"),
        ("statsforecast", "2.0.1"),
        ("torch", "2.8.0+cpu"),
    ):
        info = packages / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    wrapper = bins / "python3"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
        import json, os, pathlib, shutil, sys
        args = sys.argv[1:]
        with open(os.environ['EVENTS'], 'a') as log:
            log.write(json.dumps(['python', *args]) + '\\n')
        if args[:2] == ['-m', 'venv'] or args[:1] == ['--fake-venv']:
            if args[:2] == ['-m', 'venv'] and os.environ.get('FAIL_VENV'):
                sys.exit('ensurepip is not available')
            target = pathlib.Path(args[-1]) / 'bin'
            target.mkdir(parents=True, exist_ok=True)
            (target / 'python').symlink_to(os.environ['FAKE_PYTHON'])
            sys.exit(0)
        if args[:3] == ['-m', 'pip', 'install']:
            if os.environ.get('FAIL_INSTALL'): sys.exit(17)
            if 'uv' in args:
                target = pathlib.Path(args[args.index('--target') + 1]) / 'bin'
                target.mkdir(parents=True)
                shutil.copy2(pathlib.Path(os.environ['FAKE_BIN']) / 'uv', target / 'uv')
            if '--constraint' in args:
                with open(os.environ['EVENTS'], 'a') as log:
                    log.write(json.dumps(['constraints', pathlib.Path(args[args.index('--constraint') + 1]).read_text()]) + '\\n')
            sys.exit(0)
        if args[:3] == ['-m', 'pip', 'check']: sys.exit(0)
        if args[:3] == ['-m', 'pip', 'freeze']:
            print('fev==0.10.0\\ndatasets==3.6.0\\nstatsforecast==2.0.1\\ntorch==2.8.0+cpu')
            sys.exit(0)
        os.execv(sys.executable, [sys.executable, *args])
        """)
    )
    (bins / "uv").write_text('#!/bin/bash\nexec "$FAKE_PYTHON" --fake-venv "${@: -1}"\n')
    (bins / "git").write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
        import json, os, pathlib, sys
        args = sys.argv[1:]
        with open(os.environ['EVENTS'], 'a') as log:
            log.write(json.dumps(['git', *args]) + '\\n')
        if args[0] == 'init': pathlib.Path(args[-1]).mkdir(parents=True)
        if 'rev-parse' in args: print(os.environ.get('FETCHED_SHA', 'a' * 40))
        """)
    )
    (bins / "gcloud").write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
        import json, os, pathlib, shutil, sys
        args = sys.argv[1:]
        with open(os.environ['EVENTS'], 'a') as log:
            log.write(json.dumps(['gcloud', *args]) + '\\n')
        assert args[:2] == ['storage', 'cp']
        source, destination = args[2:]
        cloud = pathlib.Path(os.environ['CLOUD'])
        if source.startswith('gs://offline/'):
            if os.environ.get('FAIL_FETCH'): sys.exit(22)
            shutil.copy2(cloud / source.removeprefix('gs://offline/'), destination)
        else:
            if os.environ.get('FAIL_UPLOAD') == pathlib.Path(source).name: sys.exit(23)
            target = cloud / destination.removeprefix('gs://offline/')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        """)
    )
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
        "EVENTS": str(tmp_path / "events.jsonl"),
    }
    script = tmp_path / "job.sh"

    def run(options=None, **overrides):
        script.write_text(jobs.generate(**arguments(**(options or {}))))
        return subprocess.run(
            ["bash", str(script)], env={**env, **overrides}, capture_output=True, text=True
        )

    def events():
        path = Path(env["EVENTS"])
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    return run, Path(env["RUN_DIR"]) / "state/baseline/probe_1", events, cloud


def test_guest_success_pins_artifacts_publication_and_resume(machine):
    run, state, events, cloud = machine
    options = {
        "names_uri": "gs://offline/names.txt",
        "names_sha256": hashlib.sha256(NAMES).hexdigest(),
    }
    result = run(options, FAIL_VENV="1")
    assert result.returncode == 0, result.stderr
    status = json.loads((state / "status.json").read_text())
    assert status["requested"] == status["completed"] == ["beta/new", "alpha"]
    assert status["failed"] == {}
    assert status["repo_sha"] == SHA
    assert status["tasks_sha256"] == hashlib.sha256(TASKS).hexdigest()
    assert status["versions"]["fev"] == "0.10.0"
    assert status["versions"]["datasets"] == "3.6.0"
    marker = json.loads((state / "completed.json").read_text())
    assert marker["inputs"] == json.loads((state / "inputs.json").read_text())
    assert {"summaries.csv", "status.json", "inputs.json", "environment.txt"} <= marker[
        "artifacts"
    ].keys()
    for name, digest in marker["artifacts"].items():
        assert hashlib.sha256((state / name).read_bytes()).hexdigest() == digest
        assert (cloud / "state/baseline/probe_1" / name).read_bytes() == (state / name).read_bytes()
    copies = [event for event in events() if event[0] == "gcloud"]
    assert copies[-1][-1] == "gs://offline/state/baseline/probe_1/completed.json"
    installs = [e for e in events() if e[:4] == ["python", "-m", "pip", "install"]]
    assert any("https://download.pytorch.org/whl/cpu" in e for e in installs)
    assert not any("--system-site-packages" in e for e in events())
    assert not any(
        "requirements-yardsticks.txt" in arg or "cu128" in arg for e in installs for arg in e
    )
    pins = next(e[1] for e in events() if e[0] == "constraints")
    for pin in ("fev==0.10.0", "datasets==3.6.0", "statsforecast==2.0.1", "torch==2.8.0+cpu"):
        assert pin in pins and pin in (state / "environment.txt").read_text()
    before = len(events())
    assert run(options).returncode == 0
    assert not any(e[0] in ("git", "gcloud", "baseline", "task") for e in events()[before:])


@pytest.mark.parametrize("failure", ["SHORT_TASK", "SHORT_BUILD"])
def test_lack_of_history_is_reported_and_other_tasks_continue(machine, failure):
    run, state, _, _ = machine
    result = run(**{failure: "alpha"})
    assert result.returncode == 0, result.stderr
    status = json.loads((state / "status.json").read_text())
    assert status["completed"] == ["beta/new"]
    assert status["failed"] == {"alpha": HISTORY_ERROR}
    assert status["failure_kinds"] == {"alpha": "insufficient_history"}
    assert (state / "completed.json").exists()


@pytest.mark.parametrize("failure", ["FAIL_TASK", "BUILD_FAIL", "CRASH_AFTER_ROW"])
def test_failed_child_has_no_completion_marker_and_resume_preserves_rows(machine, failure):
    run, state, events, cloud = machine
    result = run(**{failure: "alpha"})
    assert result.returncode != 0
    assert not (state / "completed.json").exists()
    assert not (cloud / "state/baseline/probe_1/completed.json").exists()
    status = json.loads((state / "status.json").read_text())
    assert "alpha" in status["failed"]
    assert "beta/new" in status["completed"]
    saved = next(
        r for r in csv.DictReader((state / "summaries.csv").open()) if r["task"] == "beta/new"
    )
    before = len(events())
    result = run()
    assert result.returncode == 0, result.stderr
    assert (state / "completed.json").exists()
    assert not any(e[:2] == ["task", "beta/new"] for e in events()[before:])
    assert saved in list(csv.DictReader((state / "summaries.csv").open()))


@pytest.mark.parametrize(
    "changes",
    [
        {"FAIL_INSTALL": "1"},
        {"FAIL_FETCH": "1"},
        {"FAIL_UPLOAD": "summaries.csv"},
        {"FAIL_UPLOAD": "completed.json"},
        {"FETCHED_SHA": "b" * 40},
        {"GPU_TORCH": "1"},
    ],
)
def test_setup_or_publication_failure_leaves_no_marker(machine, changes):
    run, state, _, cloud = machine
    assert run(**changes).returncode != 0
    assert not (state / "completed.json").exists()
    assert not (cloud / "state/baseline/probe_1/completed.json").exists()


@pytest.mark.parametrize("filename", ["tasks.yaml", "names.txt"])
def test_hash_mismatch_stops_before_tasks(machine, filename):
    run, state, events, cloud = machine
    (cloud / filename).write_text("tampered\n")
    result = run(
        {"names_uri": "gs://offline/names.txt", "names_sha256": hashlib.sha256(NAMES).hexdigest()}
    )
    assert result.returncode != 0 and "sha256 mismatch" in result.stderr
    assert not any(e[0] == "task" for e in events())
    assert not (state / "completed.json").exists()


@pytest.mark.parametrize(
    "names", [b"", b"alpha, beta/new\n", b"alpha\n\n", b"alpha\nalpha\n", b"unknown\n"]
)
def test_invalid_names_stop_before_tasks(machine, names):
    run, state, events, cloud = machine
    (cloud / "names.txt").write_bytes(names)
    result = run(
        {"names_uri": "gs://offline/names.txt", "names_sha256": hashlib.sha256(names).hexdigest()}
    )
    assert result.returncode != 0
    assert not any(e[0] == "task" for e in events())
    assert not (state / "completed.json").exists()


@pytest.mark.parametrize(
    "change", [{"model": "auto_ets"}, {"repo_sha": "b" * 40}, {"tasks_sha256": "c" * 64}]
)
def test_changed_inputs_are_refused_without_mutation(machine, change):
    run, state, events, _ = machine
    assert run().returncode == 0
    saved = {p: p.read_bytes() for p in state.rglob("*") if p.is_file()}
    before = len(events())
    result = run(change)
    assert result.returncode != 0 and "different inputs" in result.stderr
    assert saved == {p: p.read_bytes() for p in state.rglob("*") if p.is_file()}
    assert not any(e[0] in ("git", "gcloud", "baseline", "task") for e in events()[before:])


def test_missing_artifact_republishes_without_recomputing_saved_rows(machine):
    run, state, events, _ = machine
    assert run().returncode == 0
    (state / "environment.txt").unlink()
    before = len(events())
    result = run()
    assert result.returncode == 0, result.stderr
    assert not any(e[0] == "task" for e in events()[before:])
    assert (state / "environment.txt").exists()


def test_fresh_guest_resumes_restored_partial_state(machine):
    run, state, events, cloud = machine
    assert run(FAIL_TASK="alpha").returncode != 0
    shutil.rmtree(state)
    shutil.copytree(cloud / "state/baseline/probe_1", state)
    before = len(events())
    assert run().returncode == 0
    assert not any(e[:2] == ["task", "beta/new"] for e in events()[before:])


@pytest.mark.parametrize(
    "uri", ["https://offline/state", "gs://offline/../state", "gs://offline/state/*"]
)
def test_invalid_state_uri_fails_before_external_calls(machine, uri):
    run, _, events, _ = machine
    assert run(STATE_URI=uri).returncode != 0
    assert not any(e[0] in ("git", "gcloud", "task") for e in events())


def test_symlink_state_refused(machine, tmp_path):
    run, state, events, _ = machine
    state.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    state.symlink_to(outside, target_is_directory=True)
    assert run().returncode != 0
    assert not list(outside.iterdir())
    assert not any(e[0] in ("git", "gcloud", "task") for e in events())


def test_all_short_tasks_still_publish_runner_csv(machine):
    run, state, _, _ = machine
    assert run(SHORT_TASK="alpha", SHORT_BUILD="beta/new").returncode == 0
    status = json.loads((state / "status.json").read_text())
    assert status["completed"] == []
    assert status["failed"] == {"alpha": HISTORY_ERROR, "beta/new": HISTORY_ERROR}
    assert (state / "summaries.csv").read_text() == "task,model_name,MASE,quantile_levels\n"
    assert run().returncode == 0  # Matching completion includes accepted history failures.


@pytest.mark.parametrize("model", ["auto_ets", "auto_theta", "median3"])
def test_model_and_subset_selection_reach_runner(machine, model):
    run, state, events, cloud = machine
    (cloud / "names.txt").write_bytes(b"beta/new\n")
    result = run(
        {
            "model": model,
            "names_uri": "gs://offline/names.txt",
            "names_sha256": hashlib.sha256(b"beta/new\n").hexdigest(),
        },
        BUILD_FAIL="alpha",
    )
    assert result.returncode == 0, result.stderr
    assert not any(e[:2] == ["task", "alpha"] for e in events())
    assert list(csv.DictReader((state / "summaries.csv").open()))[0]["model_name"] == model


def test_pin_mismatch_fails_before_tasks(machine):
    run, state, events, cloud = machine
    metadata = cloud.parent / "packages/fev-0.10.0.dist-info/METADATA"
    metadata.write_text(metadata.read_text().replace("Version: 0.10.0", "Version: 0.10.1"))
    result = run()
    assert result.returncode != 0 and "versions do not match pins" in result.stderr
    assert not (state / "completed.json").exists()
    assert not any(e[0] == "task" for e in events())


def test_corrupt_completed_summaries_are_never_accepted_as_resume_rows(machine):
    run, state, events, _ = machine
    assert run().returncode == 0
    summary = state / "summaries.csv"
    summary.write_text(summary.read_text().replace("0.75", "999"))
    before = len(events())
    for _ in range(2):
        result = run()
        assert result.returncode != 0 and "summaries sha256 mismatch" in result.stderr
    assert not any(e[0] in ("git", "gcloud", "task") for e in events()[before:])


@pytest.fixture
def guest(jobs):
    namespace = {"__name__": "baseline_guest_test"}
    exec(compile(jobs.GUEST, "baseline-guest.py", "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "definitions",
    [
        "tasks: []",
        "tasks: {}",
        "tasks: [null]",
        "tasks: [{name: x}]",
        "tasks: [{name: ../x, dataset_path: fake/data}]",
        "tasks: [{name: /absolute, dataset_path: fake/data}]",
        "tasks: [{name: 'a,b', dataset_path: fake/data}]",
        "tasks: [{name: x, task_name: y, dataset_path: fake/data}]",
        "tasks: [{name: x, dataset_path: fake/data}, {name: x, dataset_path: fake/data}]",
    ],
)
def test_task_yaml_validation(guest, tmp_path, definitions):
    pytest.importorskip("yaml")
    (tmp_path / "tasks.yaml").write_text(definitions)
    with pytest.raises(ValueError):
        guest["selected"]({"names_uri": None}, tmp_path)


def test_native_tasks_are_lazy_and_keep_shifted_windows(guest, tmp_path):
    pytest.importorskip("yaml")
    (tmp_path / "tasks.yaml").write_text(
        "tasks:\n- dataset_path: autogluon/fev_datasets\n"
        "  dataset_config: ecl\n  initial_cutoff: -168\n  window_step_size: 24\n"
        "  num_windows: 3\n  horizon: 48\n  target: [first, second]\n"
        "  quantile_levels: [0.1, 0.5, 0.9]\n  eval_metric: SQL\n"
        "  extra_metrics: [{name: WAPE, epsilon: 1.0}]\n"
    )
    spec = guest["selected"]({"names_uri": None}, tmp_path)[0]
    assert spec.name == "ecl"
    assert spec.values["initial_cutoff"] == -168
    assert spec.values["window_step_size"] == 24
    assert spec.values["num_windows"] == 3
    assert spec.values["target"] == ["first", "second"]
    assert spec.values["extra_metrics"] == [{"name": "WAPE", "epsilon": 1.0}]


def test_real_runner_uses_native_tasks_without_profiles_and_keeps_csv(guest, tmp_path):
    pytest.importorskip("fev")
    pytest.importorskip("statsforecast")
    from demand_forecast.bench.run import run_baseline
    from tests.helpers.fev_fixture import write_fev_fixture

    parquet = write_fev_fixture(tmp_path / "fixture")
    (tmp_path / "tasks.yaml").write_text(
        f"tasks:\n- name: custom/shifted\n  dataset_path: {parquet}\n"
        "  horizon: 4\n  num_windows: 2\n  initial_cutoff: -12\n"
        "  seasonality: 2\n  eval_metric: SQL\n  quantile_levels: [0.1, 0.5, 0.9]\n"
    )
    specs = guest["selected"]({"names_uri": None}, tmp_path)
    run_baseline(specs, tmp_path / "out")
    saved = (tmp_path / "out/summaries.csv").read_bytes()
    run_baseline(specs, tmp_path / "out")
    assert saved == (tmp_path / "out/summaries.csv").read_bytes()
    row = list(csv.DictReader((tmp_path / "out/summaries.csv").open()))[0]
    assert row["task"] == "custom/shifted" and row["model_name"] == "seasonal_naive"
    assert float(row["num_forecasts"]) > 0
