"""Run the generated guest offline with strict fake external commands."""

import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from demand_forecast.pretrain.v5.checkpoint import write_manifest
from demand_forecast.pretrain.v5.config import SourceConfig
from demand_forecast.pretrain.v5.corpus import listing_digest
from tests.helpers.v5_loop import config

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 40
OBJECTS = [
    ("univariate/demo/train.parquet", 4, "abcd"),
    ("covariates/demo/train.parquet", 7, "efgh"),
]
DIGEST = listing_digest(OBJECTS)


@pytest.fixture
def jobs():
    path = ROOT / "scripts/gcp/jobs/make_v5_job.py"
    assert path.exists(), "v5 generator is missing"
    spec = importlib.util.spec_from_file_location("make_v5_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / "base.yaml"
    config(steps=1).to_yaml(path)
    return path


def arguments(settings, **changes):
    return dict(
        name="unit", action="train", repo_sha=SHA, config=str(settings), leg="leg", **changes
    )


def test_bash_determinism_embedded_configuration_and_digest(jobs, settings, tmp_path):
    first = jobs.generate(**arguments(settings))
    assert first == jobs.generate(**arguments(settings))
    script = tmp_path / "job.sh"
    script.write_text(first)
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    assert "shutdown" not in first
    assert "shutdown" not in (ROOT / "scripts/gcp/jobs/make_v5_job.py").read_text()
    assert all("--exclude" in line for line in first.splitlines() if "quarantine" in line)
    assert base64.b64encode(settings.read_bytes()).decode() in first
    assert hashlib.sha256(settings.read_bytes()).hexdigest() in first
    helper = tmp_path / "helper.py"
    helper.write_text(jobs.GUEST)
    listing = tmp_path / "listing.json"
    listing.write_text(
        json.dumps(
            [
                dict(name="gs://offline/corpus/" + p, size=s, crc32c=c, generation="1")
                for p, s, c in OBJECTS
            ]
        )
    )
    result = subprocess.run(
        [sys.executable, str(helper), "listing", str(listing), "gs://offline/corpus", DIGEST],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["objects"] == [list(row) for row in sorted(OBJECTS)]
    options = dict(
        name="t1",
        action="t1",
        repo_sha=SHA,
        base=str(settings),
        path="static_category",
        arm="true",
        seed=1,
    )
    assert jobs.generate(**options, episodes=2) != jobs.generate(**options, episodes=3)


@pytest.mark.parametrize(
    "changes",
    [
        {"repo_sha": "main"},
        {"name": "../x"},
        {"corpus_uri": "gs://offline/corpus"},
        {"init_uri": "gs://offline/init"},
    ],
)
def test_invalid_inputs(jobs, settings, changes):
    values = arguments(settings)
    values.update(changes)
    with pytest.raises(ValueError):
        jobs.generate(**values)


@pytest.fixture
def machine(tmp_path):
    bins = tmp_path / "bin"
    bins.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    env = {
        **os.environ,
        "PATH": str(bins) + os.pathsep + os.environ["PATH"],
        "RUN_DIR": str(run),
        "STATE_URI": "gs://offline/state",
        "EVENTS": str(tmp_path / "events"),
        "FAKE_PYTHON": str(bins / "python3"),
        "LISTING": json.dumps(
            [
                dict(name="gs://offline/corpus/" + p, size=s, crc32c=c, generation="1")
                for p, s, c in OBJECTS
            ]
        ),
    }
    programs = {
        "python3": """
            import hashlib, json, os, pathlib, shutil, sys
            a = sys.argv[1:]
            with open(os.environ['EVENTS'], 'a') as f: f.write(json.dumps(['python', *a]) + '\\n')
            if a[:3] == ['-m', 'pip', 'install']:
                if 'uv' in a:
                    dest = pathlib.Path(a[a.index('--target') + 1]) / 'bin'
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(pathlib.Path(os.environ['FAKE_PYTHON']).with_name('uv'), dest / 'uv')
                sys.exit(0)
            if a[:3] == ['-m', 'pip', 'check']:
                assert pathlib.Path(sys.argv[0]).with_name('seeded').exists(), 'pip missing'
                sys.exit(0)
            if a[:3] == ['-m', 'pip', 'freeze']:
                assert '--exclude-editable' in a
                print('zeta==2\\nalpha==1')
                sys.exit(0)
            if a[:2] == ['-m', 'demand_forecast.pretrain.v5.job']:
                config_flag = '--base' if '--base' in a else '--config'
                assert pathlib.Path(a[a.index(config_flag) + 1]).exists()
                assert a[a.index('--code') + 1] == 'a' * 40
                assert a[a.index('--env-sha256') + 1] == hashlib.sha256(b'alpha==1\\nzeta==2\\n').hexdigest()
                sys.exit(int(os.environ.get('JOB_EXIT', '0')))
            os.execv(sys.executable, [sys.executable, *a])
        """,
        "git": """
            import json, os, pathlib, sys
            a = sys.argv[1:]
            with open(os.environ['EVENTS'], 'a') as f: f.write(json.dumps(['git', *a]) + '\\n')
            if a[0] == 'init':
                p = pathlib.Path(a[-1]); p.mkdir(parents=True, exist_ok=True)
                (p / 'requirements-v5.txt').write_text('alpha==1')
            if 'rev-parse' in a: print(os.environ.get('FETCHED_SHA', 'a' * 40))
        """,
        "uv": """
            import json, os, pathlib, sys
            a = sys.argv[1:]
            with open(os.environ['EVENTS'], 'a') as f: f.write(json.dumps(['uv', *a]) + '\\n')
            assert 'venv' in a and '--python' in a and '3.12' in a
            p = pathlib.Path(a[-1]) / 'bin'; p.mkdir(parents=True, exist_ok=True)
            (p / 'python').symlink_to(os.environ['FAKE_PYTHON'])
            if '--seed' in a: (p / 'seeded').touch()
        """,
        "gcloud": """
            import json, os, pathlib, sys
            a = sys.argv[1:]
            with open(os.environ['EVENTS'], 'a') as f: f.write(json.dumps(['gcloud', *a]) + '\\n')
            assert a[0] == 'storage'
            if a[1] == 'ls':
                assert '--json' in a, 'storage ls does not support --format=json'
                counter = pathlib.Path(os.environ['RUN_DIR']) / 'v5-cache/list-count'
                n = int(counter.read_text()) if counter.exists() else 0
                counter.parent.mkdir(parents=True, exist_ok=True); counter.write_text(str(n + 1))
                rows = json.loads(os.environ['LISTING'])
                if n and os.environ.get('CHANGE_GENERATION'): rows[0]['generation'] = '2'
                # gcloud prints a separate JSON array for every URL; object
                # fields are nested under metadata in the real SDK.
                for kind in ('univariate', 'covariates'):
                    print(json.dumps([{'url': r['name'], 'type': 'cloud_object', 'metadata': {**r, 'name': r['name'].removeprefix('gs://offline/')}} for r in rows if '/' + kind + '/' in r['name']]))
            elif a[1] == 'rsync':
                assert '--recursive' in a
                assert not (pathlib.Path(os.environ['RUN_DIR']) / 'corpus/stale.parquet').exists()
                pathlib.Path(a[-1]).mkdir(parents=True, exist_ok=True)
            elif a[1] == 'cp':
                pathlib.Path(a[-1]).write_bytes(b'initial')
            else: sys.exit(91)
        """,
    }
    for name, source in programs.items():
        path = bins / name
        path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(source))
        path.chmod(0o755)

    def execute(script, **changes):
        path = tmp_path / "job.sh"
        path.write_text(script)
        result = subprocess.run(
            ["bash", str(path)], env={**env, **changes}, capture_output=True, text=True, timeout=20
        )
        events = [json.loads(line) for line in Path(env["EVENTS"]).read_text().splitlines()]
        return result, events

    return run, execute


def test_first_run_executes_seeded_environment_and_propagates_exit(jobs, settings, machine):
    _, execute = machine
    result, events = execute(jobs.generate(**arguments(settings)), JOB_EXIT="75")
    assert result.returncode == 75, result.stderr
    final = events[-1]
    assert final[:3] == ["python", "-m", "demand_forecast.pretrain.v5.job"]
    assert final[3] == "train"
    assert final[final.index("--name") + 1] == "leg"
    assert Path(final[final.index("--config") + 1]).read_bytes() == settings.read_bytes()
    assert next(i for i, e in enumerate(events) if e[0] == "git") < next(
        i for i, e in enumerate(events) if e[0] == "uv"
    )


@pytest.mark.parametrize("fault", ["digest", "generation", "none"])
def test_corpus_listing_guard_and_stale_cache(jobs, settings, machine, fault):
    replace(config(steps=1), sources=(SourceConfig("corpus", 1),)).to_yaml(settings)
    run, execute = machine
    (run / "corpus").mkdir()
    (run / "corpus/stale.parquet").touch()
    script = jobs.generate(
        **arguments(
            settings,
            corpus_uri="gs://offline/corpus",
            corpus_digest="f" * 64 if fault == "digest" else DIGEST,
        )
    )
    result, events = execute(
        script, **({"CHANGE_GENERATION": "1"} if fault == "generation" else {})
    )
    assert result.returncode == (0 if fault == "none" else 2), result.stderr
    copies = [e for e in events if e[:3] == ["gcloud", "storage", "rsync"]]
    assert len(copies) == (0 if fault == "digest" else 2)
    if fault == "none":
        assert not (run / "corpus/stale.parquet").exists()
        assert json.loads((run / "corpus/manifest.json").read_text())["objects"] == [
            list(row) for row in sorted(OBJECTS)
        ]
        assert "--corpus-root" in events[-1]


@pytest.mark.parametrize("same", [True, False])
def test_valid_unit_skips_only_the_same_job(jobs, settings, machine, same):
    run, execute = machine
    script = jobs.generate(**arguments(settings))
    unit = run / "state/train/leg"
    unit.mkdir(parents=True)
    (unit / "checkpoint-final.pt").write_bytes(b"finished")
    (unit / "train.json").write_text("{}")
    write_manifest(
        unit / "unit.json",
        {
            "files": ["checkpoint-final.pt", "train.json"],
            "job_sha256": hashlib.sha256(script.encode()).hexdigest() if same else "d" * 64,
        },
    )
    result, events = execute(script)
    assert result.returncode == 0, result.stderr
    assert any(e[0] == "git" for e in events) == (not same)
    assert any(e[0] == "uv" for e in events) == (not same)


def test_sha_mismatch_exits_before_environment(jobs, settings, machine):
    _, execute = machine
    result, events = execute(jobs.generate(**arguments(settings)), FETCHED_SHA="e" * 40)
    assert result.returncode == 2
    assert not any(e[0] == "uv" for e in events)


@pytest.mark.parametrize("action", ["qualify", "t1"])
def test_cloud_actions_and_initial_checkpoint(jobs, settings, machine, action):
    _, execute = machine
    values = dict(name="probe", action=action, repo_sha=SHA)
    if action == "t1":
        values.update(
            base=str(settings), path="selection", arm="true", seed=3, episodes=4, max_panels=10
        )
    else:
        values.update(config=str(settings))
    result, events = execute(jobs.generate(**values))
    assert result.returncode == 0, result.stderr
    assert events[-1][3] == action
    result, events = execute(
        jobs.generate(
            **arguments(
                settings,
                init_uri="gs://offline/init.pt",
                init_sha256=hashlib.sha256(b"initial").hexdigest(),
            )
        )
    )
    assert result.returncode == 0, result.stderr
    assert "--init-file" in events[-1]


def test_generator_cli(jobs, settings, tmp_path):
    output = tmp_path / "cli.sh"
    result = subprocess.run(
        [
            sys.executable,
            jobs.__file__,
            "--name",
            "unit",
            "--action",
            "train",
            "--repo-sha",
            SHA,
            "--config",
            str(settings),
            "--leg",
            "leg",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert os.access(output, os.X_OK)
