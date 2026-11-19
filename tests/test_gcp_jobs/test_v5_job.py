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
    # An empty array under `set -u` is an error before bash 4.4 (the macOS system shell).
    guarded = '${extra[@]+"${extra[@]}"}'
    assert guarded in first and '"${extra[@]}"' not in first.replace(guarded, "")
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
    ("action", "digest"),
    [
        ("train", "faf53dd3a54b6742189737131726140c91892e8a623d87ea6ae9f045666f95e2"),
        ("qualify", "3bdc873d40a45fb9d3a92a077022aeb6ce24ff4d2ce2a5590783a47ebf8bf215"),
        ("t1", "1f2190455b0f939ffd00116e6bcae0fccf2295e16259b0a228ce951b03ba1571"),
    ],
)
def test_v5_job_byte_identical_to_ec6c5c4(jobs, tmp_path, action, digest):
    config = tmp_path / "fixed.yaml"
    config.write_bytes(b"sources: []\n")
    options = {
        "train": dict(config=str(config), leg="leg"),
        "qualify": dict(config=str(config)),
        "t1": dict(base=str(config), path="static_category", arm="true", seed=1),
    }
    script = jobs.generate(name="unit", action=action, repo_sha=SHA, **options[action])
    assert hashlib.sha256(script.encode()).hexdigest() == digest


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
                if a[2] == 'export':
                    assert pathlib.Path(a[a.index('--checkpoint-file') + 1]).read_bytes() == b'initial'
                else:
                    config_flag = '--base' if '--base' in a else '--config'
                    assert pathlib.Path(a[a.index(config_flag) + 1]).exists()
                assert a[a.index('--code') + 1] == 'a' * 40
                assert a[a.index('--env-sha256') + 1] == hashlib.sha256(b'alpha==1\\nzeta==2\\n').hexdigest()
                if os.environ.get('EXPECT_UPSTREAM'):
                    import simons, subprocess
                    source = pathlib.Path(sys.argv[0]).parents[2] / 'upstream/src'
                    assert str(source) in os.environ['PYTHONPATH'].split(os.pathsep)
                    assert pathlib.Path(simons.__file__).is_relative_to(source)
                    subprocess.run([sys.executable, '-c', 'import simons'], check=True)
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
                pin = os.environ.get('REPO_UPSTREAM_PIN', {upstream_pin!r})
                if pin:
                    target = p / 'third_party/simons/PINNED'
                    target.parent.mkdir(parents=True)
                    target.write_text(pin + '\\n')
            if a[0] == 'clone':
                p = pathlib.Path(a[-1]) / 'src/simons'; p.mkdir(parents=True)
                (p / '__init__.py').write_text('')
            if 'rev-parse' in a:
                if pathlib.Path(a[1]).name == 'upstream':
                    print(os.environ.get('UPSTREAM_SHA', {upstream_pin!r}))
                else:
                    print(os.environ.get('FETCHED_SHA', 'a' * 40))
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
    programs["git"] = programs["git"].replace(
        "{upstream_pin!r}", repr((ROOT / "third_party/simons/PINNED").read_text().strip())
    )
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


def snapshot(directory):
    return {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_t1_extension_preserves_old_unit_and_training(jobs, settings, machine):
    run, execute = machine
    options = dict(
        action="t1", repo_sha=SHA, base=str(settings), path="static_category", arm="true", seed=7
    )
    first = jobs.generate(name="first", **options, episodes=200, max_panels=1000)
    result, events = execute(first)
    assert result.returncode == 0, result.stderr
    root = run / "state/t1/static_category-true-s7"
    training = root / "train"
    unit = root / "eval-200-1000"
    for directory in (training, unit):
        directory.mkdir(parents=True)
        (directory / "checkpoint-final.pt").write_bytes(b"finished checkpoint")
        write_manifest(directory / "train.json", {"files": ["checkpoint-final.pt"]})
    (unit / "result.json").write_text('{"loss": [1.0]}')
    write_manifest(
        unit / "unit.json",
        {
            "files": ["checkpoint-final.pt", "train.json", "result.json"],
            "job_sha256": hashlib.sha256(first.encode()).hexdigest(),
        },
    )
    old_training, old_unit = snapshot(training), snapshot(unit)
    second = jobs.generate(name="extended", **options, episodes=400, max_panels=2000)
    previous_events = len(events)
    result, events = execute(second)
    assert result.returncode == 0, result.stderr
    extension = events[previous_events:]
    assert any(e[0] == "git" for e in extension)
    assert any(e[0] == "uv" for e in extension)
    preflight = next(e for e in extension if "preflight" in e)
    assert preflight[preflight.index("preflight") + 1] == str(root / "eval-400-2000")
    final = extension[-1]
    assert final[:4] == ["python", "-m", "demand_forecast.pretrain.v5.job", "t1"]
    from demand_forecast.pretrain.v5 import job

    parsed = job._parser().parse_args(final[3:])
    assert (parsed.path, parsed.arm, parsed.seed) == ("static_category", "true", 7)
    assert f"eval-{parsed.episodes}-{parsed.max_panels}" == "eval-400-2000"
    assert snapshot(training) == old_training
    assert snapshot(unit) == old_unit
    previous_events = len(events)
    result, events = execute(first)
    assert result.returncode == 0, result.stderr
    assert not any(e[0] in ("git", "uv") for e in events[previous_events:])
    assert not any("demand_forecast.pretrain.v5.job" in e for e in events[previous_events:])
    assert snapshot(training) == old_training
    assert snapshot(unit) == old_unit


@pytest.mark.parametrize("defaults", [None, (3, 17)])
def test_generator_and_cli_share_default_unit(jobs, settings, machine, monkeypatch, defaults):
    from demand_forecast.pretrain.v5 import job

    if defaults is not None:
        monkeypatch.setattr(job, "T1_EVAL_DEFAULTS", defaults, raising=False)
        monkeypatch.setattr(jobs, "T1_EVAL_DEFAULTS", defaults, raising=False)
    expected = "eval-200-1000" if defaults is None else "eval-3-17"
    run, execute = machine
    result, events = execute(
        jobs.generate(
            name="defaults",
            action="t1",
            repo_sha=SHA,
            base=str(settings),
            path="static_category",
            arm="true",
            seed=7,
        )
    )
    assert result.returncode == 0, result.stderr
    root = run / "state/t1/static_category-true-s7"
    preflight = next(e for e in events if "preflight" in e)
    assert preflight[preflight.index("preflight") + 1] == str(root / expected)
    parsed = job._parser().parse_args(events[-1][3:])
    assert f"eval-{parsed.episodes}-{parsed.max_panels}" == expected
    visited = []

    def unfinished(args, unit, training):
        visited.append(unit)
        return 75

    monkeypatch.setattr(job, "_run", unfinished)
    assert (
        job.main(
            [
                "t1",
                "--run-dir",
                str(run),
                "--code",
                SHA,
                "--job-sha256",
                "b" * 64,
                "--env-sha256",
                "c" * 64,
                "--base",
                str(settings),
                "--path",
                "static_category",
                "--arm",
                "true",
                "--seed",
                "7",
            ]
        )
        == 75
    )
    assert visited == [root / expected]
    assert (root / expected).is_dir()


def test_generator_t1_defaults_match_job(jobs):
    from demand_forecast.pretrain.v5 import job

    assert jobs.T1_EVAL_DEFAULTS == job.T1_EVAL_DEFAULTS


@pytest.mark.parametrize("action", ["help", "t1"])
def test_generator_runs_without_site_packages_or_project_imports(jobs, settings, tmp_path, action):
    output = tmp_path / "isolated.sh"
    arguments = ["--help"]
    if action == "t1":
        arguments = [
            "--name",
            "isolated",
            "--action",
            "t1",
            "--repo-sha",
            SHA,
            "--base",
            str(settings),
            "--path",
            "static_category",
            "--arm",
            "true",
            "--seed",
            "7",
            "--output",
            str(output),
        ]
    isolated = textwrap.dedent("""
        import runpy
        import sys

        blocked = []
        class BlockProjectImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'demand_forecast', 'torch', 'numpy'}:
                    blocked.append(fullname)
                    raise ModuleNotFoundError(fullname, name=fullname)

        sys.meta_path.insert(0, BlockProjectImports())
        try:
            import demand_forecast
        except ModuleNotFoundError as error:
            assert error.name == 'demand_forecast'
        else:
            raise AssertionError('project must not be importable')
        blocked.clear()
        sys.argv = sys.argv[1:]
        try:
            runpy.run_path(sys.argv[0], run_name='__main__')
        finally:
            assert not blocked, f'generator attempted forbidden imports: {blocked}'
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", isolated, jobs.__file__, *arguments],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    if action == "help":
        assert "usage:" in result.stdout
    else:
        assert os.access(output, os.X_OK)
        script = output.read_text()
        assert "state/t1/static_category-true-s7/eval-200-1000" in script
        assert base64.b64encode(settings.read_bytes()).decode() in script


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


def export_arguments(**changes):
    return {
        "name": "model",
        "action": "export",
        "repo_sha": SHA,
        "member_uri": "gs://offline/checkpoint-final.pt",
        "member_sha256": hashlib.sha256(b"initial").hexdigest(),
        **changes,
    }


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"config": "unused.yaml"}, "config"),
        ({"corpus_uri": "gs://offline/corpus", "corpus_digest": DIGEST}, "corpus"),
        ({"upstream": True}, "upstream"),
        ({"init_uri": "gs://offline/initial.pt", "init_sha256": "a" * 64}, "init"),
        ({"member_sha256": None}, "member_sha256"),
        ({"member_sha256": "invalid"}, "member_sha256"),
        ({"member_uri": None}, "member_uri"),
        ({"member_uri": "https://offline/checkpoint.pt"}, "literal gs://"),
        ({"member_uri": "gs://offline/../checkpoint.pt"}, "normalized gs://"),
    ],
)
def test_export_generator_rejects_invalid_arguments(jobs, options, message):
    with pytest.raises(ValueError, match=message):
        jobs.generate(**export_arguments(**options))


@pytest.mark.parametrize("matches", [True, False])
def test_export_cloud_job_checks_member_before_environment(jobs, machine, tmp_path, matches):
    _, execute = machine
    options = export_arguments()
    if not matches:
        options["member_sha256"] = "0" * 64
    output = tmp_path / "export.sh"
    command = [sys.executable, jobs.__file__, "--output", str(output)]
    for key, value in options.items():
        command.extend(("--" + key.replace("_", "-"), value))
    generated = subprocess.run(command, capture_output=True, text=True)
    assert generated.returncode == 0, generated.stderr
    assert subprocess.run(["bash", "-n", str(output)]).returncode == 0

    result, events = execute(output.read_text())
    fetch = next(i for i, event in enumerate(events) if event[0] == "gcloud")
    assert events[fetch][:4] == ["gcloud", "storage", "cp", options["member_uri"]]
    check = next(i for i, event in enumerate(events) if event[2:] == ["hash", events[fetch][-1]])
    assert fetch < check
    if matches:
        assert result.returncode == 0, result.stderr
        assert check < next(i for i, event in enumerate(events) if event[0] == "uv")
        final = events[-1]
        assert final[:4] == ["python", "-m", "demand_forecast.pretrain.v5.job", "export"]
        assert final[final.index("--name") + 1] == "model"
        assert final[final.index("--checkpoint-file") + 1] == events[fetch][-1]
        assert final[final.index("--checkpoint-sha256") + 1] == options["member_sha256"]
        assert "--config" not in final and "--base" not in final
    else:
        assert result.returncode == 2, result.stderr
        assert "checkpoint SHA-256 mismatch" in result.stderr
        assert not any(event[0] == "uv" for event in events)
        assert not any(event[1:3] == ["-m", "pip"] for event in events)
        assert not any("demand_forecast.pretrain.v5.job" in event for event in events)


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


@pytest.mark.parametrize("fault", ["none", "head", "repo_pin", "missing_repo_pin"])
def test_upstream_checkout_and_pin_guards(jobs, settings, machine, fault):
    replace(config(), sources=(SourceConfig("upstream", 1),)).to_yaml(settings)
    _, execute = machine
    script = jobs.generate(**arguments(settings), upstream=True)
    assert script == jobs.generate(**arguments(settings), upstream=True)
    assert "shutdown" not in script
    changes = {
        "none": {},
        "head": {"UPSTREAM_SHA": "e" * 40},
        "repo_pin": {"REPO_UPSTREAM_PIN": "e" * 40},
        "missing_repo_pin": {"REPO_UPSTREAM_PIN": ""},
    }[fault]
    result, events = execute(script, EXPECT_UPSTREAM="1", **changes)
    assert result.returncode == (0 if fault == "none" else 2), result.stderr
    pin = (ROOT / "third_party/simons/PINNED").read_text().strip()
    clones = [e for e in events if e[:2] == ["git", "clone"]]
    if fault in ("repo_pin", "missing_repo_pin"):
        assert not clones  # The cheaper guard runs first.
    else:
        (clone,) = clones
        assert clone[2:5] == [
            "--filter=blob:none",
            "--no-checkout",
            "https://github.com/Abel-ai-lab/Simons.git",
        ]
        assert Path(clone[-1]).parent.name.startswith("install-")
        checkout = next(e for e in events if e[0] == "git" and pin in e)
        assert "checkout" in checkout and "--detach" in checkout
    if fault == "none":
        assert events[-1][:3] == ["python", "-m", "demand_forecast.pretrain.v5.job"]
    else:
        assert "mismatch" in result.stderr.lower()
        assert not any("demand_forecast.pretrain.v5.job" in e for e in events)
        # Every guard runs before anything is installed.
        assert not any(e[0] == "uv" or e[:3] == ["python", "-m", "pip"] for e in events)


def test_upstream_keeps_dependency_events_and_fingerprint_unchanged(jobs, settings, machine):
    _, execute = machine
    plain = jobs.generate(**arguments(settings))
    plain_result, plain_events = execute(plain)
    assert plain_result.returncode == 0, plain_result.stderr
    replace(config(), sources=(SourceConfig("upstream", 1),)).to_yaml(settings)
    script = jobs.generate(**arguments(settings), upstream=True)
    result, all_events = execute(script)
    assert result.returncode == 0, result.stderr
    upstream_events = all_events[len(plain_events) :]

    def dependency_events(events):
        scratch = str(Path(next(e[-1] for e in events if e[0] == "uv")).parent)
        return [
            [part.replace(scratch, "$scratch") for part in event]
            for event in events
            if event[0] == "uv" or event[:3] == ["python", "-m", "pip"]
        ]

    assert dependency_events(upstream_events) == dependency_events(plain_events)
    for events in (plain_events, upstream_events):
        job = events[-1]
        assert (
            job[job.index("--env-sha256") + 1] == hashlib.sha256(b"alpha==1\nzeta==2\n").hexdigest()
        )
    assert "prepare.py" not in script
    assert not any("$upstream" in line for line in script.splitlines() if "pip install" in line)
    # The checkout becomes importable only after the fingerprint has been taken.
    lines = script.splitlines()
    fingerprint = next(i for i, line in enumerate(lines) if line.startswith("env_sha256="))
    exported = [i for i, line in enumerate(lines) if line.startswith("export PYTHONPATH=")]
    assert len(exported) == 1 and exported[0] > fingerprint
    assert "export PYTHONPATH=" not in plain


def test_upstream_is_opt_in(jobs, settings, machine):
    _, execute = machine
    result, events = execute(jobs.generate(**arguments(settings)))
    assert result.returncode == 0, result.stderr
    assert not any(any("/upstream" in part or "prepare.py" in part for part in e) for e in events)


@pytest.mark.parametrize("action", ["train", "qualify"])
@pytest.mark.parametrize("problem", ["root", "missing", "unexpected", "mapping"])
def test_generator_preserves_corpus_configuration_guards(jobs, settings, action, problem):
    if problem == "root":
        replace(config(), sources=(SourceConfig("corpus", 1, root="/local"),)).to_yaml(settings)
    elif problem == "missing":
        replace(config(), sources=(SourceConfig("corpus", 1),)).to_yaml(settings)
    elif problem == "mapping":
        settings.write_text("[]")
    options = (
        {}
        if problem != "unexpected"
        else {"corpus_uri": "gs://offline/corpus", "corpus_digest": DIGEST}
    )
    values = dict(name="guard", action=action, repo_sha=SHA, config=str(settings), **options)
    message = {
        "root": "root: null",
        "missing": "corpus sources and corpus_uri",
        "unexpected": "corpus sources and corpus_uri",
        "mapping": "requires a mapping",
    }[problem]
    with pytest.raises(ValueError, match=message):
        jobs.generate(**values)


@pytest.mark.parametrize("action", ["train", "qualify"])
@pytest.mark.parametrize("has_source,flag", [(True, False), (False, True)])
def test_generator_requires_upstream_source_and_flag_together(
    jobs, settings, action, has_source, flag
):
    if has_source:
        replace(config(), sources=(SourceConfig("upstream", 1),)).to_yaml(settings)
    with pytest.raises(ValueError, match="upstream sources and --upstream"):
        jobs.generate(name="pair", action=action, repo_sha=SHA, config=str(settings), upstream=flag)


def test_t1_rejects_upstream(jobs, settings):
    with pytest.raises(ValueError, match="--upstream.*t1"):
        jobs.generate(
            name="t1",
            action="t1",
            repo_sha=SHA,
            base=str(settings),
            path="selection",
            arm="true",
            seed=1,
            upstream=True,
        )
