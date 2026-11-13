"""Execute the complete generated adaptation job with offline guest commands."""

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
PIN = (ROOT / "third_party/simons/PINNED").read_text().strip()
TASKS = b"tasks:\n- name: alpha\n- name: beta/new\n"


@pytest.fixture
def jobs():
    spec = importlib.util.spec_from_file_location(
        "make_adapt_job", ROOT / "scripts/gcp/jobs/make_adapt_job.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(**overrides):
    return dict(
        name="pilot",
        repo_sha="a" * 40,
        upstream_pin=PIN,
        tasks_uri="gs://offline/tasks.yaml",
        tasks_sha256=hashlib.sha256(TASKS).hexdigest(),
        names=["alpha", "beta/new"],
        checkpoint_uri="gs://offline/base",
        params=["head", "all"],
        lr_multipliers=[0.02],
        seeds=[11, 29],
        max_updates=50,
        **overrides,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("name", "../pilot"),
        ("repo_sha", "main"),
        ("upstream_pin", "b" * 40),
        ("tasks_uri", "gs://offline/a*"),
        ("tasks_sha256", "bad"),
        ("checkpoint_uri", None),
        ("hub_weights", "abel/model"),
        ("hub_revision", "b" * 40),
        ("processing", "other"),
        ("names", []),
        ("names", ["alpha", "alpha"]),
        ("names", ["../alpha"]),
        ("names", ["alpha,beta"]),
        ("names", ["@file"]),
        ("params", ["body"]),
        ("params", []),
        ("lr_multipliers", [float("nan")]),
        ("lr_multipliers", [0]),
        ("lr_multipliers", [0.02, 0.02]),
        ("seeds", [-1]),
        ("seeds", [True]),
        ("max_updates", -1),
        ("max_updates", True),
    ],
)
def test_validation(jobs, key, value):
    options = arguments()
    options[key] = value
    with pytest.raises(ValueError):
        jobs.generate(**options)


@pytest.mark.parametrize("mode", ["raw", "published", "both"])
def test_shell_and_cli(jobs, tmp_path, mode):
    options = arguments()
    options["processing"] = mode
    path = tmp_path / "job.sh"
    path.write_text(jobs.generate(**options))
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/gcp/jobs/make_adapt_job.py"),
            "--name",
            "pilot",
            "--repo-sha",
            "a" * 40,
            "--upstream-pin",
            PIN,
            "--tasks-uri",
            "gs://offline/tasks.yaml",
            "--tasks-sha256",
            options["tasks_sha256"],
            "--names",
            "alpha",
            "beta/new",
            "--checkpoint-uri",
            "gs://offline/base",
            "--params",
            "head",
            "all",
            "--lr-multipliers",
            "0.02",
            "0.1",
            "--seeds",
            "11",
            "29",
            "--max-updates",
            "50",
            "--processing",
            mode,
            "--output",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert path.stat().st_mode & 0o111


@pytest.fixture
def machine(jobs, tmp_path):
    bins, cloud, packages = (tmp_path / n for n in ("bin", "cloud", "packages"))
    for path in (bins, cloud / "base", packages):
        path.mkdir(parents=True)
    (cloud / "tasks.yaml").write_bytes(TASKS)
    (cloud / "base/config.json").write_text('{"width":24}')
    (cloud / "base/model.safetensors").write_bytes(b"base weights")
    (packages / "torch.py").write_text(
        "import os, types\n__version__='2.8.0+cu128'\n"
        "version=types.SimpleNamespace(cuda='12.8')\n"
        "cuda=types.SimpleNamespace(is_available=lambda: not os.environ.get('NO_CUDA'))\n"
    )
    for name, version in (("fev", "0.10.0"), ("datasets", "3.6.0"), ("huggingface_hub", "0.36.0")):
        info = packages / (name + "-" + version + ".dist-info")
        info.mkdir()
        (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    fake = tmp_path / "fake.py"
    fake.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(r"""
        import csv, hashlib, json, os, pathlib, shutil, sys, time
        P = pathlib.Path
        args = sys.argv[1:]
        tool = P(sys.argv[0]).name
        cloud, packages = P(os.environ['CLOUD']), P(os.environ['PACKAGES'])
        def event(kind, *values):
            with open(os.environ['EVENTS'], 'a') as f: f.write(json.dumps([kind, *values]) + '\n')
        def val(flag): return args[args.index(flag) + 1]
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
        def remote(path): return cloud / path.removeprefix('gs://offline/')
        if tool == 'git':
            event('git', *args)
            if args[0] == 'init':
                P(args[-1]).mkdir(parents=True, exist_ok=True)
                (P(args[-1]) / 'requirements-yardsticks.txt').write_text('fev==0.10.0\n')
            if args[0] == 'clone':
                tree = P(args[-1])
                (tree / 'src/simons/training').mkdir(parents=True)
                (tree / 'configs').mkdir()
                for n in ('__init__.py', 'training/__init__.py', 'training/adapt.py'):
                    (tree / 'src/simons' / n).write_text('')
                (packages / 'simons').unlink(missing_ok=True)
                (packages / 'simons').symlink_to(tree / 'src/simons', target_is_directory=True)
                info = packages / 'simons-0.1.dist-info'
                info.mkdir(exist_ok=True)
                (info / 'METADATA').write_text('Name: simons\nVersion: 0.1\n')
                write(info / 'direct_url.json', {'url': tree.as_uri(), 'dir_info': {'editable': True}})
            if 'rev-parse' in args:
                print(os.environ['PIN'] if 'Simons' in args[1] else os.environ.get('REPO_SHA', 'a' * 40))
            sys.exit(0)
        if tool == 'gcloud':
            event('gcloud', *args)
            assert args[:2] == ['storage', 'cp'], args
            src, dst = args[2:4]
            source = remote(src) if src.startswith('gs://') else P(src)
            dest = remote(dst) if dst.startswith('gs://') else P(dst)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            sys.exit(0)
        if tool == 'uv':
            event('uv', *args)
            args = ['--fake-venv', args[-1]]
        if tool == 'nvidia-smi':
            print('580.105.08')
            sys.exit(0)
        if args[:2] == ['-m', 'venv'] or args[:1] == ['--fake-venv']:
            if args[0] != '--fake-venv': sys.exit(1)
            target = P(args[-1]) / 'bin'
            target.mkdir(parents=True, exist_ok=True)
            (target / 'python').unlink(missing_ok=True)
            (target / 'python').symlink_to(os.environ['WRAPPER'])
            sys.exit(0)
        if args[:3] == ['-m', 'pip', 'install']:
            event('install', *args)
            if 'uv' in args:
                target = P(val('--target')) / 'bin'
                target.mkdir(parents=True)
                shutil.copyfile(P(os.environ['BIN']) / 'uv', target / 'uv')
                (target / 'uv').chmod(0o755)
            sys.exit(0)
        if args[:3] == ['-m', 'pip', 'freeze']:
            print('fev==0.10.0\ntorch==2.8.0+cu128')
            sys.exit(0)
        if args[:3] == ['-m', 'pip', 'check']: sys.exit(0)
        if args[:1] == ['-c']:
            print('3.10')
            sys.exit(0)
        if args[:2] == ['-m', 'demand_forecast.bench.adaptation']:
            name, out = val('--task'), P(val('--output'))
            event('bundle', name)
            out.mkdir(parents=True)
            (out / 'train.parquet').write_bytes(os.environ.get('BUNDLE_CONTENT', 'bounded history').encode())
            write(out / 'manifest.json', {'schema_version': int(os.environ.get('SCHEMA', '2')),
                'fingerprint': 'fingerprint-' + name + sha(out / 'train.parquet'), 'phase': 'development',
                'task': {'name': name}, 'files': {'train.parquet': {'bytes': (out / 'train.parquet').stat().st_size, 'sha256': sha(out / 'train.parquet')}}})
            sys.exit(0)
        if args[:2] == ['-m', 'simons.training.adapt']:
            out, bundle = P(val('--output')), P(val('--bundle'))
            seed = int(val('--seed'))
            arm = val('--params')
            resumed = (out / 'checkpoints/checkpoint-slot-0.pt').exists()
            event('fit', arm, seed, resumed, str(out), os.environ.get('RUN_DIR'))
            out.mkdir(parents=True, exist_ok=True)
            (out / 'checkpoints').mkdir(exist_ok=True)
            (out / 'checkpoints/checkpoint-slot-0.pt').write_bytes(b'trainer checkpoint')
            if os.environ.get('REQUEST_CHECKPOINT'):
                control = P(os.environ['RUN_DIR'])
                control.mkdir(parents=True, exist_ok=True)
                for _ in range(200):
                    req = control / 'checkpoint-request'
                    if req.exists():
                        (control / 'checkpoint-done').write_bytes(req.read_bytes())
                        break
                    time.sleep(0.01)
            if os.environ.get('FAIL_FIT') == arm + str(seed): sys.exit(3)
            selected = arm == 'all'
            for n in ('config.json', 'model.safetensors'):
                shutil.copyfile(P(val('--base')) / n, out / n)
            if selected: (out / 'model.safetensors').write_bytes(b'adapted weights')
            manifest = json.loads((bundle / 'manifest.json').read_text())
            write(out / 'adaptation.json', {'schema_version': 1, 'seed': seed,
                'identity': {'bundle_fingerprint': manifest['fingerprint']},
                'inputs': {'bundle_manifest_sha256': sha(bundle / 'manifest.json')},
                'decision': {'selected': 'adapted' if selected else 'frozen_base', 'selected_update': 50 if selected else 0, 'reason': 'inner_gain' if selected else 'no_qualifying_gain'},
                'parameters': {'trainable_count': 123}, 'training_wall_seconds': 2.0,
                'peak_cuda_memory_bytes': 4096,
                'evaluations': [{'update': u, 'wall_seconds': 1.0, 'windows': [{'score': s}, {'score': s + .1}]} for u, s in [(0, 1.0), (50, .8)]],
                'export': {n: sha(out / n) for n in ('config.json', 'model.safetensors')}})
            sys.exit(0)
        if args and args[0].endswith('fev_yardstick.py'):
            name, mode, out = val('--names'), val('--processing'), P(val('--out-dir'))
            event('score', name, mode, val('--checkpoint'), args)
            out.mkdir(parents=True, exist_ok=True)
            if os.environ.get('FAIL_SCORE') == mode: sys.exit(4)
            window = out / name / 'window_0.npz'
            window.parent.mkdir(parents=True, exist_ok=True)
            window.write_bytes(b'prediction array')
            score = .7 if (P(val('--checkpoint')) / 'model.safetensors').read_bytes() == b'adapted weights' else 1.0
            row = {'task': name, 'model_name': val('--model-name'), 'run_identity': 'id', 'SQL': score}
            write(out / name / 'run.json', {'run_identity': 'id', 'summary': row, 'window_sha256': {'window_0.npz': sha(window)}})
            with (out / 'summaries.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
            sys.exit(0)
        if len(args) > 1 and args[0].endswith('/adapt.py') and args[1] == 'environment':
            import runpy
            sys.version_info = (3, 12, 0, 'final', 0)
            sys.argv = args
            runpy.run_path(args[0], run_name='__main__')
            sys.exit(0)
        os.execv(sys.executable, [sys.executable, *args])
        """)
    )
    fake.chmod(0o755)
    for name in ("python3", "git", "gcloud", "uv", "nvidia-smi"):
        (bins / name).symlink_to(fake)
    env = dict(
        os.environ,
        PATH=str(bins) + os.pathsep + os.environ["PATH"],
        PYTHONPATH=str(packages),
        RUN_DIR=str(tmp_path / "run"),
        STATE_URI="gs://offline/state",
        CLOUD=str(cloud),
        PACKAGES=str(packages),
        WRAPPER=str(bins / "python3"),
        BIN=str(bins),
        PIN=PIN,
        EVENTS=str(tmp_path / "events.jsonl"),
    )
    script = tmp_path / "job.sh"

    def run(options=None, **overrides):
        config = arguments()
        config["name"] = "pilot_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
        config.update(options or {})
        script.write_text(jobs.generate(**config))
        return subprocess.run(
            ["bash", str(script)], env={**env, **overrides}, capture_output=True, text=True
        )

    def events(kind):
        return [
            json.loads(s)
            for s in Path(env["EVENTS"]).read_text().splitlines()
            if kind == "*" or json.loads(s)[0] == kind
        ]

    state = (
        Path(env["RUN_DIR"])
        / "state/adapt"
        / ("pilot_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16])
    )
    yield run, state, events, cloud
    for e in events("fit"):
        work = next(
            (p for p in Path(e[4]).parents if p.name.startswith("demand-forecast-adapt-")), None
        )
        if work:
            shutil.rmtree(work, ignore_errors=True)


def test_pipeline_reuse_and_identity(machine):
    run, state, events, _ = machine
    result = run()
    assert result.returncode == 0, result.stderr + result.stdout
    assert [e[1] for e in events("bundle")] == ["alpha", "beta/new"]
    assert len(events("fit")) == 8
    scores = events("score")
    assert len(scores) == 12  # four frozen + eight selected-adapted
    assert all(e[1] in ("alpha", "beta/new") for e in scores)
    assert all("/state/" not in e[3] and "--resume" in e[4] for e in scores)
    rows = list(csv.DictReader((state / "results.csv").open()))
    assert len(rows) == 8
    assert rows[0]["outer_sql_raw"] == rows[0]["frozen_sql_raw"] == "1.0"
    assert rows[2]["outer_sql_raw"] == "0.7"
    assert rows[2]["selected_update"] == "50"
    assert rows[2]["inner_0_window_0"] == "1.0" and rows[2]["inner_selected_window_1"] == "0.9"
    assert rows[2]["trainable_parameters"] == "123"
    assert rows[2]["training_seconds"] == "2.0" and rows[2]["inner_evaluation_seconds"] == "2.0"
    assert set(jobs_columns()) <= set(rows[0])
    assert not any(
        p.suffix in (".npz", ".parquet", ".pt", ".safetensors") for p in state.rglob("*")
    )
    manifests = list(state.glob("tasks/*/bundle-manifest.json"))
    assert len(manifests) == 2 and all(
        json.loads(p.read_text())["schema_version"] == 2 for p in manifests
    )
    before = {k: len(events(k)) for k in ("git", "gcloud", "fit", "score", "bundle")}
    assert run().returncode == 0
    assert before == {k: len(events(k)) for k in before}
    result = run({"max_updates": 100})
    assert result.returncode != 0 and "different inputs" in result.stderr


def test_environment_and_task_order(machine):
    run, _, events, _ = machine
    assert run().returncode == 0
    steps = events("*")
    prepare = [i for i, e in enumerate(steps) if e[0] == "git" and "apply" in e]
    assert prepare
    repo_verified = next(
        i
        for i, e in enumerate(steps)
        if e[0] == "git" and "rev-parse" in e and "repository" in e[2]
    )
    installs = [i for i, e in enumerate(steps) if e[0] == "install" and "uv" not in e]
    downloads = [i for i, e in enumerate(steps) if e[0] == "gcloud"]
    assert (
        repo_verified
        < min(prepare)
        < max(prepare)
        < min(installs)
        <= max(installs)
        < min(downloads)
    )
    assert any(e[0] == "uv" and "3.12" in e for e in steps)
    for name in ("alpha", "beta/new"):
        bundle_index = next(i for i, e in enumerate(steps) if e == ["bundle", name])
        scores = [(i, e) for i, e in enumerate(steps) if e[0] == "score" and e[1] == name]
        assert [e[2] for _, e in scores[:2]] == ["raw", "published"]
        assert bundle_index < min(i for i, _ in scores)


def test_request_acknowledged_after_upload(machine):
    run, state, events, _ = machine
    guest_root = state.parents[2]
    guest_root.mkdir(parents=True, exist_ok=True)
    (guest_root / "checkpoint-request").write_text("nonce-1")
    assert run(REQUEST_CHECKPOINT="1").returncode == 0
    assert (guest_root / "checkpoint-done").read_text().strip() == "nonce-1"
    assert list(state.glob("tasks/*/fits/*/checkpoints.json"))
    assert any("-adapt-artifacts/" in str(e) for e in events("gcloud"))


def test_corrupt_completion_evidence_reruns_only_missing_score(machine):
    run, state, events, _ = machine
    assert run().returncode == 0
    score = next(state.glob("tasks/*/fits/all*/raw/run.json"))
    score.write_text("{")
    before = len(events("score"))
    assert run().returncode == 0
    assert len(events("fit")) == 8 and len(events("bundle")) == 2
    assert len(events("score")) == before + 1


def test_refuses_changed_base_after_failure(machine):
    run, state, _, cloud = machine
    assert run(FAIL_FIT="all11").returncode != 0
    (cloud / "base/model.safetensors").write_bytes(b"changed base")
    # The completion record independently retains the original base identity.
    (state / "checkpoint-sha256.json").unlink()
    result = run()
    assert result.returncode != 0 and "different checkpoint contents" in result.stderr


def test_incomplete_completion_hash_list_is_not_a_fast_path(machine):
    run, state, events, _ = machine
    assert run().returncode == 0
    path = next(state.glob("tasks/*/fits/all*/raw/run.json"))
    path.unlink()
    marker_path = state / "completed.json"
    marker = json.loads(marker_path.read_text())
    del marker["artifacts"][path.relative_to(state).as_posix()]
    marker_path.write_text(json.dumps(marker))
    before = len(events("score"))
    assert run().returncode == 0
    assert len(events("score")) == before + 1


def test_malformed_completion_is_repaired(machine):
    run, state, events, _ = machine
    assert run().returncode == 0
    (state / "completed.json").write_text("{")
    assert run().returncode == 0
    assert len(events("fit")) == 8 and len(events("score")) == 12


@pytest.mark.parametrize("retain_completion", [True, False])
def test_missing_manifest_never_rebinds_changed_bundle(machine, retain_completion):
    run, state, events, _ = machine
    assert run().returncode == 0
    saved = state / "tasks" / b"alpha".hex() / "bundle-manifest.json"
    saved.unlink()
    if not retain_completion:
        (state / "completed.json").unlink()
    work = Path(events("fit")[0][4]).parents[1]
    shutil.rmtree(work / "bundle")
    for _ in range(2):
        assert run(BUNDLE_CONTENT="changed history").returncode != 0
        assert not saved.exists()
    assert len(events("fit")) == 8 and len(events("score")) == 12


def test_frozen_only_restore_without_bundle_identity_refuses_reuse(machine):
    run, state, events, _ = machine
    assert run().returncode == 0
    task = state / "tasks" / b"alpha".hex()
    (task / "bundle-manifest.json").unlink()
    (state / "completed.json").unlink()
    shutil.rmtree(task / "fits")
    shutil.rmtree(Path(events("fit")[0][4]).parents[1] / "bundle")
    assert run().returncode != 0
    assert not (task / "bundle-manifest.json").exists()
    assert len(events("fit")) == 8 and len(events("score")) == 12


def test_partial_restore_without_base_identity_refuses_changed_weights(machine):
    run, state, events, cloud = machine
    assert run().returncode == 0
    (state / "completed.json").unlink()
    (state / "checkpoint-sha256.json").unlink()
    (cloud / "base/model.safetensors").write_bytes(b"changed base")
    for _ in range(2):
        assert run().returncode != 0
        assert not (state / "checkpoint-sha256.json").exists()
    assert len(events("fit")) == 8 and len(events("score")) == 12


@pytest.mark.parametrize("retain_completion", [True, False])
def test_matching_bundle_identity_recovers_from_surviving_receipts(machine, retain_completion):
    run, state, events, _ = machine
    assert run().returncode == 0
    saved = state / "tasks" / b"alpha".hex() / "bundle-manifest.json"
    original = saved.read_bytes()
    saved.unlink()
    if not retain_completion:
        (state / "completed.json").unlink()
    shutil.rmtree(Path(events("fit")[0][4]).parents[1] / "bundle")
    assert run().returncode == 0
    assert saved.read_bytes() == original
    assert len(events("fit")) == 8 and len(events("score")) == 12


@pytest.mark.parametrize("processing", ["raw", "published"])
def test_single_mode_scores_only_requested_mode(machine, processing):
    run, state, events, _ = machine
    assert run({"processing": processing}).returncode == 0
    assert len(events("score")) == 6
    assert all(e[2] == processing for e in events("score"))
    rows = list(csv.DictReader((state / "results.csv").open()))
    other = "published" if processing == "raw" else "raw"
    assert all(row["outer_sql_" + other] == "" for row in rows)


def jobs_columns():
    return [
        "task",
        "params",
        "lr_multiplier",
        "seed",
        "status",
        "selected_update",
        "decision_reason",
        "outer_sql_raw",
        "outer_sql_published",
        "frozen_sql_raw",
        "frozen_sql_published",
        "training_seconds",
        "inner_evaluation_seconds",
        "outer_evaluation_seconds",
        "peak_gpu_memory_bytes",
    ]


def test_failure_continues_and_remote_resume(machine):
    run, state, events, _ = machine
    result = run(FAIL_FIT="all11")
    assert result.returncode != 0
    assert len(events("fit")) == 8
    marker = json.loads((state / "completed.json").read_text())
    assert not marker["success"]
    rows = list(csv.DictReader((state / "results.csv").open()))
    assert sum(r["status"] == "failed" for r in rows) == 2
    # Simulate a replacement guest: only state and uploaded objects remain.
    for event in events("fit"):
        work = next(
            p for p in Path(event[4]).parents if p.name.startswith("demand-forecast-adapt-")
        )
        shutil.rmtree(work, ignore_errors=True)
    assert run().returncode == 0
    fits = events("fit")
    assert len(fits) == 10 and all(e[3] for e in fits[-2:])
    assert len(events("bundle")) == 4  # rebuild disposable data, verify saved manifest
    assert len(events("score")) == 12


def test_failed_scores_retry_without_refitting(machine):
    run, state, events, _ = machine
    assert run(FAIL_SCORE="raw").returncode != 0
    assert len(events("fit")) == 8
    before = len(events("score"))
    assert run().returncode == 0
    assert len(events("fit")) == 8
    assert len(events("score")) == before + 6
    assert json.loads((state / "completed.json").read_text())["success"]


@pytest.mark.parametrize("failure", ["tasks", "repo", "cuda"])
def test_integrity_stops_before_data_or_fit(machine, failure):
    run, _, events, cloud = machine
    overrides = {}
    if failure == "tasks":
        (cloud / "tasks.yaml").write_bytes(b"changed")
    if failure == "repo":
        overrides["REPO_SHA"] = "b" * 40
    if failure == "cuda":
        overrides["NO_CUDA"] = "1"
    assert run(**overrides).returncode != 0
    assert not events("bundle") and not events("fit") and not events("score")
