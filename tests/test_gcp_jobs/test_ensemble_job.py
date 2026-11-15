"""Offline cloud-ensemble contracts; real scoring identities, synthetic forecasts."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures/ensemble"
SHA = "a" * 40
TASKS = b"synthetic tasks\n"
NAMES = b"alpha\nbeta/new\n"
URIS = ["gs://offline/score/seed11", "gs://offline/score/seed22"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def load(path):
    return json.loads(path.read_text())


@pytest.fixture
def jobs():
    path = ROOT / "scripts/gcp/jobs/make_ensemble_job.py"
    assert path.exists(), "the ensemble generator is missing"
    spec = importlib.util.spec_from_file_location("make_ensemble_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(**overrides):
    return {
        "name": "average",
        "repo_sha": SHA,
        "tasks_uri": "gs://offline/tasks.yaml",
        "tasks_sha256": hashlib.sha256(TASKS).hexdigest(),
        "member_uris": URIS,
        **overrides,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"name": "../escape"},
        {"name": "-flag"},
        {"name": "_name"},
        {"name": "a/b"},
        {"name": ""},
        {"repo_sha": "main"},
        {"repo_sha": "A" * 40},
        {"tasks_sha256": "bad"},
        {"tasks_sha256": "F" * 64},
        {"tasks_uri": "https://bucket/tasks"},
        {"tasks_uri": "gs://bucket/*"},
        {"tasks_uri": "gs://bucket/../tasks"},
        {"tasks_uri": "gs://bucket/x\ny"},
        {"names_uri": "gs://offline/names"},
        {"names_sha256": "b" * 64},
        {"names_uri": "gs://offline/names", "names_sha256": "bad"},
        {"names_uri": "gs://bucket/*", "names_sha256": "b" * 64},
        {"member_uris": []},
        {"member_uris": URIS[:1]},
        {"member_uris": [URIS[0], URIS[0] + "/"]},
        {"member_uris": [URIS[0], "gs://bucket/../score/run"]},
        {"member_uris": [URIS[0], "https://bucket/score/run"]},
        {"member_uris": [URIS[0], "gs://bucket/not-score/run"]},
        {"processing": "other"},
    ],
)
def test_generator_validation(jobs, change):
    options = arguments()
    options.update(change)
    with pytest.raises(ValueError):
        jobs.generate(**options)


@pytest.mark.parametrize("processing", ["raw", "published", "both"])
def test_bundle_and_cli(jobs, tmp_path, processing):
    output = tmp_path / "job.sh"
    command = [sys.executable, str(ROOT / "scripts/gcp/jobs/make_ensemble_job.py")]
    for key, value in arguments().items():
        if key == "member_uris":
            for uri in value:
                command += ["--member-uri", uri]
        else:
            command += ["--" + key.replace("_", "-"), value]
    command += ["--processing", processing, "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert os.access(output, os.X_OK)
    assert subprocess.run(["bash", "-n", str(output)]).returncode == 0
    snippet = output.read_text().split("<<'ENSEMBLE_BUNDLE'\n")[1].split("\nENSEMBLE_BUNDLE")[0]
    result = subprocess.run(
        [sys.executable, "-", str(tmp_path)], input=snippet, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    config = load(tmp_path / "inputs.json")
    assert config["modes"] == (["raw", "published"] if processing == "both" else [processing])
    assert config["guest_sha256"] == sha(tmp_path / "ensemble.py")


def synthetic_member(directory, index, names=None):
    """Retain the operator's identity schema, replacing data and artifact hashes."""
    names = names or ["alpha", "beta/new"]
    inputs = load(FIXTURES / "member-seed11-inputs.json")
    inputs.update(
        name=f"seed{index}",
        tasks_sha256=hashlib.sha256(TASKS).hexdigest(),
        names_uri=None,
        names_sha256=None,
    )
    if index:
        inputs["covariates"] = "all"
    write(directory / "inputs.json", inputs)
    write(directory / "selected.json", names)
    hashes = {"config.json": "c" * 64, "model.safetensors": str(index + 1) * 64}
    write(directory / "checkpoint-sha256.json", hashes)
    for mode in ["raw", "published"]:
        out = directory / mode
        out.mkdir()
        (out / "summaries.csv").write_text(
            "task,model_name,run_identity,test_error,SQL,SQL[0.5]\n"
            + "".join(f"{n},seed{index}-{mode},identity,1,1,1\n" for n in names)
        )
        write(out / "yardstick.json", {"synthetic": True})
        for name in names:
            task = out / name
            task.mkdir(parents=True)
            (task / "window_0.npz").write_bytes(b"synthetic window")
            write(
                task / "run.json",
                {
                    "run_identity": "identity",
                    "summary": {"test_error": 1},
                    "window_sha256": {"window_0.npz": sha(task / "window_0.npz")},
                },
            )
    marker = load(FIXTURES / "member-seed11-completed.json")
    marker.update(inputs=inputs, checkpoint_sha256=hashes)
    for mode, report in marker["modes"].items():
        report.update(
            requested=names, completed=names, failed={}, exit_code=0, checkpoint_sha256=hashes
        )
    write(directory / "completed.json", marker)
    rehash(directory)


def rehash(directory):
    marker = load(directory / "completed.json")
    marker["artifacts"] = {
        str(p.relative_to(directory)): sha(p)
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "completed.json"
    }
    write(directory / "completed.json", marker)


@pytest.fixture
def members(jobs, tmp_path, monkeypatch):
    guest = {"__name__": "ensemble_test"}
    exec(jobs.GUEST, guest)
    cloud = tmp_path / "cloud"
    for i, uri in enumerate(URIS):
        synthetic_member(cloud / uri.removeprefix("gs://offline/"), i)
    root, scratch = tmp_path / "state", tmp_path / "scratch"
    root.mkdir()
    scratch.mkdir()

    def fetch(uri, target):
        shutil.copyfile(cloud / uri.removeprefix("gs://offline/"), target)

    original_output, original_run = subprocess.check_output, subprocess.run

    def listing(args, **kwargs):
        if args[:2] != ["gcloud", "storage"]:
            return original_output(args, **kwargs)
        assert args[2:4] == ["ls", "--json"]
        uri = args[4].removesuffix("/**")
        directory = cloud / uri.removeprefix("gs://offline/")
        return json.dumps(
            [
                {
                    "url": "gs://offline/" + str(p.relative_to(cloud)),
                    "type": "cloud_object",
                    "metadata": {
                        "bucket": "offline",
                        "name": str(p.relative_to(cloud)),
                        "size": str(p.stat().st_size),
                    },
                }
                for p in directory.rglob("*")
                if p.is_file()
            ]
        )

    def copy(args, **kwargs):
        if args[:2] != ["gcloud", "storage"]:
            return original_run(args, **kwargs)
        assert args[2:4] == ["cp", "--recursive"]
        source = cloud / args[4].removeprefix("gs://offline/")
        shutil.copytree(source, Path(args[5]) / source.name)
        return SimpleNamespace(returncode=0)

    monkeypatch.setitem(guest, "fetch", fetch)
    monkeypatch.setattr(subprocess, "check_output", listing)
    monkeypatch.setattr(subprocess, "run", copy)
    config = arguments()
    config.pop("member_uris")
    config.update(
        member_uris=URIS,
        names_uri=None,
        names_sha256=None,
        modes=["raw", "published"],
        guest_sha256="f" * 64,
    )
    return guest, config, root, scratch, cloud


def test_legacy_covariates_all_accepted_and_only_identity_saved(members):
    guest, config, root, scratch, _ = members
    records, directories = guest["members"](config, root, scratch)
    assert len(records) == len(directories) == 2
    assert records[0]["inputs"].get("covariates") is None
    assert records[1]["inputs"]["covariates"] == "all"
    assert {
        str(p.relative_to(root / "members/0"))
        for p in (root / "members/0").rglob("*")
        if p.is_file()
    } == {"inputs.json", "completed.json", "selected.json", "checkpoint-sha256.json"}
    assert (directories[0] / "raw/alpha/window_0.npz").exists()


def test_simons_and_single_raw_reference_member(members):
    guest, config, root, scratch, cloud = members
    directory = cloud / "score/seed22"
    inputs = load(directory / "inputs.json")
    inputs.update(
        model="timestral",
        model_package="timestral @ git+https://example.invalid/repo@" + "b" * 40,
        modes=["raw"],
    )
    write(directory / "inputs.json", inputs)
    marker = load(directory / "completed.json")
    marker["inputs"] = inputs
    marker["modes"].pop("published")
    write(directory / "completed.json", marker)
    shutil.rmtree(directory / "published")
    rehash(directory)
    config["modes"] = ["raw"]
    records, directories = guest["members"](config, root, scratch)
    assert len(records) == 2
    assert records[1]["inputs"]["model"] == "timestral"
    assert all((p / "raw/alpha/window_0.npz").is_file() for p in directories)
    assert not any((p / "published").exists() for p in directories)


@pytest.mark.parametrize(
    "field",
    ["repo_sha", "upstream_pin", "tasks_sha256", "names_sha256", "max_context", "covariates"],
)
def test_member_evaluation_mismatch(members, field):
    guest, config, root, scratch, cloud = members
    directory = cloud / "score/seed22"
    inputs = load(directory / "inputs.json")
    inputs[field] = "none" if field == "covariates" else "different"
    write(directory / "inputs.json", inputs)
    marker = load(directory / "completed.json")
    marker["inputs"] = inputs
    write(directory / "completed.json", marker)
    rehash(directory)
    with pytest.raises(ValueError, match="seed22.*" + field):
        guest["members"](config, root, scratch)


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "inputs",
        "raw",
        "requested",
        "completed",
        "failed",
        "exit_code",
        "selected.json",
        "model.safetensors",
        "name",
        "checkpoint_sha256",
    ],
)
def test_member_refusals(members, field):
    guest, config, root, scratch, cloud = members
    directory = cloud / "score/seed22"
    marker = load(directory / "completed.json")
    if field == "schema":
        marker["schema"] = 2
    elif field == "inputs":
        marker["inputs"]["name"] = "other"
    elif field == "raw":
        marker["modes"].pop("raw")
    elif field in ["requested", "completed"]:
        marker["modes"]["raw"][field] = ["alpha"]
    elif field == "failed":
        marker["modes"]["raw"]["failed"] = {"alpha": "failed"}
    elif field == "exit_code":
        marker["modes"]["raw"]["exit_code"] = 1
    elif field == "selected.json":
        write(directory / field, ["other"])
        for report in marker["modes"].values():
            report.update(requested=["other"], completed=["other"])
    elif field == "model.safetensors":
        hashes = load(cloud / "score/seed11/checkpoint-sha256.json")
        write(directory / "checkpoint-sha256.json", hashes)
        marker["checkpoint_sha256"] = hashes
    elif field == "checkpoint_sha256":
        marker[field]["config.json"] = "e" * 64
    elif field == "name":
        inputs = load(directory / "inputs.json")
        inputs["name"] = "seed0"
        write(directory / "inputs.json", inputs)
        marker["inputs"] = inputs
    write(directory / "completed.json", marker)
    rehash(directory)
    with pytest.raises(ValueError, match="seed22.*" + field.replace(".", r"\.")):
        guest["members"](config, root, scratch)


@pytest.mark.parametrize("field", ["tasks_sha256", "names_sha256"])
def test_job_and_members_input_mismatch(members, field):
    guest, config, root, scratch, _ = members
    config[field] = "f" * 64
    with pytest.raises(ValueError, match="seed11.*" + field):
        guest["members"](config, root, scratch)


@pytest.mark.parametrize(
    "damage", ["tampered", "missing", "unlisted", "identity-hash", "unsafe-path"]
)
def test_member_artifact_refusals(members, damage):
    guest, config, root, scratch, cloud = members
    directory = cloud / "score/seed11"
    window = directory / "raw/alpha/window_0.npz"
    if damage == "tampered":
        window.write_bytes(b"tampered")
    elif damage == "missing":
        window.unlink()
    elif damage == "unlisted":
        (directory / "raw/extra.txt").write_text("unlisted")
    else:
        marker = load(directory / "completed.json")
        marker["artifacts"]["inputs.json" if damage == "identity-hash" else "raw/../escape"] = (
            "f" * 64
        )
        write(directory / "completed.json", marker)
    with pytest.raises((ValueError, OSError), match="seed11"):
        guest["members"](config, root, scratch)


@pytest.mark.parametrize("filename", ["selected.json", "checkpoint-sha256.json"])
def test_malformed_member_identity_names_the_member_and_field(members, filename):
    guest, config, root, scratch, cloud = members
    directory = cloud / "score/seed22"
    (directory / filename).write_text("{")
    rehash(directory)
    with pytest.raises(ValueError, match="seed22.*" + filename.replace(".", r"\.")):
        guest["members"](config, root, scratch)


def test_saved_failure_cannot_hide_behind_successful_member_status(members):
    guest, config, root, scratch, cloud = members
    directory = cloud / "score/seed22"
    write(directory / "raw/failures.json", {"alpha": {"message": "failed"}})
    rehash(directory)
    with pytest.raises(ValueError, match="seed22.*raw.*failed"):
        guest["members"](config, root, scratch)


@pytest.mark.parametrize("size,slash", [("0", True), (0, True), ("7", True), ("0", False)])
def test_listing_only_ignores_zero_byte_folder_placeholders(members, monkeypatch, size, slash):
    guest, config, root, scratch, _ = members
    original = subprocess.check_output

    def listing(args, **kwargs):
        output = original(args, **kwargs)
        if args[-1] != URIS[0] + "/raw/**":
            return output
        rows = json.loads(output)
        name = "score/seed11/raw/placeholder" + ("/" if slash else "")
        rows += [
            {
                "url": "gs://offline/" + name,
                "type": "cloud_object",
                "metadata": {"bucket": "offline", "name": name, "size": size},
            }
        ]
        return json.dumps(rows)

    monkeypatch.setattr(subprocess, "check_output", listing)
    if int(size) == 0 and slash:
        assert len(guest["members"](config, root, scratch)[0]) == 2
    else:
        with pytest.raises(ValueError, match="seed11.*unlisted.*placeholder"):
            guest["members"](config, root, scratch)


@pytest.mark.parametrize("character", list("*?[]{}"))
@pytest.mark.parametrize("mode", ["raw", "unrequested"])
def test_wildcard_artifact_refused_before_member_forecasts(members, monkeypatch, character, mode):
    guest, config, root, scratch, cloud = members
    directory = cloud / "score/seed22"
    path = f"{mode}/task{character}/window_0.npz"
    target = directory / path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"wildcard must not reach storage cp")
    rehash(directory)
    fetched = []
    original = guest["fetch"]

    def fetch(uri, target):
        fetched.append(uri)
        return original(uri, target)

    monkeypatch.setitem(guest, "fetch", fetch)
    with pytest.raises(ValueError) as error:
        guest["members"](config, root, scratch)
    assert "seed22" in str(error.value) and path in str(error.value)
    assert [u.removeprefix(URIS[1] + "/") for u in fetched if u.startswith(URIS[1])] == [
        "inputs.json",
        "completed.json",
        "selected.json",
        "checkpoint-sha256.json",
    ]
    assert not (scratch / "members/1/raw").exists()


@pytest.mark.parametrize("damage", ["tampered", "unlisted", "evidence"])
@pytest.mark.parametrize("restored", [False, True])
def test_later_member_refusal_preserves_entire_state(members, damage, restored):
    guest, config, root, scratch, cloud = members
    write(root / "inputs.json", config)
    if restored:
        (root / "members/0").mkdir(parents=True)
        shutil.copyfile(cloud / "score/seed11/inputs.json", root / "members/0/inputs.json")
    directory = cloud / "score/seed22"
    if damage == "tampered":
        (directory / "raw/alpha/window_0.npz").write_bytes(b"tampered")
    elif damage == "unlisted":
        (directory / "published/unlisted.txt").write_text("unlisted")
    else:
        record = load(directory / "published/alpha/run.json")
        record["window_sha256"]["window_0.npz"] = "0" * 64
        write(directory / "published/alpha/run.json", record)
        rehash(directory)

    def snapshot():
        return {
            str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
            for p in root.rglob("*")
        }

    before = snapshot()
    with pytest.raises(ValueError, match="seed22"):
        guest["members"](config, root, scratch)
    assert snapshot() == before


@pytest.mark.parametrize("damage", ["missing", "extra", "tampered"])
def test_bulk_copy_output_verified_even_when_listing_matched(members, monkeypatch, damage):
    guest, config, root, scratch, _ = members
    original = subprocess.run

    def copy(args, **kwargs):
        result = original(args, **kwargs)
        directory = Path(args[-1]) / "raw"
        if damage == "missing":
            (directory / "alpha/window_0.npz").unlink()
        elif damage == "extra":
            (directory / "extra.txt").write_bytes(b"appeared after listing")
        else:
            (directory / "alpha/window_0.npz").write_bytes(b"tampered in transit")
        return result

    monkeypatch.setattr(subprocess, "run", copy)
    with pytest.raises(ValueError, match="seed11"):
        guest["members"](config, root, scratch)
    assert not list(root.iterdir())


@pytest.fixture
def machine(jobs, tmp_path):
    """Run the actual shell/guest, faking only installs, Git, storage and scoring."""
    bins, cloud, packages = (tmp_path / n for n in ["bin", "cloud", "packages"])
    bins.mkdir()
    cloud.mkdir()
    packages.mkdir()
    for index, uri in enumerate(URIS):
        synthetic_member(cloud / uri.removeprefix("gs://offline/"), index)
    (cloud / "tasks.yaml").write_bytes(TASKS)
    (cloud / "names.txt").write_bytes(NAMES)
    for name in [
        "demand_forecast",
        "demand_forecast/bench",
        "demand_forecast/bench/yardsticks",
        "fev",
    ]:
        directory = packages / name
        directory.mkdir(exist_ok=True)
        (directory / "__init__.py").write_text("")
    (packages / "demand_forecast/bench/yardsticks/runner.py").write_text(
        "from types import SimpleNamespace\n"
        "def read_names(value): return open(value[1:]).read().splitlines() if value else None\n"
        "def load_tasks(path, names): return [SimpleNamespace(task_name=n) for n in (names or ['alpha','beta/new'])]\n"
    )
    (packages / "demand_forecast/bench/yardsticks/ensemble.py").write_text(
        "def run_ensemble(): pass\n"
    )
    (packages / "fev/metrics.py").write_text(
        "AVAILABLE_METRICS = {'SQL': object(), 'WQL': object()}\n"
    )
    for name, version in [("fev", "0.10.0"), ("datasets", "3.6.0"), ("huggingface_hub", "0.36.0")]:
        directory = packages / f"{name}-{version}.dist-info"
        directory.mkdir()
        (directory / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )
    fake_ensemble = tmp_path / "averaging.py"
    fake_ensemble.write_text(
        textwrap.dedent("""\
        import csv, hashlib, json, os, pathlib, sys
        args = sys.argv[1:]
        def value(key): return args[args.index(key)+1]
        out = pathlib.Path(value('--out-dir'))
        mode = 'published' if out.name.endswith('published') else 'raw'
        parity = '--allow-identical-members' in args
        with open(os.environ['EVENTS'], 'a') as stream:
            stream.write(json.dumps(['ensemble', mode, parity, args]) + '\\n')
        if os.environ.get('INTERRUPT') == mode and not parity:
            # Kill the guest, leaving its durable status and output from the first mode.
            import signal
            os.kill(os.getppid(), signal.SIGKILL)
            sys.exit(91)
        source = pathlib.Path(args[args.index('--members')+1])
        if parity:
            assert args[args.index('--members')+2] == str(source)
        if not (out / 'yardstick.json').exists() and out.exists() and any(out.iterdir()):
            raise ValueError('out-dir is nonempty and has no ensemble manifest')
        out.mkdir(parents=True, exist_ok=True)
        if os.environ.get('MANIFEST_INTERRUPT') == mode and not parity:
            import signal, tempfile
            if os.environ.get('EMPTY_INIT') != '1':
                with tempfile.NamedTemporaryFile(mode='w', dir=out, suffix='.json', delete=False) as stream:
                    stream.write('{')
            os.kill(os.getppid(), signal.SIGKILL)
            sys.exit(91)
        (out / 'yardstick.json').write_text('{}')
        rows, failures = [], {}
        with (source / 'summaries.csv').open() as stream:
            for row in csv.DictReader(stream):
                if not parity and os.environ.get('FAIL_TASK') == row['task']:
                    failures[row['task']] = {'message': os.environ.get('FAIL_MESSAGE', 'synthetic failure')}
                    continue
                row['model_name'] = value('--model-name')
                row['test_error'] = row['SQL'] = row['SQL[0.5]'] = '1'
                task = out / row['task']
                task.mkdir(parents=True, exist_ok=True)
                window = task / 'window_0.npz'
                if not window.exists(): window.write_bytes(b'resumed window')
                record = {'run_identity': 'identity', 'summary': row, 'window_sha256': {'window_0.npz': hashlib.sha256(window.read_bytes()).hexdigest()}}
                (task / 'run.json').write_text(json.dumps(record))
                rows.append(row)
        with (out / 'summaries.csv').open('w') as stream:
            fields = ['task','model_name','run_identity','test_error','SQL','SQL[0.5]']
            writer = csv.DictWriter(stream, fieldnames=fields)
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
        import json, os, pathlib, shutil, sys
        args = sys.argv[1:]
        with open(os.environ['EVENTS'], 'a') as stream: stream.write(json.dumps(['python', *args]) + '\\n')
        if args[:2] == ['-m', 'venv'] or args[:1] == ['--fake-venv']:
            if args[:2] == ['-m', 'venv'] and os.environ.get('FAIL_VENV') == '1': sys.exit(1)
            target = pathlib.Path(args[-1]) / 'bin'
            target.mkdir(parents=True, exist_ok=True)
            (target / 'python').unlink(missing_ok=True)
            (target / 'python').symlink_to(os.environ['WRAPPER'])
            sys.exit(0)
        if args[:3] == ['-m', 'pip', 'install']:
            if 'uv' in args:
                target = pathlib.Path(args[args.index('--target')+1]) / 'bin'
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pathlib.Path(os.environ['BINS']) / 'uv', target / 'uv')
            sys.exit(0)
        if args[:3] == ['-m', 'pip', 'freeze']:
            print('fev==0.10.0\\ndatasets==3.6.0\\nhuggingface-hub==0.36.0')
            sys.exit(0)
        if args[:3] == ['-m', 'pip', 'check']: sys.exit(0)
        if args[:1] == ['-c'] and 'version_info' in args[1]:
            print('3.10' if os.environ.get('NEED_PYTHON') == '1' and 'venv' in sys.argv[0] and not pathlib.Path(os.environ['UPGRADED']).exists() else '3.12')
            sys.exit(0)
        if args and args[0].endswith('/scripts/fev_ensemble.py'):
            os.execv(sys.executable, [sys.executable, os.environ['AVERAGER'], *args[1:]])
        os.execv(sys.executable, [sys.executable, *args])
    """)
    )
    (bins / "uv").write_text(
        '#!/bin/bash\ntouch "$UPGRADED"\nexec "$WRAPPER" --fake-venv "${@: -1}"\n'
    )
    (bins / "git").write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
        import json, os, pathlib, shutil, sys
        args = sys.argv[1:]
        with open(os.environ['EVENTS'], 'a') as stream: stream.write(json.dumps(['git', *args]) + '\\n')
        if args[0] == 'init':
            repo = pathlib.Path(args[-1])
            repo.mkdir()
            shutil.copy2(os.environ['REQUIREMENTS'], repo / 'requirements-yardsticks.txt')
        if 'rev-parse' in args: print(os.environ.get('FETCHED_SHA', 'a'*40))
    """)
    )
    (bins / "gcloud").write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
        import json, os, pathlib, shutil, sys
        args = sys.argv[1:]
        with open(os.environ['EVENTS'], 'a') as stream: stream.write(json.dumps(['gcloud', *args]) + '\\n')
        assert args[0] == 'storage'
        if args[1] == 'cp':
            if args[2] == '--recursive':
                source = pathlib.Path(os.environ['CLOUD']) / args[3].removeprefix('gs://offline/')
                shutil.copytree(source, pathlib.Path(args[4]) / source.name)
                sys.exit(0)
            source, target = args[2:]
            shutil.copyfile(pathlib.Path(os.environ['CLOUD']) / source.removeprefix('gs://offline/'), target)
        elif args[1:3] == ['ls', '--json']:
            prefix = args[3].removesuffix('/**')
            cloud = pathlib.Path(os.environ['CLOUD'])
            directory = cloud / prefix.removeprefix('gs://offline/')
            print(json.dumps([{'url': 'gs://offline/' + str(p.relative_to(cloud)), 'type': 'cloud_object',
                              'metadata': {'bucket': 'offline', 'name': str(p.relative_to(cloud)), 'size': str(p.stat().st_size)}}
                              for p in directory.rglob('*') if p.is_file()]))
        else: raise AssertionError(args)
    """)
    )
    for p in bins.iterdir():
        p.chmod(0o755)
    name = "average_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    env = {
        **os.environ,
        "PATH": str(bins) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(packages),
        "RUN_DIR": str(tmp_path / "run"),
        "STATE_URI": "gs://offline/state",
        "WRAPPER": str(wrapper),
        "BINS": str(bins),
        "CLOUD": str(cloud),
        "AVERAGER": str(fake_ensemble),
        "REQUIREMENTS": str(ROOT / "requirements-yardsticks.txt"),
        "EVENTS": str(tmp_path / "events.jsonl"),
        "UPGRADED": str(tmp_path / "upgraded"),
    }
    env.pop("HOME", None)

    def run(options=None, **overrides):
        path = tmp_path / "job.sh"
        path.write_text(jobs.generate(**arguments(name=name, **(options or {}))))
        return subprocess.run(
            ["bash", str(path)], env={**env, **overrides}, capture_output=True, text=True
        )

    def events():
        path = tmp_path / "events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    return run, tmp_path / "run/state/ensemble" / name, events, cloud


def test_shell_completion_preflight_and_different_inputs(machine):
    run, state, events, _ = machine
    result = run(FAIL_VENV="1", NEED_PYTHON="1")
    assert result.returncode == 0, result.stderr + result.stdout
    calls = [e[1:3] for e in events() if e[0] == "ensemble"]
    assert calls == [["raw", True], ["published", True], ["raw", False], ["published", False]]
    marker = load(state / "completed.json")
    assert marker["schema"] == 1
    assert marker["members"][0]["inputs"]["repo_sha"] != marker["inputs"]["repo_sha"]
    assert len(marker["members"]) == 2
    assert "members/0/completed.json" in marker["artifacts"]
    assert all(
        load(state / "parity" / (mode + ".json"))["values_compared"] == 6
        for mode in ["raw", "published"]
    )
    assert '"simons_required": false' in (state / "environment.txt").read_text()
    before = len(events())
    assert run().returncode == 0
    assert not any(
        e[0] in ["git", "gcloud", "ensemble"] or e[:3] == ["python", "-m", "pip"]
        for e in events()[before:]
    )
    snapshot = {str(p.relative_to(state)): p.read_bytes() for p in state.rglob("*") if p.is_file()}
    result = run({"repo_sha": "b" * 40})
    assert result.returncode != 0 and "different inputs" in result.stderr
    assert snapshot == {
        str(p.relative_to(state)): p.read_bytes() for p in state.rglob("*") if p.is_file()
    }


def test_parity_failure_prevents_all_averaging(machine):
    run, state, events, cloud = machine
    path = cloud / "score/seed11/raw/summaries.csv"
    path.write_text(path.read_text().replace(",1,1,1", ",2,1,1"))
    rehash(cloud / "score/seed11")
    result = run()
    assert result.returncode != 0 and "parity failed" in result.stderr
    report = load(state / "parity/raw.json")
    assert report["success"] is False
    assert report["differences"] == [
        {"task": name, "column": "test_error", "recorded": "2", "recomputed": "1"}
        for name in ["alpha", "beta/new"]
    ]
    assert not any(e[0] == "ensemble" and not e[2] for e in events())
    assert not (state / "completed.json").exists()


def test_interruption_between_modes_restores_to_new_machine(machine, tmp_path):
    run, state, events, _ = machine
    result = run(INTERRUPT="published")
    assert result.returncode != 0
    assert (state / "raw-status.json").exists() and not (state / "completed.json").exists()
    window = state / "raw/alpha/window_0.npz"
    modified = window.stat().st_mtime_ns
    restored = tmp_path / "new-machine"
    shutil.copytree(state.parents[1], restored / "state")
    result = run(RUN_DIR=str(restored))
    assert result.returncode == 0, result.stderr
    target = restored / "state/ensemble" / state.name
    assert (target / "raw/alpha/window_0.npz").stat().st_mtime_ns == modified
    calls = [e[3] for e in events() if e[0] == "ensemble" and not e[2]]
    assert calls[0][calls[0].index("--members") + 1] == calls[2][calls[2].index("--members") + 1]
    assert load(target / "completed.json")["modes"]["published"]["failed"] == {}


@pytest.mark.parametrize("mode", ["raw", "published"])
@pytest.mark.parametrize("empty", [False, True])
def test_interrupted_mode_initialization_restores_to_new_machine(machine, tmp_path, mode, empty):
    run, state, _, _ = machine
    result = run(MANIFEST_INTERRUPT=mode, EMPTY_INIT=str(int(empty)))
    assert result.returncode != 0
    assert (state / mode).is_dir() and not (state / mode / "yardstick.json").exists()
    assert bool(list((state / mode).iterdir())) is not empty
    restored = tmp_path / "restored-machine"
    shutil.copytree(state.parents[1], restored / "state")
    result = run(RUN_DIR=str(restored))
    assert result.returncode == 0, result.stderr
    target = restored / "state/ensemble" / state.name
    assert load(target / "completed.json")["modes"][mode]["failed"] == {}
    assert not list((target / mode).glob("tmp*.json"))


@pytest.mark.parametrize("other", ["important.json", "tmp-not-ours.json", "subdirectory"])
def test_manifestless_mode_with_other_content_is_preserved_and_refused(machine, other):
    run, state, _, _ = machine
    # Interrupt just before the second mode starts so all prior state has an identity.
    assert run(INTERRUPT="published").returncode != 0
    output = state / "published"
    output.mkdir()
    if other == "subdirectory":
        (output / other).mkdir()
    else:
        (output / other).write_bytes(b"not ours to repair")
    snapshot = {p.name: p.read_bytes() if p.is_file() else None for p in output.iterdir()}
    result = run()
    assert result.returncode != 0 and "published" in result.stderr
    assert "manifest" in result.stderr and "refus" in result.stderr
    assert snapshot == {p.name: p.read_bytes() if p.is_file() else None for p in output.iterdir()}


@pytest.mark.parametrize("published", [False, True])
def test_mode_recovery_respects_published_manifest(members, published):
    guest, _, root, _, _ = members
    directory = root / "raw"
    directory.mkdir()
    for name in ["tmpabcdefgh.json", "interrupted.tmp"]:
        (directory / name).write_bytes(b"unfinished")
    if published:
        (directory / "yardstick.json").write_bytes(b"published identity")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    guest["recover_mode"](root, "raw")
    if published:
        assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    else:
        assert not directory.exists()


def test_one_bulk_copy_per_member_mode(machine):
    run, _, events, _ = machine
    result = run()
    assert result.returncode == 0, result.stderr
    copies = [e[3:] for e in events() if e[:3] == ["gcloud", "storage", "cp"]]
    bulk = [args for args in copies if args[0] == "--recursive"]
    assert sorted(args[1] for args in bulk) == sorted(
        uri + "/" + mode for uri in URIS for mode in ["raw", "published"]
    )
    assert len(copies) == 4 + 8 + 1  # Four mode trees, four identities/member, tasks.


@pytest.mark.parametrize("damage", ["window", "marker", "input", "unidentified"])
def test_damaged_state(machine, damage):
    run, state, events, _ = machine
    assert run().returncode == 0
    if damage == "window":
        (state / "raw/alpha/window_0.npz").unlink()
    elif damage == "marker":
        (state / "completed.json").write_text("{")
    elif damage == "input":
        (state / "inputs.json").write_text("null")
    else:
        (state / "inputs.json").write_text("null")
        (state / "completed.json").unlink()
    before = len(events())
    result = run()
    if damage == "unidentified":
        assert result.returncode != 0 and "no valid input identity" in result.stderr
        assert not any(e[0] in ["gcloud", "git", "ensemble"] for e in events()[before:])
    else:
        assert result.returncode == 0, result.stderr
        assert (state / "raw/alpha/window_0.npz").is_file()


def test_failed_task_reports_both_modes_and_retry(machine):
    run, state, events, _ = machine
    result = run(FAIL_TASK="alpha")
    assert result.returncode != 0
    marker = load(state / "completed.json")
    for mode in ["raw", "published"]:
        assert marker["modes"][mode]["failed"] == {"alpha": "synthetic failure"}
        assert marker["modes"][mode]["completed"] == ["beta/new"]
    result = run()
    assert result.returncode == 0, result.stderr
    assert len([e for e in events() if e[0] == "ensemble" and not e[2]]) == 4


def test_empty_failure_message_still_records_both_mode_statuses(machine):
    run, state, _, _ = machine
    result = run(FAIL_TASK="alpha", FAIL_MESSAGE="")
    assert result.returncode != 0
    for mode in ["raw", "published"]:
        assert load(state / (mode + "-status.json"))["failed"]["alpha"] == "task incomplete"


@pytest.mark.parametrize(
    "a,b,equal",
    [
        ("", "NaN", True),
        ("nan", "nan", True),
        ("nan", "0", False),
        ("1", "1.0000000001", True),
        ("1", "1.0001", False),
        ("0", "1e-13", True),
    ],
)
def test_metric_parity_nan_and_tolerances(jobs, a, b, equal):
    guest = {"__name__": "ensemble_test"}
    exec(jobs.GUEST, guest)
    report = guest["compare_metrics"](
        {"boomlet_1062": {"SQL": a, "inference_time_s": "1"}},
        {"boomlet_1062": {"SQL": b, "inference_time_s": "2"}},
        ["boomlet_1062"],
        {"SQL"},
    )
    assert report["success"] is equal
    assert report["columns"] == ["SQL"]


@pytest.mark.parametrize("processing", ["raw", "published"])
def test_single_mode_with_names(machine, processing):
    run, state, events, cloud = machine
    for name in ["seed11", "seed22"]:
        directory = cloud / "score" / name
        inputs = load(directory / "inputs.json")
        inputs.update(
            names_uri="gs://offline/names.txt", names_sha256=hashlib.sha256(NAMES).hexdigest()
        )
        write(directory / "inputs.json", inputs)
        marker = load(directory / "completed.json")
        marker["inputs"] = inputs
        write(directory / "completed.json", marker)
        rehash(directory)
    result = run(
        {
            "processing": processing,
            "names_uri": "gs://offline/names.txt",
            "names_sha256": hashlib.sha256(NAMES).hexdigest(),
        }
    )
    assert result.returncode == 0, result.stderr
    assert set(load(state / "completed.json")["modes"]) == {processing}
    assert (state / "names.txt").read_bytes() == NAMES
    for e in events():
        if e[0] == "ensemble":
            assert "--names" in e[3] and "--keep-going" in e[3]


def test_environment_pin_mismatch_stops_before_averaging(machine, tmp_path):
    run, _, events, _ = machine
    metadata = tmp_path / "packages/fev-0.10.0.dist-info/METADATA"
    metadata.write_text(metadata.read_text().replace("Version: 0.10.0", "Version: 0.9.0"))
    result = run()
    assert result.returncode != 0 and "fev must remain pinned" in result.stderr
    assert not any(e[0] == "ensemble" for e in events())


def test_symlink_state_refused(machine, tmp_path):
    run, state, _, _ = machine
    outside = tmp_path / "outside"
    outside.mkdir()
    state.parent.mkdir(parents=True)
    state.symlink_to(outside, target_is_directory=True)
    result = run()
    assert result.returncode != 0 and "symlink" in result.stderr
    assert not list(outside.iterdir())


@pytest.mark.parametrize(
    "case", ["ordinary", "nan", "parity-drift", "interrupted", "manifest-interrupted"]
)
def test_real_fev_parity_mean_and_resume(members, tmp_path, monkeypatch, case):
    """Use real saved windows/scoring/skip rules; replace only external storage/processes."""
    fev = pytest.importorskip("fev")
    from dataclasses import replace

    import datasets.config
    import huggingface_hub.constants
    import numpy as np
    import pandas as pd
    import yaml

    from demand_forecast.bench.yardsticks import ensemble as ensemble_module
    from demand_forecast.bench.yardsticks import run_yardstick
    from scripts.fev_ensemble import main as ensemble_main
    from tests.test_bench.test_fev_ensemble import OffsetAdapter
    from tests.test_bench.test_yardsticks import scenario

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "datasets-cache")
    monkeypatch.setattr(datasets.config, "HF_DATASETS_OFFLINE", True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    guest, config, root, scratch, cloud = members
    tasks, options, _ = scenario.__wrapped__(tmp_path)
    if case == "nan":
        from datasets import Dataset

        data = Dataset.from_parquet(tasks[0].dataset_path)
        data.map(
            lambda row: {"target": [10.0] * len(row["target"]), "other": [20.0] * len(row["other"])}
        ).to_parquet(tasks[0].dataset_path)
        tasks = [fev.Task(**{**tasks[0].to_dict(), "task_name": "boomlet_1062"})]
    (cloud / "tasks.yaml").write_text(yaml.safe_dump({"tasks": [t.to_dict() for t in tasks]}))
    config["tasks_sha256"] = sha(cloud / "tasks.yaml")
    config["name"] += "_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    for index, uri in enumerate(URIS):
        directory = cloud / uri.removeprefix("gs://offline/")
        weights = tmp_path / f"weights-{index}"
        weights.mkdir()
        (weights / "config.json").write_text("{}")
        (weights / "model.safetensors").write_bytes(f"seed {index}".encode())
        for mode in config["modes"]:
            shutil.rmtree(directory / mode)
            run_yardstick(
                tasks,
                replace(options, weights=str(weights), model_name=f"seed{index}-{mode}"),
                directory / mode,
                adapter_factory=lambda _, i=index: OffsetAdapter(i * 2),
            )
        inputs = load(directory / "inputs.json")
        inputs["tasks_sha256"] = config["tasks_sha256"]
        write(directory / "inputs.json", inputs)
        write(directory / "selected.json", [t.task_name for t in tasks])
        marker = load(directory / "completed.json")
        marker["inputs"] = inputs
        for report in marker["modes"].values():
            report.update(
                requested=[t.task_name for t in tasks], completed=[t.task_name for t in tasks]
            )
        write(directory / "completed.json", marker)
        rehash(directory)
    calls, interrupted = [], False
    original_run = subprocess.run
    original_save = ensemble_module._save_failures

    class ManifestInterruption(BaseException):
        pass

    def save(path, value):
        nonlocal interrupted
        if (
            case == "manifest-interrupted"
            and path == root / "published/yardstick.json"
            and not interrupted
        ):
            import tempfile

            interrupted = True
            # Simulate SIGKILL after NamedTemporaryFile creation, before its rename.
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, suffix=".json", delete=False
            ) as stream:
                stream.write('{"ensemble_version":')
            raise ManifestInterruption("synthetic stop during manifest initialization")
        return original_save(path, value)

    monkeypatch.setattr(ensemble_module, "_save_failures", save)

    def execute(args, **kwargs):
        nonlocal interrupted
        if len(args) < 2 or not str(args[1]).endswith("fev_ensemble.py"):
            return original_run(args, **kwargs)
        out = Path(args[args.index("--out-dir") + 1])
        is_parity = "--allow-identical-members" in args
        calls.append((out.name, is_parity))
        if case == "interrupted" and out.name == "published" and not interrupted:
            interrupted = True
            raise InterruptedError("synthetic stop between modes")
        return SimpleNamespace(returncode=ensemble_main(args[2:]))

    monkeypatch.setattr(subprocess, "run", execute)
    # A deliberate scorer drift still passes the script's member consistency
    # checks: both its recorded row and run.json summary carry the old metric.
    if case == "parity-drift":
        directory = cloud / "score/seed11/raw"
        frame = pd.read_csv(directory / "summaries.csv", float_precision="round_trip")
        frame.loc[0, "test_error"] += 1
        frame.to_csv(directory / "summaries.csv", index=False)
        record_path = directory / tasks[0].task_name / "run.json"
        record = load(record_path)
        record["summary"]["test_error"] += 1
        write(record_path, record)
        rehash(directory.parent)
    assert guest["preflight"](config, root) == 10
    (root / "environment.txt").write_text(
        "synthetic integration; fev present, install not exercised\n"
    )
    if case == "parity-drift":
        with pytest.raises(ValueError, match="parity failed"):
            guest["average"](config, root, scratch, ROOT, sys.executable)
        report = load(root / "parity/raw.json")
        assert report["differences"][0]["column"] == "test_error"
        assert all(parity for _, parity in calls)
        return
    if case in ["interrupted", "manifest-interrupted"]:
        with pytest.raises((InterruptedError, ManifestInterruption)):
            guest["average"](config, root, scratch, ROOT, sys.executable)
        if case == "manifest-interrupted":
            assert not (root / "published/yardstick.json").exists()
            assert list((root / "published").glob("tmp*.json"))
        saved = root / "raw" / tasks[0].task_name / "window_0.npz"
        before = saved.read_bytes(), saved.stat().st_mtime_ns
        restored = tmp_path / "restored"
        shutil.copytree(root, restored)
        shutil.rmtree(root)
        root = restored
        shutil.rmtree(scratch)
        scratch = tmp_path / "new-scratch"
        scratch.mkdir()
    assert guest["average"](config, root, scratch, ROOT, sys.executable) == 0
    assert guest["preflight"](config, root) == 0
    if case in ["interrupted", "manifest-interrupted"]:
        saved = root / "raw" / tasks[0].task_name / "window_0.npz"
        assert (saved.read_bytes(), saved.stat().st_mtime_ns) == before
    for mode in config["modes"]:
        report = load(root / "parity" / (mode + ".json"))
        assert report["success"] and report["tasks_compared"] == 1
        source = pd.read_csv(cloud / "score/seed11" / mode / "summaries.csv")
        if case == "nan":
            assert source[["SQL", "test_error"]].isna().all().all()
        for i in range(tasks[0].num_windows):
            name = f"window_{i}.npz"
            with np.load(root / mode / tasks[0].task_name / name) as actual:
                with np.load(cloud / "score/seed11" / mode / tasks[0].task_name / name) as first:
                    with np.load(
                        cloud / "score/seed22" / mode / tasks[0].task_name / name
                    ) as second:
                        np.testing.assert_array_equal(
                            actual["arrays"], first["arrays"] / 2 + second["arrays"] / 2
                        )
