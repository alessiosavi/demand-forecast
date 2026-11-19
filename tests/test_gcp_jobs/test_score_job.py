"""Offline execution of the generated scoring job; external installation is faked."""

import ast
import base64
import gzip
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
TIMESTRAL_PIN = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
PACKAGES = {
    "timestral": "timestral @ git+https://github.com/mistralai/timestral.git@" + TIMESTRAL_PIN,
    "chronos2": "chronos-forecasting==2.2.2",
}


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


def reference_arguments(model="timestral", **overrides):
    return arguments(
        model=model,
        checkpoint_uri=None,
        hub_weights="abel/model",
        hub_revision="b" * 40,
        **overrides,
    )


def v5_arguments(**overrides):
    return arguments(**{"model": "v5", "export_digest": "d" * 64, **overrides})


@pytest.mark.parametrize(
    "model,default,selected",
    [
        (
            "chronos2",
            "09452e7a3286f0dd19ae95eda06e3abbbf513ad08aa7f10d1127c1797cbba6e8",
            "bf833f40f4ee847a055d3d67805f4579a4de0f1a35bd20ec5ca1a4d7d097f62b",
        ),
        (
            "timestral",
            "33aa6294bd6056b613afbf4fee83a863b2861e5ae0ca7ddf26123ffdd749bedb",
            "8319e8bf6c60d58e0929b0eeb54db20fbd1529c741b8443cb81146fb0598061a",
        ),
    ],
)
def test_reference_jobs_match_base_revision(score_jobs, model, default, selected):
    selection = {
        "names_uri": "gs://offline/names.txt",
        "names_sha256": hashlib.sha256(NAMES).hexdigest(),
        "covariates": "none",
        "max_context": 1024,
    }
    for options, expected in [({}, default), (selection, selected)]:
        job = score_jobs.generate(**reference_arguments(model, **options)).encode()
        packed = job.split(b'packed = base64.b64decode("', 1)[1].split(b'")\n', 1)[0]
        guest = gzip.decompress(base64.b64decode(packed))
        # Pins from a753e18's generator. zlib output varies across platforms,
        # so pin the guest bytes and assert the base's exact local gzip encoding.
        assert packed == base64.b64encode(gzip.compress(guest, mtime=0))
        assert (
            hashlib.sha256(job.replace(packed, base64.b64encode(guest), 1)).hexdigest() == expected
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {
            "checkpoint_uri": None,
            "hub_weights": "abel/model",
            "hub_revision": "b" * 40,
            "processing": "published",
            "covariates": "none",
            "max_context": 1024,
            "names_uri": "gs://offline/names.txt",
            "names_sha256": hashlib.sha256(NAMES).hexdigest(),
        },
    ],
)
def test_simons_job_byte_identical_to_e4351d8(score_jobs, overrides):
    path = ROOT / "scripts/gcp/jobs/make_score_job.py"
    # Archive the unmodified base generator so shallow CI checkouts also run
    # the original function, with this interpreter's gzip implementation.
    source = gzip.decompress(
        (Path(__file__).parent / "fixtures/make_score_job-e4351d8.py.gz").read_bytes()
    )
    assert hashlib.sha256(source).hexdigest() == (
        "aff259642e04c9f92fae755db93d02c2bb50d63a1a83b963fef7564ee331a993"
    )
    base = {"__file__": str(path), "__name__": "base_score_job"}
    exec(compile(source, str(path), "exec"), base)
    expected = base["generate"](**arguments(**overrides)).encode()
    assert score_jobs.generate(**arguments(**overrides)).encode() == expected
    assert score_jobs.generate(**arguments(**overrides), model="simons").encode() == expected


@pytest.mark.parametrize("model", ["timestral", "chronos2"])
@pytest.mark.parametrize(
    "change",
    [
        {"checkpoint_uri": "gs://offline/export", "hub_weights": None, "hub_revision": None},
        {"hub_weights": None},
        {"hub_revision": None},
        {"hub_revision": "main"},
        {"hub_revision": "b" * 39},
        {"processing": "raw"},
        {"processing": "published"},
        {"covariates": "other"},
        {"max_context": 0},
    ],
)
def test_reference_validation(score_jobs, model, change):
    options = {**reference_arguments(model), **change}
    with pytest.raises(ValueError):
        score_jobs.generate(**options)


@pytest.mark.parametrize("model", ["timestral", "chronos2"])
@pytest.mark.parametrize("count", [0, 2], ids=["missing", "duplicated"])
@pytest.mark.parametrize(
    "target,snippet",
    [
        (
            "Simons package check",
            """    upstream = json.loads(metadata.distribution("simons").read_text("direct_url.json"))
    if upstream["vcs_info"]["commit_id"] != config["upstream_pin"]:
        raise ValueError("installed Simons commit does not match upstream pin")""",
        ),
        (
            "Simons child command",
            """"--model", "simons",
                   "--checkpoint", str(checkpoint), "--processing", mode,""",
        ),
    ],
    ids=["package-check", "child-command"],
)
def test_reference_generation_refuses_template_drift(
    score_jobs, monkeypatch, model, count, target, snippet
):
    assert score_jobs.GUEST.count(snippet) == 1
    monkeypatch.setattr(score_jobs, "GUEST", score_jobs.GUEST.replace(snippet, snippet * count))
    with pytest.raises(RuntimeError, match=target):
        score_jobs.generate(**reference_arguments(model))


@pytest.mark.parametrize("model", ["timestral", "chronos2"])
def test_reference_generation_refuses_leftover_simons_child(score_jobs, monkeypatch, model):
    monkeypatch.setattr(score_jobs, "GUEST", score_jobs.GUEST + '\nextra = ["--model", "simons"]\n')
    with pytest.raises(RuntimeError, match="Simons child"):
        score_jobs.generate(**reference_arguments(model))


@pytest.mark.parametrize("model", ["simons", "timestral", "chronos2", "unknown"])
def test_model_cli(tmp_path, model):
    options = arguments() if model == "simons" else reference_arguments(model)
    options["model"] = model
    command = [sys.executable, str(ROOT / "scripts/gcp/jobs/make_score_job.py")]
    for key, value in options.items():
        if value is not None:
            command += ["--" + key.replace("_", "-"), value]
    output = tmp_path / "job.sh"
    result = subprocess.run(command + ["--output", str(output)], capture_output=True, text=True)
    if model == "unknown":
        assert result.returncode == 2 and "invalid choice" in result.stderr
        assert not output.exists()
        return
    assert result.returncode == 0, result.stderr
    assert subprocess.run(["bash", "-n", str(output)]).returncode == 0
    snippet = output.read_text().split("<<'SCORE_BUNDLE'\n", 1)[1].split("\nSCORE_BUNDLE\n", 1)[0]
    result = subprocess.run(
        [sys.executable, "-", str(tmp_path)], input=snippet, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / "inputs.json").read_text())
    ast.parse((tmp_path / "score.py").read_text(), feature_version=(3, 10))
    if model != "simons":
        assert config["model"] == model
        assert config["modes"] == ["raw"]
        assert config["model_package"] == PACKAGES[model]
        assert PACKAGES[model] in output.read_text()


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


@pytest.mark.parametrize("variant", [None, "no_peers"])
def test_v5_cli_emits_raw_job(tmp_path, variant):
    output = tmp_path / "job.sh"
    command = [sys.executable, str(ROOT / "scripts/gcp/jobs/make_score_job.py")]
    for key, value in v5_arguments(variant=variant).items():
        if value is not None:
            command += ["--" + key.replace("_", "-"), value]
    result = subprocess.run(command + ["--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert os.access(output, os.X_OK)
    result = subprocess.run(["bash", "-n", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    job = output.read_text()
    install = next(line for line in job.splitlines() if '--editable "$repo[bench]"' in line)
    assert install.endswith('--editable "$repo[bench]"')
    assert "Simons.git" not in job
    snippet = job.split("<<'SCORE_BUNDLE'\n", 1)[1].split("\nSCORE_BUNDLE\n", 1)[0]
    result = subprocess.run(
        [sys.executable, "-", str(tmp_path)], input=snippet, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / "inputs.json").read_text())
    assert config["model"] == "v5" and config["modes"] == ["raw"]
    assert config["export_digest"] == "d" * 64 and config["variant"] == variant
    guest = (tmp_path / "score.py").read_text()
    ast.parse(guest, feature_version=(3, 10))
    assert '"--model", "simons"' not in guest
    assert 'metadata.distribution("simons")' not in guest


def test_v5_generation_loads_variants_without_torch(request, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "demand_forecast", None)
    score_jobs = request.getfixturevalue("score_jobs")
    assert score_jobs.generate(**v5_arguments(variant="no_value_stats")).startswith(
        "#!/usr/bin/env bash"
    )


@pytest.mark.parametrize(
    "change,message",
    [
        ({"checkpoint_uri": None}, "checkpoint URI"),
        ({"checkpoint_uri": None, "hub_weights": "abel/model"}, "Hub weights"),
        ({"hub_revision": "b" * 40}, "revision"),
        ({"processing": "published"}, "raw"),
        ({"export_digest": None}, "export digest"),
        ({"export_digest": "d" * 63}, "export digest"),
        ({"export_digest": "D" * 64}, "export digest"),
        ({"variant": "unknown"}, "unknown v5 variant"),
        ({"covariates": "none"}, "covariates"),
        ({"max_context": 15}, "at least 16"),
        ({"upstream_pin": "b" * 40}, "upstream pin"),
    ],
)
def test_v5_refuses_invalid_options(score_jobs, change, message):
    with pytest.raises(ValueError, match=message):
        score_jobs.generate(**v5_arguments(**change))


@pytest.mark.parametrize("model", ["simons", "timestral", "chronos2"])
@pytest.mark.parametrize(
    "change, message",
    [({"variant": "full"}, "variant"), ({"export_digest": "d" * 64}, "export digest")],
)
def test_other_models_refuse_v5_options(score_jobs, model, change, message):
    options = arguments(model=model) if model == "simons" else reference_arguments(model)
    with pytest.raises(ValueError, match=message):
        score_jobs.generate(**options, **change)


@pytest.mark.parametrize(
    "target,anchor",
    [
        ("Simons package check", '    upstream = json.loads(metadata.distribution("simons")'),
        ("Simons child command", '"--model", "simons",'),
        ("checkpoint fetch files", '    for filename in ("config.json", "model.safetensors"):'),
        ("checkpoint hash files", ' for name in ("config.json", "model.safetensors")'),
        ("checkpoint hash keys", 'set(value) == {"config.json", "model.safetensors"}'),
        ("export digest check", '}\n    prior = root / "checkpoint-sha256.json"'),
        ("task evidence scan", '            windows = record["window_sha256"]'),
        (
            "completion artifacts",
            '                required.update(mode + "/" + name + "/" + p for p in record["window_sha256"])',
        ),
    ],
)
def test_v5_generation_refuses_missing_template_site(score_jobs, monkeypatch, target, anchor):
    assert score_jobs.GUEST.count(anchor) == 1
    monkeypatch.setattr(score_jobs, "GUEST", score_jobs.GUEST.replace(anchor, ""))
    with pytest.raises(RuntimeError, match=target):
        score_jobs.generate(**v5_arguments())


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
        ("timestral", "0.1.0"),
        ("chronos_forecasting", "2.2.2"),
    ]:
        info = packages / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        (info / "direct_url.json").write_text(
            json.dumps({"vcs_info": {"commit_id": TIMESTRAL_PIN if name == "timestral" else PIN}})
        )
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
        model = value('--model')
        assert model in ('simons', 'timestral', 'chronos2', 'v5')
        assert '--resume' in args and '--keep-going' in args
        if model == 'simons':
            assert pathlib.Path(value('--checkpoint'), 'model.safetensors').is_file()
        elif model == 'v5':
            checkpoint = pathlib.Path(value('--checkpoint'))
            export = pathlib.Path(os.environ['CLOUD']) / 'export'
            for name in ('config.json', 'model.safetensors', 'parity.npz'):
                assert (checkpoint / name).read_bytes() == (export / name).read_bytes()
            assert value('--processing') == 'raw'
            assert not any(flag in args for flag in ('--weights', '--revision', '--covariates'))
            if 'EXPECTED_VARIANT' in os.environ:
                assert value('--variant') == os.environ['EXPECTED_VARIANT']
            else:
                assert '--variant' not in args
        else:
            assert value('--weights') == 'abel/model'
            assert value('--revision') == 'b' * 40
            assert '--processing' not in args and '--checkpoint' not in args
        assert pathlib.Path(value('--tasks')).read_bytes() == bytes.fromhex(os.environ['TASK_HEX'])
        if '--names' in args:
            assert pathlib.Path(value('--names')[1:]).read_text() == 'alpha\\nbeta/new\\n'
        assert not pathlib.Path(os.environ['HF_DATASETS_CACHE']).is_relative_to(pathlib.Path(os.environ['RUN_DIR']) / 'state')
        mode = value('--processing') if model in ('simons', 'v5') else 'raw'
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
            record = {'run_identity': 'identity', 'summary': row, 'window_sha256': {'window_0.npz': hashlib.sha256(window.read_bytes()).hexdigest()}}
            if model == 'v5':
                damage = os.environ.get('V5_DAMAGE') if name == 'alpha' else None
                row['test_error'] = 'NaN' if damage == 'nonfinite' else 0.5
                evidence = {'v5-report.json': b'{"run_identity": "identity"}', 'evidence_window_0.npz': b'offline evidence'}
                record['evidence_sha256'] = {p: hashlib.sha256(data).hexdigest() for p, data in evidence.items()}
                for p, data in evidence.items():
                    if damage == 'missing' and p == 'evidence_window_0.npz':
                        continue
                    (task / p).write_bytes(data)
            (task / 'run.json').write_text(json.dumps(record))
            rows.append(row)
        with (out / 'summaries.csv').open('w') as stream:
            fields = ['task', 'model_name', 'run_identity']
            if model == 'v5':
                fields.append('test_error')
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
            assert '' not in args, 'empty pip operand'
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
        path = tmp_path / "events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    yield run, Path(env["RUN_DIR"]) / "state/score" / score_name, events, cloud
    # The checkpoint path must remain stable across guest labels for the adapter's identity.
    for event in events():
        if event[0] == "score" and "--checkpoint" in event[2]:
            checkpoint = Path(event[2][event[2].index("--checkpoint") + 1])
            if checkpoint.parent.name.startswith("demand-forecast-score-"):
                shutil.rmtree(checkpoint.parent, ignore_errors=True)


@pytest.fixture
def v5_machine(machine):
    _, _, _, cloud = machine
    shutil.rmtree(cloud.parent / "packages/simons-0.1.1.dist-info")
    export = cloud / "export"
    (export / "parity.npz").write_bytes(b"offline parity")
    config = {
        "format": "v5-export-1",
        "model": {"width": 24},
        "files": {
            name: hashlib.sha256((export / name).read_bytes()).hexdigest()
            for name in ("model.safetensors", "parity.npz")
        },
    }
    (export / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    digest = hashlib.sha256((export / "config.json").read_bytes()).hexdigest()
    return machine, {"model": "v5", "export_digest": digest}


def test_v5_scores_export_and_completed_job_skips_fetch(v5_machine):
    (run, state, events, cloud), options = v5_machine
    options.update(variant="no_peers", max_context=16, processing="raw")
    result = run(options, EXPECTED_VARIANT="no_peers")
    assert result.returncode == 0, result.stderr
    steps = events()
    assert [e[3] for e in steps if e[0] == "gcloud"] == [
        "gs://offline/tasks.yaml",
        "gs://offline/export/config.json",
        "gs://offline/export/model.safetensors",
        "gs://offline/export/parity.npz",
    ]
    install = next(e for e in steps if e[:5] == ["python", "-m", "pip", "install", "-r"])
    assert len(install) == 8
    assert install[5].endswith("/requirements-yardsticks.txt")
    assert install[6] == "--editable" and install[7].endswith("/repository[bench]")
    scores = [e for e in steps if e[0] == "score"]
    assert len(scores) == 1 and scores[0][1] == "raw"
    args = scores[0][2]
    assert args[args.index("--max-context") + 1] == "16"
    marker_path = state / "completed.json"
    before = marker_path.read_bytes()
    marker = json.loads(before)
    assert marker["inputs"]["export_digest"] == options["export_digest"]
    assert marker["inputs"]["variant"] == "no_peers"
    assert marker["modes"]["raw"]["completed"] == ["alpha", "beta/new"]
    assert {"raw/alpha/v5-report.json", "raw/alpha/evidence_window_0.npz"} <= marker[
        "artifacts"
    ].keys()
    shutil.rmtree(cloud / "export")
    assert run(options).returncode == 0
    assert marker_path.read_bytes() == before
    assert not any(e[0] in ["git", "gcloud", "score", "uv"] for e in events()[len(steps) :])
    for change in [{"export_digest": "f" * 64}, {"variant": "full"}]:
        result = run({**options, **change})
        assert result.returncode != 0 and "different inputs" in result.stderr


def test_v5_interrupted_run_resumes_from_three_key_record(v5_machine):
    (run, state, events, cloud), options = v5_machine
    assert run(options, FAIL_TASK="1").returncode == 1
    marker = state / "completed.json"
    assert json.loads(marker.read_text())["modes"]["raw"]["completed"] == ["beta/new"]
    marker.unlink()
    record = state / "checkpoint-sha256.json"
    saved = record.read_bytes()
    assert json.loads(saved) == {
        name: hashlib.sha256((cloud / "export" / name).read_bytes()).hexdigest()
        for name in ("config.json", "model.safetensors", "parity.npz")
    }
    before = len(events())
    result = run(options)
    assert result.returncode == 0, result.stderr
    assert record.read_bytes() == saved
    assert json.loads(marker.read_text())["modes"]["raw"]["completed"] == ["alpha", "beta/new"]
    assert sum(e[0] == "gcloud" and "/export/" in e[3] for e in events()[before:]) == 3


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors", "parity.npz"])
@pytest.mark.parametrize("resume", [False, True], ids=["first-run", "resume"])
def test_v5_rejects_changed_export_before_committing_checkpoint(v5_machine, filename, resume):
    (run, state, events, cloud), options = v5_machine
    manifest = state / "checkpoint-sha256.json"
    if resume:
        assert run(options, FAIL_TASK="1").returncode == 1
        saved = manifest.read_bytes()
    start = len(events())
    target = cloud / "export" / filename
    target.write_bytes(target.read_bytes() + b" ")
    result = run(options)
    assert result.returncode != 0, result.stderr
    assert "sha256 mismatch" in result.stderr and filename in result.stderr
    assert not any(
        e[0] == "python" and e[1].endswith("/scripts/fev_yardstick.py") for e in events()[start:]
    )
    if resume:
        assert manifest.read_bytes() == saved
        step = next(e for e in events() if e[0] == "score")
        checkpoint = Path(step[2][step[2].index("--checkpoint") + 1])
        assert (checkpoint / filename).read_bytes() != target.read_bytes()
    else:
        assert not manifest.exists()


@pytest.mark.parametrize("damage", ["nonfinite", "missing"])
def test_v5_incomplete_task_is_reported_even_if_child_succeeds(v5_machine, damage):
    (run, state, _, _), options = v5_machine
    result = run(options, V5_DAMAGE=damage)
    assert result.returncode == 1, result.stderr
    report = json.loads((state / "completed.json").read_text())["modes"]["raw"]
    assert report["exit_code"] == 0
    assert report["completed"] == ["beta/new"] and set(report["failed"]) == {"alpha"}


@pytest.mark.parametrize("filename", ["v5-report.json", "evidence_window_0.npz"])
@pytest.mark.parametrize("damage", ["changed", "unlisted"])
def test_v5_preflight_checks_runner_digests_and_completion_manifest(v5_machine, filename, damage):
    (run, state, events, _), options = v5_machine
    assert run(options).returncode == 0
    marker_path = state / "completed.json"
    marker = json.loads(marker_path.read_text())
    relative = "raw/alpha/" + filename
    if damage == "changed":
        artifact = state / relative
        artifact.write_bytes(b"changed")
        # The outer manifest agrees with disk, but the runner's digest disagrees.
        marker["artifacts"][relative] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    else:
        del marker["artifacts"][relative]
    marker_path.write_text(json.dumps(marker))
    before = len(events())
    assert run(options).returncode == 0
    assert sum(e[0] == "score" for e in events()[before:]) == 1


@pytest.mark.parametrize("model", ["timestral", "chronos2"])
@pytest.mark.parametrize("covariates", ["all", "none"])
def test_reference_guest_windows_resume_and_identity(machine, model, covariates):
    run, state, events, _ = machine
    options = reference_arguments(
        model,
        covariates=covariates,
        max_context=2048,
        names_uri="gs://offline/names.txt",
        names_sha256=hashlib.sha256(NAMES).hexdigest(),
    )
    options.pop("name")
    result = run(options, FAIL_TASK="1")
    assert result.returncode == 1, result.stderr
    assert json.loads((state / "completed.json").read_text())["modes"]["raw"]["failed"] == {
        "alpha": "inference broke"
    }
    result = run(options)
    assert result.returncode == 0, result.stderr
    marker = json.loads((state / "completed.json").read_text())
    assert marker["inputs"]["model"] == model
    assert marker["inputs"]["model_package"] == PACKAGES[model]
    assert marker["inputs"]["covariates"] == covariates
    assert list(marker["modes"]) == ["raw"]
    assert not (state / "published").exists()
    assert marker["modes"]["raw"]["completed"] == ["alpha", "beta/new"]
    for name in ["alpha", "beta/new"]:
        record = json.loads((state / "raw" / name / "run.json").read_text())
        window = "raw/" + name + "/window_0.npz"
        assert marker["artifacts"][window] == record["window_sha256"]["window_0.npz"]
    calls = [e for e in events() if e[0] == "score"]
    assert [e[1] for e in calls] == ["raw", "raw"]
    for _, _, command, _ in calls:
        assert command[command.index("--model") + 1] == model
        assert command[command.index("--weights") + 1] == "abel/model"
        assert command[command.index("--revision") + 1] == "b" * 40
        assert "--checkpoint" not in command
        assert "--processing" not in command
        assert command[command.index("--max-context") + 1] == "2048"
        assert "--names" in command
        assert ("--covariates" in command) == (covariates == "none")
    installs = [e for e in events() if e[:4] == ["python", "-m", "pip", "install"]]
    assert any(PACKAGES[model] in e for e in installs)
    assert not any("simons @" in arg for e in installs for arg in e)
    before = len(events())
    assert run(options).returncode == 0
    assert not any(e[0] in ["score", "git", "gcloud", "uv"] for e in events()[before:])
    saved = {p: p.read_bytes() for p in state.rglob("*") if p.is_file()}
    for change in [
        {"model": "chronos2" if model == "timestral" else "timestral"},
        {"hub_revision": "c" * 40},
        {"covariates": "none" if covariates == "all" else "all"},
        {"max_context": 1024},
    ]:
        result = run({**options, **change})
        assert result.returncode != 0 and "different inputs" in result.stderr
        assert {p: p.read_bytes() for p in state.rglob("*") if p.is_file()} == saved
    (state / "raw/alpha/window_0.npz").unlink()
    result = run(options)
    assert result.returncode == 0, result.stderr
    assert len([e for e in events() if e[0] == "score"]) == 3


@pytest.mark.parametrize("model", ["timestral", "chronos2"])
def test_reference_package_pin_mismatch_stops_before_scoring(machine, model):
    run, _, events, cloud = machine
    packages = cloud.parent / "packages"
    if model == "timestral":
        (packages / "timestral-0.1.0.dist-info/direct_url.json").write_text(
            json.dumps({"vcs_info": {"commit_id": "f" * 40}})
        )
    else:
        metadata = packages / "chronos_forecasting-2.2.2.dist-info/METADATA"
        metadata.write_text(metadata.read_text().replace("Version: 2.2.2", "Version: 2.2.3"))
    result = run(reference_arguments(model))
    assert result.returncode != 0 and "does not match pin" in result.stderr
    assert not any(e[0] == "score" for e in events())


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
    config.pop("covariates")
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
    assert json.loads(marker_path.read_text())["inputs"] == config
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


@pytest.mark.parametrize("with_marker", [False, True])
def test_139f3f7_state_refuses_covariates_none_without_changes(machine, with_marker):
    run, state, events, _ = machine
    result = run()
    assert result.returncode == 0, result.stderr
    identity = state / "inputs.json"
    config = json.loads(identity.read_text())
    config.pop("covariates")
    config["guest_sha256"] = "a726e1e81f4ad804683b5164aebbf38aefb2107a7983a0b4faad8b1de200916f"
    identity.write_text(json.dumps(config, sort_keys=True) + "\n")
    marker_path = state / "completed.json"
    if with_marker:
        marker = json.loads(marker_path.read_text())
        marker["inputs"] = config
        marker["artifacts"]["inputs.json"] = hashlib.sha256(identity.read_bytes()).hexdigest()
        marker_path.write_text(json.dumps(marker))
    else:
        marker_path.unlink()
    saved = {path: path.read_bytes() for path in state.rglob("*") if path.is_file()}
    before = len(events())
    result = run({"covariates": "none"})
    assert result.returncode != 0 and "no valid input identity" in result.stderr
    assert {path: path.read_bytes() for path in state.rglob("*") if path.is_file()} == saved
    assert not any(e[0] in ["score", "git", "gcloud", "uv"] for e in events()[before:])


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
