"""Execute the generated inventory guest offline with real synthetic Parquet rows."""

import base64
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.corpus_v4 import make_inventory_job as jobs

FOLDERS = ["quarantine/covariates/eia_load", "covariates/ons_load"]


def arguments(**changes):
    options = dict(
        name="probe", repo_sha="a" * 40, corpus_uri="gs://offline/corpus", folders=FOLDERS
    )
    return {**options, **changes}


@pytest.mark.parametrize(
    "changes",
    [
        dict(name="../x"),
        dict(name="UPPER"),
        dict(repo_sha="main"),
        dict(repo_sha="A" * 40),
        dict(corpus_uri="gs://offline/../x"),
        dict(corpus_uri="gs://offline/corpus/"),
        dict(corpus_uri="gs://offline/corpus;touch"),
        dict(folders=[]),
        dict(folders="covariates/a"),
        *[
            dict(folders=[f])
            for f in [
                "../a",
                "/covariates/a",
                "covariates/..",
                "covariates/a/b",
                "covariates/a;echo",
                "covariates//a",
            ]
        ],
        dict(folders=[FOLDERS[0], FOLDERS[0]]),
    ],
)
def test_validation(changes):
    with pytest.raises(ValueError):
        jobs.generate(**arguments(**changes))


def test_cli_syntax_bundle_identity_and_deterministic_folder_order(tmp_path):
    path = tmp_path / "job.sh"
    args = ["--name", "probe", "--repo-sha", "a" * 40, "--corpus-uri", "gs://offline/corpus"]
    for folder in FOLDERS:
        args += ["--folder", folder]
    assert jobs.main([*args, "--output", str(path)]) == 0
    assert os.access(path, os.X_OK)
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    source = path.read_text()
    embedded = source.split("# guest-gzip-base64: ")[1].splitlines()[0]
    guest = gzip.decompress(base64.b64decode(embedded))
    assert guest.decode() == jobs.GUEST
    assert hashlib.sha256(guest).hexdigest() in source
    assert source == jobs.generate(**arguments(folders=list(reversed(FOLDERS))))


@pytest.fixture
def machine(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    fixtures = tmp_path / "fixtures"
    for folder in FOLDERS:
        path = fixtures / folder
        path.mkdir(parents=True)
        pq.write_table(
            pa.table(dict(item_id=["one"], source=["load"], freq=["h"], target=[[1.0, 2.0, 3.0]])),
            path / "train-0.parquet",
        )
    bins = tmp_path / "bin"
    bins.mkdir()
    events = tmp_path / "events.jsonl"
    fake = (
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json, os, pathlib, shutil, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["EVENTS"], "a") as file:
    file.write(json.dumps([name, *args]) + "\n")
if name == "git":
    if args[0] == "clone":
        target = pathlib.Path(args[-1]); (target / ".git").mkdir(parents=True)
        module = target / "scripts/corpus_v4"; module.mkdir(parents=True)
        shutil.copy(pathlib.Path(os.environ["SOURCE_REPO"]) / "scripts/corpus_v4/row_inventory.py", module)
    elif args[:2] == ["rev-parse", "HEAD"]:
        print("a" * 40)
elif name == "gcloud":
    if args[-1].startswith("gs://"):
        if os.environ.get("FAIL_UPLOAD"): sys.exit(23)
        if os.environ.get("FAIL_MARKER") and args[-1].endswith("completed.json"): sys.exit(24)
        dest = pathlib.Path(os.environ["CLOUD_MIRROR"]) / args[-1][5:]
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(args[-2], dest)
    else:
        assert args[:3] == ["storage", "cp", "--recursive"]
        relative = args[-2].removeprefix("gs://offline/corpus/")
        source = pathlib.Path(os.environ["FIXTURE_ROOT"]) / relative
        target = pathlib.Path(args[-1]) / source.name
        shutil.copytree(source, target, dirs_exist_ok=True)
elif args[:2] == ["-m", "venv"]:
    target = pathlib.Path(args[-1]) / "bin"; target.mkdir(parents=True, exist_ok=True)
    python = target / "python"
    if not python.exists(): python.symlink_to(sys.argv[0])
elif args[:2] == ["-m", "pip"]:
    if args[2] == "freeze": print("pyarrow==23.0.1")
elif args[:2] == ["-m", "scripts.corpus_v4.row_inventory"] and os.environ.get("FAIL_INVENTORY"):
    sys.exit(19)
else:
    os.execv(os.environ["REAL_PYTHON"], [os.environ["REAL_PYTHON"], *args])
"""
    )
    for name in ("python3", "git", "gcloud"):
        path = bins / name
        path.write_text(fake)
        path.chmod(0o755)
    for name, value in {
        "PATH": str(bins) + os.pathsep + os.environ["PATH"],
        "EVENTS": str(events),
        "REAL_PYTHON": sys.executable,
        "RUN_DIR": str(tmp_path / "run"),
        "STATE_URI": "gs://offline/state",
        "FIXTURE_ROOT": str(fixtures),
        "SOURCE_REPO": str(Path(__file__).resolve().parents[2]),
        "CLOUD_MIRROR": str(tmp_path / "cloud"),
    }.items():
        monkeypatch.setenv(name, value)
    job = tmp_path / "job.sh"
    job.write_text(jobs.generate(**arguments()))

    def invoke(**env):
        return subprocess.run(
            ["bash", str(job)], capture_output=True, text=True, env={**os.environ, **env}
        )

    def calls():
        return [json.loads(line) for line in events.read_text().splitlines()]

    return invoke, tmp_path / "run/state/inventory/probe", calls


def test_guest_downloads_only_requested_folders_publishes_hashes_and_resumes(machine):
    invoke, root, calls = machine
    result = invoke()
    assert result.returncode == 0, result.stderr
    record = json.loads((root / "completed.json").read_text())
    assert set(record["runs"]) == set(FOLDERS)
    assert all(r["exit_code"] == 0 for r in record["runs"].values())
    for name, sha in record["artifacts"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
    for folder in FOLDERS:
        report = root / folder.replace("/", "__") / "inventory.json"
        assert json.loads(report.read_text())["rows"] == 1
    copies = [c for c in calls() if c[0] == "gcloud"]
    downloads = [c[-2] for c in copies if c[-2].startswith("gs://")]
    assert downloads == ["gs://offline/corpus/" + f for f in sorted(FOLDERS)]
    assert all(not c[-1].startswith("gs://offline/corpus") for c in copies)
    assert copies[-1][-1] == "gs://offline/state/inventory/probe/completed.json"
    installs = [c for c in calls() if c[1:4] == ["-m", "pip", "install"]]
    assert len(installs) == 1 and installs[0][-1].startswith("pyarrow")
    before = len(calls())
    assert invoke().returncode == 0
    assert not any(c[0] in ("git", "gcloud") or c[1:3] == ["-m", "pip"] for c in calls()[before:])


@pytest.mark.parametrize("failure", ["FAIL_INVENTORY", "FAIL_UPLOAD", "FAIL_MARKER"])
def test_failure_leaves_no_completion_and_retry_completes(machine, failure):
    invoke, root, _ = machine
    assert invoke(**{failure: "1"}).returncode != 0
    assert not (root / "completed.json").exists()
    result = invoke()
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "uri",
    [
        "gs://offline/corpus/run",
        "gs://offline",
        "gs://offline/../state",
        "gs://offline/state*",
        "https://offline/state",
    ],
)
def test_invalid_destinations_refused_before_external_calls(machine, uri):
    invoke, _, calls = machine
    assert invoke(STATE_URI=uri).returncode != 0
    assert not any(c[0] in ("git", "gcloud") for c in calls())


@pytest.mark.parametrize(
    "mutation",
    ["schema", "identity", "runs", "exit_code", "artifacts", "hash", "missing", "symlink"],
)
def test_resume_requires_complete_valid_record(machine, mutation):
    invoke, root, calls = machine
    assert invoke().returncode == 0
    marker = root / "completed.json"
    record = json.loads(marker.read_text())
    if mutation == "schema":
        record["schema"] = True
    elif mutation == "identity":
        record["inputs"]["repo_sha"] = "b" * 40
    elif mutation == "runs":
        record["runs"].pop(FOLDERS[0])
    elif mutation == "exit_code":
        record["runs"][FOLDERS[0]]["exit_code"] = False
    elif mutation == "artifacts":
        record["artifacts"].pop("environment.txt")
    elif mutation == "hash":
        (root / "environment.txt").write_text("changed")
    elif mutation == "missing":
        (root / "environment.txt").unlink()
    elif mutation == "symlink":
        path = root / "environment.txt"
        path.unlink()
        path.symlink_to(root / "inputs.json")
    marker.write_text(json.dumps(record))
    before = len(calls())
    assert invoke().returncode != 0
    assert not any(c[0] in ("git", "gcloud") for c in calls()[before:])


@pytest.mark.parametrize("mode", ["different-inputs", "orphan-state"])
def test_preflight_rejects_unbound_or_changed_state(machine, mode):
    invoke, root, calls = machine
    root.mkdir(parents=True)
    if mode == "different-inputs":
        (root / "inputs.json").write_text('{"repo_sha":"different"}')
    else:
        (root / "partial.csv").write_text("orphan evidence")
    assert invoke().returncode != 0
    assert not any(c[0] in ("git", "gcloud") for c in calls())


@pytest.fixture
def completed_record(tmp_path):
    namespace = {"__name__": "inventory_guest_test"}
    exec(compile(jobs.GUEST, "inventory-guest.py", "exec"), namespace)
    config = dict(folders=["covariates/a"])
    files = {
        "inputs.json": json.dumps(config),
        "environment.txt": "pyarrow offline",
        "covariates__a/inventory.csv": "evidence",
        "covariates__a/inventory.json": "{}",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    record = dict(
        schema=1,
        inputs=config,
        runs={"covariates/a": dict(exit_code=0, wall_seconds=0, peak_rss_bytes=0)},
        artifacts={
            name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()
        },
    )
    return namespace["preflight"], config, tmp_path, record


def test_complete_zero_metrics_record_resumes(completed_record):
    preflight, config, root, record = completed_record
    (root / "completed.json").write_text(json.dumps(record))
    assert preflight(config, root) == 0


@pytest.mark.parametrize("metric", ["wall_seconds", "peak_rss_bytes"])
@pytest.mark.parametrize("value", [None, -1, True, "0", float("nan"), float("inf")])
def test_completion_refuses_missing_or_invalid_metrics(completed_record, metric, value):
    preflight, config, root, record = completed_record
    if value is None:
        record["runs"]["covariates/a"].pop(metric)
    else:
        record["runs"]["covariates/a"][metric] = value
    (root / "completed.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="completion"):
        preflight(config, root)
