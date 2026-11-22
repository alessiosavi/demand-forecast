#!/usr/bin/env python3
"""Generate a self-contained, pinned CPU classical-baseline GCP guest job."""

import argparse
import base64
import gzip
import hashlib
import json
import re
import shlex
from pathlib import Path

MODELS = (
    "seasonal_naive",
    "auto_ets",
    "auto_theta",
    "median3",
    "snaive",
    "snaive_v5lag",
    "snaive_v5point",
    "snaive_v5pairs",
    "snaive_v5patch",
)
# Only these packages are pinned here; the remaining bench/base dependencies
# resolve for the guest's Python (3.10). The full yardstick lock needs >=3.11
# and includes GPU-model packages, so must not be installed by this CPU job.
PINS = {"fev": "0.10.0", "datasets": "3.6.0", "statsforecast": "2.0.1", "torch": "2.8.0+cpu"}

GUEST = r"""
import csv
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse

HISTORY_ERROR = "All time series in the dataset are too short for the chosen cutoff, horizon and min_context_length"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def atomic(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, path)


def read(path):
    return json.loads(path.read_text())


def safe_child(root, name):
    if (not isinstance(name, str) or not name or name.startswith("/") or "\\" in name
            or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("unsafe baseline state/task path: " + str(name))
    path = root
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink in baseline state: " + str(path))
    return path


def state_root(config):
    run = pathlib.Path(os.environ["RUN_DIR"])
    if not run.is_absolute():
        raise ValueError("RUN_DIR must be absolute")
    for path in (run, *run.parents):
        if path.is_symlink():
            raise ValueError("symlink in baseline state: " + str(path))
    root = safe_child(run, "state/baseline/" + config["name"])
    root.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("symlink in baseline state")
    uri = os.environ["STATE_URI"].rstrip("/")
    if (not re.fullmatch(r"gs://[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9_./:@+=-]+)?", uri)
            or any(p in ("", ".", "..") for p in uri[5:].split("/"))):
        raise ValueError("invalid baseline STATE_URI")
    destination = uri + "/baseline/" + config["name"]
    for source in (config["tasks_uri"], config["names_uri"]):
        if source and (source == destination or source.startswith(destination + "/")):
            raise ValueError("baseline state destination overlaps an input")
    return root, destination


def artifact_names(config):
    names = {"inputs.json", "environment.txt", "tasks.yaml", "selected.json",
             "summaries.csv", "progress.json", "status.json", "baseline.log"}
    if config["names_uri"]:
        names.add("names.txt")
    return names


def summary_rows(config, root, names):
    with (root / "summaries.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"task", "model_name"} <= set(reader.fieldnames or []):
            raise ValueError("invalid baseline summary columns")
        rows = {}
        for row in reader:
            name = row.get("task")
            if (name not in names or name in rows or row.get("model_name") != config["model"]
                    or None in row or any(v is None for k, v in row.items() if k != "test_error")):
                raise ValueError("invalid or duplicate baseline summary: " + str(name))
            rows[name] = row
        return reader.fieldnames, rows


def validate_score(row):
    try:
        if math.isfinite(float(row.get("test_error"))):
            return
    except (ValueError, TypeError):
        pass
    raise ValueError("non-finite baseline score")


def row_digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def progress(root):
    try:
        value = read(root / "progress.json")
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_rows(root, columns, rows):
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=root, suffix=".tmp", delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows.values())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, root / "summaries.csv")


def checkpoint(config, root, names):
    _, rows = summary_rows(config, root, names)
    for row in rows.values():
        validate_score(row)
    atomic(root / "progress.json", {name: row_digest(row) for name, row in rows.items()})


def resume(config, root, names):
    if not (root / "summaries.csv").exists():
        return
    columns, rows = summary_rows(config, root, names)
    recorded, verified = progress(root), {}
    for name, row in rows.items():
        try:
            validate_score(row)
        except ValueError:
            continue
        if recorded.get(name) == row_digest(row):
            verified[name] = row
    if verified != rows:
        save_rows(root, columns, verified)
    atomic(root / "progress.json", {name: row_digest(row) for name, row in verified.items()})


def evidence(config, root, names):
    _, rows = summary_rows(config, root, names)
    recorded = progress(root)
    for row in rows.values():
        validate_score(row)
    if (set(recorded) != set(rows)
            or any(recorded.get(name) != row_digest(row) for name, row in rows.items())):
        raise ValueError("baseline summary progress sha256 mismatch")
    return [name for name in names if name in rows]


def complete(config, root, marker):
    try:
        hashes = marker["artifacts"]
        if (type(marker["schema"]) is not int or marker["schema"] != 1 or marker["inputs"] != config
                or set(hashes) != artifact_names(config)
                or not all(digest(safe_child(root, p)) == h for p, h in hashes.items())):
            return False
        if digest(root / "tasks.yaml") != config["tasks_sha256"]:
            return False
        if config["names_uri"] and digest(root / "names.txt") != config["names_sha256"]:
            return False
        names, status = read(root / "selected.json"), read(root / "status.json")
        if (not isinstance(names, list) or not names or len(set(names)) != len(names)
                or any(not isinstance(n, str) for n in names)):
            return False
        for name in names:
            safe_child(root, name)
        failed, kinds = status["failed"], status["failure_kinds"]
        return (status["requested"] == names and status["completed"] == evidence(config, root, names)
                and isinstance(failed, dict) and isinstance(kinds, dict)
                and set(kinds) == set(failed)
                and all(isinstance(reason, str) and reason == HISTORY_ERROR for reason in failed.values())
                and all(kind == "insufficient_history" for kind in kinds.values())
                and not set(status["completed"]) & set(failed)
                and set(status["completed"]) | set(failed) == set(names)
                and type(status["exit_code"]) is int and status["exit_code"] == 0
                and status["repo_sha"] == config["repo_sha"]
                and status["tasks_sha256"] == config["tasks_sha256"]
                and status["model"] == config["model"] and status["versions"] == config["pins"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError, csv.Error):
        return False


def preflight(config, root):
    identity, marker_path = root / "inputs.json", root / "completed.json"
    try:
        marker = read(marker_path) if marker_path.exists() else {}
    except ValueError:
        marker = {}
    if not isinstance(marker, dict):
        marker = {}
    if marker.get("inputs", config) != config:
        raise ValueError("different inputs under the same baseline name; choose a new name")
    if identity.exists():
        if read(identity) != config:
            raise ValueError("different inputs under the same baseline name; choose a new name")
    else:
        if marker.get("inputs") != config and any(root.iterdir()):
            raise ValueError("baseline state has no valid input identity; refusing to overwrite")
        atomic(identity, config)
    if complete(config, root, marker):
        return 0
    expected = marker.get("artifacts", {}).get("summaries.csv")
    summary = root / "summaries.csv"
    if expected and summary.exists() and digest(summary) != expected:
        # Retain the invalid marker's checksum evidence so repeated retries
        # cannot silently trust damaged rows after the first refusal.
        raise ValueError("baseline summaries sha256 mismatch; restore evidence or choose a new name")
    # Invalidate completion before installation, children, or publication can fail.
    marker_path.unlink(missing_ok=True)
    return 10


def environment(config, root, scratch, python):
    from importlib import metadata
    import torch
    versions = {name: metadata.version(name) for name in config["pins"]}
    if versions != config["pins"]:
        raise ValueError("baseline package versions do not match pins: " + str(versions))
    if torch.version.cuda is not None or torch.__version__ != config["pins"]["torch"]:
        raise ValueError("baseline requires CPU-only torch")
    freeze = subprocess.check_output([python, "-m", "pip", "freeze"], text=True)
    facts = {"python": sys.version.split()[0], "versions": versions,
             "repo_sha": config["repo_sha"], "device": "cpu", "cuda": torch.version.cuda}
    path = scratch / "environment.txt"
    path.write_text(freeze + "\n# runtime " + json.dumps(facts, sort_keys=True) + "\n")
    os.replace(path, root / "environment.txt")


def fetch_inputs(config, root, scratch):
    for key, filename in (("tasks", "tasks.yaml"), ("names", "names.txt")):
        if config[key + "_uri"]:
            target = scratch / filename
            subprocess.run(["gcloud", "storage", "cp", config[key + "_uri"], str(target)], check=True)
            if digest(target) != config[key + "_sha256"]:
                raise ValueError(key + " sha256 mismatch")
            os.replace(target, root / filename)


class TaskSpec:
    # run_baseline only needs this protocol. Native fev definitions need no
    # training role profiles, and construction must happen inside each child.
    def __init__(self, values):
        self.values = values
        self.name = values["task_name"]

    def to_task(self):
        import fev
        return fev.Task(**self.values)


def selected(config, root):
    import yaml
    data = yaml.safe_load((root / "tasks.yaml").read_text())
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list) or not data["tasks"]:
        raise ValueError("task YAML must contain a nonempty tasks list")
    tasks = {}
    for raw in data["tasks"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("dataset_path"), str) or not raw["dataset_path"]:
            raise ValueError("each task must contain a dataset_path")
        values = dict(raw)
        alias = values.pop("name", None)
        if alias is not None:
            if values.get("task_name") not in (None, alias):
                raise ValueError("conflicting name and task_name")
            values["task_name"] = alias
        source = values["dataset_path"]
        dataset = pathlib.Path(source).expanduser()
        if not urlparse(source).scheme and (
                dataset.suffix in (".parquet", ".arrow") or source.startswith(("./", "../", "~", "/"))):
            values["dataset_path"] = str((root / dataset).resolve())
            values["dataset_config"] = None
        name = values.get("task_name")
        if name is None:
            name = values.get("dataset_config")
            if name is None:
                name = "/".join(pathlib.Path(values["dataset_path"]).parts[-3:-1])
        safe_child(root, name)
        if name.strip() != name or "," in name or any(c in name for c in "\r\n"):
            raise ValueError("invalid task name")
        if name in tasks:
            raise ValueError("duplicate task name: " + name)
        values["task_name"] = name
        tasks[name] = TaskSpec(values)
    names = list(tasks)
    if config["names_uri"]:
        names = (root / "names.txt").read_text().splitlines()
        if (not names or any(not n.strip() or n != n.strip() or "," in n for n in names)
                or len(set(names)) != len(names) or set(names) - tasks.keys()):
            raise ValueError("names file must contain one unique, known task name per line")
    return [tasks[name] for name in names]


def task(config, root, scratch, index):
    specs = selected(config, root)
    spec = specs[index]
    result = {"name": spec.name}
    code = 0
    try:
        from demand_forecast.bench.run import run_baseline
        run_baseline([spec], root, model=config["model"])
        _, rows = summary_rows(config, root, [s.name for s in specs])
        if spec.name not in rows:
            raise ValueError("baseline child produced no summary")
        validate_score(rows[spec.name])
        result["completed"] = True
    except Exception as error:
        expected = isinstance(error, ValueError) and str(error) == HISTORY_ERROR
        result.update(reason=" ".join(str(error).splitlines()) or type(error).__name__,
                      kind="insufficient_history" if expected else "error")
        code = 0 if expected else 1
    atomic(scratch / "task-result.json", result)
    return code


def run(config, root, scratch, repo, python, destination):
    from demand_forecast.bench.run import run_baseline
    fetch_inputs(config, root, scratch)
    specs = selected(config, root)
    names = [spec.name for spec in specs]
    atomic(root / "selected.json", names)
    resume(config, root, names)
    # Even an all-short batch gets the runner's own empty summaries.csv schema.
    run_baseline([], root, model=config["model"])
    checkpoint(config, root, names)
    failed, kinds, unsuccessful = {}, {}, set()
    started = time.monotonic()

    def report():
        completed = [n for n in evidence(config, root, names) if n not in failed and n not in unsuccessful]
        status = {"requested": names, "completed": completed, "failed": failed,
                  "failure_kinds": kinds, "exit_code": int(bool(unsuccessful)),
                  "repo_sha": config["repo_sha"], "tasks_sha256": config["tasks_sha256"],
                  "model": config["model"], "versions": config["pins"],
                  "wall_seconds": time.monotonic() - started}
        atomic(root / "status.json", status)
        return status

    for index, name in enumerate(names):
        result_path = scratch / "task-result.json"
        result_path.unlink(missing_ok=True)
        command = [python, __file__, "task", sys.argv[2], str(scratch), str(repo), python, str(index)]
        with (root / "baseline.log").open("a") as stream:
            code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
        try:
            result = read(result_path)
            if not isinstance(result, dict) or result.get("name") != name:
                raise ValueError("invalid task result")
        except (OSError, ValueError):
            result = {}
        if code or result.get("completed") is not True:
            expected = (code == 0 and result.get("kind") == "insufficient_history"
                        and result.get("reason") == HISTORY_ERROR)
            failed[name] = result.get("reason") or "baseline child exit " + str(code) + "; see baseline.log"
            kinds[name] = "insufficient_history" if expected else "error"
            if not expected:
                unsuccessful.add(name)
        elif name not in summary_rows(config, root, names)[1]:
            failed[name], kinds[name] = "baseline child produced no summary", "error"
            unsuccessful.add(name)
        if name in failed:
            columns, rows = summary_rows(config, root, names)
            if name in rows:
                del rows[name]
                save_rows(root, columns, rows)
        # The runner may rewrite existing CSV rows (or add metric columns), so
        # checkpoint its output only after validating the child and removing failures.
        checkpoint(config, root, names)
        report()
    status = report()
    artifacts = {name: digest(root / name) for name in sorted(artifact_names(config))}
    # Publish resumable rows/status even after a failed child, but never completion.
    for name in artifacts:
        subprocess.run(["gcloud", "storage", "cp", str(root / name), destination + "/" + name], check=True)
    if status["exit_code"]:
        return status["exit_code"]
    marker = root / "completed.json"
    atomic(marker, {"schema": 1, "inputs": config, "artifacts": artifacts})
    try:
        subprocess.run(["gcloud", "storage", "cp", str(marker), destination + "/completed.json"], check=True)
    except BaseException:
        marker.unlink(missing_ok=True)
        raise
    return 0


def main():
    action, config_file, scratch_dir, repo_dir, python = sys.argv[1:6]
    config, scratch, repo = read(pathlib.Path(config_file)), pathlib.Path(scratch_dir), pathlib.Path(repo_dir)
    root, destination = state_root(config)
    if action == "preflight":
        return preflight(config, root)
    if action == "environment":
        environment(config, root, scratch, python)
        return 0
    if action == "task":
        return task(config, root, scratch, int(sys.argv[6]))
    if action == "run":
        return run(config, root, scratch, repo, python, destination)
    raise ValueError("unknown baseline guest action")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        sys.exit("Baseline job failed: " + " ".join(str(error).splitlines()))
"""


def _sha(value, length, label):
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"{label} must be full {length}-character lowercase hex")


def _uri(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"gs://[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9_./:@+=-]+", value)
        or any(part in ("", ".", "..") for part in value[5:].split("/"))
    ):
        raise ValueError("URI must name an explicit gs:// bucket object without dot components")


def generate(
    *,
    name,
    repo_sha,
    tasks_uri,
    tasks_sha256,
    names_uri=None,
    names_sha256=None,
    model="seasonal_naive",
):
    """Return a validated guest script; generation performs no external calls."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError(
            "name must start with a letter/digit and contain only letters, digits, _ or -"
        )
    _sha(repo_sha, 40, "repo SHA")
    _sha(tasks_sha256, 64, "tasks sha256")
    _uri(tasks_uri)
    if model not in MODELS:
        raise ValueError("unknown baseline model; choose from " + ", ".join(MODELS))
    if (names_uri is None) != (names_sha256 is None):
        raise ValueError("names URI and sha256 are required together")
    if names_uri is not None:
        _uri(names_uri)
        _sha(names_sha256, 64, "names sha256")
    content = GUEST.encode()
    guest_sha = hashlib.sha256(content).hexdigest()
    config = dict(
        name=name,
        repo_sha=repo_sha,
        tasks_uri=tasks_uri,
        tasks_sha256=tasks_sha256,
        names_uri=names_uri,
        names_sha256=names_sha256,
        model=model,
        pins=PINS,
        guest_sha256=guest_sha,
    )
    encoded_config = base64.b64encode(json.dumps(config, sort_keys=True).encode()).decode()
    encoded_guest = base64.b64encode(gzip.compress(content, mtime=0)).decode()
    pins = "".join(f"{name}=={version}\n" for name, version in sorted(PINS.items()))
    return f'''#!/usr/bin/env bash
# Classical baseline {name}; repository {repo_sha}; CPU only
set -euo pipefail
umask 077
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export STATE_URI}}"
case "$RUN_DIR" in /*) ;; *) echo 'RUN_DIR must be absolute' >&2; exit 2 ;; esac
export RUN_DIR STATE_URI
export HF_HOME="$RUN_DIR/baseline-cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export UV_CACHE_DIR="$RUN_DIR/baseline-cache/uv"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE HF_TOKEN HUGGING_FACE_HUB_TOKEN
mkdir -p "$RUN_DIR"
scratch=$(mktemp -d "$RUN_DIR/baseline-install-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
python3 - "$scratch" <<'BASELINE_BUNDLE'
import base64, gzip, hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
content = gzip.decompress(base64.b64decode("{encoded_guest}"))
assert hashlib.sha256(content).hexdigest() == "{guest_sha}"
(root / "baseline.py").write_bytes(content)
(root / "inputs.json").write_bytes(base64.b64decode("{encoded_config}"))
BASELINE_BUNDLE
repo="$scratch/repository"
python="$scratch/venv/bin/python"
guest() {{
    "$1" "$scratch/baseline.py" "$2" "$scratch/inputs.json" "$scratch" "$repo" "$python"
}}
status=0
guest python3 preflight || status=$?
if [[ $status == 0 ]]; then exit 0; fi
if [[ $status != 10 ]]; then exit "$status"; fi
mkdir -p "$HF_DATASETS_CACHE" "$UV_CACHE_DIR"
git init "$repo"
git -C "$repo" remote add origin https://github.com/alessiosavi/demand-forecast.git
git -C "$repo" fetch --depth=1 origin {shlex.quote(repo_sha)}
git -C "$repo" checkout --detach FETCH_HEAD
[[ $(git -C "$repo" rev-parse HEAD) == {shlex.quote(repo_sha)} ]] || {{ echo 'repository SHA mismatch' >&2; exit 2; }}
# Isolate from image CUDA packages; retain the CPython 3.10 guest interpreter.
if ! python3 -m venv "$scratch/venv"; then
    python3 -m pip install --target "$scratch/uv-bootstrap" uv
    "$scratch/uv-bootstrap/bin/uv" venv --clear --seed --python "$(command -v python3)" \\
        --no-python-downloads "$scratch/venv"
fi
cat > "$scratch/pins.txt" <<'BASELINE_PINS'
{pins}BASELINE_PINS
"$python" -m pip install --upgrade pip
"$python" -m pip install 'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
"$python" -m pip install --constraint "$scratch/pins.txt" --editable "$repo[bench]"
"$python" -m pip check
guest "$python" environment
guest "$python" run
'''


def main():
    """Write one executable job without touching cloud resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("name", "repo-sha", "tasks-uri", "tasks-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--names-uri")
    parser.add_argument("--names-sha256")
    parser.add_argument("--model", choices=MODELS, default="seasonal_naive")
    parser.add_argument("--output", required=True, type=Path)
    args = vars(parser.parse_args())
    output = args.pop("output")
    try:
        script = generate(**args)
    except ValueError as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script)
    output.chmod(0o755)


if __name__ == "__main__":
    main()
