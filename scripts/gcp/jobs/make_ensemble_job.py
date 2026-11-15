#!/usr/bin/env python3
"""Generate an offline, self-contained cloud job averaging completed scoring runs."""

import argparse
import ast
import base64
import gzip
import hashlib
import json
import re
import shlex
from pathlib import Path

try:
    from scripts.gcp.jobs.make_score_job import GUEST as SCORE_GUEST
    from scripts.gcp.jobs.make_score_job import _sha, _uri
except ModuleNotFoundError:  # Direct execution without the repository on sys.path.
    from make_score_job import GUEST as SCORE_GUEST
    from make_score_job import _sha, _uri

# Bundle the scoring job's actual evidence/path helpers, without changing that
# generator (its guest hash identifies existing scoring runs). No guest imports
# the generator, and preflight remains stdlib-only on the system interpreter.
_SHARED = {
    "digest",
    "atomic",
    "read",
    "read_object",
    "safe_child",
    "evidence",
    "fetch",
    "valid_checkpoint_hashes",
}
_COMMON = "\n\n".join(
    ast.get_source_segment(SCORE_GUEST, node)
    for node in ast.parse(SCORE_GUEST).body
    if isinstance(node, ast.FunctionDef) and node.name in _SHARED
)
GUEST = (
    r"""
import csv
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
"""
    + _COMMON
    + r"""

IDENTITY_FILES = ("inputs.json", "completed.json", "selected.json", "checkpoint-sha256.json")
EVALUATION_FIELDS = ("repo_sha", "upstream_pin", "tasks_sha256", "names_sha256", "max_context", "covariates")


def state_root(config):
    root = safe_child(pathlib.Path(os.environ["RUN_DIR"]), "state/ensemble/" + config["name"])
    root.mkdir(parents=True, exist_ok=True)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in ensemble state: " + str(path))
    return root


def artifacts(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p != root / "completed.json" and not p.name.endswith(".tmp")}


def refuse(uri, field, detail):
    raise ValueError("member " + uri + ": " + field + ": " + detail)


def member_json(uri, directory, filename):
    try:
        return read(directory / filename)
    except (OSError, ValueError) as error:
        refuse(uri, filename, str(error))


def checked_member(config, uri, directory):
    # Validate identity without optional scoring packages or forecasts.
    inputs = read_object(directory / "inputs.json")
    marker = read_object(directory / "completed.json")
    if type(marker.get("schema")) is not int or marker["schema"] != 1:
        refuse(uri, "schema", "expected completed.json schema 1")
    if not inputs or marker.get("inputs") != inputs:
        refuse(uri, "inputs", "completed.json differs from inputs.json")
    hashes = marker.get("artifacts")
    if not isinstance(hashes, dict) or not hashes:
        refuse(uri, "artifacts", "missing hash manifest")
    for name, value in hashes.items():
        try:
            safe_child(directory, name)
            if pathlib.PurePosixPath(name).as_posix() != name:
                raise ValueError("noncanonical artifact path")
            if any(c in name for c in "*?[]{}"):
                raise ValueError("wildcard character in artifact path: " + name)
            if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
                raise ValueError("invalid sha256")
        except (TypeError, ValueError) as error:
            refuse(uri, "artifacts", str(error))
    for name in IDENTITY_FILES:
        if name != "completed.json" and hashes.get(name) != digest(directory / name):
            refuse(uri, name, "sha256 mismatch or unlisted identity file")
    selected = member_json(uri, directory, "selected.json")
    if (not isinstance(selected, list) or not selected
            or any(not isinstance(n, str) or not n for n in selected)
            or len(set(selected)) != len(selected)):
        refuse(uri, "selected.json", "expected nonempty unique task names")
    for name in selected:
        try:
            safe_child(directory, name)
        except ValueError as error:
            refuse(uri, "selected.json", str(error))
    checkpoints = member_json(uri, directory, "checkpoint-sha256.json")
    if not valid_checkpoint_hashes(checkpoints) or marker.get("checkpoint_sha256") != checkpoints:
        refuse(uri, "checkpoint_sha256", "checkpoint-sha256.json differs or is invalid")
    for field in EVALUATION_FIELDS:
        if field != "covariates" and field not in inputs:
            refuse(uri, field, "missing evaluation field")
    if not isinstance(inputs.get("name"), str) or not inputs["name"]:
        refuse(uri, "name", "missing member name")
    for field in ("tasks_sha256", "names_sha256"):
        if inputs[field] != config[field]:
            refuse(uri, field, "differs from job inputs")
    for mode in config["modes"]:
        reports = marker.get("modes")
        report = reports.get(mode) if isinstance(reports, dict) else None
        if not isinstance(report, dict) or mode not in inputs.get("modes", []):
            refuse(uri, mode, "requested mode is missing")
        for field in ("requested", "completed"):
            if report.get(field) != selected:
                refuse(uri, mode + "." + field, "differs from selected.json")
        if report.get("failed") != {}:
            refuse(uri, mode + ".failed", "member has failures or missing failure report")
        if report.get("exit_code") != 0:
            refuse(uri, mode + ".exit_code", "member did not exit successfully")
    return {"uri": uri, "name": inputs["name"], "inputs": inputs,
            "checkpoint_sha256": checkpoints, "completed_sha256": digest(directory / "completed.json")}


def objects(prefix):
    # Native JSON includes raw GCS metadata; --format=json is not supported by ls.
    output = subprocess.check_output(["gcloud", "storage", "ls", "--json", prefix + "/**"], text=True)
    records = json.loads(output)
    if not isinstance(records, list):
        raise ValueError("object listing must be a JSON list")
    result = set()
    for record in records:
        metadata = record["metadata"]
        name, size = metadata["name"], metadata["size"]
        if (not isinstance(name, str) or not name
                or isinstance(size, bool) or not isinstance(size, (int, str))
                or not str(size).isdigit()):
            raise ValueError("object listing has invalid name or size")
        uri = "gs://" + metadata["bucket"] + "/" + name
        if not uri.startswith(prefix + "/"):
            raise ValueError("object listing is outside mode prefix: " + uri)
        if name.endswith("/") and int(size) == 0:
            continue
        result.add(uri)
    return result


def members(config, root, scratch):
    entries, directories = [], []
    for index, uri in enumerate(config["member_uris"]):
        directory = scratch / "members" / str(index)
        directory.mkdir(parents=True, exist_ok=True)
        for filename in IDENTITY_FILES:
            try:
                fetch(uri + "/" + filename, directory / filename)
            except Exception as error:
                refuse(uri, filename, "fetch failed: " + str(error))
        entry = checked_member(config, uri, directory)
        if entries:
            for field in EVALUATION_FIELDS:
                if entry["inputs"].get(field, "all" if field == "covariates" else None) != entries[0]["inputs"].get(field, "all" if field == "covariates" else None):
                    refuse(uri, field, "differs from first member")
            if read(directory / "selected.json") != read(directories[0] / "selected.json"):
                refuse(uri, "selected.json", "differs from first member")
        if any(e["name"] == entry["name"] for e in entries):
            refuse(uri, "name", "member names must differ")
        if any(e["checkpoint_sha256"]["model.safetensors"] == entry["checkpoint_sha256"]["model.safetensors"] for e in entries):
            refuse(uri, "model.safetensors", "member weight hashes must differ")
        saved = root / "members" / str(index)
        for filename in IDENTITY_FILES:
            prior = saved / filename
            if prior.exists() and digest(prior) != digest(directory / filename):
                refuse(uri, filename, "different member identity under the same ensemble name")
        entries.append(entry)
        directories.append(directory)
    # Validate all identities before fetching the large forecast folders.
    for entry, directory in zip(entries, directories):
        uri = entry["uri"]
        hashes = read(directory / "completed.json")["artifacts"]
        for mode in config["modes"]:
            expected = {uri + "/" + p for p in hashes if p.startswith(mode + "/")}
            try:
                actual = objects(uri + "/" + mode)
            except Exception as error:
                refuse(uri, mode, "object listing failed: " + str(error))
            if actual != expected:
                refuse(uri, mode + ".artifacts", "missing or unlisted files: " + str(sorted(actual ^ expected)))
            try:
                # Copy into the existing member directory: cp retains the mode's
                # leaf name. One process per mode, independent of task/file count.
                subprocess.run(["gcloud", "storage", "cp", "--recursive", uri + "/" + mode, str(directory)], check=True)
                local = {uri + "/" + str(p.relative_to(directory))
                         for p in (directory / mode).rglob("*") if p.is_file()}
                if local != expected:
                    raise ValueError("missing or unlisted downloaded files: " + str(sorted(local ^ expected)))
            except Exception as error:
                refuse(uri, mode + ".artifacts", str(error))
            for path in sorted(p for p in hashes if p.startswith(mode + "/")):
                target = safe_child(directory, path)
                try:
                    if digest(target) != hashes[path]:
                        raise ValueError("sha256 mismatch")
                except Exception as error:
                    refuse(uri, path, str(error))
            names = read(directory / "selected.json")
            failure_file = mode + "/failures.json"
            if (directory / failure_file).exists() and member_json(uri, directory, failure_file) != {}:
                refuse(uri, mode + ".failed", "saved failures.json is nonempty or invalid")
            try:
                completed = evidence(directory, mode, names, entry["name"] + "-" + mode)
            except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
                refuse(uri, mode + ".evidence", str(error))
            if completed != names:
                refuse(uri, mode + ".evidence", "incomplete summary/run/window evidence")
    # Refusing any member leaves all saved identity files untouched. Publish
    # only after every identity, listing, download hash and evidence check passes.
    for index, directory in enumerate(directories):
        saved = root / "members" / str(index)
        saved.mkdir(parents=True, exist_ok=True)
        for filename in IDENTITY_FILES:
            target = saved / (filename + ".tmp")
            shutil.copyfile(directory / filename, target)
            os.replace(target, saved / filename)
    return entries, directories


def valid_report(report, names):
    return (report["requested"] == names and report["completed"] == names
            and report["failed"] == {} and report["exit_code"] == 0
            and math.isfinite(report["wall_seconds"]) and report["wall_seconds"] >= 0)


def preflight(config, root):
    identity = root / "inputs.json"
    marker = read_object(root / "completed.json")
    recorded, marker_inputs = read_object(identity), marker.get("inputs")
    for prior in (recorded, marker_inputs):
        if isinstance(prior, dict) and prior.keys() == config.keys() and prior != config:
            raise ValueError("different inputs under the same ensemble name; choose a new name")
    if recorded != config:
        if marker_inputs != config and any(not p.name.endswith(".tmp") for p in root.iterdir()):
            raise ValueError("ensemble state has no valid input identity; refusing to overwrite")
        atomic(identity, config)
    try:
        if marker["schema"] != 1 or marker["inputs"] != config or marker["artifacts"] != artifacts(root):
            return 10
        required = {"inputs.json", "environment.txt", "tasks.yaml"}
        if config["names_uri"]:
            required.add("names.txt")
        if digest(root / "tasks.yaml") != config["tasks_sha256"]:
            return 10
        if config["names_uri"] and digest(root / "names.txt") != config["names_sha256"]:
            return 10
        entries = []
        for index, uri in enumerate(config["member_uris"]):
            directory = root / "members" / str(index)
            entries.append(checked_member(config, uri, directory))
            required.update("members/" + str(index) + "/" + p for p in IDENTITY_FILES)
        if entries != marker["members"]:
            return 10
        names = read(root / "members/0/selected.json")
        if set(marker["modes"]) != set(config["modes"]) or set(marker["parity"]) != set(config["modes"]):
            return 10
        for mode in config["modes"]:
            report = marker["modes"][mode]
            parity = read(root / "parity" / (mode + ".json"))
            if (not valid_report(report, names) or report != read(root / (mode + "-status.json"))
                    or not parity["success"] or parity != marker["parity"][mode]
                    or parity["member"] != entries[0]["uri"] or parity["tasks_compared"] != len(names)
                    or parity["columns_compared"] < 1):
                return 10
            if evidence(root, mode, names, config["name"] + "-" + mode) != names:
                return 10
            required.update({mode + "/yardstick.json", mode + "/summaries.csv", mode + "-status.json", "parity/" + mode + ".json"})
            for name in names:
                record = read(root / mode / name / "run.json")
                required.add(mode + "/" + name + "/run.json")
                required.update(mode + "/" + name + "/" + p for p in record["window_sha256"])
        return 0 if required <= marker["artifacts"].keys() else 10
    except (OSError, ValueError, KeyError, TypeError, AttributeError, csv.Error):
        return 10


def environment(config, root, python):
    from importlib import metadata
    for name, expected in (("fev", "0.10.0"), ("datasets", "3.6.0"), ("huggingface-hub", "0.36.0")):
        if metadata.version(name) != expected:
            raise ValueError(name + " must remain pinned to " + expected)
    from demand_forecast.bench.yardsticks.ensemble import run_ensemble
    facts = {"python": sys.version.split()[0], "repo_sha": config["repo_sha"],
             "ensemble_import": run_ensemble.__module__, "simons_required": False,
             "decision": "CPU scoring only; no inference packages, model weights or CUDA required"}
    freeze = subprocess.check_output([python, "-m", "pip", "freeze"], text=True)
    temporary = root / "environment.txt.tmp"
    temporary.write_text(freeze + "\n# runtime " + json.dumps(facts, sort_keys=True) + "\n")
    os.replace(temporary, root / "environment.txt")


def summaries(path, label, names):
    rows = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row.get("model_name") == label:
                name = row["task"]
                if name in rows:
                    raise ValueError("duplicate summary row: " + name)
                rows[name] = row
    if set(rows) != set(names):
        raise ValueError("summary task selection differs")
    return rows


def compare_metrics(source, output, names, metrics):
    failures, columns, largest, count = [], set(), 0.0, 0
    for name in names:
        shared = source[name].keys() & output[name].keys()
        selected = sorted(c for c in shared if c == "test_error" or c.split("[", 1)[0] in metrics)
        if not selected:
            raise ValueError("no shared metric columns for " + name)
        for column in selected:
            old, new = source[name][column], output[name][column]
            # pandas writes undefined float metrics as empty CSV cells.
            a, b = float(old) if old else math.nan, float(new) if new else math.nan
            equal = (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
            difference = abs(a - b) if math.isfinite(a) and math.isfinite(b) else 0.0
            largest = max(largest, difference)
            columns.add(column)
            count += 1
            if not equal:
                failures.append({"task": name, "column": column, "recorded": old or "NaN", "recomputed": new or "NaN"})
    return {"success": not failures, "tasks_compared": len(names), "columns_compared": len(columns),
            "values_compared": count, "columns": sorted(columns), "largest_difference": largest,
            "differences": failures, "rtol": 1e-9, "atol": 1e-12}


def command(python, repo, root, config, directories, label, output):
    args = [python, str(repo / "scripts/fev_ensemble.py"), "--members", *map(str, directories),
            "--tasks", str(root / "tasks.yaml"), "--model-name", label,
            "--out-dir", str(output), "--keep-going"]
    if config["names_uri"]:
        args += ["--names", "@" + str(root / "names.txt")]
    return args


def parity(config, root, scratch, repo, python, mode, entry, directory, names):
    from fev.metrics import AVAILABLE_METRICS
    label = config["name"] + "-parity-" + mode
    output = scratch / ("parity-" + mode)
    args = command(python, repo, root, config, [directory, directory], label, output)
    args += ["--allow-identical-members"]
    path = root / "parity" / (mode + ".json")
    path.parent.mkdir(exist_ok=True)
    report = {"success": False, "member": entry["uri"], "member_name": entry["name"]}
    try:
        status = subprocess.run(args, check=False).returncode
        if status or evidence(scratch, output.name, names, label) != names:
            raise ValueError("self-ensemble did not complete; exit " + str(status))
        original = summaries(directory / "summaries.csv", entry["name"] + "-" + mode, names)
        recomputed = summaries(output / "summaries.csv", label, names)
        report.update(compare_metrics(original, recomputed, names, AVAILABLE_METRICS))
    except Exception as error:
        report.update(success=False, error=str(error))
    atomic(path, report)
    if not report["success"]:
        raise ValueError("scorer parity failed for " + mode + "; see " + str(path))
    return report


def recover_mode(root, mode):
    directory = safe_child(root, mode)
    if not directory.exists() or (directory / "yardstick.json").exists():
        return
    # run_ensemble initializes its manifest with NamedTemporaryFile(suffix=".json").
    # A process kill can leave this file before replace, or just an empty folder.
    # Do not infer ownership of arbitrary JSON files or directories.
    children = list(directory.iterdir())
    if any(p.is_symlink() or not p.is_file()
           or not (p.name.endswith(".tmp") or re.fullmatch(r"tmp[a-z0-9_]{8}\.json", p.name))
           for p in children):
        raise ValueError("mode " + mode + " has files without a published yardstick.json manifest; refusing to repair " + str(directory))
    shutil.rmtree(directory)


def average(config, root, scratch, repo, python):
    from demand_forecast.bench.yardsticks.runner import load_tasks, read_names
    for key, filename in (("tasks", "tasks.yaml"), ("names", "names.txt")):
        if config[key + "_uri"]:
            target = scratch / filename
            fetch(config[key + "_uri"], target)
            if digest(target) != config[key + "_sha256"]:
                raise ValueError(key + " sha256 mismatch")
            os.replace(target, root / filename)
    selection = "@" + str(root / "names.txt") if config["names_uri"] else None
    if selection and any("," in line or not line.strip() for line in (root / "names.txt").read_text().splitlines()):
        raise ValueError("names file must contain one task name per line")
    names = [task.task_name for task in load_tasks(root / "tasks.yaml", read_names(selection))]
    if not names or len(set(names)) != len(names):
        raise ValueError("tasks must be nonempty and unique")
    for name in names:
        safe_child(root, name)
    # Member directory spellings are part of ensemble identities. Keep them
    # stable across RUN_DIR/guest labels, as scoring does for checkpoint paths.
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    base = safe_child(pathlib.Path("/tmp"), "demand-forecast-ensemble-" + identity)
    if base.exists():
        for p in base.rglob("*"):
            if p.is_symlink():
                raise ValueError("symlink in member scratch: " + str(p))
        shutil.rmtree(base)
    base.mkdir(mode=0o700)
    try:
        entries, directories = members(config, root, base)
        if read(directories[0] / "selected.json") != names:
            refuse(entries[0]["uri"], "selected.json", "differs from job task selection")
        # Sort exactly as run_ensemble does (lexicographic directory spelling).
        first = min(range(len(directories)), key=lambda i: str(directories[i]))
        parity_reports = {}
        # Every parity gate passes before *any* real averaging is started.
        for mode in config["modes"]:
            parity_reports[mode] = parity(config, root, scratch, repo, python, mode,
                                          entries[first], directories[first] / mode, names)
        reports = {}
        for mode in config["modes"]:
            recover_mode(root, mode)
            label = config["name"] + "-" + mode
            args = command(python, repo, root, config, [d / mode for d in directories], label, root / mode)
            start = time.monotonic()
            log = root / (mode + ".log")
            with log.open("w") as stream:
                status = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT).returncode
            with log.open(errors="replace") as stream:
                message = next((line.strip() for line in stream if line.strip()), "no ensemble output")
            try:
                completed = evidence(root, mode, names, label)
                failure_path = root / mode / "failures.json"
                failures = read(failure_path) if failure_path.exists() else {}
                if not isinstance(failures, dict) or any(not isinstance(v, dict) for v in failures.values()):
                    raise ValueError("invalid ensemble failures.json")
            except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
                completed, failures, status = [], {}, status or 2
                message = "ensemble evidence invalid: " + str(error)
            failed = {n: (str(failures.get(n, {}).get("message", "ensemble exit " + str(status) + ": " + message)).splitlines()
                          or ["task incomplete"])[0]
                      for n in names if n in failures or n not in completed}
            report = {"requested": names, "completed": [n for n in completed if n not in failed],
                      "failed": failed, "exit_code": status, "wall_seconds": time.monotonic() - start}
            reports[mode] = report
            atomic(root / (mode + "-status.json"), report)
        atomic(root / "completed.json", {"schema": 1, "inputs": config, "members": entries,
                                         "parity": parity_reports, "modes": reports, "artifacts": artifacts(root)})
        return int(any(r["failed"] or r["exit_code"] for r in reports.values()))
    finally:
        shutil.rmtree(base)


def main():
    action, config_file, scratch_dir, repo_dir, python = sys.argv[1:]
    config, scratch, repo = read(pathlib.Path(config_file)), pathlib.Path(scratch_dir), pathlib.Path(repo_dir)
    root = state_root(config)
    if action == "preflight":
        return preflight(config, root)
    if action == "environment":
        environment(config, root, python)
        return 0
    if action == "average":
        return average(config, root, scratch, repo, python)
    raise ValueError("unknown guest action: " + action)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        sys.exit("Ensemble job failed: " + " ".join(str(error).splitlines()))
"""
)


def generate(
    *,
    name,
    repo_sha,
    tasks_uri,
    tasks_sha256,
    member_uris,
    names_uri=None,
    names_sha256=None,
    processing="both",
):
    """Validate and bundle the job without contacting code, cloud state or data."""
    _sha(repo_sha, 40, "repo SHA")
    _sha(tasks_sha256, 64, "tasks sha256")
    _uri(tasks_uri)
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError(
            "name must start with a letter or digit and contain only letters, digits, underscore or hyphen"
        )
    if (names_uri is None) != (names_sha256 is None):
        raise ValueError("names URI and sha256 are required together")
    if names_uri is not None:
        _uri(names_uri)
        _sha(names_sha256, 64, "names sha256")
    if not isinstance(member_uris, (list, tuple)) or len(member_uris) < 2:
        raise ValueError("at least two member URIs are required")
    normalized = []
    for uri in member_uris:
        _uri(uri)
        uri = uri.rstrip("/")
        if not re.search(r"/score/[A-Za-z0-9][A-Za-z0-9_-]*$", uri):
            raise ValueError("member URI must end in /score/<name>")
        normalized.append(uri)
    if len(set(normalized)) != len(normalized):
        raise ValueError("member URIs must be pairwise different")
    if processing not in ("raw", "published", "both"):
        raise ValueError("processing must be raw, published or both")
    guest_sha = hashlib.sha256(GUEST.encode()).hexdigest()
    config = dict(
        name=name,
        repo_sha=repo_sha,
        tasks_uri=tasks_uri,
        tasks_sha256=tasks_sha256,
        names_uri=names_uri,
        names_sha256=names_sha256,
        member_uris=normalized,
        modes=["raw", "published"] if processing == "both" else [processing],
        guest_sha256=guest_sha,
    )
    encoded_config = base64.b64encode(json.dumps(config, sort_keys=True).encode()).decode()
    encoded_guest = base64.b64encode(gzip.compress(GUEST.encode(), mtime=0)).decode()
    return f'''#!/usr/bin/env bash
# Forecast ensemble {name}; repository {repo_sha}
set -euo pipefail
umask 077
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export STATE_URI}}"
case "$RUN_DIR" in /*) ;; *) echo "RUN_DIR must be absolute" >&2; exit 2 ;; esac
export RUN_DIR
export HOME="${{HOME:-$RUN_DIR/home}}"
export HF_HOME="$RUN_DIR/ensemble-cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE HF_TOKEN HUGGING_FACE_HUB_TOKEN
export UV_CACHE_DIR="$RUN_DIR/ensemble-cache/uv"
export UV_PYTHON_INSTALL_DIR="$RUN_DIR/ensemble-cache/python"
mkdir -p "$RUN_DIR"
scratch=$(mktemp -d "$RUN_DIR/ensemble-install-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
python3 - "$scratch" <<'ENSEMBLE_BUNDLE'
import base64, gzip, hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
content = gzip.decompress(base64.b64decode("{encoded_guest}"))
assert hashlib.sha256(content).hexdigest() == "{guest_sha}"
(root / "ensemble.py").write_bytes(content)
(root / "inputs.json").write_bytes(base64.b64decode("{encoded_config}"))
ENSEMBLE_BUNDLE
repo="$scratch/repository"
python="$scratch/venv/bin/python"
guest() {{
    "$1" "$scratch/ensemble.py" "$2" "$scratch/inputs.json" "$scratch" "$repo" "$python"
}}
status=0
guest python3 preflight || status=$?
if [[ $status == 0 ]]; then exit 0; fi
if [[ $status != 10 ]]; then exit "$status"; fi
mkdir -p "$HOME" "$HF_DATASETS_CACHE" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
bootstrap_uv() {{
    if [[ ! -x $scratch/uv-bootstrap/bin/uv ]]; then
        python3 -m pip install --target "$scratch/uv-bootstrap" uv
    fi
}}
if ! python3 -m venv "$scratch/venv"; then
    bootstrap_uv
    "$scratch/uv-bootstrap/bin/uv" venv --clear --seed \\
        --python "$(command -v python3)" --no-python-downloads "$scratch/venv"
fi
if [[ $("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])') != 3.12 ]]; then
    bootstrap_uv
    "$scratch/uv-bootstrap/bin/uv" venv --clear --seed --python 3.12 "$scratch/venv"
fi
[[ $("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])') == 3.12 ]] || {{ echo 'Python 3.12 required' >&2; exit 2; }}
git init "$repo"
git -C "$repo" remote add origin https://github.com/alessiosavi/demand-forecast.git
git -C "$repo" fetch --depth=1 origin {shlex.quote(repo_sha)}
git -C "$repo" checkout --detach FETCH_HEAD
[[ $(git -C "$repo" rev-parse HEAD) == {shlex.quote(repo_sha)} ]] || {{ echo 'repository SHA mismatch' >&2; exit 2; }}
"$python" -m pip install --upgrade pip
# Constraints keep the numerical scoring stack identical, without installing
# CUDA wheels or the optional inference adapters in requirements-yardsticks.
"$python" -m pip install -c "$repo/requirements-yardsticks.txt" \\
    fev datasets huggingface-hub numpy pandas pyarrow scipy scikit-learn pyyaml pydantic pydantic-settings
"$python" -m pip check
# Register repository metadata/imports without pulling in training/inference
# extras. Their declared dependencies intentionally are not this job's runtime.
"$python" -m pip install --no-deps --editable "$repo"
export PYTHONPATH="$repo${{PYTHONPATH:+:$PYTHONPATH}}"
guest "$python" environment
guest "$python" average
'''


def main():
    """Write one executable job; generation is entirely offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("name", "repo-sha", "tasks-uri", "tasks-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--member-uri", dest="member_uris", action="append", required=True)
    parser.add_argument("--names-uri")
    parser.add_argument("--names-sha256")
    parser.add_argument("--processing", choices=["raw", "published", "both"], default="both")
    parser.add_argument("--output", type=Path, required=True)
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
