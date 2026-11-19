#!/usr/bin/env python3
"""Generate a CPU guest job for Stage D's history-only diagnostics."""

import argparse
import ast
import base64
import gzip
import hashlib
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile


def _name_validator_source():
    # Preflight and resume run before installation. Bundle the same validator
    # used by the diagnostics without importing the bench dependencies.
    path = Path(__file__).resolve().parents[3] / "demand_forecast/bench/static_diagnostics.py"
    source = path.read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_safe_name":
            return ast.get_source_segment(source, node)
    raise RuntimeError("missing diagnostics task name validator")


GUEST = (
    _name_validator_source()
    + "\n\n"
    + r"""
import csv
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPOSITORY = "https://github.com/alessiosavi/demand-forecast.git"
REQUIRED_REPORTS = {"attributes.csv", "tasks.csv", "level.json"}
INPUTS = {"tasks": "tasks.yaml", "names": "names.txt"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic(path, value):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = pathlib.Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def state(config):
    run = pathlib.Path(os.environ["RUN_DIR"])
    root = run / "state" / "diagnose" / config["name"]
    if not run.is_absolute() or any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("diagnostic state requires an absolute RUN_DIR without symlinks")
    root.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("symlink in diagnostic state")
    uri = os.environ["STATE_URI"].rstrip("/")
    if (not re.fullmatch(r"gs://[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9_./:@+=-]+)?", uri)
            or any(part in ("", ".", "..") for part in uri[5:].split("/"))):
        raise ValueError("invalid diagnostic STATE_URI")
    destination = uri + "/diagnose/" + config["name"]
    for source in (config["tasks_uri"], config["names_uri"]):
        if source and (source == destination or source.startswith(destination + "/")):
            raise ValueError("diagnostic destination overlaps an input")
    return root, destination


def report_names(directory):
    with (directory / "tasks.csv").open(encoding="utf-8", newline="") as stream:
        tasks = {row["task"] for row in csv.DictReader(stream)}
    reports = REQUIRED_REPORTS.copy()
    for name in tasks:
        _safe_name(name)
        reports.add(name + ".json")
    if not tasks or len(reports) != len(tasks) + len(REQUIRED_REPORTS):
        raise ValueError("diagnostic tasks.csv lacks distinct report paths")
    return reports


def input_names(config):
    return {"inputs.json", *(filename for key, filename in INPUTS.items() if config[key + "_uri"])}


def completed(config, root):
    path = root / "completed.json"
    if not path.exists():
        return False
    try:
        marker = read(path)
        files = report_names(root) | input_names(config)
        hashes = marker["artifacts"]
        if (type(marker["schema"]) is int and marker["schema"] == 1
                and marker["inputs"] == config and set(hashes) == files
                and read(root / "inputs.json") == config
                and all(hashes[name] == digest(root / name) for name in files)
                and all(hashes[filename] == config[key + "_sha256"]
                        for key, filename in INPUTS.items() if config[key + "_uri"])):
            return True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    raise ValueError("damaged diagnostic completed unit; restore its artifacts or use a new name")


def copy(source, destination):
    subprocess.run(["gcloud", "storage", "cp", str(source), str(destination)], check=True)


def fetch(config, scratch):
    for key, filename in INPUTS.items():
        uri = config[key + "_uri"]
        if uri:
            target = scratch / filename
            copy(uri, target)
            if digest(target) != config[key + "_sha256"]:
                raise ValueError(key + " sha256 mismatch")


def install(config, scratch):
    repo = scratch / "repository"
    subprocess.run(["git", "init", str(repo)], check=True)
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "remote", "add", "origin", REPOSITORY], check=True)
    subprocess.run([*git, "fetch", "--depth=1", "origin", config["repo_sha"]], check=True)
    subprocess.run([*git, "checkout", "--detach", "FETCH_HEAD"], check=True)
    if subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != config["repo_sha"]:
        raise ValueError("repository SHA mismatch")

    uv = shutil.which("uv")
    if uv is None:
        bootstrap = scratch / "uv-bootstrap"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", str(bootstrap), "uv"], check=True
        )
        uv = str(bootstrap / "bin" / "uv")
    venv = scratch / "venv"
    python = venv / "bin" / "python"
    subprocess.run([uv, "venv", "--python", "3.12", str(venv)], check=True)
    pins = scratch / "pins.txt"
    requirements = []
    torch = None
    for line in (repo / "requirements-yardsticks.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "chronos-forecasting==", "timestral @")):
            continue
        if line.startswith("torch=="):
            torch = line + "+cpu"
            line = torch
        requirements.append(line)
    if torch is None:
        raise ValueError("missing torch pin in requirements-yardsticks.txt")
    pins.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    pip = [uv, "pip", "install", "--python", str(python)]
    subprocess.run([*pip, "--index", "https://download.pytorch.org/whl/cpu", torch], check=True)
    subprocess.run([*pip, "-r", str(pins), "-e", f"{repo}[bench]"], check=True)
    subprocess.run([uv, "pip", "check", "--python", str(python)], check=True)
    subprocess.run(
        [str(python), "-c",
         "import sys, torch; assert sys.version_info[:2] == (3, 12); "
         "assert torch.version.cuda is None and torch.__version__ == sys.argv[1]",
         torch.partition("==")[2]],
        check=True,
    )
    return python


def run(config, root, destination, scratch):
    if completed(config, root):
        return
    identity = root / "inputs.json"
    if identity.exists() and read(identity) != config:
        raise ValueError("different inputs under the same diagnostic name; choose a new name")
    if not identity.exists() and any(root.iterdir()):
        raise ValueError("diagnostic state lacks input identity; refusing to overwrite it")
    fetch(config, scratch)
    if not identity.exists():
        atomic(identity, config)
    python = install(config, scratch)
    reports = scratch / "reports"
    command = [
        str(python), "-m", "demand_forecast.bench.static_diagnostics",
        "--tasks", str(scratch / "tasks.yaml"), "--out", str(reports),
    ]
    if config["names_uri"]:
        command.extend(["--names", "@" + str(scratch / "names.txt")])
    subprocess.run(command, check=True)

    artifacts = {name: reports / name for name in report_names(reports)}
    artifacts.update((name, scratch / name) for name in input_names(config) - {"inputs.json"})
    hashes = {name: digest(path) for name, path in artifacts.items()}
    hashes["inputs.json"] = digest(identity)
    copy(identity, destination + "/inputs.json")
    for name, source in sorted(artifacts.items()):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        copy(target, destination + "/" + name)
    marker = scratch / "completed.json"
    atomic(marker, {"schema": 1, "inputs": config, "artifacts": hashes})
    copy(marker, destination + "/completed.json")
    marker.replace(root / "completed.json")


def main():
    scratch = pathlib.Path(sys.argv[1])
    config = read(scratch / "inputs.json")
    root, destination = state(config)
    lock = root.parents[2] / (".diagnose-" + config["name"] + ".lock")
    if lock.is_symlink():
        raise ValueError("symlink in diagnostic lock")
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        run(config, root, destination, scratch)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.exit("Diagnostic job failed: " + " ".join(str(error).splitlines()))
"""
)


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


def generate(*, name, repo_sha, tasks_uri, tasks_sha256, names_uri=None, names_sha256=None):
    """Return the guest shell script without contacting any external service."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError(
            "name must start with a letter or digit and contain only letters, digits, _ or -"
        )
    _sha(repo_sha, 40, "repo SHA")
    _sha(tasks_sha256, 64, "tasks sha256")
    _uri(tasks_uri)
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
        guest_sha256=guest_sha,
    )
    encoded_config = base64.b64encode(json.dumps(config, sort_keys=True).encode()).decode()
    encoded_guest = base64.b64encode(gzip.compress(content, mtime=0)).decode()
    return f'''#!/usr/bin/env bash
# Stage D diagnostics {name}; repository {repo_sha}; CPU only
set -euo pipefail
umask 077
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export STATE_URI}}"
case "$RUN_DIR" in /*) ;; *) echo 'RUN_DIR must be absolute' >&2; exit 2 ;; esac
export RUN_DIR STATE_URI
export HF_HOME="$RUN_DIR/diagnose-cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export UV_CACHE_DIR="$RUN_DIR/diagnose-cache/uv"
export UV_PYTHON_INSTALL_DIR="$RUN_DIR/diagnose-cache/python"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE HF_TOKEN HUGGING_FACE_HUB_TOKEN
mkdir -p "$RUN_DIR"
scratch=$(mktemp -d "$RUN_DIR/diagnose-install-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
python3 - "$scratch" <<'DIAGNOSE_BUNDLE'
import base64, gzip, hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
content = gzip.decompress(base64.b64decode("{encoded_guest}"))
if hashlib.sha256(content).hexdigest() != "{guest_sha}":
    raise SystemExit("diagnostic guest bundle sha256 mismatch")
(root / "diagnose.py").write_bytes(content)
(root / "inputs.json").write_bytes(base64.b64decode("{encoded_config}"))
DIAGNOSE_BUNDLE
python3 "$scratch/diagnose.py" "$scratch"
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("name", "repo-sha", "tasks-uri", "tasks-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--names-uri")
    parser.add_argument("--names-sha256")
    parser.add_argument("--output", required=True, type=Path)
    args = vars(parser.parse_args())
    output = args.pop("output")
    try:
        script = generate(**args)
    except ValueError as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(script)
            stream.close()
            temporary.chmod(0o755)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
