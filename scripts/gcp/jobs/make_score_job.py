#!/usr/bin/env python3
"""Generate an offline, self-contained Simons fev scoring job for the GCP guest."""

import argparse
import base64
import gzip
import hashlib
import json
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# This payload runs on the guest. Preflight uses only the system Python stdlib;
# optional imports occur only after the pinned scoring environment is ready.
GUEST = r"""
import csv
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time


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
    parts = pathlib.PurePosixPath(name).parts
    if not parts or name.startswith("/") or any(p in (".", "..") for p in name.split("/")) or "\\" in name:
        raise ValueError("unsafe evidence/task path: " + name)
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink in score state: " + str(path))
    return path


def state_root(config):
    run = pathlib.Path(os.environ["RUN_DIR"])
    root = safe_child(run, "state/score/" + config["name"])
    root.mkdir(parents=True, exist_ok=True)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in score state: " + str(path))
    return root


def evidence(root, mode, names, label):
    directory = root / mode
    rows = {}
    summary = directory / "summaries.csv"
    if summary.exists():
        with summary.open() as stream:
            for row in csv.DictReader(stream):
                if row.get("model_name") == label:
                    name = row.get("task")
                    if name in rows:
                        raise ValueError("duplicate task summary: " + name)
                    rows[name] = row
    completed = []
    for name in names:
        task = safe_child(directory, name)
        try:
            record = read(task / "run.json")
            windows = record["window_sha256"]
            valid = (name in rows and record.get("summary") is not None
                     and rows[name].get("run_identity") == record["run_identity"]
                     and bool(windows)
                     and all(digest(safe_child(task, p)) == h for p, h in windows.items()))
            if valid:
                completed.append(name)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return completed


def preflight(config, root):
    identity = root / "inputs.json"
    marker_path = root / "completed.json"
    marker = None
    if marker_path.exists():
        try:
            marker = read(marker_path)
        except (ValueError, OSError):
            pass  # A partial remote copy is repaired using the durable identity.
        if isinstance(marker, dict) and marker.get("inputs") != config:
            raise ValueError("different inputs under the same score name; choose a new name")
    if identity.exists():
        if read(identity) != config:
            raise ValueError("different inputs under the same score name; choose a new name")
    else:
        if not isinstance(marker, dict) and any(
                p.name != "completed.json" and not p.name.endswith(".tmp") for p in root.iterdir()):
            raise ValueError("score state has no input identity; refusing to overwrite")
        # A matching completion marker carries the full identity independently;
        # remote copies can deliver it and task evidence before inputs.json.
        atomic(identity, config)
    try:
        if marker["schema"] != 1 or marker["inputs"] != config:
            return 10
        hashes = marker["artifacts"]
        if not hashes or not all(digest(safe_child(root, p)) == h for p, h in hashes.items()):
            return 10
        required = {"inputs.json", "environment.txt", "checkpoint-sha256.json", "tasks.yaml", "selected.json"}
        if config["names_uri"]:
            required.add("names.txt")
        if not required <= hashes.keys():
            return 10
        if digest(root / "tasks.yaml") != config["tasks_sha256"]:
            return 10
        if config["names_uri"] and digest(root / "names.txt") != config["names_sha256"]:
            return 10
        names = read(root / "selected.json")
        if not names or marker["checkpoint_sha256"] != read(root / "checkpoint-sha256.json"):
            return 10
        if set(marker["modes"]) != set(config["modes"]):
            return 10
        for mode in config["modes"]:
            report = marker["modes"][mode]
            if (report["requested"] != names or report["completed"] != names or report["failed"]
                    or report["exit_code"] != 0 or not math.isfinite(report["wall_seconds"])
                    or report["wall_seconds"] < 0):
                return 10
            if evidence(root, mode, names, config["name"] + "-" + mode) != names:
                return 10
            for name in names:
                record = read(root / mode / name / "run.json")
                required.add(mode + "/" + name + "/run.json")
                required.update(mode + "/" + name + "/" + p for p in record["window_sha256"])
            required.add(mode + "/summaries.csv")
        return 0 if required <= hashes.keys() else 10
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return 10


def image(scratch):
    import torch
    atomic(scratch / "image.json", {"python": sys.version.split()[0], "torch": torch.__version__,
                                  "torch_path": str(pathlib.Path(torch.__file__).resolve()),
                                  "cuda": torch.version.cuda})


def environment(config, root, scratch, python):
    from importlib import metadata
    import torch
    for name, expected in (("fev", "0.10.0"), ("datasets", "3.6.0"), ("huggingface-hub", "0.36.0")):
        if metadata.version(name) != expected:
            raise ValueError(name + " must remain pinned to " + expected)
    upstream = json.loads(metadata.distribution("simons").read_text("direct_url.json"))
    if upstream["vcs_info"]["commit_id"] != config["upstream_pin"]:
        raise ValueError("installed Simons commit does not match upstream pin")
    if torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8":
        raise ValueError("scoring requires stage-1 torch 2.8.0+cu128 / CUDA 12.8")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    drivers = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    if not drivers:
        raise ValueError("NVIDIA driver version is unavailable")
    facts = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
             "cuda_available": True, "nvidia_driver": drivers, "image": read(scratch / "image.json"),
             "repo_sha": config["repo_sha"], "upstream_pin": config["upstream_pin"],
             "decision": "Python 3.12 provisioned for stage-1 pins; scipy 1.16.3 requires >=3.11; image cp310 torch cannot be reused"}
    freeze = subprocess.check_output([python, "-m", "pip", "freeze"], text=True)
    path = scratch / "environment.txt"
    path.write_text(freeze + "\n# runtime " + json.dumps(facts, sort_keys=True) + "\n")
    os.replace(path, root / "environment.txt")


def fetch(uri, target):
    subprocess.run(["gcloud", "storage", "cp", uri, str(target)], check=True)


def inputs(config, root, scratch):
    for key, filename in (("tasks", "tasks.yaml"), ("names", "names.txt")):
        if not config[key + "_uri"]:
            continue
        target = scratch / filename
        fetch(config[key + "_uri"], target)
        if digest(target) != config[key + "_sha256"]:
            raise ValueError(key + " sha256 mismatch")
        os.replace(target, root / filename)
    download = scratch / "checkpoint"
    download.mkdir()
    for filename in ("config.json", "model.safetensors"):
        if config["checkpoint_uri"]:
            fetch(config["checkpoint_uri"].rstrip("/") + "/" + filename, download / filename)
        else:
            from huggingface_hub import hf_hub_download
            hf_hub_download(config["hub_weights"], filename, revision=config["hub_revision"],
                            token=False, local_dir=str(download))
        if not (download / filename).stat().st_size:
            raise ValueError("empty checkpoint file: " + filename)
    hashes = {name: digest(download / name) for name in ("config.json", "model.safetensors")}
    prior = root / "checkpoint-sha256.json"
    if prior.exists() and read(prior) != hashes:
        raise ValueError("different checkpoint contents under the same score name")
    atomic(prior, hashes)
    # The adapter hashes its absolute checkpoint path in the resume identity.
    # A fixed disposable path survives a new guest label/RUN_DIR, unlike mktemp.
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    base = pathlib.Path("/tmp") / ("demand-forecast-score-" + identity)
    checkpoint = safe_child(base.parent, base.name + "/checkpoint")
    checkpoint.mkdir(parents=True, mode=0o700, exist_ok=True)
    for filename in hashes:
        target = safe_child(checkpoint, filename)
        temporary = checkpoint / (filename + ".tmp")
        if temporary.is_symlink():
            raise ValueError("symlink in checkpoint scratch")
        shutil.copyfile(download / filename, temporary)
        os.replace(temporary, target)
    return checkpoint, hashes


def score(config, root, scratch, repo, python):
    from demand_forecast.bench.yardsticks.runner import load_tasks, read_names
    checkpoint, hashes = inputs(config, root, scratch)
    selection = "@" + str(root / "names.txt") if config["names_uri"] else None
    if selection:
        lines = (root / "names.txt").read_text().splitlines()
        if any("," in line or not line.strip() for line in lines):
            raise ValueError("names file must contain one task name per line")
    tasks = load_tasks(root / "tasks.yaml", read_names(selection))
    names = [task.task_name for task in tasks]
    if not names or len(set(names)) != len(names):
        raise ValueError("tasks must be nonempty and unique")
    for name in names:
        safe_child(root, name)
    atomic(root / "selected.json", names)
    reports = {}
    for mode in config["modes"]:
        directory = root / mode
        command = [python, str(repo / "scripts/fev_yardstick.py"), "--model", "simons",
                   "--checkpoint", str(checkpoint), "--processing", mode,
                   "--tasks", str(root / "tasks.yaml"), "--model-name", config["name"] + "-" + mode,
                   "--out-dir", str(directory), "--resume", "--keep-going"]
        if selection:
            command += ["--names", selection]
        if config["max_context"] is not None:
            command += ["--max-context", str(config["max_context"])]
        start = time.monotonic()
        log = root / (mode + ".log")
        with log.open("w") as stream:
            status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
        with log.open(errors="replace") as stream:
            first_line = next((line.strip() for line in stream if line.strip()), "no scorer output")
        try:
            completed = evidence(root, mode, names, config["name"] + "-" + mode)
            failure_path = directory / "failures.json"
            failures = read(failure_path) if failure_path.exists() else {}
            if not isinstance(failures, dict) or any(not isinstance(v, dict) for v in failures.values()):
                raise ValueError("invalid scorer failures.json")
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
            # A damaged restored report must not prevent attempting the other mode.
            completed, failures = [], {}
            status = status or 2
            first_line = "scorer evidence invalid: " + str(error).splitlines()[0] + "; " + first_line
        failed = {}
        for name in names:
            if name in failures or name not in completed:
                message = failures.get(name, {}).get("message", "scorer exit " + str(status) + ": " + first_line)
                failed[name] = str(message).splitlines()[0] if str(message).splitlines() else "task incomplete"
        completed = [name for name in completed if name not in failed]
        report = {"requested": names, "completed": completed, "failed": failed,
                  "wall_seconds": time.monotonic() - start, "exit_code": status,
                  "checkpoint_sha256": hashes, "repo_sha": config["repo_sha"],
                  "upstream_pin": config["upstream_pin"], "tasks_sha256": config["tasks_sha256"]}
        reports[mode] = report
        atomic(root / (mode + "-status.json"), report)
    artifacts = {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*"))
                 if path.is_file() and path.name != "completed.json" and not path.name.endswith(".tmp")}
    atomic(root / "completed.json", {"schema": 1, "inputs": config, "checkpoint_sha256": hashes,
                                     "modes": reports, "artifacts": artifacts})
    return int(any(report["failed"] or report["exit_code"] for report in reports.values()))


def main():
    action, config_file, scratch_dir, repo_dir, python = sys.argv[1:]
    config, scratch, repo = read(pathlib.Path(config_file)), pathlib.Path(scratch_dir), pathlib.Path(repo_dir)
    root = state_root(config)
    if action == "preflight":
        return preflight(config, root)
    if action == "image":
        image(scratch)
    elif action == "environment":
        environment(config, root, scratch, python)
    elif action == "score":
        return score(config, root, scratch, repo, python)
    else:
        raise ValueError("unknown guest action")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        sys.exit("Score job failed: " + " ".join(str(error).splitlines()))
"""


def _sha(value, length, label):
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"{label} must be full {length}-character lowercase hex")


def _uri(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"gs://[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9_./:@+=-]+", value
    ):
        raise ValueError("URI must name an explicit gs:// bucket object or prefix")
    if any(part in (".", "..") for part in value[5:].split("/")):
        raise ValueError("URI cannot contain dot path components")


def generate(
    *,
    name,
    repo_sha,
    upstream_pin,
    tasks_uri,
    tasks_sha256,
    checkpoint_uri=None,
    hub_weights=None,
    hub_revision=None,
    names_uri=None,
    names_sha256=None,
    processing="both",
    max_context=None,
):
    """Validate inputs and return a shell job, without fetching code, data or cloud state."""
    _sha(repo_sha, 40, "repo SHA")
    _sha(upstream_pin, 40, "upstream pin")
    if upstream_pin != (ROOT / "third_party/simons/PINNED").read_text().strip():
        raise ValueError("upstream pin must equal third_party/simons/PINNED")
    _sha(tasks_sha256, 64, "tasks sha256")
    _uri(tasks_uri)
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError(
            "name must start with a letter or digit and contain only letters, digits, underscore or hyphen"
        )
    if bool(checkpoint_uri) == bool(hub_weights):
        raise ValueError("choose exactly one checkpoint URI or Hub weights repository")
    if checkpoint_uri:
        _uri(checkpoint_uri)
        if hub_revision is not None:
            raise ValueError("Hub revision requires Hub weights")
    else:
        if not isinstance(hub_weights, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?", hub_weights
        ):
            raise ValueError("invalid Hub repository ID")
        _sha(hub_revision, 40, "Hub revision")
    if (names_uri is None) != (names_sha256 is None):
        raise ValueError("names URI and sha256 are required together")
    if names_uri is not None:
        _uri(names_uri)
        _sha(names_sha256, 64, "names sha256")
    if processing not in ("raw", "published", "both"):
        raise ValueError("processing must be raw, published or both")
    if max_context is not None and (type(max_context) is not int or max_context < 1):
        raise ValueError("max-context must be a positive integer")
    payload = gzip.compress(GUEST.encode(), mtime=0)
    guest_sha = hashlib.sha256(payload).hexdigest()
    config = dict(
        name=name,
        repo_sha=repo_sha,
        upstream_pin=upstream_pin,
        tasks_uri=tasks_uri,
        tasks_sha256=tasks_sha256,
        checkpoint_uri=checkpoint_uri.rstrip("/") if checkpoint_uri else None,
        hub_weights=hub_weights,
        hub_revision=hub_revision,
        names_uri=names_uri,
        names_sha256=names_sha256,
        modes=["raw", "published"] if processing == "both" else [processing],
        max_context=max_context,
        guest_sha256=guest_sha,
    )
    encoded_config = base64.b64encode(json.dumps(config, sort_keys=True).encode()).decode()
    encoded_guest = base64.b64encode(payload).decode()
    return f'''#!/usr/bin/env bash
# Simons scoring {name}; repository {repo_sha}; upstream {upstream_pin}
set -euo pipefail
umask 077
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export STATE_URI}}"
case "$RUN_DIR" in /*) ;; *) echo "RUN_DIR must be absolute" >&2; exit 2 ;; esac
export RUN_DIR
export HOME="${{HOME:-$RUN_DIR/home}}"
export HF_HOME="$RUN_DIR/score-cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE HF_TOKEN HUGGING_FACE_HUB_TOKEN
export UV_CACHE_DIR="$RUN_DIR/score-cache/uv"
export UV_PYTHON_INSTALL_DIR="$RUN_DIR/score-cache/python"
mkdir -p "$RUN_DIR"
scratch=$(mktemp -d "$RUN_DIR/score-install-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
python3 - "$scratch" <<'SCORE_BUNDLE'
import base64, gzip, hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
packed = base64.b64decode("{encoded_guest}")
assert hashlib.sha256(packed).hexdigest() == "{guest_sha}"
(root / "score.py").write_bytes(gzip.decompress(packed))
(root / "inputs.json").write_bytes(base64.b64decode("{encoded_config}"))
SCORE_BUNDLE
repo="$scratch/repository"
python="$scratch/venv/bin/python"
guest() {{
    "$1" "$scratch/score.py" "$2" "$scratch/inputs.json" "$scratch" "$repo" "$python"
}}
status=0
guest python3 preflight || status=$?
if [[ $status == 0 ]]; then exit 0; fi
if [[ $status != 10 ]]; then exit "$status"; fi
mkdir -p "$HOME" "$HF_DATASETS_CACHE" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
guest python3 image
bootstrap_uv() {{
    if [[ ! -x $scratch/uv-bootstrap/bin/uv ]]; then
        python3 -m pip install --target "$scratch/uv-bootstrap" uv
    fi
}}
# Start with the training job's proven system-site-packages / ensurepip fallback.
if ! python3 -m venv --system-site-packages "$scratch/venv"; then
    bootstrap_uv
    "$scratch/uv-bootstrap/bin/uv" venv --clear --seed --system-site-packages \\
        --python "$(command -v python3)" --no-python-downloads "$scratch/venv"
fi
# scipy==1.16.3 cannot run on the image's 3.10. Use stage 1's Python 3.12;
# cp310 torch is not ABI compatible, so install the stage-1 CUDA wheel explicitly.
if [[ $("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])') != 3.12 ]]; then
    bootstrap_uv
    "$scratch/uv-bootstrap/bin/uv" venv --clear --seed --system-site-packages \\
        --python 3.12 "$scratch/venv"
fi
git init "$repo"
git -C "$repo" remote add origin https://github.com/alessiosavi/demand-forecast.git
git -C "$repo" fetch --depth=1 origin {shlex.quote(repo_sha)}
git -C "$repo" checkout --detach FETCH_HEAD
[[ $(git -C "$repo" rev-parse HEAD) == {shlex.quote(repo_sha)} ]] || {{ echo 'repository SHA mismatch' >&2; exit 2; }}
"$python" -m pip install --upgrade pip
"$python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$python" -m pip install -r "$repo/requirements-yardsticks.txt" --editable "$repo[bench]" \\
    {shlex.quote("simons @ git+https://github.com/Abel-ai-lab/Simons.git@" + upstream_pin)}
"$python" -m pip check
guest "$python" environment
guest "$python" score
'''


def main():
    """Write one executable job; generation never contacts an external service."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("name", "repo-sha", "upstream-pin", "tasks-uri", "tasks-sha256"):
        parser.add_argument("--" + key, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-uri")
    source.add_argument("--hub-weights")
    for key in ("hub-revision", "names-uri", "names-sha256"):
        parser.add_argument("--" + key)
    parser.add_argument("--processing", choices=["raw", "published", "both"], default="both")
    parser.add_argument("--max-context", type=int)
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
