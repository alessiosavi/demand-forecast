"""Generate a pinned, self-contained folder row-inventory guest job."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import re
import shlex
from pathlib import Path

from scripts.corpus_v4.make_corpus_job import REPOSITORY
from scripts.corpus_v4.make_sketch_job import _prefix

GUEST = r"""
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def destinations(config):
    uris = [config["corpus_uri"], os.environ["STATE_URI"].rstrip("/")]
    for uri in uris:
        if (not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[A-Za-z0-9._/-]+)?", uri)
                or any(p in ("", ".", "..") for p in uri[5:].split("/"))):
            raise ValueError("invalid inventory destination")
    a, b = uris
    if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
        raise ValueError("state and corpus destinations overlap")


def root_for(config):
    root = pathlib.Path(os.environ["RUN_DIR"]) / "state/inventory" / config["name"]
    for path in [root, *root.parents]:
        if path.is_symlink():
            raise ValueError("symlink in inventory state: " + str(path))
    root.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("symlink in inventory state")
    return root


def artifact_names(config):
    names = {"inputs.json", "environment.txt"}
    names.update(folder.replace("/", "__") + "/inventory." + ext
                 for folder in config["folders"] for ext in ("csv", "json"))
    return names


def preflight(config, root):
    identity = root / "inputs.json"
    if identity.exists():
        if json.loads(identity.read_text()) != config:
            raise ValueError("inventory state has different inputs")
    elif any(root.iterdir()):
        raise ValueError("inventory state has no input identity")
    marker = root / "completed.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if (not isinstance(record, dict) or type(record.get("schema")) is not int
                or record["schema"] != 1 or record.get("inputs") != config):
            raise ValueError("invalid inventory completion identity/schema")
        runs = record.get("runs")
        if not isinstance(runs, dict) or set(runs) != set(config["folders"]):
            raise ValueError("invalid inventory completion runs")
        for name, report in runs.items():
            if (not isinstance(report, dict) or type(report.get("exit_code")) is not int
                    or report["exit_code"] != 0):
                raise ValueError("unsuccessful inventory completion run: " + name)
            for metric in ("wall_seconds", "peak_rss_bytes"):
                value = report.get(metric)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid inventory completion metric: " + name)
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != artifact_names(config):
            raise ValueError("invalid inventory artifact manifest")
        for name, value in artifacts.items():
            if (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                    or digest(root / name) != value):
                raise ValueError("inventory artifact sha256 mismatch: " + name)
        return 0
    atomic(identity, config)
    return 10


def run(config, root, scratch, repo, python):
    runs = {}
    env = {**os.environ, "PYTHONPATH": str(repo)}
    for folder in config["folders"]:
        local = scratch / "corpus" / folder
        # A retry must not retain shards no longer present in the downloaded folder.
        if local.exists():
            shutil.rmtree(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "cp", "--recursive",
                        config["corpus_uri"] + "/" + folder, str(local.parent)], check=True)
        output = scratch / "outputs" / folder.replace("/", "__")
        output.mkdir(parents=True, exist_ok=True)
        args = [str(python), "-m", "scripts.corpus_v4.row_inventory", "--folder", str(local),
                "--folder-name", folder, "--output", str(output)]
        started = time.monotonic()
        child = subprocess.Popen(args, cwd=repo, env=env)
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        runs[folder] = {"exit_code": child.returncode, "wall_seconds": time.monotonic() - started,
                        "peak_rss_bytes": int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))}
        if child.returncode:
            raise RuntimeError("inventory failed: " + folder + " exit " + str(child.returncode))
        summary = json.loads((output / "inventory.json").read_text())
        if (summary.get("folder") != folder or summary.get("csv_sha256") != digest(output / "inventory.csv")
                or type(summary.get("rows")) is not int or summary["rows"] < 0):
            raise ValueError("invalid inventory output: " + folder)
        destination = root / folder.replace("/", "__")
        destination.mkdir(parents=True, exist_ok=True)
        for ext in ("csv", "json"):
            (output / ("inventory." + ext)).replace(destination / ("inventory." + ext))
    names = sorted(artifact_names(config))
    artifacts = {name: digest(root / name) for name in names}
    destination = os.environ["STATE_URI"].rstrip("/") + "/inventory/" + config["name"]
    for name in names:
        subprocess.run(["gcloud", "storage", "cp", str(root / name), destination + "/" + name], check=True)
    marker = root / "completed.json"
    atomic(marker, {"schema": 1, "inputs": config, "runs": runs, "artifacts": artifacts})
    try:
        subprocess.run(["gcloud", "storage", "cp", str(marker), destination + "/completed.json"], check=True)
    except BaseException:
        marker.unlink()
        raise


def main():
    config = json.loads(pathlib.Path(sys.argv[2]).read_text())
    destinations(config)
    root = root_for(config)
    if sys.argv[1] == "preflight":
        return preflight(config, root)
    scratch, repo, python = map(pathlib.Path, sys.argv[3:])
    run(config, root, scratch, repo, python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def generate(*, name, repo_sha, corpus_uri, folders):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
        raise ValueError("invalid inventory name")
    if not isinstance(repo_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", repo_sha):
        raise ValueError("revision must be a full lowercase commit hash")
    corpus_uri = _prefix(corpus_uri)
    if (
        not isinstance(folders, (list, tuple))
        or not folders
        or any(
            not isinstance(f, str)
            or not re.fullmatch(
                r"(?:quarantine/)?(?:univariate|covariates)/[A-Za-z0-9][A-Za-z0-9._-]*", f
            )
            for f in folders
        )
    ):
        raise ValueError("folders must be canonical native folder paths")
    if len(set(folders)) != len(folders) or len({f.replace("/", "__") for f in folders}) != len(
        folders
    ):
        raise ValueError("folders must be unique without publication path collisions")
    config = dict(
        schema=1,
        name=name,
        repo_sha=repo_sha,
        corpus_uri=corpus_uri,
        folders=sorted(folders),
        guest_sha256=hashlib.sha256(GUEST.encode()).hexdigest(),
    )
    encoded = base64.b64encode(gzip.compress(GUEST.encode(), mtime=0)).decode()
    identity = shlex.quote(json.dumps(config, sort_keys=True))
    return f'''#!/usr/bin/env bash
set -euo pipefail
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export STATE_URI}}"
mkdir -p "$RUN_DIR/inventory-bootstrap"
bootstrap="$RUN_DIR/inventory-bootstrap/{name}"
mkdir -p "$bootstrap"
# guest-gzip-base64: {encoded}
python3 - "$bootstrap/guest.py" <<'INVENTORY_BUNDLE'
import base64, gzip, hashlib, pathlib, sys
content = gzip.decompress(base64.b64decode("{encoded}"))
assert hashlib.sha256(content).hexdigest() == "{config["guest_sha256"]}"
pathlib.Path(sys.argv[1]).write_bytes(content)
INVENTORY_BUNDLE
printf '%s\\n' {identity} > "$bootstrap/config.json"
if python3 "$bootstrap/guest.py" preflight "$bootstrap/config.json"; then
    exit 0
else
    status=$?
    if [[ "$status" != 10 ]]; then exit "$status"; fi
fi
scratch="$RUN_DIR/inventory-work/{name}"
mkdir -p "$scratch"
repo="$scratch/repo"
if [[ ! -d "$repo/.git" ]]; then
    git clone --no-checkout {shlex.quote(REPOSITORY)} "$repo"
fi
cd "$repo"
git fetch origin '{repo_sha}'
git checkout --detach '{repo_sha}'
test "$(git rev-parse HEAD)" = '{repo_sha}'
if ! python3 -m venv "$scratch/venv"; then
    python3 -m pip install --target "$scratch/uv-bootstrap" uv
    "$scratch/uv-bootstrap/bin/uv" venv --clear --seed --python "$(command -v python3)" \\
        --no-python-downloads "$scratch/venv"
fi
python="$scratch/venv/bin/python"
"$python" -m pip install 'pyarrow<26'
"$python" -m pip freeze > "$RUN_DIR/state/inventory/{name}/environment.txt"
export ARROW_NUM_THREADS=1
exec "$python" "$bootstrap/guest.py" run "$bootstrap/config.json" "$scratch" "$repo" "$python"
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("name", "repo-sha", "corpus-uri"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--folder", dest="folders", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    source = generate(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    output.chmod(0o755)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
