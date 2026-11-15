"""Generate a pinned, self-contained real-only covariate exposure guest job."""

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
from scripts.gcp.jobs.make_simons_job import canonical_condition

GUEST = r"""
import hashlib
import json
import math
import os
import pathlib
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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def destinations(config):
    import re
    uris = [config["corpus_uri"], os.environ["STATE_URI"].rstrip("/")]
    for uri in uris:
        if (not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[A-Za-z0-9._/-]+)?", uri)
                or any(p in ("", ".", "..") for p in uri[5:].split("/"))):
            raise ValueError("invalid exposure destination")
    a, b = uris
    if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
        raise ValueError("state and corpus destinations overlap")


def root_for(config):
    root = pathlib.Path(os.environ["RUN_DIR"]) / "state/exposure" / config["name"]
    for path in [root, *root.parents]:
        if path.is_symlink():
            raise ValueError("symlink in exposure state: " + str(path))
    root.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("symlink in exposure state")
    return root


def preflight(config, root):
    identity = root / "inputs.json"
    if identity.exists():
        if json.loads(identity.read_text()) != config:
            raise ValueError("exposure state has different inputs")
    elif any(root.iterdir()):
        raise ValueError("exposure state has no input identity")
    marker = root / "completed.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if not isinstance(record, dict) or record.get("schema") != 1:
            raise ValueError("invalid exposure completion schema")
        expected_runs = {stage + "-" + leg + "-seed" + str(seed)
                         for seed in config["seeds"] for stage, leg in config["passes"]}
        runs = record.get("runs")
        if not isinstance(runs, dict) or set(runs) != expected_runs:
            raise ValueError("invalid exposure completion runs")
        for name, report in runs.items():
            if not isinstance(report, dict) or report.get("exit_code") != 0:
                raise ValueError("unsuccessful exposure completion run: " + name)
            for metric in ("wall_seconds", "peak_rss_bytes"):
                value = report.get(metric)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid exposure completion " + metric + ": " + name)
        if record.get("inputs") != config or record.get("replay") != config["replay"]:
            raise ValueError("invalid exposure completion identity")
        expected = {"inputs.json", "environment.txt"}
        expected.update(name + ext for name in expected_runs for ext in (".json", ".csv"))
        if set(record.get("artifacts", {})) != expected:
            raise ValueError("invalid exposure artifact manifest")
        for name, value in record["artifacts"].items():
            if digest(root / name) != value:
                raise ValueError("exposure artifact sha256 mismatch: " + name)
        return 0
    atomic(identity, config)
    return 10


def run(config, root, scratch, repo, upstream, python):
    runs = {}
    env = {**os.environ, "PYTHONPATH": str(repo)}
    for seed in config["seeds"]:
        for stage, leg in config["passes"]:
            name = stage + "-" + leg + "-seed" + str(seed)
            output = scratch / name
            output.mkdir(parents=True, exist_ok=True)
            args = [str(python), "-m", "scripts.corpus_v4.covariate_exposure",
                    "--corpus", str(scratch / "corpus"), "--upstream", str(upstream),
                    "--stage", stage, "--size", "25m", "--pass", leg,
                    "--batches", str(config["batches"]), "--seed", str(seed), "--output", str(output)]
            for override in config["overrides"]:
                args.extend(["--override", override])
            started = time.monotonic()
            child = subprocess.Popen(args, cwd=repo, env=env)
            # wait4 reports this child's peak, rather than the maximum of earlier runs.
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
            runs[name] = {"wall_seconds": time.monotonic() - started,
                          "peak_rss_bytes": int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
                          "exit_code": child.returncode}
            if child.returncode:
                raise RuntimeError("audit failed: " + name + " exit " + str(child.returncode))
            for extension in (".json", ".csv"):
                source = output / (leg + extension)
                if not source.is_file():
                    raise RuntimeError("missing audit artifact: " + str(source))
                source.replace(root / (name + extension))
    names = ["inputs.json", "environment.txt"] + [n + e for n in runs for e in (".json", ".csv")]
    artifacts = {name: digest(root / name) for name in sorted(names)}
    destination = os.environ["STATE_URI"].rstrip("/") + "/exposure/" + config["name"]
    # Publish evidence before the final marker; corpus is never a destination.
    for name in names:
        subprocess.run(["gcloud", "storage", "cp", str(root / name), destination + "/" + name], check=True)
    marker = root / "completed.json"
    atomic(marker, {"schema": 1, "replay": config["replay"], "inputs": config,
                    "runs": runs, "artifacts": artifacts})
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
    scratch, repo, upstream, python = map(pathlib.Path, sys.argv[3:])
    run(config, root, scratch, repo, upstream, python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def generate(*, name, repo_sha, upstream_pin, corpus_uri, batches, seeds, overrides=()):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
        raise ValueError("invalid exposure name")
    for value in (repo_sha, upstream_pin):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("revisions must be full lowercase commit hashes")
    corpus_uri = _prefix(corpus_uri)
    if type(batches) is not int or batches < 1:
        raise ValueError("batches must be a positive integer")
    if (
        not seeds
        or any(type(s) is not int or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be unique nonnegative integers")
    parsed = canonical_condition("exposure", overrides) if overrides else None
    overrides = [k + "=" + v for k, v in parsed["overrides"].items()] if parsed else []
    config = dict(
        schema=1,
        replay="real-only statistical",
        name=name,
        repo_sha=repo_sha,
        upstream_pin=upstream_pin,
        corpus_uri=corpus_uri,
        batches=batches,
        seeds=seeds,
        overrides=overrides,
        passes=[
            ["stage3", "main"],
            ["stage4", "main"],
            ["stage4-long", "main"],
            ["stage4-long", "long"],
        ],
        guest_sha256=hashlib.sha256(GUEST.encode()).hexdigest(),
    )
    encoded = base64.b64encode(gzip.compress(GUEST.encode(), mtime=0)).decode()
    identity = shlex.quote(json.dumps(config, sort_keys=True))
    return f'''#!/usr/bin/env bash
# replay: real-only statistical
set -euo pipefail
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export STATE_URI}}"
mkdir -p "$RUN_DIR/exposure-bootstrap"
bootstrap="$RUN_DIR/exposure-bootstrap/{name}"
mkdir -p "$bootstrap"
# guest-gzip-base64: {encoded}
python3 - "$bootstrap/guest.py" <<'EXPOSURE_BUNDLE'
import base64, gzip, hashlib, pathlib, sys
content = gzip.decompress(base64.b64decode("{encoded}"))
assert hashlib.sha256(content).hexdigest() == "{config["guest_sha256"]}"
pathlib.Path(sys.argv[1]).write_bytes(content)
EXPOSURE_BUNDLE
printf '%s\\n' {identity} > "$bootstrap/config.json"
if python3 "$bootstrap/guest.py" preflight "$bootstrap/config.json"; then
    exit 0
else
    status=$?
    if [[ "$status" != 10 ]]; then exit "$status"; fi
fi
scratch="$RUN_DIR/exposure-work/{name}"
mkdir -p "$scratch"
repo="$scratch/repo"
upstream="$scratch/upstream"
if [[ ! -d "$repo/.git" ]]; then
    git clone --no-checkout {shlex.quote(REPOSITORY)} "$repo"
fi
cd "$repo"
git fetch origin '{repo_sha}'
git checkout --detach '{repo_sha}'
test "$(git rev-parse HEAD)" = '{repo_sha}'
# Fresh upstream scratch on a retry, so patches are applied exactly once.
rm -rf "$upstream"
git clone --no-checkout https://github.com/Abel-ai-lab/Simons "$upstream"
cd "$upstream"
git fetch origin '{upstream_pin}'
git checkout --detach '{upstream_pin}'
test "$(git rev-parse HEAD)" = '{upstream_pin}'
python3 "$repo/third_party/simons/prepare.py" "$upstream" "$repo/third_party/simons"
if ! python3 -m venv "$scratch/venv"; then
    python3 -m pip install --target "$scratch/uv-bootstrap" uv
    "$scratch/uv-bootstrap/bin/uv" venv --clear --seed --python "$(command -v python3)" \\
        --no-python-downloads "$scratch/venv"
fi
python="$scratch/venv/bin/python"
"$python" -m pip install 'torch>=2.5' --index-url https://download.pytorch.org/whl/cpu
"$python" -m pip install --editable "$upstream"
printf '# replay: real-only statistical\\n' > "$RUN_DIR/state/exposure/{name}/environment.txt"
"$python" -m pip freeze >> "$RUN_DIR/state/exposure/{name}/environment.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
mkdir -p "$scratch/corpus"
# One recursive download, restricted to covariates; no univariate request.
gcloud storage cp --recursive '{corpus_uri}/covariates' "$scratch/corpus"
exec "$python" "$bootstrap/guest.py" run "$bootstrap/config.json" "$scratch" "$repo" "$upstream" "$python"
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("name", "repo-sha", "upstream-pin", "corpus-uri"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--seed", dest="seeds", type=int, action="append", required=True)
    parser.add_argument("--override", dest="overrides", action="append", default=[])
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
