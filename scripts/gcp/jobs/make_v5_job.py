"""Generate a pinned native-v5 guest job without contacting external services."""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import shlex
from pathlib import Path

import yaml

# This helper deliberately uses only the image interpreter's standard library.
# It runs before repository or environment installation, including on a retry.
GUEST = r"""
import base64
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

def valid(unit, job):
    try:
        record = json.loads((unit / 'unit.json').read_text())
        files = record['files']
        if record['job_sha256'] != job or not isinstance(files, dict) or not files:
            return False
        for name, metadata in files.items():
            if not isinstance(name, str) or not name or '..' in name or name.startswith('/') or '\\' in name:
                return False
            target = unit
            for part in pathlib.Path(name).parts:
                target /= part
                if target.is_symlink():
                    return False
            if not target.is_file() or type(metadata['bytes']) is not int:
                return False
            if target.stat().st_size != metadata['bytes'] or digest(target) != metadata['sha256']:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False

def listing(source, uri, expected):
    # storage ls --json emits one array per URL, rather than one combined
    # document. Decode each complete array from the output stream.
    pending = pathlib.Path(source).read_text().strip()
    decoder, rows = json.JSONDecoder(), []
    while pending:
        frame, end = decoder.raw_decode(pending)
        if not isinstance(frame, list):
            raise ValueError('listing requires JSON arrays')
        rows.extend(frame)
        pending = pending[end:].lstrip()
    objects, generations, seen = [], [], set()
    for entry in rows:
        metadata = entry.get('metadata', entry)
        name = entry.get('storage_url') or metadata.get('name')
        if not isinstance(name, str):
            raise ValueError('listing lacks object name')
        if name.startswith('gs://'):
            if not name.startswith(uri.rstrip('/') + '/'):
                raise ValueError('object outside corpus prefix')
            name = name[len(uri.rstrip('/')) + 1:]
        else:
            prefix = uri.split('/', 3)[3] if len(uri.split('/', 3)) == 4 else ''
            if prefix:
                if not name.startswith(prefix.rstrip('/') + '/'):
                    raise ValueError('object outside corpus prefix')
                name = name[len(prefix.rstrip('/')) + 1:]
        parts = pathlib.PurePosixPath(name).parts
        if not parts or parts[0] not in ('univariate', 'covariates') or '..' in parts or name.startswith('/'):
            raise ValueError('object outside training trees')
        if name.endswith('/'):
            continue
        if name in seen:
            raise ValueError('duplicate listing object')
        seen.add(name)
        size = int(metadata['size'])
        checksum = metadata.get('crc32c') or metadata.get('crc32c_hash')
        generation = str(metadata['generation'])
        if size < 0 or not checksum or not generation.isdigit():
            raise ValueError('incomplete object metadata')
        objects.append([name, size, checksum])
        generations.append([name, size, checksum, generation])
    objects.sort(key=lambda row: row[0])
    generations.sort(key=lambda row: row[0])
    value = hashlib.sha256()
    for name, size, checksum in objects:
        value.update(f'{name}\t{size}\t{checksum}\n'.encode())
    if value.hexdigest() != expected:
        raise ValueError('corpus digest mismatch')
    return {'objects': objects, 'generations': generations}

def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = pathlib.Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def main():
    action, *args = sys.argv[1:]
    if action == 'hash':
        print(digest(pathlib.Path(args[0])))
    elif action == 'preflight':
        return 0 if valid(pathlib.Path(args[0]), args[1]) else 10
    elif action == 'listing':
        print(json.dumps(listing(*args), sort_keys=True))
    elif action == 'manifest':
        before, after, destination = map(pathlib.Path, args)
        record = json.loads(before.read_text())
        if record != json.loads(after.read_text()):
            raise ValueError('corpus generation changed during copy')
        atomic(destination, json.dumps({'objects': record['objects']}).encode())
    elif action == 'environment':
        lines = sorted(line.strip() for line in sys.stdin if line.strip())
        print(hashlib.sha256(('\n'.join(lines) + '\n').encode()).hexdigest())
    elif action == 'config':
        encoded, expected, destination = args
        data = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('configuration digest mismatch')
        atomic(pathlib.Path(destination), data)
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
""".lstrip()


def _name(value: str, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"{field} requires a safe name")
    return value


def _hex(value: str, size: int, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{size}}}", value):
        raise ValueError(f"{field} requires {size} lowercase hexadecimal characters")
    return value


def _uri(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"gs://[a-z0-9][a-z0-9._-]*(?:/[A-Za-z0-9._/-]+)?", value
    ):
        raise ValueError("expected a literal gs:// URI")
    if value.endswith("/") or any(p in (".", "..", "") for p in value[5:].split("/")):
        raise ValueError("expected a normalized gs:// URI")
    return value


def _count(value: int | None, default: int) -> int:
    if value is None:
        return default
    if type(value) is not int or value < 1:
        raise ValueError("evaluation counts must be positive integers")
    return value


def generate(
    *,
    name: str,
    action: str,
    repo_sha: str,
    config: str | None = None,
    base: str | None = None,
    leg: str | None = None,
    init_uri: str | None = None,
    init_sha256: str | None = None,
    path: str | None = None,
    arm: str | None = None,
    seed: int | None = None,
    episodes: int | None = None,
    max_panels: int | None = None,
    corpus_uri: str | None = None,
    corpus_digest: str | None = None,
) -> str:
    """Return one deterministic shell script; config/base name local YAML files."""
    _name(name, "name")
    _hex(repo_sha, 40, "repo_sha")
    if action not in ("train", "qualify", "t1"):
        raise ValueError("action must be train, qualify or t1")
    if (corpus_uri is None) != (corpus_digest is None):
        raise ValueError("corpus_uri and corpus_digest must be supplied together")
    if (init_uri is None) != (init_sha256 is None):
        raise ValueError("init_uri and init_sha256 must be supplied together")
    if corpus_uri is not None:
        _uri(corpus_uri)
        _hex(corpus_digest, 64, "corpus_digest")
    if init_uri is not None:
        _uri(init_uri)
        _hex(init_sha256, 64, "init_sha256")
        if action != "train":
            raise ValueError("initial checkpoints apply only to train")
    command = [action]
    if action == "t1":
        if base is None or config is not None or leg is not None or corpus_uri is not None:
            raise ValueError("t1 requires base, without config, leg or corpus")
        _name(path, "path")
        _name(arm, "arm")
        if type(seed) is not int:
            raise ValueError("t1 requires an integer seed")
        episodes, max_panels = _count(episodes, 200), _count(max_panels, 1000)
        unit = f"state/t1/{path}-{arm}-s{seed}/eval-{episodes}-{max_panels}"
        command += [
            "--path",
            path,
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--episodes",
            str(episodes),
            "--max-panels",
            str(max_panels),
        ]
        config_path, config_flag = base, "--base"
    else:
        if config is None or any(
            v is not None for v in (base, path, arm, seed, episodes, max_panels)
        ):
            raise ValueError("train/qualify require config and no T1 parameters")
        if action == "qualify" and leg is not None:
            raise ValueError("leg applies only to train")
        unit_name = _name(leg or name, "leg") if action == "train" else name
        unit = f"state/{action}/{unit_name}"
        command += ["--name", unit_name]
        config_path, config_flag = config, "--config"
    data = Path(config_path).read_bytes()
    parsed = yaml.safe_load(data)
    if not isinstance(parsed, dict):
        raise ValueError("configuration requires a mapping")
    sources = parsed.get("sources", [])
    if any(s.get("kind") == "corpus" and s.get("root") is not None for s in sources):
        raise ValueError("cloud corpus sources must carry root: null")
    if action != "t1" and any(s.get("kind") == "corpus" for s in sources) != (
        corpus_uri is not None
    ):
        raise ValueError("corpus sources and corpus_uri must be supplied together")
    packed_helper = base64.b64encode(GUEST.encode()).decode()
    packed_config = base64.b64encode(data).decode()
    config_sha = hashlib.sha256(data).hexdigest()
    command_text = shlex.join(command)
    script = f'''#!/usr/bin/env bash
# v5 job {name}; repository {repo_sha}
set -euo pipefail
umask 077
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export STATE_URI}}"
case "$RUN_DIR" in /*) ;; *) echo 'RUN_DIR must be absolute' >&2; exit 2 ;; esac
case "$STATE_URI" in gs://*) ;; *) echo 'STATE_URI must be a bucket URI' >&2; exit 2 ;; esac
export RUN_DIR
cache="$RUN_DIR/v5-cache"
export HOME="$cache/home"
export HF_HOME="$cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE HF_TOKEN HUGGING_FACE_HUB_TOKEN
export UV_CACHE_DIR="$cache/uv"
export UV_PYTHON_INSTALL_DIR="$cache/python"
export PIP_CACHE_DIR="$cache/pip"
export TMPDIR="$cache/tmp"
mkdir -p "$HOME" "$HF_DATASETS_CACHE" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
scratch=$(mktemp -d "$cache/install-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
python3 - "$scratch/helper.py" <<'V5_HELPER'
import base64, pathlib, sys
pathlib.Path(sys.argv[1]).write_bytes(base64.b64decode("{packed_helper}"))
V5_HELPER
helper="$scratch/helper.py"
job_sha256=$(python3 "$helper" hash "$0")
status=0
python3 "$helper" preflight "$RUN_DIR/{unit}" "$job_sha256" || status=$?
if [[ $status == 0 ]]; then exit 0; fi
if [[ $status != 10 ]]; then exit "$status"; fi
repo="$scratch/repository"
git init "$repo"
git -C "$repo" remote add origin https://github.com/alessiosavi/demand-forecast.git
git -C "$repo" fetch --depth=1 origin {repo_sha}
git -C "$repo" checkout --detach FETCH_HEAD
[[ $(git -C "$repo" rev-parse HEAD) == {repo_sha} ]] || {{ echo 'repository SHA mismatch' >&2; exit 2; }}
if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --target "$scratch/uv-bootstrap" uv
    export PATH="$scratch/uv-bootstrap/bin:$PATH"
fi
uv venv --python 3.12 --seed "$scratch/venv"
python="$scratch/venv/bin/python"
"$python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$python" -m pip install -r "$repo/requirements-v5.txt" --editable "$repo[corpus]"
"$python" -m pip check
env_sha256=$("$python" -m pip freeze --exclude-editable | python3 "$helper" environment)
extra=()
'''
    if corpus_uri is not None:
        uri = shlex.quote(corpus_uri)
        script += f"""corpus_uri={uri}
gcloud storage ls --recursive --json "$corpus_uri/univariate/**" "$corpus_uri/covariates/**" > "$scratch/before.json"
python3 "$helper" listing "$scratch/before.json" "$corpus_uri" {corpus_digest} > "$scratch/before.checked.json"
rm -rf -- "$RUN_DIR/corpus"
mkdir -p "$RUN_DIR/corpus"
gcloud storage rsync --recursive --exclude='(^|/)quarantine(/|$)' "$corpus_uri/univariate" "$RUN_DIR/corpus/univariate"
gcloud storage rsync --recursive --exclude='(^|/)quarantine(/|$)' "$corpus_uri/covariates" "$RUN_DIR/corpus/covariates"
gcloud storage ls --recursive --json "$corpus_uri/univariate/**" "$corpus_uri/covariates/**" > "$scratch/after.json"
python3 "$helper" listing "$scratch/after.json" "$corpus_uri" {corpus_digest} > "$scratch/after.checked.json"
python3 "$helper" manifest "$scratch/before.checked.json" "$scratch/after.checked.json" "$RUN_DIR/corpus/manifest.json"
extra+=(--corpus-root "$RUN_DIR/corpus")
"""
    if init_uri is not None:
        script += f"""gcloud storage cp {shlex.quote(init_uri)} "$scratch/initial.pt"
extra+=(--init-file "$scratch/initial.pt" --init-sha256 {init_sha256})
"""
    script += f"""python3 "$helper" config {shlex.quote(packed_config)} {config_sha} "$scratch/config.yaml"
exec "$python" -m demand_forecast.pretrain.v5.job {command_text} \\
    {config_flag} "$scratch/config.yaml" --run-dir "$RUN_DIR" --code {repo_sha} \\
    --job-sha256 "$job_sha256" --env-sha256 "$env_sha256" "${{extra[@]}}"
"""
    return script


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("name", "action", "repo-sha"):
        parser.add_argument("--" + key, required=True)
    for key in (
        "config",
        "base",
        "leg",
        "init-uri",
        "init-sha256",
        "path",
        "arm",
        "corpus-uri",
        "corpus-digest",
    ):
        parser.add_argument("--" + key)
    for key in ("seed", "episodes", "max-panels"):
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    try:
        script = generate(**args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script)
    output.chmod(0o755)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
