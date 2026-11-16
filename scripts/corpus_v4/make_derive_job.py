"""Generate a CPU-only, pinned v4.1 derivation guest; generation makes no cloud calls."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import re
import shlex
from pathlib import Path

from scripts.corpus_v4.derive_corpus import ENVIRONMENT
from scripts.corpus_v4.make_corpus_job import REPOSITORY
from scripts.corpus_v4.make_sketch_job import _prefix

GUEST = r"""
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def destinations(config):
    uris = [config["parent_uri"], config["output_uri"], os.environ["STATE_URI"].rstrip("/")]
    for uri in uris:
        if (not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[A-Za-z0-9._/-]+)?", uri)
                or any(p in ("", ".", "..") for p in uri[5:].split("/"))):
            raise ValueError("invalid derivation destination")
    for i, a in enumerate(uris):
        for b in uris[i + 1:]:
            if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
                raise ValueError("parent, output and state destinations overlap")


def root_for(config):
    root = pathlib.Path(os.environ["RUN_DIR"]) / "state/derive" / config["name"]
    if any(p.is_symlink() for p in [root, *root.parents]):
        raise ValueError("symlink in derivation state")
    root.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("symlink in derivation state")
    return root


def preflight(config, root):
    identity = root / "inputs.json"
    if identity.exists():
        if identity.read_bytes() != encoded(config):
            raise ValueError("derivation state has different inputs")
    elif any(root.iterdir()):
        raise ValueError("derivation state has no inputs identity")
    completion = root / "completed.json"
    if completion.exists():
        record = json.loads(completion.read_text())
        if (type(record.get("schema")) is not int or record["schema"] != 1
                or record.get("inputs") != config
                or set(record.get("artifacts", {})) != {"inputs.json", "environment.txt"}):
            raise ValueError("invalid derivation completion identity/artifacts")
        for field in ("derivation_sha256", "status_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(record.get(field))):
                raise ValueError("invalid derivation completion " + field)
        for name, digest in record["artifacts"].items():
            if sha(root / name) != digest:
                raise ValueError("derivation state artifact sha256 mismatch: " + name)
    atomic(identity, config)
    # A local completion is never authority for the remote publication.


class Store:
    def __init__(self, prefix):
        self.prefix = prefix

    def objects(self):
        result = subprocess.run(["gcloud", "storage", "ls", self.prefix + "/**"],
                                capture_output=True, text=True)
        if result.returncode:
            if (result.returncode == 1 and not result.stdout.strip()
                    and "One or more URLs matched no objects" in result.stderr):
                return set()
            raise RuntimeError("cannot list output prefix: " + result.stderr)
        names = set()
        for uri in result.stdout.splitlines():
            if not uri.startswith(self.prefix + "/"):
                raise ValueError("unexpected object listing: " + uri)
            names.add(uri[len(self.prefix) + 1:])
        return names

    def read(self, name):
        return subprocess.run(["gcloud", "storage", "cat", self.prefix + "/" + name],
                              capture_output=True, check=True).stdout

    def put(self, path, name):
        # Every output write is create-only, including payloads. Concurrent retries
        # cannot overwrite a published byte, even between listing and upload.
        result = subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0",
                                 str(path), self.prefix + "/" + name], capture_output=True)
        if result.returncode:
            if name not in self.objects() or self.read(name) != path.read_bytes():
                raise RuntimeError("create-only upload failed/different bytes: " + name)
        # Always reconcile ambiguous command success with the object bytes.
        if name.endswith(("publishing.json", "derivation.json", "attempt-status.json")):
            if self.read(name) != path.read_bytes():
                raise ValueError("uploaded publication record differs: " + name)

    def copy_parent(self, parent_uri, folder, files):
        names = [pathlib.PurePosixPath(f.path).name for f in files] + ["receipt.json", "stats.json"]
        present = self.objects()
        for name in names:
            relative = folder + "/" + name
            if relative in present:
                continue  # The complete read-back will verify this surviving object.
            result = subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0",
                                     parent_uri + "/" + relative, self.prefix + "/" + relative],
                                    capture_output=True)
            if result.returncode and relative not in self.objects():
                raise RuntimeError("parent server-side copy failed: " + relative)

    def download(self, root):
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        subprocess.run(["gcloud", "storage", "rsync", "--recursive", self.prefix, str(root)], check=True)


def complete(config, state, downloaded, verified):
    status = downloaded / "attempt-status.json"
    measurements_path = state / "measurements.json"
    destination = os.environ["STATE_URI"].rstrip("/") + "/derive/" + config["name"]
    measurements, measurements_source = {}, "unavailable"
    if measurements_path.exists():
        measurements = json.loads(measurements_path.read_text())
        measurements_source = "local"
    else:
        try:
            previous = json.loads(Store(destination).read("completed.json"))
        except (OSError, ValueError, subprocess.CalledProcessError):
            previous = None
        # Previous state supplies measurements only; all other evidence is fresh.
        if (isinstance(previous, dict) and type(previous.get("schema")) is int
                and previous["schema"] == 1 and previous.get("inputs") == config
                and previous.get("derivation_sha256") == verified["derivation_sha256"]
                and isinstance(previous.get("measurements"), dict)):
            measurements = previous["measurements"]
            measurements_source = "previous-completion"
    for name in ("inputs.json", "environment.txt"):
        subprocess.run(["gcloud", "storage", "cp", str(state / name),
                        destination + "/" + name], check=True)
    record = dict(schema=1, inputs=config, derivation_sha256=verified["derivation_sha256"],
                  status_sha256=sha(status), verification=verified,
                  measurements=measurements, measurements_source=measurements_source,
                  artifacts={name: sha(state / name) for name in ("inputs.json", "environment.txt")})
    candidate = state / "completion-candidate.json"
    atomic(candidate, record)
    try:
        subprocess.run(["gcloud", "storage", "cp", str(candidate),
                        destination + "/completed.json"], check=True)
    finally:
        candidate.unlink()
    atomic(state / "completed.json", record)


def run(config, state, scratch, repo):
    sys.path.insert(0, str(repo))
    from scripts.corpus_v4 import derive_corpus as derive
    from scripts.corpus_v4.fetch_native import inventory
    from scripts.corpus_v4.freeze_manifest import read_manifest
    from scripts.corpus_v4.verify_published import PinnedInventory
    store = Store(config["output_uri"])
    publishing = dict(inputs=config, attempt_id=derive.attempt_id(config))
    present = store.objects()
    if present and ("derivation/publishing.json" not in present
                    or store.read("derivation/publishing.json") != encoded(publishing)):
        raise ValueError("populated output has no matching publishing identity")
    downloaded = scratch / "readback"
    if "attempt-status.json" in present:
        store.download(downloaded)
        verified = derive.verify_derived(downloaded, repo)
        if (downloaded / "derivation/publishing.json").read_bytes() != encoded(publishing):
            raise ValueError("read-back publishing identity changed")
        complete(config, state, downloaded, verified)
        return
    parent = scratch / "parent"
    staging = scratch / "staging"
    for local in (parent, staging):
        if local.exists():
            shutil.rmtree(local)
        local.mkdir(parents=True)
    contract = derive.read_contract(repo)
    for name in ("corpus-identity.json", "attempt-status.json"):
        subprocess.run(["gcloud", "storage", "cp", config["parent_uri"] + "/" + name,
                        str(parent / name)], check=True)
    full_parent = read_manifest(repo / contract["parent"]["manifest"])
    expected_identity = {key: full_parent[key] for key in ("repo_id", "revision", "folders")}
    if json.loads((parent / "corpus-identity.json").read_text()) != expected_identity:
        raise ValueError("parent corpus-identity differs from full strict manifest")
    parent_status = json.loads((parent / "attempt-status.json").read_text())
    if (not isinstance(parent_status, dict) or not isinstance(parent_status.get("attempt_id"), str)
            or not parent_status["attempt_id"]):
        raise ValueError("parent attempt status lacks identity")
    for release in contract["releases"]:
        local = parent / release["source_folder"]
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "cp", "--recursive",
                        config["parent_uri"] + "/" + release["source_folder"], str(local.parent)], check=True)
    prior_stats = {}
    for release in contract["releases"]:
        name = release["folder"] + "/stats.json"
        if name in present:
            prior_stats[release["folder"]] = json.loads(store.read(name))
    measurements = {}
    records = derive.prepare_releases(parent, staging, repo, prior_stats=prior_stats,
                                      measurements=measurements)
    atomic(state / "measurements.json", measurements)
    atomic(staging / "derivation/publishing.json", publishing)
    # First output write. Even an ambiguous CLI failure must reconcile the marker.
    store.put(staging / "derivation/publishing.json", "derivation/publishing.json")
    pinned = json.loads((repo / "scripts/corpus_v4/pinned_inventory.json").read_text())
    groups = inventory(full_parent, PinnedInventory(**pinned))
    for entry in full_parent["folders"]:
        if entry["status"] == "allowed":
            store.copy_parent(config["parent_uri"], entry["folder"], groups[entry["folder"]])
    # Persist stable stats before their receipts so retries reuse published bytes.
    paths = sorted(staging.rglob("*"), key=lambda p: (p.name != "stats.json", p.as_posix()))
    for path in paths:
        if path.is_file() and path.name != "publishing.json":
            store.put(path, path.relative_to(staging).as_posix())
    # The source-only download cannot know SHA256 of the parent's receipts/stats.
    # Finalise the payload map from the one whole-prefix read-back, then confirm
    # the separately uploaded ledger bytes before any training status is created.
    store.download(downloaded)
    if (downloaded / "derivation/publishing.json").read_bytes() != encoded(publishing):
        raise ValueError("read-back publishing identity changed")
    old_ledger = ((downloaded / "derivation/derivation.json").read_bytes()
                  if (downloaded / "derivation/derivation.json").exists() else None)
    derive.write_derivation(downloaded, repo, releases=records, publishing=publishing,
                            parent_identity=parent / "corpus-identity.json", parent_status=parent_status)
    ledger = downloaded / "derivation/derivation.json"
    if old_ledger is not None and old_ledger != ledger.read_bytes():
        raise ValueError("existing derivation record differs")
    store.put(ledger, "derivation/derivation.json")
    verified = derive.verify_derived(downloaded, repo)
    status = downloaded / "attempt-status.json"
    atomic(status, derive.status_for(downloaded))
    store.put(status, "attempt-status.json")  # Last output write, always create-only.
    complete(config, state, downloaded, verified)


def main():
    config = json.loads(pathlib.Path(sys.argv[2]).read_text())
    destinations(config)
    state = root_for(config)
    if sys.argv[1] == "preflight":
        preflight(config, state)
    else:
        scratch, repo = map(pathlib.Path, sys.argv[3:])
        run(config, state, scratch, repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def generate(*, name, repo_sha, parent_uri, output_uri):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
        raise ValueError("invalid derivation name")
    if not isinstance(repo_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", repo_sha):
        raise ValueError("revision must be a full lowercase commit hash")
    parent_uri, output_uri = _prefix(parent_uri), _prefix(output_uri)
    if (
        parent_uri == output_uri
        or parent_uri.startswith(output_uri + "/")
        or output_uri.startswith(parent_uri + "/")
    ):
        raise ValueError("parent and output destinations overlap")
    config = dict(
        schema=1,
        name=name,
        repo_sha=repo_sha,
        parent_uri=parent_uri,
        output_uri=output_uri,
        environment=ENVIRONMENT,
        guest_sha256=hashlib.sha256(GUEST.encode()).hexdigest(),
    )
    bundle = base64.b64encode(gzip.compress(GUEST.encode(), mtime=0)).decode()
    identity = shlex.quote(json.dumps(config, sort_keys=True))
    pins = " ".join(shlex.quote(name + "==" + version) for name, version in ENVIRONMENT.items())
    return f'''#!/usr/bin/env bash
set -euo pipefail
# CPU budget: about 25 GB working data, 100 GB boot disk, 1.5 hours.
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export STATE_URI}}"
bootstrap="$RUN_DIR/derive-bootstrap/{name}"
mkdir -p "$bootstrap"
# guest-gzip-base64: {bundle}
python3 - "$bootstrap/guest.py" <<'DERIVE_BUNDLE'
import base64, gzip, hashlib, pathlib, sys
content = gzip.decompress(base64.b64decode("{bundle}"))
assert hashlib.sha256(content).hexdigest() == "{config["guest_sha256"]}"
pathlib.Path(sys.argv[1]).write_bytes(content)
DERIVE_BUNDLE
printf '%s\\n' {identity} > "$bootstrap/config.json"
python3 "$bootstrap/guest.py" preflight "$bootstrap/config.json"
scratch="$RUN_DIR/derive-work/{name}"
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
"$python" -m pip install {pins}
"$python" -m pip freeze > "$RUN_DIR/state/derive/{name}/environment.txt"
export ARROW_NUM_THREADS=1
export PYTHONPATH="$repo"
exec "$python" "$bootstrap/guest.py" run "$bootstrap/config.json" "$scratch" "$repo"
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("name", "repo-sha", "parent-uri", "output-uri"):
        parser.add_argument("--" + name, required=True)
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
