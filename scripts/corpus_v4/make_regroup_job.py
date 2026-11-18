"""Generate an offline CPU guest for publishing the regrouped training corpus."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

import yaml

from scripts.corpus_v4.derive_corpus import ENVIRONMENT
from scripts.corpus_v4.make_corpus_job import REPOSITORY
from scripts.corpus_v4.make_sketch_job import _overlap, _prefix

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/corpus-v4.2.yaml"
PARAMETERS = ("threshold_bytes", "group_bytes", "group_rows")

# The image interpreter can check a completion and read the parent before installing C1.
GUEST = r"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

TREES = ("univariate", "covariates")
ATTRIBUTION = {"LICENSE.md", "articles.csv"}
CLAIM = "regroup-claim.json"
LEDGER = "regroup.json"
STATUS = "attempt-status.json"
COMPLETED = "completed.json"


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def state_destination(inputs):
    uri = os.environ["STATE_URI"].rstrip("/")
    if (not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[A-Za-z0-9._/-]+)?", uri)
            or any(part in ("", ".", "..") for part in uri[5:].split("/"))):
        raise ValueError("invalid regroup state destination")
    for other in (inputs["parent_uri"], inputs["output_uri"]):
        if uri == other or uri.startswith(other + "/") or other.startswith(uri + "/"):
            raise ValueError("parent, output and state destinations overlap")
    return f'{uri}/regroup/{inputs["name"]}'


def training_file(name):
    tree, separator, _ = name.partition("/")
    return bool(separator) and tree in TREES


def parent_file(name):
    return not name.endswith("/") and (
        training_file(name)
        or (PurePosixPath(name).name in ATTRIBUTION and "quarantine" not in name.lower())
    )


def listing_rows(text):
    # As in the training guest, accept consecutive arrays from storage ls --json.
    decoder = json.JSONDecoder()
    remaining = text.strip()
    while remaining:
        rows, end = decoder.raw_decode(remaining)
        if not isinstance(rows, list):
            raise ValueError("listing requires JSON arrays")
        yield from rows
        remaining = remaining[end:].lstrip()


def fingerprints(listing):
    return {name: [size, crc] for name, (size, crc, _) in listing.items()}


def corpus_digest(objects):
    # Same relative names, ordering and framing as make_v5_job's listing helper.
    value = hashlib.sha256()
    for name, (size, checksum) in sorted(objects.items()):
        if training_file(name):
            value.update(f'{name}\t{size}\t{checksum}\n'.encode())
    return value.hexdigest()


class Store:
    def __init__(self, prefix):
        self.prefix = prefix

    def listing(self, *, is_parent=False):
        result = subprocess.run(
            ["gcloud", "storage", "ls", "--recursive", "--json", self.prefix + "/**"],
            capture_output=True, text=True,
        )
        if result.returncode:
            if (result.returncode == 1 and not result.stdout.strip()
                    and "One or more URLs matched no objects" in result.stderr):
                return {}
            raise RuntimeError(f"cannot list {self.prefix}: {result.stderr.strip()}")
        objects = {}
        key_prefix = self.prefix.partition("//")[2].partition("/")[2]
        key_prefix = key_prefix + "/" if key_prefix else ""
        for entry in listing_rows(result.stdout):
            if entry.get("type") in ("prefix", "folder", "managed_folder"):
                continue
            metadata = entry.get("metadata", entry)
            name = entry.get("storage_url") or metadata["name"]
            base = self.prefix + "/" if name.startswith("gs://") else key_prefix
            if not name.startswith(base):
                raise ValueError(f"object outside {self.prefix}: {name}")
            name = name[len(base):]
            if is_parent and not parent_file(name):
                continue
            if (any(part in ("", ".", "..") for part in name.split("/"))
                    or re.search(r'[\x00-\x1f\x7f\\*?\[\]#]', name)):
                raise ValueError(f"unsafe object name: {name!r}")
            size = int(metadata["size"])
            crc = metadata.get("crc32c") or metadata.get("crc32c_hash")
            generation = str(metadata["generation"])
            if name in objects or size < 0 or not isinstance(crc, str) or not crc or not generation.isdigit():
                raise ValueError(f"invalid or duplicate listing metadata: {name}")
            objects[name] = (size, crc, generation)
        return objects

    def read(self, name):
        return subprocess.run(
            ["gcloud", "storage", "cat", self.prefix + "/" + name], capture_output=True, check=True,
        ).stdout

    def put(self, path, name, fingerprint=None):
        # Every upload is create-only. A racing or ambiguous result must reconcile.
        result = subprocess.run(
            ["gcloud", "storage", "cp", "--if-generation-match=0", str(path), self.prefix + "/" + name],
            capture_output=True, text=True,
        )
        if result.returncode:
            existing = fingerprints(self.listing()).get(name)
            if existing is None or (fingerprint is not None and existing != fingerprint):
                raise RuntimeError(f"create-only upload failed or differs: {name}: {result.stderr.strip()}")
        if fingerprint is None and self.read(name) != path.read_bytes():
            raise ValueError(f"uploaded publication record differs: {name}")


def local_files(root):
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"unexpected symlink in local corpus: {path}")
        if path.is_file():
            files.append(path)
    return files


def local_fingerprints(root, files):
    objects = {}
    for start in range(0, len(files), 128):
        paths = files[start:start + 128]
        result = subprocess.check_output(
            ["gcloud", "storage", "hash", "--skip-md5", "--format=json", *map(str, paths)], text=True,
        )
        hashes = {}
        for row in json.loads(result):
            path, crc = Path(row["url"]), row["crc32c_hash"]
            if path in hashes or row["digest_format"] != "base64" or not crc:
                raise ValueError(f"invalid local CRC32C: {path}")
            hashes[path] = crc
        if hashes.keys() != set(paths):
            raise ValueError("local CRC32C listing does not match requested files")
        for path in paths:
            objects[path.relative_to(root).as_posix()] = [path.stat().st_size, hashes[path]]
    return objects


def download_parent(inputs, root):
    parent = Store(inputs["parent_uri"])
    before = parent.listing(is_parent=True)
    expected = fingerprints(before)
    if corpus_digest(expected) != inputs["parent_digest"]:
        raise ValueError("parent corpus digest mismatch")
    root.mkdir()
    for tree in TREES:
        if any(name.startswith(tree + "/") for name in before):
            target = root / tree
            target.mkdir()
            subprocess.run(
                ["gcloud", "storage", "rsync", "--recursive", f"{parent.prefix}/{tree}", str(target)],
                check=True,
            )
    for name in before:
        if not training_file(name):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["gcloud", "storage", "cp", f"{parent.prefix}/{name}", str(target)], check=True)
    if parent.listing(is_parent=True) != before:
        raise ValueError("parent corpus generation changed during copy")
    if local_fingerprints(root, local_files(root)) != expected:
        raise ValueError("parent copy differs from listed sizes or CRC32C")


def stage_corpus(work, inputs, repository, pins):
    repo, venv = work / "repo", work / "venv"
    subprocess.run(["git", "clone", "--no-checkout", repository, str(repo)], check=True)
    revision = inputs["repo_sha"]
    subprocess.run(["git", "fetch", "origin", revision], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "--detach", revision], cwd=repo, check=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != revision:
        raise ValueError("repository SHA mismatch")
    if subprocess.run(["python3", "-m", "venv", str(venv)]).returncode:
        bootstrap = work / "uv-bootstrap"
        subprocess.run(["python3", "-m", "pip", "install", "--target", str(bootstrap), "uv"], check=True)
        subprocess.run([
            str(bootstrap / "bin/uv"), "venv", "--clear", "--seed", "--python", shutil.which("python3"),
            "--no-python-downloads", str(venv),
        ], check=True)
    python = str(venv / "bin/python")
    subprocess.run([python, "-m", "pip", "install", *pins], check=True)
    out = work / "out"
    flags = []
    for key, value in inputs["parameters"].items():
        flags.extend(("--" + key.replace("_", "-"), str(value)))
    subprocess.run(
        [python, "-m", "scripts.corpus_v4.regroup_corpus", str(work / "parent"), str(out), *flags],
        cwd=repo, env={**os.environ, "ARROW_NUM_THREADS": "1", "PYTHONPATH": str(repo)}, check=True,
    )
    return out


def require_claim(store, present, claim):
    if present and (CLAIM not in present or store.read(CLAIM) != encoded(claim)):
        raise ValueError("populated output has no matching regroup claim")


def reconcile(present, expected, required, label):
    differences = (required - present.keys()) | {
        name for name, fingerprint in present.items() if expected.get(name) != fingerprint
    }
    if differences:
        raise ValueError(f"{label}: {min(differences)}")


def folder_groups(ledger):
    groups = {}
    for name, shard in ledger["shards"].items():
        folder = PurePosixPath(name).parent.as_posix()
        counts = groups.setdefault(folder, {"before": 0, "after": 0})
        if shard.get("copied"):
            counts["before"] += shard["groups"]
            counts["after"] += shard["groups"]
        else:
            counts["before"] += shard["source_groups"]
            counts["after"] += shard["output_groups"]
    return groups


def write_completion(state, work, claim, final, groups):
    completion = dict(
        claim, corpus_digest=corpus_digest(final), output_listing=final,
        folder_groups=groups,
    )
    path = work / COMPLETED
    path.write_bytes(encoded(completion))
    state.put(path, COMPLETED)


def status_record(claim, listing, ledger_bytes):
    return dict(
        attempt_id=f'regroup-{claim["inputs"]["name"]}-{claim["identity"][:12]}',
        identity=claim["identity"], repo_sha=claim["inputs"]["repo_sha"],
        complete=False, exit_code=-1, mode="regroup", corpus_digest=corpus_digest(listing),
        regroup_sha256=hashlib.sha256(ledger_bytes).hexdigest(), output_listing=listing,
    )


def complete_from_status(output, state, work, claim, listed):
    status_bytes = output.read(STATUS)
    status = json.loads(status_bytes)
    if not isinstance(status, dict) or status.get("identity") != claim["identity"]:
        raise ValueError("regroup status has different identity")
    expected = status.get("output_listing")
    if not isinstance(expected, dict):
        raise ValueError("regroup status lacks output fingerprints")
    if LEDGER not in listed:
        raise ValueError(f"status output listing differs: {LEDGER}")
    ledger_bytes = output.read(LEDGER)
    verified_status = status_record(claim, expected, ledger_bytes)
    if status.get("regroup_sha256") != verified_status["regroup_sha256"]:
        raise ValueError("regroup ledger SHA-256 differs")
    ledger = json.loads(ledger_bytes)
    if ledger["parameters"] != claim["inputs"]["parameters"]:
        raise ValueError("regroup ledger parameters differ")

    final = fingerprints(listed)
    without_status = {name: fingerprint for name, fingerprint in final.items() if name != STATUS}
    ledger_names = ledger["shards"].keys() | ledger["other"].keys() | {CLAIM, LEDGER}
    if expected.keys() != ledger_names:
        raise ValueError("regroup ledger inventory differs from status")
    reconcile(without_status, expected, ledger_names, "status output listing differs")
    for name, shard in ledger["shards"].items():
        if final[name][0] != shard["bytes" if shard.get("copied") else "output_bytes"]:
            raise ValueError(f"regroup ledger size differs: {name}")
    if status_bytes != encoded(verified_status):
        raise ValueError("regroup status record differs")

    # The status cannot contain its own fingerprint. Hash the bytes just read.
    path = work / STATUS
    path.write_bytes(status_bytes)
    if local_fingerprints(work, [path])[STATUS] != final[STATUS] or output.listing() != listed:
        raise ValueError("regroup output changed during status verification")
    write_completion(state, work, claim, final, folder_groups(ledger))


def publish(output, state, claim, staging):
    ledger_bytes = (staging / LEDGER).read_bytes()
    groups = folder_groups(json.loads(ledger_bytes))
    (staging / CLAIM).write_bytes(encoded(claim))
    expected = local_fingerprints(staging, local_files(staging))
    payload = expected.keys() - {CLAIM, LEDGER}
    status = status_record(claim, expected, ledger_bytes)
    (staging / STATUS).write_bytes(encoded(status))
    expected = expected | local_fingerprints(staging, [staging / STATUS])

    present = fingerprints(output.listing())
    require_claim(output, present, claim)
    # A status can only be carried forward if the full publication survived.
    required = expected.keys() if STATUS in present else set()
    reconcile(present, expected, required, "existing regroup output differs")
    if CLAIM not in present:
        output.put(staging / CLAIM, CLAIM)
    for name in sorted(payload - present.keys()):
        output.put(staging / name, name, expected[name])

    uploaded = fingerprints(output.listing())
    required = expected.keys() if STATUS in uploaded else payload | {CLAIM}
    reconcile(uploaded, expected, required, "payload read-back differs")
    for name in (LEDGER, STATUS):
        if name not in uploaded:
            output.put(staging / name, name)
    final = fingerprints(output.listing())
    reconcile(final, expected, expected.keys(), "final regroup output differs")
    write_completion(state, staging.parent, claim, final, groups)


def main():
    scratch, config, repository, *pins = sys.argv[1:]
    work = Path(scratch).resolve()
    inputs = json.loads(config)
    state = Store(state_destination(inputs))
    output = Store(inputs["output_uri"])
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    claim = dict(schema=1, identity=identity, inputs=inputs)
    if COMPLETED in state.listing():
        completed = json.loads(state.read(COMPLETED))
        if any(completed.get(key) != value for key, value in claim.items()):
            raise ValueError("regroup completion has different inputs")
        present = fingerprints(output.listing())
        if (not {CLAIM, LEDGER, STATUS} <= present.keys()
                or completed.get("output_listing") != present
                or completed.get("corpus_digest") != corpus_digest(present)):
            raise ValueError("completed output listing differs")
        require_claim(output, present, claim)
        return
    listed = output.listing()
    require_claim(output, listed, claim)
    if STATUS in listed:
        complete_from_status(output, state, work, claim, listed)
        return
    download_parent(inputs, work / "parent")
    # C1 writes its ledger only after checking the entire staged corpus.
    staging = stage_corpus(work, inputs, repository, pins)
    publish(output, state, claim, staging)


main()
""".lstrip()


def generate(*, name, repo_sha, parent_uri, parent_digest, output_uri):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
        raise ValueError("invalid regroup name")
    if not isinstance(repo_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", repo_sha):
        raise ValueError("revision must be a full lowercase commit hash")
    if not isinstance(parent_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", parent_digest):
        raise ValueError("parent digest must be a lowercase SHA-256")
    parent_uri, output_uri = _prefix(parent_uri), _prefix(output_uri)
    if _overlap(parent_uri, output_uri):
        raise ValueError("parent and output destinations overlap")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("parent") != "corpus-v4.1":
        raise ValueError("regroup config must name corpus-v4.1 as the parent")
    parameters = {key: config.get(key) for key in PARAMETERS}
    for key, value in parameters.items():
        minimum = 0 if key == "threshold_bytes" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"invalid regroup parameter: {key}")
    inputs = dict(
        name=name,
        repo_sha=repo_sha,
        parent_uri=parent_uri,
        parent_digest=parent_digest,
        output_uri=output_uri,
        parameters=parameters,
    )
    pins = " ".join(
        shlex.quote(f"{package}=={version}") for package, version in ENVIRONMENT.items()
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail
# CPU regroup of {config["parent"]} into the corpus 4.2 training view.
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export STATE_URI}}"
case "$RUN_DIR" in /*) ;; *) echo 'RUN_DIR must be absolute' >&2; exit 2 ;; esac
work="$RUN_DIR/regroup-work"
mkdir -p "$work"
scratch=$(mktemp -d "$work/{name}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
inputs={shlex.quote(json.dumps(inputs, sort_keys=True, indent=2))}
python3 - "$scratch" "$inputs" {shlex.quote(REPOSITORY)} {pins} <<'REGROUP_GUEST'
{GUEST}REGROUP_GUEST
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("name", "repo-sha", "parent-uri", "parent-digest", "output-uri"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    source = generate(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")
    output.chmod(0o755)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
