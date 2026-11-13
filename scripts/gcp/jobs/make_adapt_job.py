#!/usr/bin/env python3
"""Generate a self-contained guest job: task bundle, resumable adaptation, outer scoring."""

import argparse
import base64
import gzip
import hashlib
import json
import math
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "third_party/simons"

# System Python executes preflight; heavy imports occur only in the unified venv.
GUEST = r"""
import csv
import hashlib
import itertools
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

FILES = ("config.json", "model.safetensors")
COLUMNS = ("task", "params", "lr_multiplier", "seed", "status", "selected", "selected_update",
           "decision_reason", "inner_0_window_0", "inner_0_window_1", "inner_selected_window_0",
           "inner_selected_window_1", "outer_sql_raw", "outer_sql_published", "frozen_sql_raw",
           "frozen_sql_published", "trainable_parameters", "training_seconds",
           "inner_evaluation_seconds", "outer_evaluation_seconds", "peak_gpu_memory_bytes", "error")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def read_object(path):
    try:
        value = read(path)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


def copy_atomic(source, target):
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as f:
        with source.open("rb") as original:
            shutil.copyfileobj(original, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, target)


def child(root, name):
    if (not name or name.startswith("/") or "\\" in name
            or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("unsafe artifact path: " + name)
    path = root
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink in artifact path: " + str(path))
    return path


def slug(name):
    # Full hex is reversible, collision-free, and has no special path components.
    return name.encode().hex()


def valid(path):
    try:
        record = read(path)
        return (record if record["status"] == "ok" and record["hashes"]
                and all(sha(child(path.parent, p)) == h for p, h in record["hashes"].items()) else None)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def copy_cloud(source, destination):
    subprocess.run(["gcloud", "storage", "cp", str(source), str(destination)], check=True, timeout=120)


class Job:
    def __init__(self, config, scratch, repo, python):
        self.c, self.scratch, self.repo, self.python = config, scratch, repo, python
        self.run = pathlib.Path(os.environ["RUN_DIR"])
        self.root = child(self.run, "state/adapt/" + config["name"])
        self.root.mkdir(parents=True, exist_ok=True)
        for p in self.root.rglob("*"):
            if p.is_symlink():
                raise ValueError("symlink in adaptation state")
        self.uri = os.environ["STATE_URI"].rstrip("/") + "-adapt-artifacts/" + config["name"]
        identity = hashlib.sha256(json.dumps([config, self.uri], sort_keys=True).encode()).hexdigest()
        self.work = pathlib.Path("/tmp") / ("demand-forecast-adapt-" + identity)
        child(self.work.parent, self.work.name)
        self.work.mkdir(mode=0o700, exist_ok=True)
        self.base = self.work / "base"
        self.checkpoint_stats = {}

    def preflight(self):
        path = self.root / "inputs.json"
        marker_path = self.root / "completed.json"
        marker = read_object(marker_path)
        if "inputs" in marker and marker["inputs"] != self.c:
            raise ValueError("different inputs under the same adaptation name")
        if path.exists():
            if read(path) != self.c:
                raise ValueError("different inputs under the same adaptation name")
        else:
            if any(self.root.iterdir()) and marker.get("inputs") != self.c:
                raise ValueError("adaptation state has no valid identity; refusing overwrite")
            atomic(path, self.c)
        try:
            if marker.get("schema") != 1 or marker.get("success") is not True:
                return 10
            hashes = marker["artifacts"]
            required = {"inputs.json", "environment.txt", "checkpoint-sha256.json", "tasks.yaml", "results.csv"}
            expected = set()
            for name in self.c["names"]:
                prefix = "tasks/" + slug(name)
                required.add(prefix + "/bundle-manifest.json")
                for mode in self.c["modes"]:
                    relative = prefix + "/frozen/" + mode
                    score = valid(self.root / relative / "status.json")
                    if not score:
                        return 10
                    required.add(relative + "/status.json")
                    required.update(relative + "/" + p for p in score["hashes"])
                for params, lr, seed in itertools.product(self.c["params"], self.c["lr_multipliers"], self.c["seeds"]):
                    key = params + "-lr" + str(lr) + "-seed" + str(seed)
                    expected.add(slug(name) + "/" + key)
                    relative = prefix + "/fits/" + key
                    fit = valid(self.root / relative / "fit.json")
                    if not fit:
                        return 10
                    required.update((relative + "/fit.json", relative + "/adaptation.json"))
                    report = read(self.root / relative / "adaptation.json")
                    for mode in self.c["modes"]:
                        if report["decision"]["selected"] == "frozen_base":
                            required.add(relative + "/" + mode + "-reuse.json")
                        else:
                            score = valid(self.root / relative / mode / "status.json")
                            if not score:
                                return 10
                            required.add(relative + "/" + mode + "/status.json")
                            required.update(relative + "/" + mode + "/" + p for p in score["hashes"])
            with (self.root / "results.csv").open() as f:
                rows = list(csv.DictReader(f))
            if (marker["schema"] == 1 and marker["success"] is True
                    and marker["inputs"] == self.c and required <= hashes.keys()
                    and set(marker["fits"]) == expected and len(rows) == len(expected)
                    and all(r["status"] == "ok" for r in marker["fits"].values())
                    and sha(self.root / "tasks.yaml") == self.c["tasks_sha256"]
                    and all(sha(child(self.root, p)) == h for p, h in hashes.items())):
                return 0
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return 10

    def environment(self):
        from importlib import metadata
        from packaging.requirements import Requirement
        import simons
        import torch
        from simons.training import adapt
        if sys.version_info[:2] != (3, 12):
            raise ValueError("adaptation requires Python 3.12")
        for package, version in (("fev", "0.10.0"), ("datasets", "3.6.0"), ("huggingface-hub", "0.36.0")):
            if metadata.version(package) != version:
                raise ValueError(package + " pin mismatch")
        for line in (self.repo / "requirements-yardsticks.txt").read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            requirement = Requirement(line)
            if requirement.url:
                direct = json.loads(metadata.distribution(requirement.name).read_text("direct_url.json"))
                if direct["vcs_info"]["commit_id"] != requirement.url.rsplit("@", 1)[1]:
                    raise ValueError(requirement.name + " source pin mismatch")
            elif not requirement.specifier.contains(metadata.version(requirement.name)):
                raise ValueError(requirement.name + " version pin mismatch")
        tree = self.scratch / "Simons"
        direct = json.loads(metadata.distribution("simons").read_text("direct_url.json"))
        if (direct.get("url") != tree.as_uri() or direct.get("dir_info", {}).get("editable") is not True
                or pathlib.Path(simons.__file__).resolve() != tree / "src/simons/__init__.py"
                or pathlib.Path(adapt.__file__).resolve() != tree / "src/simons/training/adapt.py"):
            raise ValueError("installed Simons is not the patched editable tree")
        if torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8" or not torch.cuda.is_available():
            raise ValueError("requires CUDA and torch 2.8.0+cu128 / CUDA 12.8")
        drivers = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip().splitlines()
        if not drivers:
            raise ValueError("NVIDIA driver unavailable")
        facts = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                     cuda_available=True, nvidia_driver=drivers, simons_path=str(tree),
                     repo_sha=self.c["repo_sha"], upstream_pin=self.c["upstream_pin"],
                     patch_bundle_sha256=self.c["patch_bundle_sha256"])
        freeze = subprocess.check_output([self.python, "-m", "pip", "freeze"], text=True)
        temp = self.scratch / "environment.txt"
        temp.write_text(freeze + "\n# runtime " + json.dumps(facts, sort_keys=True) + "\n")
        os.replace(temp, self.root / "environment.txt")

    def inputs(self):
        path = self.scratch / "tasks.yaml"
        copy_cloud(self.c["tasks_uri"], path)
        if sha(path) != self.c["tasks_sha256"]:
            raise ValueError("task YAML sha256 mismatch")
        os.replace(path, self.root / "tasks.yaml")
        base = self.scratch / "base"
        base.mkdir()
        for name in FILES:
            if self.c["checkpoint_uri"]:
                copy_cloud(self.c["checkpoint_uri"] + "/" + name, base / name)
            else:
                from huggingface_hub import hf_hub_download
                hf_hub_download(self.c["hub_weights"], name, revision=self.c["hub_revision"],
                                token=False, local_dir=str(base))
            if not (base / name).stat().st_size:
                raise ValueError("empty base export")
        hashes = {n: sha(base / n) for n in FILES}
        prior = self.root / "checkpoint-sha256.json"
        if prior.exists() and read(prior) != hashes:
            raise ValueError("different checkpoint contents under the same adaptation name")
        recorded = read_object(self.root / "completed.json").get("checkpoint_sha256")
        if recorded is not None and recorded != hashes:
            raise ValueError("different checkpoint contents under the same adaptation name")
        if not prior.exists() and recorded is None and any(
                p.is_file() for p in (self.root / "tasks").rglob("*")):
            raise ValueError("base hash identity missing from restored state; refusing to rebind existing results")
        atomic(prior, hashes)
        self.base.mkdir(exist_ok=True)
        for name in FILES:
            shutil.copyfile(base / name, self.base / name)

    def publish(self, path):
        # Snapshot an atomically replaced trainer file before hashing/uploading.
        with tempfile.TemporaryDirectory(dir=self.work) as temp:
            snapshot = pathlib.Path(temp) / path.name
            shutil.copyfile(path, snapshot)
            digest = sha(snapshot)
            uri = self.uri + "/blobs/" + digest
            copy_cloud(snapshot, uri)
        return dict(sha256=digest, uri=uri)

    def restore(self, record, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and sha(path) == record["sha256"]:
            return
        temp = path.with_suffix(path.suffix + ".download")
        copy_cloud(record["uri"], temp)
        if sha(temp) != record["sha256"]:
            raise ValueError("restored artifact sha256 mismatch")
        os.replace(temp, path)

    def bundle(self, name, state, work):
        output = work / "bundle"
        if not output.exists():
            command = [self.python, "-m", "demand_forecast.bench.adaptation", "--tasks",
                       str(self.root / "tasks.yaml"), "--task", name, "--phase", "development",
                       "--output", str(output)]
            self.command(command, state / "bundle.log")
        manifest = read(output / "manifest.json")
        version = manifest.get("schema_version")
        if type(version) is not int or version != 2:
            raise ValueError("bundle schema_version must be 2, found " + repr(version))
        if manifest["task"]["name"] != name or manifest["phase"] != "development":
            raise ValueError("bundle manifest identity mismatch")
        for file, expected in manifest["files"].items():
            path = child(output, file)
            if sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError("bundle payload mismatch: " + file)
        saved = state / "bundle-manifest.json"
        expected = {sha(saved)} if saved.exists() else set()
        completion = read_object(self.root / "completed.json")
        if completion.get("inputs") == self.c:
            recorded = completion.get("artifacts", {}).get(saved.relative_to(self.root).as_posix())
            if recorded:
                expected.add(recorded)
        for receipt in state.glob("fits/*/fit.json"):
            if valid(receipt):
                expected.add(read(receipt.parent / "adaptation.json")["inputs"]["bundle_manifest_sha256"])
        if expected and expected != {sha(output / "manifest.json")}:
            raise ValueError("rebuilt bundle changed; refusing to reuse fits")
        if not expected and any(p.is_file() for folder in ("frozen", "fits")
                                for p in (state / folder).rglob("*")):
            raise ValueError("bundle identity missing from restored state; refusing to rebind existing results")
        copy_atomic(output / "manifest.json", saved)
        return output, manifest

    def command(self, command, log, env=None):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            raise ValueError("command exit " + str(result.returncode) + "; see " + str(log))

    def score(self, name, mode, checkpoint, state, work, label):
        status = state / "status.json"
        cached = valid(status)
        if cached:
            return cached
        state.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            command = [self.python, str(self.repo / "scripts/fev_yardstick.py"), "--model", "simons",
                       "--checkpoint", str(checkpoint), "--processing", mode, "--tasks",
                       str(self.root / "tasks.yaml"), "--names", name, "--model-name", label,
                       "--out-dir", str(work), "--resume", "--keep-going"]
            self.command(command, state / "score.log")
            record = read(child(work, name + "/run.json"))
            summary = record["summary"]
            with (work / "summaries.csv").open() as f:
                rows = list(csv.DictReader(f))
            if (len(rows) != 1 or rows[0]["task"] != name or summary["task"] != name
                    or summary["model_name"] != label or rows[0]["model_name"] != label
                    or rows[0]["run_identity"] != record["run_identity"]
                    or not math.isfinite(float(summary["SQL"]))
                    or float(rows[0]["SQL"]) != float(summary["SQL"])
                    or not record["window_sha256"]
                    or not all(sha(child(child(work, name), p)) == h
                               for p, h in record["window_sha256"].items())):
                raise ValueError("invalid outer scoring evidence")
            for filename, source in (("run.json", child(work, name + "/run.json")),
                                     ("summaries.csv", work / "summaries.csv")):
                copy_atomic(source, state / filename)
            report = dict(status="ok", sql=float(summary["SQL"]), wall_seconds=time.monotonic() - started,
                          hashes={n: sha(state / n) for n in ("run.json", "summaries.csv")})
        except Exception as error:
            report = dict(status="failed", error=str(error), wall_seconds=time.monotonic() - started)
        atomic(status, report)
        return report

    def checkpoints(self, output, state):
        path = state / "checkpoints.json"
        refs = read(path) if path.exists() else {}
        for checkpoint in sorted((output / "checkpoints").glob("checkpoint-*.pt")):
            stat = checkpoint.stat()
            signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if self.checkpoint_stats.get(checkpoint) == signature:
                continue
            digest = sha(checkpoint)
            if refs.get(checkpoint.name, {}).get("sha256") != digest:
                refs[checkpoint.name] = self.publish(checkpoint)
                atomic(path, refs)
            self.checkpoint_stats[checkpoint] = signature

    def acknowledge(self, output, state, control):
        request, done = self.run / "checkpoint-request", control / "checkpoint-done"
        if (done.exists() and request.exists() and request.read_text().strip()
                and done.read_text().strip() == request.read_text().strip()):
            self.checkpoints(output, state)
            copy_atomic(done, self.run / "checkpoint-done")

    def train(self, bundle, output, state, params, lr, seed):
        refs = state / "checkpoints.json"
        if refs.exists():
            for name, record in read(refs).items():
                self.restore(record, child(output, "checkpoints/" + name))
        output.mkdir(parents=True, exist_ok=True)
        command = [self.python, "-m", "simons.training.adapt", "--bundle", str(bundle),
                   "--base", str(self.base), "--output", str(output), "--params", params,
                   "--lr-multiplier", str(lr), "--seed", str(seed), "--max-updates",
                   str(self.c["max_updates"]), "--device", "cuda"]
        # Keep control files outside trainer output: nonempty fresh outputs are refused.
        control = output.parent / (output.name + "-control")
        control.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, RUN_DIR=str(control))
        with (state / "train.log").open("a") as f:
            process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, env=env)
            try:
                while process.poll() is None:
                    request = self.run / "checkpoint-request"
                    if request.exists():
                        nonce = request.read_text().strip()
                        if nonce:
                            (control / "checkpoint-request").write_text(nonce)
                    self.checkpoints(output, state)
                    self.acknowledge(output, state, control)
                    time.sleep(0.2)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                self.checkpoints(output, state)
                self.acknowledge(output, state, control)
        if process.returncode:
            raise ValueError("trainer exit " + str(process.returncode) + "; see train.log")

    def fit(self, bundle, manifest, state, output, params, lr, seed):
        state.mkdir(parents=True, exist_ok=True)
        cached = valid(state / "fit.json")
        if cached:
            report = read(state / "adaptation.json")
            if (report["identity"]["bundle_fingerprint"] != manifest["fingerprint"]
                    or report["inputs"]["bundle_manifest_sha256"] != sha(bundle / "manifest.json")):
                result = dict(status="failed", error="cached fit bundle identity mismatch")
                atomic(state / "fit.json", result)
                return None, result
            return report, cached
        try:
            self.train(bundle, output, state, params, lr, seed)
            report = read(output / "adaptation.json")
            if (report["schema_version"] != 1 or report["seed"] != seed
                    or report["identity"]["bundle_fingerprint"] != manifest["fingerprint"]
                    or report["inputs"]["bundle_manifest_sha256"] != sha(bundle / "manifest.json")
                    or report["decision"]["selected"] not in ("adapted", "frozen_base")
                    or set(report["export"]) != set(FILES)
                    or any(sha(output / n) != h for n, h in report["export"].items())):
                raise ValueError("invalid trainer receipt/export identity")
            if report["decision"]["selected"] == "frozen_base" and report["export"] != read(self.root / "checkpoint-sha256.json"):
                raise ValueError("frozen decision changed base weights")
            exports = {n: self.publish(output / n) for n in FILES} if report["decision"]["selected"] == "adapted" else {}
            copy_atomic(output / "adaptation.json", state / "adaptation.json")
            result = dict(status="ok", exports=exports,
                          hashes={"adaptation.json": sha(state / "adaptation.json")})
        except Exception as error:
            report, result = None, dict(status="failed", error=str(error))
        atomic(state / "fit.json", result)
        return report, result

    def rows(self, rows):
        temp = self.root / "results.csv.tmp"
        with temp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.root / "results.csv")

    def execute(self):
        self.inputs()
        rows, fits, bundles, failed = [], {}, {}, False
        for name in self.c["names"]:
            task = self.root / "tasks" / slug(name)
            task.mkdir(parents=True, exist_ok=True)
            work = self.work / "tasks" / slug(name)
            work.mkdir(parents=True, exist_ok=True)
            frozen = {}
            try:
                bundle, manifest = self.bundle(name, task, work)
                bundles[name] = dict(schema_version=manifest["schema_version"],
                                     fingerprint=manifest["fingerprint"], manifest_sha256=sha(bundle / "manifest.json"))
                bundle_error = None
            except Exception as error:
                bundle_error, failed = str(error), True
                bundles[name] = dict(error=bundle_error)
            if bundle_error is None:
                for mode in self.c["modes"]:
                    frozen[mode] = self.score(name, mode, self.base, task / "frozen" / mode,
                                              work / "frozen" / mode, self.c["name"] + "-frozen-" + mode)
                    failed |= frozen[mode]["status"] != "ok"
            for params, lr, seed in itertools.product(self.c["params"], self.c["lr_multipliers"], self.c["seeds"]):
                key = params + "-lr" + str(lr) + "-seed" + str(seed)
                state, output = task / "fits" / key, work / "fits" / key
                row = dict(task=name, params=params, lr_multiplier=lr, seed=seed, status="failed")
                scores = {}
                for mode, score in frozen.items():
                    row["frozen_sql_" + mode] = score.get("sql")
                if bundle_error is None:
                    report, result = self.fit(bundle, manifest, state, output, params, lr, seed)
                else:
                    report, result = None, dict(status="failed", error=bundle_error)
                if report is None:
                    row.update(error=result["error"], decision_reason="fit_failed")
                    failed = True
                else:
                    decision = report["decision"]
                    row.update(status="ok", selected=decision["selected"], selected_update=decision["selected_update"],
                               decision_reason=decision["reason"], trainable_parameters=report["parameters"]["trainable_count"],
                               training_seconds=report["training_wall_seconds"], peak_gpu_memory_bytes=report["peak_cuda_memory_bytes"],
                               inner_evaluation_seconds=sum(e["wall_seconds"] for e in report["evaluations"]),
                               outer_evaluation_seconds=0.0)
                    for evaluation in report["evaluations"]:
                        for label, update in (("0", 0), ("selected", decision["selected_update"])):
                            if evaluation["update"] == update:
                                for i, window in enumerate(evaluation["windows"]):
                                    row["inner_" + label + "_window_" + str(i)] = window["score"]
                    for mode in self.c["modes"]:
                        base_score = frozen[mode]
                        row["frozen_sql_" + mode] = base_score.get("sql")
                        if decision["selected"] == "frozen_base":
                            score = dict(base_score, reused_frozen=True)
                            atomic(state / (mode + "-reuse.json"), score)
                        else:
                            try:
                                if not valid(state / mode / "status.json"):
                                    for n, ref in result["exports"].items():
                                        self.restore(ref, output / n)
                                score = self.score(name, mode, output, state / mode, work / "scores" / key / mode,
                                                   self.c["name"] + "-" + key + "-" + mode)
                            except Exception as error:
                                score = dict(status="failed", error=str(error))
                        row["outer_sql_" + mode] = score.get("sql")
                        scores[mode] = score
                        if decision["selected"] != "frozen_base":
                            row["outer_evaluation_seconds"] += score.get("wall_seconds", 0)
                        if score["status"] != "ok" or base_score["status"] != "ok":
                            failed, row["status"] = True, "failed"
                            row["error"] = score.get("error", base_score.get("error", "outer score failed"))
                fits[slug(name) + "/" + key] = dict(status=row["status"], fit=result, scores=scores)
                rows.append(row)
                self.rows(rows)
        artifacts = {p.relative_to(self.root).as_posix(): sha(p) for p in sorted(self.root.rglob("*"))
                     if p.is_file() and p.name != "completed.json" and not p.name.endswith(".tmp")}
        atomic(self.root / "completed.json", dict(schema=1, inputs=self.c, success=not failed,
               checkpoint_sha256=read(self.root / "checkpoint-sha256.json"), bundles=bundles, fits=fits,
               artifacts=artifacts, artifact_uri=self.uri))
        return int(failed)


def main():
    action, config, scratch, repo, python = sys.argv[1:]
    job = Job(read(pathlib.Path(config)), pathlib.Path(scratch), pathlib.Path(repo), python)
    if action == "preflight":
        return job.preflight()
    if action == "environment":
        job.environment()
        return 0
    if action == "execute":
        return job.execute()
    raise ValueError("unknown guest action")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        sys.exit("Adaptation job failed: " + " ".join(str(error).splitlines()))
"""


def _sha(value, length, label):
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"{label} must be full {length}-character lowercase hex")


def _uri(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"gs://[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9_./:@+=-]+", value)
        or any(p in (".", "..") for p in value[5:].split("/"))
    ):
        raise ValueError("URI must name an explicit gs:// object or prefix without dot components")


def generate(
    *,
    name,
    repo_sha,
    upstream_pin,
    tasks_uri,
    tasks_sha256,
    names,
    checkpoint_uri=None,
    hub_weights=None,
    hub_revision=None,
    params=("head", "all"),
    lr_multipliers=(0.02, 0.1),
    seeds=(11, 29),
    max_updates=500,
    processing="both",
):
    """Return one deterministic executable job; make no network/cloud calls."""
    _sha(repo_sha, 40, "repo SHA")
    _sha(upstream_pin, 40, "upstream pin")
    if upstream_pin != (BUNDLE / "PINNED").read_text().strip():
        raise ValueError("upstream pin must equal third_party/simons/PINNED")
    _sha(tasks_sha256, 64, "tasks sha256")
    _uri(tasks_uri)
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("invalid run name")
    if bool(checkpoint_uri) == bool(hub_weights):
        raise ValueError("choose exactly one checkpoint URI or Hub repository")
    if checkpoint_uri:
        _uri(checkpoint_uri)
        if hub_revision is not None:
            raise ValueError("Hub revision requires Hub repository")
    else:
        if not isinstance(hub_weights, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?", hub_weights
        ):
            raise ValueError("invalid Hub repository")
        _sha(hub_revision, 40, "Hub revision")
    for values, label in (
        (names, "names"),
        (params, "params"),
        (lr_multipliers, "lr multipliers"),
        (seeds, "seeds"),
    ):
        if not isinstance(values, (list, tuple)) or not values or len(set(values)) != len(values):
            raise ValueError(label + " must be a nonempty unique list")
    for task in names:
        if (
            not isinstance(task, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", task)
            or any(p in ("", ".", "..") for p in task.split("/"))
        ):
            raise ValueError("invalid task name")
    if any(p not in ("head", "all") for p in params):
        raise ValueError("params must contain head or all")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in lr_multipliers):
        raise ValueError("lr multipliers must be finite and positive")
    if any(type(x) is not int or x < 0 for x in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if type(max_updates) is not int or max_updates < 0:
        raise ValueError("max updates must be a nonnegative integer")
    if processing not in ("raw", "published", "both"):
        raise ValueError("processing must be raw, published or both")
    paths = [BUNDLE / p for p in ("PINNED", "NOTICE", "prepare.py")]
    patches = sorted((BUNDLE / "patches").glob("*.patch"))
    if not patches:
        raise ValueError("missing patch bundle")
    files = {p.relative_to(BUNDLE).as_posix(): p.read_text() for p in [*paths, *patches]}
    packed = gzip.compress(json.dumps(files, sort_keys=True).encode(), mtime=0)
    bundle_sha = hashlib.sha256(packed).hexdigest()
    guest = gzip.compress(GUEST.encode(), mtime=0)
    guest_sha = hashlib.sha256(guest).hexdigest()
    config = dict(
        name=name,
        repo_sha=repo_sha,
        upstream_pin=upstream_pin,
        tasks_uri=tasks_uri,
        tasks_sha256=tasks_sha256,
        names=list(names),
        params=list(params),
        lr_multipliers=list(lr_multipliers),
        seeds=list(seeds),
        max_updates=max_updates,
        checkpoint_uri=checkpoint_uri.rstrip("/") if checkpoint_uri else None,
        hub_weights=hub_weights,
        hub_revision=hub_revision,
        modes=["raw", "published"] if processing == "both" else [processing],
        patch_bundle_sha256=bundle_sha,
        guest_sha256=guest_sha,
    )
    encoded_config = base64.b64encode(json.dumps(config, sort_keys=True).encode()).decode()
    return f'''#!/usr/bin/env bash
# Adaptation {name}; repository {repo_sha}; upstream {upstream_pin}
set -euo pipefail
umask 077
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export STATE_URI}}"
case "$RUN_DIR" in /*) ;; *) echo 'RUN_DIR must be absolute' >&2; exit 2 ;; esac
export RUN_DIR
export HOME="${{HOME:-$RUN_DIR/home}}"
export HF_HOME="$RUN_DIR/adapt-cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE HF_TOKEN HUGGING_FACE_HUB_TOKEN
export UV_CACHE_DIR="$RUN_DIR/adapt-cache/uv"
export UV_PYTHON_INSTALL_DIR="$RUN_DIR/adapt-cache/python"
export PYTORCH_KERNEL_CACHE_PATH="$RUN_DIR/adapt-cache/kernels"
mkdir -p "$RUN_DIR"
scratch=$(mktemp -d "$RUN_DIR/adapt-install-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
python3 - "$scratch" <<'ADAPT_BUNDLE'
import base64, gzip, hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for encoded, expected, target in (("{base64.b64encode(guest).decode()}", "{guest_sha}", "adapt.py"),
                                   ("{base64.b64encode(packed).decode()}", "{bundle_sha}", None)):
    payload = base64.b64decode(encoded)
    assert hashlib.sha256(payload).hexdigest() == expected
    content = gzip.decompress(payload)
    if target:
        (root / target).write_bytes(content)
    else:
        for name, text in json.loads(content).items():
            path = root / "bundle" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
(root / "inputs.json").write_bytes(base64.b64decode("{encoded_config}"))
ADAPT_BUNDLE
repo="$scratch/repository"
upstream="$scratch/Simons"
python="$scratch/venv/bin/python"
guest() {{ "$1" "$scratch/adapt.py" "$2" "$scratch/inputs.json" "$scratch" "$repo" "$python"; }}
status=0
guest python3 preflight || status=$?
if [[ $status == 0 ]]; then exit 0; fi
if [[ $status != 10 ]]; then exit "$status"; fi
mkdir -p "$HOME" "$HF_DATASETS_CACHE" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PYTORCH_KERNEL_CACHE_PATH"
bootstrap_uv() {{
    if [[ ! -x $scratch/uv-bootstrap/bin/uv ]]; then
        python3 -m pip install --target "$scratch/uv-bootstrap" uv
    fi
}}
if ! python3 -m venv --system-site-packages "$scratch/venv"; then
    bootstrap_uv
    "$scratch/uv-bootstrap/bin/uv" venv --clear --seed --system-site-packages \\
        --python "$(command -v python3)" --no-python-downloads "$scratch/venv"
fi
if [[ $("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])') != 3.12 ]]; then
    bootstrap_uv
    "$scratch/uv-bootstrap/bin/uv" venv --clear --seed --system-site-packages --python 3.12 "$scratch/venv"
fi
git init "$repo"
git -C "$repo" remote add origin https://github.com/alessiosavi/demand-forecast.git
git -C "$repo" fetch --depth=1 origin {shlex.quote(repo_sha)}
git -C "$repo" checkout --detach FETCH_HEAD
[[ $(git -C "$repo" rev-parse HEAD) == {shlex.quote(repo_sha)} ]] || {{ echo 'repository SHA mismatch' >&2; exit 2; }}
git clone --filter=blob:none --no-checkout https://github.com/Abel-ai-lab/Simons.git "$upstream"
git -C "$upstream" sparse-checkout set --no-cone /src/ /configs/ /pyproject.toml /README.md /LICENSE /NOTICE
git -C "$upstream" checkout --detach {shlex.quote(upstream_pin)}
"$python" "$scratch/bundle/prepare.py" "$upstream" "$scratch/bundle"
"$python" -m pip install --upgrade pip
"$python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$python" -m pip install -r "$repo/requirements-yardsticks.txt" --editable "$repo[bench]" --editable "$upstream"
"$python" -m pip check
guest "$python" environment
guest "$python" execute
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("name", "repo-sha", "upstream-pin", "tasks-uri", "tasks-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-uri")
    source.add_argument("--hub-weights")
    parser.add_argument("--hub-revision")
    parser.add_argument("--params", nargs="+", choices=["head", "all"], default=["head", "all"])
    parser.add_argument("--lr-multipliers", nargs="+", type=float, default=[0.02, 0.1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29])
    parser.add_argument("--max-updates", type=int, default=500)
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
