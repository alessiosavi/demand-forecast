"""Execute the derived guest against a local filesystem gcloud double, never GCP."""

import base64
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.corpus_v4 import make_derive_job as jobs
from tests.test_corpus_v4.test_derive_corpus import synthetic  # noqa: F401


def arguments(**changes):
    return (
        dict(
            name="probe",
            repo_sha="a" * 40,
            parent_uri="gs://offline/parent",
            output_uri="gs://offline/output",
        )
        | changes
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(name="../bad"),
        dict(name="UPPER"),
        dict(repo_sha="main"),
        dict(repo_sha="A" * 40),
        dict(parent_uri="gs://offline/p/"),
        dict(output_uri="gs://offline/p;touch"),
        dict(output_uri="gs://offline/parent"),
        dict(output_uri="gs://offline/parent/sub"),
        dict(output_uri="gs://offline"),
        dict(output_uri="gs://offline/../output"),
    ],
)
def test_arguments(changes):
    with pytest.raises(ValueError):
        jobs.generate(**arguments(**changes))


def test_cli_bundle_hash_pins_and_bash_syntax(tmp_path):
    path = tmp_path / "job.sh"
    args = [a for k, v in arguments().items() for a in ("--" + k.replace("_", "-"), v)]
    assert jobs.main([*args, "--output", str(path)]) == 0
    assert os.access(path, os.X_OK)
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    script = path.read_text()
    bundle = script.split("# guest-gzip-base64: ")[1].splitlines()[0]
    guest = gzip.decompress(base64.b64decode(bundle))
    assert guest.decode() == jobs.GUEST
    assert hashlib.sha256(guest).hexdigest() in script
    assert script == jobs.generate(**arguments())
    for pin in ("pyarrow==25.0.1", "numpy==2.2.6", "PyYAML==6.0.3"):
        assert pin in script
    assert "100 GB" in script and "1.5 hours" in script


@pytest.fixture
def machine(synthetic, monkeypatch):  # noqa: F811
    case = synthetic
    tmp = case["tmp"]
    cloud = tmp / "cloud"
    remote_parent = cloud / "offline/parent"
    remote_parent.parent.mkdir(parents=True)
    shutil.copytree(case["parent"], remote_parent)
    # The guest clones only code/configuration; all data come from this fake store.
    source = case["repo"]
    modules = source / "scripts/corpus_v4"
    real = Path(__file__).resolve().parents[2]
    for name in (
        "__init__.py",
        "derive_corpus.py",
        "row_inventory.py",
        "native.py",
        "freeze_manifest.py",
        "verify_published.py",
        "fetch_native.py",
    ):
        shutil.copyfile(real / "scripts/corpus_v4" / name, modules / name)
    bins = tmp / "bin"
    bins.mkdir()
    events = tmp / "events.jsonl"
    fake = (
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json, os, pathlib, shutil, sys, tempfile
args = sys.argv[1:]
name = pathlib.Path(sys.argv[0]).name
with open(os.environ["EVENTS"], "a") as file: file.write(json.dumps([name, *args]) + "\n")
def local(uri):
    assert uri.startswith("gs://offline/")
    return pathlib.Path(os.environ["CLOUD"]) / uri[5:]
def fault(boundary):
    if os.environ.get("CRASH") == boundary: sys.exit(79)
if name == "git":
    if args[0] == "clone":
        target = pathlib.Path(args[-1]); shutil.copytree(os.environ["SOURCE"], target)
        (target / ".git").mkdir()
    elif args[:2] == ["rev-parse", "HEAD"]: print("a" * 40)
elif name == "gcloud":
    assert args[0] == "storage"
    operation = args[1]
    if operation == "ls":
        root = local(args[-1].removesuffix("/**"))
        if not root.exists() or not any(p.is_file() for p in root.rglob("*")):
            print("One or more URLs matched no objects.", file=sys.stderr); sys.exit(1)
        for p in sorted(root.rglob("*")):
            if p.is_file(): print("gs://" + p.relative_to(os.environ["CLOUD"]).as_posix())
    elif operation == "cat":
        sys.stdout.buffer.write(local(args[-1]).read_bytes())
    elif operation == "rsync":
        assert args[2] == "--recursive" and args[-2].startswith("gs://")
        fault("readback")
        shutil.copytree(local(args[-2]), args[-1], dirs_exist_ok=True)
    elif operation == "cp":
        src, dst = args[-2:]
        if dst.startswith("gs://"):
            path = local(dst)
            if dst.endswith("/derivation/publishing.json"): fault("ownership")
            elif dst.endswith("/attempt-status.json"): fault("status")
            elif dst.endswith("/completed.json"): fault("completion")
            elif dst.endswith("/derivation/derivation.json"): fault("ledger")
            elif src.startswith("gs://"): fault("copy-parent")
            else: fault("upload")
            if (os.environ.get("FAIL_AFTER_UPLOADS") and dst.startswith("gs://offline/output/")
                    and not src.startswith("gs://") and not dst.endswith("/publishing.json")):
                output = local("gs://offline/output")
                uploaded = [p for p in output.rglob("*") if p.is_file()
                            and not p.relative_to(output).as_posix().startswith("univariate/")
                            and p.name != "publishing.json"]
                if len(uploaded) >= int(os.environ["FAIL_AFTER_UPLOADS"]): sys.exit(79)
            path.parent.mkdir(parents=True, exist_ok=True)
            source = local(src) if src.startswith("gs://") else src
            if "--if-generation-match=0" in args:
                # GCS makes objects visible atomically. Keep temporary bytes outside
                # the fake prefix so list/read cannot observe an unfinished upload.
                with tempfile.NamedTemporaryFile(dir=os.environ["CLOUD"]) as temporary:
                    temporary.write(pathlib.Path(source).read_bytes()); temporary.flush()
                    try: os.link(temporary.name, path)
                    except FileExistsError: sys.exit(18)
            else: shutil.copyfile(source, path)
            if os.environ.get("AMBIGUOUS") and dst.endswith(os.environ["AMBIGUOUS"]): sys.exit(23)
        else:
            fault("download")
            source = local(src)
            if "--recursive" in args: shutil.copytree(source, pathlib.Path(dst) / source.name, dirs_exist_ok=True)
            else: shutil.copyfile(source, dst)
    else: raise AssertionError(args)
elif args[:2] == ["-m", "venv"]:
    target = pathlib.Path(args[-1]) / "bin"; target.mkdir(parents=True, exist_ok=True)
    python = target / "python"
    if not python.exists(): python.symlink_to(sys.argv[0])
elif args[:2] == ["-m", "pip"]:
    if args[2] == "freeze": print("offline dependencies")
else:
    os.execv(os.environ["REAL_PYTHON"], [os.environ["REAL_PYTHON"], *args])
"""
    )
    for name in ("gcloud", "git", "python3"):
        path = bins / name
        path.write_text(fake)
        path.chmod(0o755)
    for key, value in dict(
        PATH=str(bins) + os.pathsep + os.environ["PATH"],
        EVENTS=str(events),
        CLOUD=str(cloud),
        SOURCE=str(source),
        REAL_PYTHON=sys.executable,
        RUN_DIR=str(tmp / "run"),
        STATE_URI="gs://offline/state",
    ).items():
        monkeypatch.setenv(key, value)
    script = tmp / "job.sh"
    script.write_text(jobs.generate(**arguments()))

    def invoke(**env):
        return subprocess.run(
            ["bash", str(script)], capture_output=True, text=True, env={**os.environ, **env}
        )

    def calls():
        return [json.loads(line) for line in events.read_text().splitlines()]

    return dict(
        case=case,
        invoke=invoke,
        calls=calls,
        output=cloud / "offline/output",
        state=tmp / "run/state/derive/probe",
        script=script,
        cloud=cloud,
    )


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_publication_order_identity_and_immutable_resume(machine):
    m = machine
    parent_before = snapshot(m["cloud"] / "offline/parent")
    result = m["invoke"]()
    assert result.returncode == 0, result.stderr
    calls = m["calls"]()
    writes = [
        c[-1]
        for c in calls
        if c[:3] == ["gcloud", "storage", "cp"] and c[-1].startswith("gs://offline/output/")
    ]
    assert writes[0].endswith("/derivation/publishing.json")
    assert writes[-1].endswith("/attempt-status.json")
    assert all(
        "--if-generation-match=0" in c
        for c in calls
        if c[:3] == ["gcloud", "storage", "cp"] and c[-1].startswith("gs://offline/output/")
    )
    assert len([c for c in calls if c[:3] == ["gcloud", "storage", "rsync"]]) == 1
    downloads = [
        c[-2]
        for c in calls
        if c[:3] == ["gcloud", "storage", "cp"]
        and c[-2].startswith("gs://offline/parent/")
        and not c[-1].startswith("gs://")
    ]
    assert set(downloads) == {
        "gs://offline/parent/" + f
        for f in (
            "corpus-identity.json",
            "attempt-status.json",
            *[r["source_folder"] for r in m["case"]["contract"]["releases"]],
        )
    }
    status = json.loads((m["output"] / "attempt-status.json").read_text())
    assert status["attempt_id"].startswith("derived-probe-")
    assert status["complete"] is False and status["exit_code"] == -1
    assert status["parent_attempt_id"] == "parent"
    assert (
        status["derivation_sha256"]
        == hashlib.sha256((m["output"] / "derivation/derivation.json").read_bytes()).hexdigest()
    )
    assert (
        json.loads((m["state"] / "completed.json").read_text())["derivation_sha256"]
        == status["derivation_sha256"]
    )
    assert snapshot(m["cloud"] / "offline/parent") == parent_before
    before, offset = snapshot(m["output"]), len(calls)
    assert m["invoke"]().returncode == 0
    assert snapshot(m["output"]) == before
    assert not any(
        c[:3] == ["gcloud", "storage", "cp"] and c[-1].startswith("gs://offline/output/")
        for c in m["calls"]()[offset:]
    )


def test_volatile_measurements_are_saved_in_state_only(machine):
    m = machine
    result = m["invoke"]()
    assert result.returncode == 0, result.stderr
    completion = json.loads((m["state"] / "completed.json").read_text())
    assert completion["measurements_source"] == "local"
    metrics = completion["measurements"]
    assert set(metrics) == {r["folder"] for r in m["case"]["contract"]["releases"]}
    for folder, measurement in metrics.items():
        assert set(measurement) == {"wall_seconds", "peak_rss_bytes"}
        assert measurement["wall_seconds"] > 0 and measurement["peak_rss_bytes"] > 0
        stats = json.loads((m["output"] / folder / "stats.json").read_text())
        assert "wall_seconds" not in stats["measurement"]
        assert "peak_rss_bytes" not in stats["measurement"]
    ledger = (m["output"] / "derivation/derivation.json").read_text()
    assert "wall_seconds" not in ledger and "peak_rss_bytes" not in ledger
    assert (
        json.loads((m["cloud"] / "offline/state/derive/probe/completed.json").read_text())
        == completion
    )
    remote = m["cloud"] / "offline/state/derive/probe/completed.json"
    remote.write_text(json.dumps(dict(completion, measurements={})))
    before = snapshot(m["output"])
    result = m["invoke"]()
    assert result.returncode == 0, result.stderr
    assert json.loads((m["state"] / "completed.json").read_text()) == completion
    assert json.loads(remote.read_text()) == completion
    assert snapshot(m["output"]) == before


def test_state_loss_retry_preserves_previous_completion_measurements_only(machine):
    m = machine
    result = m["invoke"]()
    assert result.returncode == 0, result.stderr
    remote = m["cloud"] / "offline/state/derive/probe/completed.json"
    original = json.loads(remote.read_text())
    assert original["measurements"]
    previous = dict(
        original, status_sha256="0" * 64, verification={"untrusted": True}, artifacts={}
    )
    remote.write_text(json.dumps(previous))
    before, offset = snapshot(m["output"]), len(m["calls"]())
    shutil.rmtree(m["case"]["tmp"] / "run")

    result = m["invoke"]()

    assert result.returncode == 0, result.stderr
    completion = json.loads(remote.read_text())
    assert completion["measurements"] == original["measurements"]
    assert completion["measurements_source"] == "previous-completion"
    assert completion == dict(original, measurements_source="previous-completion")
    assert json.loads((m["state"] / "completed.json").read_text()) == completion
    assert snapshot(m["output"]) == before
    assert not any(
        c[:3] == ["gcloud", "storage", "cp"] and c[-1].startswith("gs://offline/output/")
        for c in m["calls"]()[offset:]
    )


@pytest.mark.parametrize(
    "mode",
    [
        "different-derivation",
        "different-inputs",
        "unparsable",
        "non-object",
        "invalid-schema",
        "invalid-measurements",
        "missing",
    ],
)
def test_state_loss_retry_discards_untrusted_previous_measurements(machine, mode):
    m = machine
    result = m["invoke"]()
    assert result.returncode == 0, result.stderr
    remote = m["cloud"] / "offline/state/derive/probe/completed.json"
    previous = json.loads(remote.read_text())
    assert previous["measurements"]
    if mode == "different-derivation":
        previous["derivation_sha256"] = "b" * 64
    elif mode == "different-inputs":
        previous["inputs"] = dict(previous["inputs"], name="other")
    elif mode == "invalid-schema":
        previous["schema"] = True
    elif mode == "invalid-measurements":
        previous["measurements"] = ["untrusted"]
    if mode == "unparsable":
        remote.write_text("{")
    elif mode == "non-object":
        remote.write_text("[]")
    elif mode == "missing":
        remote.unlink()
    else:
        remote.write_text(json.dumps(previous))
    before = snapshot(m["output"])
    shutil.rmtree(m["case"]["tmp"] / "run")

    result = m["invoke"]()

    assert result.returncode == 0, result.stderr
    completion = json.loads(remote.read_text())
    assert completion["measurements"] == {}
    assert completion["measurements_source"] == "unavailable"
    assert json.loads((m["state"] / "completed.json").read_text()) == completion
    assert snapshot(m["output"]) == before


@pytest.mark.parametrize(
    "boundary",
    [
        "download",
        "ownership",
        "copy-parent",
        "upload",
        "readback",
        "ledger",
        "status",
        "completion",
    ],
)
def test_crash_boundaries_retry_and_status_last(machine, boundary):
    m = machine
    result = m["invoke"](CRASH=boundary)
    assert result.returncode != 0, result.stdout
    assert (m["output"] / "attempt-status.json").exists() is (boundary == "completion")
    published = snapshot(m["output"]) if boundary == "completion" else None
    assert not (m["state"] / "completed.json").exists()
    retry = m["invoke"]()
    assert retry.returncode == 0, retry.stderr
    if published:
        assert snapshot(m["output"]) == published


@pytest.mark.parametrize(
    "suffix", ["/derivation/publishing.json", "/derivation/derivation.json", "/attempt-status.json"]
)
def test_ambiguous_success_reconciles_identical_bytes(machine, suffix):
    result = machine["invoke"](AMBIGUOUS=suffix)
    assert result.returncode == 0, result.stderr
    assert (machine["state"] / "completed.json").is_file()


@pytest.mark.parametrize(
    "mode", ["state-lost", "completion-present-status-missing", "partial-state-lost"]
)
def test_resume_reconciles_remote_without_trusting_local_completion(machine, mode):
    m = machine
    if mode == "partial-state-lost":
        assert m["invoke"](CRASH="ledger").returncode != 0
    else:
        result = m["invoke"]()
        assert result.returncode == 0, result.stderr
    if mode == "completion-present-status-missing":
        (m["output"] / "attempt-status.json").unlink()
    else:
        shutil.rmtree(m["case"]["tmp"] / "run")
    result = m["invoke"]()
    assert result.returncode == 0, result.stderr
    assert (m["output"] / "attempt-status.json").is_file()


@pytest.mark.parametrize(
    "mode",
    [
        "foreign",
        "different-marker",
        "different-status",
        "changed-inputs",
        "wrong-parent",
        "extra-object",
    ],
)
def test_refuse_foreign_or_changed_objects_without_mutation(machine, mode):
    m = machine
    if mode in ("different-status", "changed-inputs", "extra-object"):
        result = m["invoke"]()
        assert result.returncode == 0, result.stderr
    if mode == "foreign":
        m["output"].mkdir(parents=True)
        (m["output"] / "stray").write_text("foreign")
    elif mode == "different-marker":
        (m["output"] / "derivation").mkdir(parents=True)
        (m["output"] / "derivation/publishing.json").write_text("{}")
    elif mode == "different-status":
        (m["output"] / "attempt-status.json").write_text("{}")
    elif mode == "changed-inputs":
        m["script"].write_text(jobs.generate(**arguments(repo_sha="b" * 40)))
    elif mode == "extra-object":
        (m["output"] / "extra").write_text("extra")
    else:
        (m["cloud"] / "offline/parent/corpus-identity.json").write_text("{}")
    before = snapshot(m["output"])
    assert m["invoke"]().returncode != 0
    assert snapshot(m["output"]) == before


@pytest.mark.parametrize(
    "uri",
    [
        "gs://offline/parent/run",
        "gs://offline/output/run",
        "gs://offline",
        "gs://offline/../state",
        "https://offline/state",
    ],
)
def test_state_overlap_preflight_before_external_calls(machine, uri):
    m = machine
    assert m["invoke"](STATE_URI=uri).returncode != 0
    assert not any(c[0] in ("git", "gcloud") for c in m["calls"]())


def test_concurrent_same_identity_publishers_cannot_overwrite(machine):
    from concurrent.futures import ThreadPoolExecutor

    m = machine
    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = [
            pool.submit(m["invoke"], RUN_DIR=str(m["case"]["tmp"] / f"run-{i}")) for i in range(2)
        ]
        results = [attempt.result() for attempt in attempts]
    assert any(r.returncode == 0 for r in results), [r.stderr for r in results]
    before = snapshot(m["output"])
    for i in range(2):
        result = m["invoke"](RUN_DIR=str(m["case"]["tmp"] / f"run-{i}"))
        assert result.returncode == 0, result.stderr
    assert snapshot(m["output"]) == before


def test_damaged_readback_never_opens_training_gate(machine):
    m = machine
    assert m["invoke"](CRASH="readback").returncode != 0
    shard = m["output"] / "univariate/demo/train-00000.parquet"
    shard.write_bytes(b"damaged")
    assert m["invoke"]().returncode != 0
    assert not (m["output"] / "attempt-status.json").exists()


def test_lost_state_does_not_mask_different_published_identity(machine):
    m = machine
    result = m["invoke"]()
    assert result.returncode == 0, result.stderr
    shutil.rmtree(m["case"]["tmp"] / "run")
    m["script"].write_text(jobs.generate(**arguments(repo_sha="b" * 40)))
    before = snapshot(m["output"])
    assert m["invoke"]().returncode != 0
    assert snapshot(m["output"]) == before


@pytest.mark.parametrize("mode", ["orphan", "identity", "symlink"])
def test_invalid_local_state_refused_before_external_calls(machine, mode):
    m = machine
    m["state"].mkdir(parents=True)
    if mode == "orphan":
        (m["state"] / "orphan.json").write_text("{}")
    elif mode == "identity":
        (m["state"] / "inputs.json").write_text("{}")
    else:
        (m["state"] / "inputs.json").symlink_to(m["case"]["repo"] / "configs/corpus-v4.yaml")
    assert m["invoke"]().returncode != 0
    assert not any(c[0] in ("git", "gcloud") for c in m["calls"]())


@pytest.mark.parametrize(
    "stage",
    [
        "check_release",
        "filter_folder",
        "verify_filtered",
        "prepare_releases",
        "write_derivation",
        "verify_derived",
    ],
)
def test_library_boundary_failure_cannot_publish_status(machine, stage):
    m = machine
    module = m["case"]["repo"] / "scripts/corpus_v4/derive_corpus.py"
    with module.open("a") as stream:
        stream.write(f'''
import os as _test_os
_test_original = {stage}
def _test_failure(*args, **kwargs):
    result = _test_original(*args, **kwargs)
    if _test_os.environ.get("FAIL_STAGE") == "{stage}":
        raise RuntimeError("injected {stage} failure")
    return result
{stage} = _test_failure
''')
    assert m["invoke"](FAIL_STAGE=stage).returncode != 0
    assert not (m["output"] / "attempt-status.json").exists()
    result = m["invoke"]()
    assert result.returncode == 0, result.stderr


def test_output_listing_uses_flat_double_star_without_recursive_headers(monkeypatch):
    from types import SimpleNamespace

    namespace = {"__name__": "derived_guest_test"}
    exec(compile(jobs.GUEST, "derive-guest.py", "exec"), namespace)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0, stdout="gs://offline/output/derivation/publishing.json\n", stderr=""
        )

    monkeypatch.setattr(namespace["subprocess"], "run", run)
    assert namespace["Store"]("gs://offline/output").objects() == {"derivation/publishing.json"}
    assert calls == [["gcloud", "storage", "ls", "gs://offline/output/**"]]


@pytest.mark.parametrize("uploaded", range(15))
def test_every_partial_payload_upload_resumes_without_volatile_stats_drift(machine, uploaded):
    m = machine
    assert m["invoke"](FAIL_AFTER_UPLOADS=str(uploaded)).returncode != 0
    assert not (m["output"] / "attempt-status.json").exists()
    # Also exercise loss of the prepared local measurements and checkpoint state.
    shutil.rmtree(m["case"]["tmp"] / "run")
    result = m["invoke"]()
    assert result.returncode == 0, result.stderr
