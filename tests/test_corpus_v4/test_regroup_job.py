"""Run the generated CPU guest with real synthetic Parquet and a fake cloud CLI."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")
pytest.importorskip("pyarrow.parquet")

from scripts.corpus_v4 import make_regroup_job as jobs  # noqa: E402
from scripts.gcp.jobs import make_v5_job  # noqa: E402
from tests.helpers.corpus_v4_fixture import write_rows  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SHA = "a" * 40
PARENT = "gs://example-bucket/corpus-v4.1"
OUTPUT = "gs://example-bucket/corpus-v4.2"
STATE = "gs://example-bucket/state"
UNI = "univariate/a/train.parquet"
COV = "covariates/b/train.parquet"
ATTRIBUTION = ("univariate/a/LICENSE.md", "covariates/b/articles.csv", "LICENSE.md")
DEFAULTS = dict(threshold_bytes=96 << 20, group_bytes=32 << 20, group_rows=4096)
SMALL = dict(threshold_bytes=500, group_bytes=300, group_rows=2)

# Model the derive job's executable shims, with JSON listings, generations and CRC32C.
FAKE = r"""
import base64, json, os, pathlib, shutil, sys, tempfile

args = sys.argv[1:]
name = pathlib.Path(sys.argv[0]).name
cloud = pathlib.Path(os.environ['FAKE_CLOUD'])
with open(os.environ['FAKE_EVENTS'], 'a') as stream:
    stream.write(json.dumps([name, *args]) + '\n')

def local(uri):
    assert uri.startswith('gs://example-bucket/'), uri
    return cloud / uri[5:]

def crc32c(data):
    value = 0xffffffff
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (0x82f63b78 if value & 1 else 0)
    return base64.b64encode((value ^ 0xffffffff).to_bytes(4, 'big')).decode()

def listing(pattern):
    assert pattern.endswith('/**'), pattern
    rows = []
    for path in sorted(local(pattern[:-3]).rglob('*')):
        if path.is_file():
            relative = path.relative_to(cloud)
            key = relative.relative_to(relative.parts[0]).as_posix()
            values = dict(size=str(path.stat().st_size), crc32c=crc32c(path.read_bytes()),
                          generation=str(path.stat().st_ino))
            if len(rows) % 2:
                rows.append(dict(storage_url='gs://' + relative.as_posix(), **values))
            else:
                rows.append(dict(url='gs://' + relative.as_posix(), type='object',
                                 metadata=dict(name=key, **values)))
    return rows

if name == 'git':
    if args[0] == 'clone':
        repo = pathlib.Path(args[-1])
        shutil.copytree(os.environ['FAKE_REPO'], repo)
        (repo / '.git').mkdir()
    elif args[:2] == ['rev-parse', 'HEAD']:
        print('a' * 40)
    else:
        assert args[0] in ('fetch', 'checkout'), args
elif name == 'gcloud':
    assert args[0] == 'storage', args
    verb, *options = args[1:]
    fault = os.environ.get('FAKE_FAULT')
    if verb == 'ls':
        assert '--json' in options, options
        rows = listing(options[-1])
        if not rows:
            print('One or more URLs matched no objects.', file=sys.stderr)
            sys.exit(1)
        # gcloud emits one array per URL. Exercise the frame decoder too.
        print(json.dumps(rows[:1]))
        print(json.dumps(rows[1:]))
    elif verb == 'hash':
        assert options[:2] == ['--skip-md5', '--format=json'], options
        print(json.dumps([dict(url=p, digest_format='base64',
                               crc32c_hash=crc32c(pathlib.Path(p).read_bytes()))
                          for p in options[2:]]))
    elif verb == 'rsync':
        assert options[0] == '--recursive', options
        src, dst = local(options[-2]), pathlib.Path(options[-1])
        shutil.copytree(src, dst, dirs_exist_ok=True)
        if src.name == 'univariate' and fault == 'generation':
            shard = src / 'a/train.parquet'
            replacement = cloud / 'replacement'
            replacement.write_bytes(shard.read_bytes())
            replacement.replace(shard)
        if src.name == 'univariate' and fault == 'download':
            shard = dst / 'a/train.parquet'
            data = shard.read_bytes()
            shard.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    elif verb == 'cat':
        sys.stdout.buffer.write(local(options[-1]).read_bytes())
    elif verb == 'cp':
        src, dst = options[-2:]
        if not dst.startswith('gs://'):
            shutil.copyfile(local(src), dst)
            sys.exit(0)
        assert '--if-generation-match=0' in options, options
        path = local(dst)
        if fault == 'completion' and path.name == 'completed.json':
            sys.exit(79)
        if fault == 'partial' and dst.endswith('/covariates/b/train.parquet'):
            sys.exit(79)
        data = pathlib.Path(src).read_bytes()
        if dst.endswith('/covariates/b/train.parquet'):
            if fault == 'upload-size':
                data += b'damage'
            if fault == 'upload-crc':
                data = bytes([data[0] ^ 1]) + data[1:]
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cloud) as temporary:
            temporary.write(data)
            temporary.flush()
            try:
                os.link(temporary.name, path)
            except FileExistsError:
                sys.exit(18)
        if fault == 'ambiguous' and path.name in ('regroup-claim.json', 'attempt-status.json'):
            sys.exit(23)
    else:
        raise AssertionError(args)
elif args[:2] == ['-m', 'venv']:
    bin_dir = pathlib.Path(args[-1]) / 'bin'
    bin_dir.mkdir(parents=True)
    (bin_dir / 'python').symlink_to(sys.argv[0])
elif args[:2] == ['-m', 'pip']:
    assert args[2:] == ['install', 'pyarrow==25.0.1', 'numpy==2.2.6', 'PyYAML==6.0.3'], args
else:
    # Preserve stdin when running the guest's Python heredoc.
    os.execv(os.environ['FAKE_PYTHON'], [os.environ['FAKE_PYTHON'], *args])
"""


def generate(path, digest, **changes):
    arguments = dict(name="probe", repo_sha=SHA, parent_uri=PARENT, parent_digest=digest)
    arguments.update(output_uri=OUTPUT)
    arguments.update(changes)
    options = ["--output", str(path)]
    for key, value in arguments.items():
        options.extend(("--" + key.replace("_", "-"), value))
    assert jobs.main(options) == 0


def read_json(path):
    return json.loads(path.read_text())


def training_listing(env, work, uri, expected=None):
    frames = []
    for kind in ("univariate", "covariates"):
        frames.append(
            subprocess.check_output(
                ["gcloud", "storage", "ls", "--recursive", "--json", f"{uri}/{kind}/**"],
                env=env,
                text=True,
            )
        )
    listing_file = work / "listing.json"
    listing_file.write_text("".join(frames))
    namespace = {"__name__": "training_guest"}
    exec(make_v5_job.GUEST, namespace)

    if expected is None:
        # Derive the pinned digest from the fixture files emitted by the fake CLI.
        rows = []
        for line in listing_file.read_text().splitlines():
            for row in json.loads(line):
                meta = row.get("metadata", row)
                name = row.get("storage_url", row.get("url"))[len(uri) + 1 :]
                rows.append((name, meta["size"], meta["crc32c"]))
        expected = hashlib.sha256(
            "".join(f"{name}\t{size}\t{crc}\n" for name, size, crc in sorted(rows)).encode()
        ).hexdigest()
    return expected, namespace["listing"](listing_file, uri, expected)["objects"]


@pytest.fixture
def machine(tmp_path):
    cloud = tmp_path / "cloud"
    bucket = cloud / "example-bucket"
    parent = bucket / "corpus-v4.1"
    write_rows(parent / UNI, [[[float(i), -0.0, float(i)]] for i in range(6)], row_group_size=6)
    write_rows(parent / COV, [[1.0]], kind="covariates", covariates=[[[2.0]]])
    for name in (*ATTRIBUTION, "univariate/a/receipt.json", "covariates/b/stats.json"):
        (parent / name).write_bytes(b"fixture\r\n")
    (parent / "quarantine/a").mkdir(parents=True)
    (parent / "quarantine/a/LICENSE.md").write_text("quarantined attribution")
    source = tmp_path / "source"
    module = Path("scripts/corpus_v4/regroup_corpus.py")
    (source / module).parent.mkdir(parents=True)
    shutil.copyfile(REPO / module, source / module)
    bins = tmp_path / "bin"
    bins.mkdir()
    for name in ("gcloud", "git", "python3"):
        fake = bins / name
        fake.write_text(f"#!{sys.executable}\n{FAKE}")
        fake.chmod(0o755)
    events = tmp_path / "events.jsonl"
    env = dict(
        os.environ,
        PATH=f"{bins}{os.pathsep}{os.environ['PATH']}",
        FAKE_CLOUD=str(cloud),
        FAKE_EVENTS=str(events),
        FAKE_REPO=str(source),
        FAKE_PYTHON=sys.executable,
        RUN_DIR=str(tmp_path / "run"),
        STATE_URI=STATE,
    )
    digest, _ = training_listing(env, tmp_path, PARENT)
    events.write_text("")
    return dict(
        parent=parent,
        source=source / module,
        output=bucket / "corpus-v4.2",
        completion=bucket / "state/regroup/probe/completed.json",
        script=tmp_path / "job.sh",
        digest=digest,
        events=events,
        env=env,
        work=tmp_path,
    )


def invoke(machine, fault="", **env):
    if not machine["script"].exists():
        generate(machine["script"], machine["digest"])
    return subprocess.run(
        ["bash", str(machine["script"])],
        capture_output=True,
        text=True,
        env={**machine["env"], "FAKE_FAULT": fault, **env},
    )


def calls(machine):
    return [json.loads(line) for line in machine["events"].read_text().splitlines()]


def writes(machine):
    return [
        call[-1]
        for call in calls(machine)
        if call[:3] == ["gcloud", "storage", "cp"] and call[-1].startswith("gs://")
    ]


def test_cli_embeds_yaml_and_has_portable_shell(tmp_path):
    script = tmp_path / "job.sh"
    generate(script, "0" * 64)
    text = script.read_text()
    config = yaml.safe_load((REPO / "configs/corpus-v4.2.yaml").read_text())
    assert config["parent"] == "corpus-v4.1"
    assert {key: config[key] for key in DEFAULTS} == DEFAULTS
    for key, value in DEFAULTS.items():
        assert f'"{key}": {value}' in text
    assert os.access(script, os.X_OK)
    for shell in (shutil.which("bash"), "/bin/bash"):
        result = subprocess.run([shell, "-n", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_publication_order_identity_digest_groups_and_noop(machine, monkeypatch):
    config = machine["work"] / "config.yaml"
    config.write_text(yaml.safe_dump(dict(parent="corpus-v4.1", **SMALL)))
    monkeypatch.setattr(jobs, "CONFIG_PATH", config)
    generate(machine["script"], machine["digest"])
    config.unlink()  # The guest uses the parameter values captured at generation time.
    result = invoke(machine)
    assert result.returncode == 0, result.stderr
    output = machine["output"]
    claim = read_json(output / "regroup-claim.json")
    status = read_json(output / "attempt-status.json")
    completion = read_json(machine["completion"])
    expected_inputs = dict(
        name="probe",
        repo_sha=SHA,
        parent_uri=PARENT,
        parent_digest=machine["digest"],
        output_uri=OUTPUT,
        parameters=SMALL,
    )
    assert claim["inputs"] == expected_inputs
    identity = hashlib.sha256(
        json.dumps(expected_inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claim["identity"] == completion["identity"] == status["identity"] == identity
    assert read_json(output / "regroup.json")["parameters"] == SMALL
    assert completion["folder_groups"] == {
        "covariates/b": {"before": 1, "after": 1},
        "univariate/a": {"before": 1, "after": 3},
    }
    assert completion["corpus_digest"] == status["corpus_digest"]
    assert writes(machine) == [
        f"{OUTPUT}/regroup-claim.json",
        *(f"{OUTPUT}/{name}" for name in sorted((UNI, COV, *ATTRIBUTION))),
        f"{OUTPUT}/regroup.json",
        f"{OUTPUT}/attempt-status.json",
        f"{STATE}/regroup/probe/completed.json",
    ]
    for name in ATTRIBUTION:
        assert (output / name).read_bytes() == (machine["parent"] / name).read_bytes()
    assert not list(output.rglob("receipt.json"))
    assert not list(output.rglob("stats.json"))
    assert not (output / "quarantine").exists()
    _, objects = training_listing(
        machine["env"], machine["work"], OUTPUT, completion["corpus_digest"]
    )
    assert {name for name, _, _ in objects} == {COV, UNI, *ATTRIBUTION[:2]}

    machine["events"].write_text("")
    shutil.rmtree(machine["env"]["RUN_DIR"])
    retry = invoke(machine)
    assert retry.returncode == 0, retry.stderr
    assert writes(machine) == []
    assert not any(c[0] == "git" or c[:3] == ["gcloud", "storage", "rsync"] for c in calls(machine))


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("digest", "corpus digest mismatch"),
        ("generation", "generation changed"),
        ("download", "parent copy differs"),
        ("verify", "injected verification failure"),
    ],
)
def test_parent_or_c1_failure_publishes_nothing(machine, fault, message):
    if fault == "digest":
        generate(machine["script"], "0" * 64)
    elif fault == "verify":
        source = machine["source"]
        source.write_text(
            source.read_text().replace(
                "    verify_tree(args.parent, args.out, record)",
                "    raise ValueError('injected verification failure')",
            )
        )
    result = invoke(machine, fault)
    assert result.returncode != 0
    assert message in result.stderr
    assert writes(machine) == []


@pytest.mark.parametrize("existing", ["foreign", "regroup-claim.json"])
def test_foreign_output_or_claim_refused(machine, existing):
    machine["output"].mkdir()
    (machine["output"] / existing).write_text("{}")
    result = invoke(machine)
    assert result.returncode != 0
    assert "matching regroup claim" in result.stderr
    assert writes(machine) == []


@pytest.mark.parametrize("fault", ["partial", "completion"])
def test_interrupted_publication_resumes_without_rewriting(machine, fault):
    result = invoke(machine, fault)
    assert result.returncode != 0
    output = machine["output"]
    assert (output / "regroup-claim.json").is_file()
    assert (output / ATTRIBUTION[0]).exists() is (fault == "completion")
    assert (output / ATTRIBUTION[1]).exists()
    assert (output / COV).exists() is (fault == "completion")
    assert (output / "attempt-status.json").is_file() is (fault == "completion")
    assert not machine["completion"].exists()
    shutil.rmtree(machine["env"]["RUN_DIR"])
    machine["events"].write_text("")
    retry = invoke(machine)
    assert retry.returncode == 0, retry.stderr
    assert machine["completion"].is_file()
    assert f"{OUTPUT}/regroup-claim.json" not in writes(machine)
    assert f"{OUTPUT}/{ATTRIBUTION[1]}" not in writes(machine)
    guest, *commands = calls(machine)
    assert guest[:2] == ["python3", "-"]
    if fault == "completion":
        state = f"{STATE}/regroup/probe"
        assert writes(machine) == [f"{state}/completed.json"]
        # The only Python entry point is the guest itself. All other calls inspect
        # the publication records or create and read back the completion.
        assert all(command[:2] == ["gcloud", "storage"] for command in commands)
        assert [(command[2], command[-1]) for command in commands] == [
            ("ls", f"{state}/**"),
            ("ls", f"{OUTPUT}/**"),
            ("cat", f"{OUTPUT}/regroup-claim.json"),
            ("cat", f"{OUTPUT}/attempt-status.json"),
            ("cat", f"{OUTPUT}/regroup.json"),
            ("hash", f"{guest[2]}/attempt-status.json"),
            ("ls", f"{OUTPUT}/**"),
            ("cp", f"{state}/completed.json"),
            ("cat", f"{state}/completed.json"),
        ]
    else:
        assert [
            command[-2] for command in commands if command[:3] == ["gcloud", "storage", "rsync"]
        ] == [
            f"{PARENT}/univariate",
            f"{PARENT}/covariates",
        ]
        assert any(command[:2] == ["git", "clone"] for command in commands)
        assert any(
            command[:3] == ["python", "-m", "scripts.corpus_v4.regroup_corpus"]
            for command in commands
        )


def test_status_only_retry_completes_without_parent_or_regroup(machine, monkeypatch):
    config = machine["work"] / "config.yaml"
    config.write_text(yaml.safe_dump(dict(parent="corpus-v4.1", **SMALL)))
    monkeypatch.setattr(jobs, "CONFIG_PATH", config)
    failed = invoke(machine, "completion")
    assert failed.returncode != 0
    assert not machine["completion"].exists()
    status = read_json(machine["output"] / "attempt-status.json")
    shutil.rmtree(machine["env"]["RUN_DIR"])
    shutil.rmtree(machine["parent"])
    machine["events"].write_text("")

    retry = invoke(machine)
    assert retry.returncode == 0, retry.stderr
    assert [command for command in calls(machine)[1:] if command[0] != "gcloud"] == []
    assert not any(command[:3] == ["gcloud", "storage", "rsync"] for command in calls(machine))
    assert writes(machine) == [f"{STATE}/regroup/probe/completed.json"]
    completion = read_json(machine["completion"])
    assert completion["identity"] == status["identity"]
    assert completion["corpus_digest"] == status["corpus_digest"]
    assert completion["folder_groups"] == {
        "covariates/b": {"before": 1, "after": 1},
        "univariate/a": {"before": 1, "after": 3},
    }
    training_listing(machine["env"], machine["work"], OUTPUT, completion["corpus_digest"])


@pytest.mark.parametrize(
    "damage",
    [
        "identity",
        "repo-sha",
        "inventory",
        "size",
        "crc",
        "missing",
        "extra",
        "attribution",
        "ledger",
    ],
)
def test_status_only_retry_refuses_foreign_status_or_changed_output(machine, damage):
    failed = invoke(machine, "completion")
    assert failed.returncode != 0
    assert not machine["completion"].exists()
    output = machine["output"]
    status_path = output / "attempt-status.json"
    assert status_path.is_file()
    if damage in {"identity", "repo-sha", "inventory"}:
        status = read_json(status_path)
        if damage == "identity":
            status["identity"] = "0" * 64
        elif damage == "repo-sha":
            status["repo_sha"] = "b" * 40
        else:
            del status["output_listing"][ATTRIBUTION[0]]
        status_path.write_bytes(
            (json.dumps(status, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        )
    elif damage == "size":
        (output / UNI).write_bytes((output / UNI).read_bytes() + b"damage")
    elif damage == "crc":
        shard = output / UNI
        data = shard.read_bytes()
        shard.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    elif damage == "missing":
        (output / UNI).unlink()
    elif damage == "extra":
        (output / "unexpected").write_text("foreign")
    elif damage == "attribution":
        (output / "LICENSE.md").write_bytes(b"changed\r\n")
    else:
        ledger = output / "regroup.json"
        ledger.write_bytes(ledger.read_bytes() + b"\n")
    machine["events"].write_text("")

    retry = invoke(machine)
    assert retry.returncode != 0, retry.stderr
    message = {
        "identity": "regroup status has different identity",
        "repo-sha": "regroup status record differs",
        "inventory": "regroup ledger inventory differs from status",
        "ledger": "regroup ledger SHA-256 differs",
    }.get(damage, "status output listing differs")
    assert message in retry.stderr, retry.stderr
    assert writes(machine) == []
    assert not machine["completion"].exists()
    assert all(
        command[:3]
        in (
            ["gcloud", "storage", "ls"],
            ["gcloud", "storage", "cat"],
            ["gcloud", "storage", "hash"],
        )
        for command in calls(machine)[1:]
    )


@pytest.mark.parametrize("fault", ["upload-size", "upload-crc"])
def test_readback_refuses_corrupt_payload_before_status(machine, fault):
    result = invoke(machine, fault)
    assert result.returncode != 0
    assert "payload read-back differs" in result.stderr
    assert (machine["output"] / COV).is_file()
    assert not (machine["output"] / "regroup.json").exists()
    assert not (machine["output"] / "attempt-status.json").exists()
    assert not machine["completion"].exists()


@pytest.mark.parametrize("damage", ["same-size-data", "status", "extra"])
def test_completion_with_changed_listing_is_an_error(machine, damage):
    result = invoke(machine)
    assert result.returncode == 0, result.stderr
    output = machine["output"]
    if damage == "same-size-data":
        shard = output / UNI
        data = shard.read_bytes()
        shard.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    elif damage == "status":
        (output / "attempt-status.json").unlink()
    else:
        (output / "unexpected").write_text("foreign")
    machine["events"].write_text("")
    retry = invoke(machine)
    assert retry.returncode != 0
    assert "completed output listing differs" in retry.stderr
    assert writes(machine) == []


def test_ambiguous_create_only_marker_writes_reconcile(machine):
    result = invoke(machine, "ambiguous")
    assert result.returncode == 0, result.stderr
    assert machine["completion"].is_file()
