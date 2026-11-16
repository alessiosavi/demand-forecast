"""Derived policy and streaming materialisation, using only synthetic local data."""

import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from scripts.corpus_v4 import derive_corpus as derive  # noqa: E402
from scripts.corpus_v4 import row_inventory as ri  # noqa: E402
from scripts.corpus_v4.fetch_native import inventory  # noqa: E402
from scripts.corpus_v4.freeze_manifest import read_manifest, validate_manifest  # noqa: E402
from scripts.corpus_v4.native import atomic_json, digest  # noqa: E402
from scripts.corpus_v4.verify_published import PinnedInventory, verify_published  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NAMES = ["eia930_ba_load_weather", "eia930_subregion_load_weather"]
REVIEWED = [
    (
        "037184bf268ce895a0b5082121a8c0538acf2fe0eeea839294efa0c0aecd4619",
        "d89f58357a5fcddd3383a02ce15cd98cc495d50faf05db3a04adef485fd57cf8",
    ),
    (
        "7cf607dae82da7159a298abe03ddb8e7fe0448162a221bb483655f5f71901cfa",
        "4981c27bd11f720f3dcaee6094dfa4707ea57eaf2ec3363f7de9be73ad3f976a",
    ),
]


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table(*, dictionary=False):
    result = pa.table(
        dict(
            item_id=["CISO|2021-2022", "PJM|2021-2022", "NYIS|2021-2022", "MISO|2021-2022"],
            source=["west", "excluded", "east", "central"],
            freq=["h"] * 4,
            target=[[1.0, None, float("nan")], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
            covariates=[[[10.0, None, float("nan")], [1.0, 2.0, 3.0]]] * 4,
            covariate_names=[["temperature", "wind"]] * 4,
            station=[1, 2, 3, 4],
        )
    )
    for name in ("item_id", "source", "freq") if dictionary else ():
        result = result.set_column(
            result.schema.get_field_index(name), name, result[name].dictionary_encode()
        )
    return result.replace_schema_metadata({b"custom": b"retained", b"notice": b"schema metadata"})


@pytest.fixture
def synthetic(tmp_path):
    repo, parent, out = (tmp_path / name for name in ("repo", "parent", "out"))
    config = repo / "configs/corpus-v4.1"
    config.mkdir(parents=True)
    original = read_manifest()
    allowed = copy.deepcopy(
        next(
            e
            for e in original["folders"]
            if e["status"] == "allowed" and e["folder"].startswith("univariate/")
        )
    )
    allowed["folder"] = "univariate/demo"
    entries = [allowed] + [
        copy.deepcopy(next(e for e in original["folders"] if e["folder"] == "covariates/" + name))
        for name in NAMES
    ]
    original["folders"] = entries
    (repo / "configs/corpus-v4.yaml").write_text(yaml.safe_dump(original, sort_keys=False))
    parent.mkdir()
    atomic_json(
        parent / "corpus-identity.json",
        {k: original[k] for k in ("repo_id", "revision", "folders")},
    )
    atomic_json(
        parent / "attempt-status.json", dict(attempt_id="parent", complete=False, exit_code=-1)
    )
    files, releases = [], []
    for entry in entries:
        name = entry["folder"].split("/")[-1]
        folder = parent / (
            ("quarantine/" if entry["status"] == "quarantined" else "") + entry["folder"]
        )
        folder.mkdir(parents=True)
        path = folder / "train-00000.parquet"
        data = (
            table()
            if entry["status"] == "quarantined"
            else pa.table(
                dict(item_id=["one"], source=["demo"], freq=["h"], target=[[1.0, 2.0, 3.0]])
            )
        )
        with pq.ParquetWriter(path, data.schema) as writer:
            writer.add_key_value_metadata({b"outside_schema": b"also retained"})
            writer.write_table(data, row_group_size=2)
        file = dict(
            path=entry["folder"] + "/" + path.name,
            size=path.stat().st_size,
            hash=digest(path),
            algorithm="sha256",
        )
        files.append(file)
        atomic_json(folder / "stats.json", {})
        atomic_json(
            folder / "receipt.json",
            dict(
                repo_id=original["repo_id"],
                revision=original["revision"],
                folder=entry["folder"],
                status=entry["status"],
                files=[file],
                stats_sha256=digest(folder / "stats.json"),
            ),
        )
        if entry["status"] != "quarantined":
            continue
        rules = config / ("rules-" + name + ".yaml")
        rules.write_bytes((REPO / "configs/corpus-v4.1" / rules.name).read_bytes())
        evidence = tmp_path / (name + "-inventory")
        ri.build_inventory(folder, evidence, folder_name="quarantine/" + entry["folder"])
        summary = ri.write_manifest(evidence / "inventory.csv", rules, evidence)
        manifest = config / ("row-manifest-" + name + ".csv")
        manifest.write_bytes((evidence / "row-manifest.csv").read_bytes())
        releases.append(
            dict(
                folder=entry["folder"],
                source_folder="quarantine/" + entry["folder"],
                rules=rules.relative_to(repo).as_posix(),
                rules_sha256=digest(rules),
                row_manifest=manifest.relative_to(repo).as_posix(),
                row_manifest_sha256=digest(manifest),
                inventory_sha256=summary["inventory_sha256"],
                rows=4,
                kept=3,
                excluded=1,
                parent_shards=[dict(path=path.name, size=file["size"], sha256=file["hash"])],
                entry=dict(reason="released", uncertain=True, uncertainty="rule L not claimed"),
            )
        )
    contract = dict(
        version="4.1",
        standard="F-row",
        parent=dict(
            manifest="configs/corpus-v4.yaml",
            manifest_sha256=digest(repo / "configs/corpus-v4.yaml"),
            repo_id=original["repo_id"],
            revision=original["revision"],
        ),
        releases=releases,
    )
    (repo / "configs/corpus-v4.1.yaml").write_text(yaml.safe_dump(contract, sort_keys=False))
    pinned = repo / "scripts/corpus_v4/pinned_inventory.json"
    pinned.parent.mkdir(parents=True)
    atomic_json(
        pinned, dict(repo_id=original["repo_id"], revision=original["revision"], files=files)
    )
    return dict(repo=repo, parent=parent, out=out, contract=contract, files=files, tmp=tmp_path)


def materialize(case):
    import shutil

    root = case["out"]
    records = derive.prepare_releases(case["parent"], root, case["repo"])
    shutil.copytree(case["parent"] / "univariate", root / "univariate")
    publishing = dict(
        inputs=dict(
            name="test",
            repo_sha="a" * 40,
            guest_sha256="b" * 64,
            parent_uri="gs://offline/parent",
            output_uri="gs://offline/output",
            environment=derive.ENVIRONMENT,
            schema=1,
        )
    )
    publishing["attempt_id"] = derive.attempt_id(publishing["inputs"])
    atomic_json(root / "derivation/publishing.json", publishing)
    derive.write_derivation(
        root,
        case["repo"],
        releases=records,
        publishing=publishing,
        parent_identity=case["parent"] / "corpus-identity.json",
        parent_status=json.loads((case["parent"] / "attempt-status.json").read_text()),
    )
    return root


def refresh_payload(root):
    path = root / "derivation/derivation.json"
    ledger = json.loads(path.read_text())
    ledger["payload"] = derive.payload_map(root)
    atomic_json(path, ledger)


def test_reviewed_inputs_and_deterministic_identity():
    contract = derive.read_contract(REPO)
    for release, (rules_hash, manifest_hash) in zip(contract["releases"], REVIEWED):
        assert release["rules_sha256"] == digest(REPO / release["rules"]) == rules_hash
        assert (
            release["row_manifest_sha256"]
            == digest(REPO / release["row_manifest"])
            == manifest_hash
        )
        rows = csv_rows(REPO / release["row_manifest"])
        assert len(rows) == release["rows"]
        assert Counter(r["action"] for r in rows) == {
            "keep": release["kept"],
            "exclude": release["excluded"],
        }
    identity = derive.derived_identity(REPO)
    assert identity == derive.derived_identity(REPO)
    assert len(identity["folders"]) == 99
    expected = [
        e["folder"]
        for e in read_manifest()["folders"]
        if e["status"] == "allowed" or e["folder"] in {r["folder"] for r in contract["releases"]}
    ]
    assert [e["folder"] for e in identity["folders"]] == expected
    assert derive.validate_derived_manifest(identity, REPO) == identity
    with pytest.raises(ValueError, match="unresolved"):
        validate_manifest(identity)
    with pytest.raises(ValueError, match="unresolved"):
        verify_published("unused", identity)
    with pytest.raises(ValueError, match="unresolved"):
        inventory(identity, PinnedInventory(identity["repo_id"], identity["revision"], []))


@pytest.mark.parametrize(
    "damage",
    [
        "parent",
        "repo",
        "revision",
        "duplicate",
        "unsafe",
        "status",
        "rule_f",
        "rule_l",
        "missing-derived",
        "hash",
        "counts",
        "boolean",
        "source",
        "unapproved",
        "strict-derived",
        "missing-release",
    ],
)
def test_derived_policy_refusals(synthetic, damage):
    case = synthetic
    identity = derive.derived_identity(case["repo"])
    entry = identity["folders"][1]
    if damage == "parent":
        path = case["repo"] / "configs/corpus-v4.yaml"
        path.write_text(path.read_text() + "\n")
    elif damage in ("repo", "revision"):
        identity["repo_id" if damage == "repo" else "revision"] = "wrong"
    elif damage == "duplicate":
        identity["folders"].append(copy.deepcopy(entry))
    elif damage == "unsafe":
        entry["folder"] = "covariates/../bad"
    elif damage == "status":
        entry["status"] = "quarantined"
    elif damage in ("rule_f", "rule_l"):
        entry[damage] = "no_identified_overlap"
    elif damage == "missing-derived":
        entry.pop("derived")
    elif damage == "hash":
        entry["derived"]["rules_sha256"] = "x" * 64
    elif damage == "counts":
        entry["derived"]["kept"] = 4
    elif damage == "boolean":
        entry["derived"]["excluded"] = True
    elif damage == "source":
        entry["derived"]["source_folder"] = "quarantine/covariates/wrong"
    elif damage == "unapproved":
        entry["folder"] = "covariates/wrong"
    elif damage == "strict-derived":
        identity["folders"][0]["derived"] = copy.deepcopy(entry["derived"])
    elif damage == "missing-release":
        identity["folders"].pop()
    with pytest.raises(ValueError):
        derive.validate_derived_manifest(identity, case["repo"])


@pytest.mark.parametrize("store_schema", [False, True])
@pytest.mark.parametrize("extra_footer", [False, True])
def test_filter_streams_metadata_nan_null_dictionary_and_determinism(
    synthetic, monkeypatch, store_schema, extra_footer
):
    case = synthetic
    release = case["contract"]["releases"][0]
    source, manifest = (
        case["parent"] / release["source_folder"],
        case["repo"] / release["row_manifest"],
    )
    data = table(dictionary=True)
    with pq.ParquetWriter(
        source / "train-00000.parquet", data.schema, store_schema=store_schema
    ) as writer:
        if extra_footer:
            writer.add_key_value_metadata({b"outside_schema": b"also retained"})
        writer.write_table(data, row_group_size=2)
    release["parent_shards"][0].update(
        size=(source / "train-00000.parquet").stat().st_size,
        sha256=digest(source / "train-00000.parquet"),
    )
    evidence = derive.check_release(source, release, case["repo"])
    assert evidence["counts_by_action"] == {"keep": 3, "exclude": 1}
    original = pq.ParquetFile
    groups = []

    class Guard:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

        def __getattr__(self, name):
            if name in {"read", "read_row_group", "read_row_groups"}:
                pytest.fail("unbounded read")
            return getattr(self.inner, name)

        def iter_batches(self, **kwargs):
            assert kwargs["batch_size"] == 1 and len(kwargs["row_groups"]) == 1
            groups.extend(kwargs["row_groups"])
            return self.inner.iter_batches(**kwargs)

    monkeypatch.setattr(pq, "ParquetFile", Guard)
    maps = [derive.filter_folder(source, manifest, case["tmp"] / f"filtered-{i}") for i in range(2)]
    assert maps[0] == maps[1]
    assert [r["source_row_index"] for r in maps[0]] == [0, 2, 3]
    assert [r["output_row_index"] for r in maps[0]] == [0, 1, 2]
    assert all(len(r["covariates_sha256"]) == 64 for r in maps[0])
    name = "train-00000.parquet"
    a, b = [case["tmp"] / f"filtered-{i}" / name for i in range(2)]
    assert a.read_bytes() == b.read_bytes()
    with original(source / name) as before, original(a) as after:
        assert (b"ARROW:schema" in (before.metadata.metadata or {})) is store_schema
        assert before.schema_arrow.equals(after.schema_arrow, check_metadata=True)
        assert before.metadata.metadata == after.metadata.metadata
    assert {0, 1} <= set(groups)
    derive.verify_filtered(case["tmp"] / "filtered-0", maps[0])


@pytest.mark.parametrize("duplicate_value", [b"retained", b"lost"])
@pytest.mark.parametrize("duplicate_shard", ["train-00000.parquet", "validation-00000.parquet"])
def test_filter_refuses_duplicate_footer_keys_before_writing(
    synthetic, duplicate_value, duplicate_shard
):
    case = synthetic
    release = case["contract"]["releases"][0]
    source = case["parent"] / release["source_folder"]
    data = table().replace_schema_metadata(
        pa.KeyValueMetadata([(b"custom", b"retained"), (b"custom", duplicate_value)])
    )
    path = source / duplicate_shard
    pq.write_table(data, path)
    raw = path.read_bytes()
    footer = raw[-8 - int.from_bytes(raw[-8:-4], "little") : -8]
    # Inspect the serialized entries independently of PyArrow's dictionary view.
    assert footer.count(b"\x18\x06custom\x18") == 2
    with pytest.raises(ValueError, match="duplicate.*custom"):
        derive.filter_folder(source, case["repo"] / release["row_manifest"], case["out"])
    assert not case["out"].exists() or not list(case["out"].iterdir())


@pytest.mark.parametrize("damage", ["duplicate", "order"])
def test_filter_detects_readback_entry_changes_hidden_by_dictionary_view(
    synthetic, monkeypatch, damage
):
    case = synthetic
    release = case["contract"]["releases"][0]
    source = case["parent"] / release["source_folder"]
    data = table().replace_schema_metadata({b"custom": b"retained", b"notice": b"separate"})
    pq.write_table(data, source / "train-00000.parquet")
    expected_metadata = pq.read_metadata(source / "train-00000.parquet").metadata
    original = pq.ParquetWriter

    @contextmanager
    def damaged_writer(path, schema, **kwargs):
        with original(path, schema, **kwargs) as writer:
            yield writer
        raw = path.read_bytes()
        start = len(raw) - 8 - int.from_bytes(raw[-8:-4], "little")
        footer = raw[start:-8]
        entry = b"\x18\x06custom\x18\x08retained\x00"
        metadata = b"\x19\x3c\x18"  # field 5, custom, notice and ARROW:schema entries
        assert footer.count(metadata) == 1 and footer.count(entry) == 1
        if damage == "duplicate":
            changed = footer.replace(metadata, b"\x19\x4c\x18").replace(entry, entry * 2)
        else:
            other = b"\x18\x06notice\x18\x08separate\x00"
            first, second = footer.index(entry), footer.index(other)
            changed = bytearray(footer)
            changed[first : first + len(entry)] = other
            changed[second : second + len(other)] = entry
            changed = bytes(changed)
        path.write_bytes(raw[:start] + changed + len(changed).to_bytes(4, "little") + b"PAR1")
        # Both old equality checks miss the changed serialized entry list.
        assert pq.read_metadata(path).metadata == expected_metadata
        assert schema.equals(pq.read_schema(path), check_metadata=True)

    monkeypatch.setattr(pq, "ParquetWriter", damaged_writer)
    with pytest.raises(ValueError, match="Parquet key-value metadata mismatch"):
        derive.filter_folder(source, case["repo"] / release["row_manifest"], case["out"])


@pytest.mark.parametrize("damage", ["schema", "footer"])
def test_filter_refuses_readback_metadata_difference(synthetic, monkeypatch, damage):
    case = synthetic
    release = case["contract"]["releases"][0]
    original = pq.ParquetWriter

    @contextmanager
    def damaged_writer(path, schema, **kwargs):
        with original(path, schema, **kwargs) as writer:
            if damage == "footer":
                writer.add_key_value_metadata({b"unexpected": b"footer entry"})
            yield writer
        if damage == "schema":
            data = pq.read_table(path).replace_schema_metadata({b"changed": b"schema"})
            with original(path, data.schema, **kwargs) as writer:
                writer.write_table(data)

    monkeypatch.setattr(pq, "ParquetWriter", damaged_writer)
    message = (
        "Arrow schema mismatch" if damage == "schema" else "Parquet key-value metadata mismatch"
    )
    with pytest.raises(ValueError, match=message):
        derive.filter_folder(
            case["parent"] / release["source_folder"],
            case["repo"] / release["row_manifest"],
            case["out"],
        )


def test_fresh_materialisations_have_identical_payload_and_ledger(synthetic):
    case = synthetic
    roots = [materialize({**case, "out": case["tmp"] / f"fresh-{i}"}) for i in range(2)]
    snapshots = [
        {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        for root in roots
    ]
    assert snapshots[0] == snapshots[1]
    for root in roots:
        for release in case["contract"]["releases"]:
            folder = root / release["folder"]
            measurement = json.loads((folder / "stats.json").read_text())["measurement"]
            assert "wall_seconds" not in measurement and "peak_rss_bytes" not in measurement
            receipt = json.loads((folder / "receipt.json").read_text())
            assert receipt["stats_sha256"] == digest(folder / "stats.json")
        derive.verify_derived(root, case["repo"], case["files"])


@pytest.mark.parametrize("legacy_volatile", [False, True])
def test_retry_reuses_published_stats_and_rechecks_stable_measurements(synthetic, legacy_volatile):
    case = synthetic
    derive.prepare_releases(case["parent"], case["out"], case["repo"])
    prior = {
        release["folder"]: json.loads((case["out"] / release["folder"] / "stats.json").read_text())
        for release in case["contract"]["releases"]
    }
    if legacy_volatile:
        for stats in prior.values():
            stats["measurement"].update(wall_seconds=0, peak_rss_bytes=1)
    metrics = {}
    retry = case["tmp"] / "retry"
    derive.prepare_releases(
        case["parent"], retry, case["repo"], prior_stats=prior, measurements=metrics
    )
    for folder, previous in prior.items():
        stats_path = retry / folder / "stats.json"
        assert stats_path.read_text() == json.dumps(previous, sort_keys=True, indent=2) + "\n"
        assert json.loads((retry / folder / "receipt.json").read_text())["stats_sha256"] == digest(
            stats_path
        )
        assert set(metrics[folder]) == {"wall_seconds", "peak_rss_bytes"}
        assert metrics[folder]["wall_seconds"] > 0 and metrics[folder]["peak_rss_bytes"] > 1
    prior[case["contract"]["releases"][0]["folder"]]["rows"] += 1
    with pytest.raises(ValueError, match="resumed stats differ from measured release"):
        derive.prepare_releases(
            case["parent"], case["tmp"] / "bad-retry", case["repo"], prior_stats=prior
        )


@pytest.mark.parametrize(
    "field",
    [
        "parent_shards",
        "inventory_sha256",
        "rules_sha256",
        "row_manifest_sha256",
        "manifest_bytes",
        "rows",
        "kept",
        "excluded",
    ],
)
def test_check_release_refusal_names_field(synthetic, field):
    case = synthetic
    release = copy.deepcopy(case["contract"]["releases"][0])
    if field == "parent_shards":
        release[field][0]["sha256"] = "0" * 64
    elif field == "manifest_bytes":
        path = case["repo"] / release["row_manifest"]
        path.write_text(path.read_text().replace("default", "changed"))
        release["row_manifest_sha256"] = digest(path)
    elif field in ("rows", "kept", "excluded"):
        release[field] += 1
    else:
        release[field] = "0" * 64
    with pytest.raises(ValueError, match="row_manifest" if field == "manifest_bytes" else field):
        derive.check_release(case["parent"] / release["source_folder"], release, case["repo"])


@pytest.mark.parametrize(
    "damage,message",
    [
        ("shard", "shard"),
        ("row_index", "row_index"),
        ("item_id", "item_id"),
        ("target_sha256", "target_sha256"),
        ("action", "action"),
        ("missing", "missing"),
        ("extra", "missing"),
        ("zero", "zero kept"),
    ],
)
def test_filter_refusals(synthetic, damage, message):
    case = synthetic
    release = case["contract"]["releases"][0]
    rows = csv_rows(case["repo"] / release["row_manifest"])
    if damage in ("shard", "row_index", "item_id", "target_sha256", "action"):
        rows[0][damage] = "wrong"
    elif damage == "missing":
        rows.pop()
    elif damage == "extra":
        rows.append(rows[-1])
    elif damage == "zero":
        for row in rows:
            row["action"] = "exclude"
    path = case["tmp"] / "bad.csv"
    csv_write(path, rows)
    with pytest.raises(ValueError, match=message):
        derive.filter_folder(case["parent"] / release["source_folder"], path, case["out"])


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (None, None),
        ("bytes", "hash mismatch: covariates/" + NAMES[0]),
        ("missing", "incomplete or extra published files: covariates/" + NAMES[0]),
        ("stats", "published stats hash mismatch: covariates/" + NAMES[0]),
        ("revision", "receipt differs from pinned inventory/manifest: covariates/" + NAMES[0]),
        ("omitted", "receipt differs from pinned inventory/manifest: covariates/" + NAMES[0]),
        ("receipt-extra", "receipt differs from pinned inventory/manifest: covariates/" + NAMES[0]),
        ("extra", "incomplete or extra published files: covariates/" + NAMES[0]),
        ("marker", "corpus-identity.json"),
        ("quarantine", "quarantine directory is forbidden"),
        ("extra-object", "payload object set/size/sha256 mismatch"),
        ("allowed-byte", "hash mismatch: univariate/demo/"),
        ("covariate", "read-back covariates_sha256 mismatch"),
        ("excluded", "kept count mismatch"),
        ("row-map", "row map source_row_index mismatch"),
        ("evidence", "evidence rules mismatch"),
        ("payload", "payload object set/size/sha256 mismatch"),
    ],
)
def test_derived_tree_verifies_and_refuses_damage(synthetic, damage, message):
    case = synthetic
    root = materialize(case)
    folder = root / "covariates" / NAMES[0]
    receipt = folder / "receipt.json"
    if damage == "bytes":
        path = folder / "train-00000.parquet"
        path.write_bytes(b"x" * path.stat().st_size)
    elif damage == "missing":
        (folder / "train-00000.parquet").unlink()
    elif damage == "stats":
        (folder / "stats.json").write_text("{}")
    elif damage in ("revision", "omitted", "receipt-extra"):
        value = json.loads(receipt.read_text())
        if damage == "receipt-extra":
            value["files"].append(
                {**value["files"][0], "path": "covariates/" + NAMES[0] + "/extra.parquet"}
            )
        else:
            value["revision" if damage == "revision" else "files"] = (
                "wrong" if damage == "revision" else []
            )
        atomic_json(receipt, value)
    elif damage == "extra":
        (folder / "train-extra.parquet").write_bytes(b"x")
    elif damage == "marker":
        (root / "corpus-identity.json").unlink()
    elif damage == "quarantine":
        (root / "quarantine").mkdir()
    elif damage == "extra-object":
        (root / "stray").write_text("x")
    elif damage == "allowed-byte":
        path = root / "univariate/demo/train-00000.parquet"
        path.write_bytes(b"x" * path.stat().st_size)
    elif damage in ("covariate", "excluded"):
        path = folder / "train-00000.parquet"
        data = pq.read_table(path)
        if damage == "covariate":
            data = data.set_column(
                data.schema.get_field_index("covariates"),
                "covariates",
                pa.array([[[42.0] * 3] * 2] * 3),
            )
        else:
            data = table()
        pq.write_table(data, path)
        # Repair file-level hashes to require the independent row evidence to catch it.
        value = json.loads(receipt.read_text())
        value["files"][0].update(size=path.stat().st_size, hash=digest(path))
        atomic_json(receipt, value)
        ledger = root / "derivation/derivation.json"
        value = json.loads(ledger.read_text())
        value["releases"][0]["shards"][0].update(size=path.stat().st_size, sha256=digest(path))
        value["payload"] = derive.payload_map(root)
        atomic_json(ledger, value)
    elif damage in ("row-map", "payload"):
        path = root / "derivation/derivation.json"
        value = json.loads(path.read_text())
        if damage == "row-map":
            value["releases"][0]["row_map"][0]["source_row_index"] = 1
        else:
            value["payload"].pop("corpus-identity.json")
        atomic_json(path, value)
    elif damage == "evidence":
        (root / "derivation" / ("rules-" + NAMES[0] + ".yaml")).write_text("wrong")
    if damage:
        # Keep payload hashes current so receipt/pinned-file and row checks run.
        if damage != "payload":
            refresh_payload(root)
        with pytest.raises((ValueError, OSError), match=message):
            derive.verify_derived(root, case["repo"], case["files"])
    else:
        result = derive.verify_derived(root, case["repo"], case["files"])
        assert result["folders"] == 3 and len(result["receipt_fingerprint"]) == 64
        assert result == derive.verify_derived(root, case["repo"], case["files"])


def test_derived_requires_shared_integrity_even_with_self_consistent_payload(synthetic):
    case = synthetic
    root = materialize(case)
    folder = root / "univariate/demo"
    path = folder / "train-00000.parquet"
    path.write_bytes(b"x" * path.stat().st_size)
    receipt = json.loads((folder / "receipt.json").read_text())
    receipt["files"][0]["hash"] = digest(path)
    atomic_json(folder / "receipt.json", receipt)
    refresh_payload(root)
    with pytest.raises(
        ValueError, match="receipt differs from pinned inventory/manifest: univariate/demo"
    ):
        derive.verify_derived(root, case["repo"], case["files"])


@pytest.mark.parametrize("damage", ["byte", "receipt", "missing", "extra"])
def test_stale_payload_ledger_is_refused(synthetic, damage):
    root = materialize(synthetic)
    folder = root / "univariate/demo"
    path = folder / "train-00000.parquet"
    if damage == "byte":
        path.write_bytes(b"x" * path.stat().st_size)
    elif damage == "receipt":
        (folder / "receipt.json").write_text("{}")
    elif damage == "missing":
        path.unlink()
    else:
        (folder / "extra.parquet").write_bytes(b"extra")
    with pytest.raises(ValueError, match="payload object set/size/sha256 mismatch"):
        derive.verify_derived(root, synthetic["repo"], synthetic["files"])


def test_readback_checks_covariates_independently(synthetic):
    case = synthetic
    release = case["contract"]["releases"][0]
    rows = derive.filter_folder(
        case["parent"] / release["source_folder"],
        case["repo"] / release["row_manifest"],
        case["out"],
    )
    rows[0]["covariates_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="covariates_sha256"):
        derive.verify_filtered(case["out"], rows)


def test_upstream_loader_reads_retained_sources(synthetic):
    upstream = os.environ.get("SIMONS_UPSTREAM")
    if not upstream:
        pytest.skip("SIMONS_UPSTREAM not set")
    pytest.importorskip("torch")
    case = synthetic
    release = case["contract"]["releases"][0]
    derive.filter_folder(
        case["parent"] / release["source_folder"],
        case["repo"] / release["row_manifest"],
        case["out"],
    )
    # Keep the clean loader isolated from the patched training guest's module cache.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from simons.training.covariates.windows import CovariateCorpus; "
            "dataset = CovariateCorpus(sys.argv[1]); "
            "assert set(dataset.sources) == {'west', 'east', 'central'}; "
            "assert len(dataset.recs) == 3",
            str(case["out"]),
        ],
        env={**os.environ, "PYTHONPATH": str(Path(upstream) / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_reviewed_long_rows_and_documented_uncertainties():
    contract = derive.read_contract(REPO)
    totals = []
    for release in contract["releases"]:
        kept = [r for r in csv_rows(REPO / release["row_manifest"]) if r["action"] == "keep"]
        totals.append(sum(json.loads(r["target_shape"])[-1] >= 16384 for r in kept))
    assert totals == [119, 96]
    docs = (REPO / "docs/corpus-v4.md").read_text().split("## Derived corpus v4.1\n", 1)[1]
    assert "d3e7e720f4158eab03ebcb021f1cce779cd92acc" in docs
    assert "conditional daylight" in docs and "unverified" in docs
    assert "gs://" not in docs and "/private/" not in docs and "/Users/" not in docs


def test_shared_integrity_preserves_v40_fingerprint(synthetic):
    case = synthetic
    parent = read_manifest(case["repo"] / "configs/corpus-v4.yaml")
    parent["folders"] = [e for e in parent["folders"] if e["status"] == "allowed"]
    root = case["tmp"] / "strict"
    import shutil

    shutil.copytree(case["parent"] / "univariate", root / "univariate")
    identity = {key: parent[key] for key in ("repo_id", "revision", "folders")}
    atomic_json(root / "corpus-identity.json", identity)
    files = [f for f in case["files"] if f["path"].startswith("univariate/")]
    receipts = {"univariate/demo": json.loads((root / "univariate/demo/receipt.json").read_text())}
    expected = hashlib.sha256(
        json.dumps(dict(identity=identity, receipts=receipts), sort_keys=True).encode()
    ).hexdigest()
    assert verify_published(root, parent, pinned_files=files) == dict(
        repo_id=identity["repo_id"],
        revision=identity["revision"],
        folders=1,
        files=1,
        receipt_fingerprint=expected,
    )


def test_derivation_record_refuses_boolean_counts(synthetic):
    root = materialize(synthetic)
    path = root / "derivation/derivation.json"
    value = json.loads(path.read_text())
    value["releases"][0]["excluded"] = True
    atomic_json(path, value)
    with pytest.raises(ValueError, match="excluded"):
        derive.verify_derived(root, synthetic["repo"], synthetic["files"])
