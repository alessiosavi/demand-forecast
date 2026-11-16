"""Offline F-row materialisation and verification of the reviewed v4.1 release."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

import yaml

from scripts.corpus_v4 import row_inventory as ri
from scripts.corpus_v4.fetch_native import HubFile, inventory
from scripts.corpus_v4.freeze_manifest import read_manifest, safe_folder, validate_manifest
from scripts.corpus_v4.native import atomic_json, digest, measure_folder
from scripts.corpus_v4.verify_published import PinnedInventory, verify_integrity

ENVIRONMENT = {"pyarrow": "25.0.1", "numpy": "2.2.6", "PyYAML": "6.0.3"}
RELEASES = {"covariates/eia930_ba_load_weather", "covariates/eia930_subregion_load_weather"}
DERIVED_FIELDS = (
    "source_folder",
    "rules_sha256",
    "row_manifest_sha256",
    "inventory_sha256",
    "rows",
    "kept",
    "excluded",
    "parent_shards",
)
EXCLUDED_PAYLOAD = {
    "attempt-status.json",
    "derivation/derivation.json",
    "derivation/publishing.json",
}


def _sha(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"invalid {field}: expected sha256")


def _relative(value):
    if not isinstance(value, str):
        raise ValueError("unsafe path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(p in ("", ".", "..") for p in path.parts)
        or "\\" in value
        or not path.parts
    ):
        raise ValueError(f"unsafe path: {value}")
    return value


def _repo_file(repo, name):
    path = Path(repo) / _relative(name)
    if not path.resolve().is_relative_to(Path(repo).resolve()) or path.is_symlink():
        raise ValueError(f"unsafe repository path: {name}")
    return path


def _identity(manifest):
    return {k: manifest[k] for k in ("repo_id", "revision", "folders")}


def _pinned(repo):
    return json.loads((Path(repo) / "scripts/corpus_v4/pinned_inventory.json").read_text())


def _block(release):
    return dict(standard="F-row", **{k: copy.deepcopy(release[k]) for k in DERIVED_FIELDS})


def _validate_block(block, folder):
    if (
        not isinstance(block, dict)
        or set(block) != {"standard", *DERIVED_FIELDS}
        or block["standard"] != "F-row"
    ):
        raise ValueError(f"incomplete derived block: {folder}")
    if block["source_folder"] != "quarantine/" + folder:
        raise ValueError(f"invalid source_folder: {folder}")
    for key in ("rules_sha256", "row_manifest_sha256", "inventory_sha256"):
        _sha(block[key], key)
    for key in ("rows", "kept", "excluded"):
        if type(block[key]) is not int or block[key] < 0:
            raise ValueError(f"invalid {key}: {folder}")
    if block["kept"] == 0 or block["kept"] + block["excluded"] != block["rows"]:
        raise ValueError(f"invalid counts: {folder}")
    shards = block["parent_shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"invalid parent_shards: {folder}")
    seen = set()
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != {"path", "size", "sha256"}:
            raise ValueError(f"invalid parent_shards: {folder}")
        name = shard["path"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"(?:train|validation)-[A-Za-z0-9._-]+\.parquet", name)
            or name in seen
            or type(shard["size"]) is not int
            or shard["size"] <= 0
        ):
            raise ValueError(f"invalid parent_shards: {folder}")
        seen.add(name)
        _sha(shard["sha256"], "parent_shards sha256")


def read_contract(repo_root):
    """Bind the release policy to repository bytes and the strict parent inventory."""
    repo = Path(repo_root)
    contract = yaml.safe_load((repo / "configs/corpus-v4.1.yaml").read_text())
    if (
        not isinstance(contract, dict)
        or set(contract) != {"version", "standard", "parent", "releases"}
        or contract["version"] != "4.1"
        or contract["standard"] != "F-row"
    ):
        raise ValueError("invalid derived contract")
    binding = contract["parent"]
    if (
        not isinstance(binding, dict)
        or set(binding) != {"manifest", "manifest_sha256", "repo_id", "revision"}
        or binding["manifest"] != "configs/corpus-v4.yaml"
    ):
        raise ValueError("invalid parent binding")
    parent_path = _repo_file(repo, binding["manifest"])
    parent = read_manifest(parent_path)
    if digest(parent_path) != binding["manifest_sha256"] or any(
        binding[k] != parent[k] for k in ("repo_id", "revision")
    ):
        raise ValueError("parent binding mismatch")
    pinned = _pinned(repo)
    groups = inventory(parent, PinnedInventory(**pinned))
    entries = {e["folder"]: e for e in parent["folders"]}
    releases = contract["releases"]
    if (
        not isinstance(releases, list)
        or len(releases) != 2
        or {r.get("folder") for r in releases} != RELEASES
    ):
        raise ValueError("unapproved release set")
    for release in releases:
        if set(release) != {"folder", "rules", "row_manifest", "entry", *DERIVED_FIELDS}:
            raise ValueError("incomplete release")
        folder = safe_folder(release["folder"])
        _validate_block(_block(release), folder)
        if entries[folder]["status"] != "quarantined" or entries[folder]["rule_l"] != "unresolved":
            raise ValueError("release is not an unresolved quarantined parent entry")
        name = folder.split("/")[-1]
        for field, filename in (
            ("rules", f"rules-{name}.yaml"),
            ("row_manifest", f"row-manifest-{name}.csv"),
        ):
            if (
                release[field] != "configs/corpus-v4.1/" + filename
                or digest(_repo_file(repo, release[field])) != release[field + "_sha256"]
            ):
                raise ValueError(f"{field}_sha256 mismatch")
        shards = [
            dict(path=Path(f.path).name, size=f.size, sha256=f.hash)
            for f in groups[folder]
            if f.algorithm == "sha256"
        ]
        if release["parent_shards"] != shards or any(
            not s["path"].endswith(".parquet") for s in shards
        ):
            raise ValueError("parent_shards differ from pinned inventory")
        entry = release["entry"]
        if (
            not isinstance(entry, dict)
            or set(entry) != {"reason", "uncertain", "uncertainty"}
            or entry["uncertain"] is not True
            or any(not isinstance(entry[k], str) or not entry[k] for k in ("reason", "uncertainty"))
        ):
            raise ValueError("invalid release entry")
    return contract


def derived_identity(repo_root):
    contract = read_contract(repo_root)
    parent = read_manifest(Path(repo_root) / contract["parent"]["manifest"])
    releases = {r["folder"]: r for r in contract["releases"]}
    result = _identity(copy.deepcopy(parent))
    result["folders"] = []
    for original in parent["folders"]:
        entry = copy.deepcopy(original)
        if entry["folder"] in releases:
            release = releases[entry["folder"]]
            entry.update(
                status="allowed",
                rule_f="row_filtered_no_identified_overlap",
                derived=_block(release),
                **release["entry"],
            )
        elif entry["status"] != "allowed":
            continue
        result["folders"].append(entry)
    return result


def validate_derived_manifest(manifest, repo_root):
    """Keep F-row permission separate from the unchanged strict v4.0 validator."""
    expected = derived_identity(repo_root)
    seen, released = set(), set()
    for entry in manifest["folders"]:
        folder = safe_folder(entry["folder"])
        if folder in seen or entry["status"] != "allowed":
            raise ValueError("duplicate folder or non-allowed derived entry")
        seen.add(folder)
        if "derived" in entry:
            if (
                folder not in RELEASES
                or entry["rule_f"] != "row_filtered_no_identified_overlap"
                or entry["rule_l"] != "unresolved"
            ):
                raise ValueError("unapproved derived release/rules")
            _validate_block(entry["derived"], folder)
            released.add(folder)
        else:
            validate_manifest(
                dict(repo_id=manifest["repo_id"], revision=manifest["revision"], folders=[entry])
            )
    if released != RELEASES or manifest != expected:
        raise ValueError("derived identity differs from parent/release contract")
    return manifest


def check_release(source_folder_dir, release, repo_root):
    """Replay the reviewed inventory and decisions against pinned source bytes."""
    source, repo = Path(source_folder_dir), Path(repo_root)
    expected = {s["path"] for s in release["parent_shards"]}
    if {p.name for p in source.glob("*.parquet")} != expected:
        raise ValueError("parent_shards names mismatch")
    for shard in release["parent_shards"]:
        path = source / shard["path"]
        if (
            path.is_symlink()
            or path.stat().st_size != shard["size"]
            or digest(path) != shard["sha256"]
        ):
            raise ValueError(f"parent_shards size/sha256 mismatch: {shard['path']}")
    rules = _repo_file(repo, release["rules"])
    manifest = _repo_file(repo, release["row_manifest"])
    for field, path in (("rules_sha256", rules), ("row_manifest_sha256", manifest)):
        if digest(path) != release[field]:
            raise ValueError(f"{field} mismatch")
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        header = ri.build_inventory(source, output, folder_name=release["source_folder"])
        if header["csv_sha256"] != release["inventory_sha256"]:
            raise ValueError("inventory_sha256 mismatch")
        summary = ri.write_manifest(output / "inventory.csv", rules, output)
        if (output / "row-manifest.csv").read_bytes() != manifest.read_bytes():
            raise ValueError("row_manifest bytes mismatch")
        counts = dict(
            rows=summary["rows"],
            kept=summary["counts_by_action"].get("keep", 0),
            excluded=summary["counts_by_action"].get("exclude", 0),
        )
        for field, value in counts.items():
            if type(release[field]) is not int or release[field] != value:
                raise ValueError(f"{field} mismatch")
        return dict(
            **summary, inventory_csv=(output / "inventory.csv").read_text(), inventory_json=header
        )


def _batches(path):
    import pyarrow.parquet as pq

    with pq.ParquetFile(path, buffer_size=65536, pre_buffer=False) as parquet:
        for group in range(parquet.num_row_groups):
            yield from parquet.iter_batches(batch_size=1, row_groups=[group], use_threads=False)


def _row_evidence(batch):
    result = {key: batch.column(key)[0].as_py() for key in ("item_id", "source", "freq")}
    result["target_sha256"] = ri.target_evidence(batch.column("target")[0])[3]
    result["covariates_sha256"] = ri.target_evidence(batch.column("covariates")[0])[3]
    return result


class _CompactFooter:
    """Skip Thrift compact values without building row-group/schema objects."""

    def __init__(self, data):
        self.data = memoryview(data)
        self.position = 0

    def take(self, size):
        start = self.position
        self.position += size
        if size < 0 or self.position > len(self.data):
            raise ValueError("truncated Parquet footer")
        return self.data[start : self.position]

    def byte(self):
        return self.take(1)[0]

    def varint(self):
        value = 0
        for shift in range(0, 70, 7):
            byte = self.byte()
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError("invalid Parquet footer varint")

    def fields(self):
        field = 0
        while header := self.byte():
            delta, kind = header >> 4, header & 15
            if delta:
                field += delta
            else:
                encoded = self.varint()
                field = (encoded >> 1) ^ -(encoded & 1)
            yield field, kind

    def collection(self):
        header = self.byte()
        size, kind = header >> 4, header & 15
        return (self.varint() if size == 15 else size), kind

    def skip(self, kind, *, collection=False):
        if kind in (1, 2):  # struct booleans live in the field header
            if collection:
                self.take(1)
        elif kind == 3:
            self.take(1)
        elif kind in (4, 5, 6):
            self.varint()
        elif kind == 7:
            self.take(8)
        elif kind == 8:
            self.take(self.varint())
        elif kind in (9, 10):
            size, element = self.collection()
            for _ in range(size):
                self.skip(element, collection=True)
        elif kind == 11:
            size = self.varint()
            if size:
                types = self.byte()
                for _ in range(size):
                    self.skip(types >> 4, collection=True)
                    self.skip(types & 15, collection=True)
        elif kind == 12:
            for _, element in self.fields():
                self.skip(element)
        else:
            raise ValueError(f"invalid Parquet footer type: {kind}")


def _footer_entries(path):
    """Read FileMetaData field 5 as ordered pairs; PyArrow's dict loses duplicates."""
    with Path(path).open("rb") as stream:
        stream.seek(-8, 2)
        trailer = stream.read(8)
        size = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != b"PAR1" or size > stream.tell() - 12:
            raise ValueError(f"invalid Parquet footer: {path}")
        stream.seek(-8 - size, 2)
        footer = _CompactFooter(stream.read(size))
    for field, kind in footer.fields():
        if field != 5:
            footer.skip(kind)
            continue
        if kind != 9:
            raise ValueError("invalid Parquet key-value metadata list")
        size, element = footer.collection()
        if element != 12:
            raise ValueError("invalid Parquet key-value metadata entry")
        entries = []
        for _ in range(size):
            key, value = None, None
            for field, kind in footer.fields():
                if field in (1, 2) and kind == 8:
                    binary = footer.take(footer.varint()).tobytes()
                    if field == 1:
                        key = binary
                    else:
                        value = binary
                else:
                    footer.skip(kind)
            if key is None:
                raise ValueError("Parquet key-value metadata entry has no key")
            entries.append((key, value))
        return entries
    return []


def filter_folder(source_folder_dir, row_manifest_csv, out_dir):
    """Read one Arrow row at a time, retaining each shard's full schema/metadata."""
    import pyarrow.parquet as pq

    source, out = Path(source_folder_dir), Path(out_dir)
    if (
        source.resolve() == out.resolve()
        or source.resolve() in out.resolve().parents
        or out.resolve() in source.resolve().parents
    ):
        raise ValueError("source and output folders overlap")
    if out.exists() and any(out.iterdir()):
        raise ValueError("output folder must be empty")
    shard_metadata = {}
    for path in sorted(source.glob("*.parquet")):
        if path.is_symlink():
            raise ValueError("symlink shard")
        entries = _footer_entries(path)
        seen = set()
        for key, _ in entries:
            if key in seen:
                raise ValueError(f"duplicate Parquet footer key {key!r}: {path.name}")
            seen.add(key)
        shard_metadata[path] = entries
    out.mkdir(parents=True, exist_ok=True)
    row_map = []
    with Path(row_manifest_csv).open(newline="") as stream:
        reader = csv.DictReader(stream)
        for path, metadata in shard_metadata.items():
            with pq.ParquetFile(path, buffer_size=65536, pre_buffer=False) as parquet:
                schema = parquet.schema_arrow
            with pq.ParquetWriter(
                out / path.name,
                schema,
                use_compliant_nested_type=False,
                store_schema=any(key == b"ARROW:schema" for key, _ in metadata),
            ) as writer:
                if metadata:
                    writer.add_key_value_metadata(dict(metadata))
                output_index = 0
                for index, batch in enumerate(_batches(path)):
                    row = next(reader, None)
                    if row is None:
                        raise ValueError("manifest row missing")
                    evidence = _row_evidence(batch)
                    fields = dict(
                        shard=path.name,
                        row_index=str(index),
                        item_id=evidence["item_id"],
                        target_sha256=evidence["target_sha256"],
                    )
                    for field, value in fields.items():
                        if row.get(field) != value:
                            raise ValueError(f"{field} mismatch: {path.name}:{index}")
                    if row["action"] not in ("keep", "exclude"):
                        raise ValueError(f"unknown action: {row['action']}")
                    if row["action"] == "keep":
                        writer.write_batch(batch)
                        row_map.append(
                            dict(
                                source_shard=path.name,
                                source_row_index=index,
                                output_row_index=output_index,
                                **evidence,
                            )
                        )
                        output_index += 1
                    del batch
            with pq.ParquetFile(out / path.name, buffer_size=65536, pre_buffer=False) as filtered:
                if not schema.equals(filtered.schema_arrow, check_metadata=True):
                    raise ValueError(f"read-back Arrow schema mismatch: {path.name}")
                if metadata != _footer_entries(out / path.name):
                    raise ValueError(f"read-back Parquet key-value metadata mismatch: {path.name}")
        if next(reader, None) is not None:
            raise ValueError("shard row missing for manifest row")
    if not row_map:
        raise ValueError("zero kept rows")
    verify_filtered(out, row_map)
    return row_map


def verify_filtered(folder, row_map):
    """Independent read-back using canonical hashes, including NaNs and nulls."""
    expected = iter(row_map)
    for path in sorted(Path(folder).glob("*.parquet")):
        for index, batch in enumerate(_batches(path)):
            record = next(expected, None)
            if record is None:
                raise ValueError("read-back kept count mismatch")
            actual = dict(source_shard=path.name, output_row_index=index, **_row_evidence(batch))
            for field, value in actual.items():
                if record.get(field) != value:
                    raise ValueError(f"read-back {field} mismatch")
            del batch
    if next(expected, None) is not None:
        raise ValueError("read-back kept count mismatch")


def evidence_names(release):
    name = release["folder"].split("/")[-1]
    return {
        "rules": f"rules-{name}.yaml",
        "row_manifest": f"row-manifest-{name}.csv",
        "inventory_csv": f"inventory-{name}.csv",
        "inventory_json": f"inventory-{name}.json",
    }


def _stable_stats(stats):
    value = copy.deepcopy(stats)
    for key in ("wall_seconds", "peak_rss_bytes"):
        value.get("measurement", {}).pop(key, None)
    return value


def prepare_releases(parent_root, output_root, repo_root, *, prior_stats=None, measurements=None):
    """Stage only releases and evidence; unchanged parent folders stay server-side."""
    parent, root, repo = Path(parent_root), Path(output_root), Path(repo_root)
    contract, identity = read_contract(repo), derived_identity(repo)
    root.mkdir(parents=True, exist_ok=True)
    evidence_dir = root / "derivation"
    evidence_dir.mkdir(exist_ok=True)
    atomic_json(root / "corpus-identity.json", identity)
    records = []
    for release in contract["releases"]:
        evidence = check_release(parent / release["source_folder"], release, repo)
        folder = root / release["folder"]
        row_map = filter_folder(
            parent / release["source_folder"], repo / release["row_manifest"], folder
        )
        names = evidence_names(release)
        for field in ("rules", "row_manifest"):
            shutil.copyfile(repo / release[field], evidence_dir / names[field])
        (evidence_dir / names["inventory_csv"]).write_text(evidence["inventory_csv"])
        atomic_json(evidence_dir / names["inventory_json"], evidence["inventory_json"])
        measured = measure_folder(folder, "covariates")
        if measurements is not None:
            measurements[release["folder"]] = {
                key: measured["measurement"][key] for key in ("wall_seconds", "peak_rss_bytes")
            }
        stats = _stable_stats(measured)
        if prior_stats and release["folder"] in prior_stats:
            previous = prior_stats[release["folder"]]
            if _stable_stats(previous) != stats:
                raise ValueError("resumed stats differ from measured release")
            stats = previous
        atomic_json(folder / "stats.json", stats)
        shards = [
            dict(path=p.name, size=p.stat().st_size, sha256=digest(p))
            for p in sorted(folder.glob("*.parquet"))
        ]
        atomic_json(
            folder / "receipt.json",
            dict(
                repo_id=identity["repo_id"],
                revision=identity["revision"],
                folder=release["folder"],
                status="allowed",
                files=[
                    dict(
                        path=release["folder"] + "/" + s["path"],
                        size=s["size"],
                        hash=s["sha256"],
                        algorithm="sha256",
                    )
                    for s in shards
                ],
                stats_sha256=digest(folder / "stats.json"),
                derived=_block(release),
            ),
        )
        records.append(
            dict(
                folder=release["folder"],
                **_block(release),
                counts_by_rule=evidence["counts_by_rule"],
                shards=shards,
                row_map=row_map,
            )
        )
    return records


def payload_map(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink object: {path}")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name not in EXCLUDED_PAYLOAD:
                result[name] = dict(size=path.stat().st_size, sha256=digest(path))
    return result


def attempt_id(inputs):
    identity = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return "derived-" + inputs["name"] + "-" + identity[:12]


def write_derivation(root, repo_root, *, releases, publishing, parent_identity, parent_status):
    read_contract(repo_root)
    inputs = publishing["inputs"]
    value = dict(
        version="4.1",
        repo_sha=inputs["repo_sha"],
        guest_sha256=inputs["guest_sha256"],
        environment=inputs["environment"],
        parent_uri=inputs["parent_uri"],
        parent_identity_sha256=digest(parent_identity),
        parent_attempt_status=parent_status,
        releases=releases,
        payload=payload_map(root),
    )
    atomic_json(Path(root) / "derivation/derivation.json", value)
    return value


def status_for(root):
    root = Path(root)
    publishing = json.loads((root / "derivation/publishing.json").read_text())
    ledger = json.loads((root / "derivation/derivation.json").read_text())
    return dict(
        attempt_id=publishing["attempt_id"],
        repo_sha=ledger["repo_sha"],
        exit_code=-1,
        complete=False,
        mode="derived",
        parent_attempt_id=ledger["parent_attempt_status"]["attempt_id"],
        derivation_sha256=digest(root / "derivation/derivation.json"),
    )


def _check_evidence(root, repo, release, record):
    """Replay source decisions; compare inventories through the source/output map."""
    evidence_dir = root / "derivation"
    names = evidence_names(release)
    for field in ("rules", "row_manifest"):
        if (evidence_dir / names[field]).read_bytes() != (repo / release[field]).read_bytes():
            raise ValueError(f"evidence {field} mismatch")
    source_inventory = evidence_dir / names["inventory_csv"]
    if digest(source_inventory) != release["inventory_sha256"]:
        raise ValueError("evidence inventory_sha256 mismatch")
    source_header = json.loads((evidence_dir / names["inventory_json"]).read_text())
    if source_header.get("folder") != release["source_folder"]:
        raise ValueError("source inventory folder mismatch")
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        shutil.copyfile(source_inventory, work / "inventory.csv")
        shutil.copyfile(evidence_dir / names["inventory_json"], work / "inventory.json")
        summary = ri.write_manifest(work / "inventory.csv", evidence_dir / names["rules"], work)
        if (work / "row-manifest.csv").read_bytes() != (
            evidence_dir / names["row_manifest"]
        ).read_bytes() or summary["counts_by_rule"] != record["counts_by_rule"]:
            raise ValueError("regenerated row_manifest/counts_by_rule mismatch")
        ri.build_inventory(root / release["folder"], work, folder_name=release["folder"])
        with (work / "inventory.csv").open(newline="") as stream:
            outputs = list(csv.DictReader(stream))
    with (evidence_dir / names["row_manifest"]).open(newline="") as stream:
        kept = [row for row in csv.DictReader(stream) if row["action"] == "keep"]
    if (
        len(kept) != release["kept"]
        or len(outputs) != len(kept)
        or len(record["row_map"]) != len(kept)
    ):
        raise ValueError("kept count mismatch")
    indexes = Counter()
    for source, actual, mapping in zip(kept, outputs, record["row_map"]):
        expected = {k: v for k, v in source.items() if k not in ri.DECISION_FIELDS}
        shard = source["shard"]
        expected.update(folder=release["folder"], row_index=str(indexes[shard]))
        if actual != expected:
            raise ValueError("retained inventory row mismatch")
        for field, value in dict(
            source_shard=shard,
            source_row_index=int(source["row_index"]),
            output_row_index=indexes[shard],
            item_id=source["item_id"],
            source=source["source"],
            freq=source["freq"],
            target_sha256=source["target_sha256"],
        ).items():
            if type(mapping.get(field)) is not type(value) or mapping[field] != value:
                raise ValueError(f"row map {field} mismatch")
        _sha(mapping.get("covariates_sha256"), "covariates_sha256")
        indexes[shard] += 1
    verify_filtered(root / release["folder"], record["row_map"])


def verify_derived(root, repo_root, pinned_files=None):
    """Verify every published byte and each retained row without any cloud access."""
    root, repo = Path(root), Path(repo_root)
    identity = derived_identity(repo)
    validate_derived_manifest(json.loads((root / "corpus-identity.json").read_text()), repo)
    if (root / "quarantine").exists():
        raise ValueError("quarantine directory is forbidden")
    contract = read_contract(repo)
    ledger_path = root / "derivation/derivation.json"
    ledger = json.loads(ledger_path.read_text())
    publishing = json.loads((root / "derivation/publishing.json").read_text())
    inputs = publishing["inputs"]
    if (
        publishing.get("attempt_id") != attempt_id(inputs)
        or ledger.get("version") != "4.1"
        or any(
            ledger.get(k) != inputs[k]
            for k in ("repo_sha", "guest_sha256", "environment", "parent_uri")
        )
        or ledger["environment"] != ENVIRONMENT
    ):
        raise ValueError("derivation/publishing identity mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", str(ledger["repo_sha"])):
        raise ValueError("invalid repo SHA")
    _sha(ledger["guest_sha256"], "guest_sha256")
    _sha(ledger.get("parent_identity_sha256"), "parent_identity_sha256")
    parent = read_manifest(repo / contract["parent"]["manifest"])
    # fetch_native writes exactly these deterministic identity bytes.
    parent_bytes = (json.dumps(_identity(parent), sort_keys=True, indent=2) + "\n").encode()
    if hashlib.sha256(parent_bytes).hexdigest() != ledger["parent_identity_sha256"]:
        raise ValueError("parent identity sha256 mismatch")
    if (
        not isinstance(ledger.get("parent_attempt_status"), dict)
        or not isinstance(ledger["parent_attempt_status"].get("attempt_id"), str)
        or not ledger["parent_attempt_status"]["attempt_id"]
    ):
        raise ValueError("invalid parent attempt status")
    records = ledger["releases"]
    if not isinstance(records, list) or [r.get("folder") for r in records] != [
        r["folder"] for r in contract["releases"]
    ]:
        raise ValueError("derivation release set mismatch")
    files = _pinned(repo)["files"] if pinned_files is None else pinned_files
    allowed = {e["folder"] for e in parent["folders"] if e["status"] == "allowed"}
    parent_view = {**parent, "folders": [e for e in parent["folders"] if e["folder"] in allowed]}
    expected = inventory(
        parent_view,
        PinnedInventory(
            parent["repo_id"],
            parent["revision"],
            [f for f in files if str(PurePosixPath(f["path"]).parent) in allowed],
        ),
    )
    permitted = {"corpus-identity.json", *EXCLUDED_PAYLOAD}
    for release, record in zip(contract["releases"], records):
        recorded_block = {k: record.get(k) for k in ("standard", *DERIVED_FIELDS)}
        _validate_block(recorded_block, release["folder"])
        if recorded_block != _block(release):
            raise ValueError("derivation release block mismatch")
        expected[release["folder"]] = []
        for shard in record["shards"]:
            name = _relative(shard["path"])
            if "/" in name or name not in {s["path"] for s in release["parent_shards"]}:
                raise ValueError("derived shard name mismatch")
            _sha(shard["sha256"], "derived shard sha256")
            if type(shard["size"]) is not int or shard["size"] <= 0:
                raise ValueError("invalid derived shard size")
            expected[release["folder"]].append(
                HubFile(release["folder"] + "/" + name, shard["size"], shard["sha256"], "sha256")
            )
        if len(expected[release["folder"]]) != len(release["parent_shards"]):
            raise ValueError("derived shard set mismatch")
        permitted.update("derivation/" + name for name in evidence_names(release).values())
    for entry in identity["folders"]:
        permitted.update(entry["folder"] + "/" + name for name in ("receipt.json", "stats.json"))
        permitted.update(file.path for file in expected[entry["folder"]])
    actual = payload_map(root)
    if actual != ledger["payload"]:
        raise ValueError("payload object set/size/sha256 mismatch")
    result = verify_integrity(root, identity, expected, derived_receipts=True)
    if set(actual) != permitted - EXCLUDED_PAYLOAD:
        raise ValueError("payload object set/size/sha256 mismatch")
    directories = {
        p.as_posix() for name in permitted for p in PurePosixPath(name).parents if str(p) != "."
    }
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if path.is_dir() and name not in directories:
            raise ValueError(f"unlisted directory: {name}")
    for release, record in zip(contract["releases"], records):
        _check_evidence(root, repo, release, record)
    status = root / "attempt-status.json"
    if status.exists():
        expected_status = json.dumps(status_for(root), sort_keys=True, indent=2) + "\n"
        if status.read_text() != expected_status:
            raise ValueError("attempt-status differs from derived publication")
    return dict(**result, derivation_sha256=digest(ledger_path))
