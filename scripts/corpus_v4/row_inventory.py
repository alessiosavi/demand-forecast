"""Stream released Parquet rows into deterministic inventory and exclusion evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE_FIELDS = [
    "folder",
    "shard",
    "row_index",
    "source",
    "item_id",
    "freq",
    "target_shape",
    "observed_target_values",
    "covariate_channels",
    "target_dtype",
    "target_sha256",
]
DECISION_FIELDS = ["action", "rule", "reason"]
TARGET_ENCODING_VERSION = "arrow-target-v1"
TARGET_ENCODING = (
    "Sorted compact JSON {encoding,dtype,shape} plus newline; one validity byte (0/1) "
    "per C-order value, then little-endian stored numeric bytes with null slots zeroed; "
    "null-typed leaves have no numeric bytes"
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def scalar_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def target_evidence(scalar):
    import pyarrow as pa
    import pyarrow.compute as pc

    if not scalar.is_valid:
        raise ValueError("target must have nonempty shape")
    values = scalar.values
    if (
        pa.types.is_list(values.type)
        or pa.types.is_large_list(values.type)
        or pa.types.is_fixed_size_list(values.type)
    ):
        lengths = [len(channel.values) for channel in values if channel.is_valid]
        if values.null_count or not lengths or len(set(lengths)) != 1:
            raise ValueError("target must have aligned non-null channels")
        shape = [len(values), lengths[0]]
        values = values.flatten()
    else:
        shape = [len(values)]
    if not all(shape) or not (
        pa.types.is_integer(values.type)
        or pa.types.is_floating(values.type)
        or pa.types.is_null(values.type)
    ):
        raise ValueError("target must have nonempty numeric shape [T] or [V,T]")
    if sys.byteorder != "little":
        raise ValueError("target encoding requires a little-endian host")
    dtype = str(values.type)
    header = (
        json.dumps(
            dict(encoding=TARGET_ENCODING_VERSION, dtype=dtype, shape=shape),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    fingerprint = hashlib.sha256(header)
    validity = pc.cast(pc.is_valid(values), pa.uint8())
    fingerprint.update(
        memoryview(validity.buffers()[1])[validity.offset : validity.offset + len(validity)]
    )
    if pa.types.is_null(values.type):
        observed = 0
    else:
        if pa.types.is_integer(values.type):
            # Every stored integer is finite; do not depend on an is_finite kernel for integers.
            observed = len(values) - values.null_count
        else:
            observed = int(pc.sum(pc.is_finite(values)).as_py() or 0)
        if values.null_count:
            # Canonicalize unspecified null-slot bytes without converting stored values.
            values = pc.fill_null(values, pa.scalar(0, type=values.type))
        width = values.type.bit_width // 8
        fingerprint.update(
            memoryview(values.buffers()[1])[
                values.offset * width : (values.offset + len(values)) * width
            ]
        )
    return shape, observed, dtype, fingerprint.hexdigest()


def _folder_identity(name):
    if not isinstance(name, str) or not re.fullmatch(
        r"(?:quarantine/)?(?:univariate|covariates)/[A-Za-z0-9][A-Za-z0-9._-]*", name
    ):
        raise ValueError("folder must be a canonical corpus-relative native folder path")
    return name


def build_inventory(folder, output, *, folder_name):
    """Write inventory.csv/json; metadata discovery never decodes target payloads."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder_name = _folder_identity(folder_name)
    folder, output = Path(folder), Path(output)
    paths = sorted(folder.glob("*.parquet"))
    if not paths or not any(p.name.startswith("train-") for p in paths):
        raise ValueError("folder has no train shards")
    columns, scalar_columns = set(), set()
    for path in paths:
        if path.is_symlink() or not path.name.startswith(("train-", "validation-")):
            raise ValueError(f"unsupported shard: {path}")
        with pq.ParquetFile(path, buffer_size=65536, pre_buffer=False) as parquet:
            schema = parquet.schema_arrow
            if not {"item_id", "source", "freq", "target"} <= set(schema.names):
                raise ValueError(f"missing native columns: {path}")
            if len(set(schema.names)) != len(schema.names):
                raise ValueError(f"duplicate schema columns: {path}")
            columns.update(schema.names)
            scalar_columns.update(
                field.name
                for field in schema
                if not pa.types.is_nested(field.type)
                and field.name
                not in {"source", "item_id", "freq", "target", "covariates", "covariate_names"}
            )
    extras = {}
    used = set(BASE_FIELDS + DECISION_FIELDS + ["covariate_names"])
    for name in sorted(scalar_columns):
        field = name
        while field in used:
            field = "scalar." + field
        extras[name] = field
        used.add(field)
    fields = (
        BASE_FIELDS
        + (["covariate_names"] if "covariate_names" in columns else [])
        + list(extras.values())
    )
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "inventory.csv.tmp"
    counts = Counter()
    total = observed_total = target_values = 0
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for path in paths:
            with pq.ParquetFile(path, buffer_size=65536, pre_buffer=False) as parquet:
                number = 0
                for group in range(parquet.num_row_groups):
                    for batch in parquet.iter_batches(
                        batch_size=1, row_groups=[group], use_threads=False
                    ):
                        identity = {
                            n: batch.column(n)[0].as_py() for n in ("source", "item_id", "freq")
                        }
                        if any(not isinstance(v, str) or not v for v in identity.values()):
                            raise ValueError(f"invalid native identity: {path}:{number}")
                        shape, observed, dtype, sha = target_evidence(batch.column("target")[0])
                        cov = (
                            batch.column("covariates")[0]
                            if "covariates" in batch.schema.names
                            else None
                        )
                        if cov is not None and not cov.is_valid:
                            raise ValueError(f"null covariates: {path}:{number}")
                        row = dict(
                            folder=folder_name,
                            shard=path.name,
                            row_index=number,
                            **identity,
                            target_shape=json.dumps(shape, separators=(",", ":")),
                            observed_target_values=observed,
                            covariate_channels=len(cov.values) if cov is not None else 0,
                            target_dtype=dtype,
                            target_sha256=sha,
                        )
                        if "covariate_names" in batch.schema.names:
                            row["covariate_names"] = json.dumps(
                                batch.column("covariate_names")[0].as_py(),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        for name, field in extras.items():
                            row[field] = (
                                scalar_text(batch.column(name)[0].as_py())
                                if name in batch.schema.names
                                else ""
                            )
                        writer.writerow(row)
                        counts[identity["source"]] += 1
                        observed_total += observed
                        target_values += shape[0] * shape[1] if len(shape) == 2 else shape[0]
                        total += 1
                        number += 1
                        del batch, cov, row
    temporary.replace(output / "inventory.csv")
    summary = dict(
        schema=1,
        folder=folder_name,
        rows=total,
        shards=[p.name for p in paths],
        schema_columns=sorted(columns),
        scalar_columns=sorted(scalar_columns),
        scalar_fields=extras,
        csv_fields=fields,
        per_source_rows=dict(sorted(counts.items())),
        target_values=target_values,
        observed_target_values=observed_total,
        target_encoding=TARGET_ENCODING,
        target_encoding_version=TARGET_ENCODING_VERSION,
        csv_sha256=digest(output / "inventory.csv"),
    )
    atomic_json(output / "inventory.json", summary)
    return summary


def _inventory_fields(fields):
    if (
        not isinstance(fields, (list, tuple))
        or any(not isinstance(field, str) or not field for field in fields)
        or len(set(fields)) != len(fields)
        or not set(BASE_FIELDS) <= set(fields)
        or set(fields) & set(DECISION_FIELDS)
    ):
        raise ValueError("invalid inventory header")
    return list(fields)


def _compile_rules(document, fields, folder_name):
    if not isinstance(document, dict) or set(document) - {"folder", "default", "rules", "missing"}:
        raise ValueError("rules must be a document with folder, default and ordered rules")
    if _folder_identity(document.get("folder")) != folder_name:
        raise ValueError(
            f"rules folder {document['folder']!r} differs from inventory folder {folder_name!r}"
        )
    missing = document.get("missing", "refuse")
    if missing not in ("refuse", "exclude"):
        raise ValueError("missing policy must be refuse or exclude")
    default = document.get("default")
    if isinstance(default, str):
        default = dict(action=default, reason="default action")
    if (
        not isinstance(default, dict)
        or set(default) != {"action", "reason"}
        or default["action"] not in ("keep", "exclude")
        or not isinstance(default["reason"], str)
        or not default["reason"].strip()
    ):
        raise ValueError("rules require a default action (keep or exclude)")
    rules = document.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("rules must be an ordered list")
    compiled = []
    for rule in rules:
        if (
            not isinstance(rule, dict)
            or set(rule) != {"match", "action", "reason"}
            or rule["action"] not in ("keep", "exclude")
            or not isinstance(rule["reason"], str)
            or not rule["reason"].strip()
            or not isinstance(rule["match"], dict)
            or not rule["match"]
        ):
            raise ValueError("invalid rule entry")
        match = []
        for field, pattern in rule["match"].items():
            if field not in fields:
                raise ValueError(f"unknown inventory field: {field}")
            if not isinstance(pattern, str):
                raise ValueError("rule regex must be text")
            try:
                match.append((field, re.compile(pattern)))
            except re.error as exc:
                raise ValueError(f"invalid regex for {field}: {exc}") from exc
        compiled.append((match, rule["action"], rule["reason"]))
    return default, compiled, missing


def apply_rules(inventory_rows, rules, *, fields, folder_name):
    """Validate against the header before first-match regex search; missing never keeps."""
    fields = _inventory_fields(fields)
    folder_name = _folder_identity(folder_name)
    default, compiled, missing_policy = _compile_rules(rules, fields, folder_name)
    referenced = {field for match, _, _ in compiled for field, _ in match}
    required = referenced | {"folder", "shard", "row_index", "source", "item_id", "freq"}
    for row in inventory_rows:
        identity = ", ".join(
            f"{key}={row.get(key, '<missing>')!r}" for key in ("folder", "shard", "row_index")
        )
        unexpected = set(row) - set(fields)
        if unexpected:
            raise ValueError(
                f"inventory row ({identity}): unexpected columns {sorted(unexpected, key=str)!r}"
            )
        missing = [
            field
            for field in fields
            if field not in row
            or row[field] is None
            or (field in required and not scalar_text(row[field]).strip())
        ]
        if "folder" not in missing and row["folder"] != folder_name:
            raise ValueError(
                f"inventory row ({identity}): folder differs from inventory folder {folder_name!r}"
            )
        if missing:
            reason = "missing inventory values: " + ", ".join(missing)
            if missing_policy == "refuse":
                raise ValueError(f"inventory row ({identity}): {reason}")
            yield {**row, "action": "exclude", "rule": "missing", "reason": reason}
            continue
        action, reason, decided = default["action"], default["reason"], "default"
        for index, (match, candidate, explanation) in enumerate(compiled):
            if all(pattern.search(scalar_text(row[field])) for field, pattern in match):
                action, reason, decided = candidate, explanation, str(index)
                break
        yield {**row, "action": action, "rule": decided, "reason": reason}


def write_manifest(inventory_csv, rules_path, output):
    """Verify inventory provenance before decisions; preserve row order and columns."""
    inventory_csv, rules_path, output = Path(inventory_csv), Path(rules_path), Path(output)
    header = json.loads(inventory_csv.with_suffix(".json").read_text())
    inventory_sha256 = digest(inventory_csv)
    if inventory_sha256 != header.get("csv_sha256"):
        raise ValueError("inventory CSV sha256 differs from inventory JSON sha256")
    folder_name = _folder_identity(header["folder"])
    expected_fields = _inventory_fields(header["csv_fields"])
    raw = rules_path.read_bytes()
    if rules_path.suffix.lower() == ".json":
        rules = json.loads(raw)
    else:
        import yaml

        rules = yaml.safe_load(raw)
    actions, decided = Counter(), Counter()
    temporary = output / "row-manifest.csv.tmp"
    with inventory_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = _inventory_fields(reader.fieldnames)
        if fields != expected_fields:
            raise ValueError("inventory CSV header differs from inventory JSON header")
        if sum(1 for _ in reader) != header.get("rows"):
            raise ValueError("inventory CSV row count differs from inventory JSON row count")
        source.seek(0)
        reader = csv.DictReader(source)
        _compile_rules(rules, fields, folder_name)
        output.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fields + DECISION_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for row in apply_rules(reader, rules, fields=fields, folder_name=folder_name):
                writer.writerow(row)
                actions[row["action"]] += 1
                decided[row["rule"]] += 1
    temporary.replace(output / "row-manifest.csv")
    summary = dict(
        schema=1,
        folder=folder_name,
        rows=sum(actions.values()),
        counts_by_action=dict(actions),
        counts_by_rule=dict(decided),
        inventory_sha256=inventory_sha256,
        rules_sha256=hashlib.sha256(raw).hexdigest(),
        csv_sha256=digest(output / "row-manifest.csv"),
    )
    atomic_json(output / "row-manifest.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument(
        "--folder-name", required=True, help="canonical folder path relative to the corpus prefix"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rules", type=Path, help="optional YAML/JSON row rules")
    args = parser.parse_args(argv)
    build_inventory(args.folder, args.output, folder_name=args.folder_name)
    if args.rules:
        write_manifest(args.output / "inventory.csv", args.rules, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
