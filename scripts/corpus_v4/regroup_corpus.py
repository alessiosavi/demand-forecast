"""Rewrite native corpus shards in bounded Parquet row groups and verify the result."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

THRESHOLD_BYTES = 96 << 20
GROUP_BYTES = 32 << 20
GROUP_ROWS = 4096
ADMISSION_BYTES = 256 << 20

_NUMERIC_COLUMNS = ("target", "covariates")
_KINDS = ("univariate", "covariates")
_ATTRIBUTION = ("LICENSE.md", "articles.csv")
_ARROW_SCHEMA_KEY = b"ARROW:schema"
_LEDGER_NAME = "regroup.json"


def _estimate(entries, rows):
    return 24 * entries + 64 * rows


def group_estimate(metadata, group: int) -> int:
    """Match the v5 reader's admission estimate for both native corpus schemas.

    For any schema the reader accepts, this estimate is at least as large as
    the reader's, so checking a bound here is at least as strict.
    """
    row_group = metadata.row_group(group)
    entries = 0
    for i in range(row_group.num_columns):
        column = row_group.column(i)
        if column.path_in_schema.split(".", 1)[0] in _NUMERIC_COLUMNS:
            entries += column.num_values
    return _estimate(entries, row_group.num_rows)


def _is_list(dtype):
    return (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_fixed_size_list(dtype)
    )


def _entries(cell):
    # A null or empty list contributes a Parquet definition level even without a value.
    if not _is_list(cell.type) or not cell.is_valid:
        return 1
    values = cell.values
    if _is_list(values.type):
        return sum(_entries(value) for value in values) or 1
    return len(values) or 1


def _framed(digest, data):
    digest.update(len(data).to_bytes(8, "little"))
    digest.update(data)


def _hash_values(digest, array):
    dtype = array.type
    if len(array) == array.null_count:
        digest.update(b"\x00" * len(array))
        return
    valid = array.is_valid().to_numpy(zero_copy_only=False) if array.null_count else None
    digest.update(valid.tobytes() if valid is not None else b"\x01" * len(array))
    if _is_list(dtype):
        for cell in array:
            if cell.is_valid:
                values = cell.values
                digest.update(len(values).to_bytes(8, "little"))
                _hash_values(digest, values)
    elif pa.types.is_dictionary(dtype):
        # Hash the logical values even when the file uses a different dictionary per group.
        _hash_values(digest, array.dictionary_decode())
    elif isinstance(dtype, pa.BaseExtensionType):
        _hash_values(digest, array.storage)
    elif pa.types.is_boolean(dtype):
        digest.update(bytes(cell.as_py() for cell in array if cell.is_valid))
    elif (
        pa.types.is_primitive(dtype)
        or pa.types.is_decimal(dtype)
        or pa.types.is_fixed_size_binary(dtype)
    ):
        width = dtype.byte_width
        data = memoryview(array.buffers()[1])[
            array.offset * width : (array.offset + len(array)) * width
        ]
        if valid is not None:
            # Copy valid element bytes without converting or normalising any numeric value.
            data = np.frombuffer(data, dtype=f"V{width}")[valid].tobytes()
        digest.update(data)
    elif (
        pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_string_view(dtype)
        or pa.types.is_binary(dtype)
        or pa.types.is_large_binary(dtype)
        or pa.types.is_binary_view(dtype)
    ):
        for cell in array:
            if cell.is_valid:
                _framed(digest, cell.as_buffer())
    else:
        raise ValueError(f"unsupported Arrow cell type: {dtype}")


def row_digest(batch: pa.RecordBatch) -> bytes:
    """Hash one logical Arrow row, including exact bits for valid numeric values."""
    if batch.num_rows != 1:
        raise ValueError("row_digest requires exactly one row")
    digest = hashlib.sha256()
    digest.update(batch.num_columns.to_bytes(8, "little"))
    for field, column in zip(batch.schema, batch.columns, strict=True):
        _framed(digest, field.name.encode("utf-8"))
        _framed(digest, str(field.type).encode("utf-8"))
        _hash_values(digest, column)
    return digest.digest()


def _contains_fixed_size_list(dtype):
    if pa.types.is_fixed_size_list(dtype):
        return True
    if pa.types.is_dictionary(dtype):
        return _contains_fixed_size_list(dtype.value_type)
    if isinstance(dtype, pa.BaseExtensionType):
        return _contains_fixed_size_list(dtype.storage_type)
    return any(_contains_fixed_size_list(dtype.field(i).type) for i in range(dtype.num_fields))


@contextmanager
def _open_parquet(path):
    with pq.ParquetFile(path, pre_buffer=False, buffer_size=64 << 10) as parquet:
        for field in parquet.schema_arrow:
            if field.name in _NUMERIC_COLUMNS and _contains_fixed_size_list(field.type):
                raise ValueError(f"{path}: {field.name} contains an unsupported fixed_size_list")
        yield parquet


def _batches(parquet):
    for group in range(parquet.num_row_groups):
        yield from parquet.iter_batches(batch_size=1, row_groups=[group], use_threads=False)


def _rows_evidence(parquet):
    digest = hashlib.sha256()
    rows = 0
    for batch in _batches(parquet):
        digest.update(row_digest(batch))
        rows += batch.num_rows
    return rows, digest.hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_schema(source, target, before, after):
    if not before.schema.equals(after.schema) or not before.schema_arrow.equals(
        after.schema_arrow, check_metadata=True
    ):
        raise ValueError(f"schema mismatch: {source} -> {target}")
    before_footer = before.metadata.metadata or {}
    after_footer = after.metadata.metadata or {}
    stored_schema_matches = (_ARROW_SCHEMA_KEY in before_footer) == (
        _ARROW_SCHEMA_KEY in after_footer
    )
    # Equal Arrow schemas can have different serializations.
    before_footer.pop(_ARROW_SCHEMA_KEY, None)
    after_footer.pop(_ARROW_SCHEMA_KEY, None)
    if not stored_schema_matches or before_footer != after_footer:
        raise ValueError(f"footer metadata mismatch: {source} -> {target}")


def _check_groups(metadata, label, *, group_bytes=None, group_rows=None):
    largest = oversized = 0
    for i in range(metadata.num_row_groups):
        estimate = group_estimate(metadata, i)
        rows = metadata.row_group(i).num_rows
        if estimate > ADMISSION_BYTES:
            raise ValueError(f"{label}: group {i} exceeds the admission limit ({estimate} bytes)")
        if group_rows is not None and rows > group_rows:
            raise ValueError(f"{label}: group {i} exceeds the row bound ({rows} rows)")
        if group_bytes is not None and estimate > group_bytes:
            if rows != 1:
                raise ValueError(f"{label}: group {i} exceeds the byte bound ({estimate} bytes)")
            oversized += 1
        largest = max(largest, estimate)
    return largest, oversized


def _check_parameters(
    *, threshold_bytes=THRESHOLD_BYTES, group_bytes=GROUP_BYTES, group_rows=GROUP_ROWS
):
    if threshold_bytes < 0 or group_bytes <= 0 or group_rows <= 0:
        raise ValueError(
            "threshold_bytes must be nonnegative and group_bytes and group_rows positive"
        )
    for name, value in (("group_bytes", group_bytes), ("threshold_bytes", threshold_bytes)):
        if value > ADMISSION_BYTES:
            raise ValueError(f"{name} exceeds the admission limit ({ADMISSION_BYTES} bytes)")


def needs_regroup(shard, *, threshold_bytes=THRESHOLD_BYTES) -> bool:
    _check_parameters(threshold_bytes=threshold_bytes)
    with _open_parquet(shard) as parquet:
        return any(
            group_estimate(parquet.metadata, i) > threshold_bytes
            for i in range(parquet.num_row_groups)
        )


def _write_groups(parquet, writer, *, group_bytes, group_rows):
    batches = []
    group_estimate_bytes = rows = 0
    digest = hashlib.sha256()
    schema = parquet.schema_arrow
    numeric = [i for i, name in enumerate(schema.names) if name in _NUMERIC_COLUMNS]
    for batch in _batches(parquet):
        estimate = _estimate(sum(_entries(batch.column(i)[0]) for i in numeric), 1)
        if batches and (
            group_estimate_bytes + estimate > group_bytes or len(batches) >= group_rows
        ):
            writer.write_table(
                pa.Table.from_batches(batches, schema=schema), row_group_size=len(batches)
            )
            batches.clear()
            group_estimate_bytes = 0
        batches.append(batch)
        group_estimate_bytes += estimate
        digest.update(row_digest(batch))
        rows += 1
    if batches:
        writer.write_table(
            pa.Table.from_batches(batches, schema=schema), row_group_size=len(batches)
        )
    return rows, digest.hexdigest()


def regroup_shard(source, target, *, group_bytes=GROUP_BYTES, group_rows=GROUP_ROWS) -> dict:
    """Retain all rows and rewrite only their Parquet grouping."""
    _check_parameters(group_bytes=group_bytes, group_rows=group_rows)
    source, target = _safe_path(source), _safe_path(target)
    if source == target or (target.exists() and target.samefile(source)):
        raise ValueError(f"source and target are the same file: {source}")
    with _open_parquet(source) as before:
        metadata = before.metadata.metadata or {}
        # The Arrow schema exposes the Parquet child names, including "element"
        # for compliant lists. Use the names as read, even in a mixed schema.
        with pq.ParquetWriter(
            target,
            before.schema_arrow,
            store_schema=_ARROW_SCHEMA_KEY in metadata,
            use_compliant_nested_type=False,
        ) as writer:
            if metadata:
                writer.add_key_value_metadata(metadata)
            rows, rows_sha256 = _write_groups(
                before, writer, group_bytes=group_bytes, group_rows=group_rows
            )

        with _open_parquet(target) as after:
            largest, oversized = _check_groups(
                after.metadata, target, group_bytes=group_bytes, group_rows=group_rows
            )
            _check_schema(source, target, before, after)
            if rows != before.metadata.num_rows or rows != after.metadata.num_rows:
                raise ValueError(f"row count mismatch: {source} -> {target}")
            return dict(
                rows=rows,
                source_groups=before.num_row_groups,
                output_groups=after.num_row_groups,
                largest_output_estimate=largest,
                oversized_rows=oversized,
                rows_sha256=rows_sha256,
                source_sha256=_sha256(source),
                source_bytes=source.stat().st_size,
                output_sha256=_sha256(target),
                output_bytes=target.stat().st_size,
            )


def _safe_path(path):
    path = Path(path).absolute()
    for part in (*reversed(path.parents), path):
        if part.is_symlink():
            raise ValueError(f"symlink path: {part}")
    resolved = path.resolve()
    if "quarantine" in str(path).lower() or "quarantine" in str(resolved).lower():
        raise ValueError(f"quarantine path: {path}")
    return resolved


def _roots(parent, out):
    parent, out = _safe_path(parent), _safe_path(out)
    if parent == out or parent in out.parents or out in parent.parents:
        raise ValueError(f"parent and output roots overlap: {parent}, {out}")
    return parent, out


def _inventory(root):
    if not root.is_dir():
        raise ValueError(f"corpus root is not a directory: {root}")
    files = {}
    directories = set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"symlink path: {path}")
        if "quarantine" in relative.lower():
            raise ValueError(f"quarantine path: {path}")
        if path.is_file():
            files[relative] = path
        elif path.is_dir():
            directories.add(relative)
    return files, directories


def _parent_inventory(parent):
    files, _ = _inventory(parent)
    shards = {
        name: path
        for name, path in files.items()
        if name.split("/", 1)[0] in _KINDS and path.suffix == ".parquet"
    }
    other = {name: path for name, path in files.items() if path.name in _ATTRIBUTION}
    return shards, other


def _copy_shard(source, target):
    with _open_parquet(source) as parquet:
        largest, _ = _check_groups(parquet.metadata, source)
        rows, rows_sha256 = _rows_evidence(parquet)
        if rows != parquet.metadata.num_rows:
            raise ValueError(f"row count mismatch: {source}")
        record = dict(
            copied=True,
            sha256=_sha256(source),
            bytes=source.stat().st_size,
            groups=parquet.num_row_groups,
            largest_estimate=largest,
            rows_sha256=rows_sha256,
        )
    shutil.copyfile(source, target)
    return record


def regroup_tree(parent, out, *, threshold_bytes, group_bytes, group_rows) -> dict:
    """Create a fresh tree with the same shards and required attribution."""
    parameters = dict(
        threshold_bytes=threshold_bytes, group_bytes=group_bytes, group_rows=group_rows
    )
    _check_parameters(**parameters)
    parent, out = _roots(parent, out)
    if out.exists():
        raise ValueError(f"output already exists: {out}")
    shards, other = _parent_inventory(parent)
    # Inspect every shard's schema and footer before creating any output paths.
    to_regroup = {
        name
        for name, source in shards.items()
        if needs_regroup(source, threshold_bytes=threshold_bytes)
    }
    out.mkdir(parents=True)
    shard_records, other_records = {}, {}
    for name, source in shards.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in to_regroup:
            shard_records[name] = regroup_shard(
                source, target, group_bytes=group_bytes, group_rows=group_rows
            )
        else:
            shard_records[name] = _copy_shard(source, target)
    for name, source in other.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        other_records[name] = _sha256(source)
    return dict(parameters=parameters, shards=shard_records, other=other_records)


def _check_inventory(expected, present, label, *, kind="file"):
    expected, present = set(expected), set(present)
    if missing := sorted(expected - present):
        raise ValueError(f"{label}: missing {kind} {missing[0]}")
    if extra := sorted(present - expected, key=str):
        raise ValueError(f"{label}: extra {kind} {extra[0]}")


def _check_record_keys(record, label, *required, exact=False):
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be an object")
    for key in required:
        if key not in record:
            raise ValueError(f"{label}: missing key {key}")
    if exact:
        for key in record:
            if key not in required:
                raise ValueError(f"{label}: unknown key {key}")


def _check_record(record, **observed):
    for key, value in observed.items():
        if record.get(key) != value:
            raise ValueError(f"regroup record {key} mismatch")


def _verify_rewritten(source, target, record, *, group_bytes, group_rows):
    with _open_parquet(source) as before, _open_parquet(target) as after:
        rows = before.metadata.num_rows
        if rows != after.metadata.num_rows:
            raise ValueError("row count mismatch")
        before_rows, before_digest = _rows_evidence(before)
        after_rows, after_digest = _rows_evidence(after)
        if before_rows != rows or after_rows != rows:
            raise ValueError("streamed row count mismatch")
        if before_digest != after_digest:
            raise ValueError("rows_sha256 mismatch")
        _check_schema(source, target, before, after)
        largest, oversized = _check_groups(
            after.metadata, target.name, group_bytes=group_bytes, group_rows=group_rows
        )
        _check_record(
            record,
            rows=rows,
            rows_sha256=before_digest,
            source_groups=before.num_row_groups,
            output_groups=after.num_row_groups,
            largest_output_estimate=largest,
            oversized_rows=oversized,
            source_sha256=_sha256(source),
            source_bytes=source.stat().st_size,
            output_sha256=_sha256(target),
            output_bytes=target.stat().st_size,
        )


def _verify_copied(source, target, sha256):
    if _sha256(source) != sha256 or _sha256(target) != sha256:
        raise ValueError("copied file SHA-256 mismatch")


def verify_tree(parent, out, record) -> None:
    """Re-read the parent and the written tree, including the ledger if present."""
    parent, out = _roots(parent, out)
    shards, other = _parent_inventory(parent)
    outputs, directories = _inventory(out)
    if ledger := outputs.pop(_LEDGER_NAME, None):
        try:
            published = json.loads(ledger.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"output inventory: invalid {_LEDGER_NAME}") from error
        if published != record:
            raise ValueError(f"output inventory: {_LEDGER_NAME} differs from the record")
    expected = shards | other
    _check_inventory(expected, outputs, "output inventory")
    expected_directories = {
        directory.as_posix()
        for name in expected
        for directory in PurePosixPath(name).parents
        if directory.parts
    }
    _check_inventory(expected_directories, directories, "output inventory", kind="directory")
    _check_record_keys(record, "regroup record", "parameters", "shards", "other")
    parameters = record["parameters"]
    _check_record_keys(
        parameters,
        "regroup record parameters",
        "threshold_bytes",
        "group_bytes",
        "group_rows",
        exact=True,
    )
    try:
        _check_parameters(**parameters)
    except TypeError as error:
        raise ValueError(f"invalid regroup record parameters: {error}") from error
    shard_records, other_records = record["shards"], record["other"]
    _check_record_keys(shard_records, "regroup record shards")
    _check_record_keys(other_records, "regroup record other")
    _check_inventory(shards, shard_records, "shard record inventory")
    _check_inventory(other, other_records, "attribution record inventory")
    for name, source in shards.items():
        target = outputs[name]
        shard = shard_records[name]
        try:
            _check_record_keys(shard, "shard record")
            copied = not needs_regroup(source, threshold_bytes=parameters["threshold_bytes"])
            if shard.get("copied", False) != copied:
                raise ValueError("copy or rewrite decision mismatch")
            if copied:
                _check_record_keys(shard, "shard record", "sha256", "rows_sha256")
                _verify_copied(source, target, shard["sha256"])
                metadata = pq.read_metadata(target)
                largest, _ = _check_groups(metadata, target.name)
                _check_record(
                    shard,
                    bytes=target.stat().st_size,
                    groups=metadata.num_row_groups,
                    largest_estimate=largest,
                )
            else:
                _verify_rewritten(
                    source,
                    target,
                    shard,
                    group_bytes=parameters["group_bytes"],
                    group_rows=parameters["group_rows"],
                )
        except (ValueError, OSError) as error:
            raise ValueError(f"{name}: {error}") from error
    for name, source in other.items():
        try:
            _verify_copied(source, outputs[name], other_records[name])
        except (ValueError, OSError) as error:
            raise ValueError(f"{name}: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--threshold-bytes", type=int, default=THRESHOLD_BYTES)
    parser.add_argument("--group-bytes", type=int, default=GROUP_BYTES)
    parser.add_argument("--group-rows", type=int, default=GROUP_ROWS)
    args = parser.parse_args(argv)
    record = regroup_tree(
        args.parent,
        args.out,
        threshold_bytes=args.threshold_bytes,
        group_bytes=args.group_bytes,
        group_rows=args.group_rows,
    )
    verify_tree(args.parent, args.out, record)
    with (args.out / _LEDGER_NAME).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
