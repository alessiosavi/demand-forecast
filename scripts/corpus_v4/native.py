"""Streaming native Parquet validation, measurement and integrity primitives."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

DEFAULT_MEMORY_BUDGET_BYTES = 32 * 1024**3
READ_BUFFER_BYTES = 64 * 1024
VALUE_CHUNK_POINTS = 65536
MEASUREMENT_VERSION = "buffered-arrow-1"


def peak_rss_bytes():
    """getrusage reports bytes on macOS and KiB on Linux."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
        1 if sys.platform == "darwin" else 1024
    )


class MemoryBudget:
    """Process RSS budget, with a hard per-row decoded-size allowance for Arrow.

    Arrow must decode a whole row. Its numeric storage is a proven lower bound;
    decoder overhead is measured by the worker watchdog rather than guessed
    from a group-size multiplier. Pages are buffered, independently of groups.
    """

    def __init__(self, limit=DEFAULT_MEMORY_BUDGET_BYTES):
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("memory budget must be a positive integer byte count")
        self.limit = limit
        self.check()

    def check(self, *, row_bytes=0, context="reader"):
        rss = peak_rss_bytes()
        if rss > self.limit or row_bytes > self.limit:
            raise ValueError(
                f"memory budget exceeded at {context}: peak_rss_bytes={rss}, "
                f"row_bytes={row_bytes}, hard_row_bytes={self.limit}, "
                f"budget_bytes={self.limit}; buffered Arrow rows require whole-row decoding"
            )


def start_memory_watchdog(limit):
    """Stop a disposable folder worker if a decoder allocation crosses its RSS budget.

    Sampling is every 10 ms; RSS can overshoot by allocations between samples.
    Running it in the worker also covers Arrow's atomic whole-row decode before
    iter_batches yields. The parent records the error and continues other folders.
    """
    MemoryBudget(limit)

    def watch():
        while True:
            rss = peak_rss_bytes()
            if rss > limit:
                os.write(
                    2,
                    f"memory budget exceeded: peak_rss_bytes={rss}, budget_bytes={limit}\n".encode(),
                )
                os._exit(97)
            time.sleep(0.01)

    threading.Thread(target=watch, daemon=True).start()


def memory_budget_arg(value):
    """CLI GiB value (including fractions), expressed internally as integer bytes."""
    import argparse
    import math

    try:
        gib = float(value)
        if not math.isfinite(gib) or gib <= 0:
            raise ValueError
        return max(1, int(gib * 1024**3))
    except (ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("memory budget must be positive GiB") from exc


def value_chunks(values):
    """Nonoverlapping views for exact statistics and bounded validation temporaries."""
    flat = values.reshape(-1)
    for start in range(0, flat.size, VALUE_CHUNK_POINTS):
        yield flat[start : start + VALUE_CHUNK_POINTS]


def atomic_json(path, value):
    """Publish only complete JSON, including during preemption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path, algorithm="sha256"):
    """Stream a file's SHA256 or Hub Git-blob SHA1 without retaining its payload."""
    h = hashlib.new("sha1" if algorithm == "git-sha1" else algorithm)
    if algorithm == "git-sha1":
        h.update(f"blob {Path(path).stat().st_size}\0".encode())
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def folder_path(root, entry):
    """Resolve the sole permitted location; quarantine is never a reader root."""
    root = Path(root)
    path = root / ("quarantine" if entry["status"] == "quarantined" else "") / entry["folder"]
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
        raise ValueError(f"symlink in corpus destination: {path}")
    return path


def _matrix(scalar, *, nested, name):
    """Use Arrow numeric storage directly; never build Python floats per value."""
    import pyarrow as pa

    if not scalar.is_valid:
        raise ValueError(f"{name} must have nonempty shape")
    array = scalar.values
    if nested:
        if array.null_count:
            raise ValueError(f"{name} has null/ragged channels")
        lengths = np.array([len(channel.values) for channel in array], dtype=np.int64)
        if not len(lengths):
            return np.empty((0, 0))
        if not np.all(lengths == lengths[0]):
            raise ValueError(f"{name} must have aligned/ragged-free shape")
        # flatten respects slices/offsets, unlike .values on a sliced ListArray.
        if pa.types.is_fixed_size_list(array.type):
            leaf = array.values.slice(
                array.offset * array.type.list_size, len(array) * array.type.list_size
            )
        else:
            leaf = array.flatten()
        numeric = (
            np.full(len(leaf), np.nan)
            if pa.types.is_null(leaf.type)
            else leaf.to_numpy(zero_copy_only=False)
        )
        return numeric.reshape(len(lengths), int(lengths[0]))
    numeric = (
        np.full(len(array), np.nan)
        if pa.types.is_null(array.type)
        else array.to_numpy(zero_copy_only=False)
    )
    return numeric[None, :]


def parquet_paths(folder):
    paths = sorted(Path(folder).glob("*.parquet"))
    if not paths or not any(p.name.startswith("train-") for p in paths):
        raise ValueError("folder has no train shards")
    for path in paths:
        if not path.name.startswith(("train-", "validation-")) or path.is_symlink():
            raise ValueError(f"unsupported shard: {path}")
    return paths


def folder_frequencies(folder):
    """Discover aliases from scalar columns only, without decoding numeric payloads."""
    import pyarrow.parquet as pq

    aliases = set()
    for path in parquet_paths(folder):
        with pq.ParquetFile(path, buffer_size=READ_BUFFER_BYTES, pre_buffer=False) as parquet:
            for batch in parquet.iter_batches(batch_size=4096, columns=["freq"], use_threads=False):
                aliases.update(batch.column(0).to_pylist())
    if any(not isinstance(alias, str) or not alias for alias in aliases):
        raise ValueError("invalid native frequency")
    return sorted(aliases)


def iter_rows(folder, kind, *, memory_budget_bytes=DEFAULT_MEMORY_BUDGET_BYTES):
    """Page-buffered native rows; numeric arrays are views, consumers chunk work.

    batch_size=1 alone still retains whole compressed column chunks. A positive
    buffer_size AND disabled pre_buffer avoid that allocation (measured offline).
    Whole-row Arrow decoding remains the stated hard limit, not row-group size.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    budget = MemoryBudget(memory_budget_bytes)
    required = {"item_id", "source", "freq", "target"}
    if kind == "covariates":
        required.add("covariates")
    for path in parquet_paths(folder):
        parquet = pq.ParquetFile(path, buffer_size=READ_BUFFER_BYTES, pre_buffer=False)
        schema = parquet.schema_arrow
        if not required <= set(schema.names):
            raise ValueError(f"missing native columns: {path}")
        for name in ("item_id", "source", "freq"):
            if not (
                pa.types.is_string(schema.field(name).type)
                or pa.types.is_large_string(schema.field(name).type)
            ):
                raise ValueError(f"{name} must be a string: {path}")
        for name, depth in (("target", 1 if kind == "covariates" else 2), ("covariates", 2)):
            if name not in required:
                continue
            dtype = schema.field(name).type
            for _ in range(depth):
                if not (
                    pa.types.is_list(dtype)
                    or pa.types.is_large_list(dtype)
                    or pa.types.is_fixed_size_list(dtype)
                ):
                    raise ValueError(f"{name} must be nested numeric lists: {path}")
                dtype = dtype.value_type
            if not (
                pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_null(dtype)
            ):
                raise ValueError(f"{name} must be numeric: {path}")
        columns = sorted(required | ({"domain", "subdomain"} & set(schema.names)))
        try:
            number = 0
            for group in range(parquet.num_row_groups):
                metadata = parquet.metadata.row_group(group)
                group_values = sum(
                    metadata.column(i).num_values for i in range(metadata.num_columns)
                )
                # A one-row group proves a minimum row size before decoding.
                minimum = (
                    sum(
                        metadata.column(i).num_values
                        * (4 if metadata.column(i).physical_type in {"FLOAT", "INT32"} else 8)
                        for i in range(metadata.num_columns)
                        if metadata.column(i).path_in_schema.split(".")[0]
                        in {"target", "covariates"}
                    )
                    if metadata.num_rows == 1
                    else 0
                )
                budget.check(row_bytes=minimum, context=f"{path}:group={group}")
                for batch in parquet.iter_batches(
                    batch_size=1, row_groups=[group], columns=columns, use_threads=False
                ):
                    row = {
                        name: batch.column(name)[0].as_py()
                        for name in columns
                        if name not in {"target", "covariates"}
                    }
                    if any(
                        not isinstance(row[n], str) or not row[n]
                        for n in ("item_id", "source", "freq")
                    ):
                        raise ValueError(f"invalid native identity: {path}:{number}")
                    try:
                        target = _matrix(
                            batch.column("target")[0], nested=kind != "covariates", name="target"
                        )
                        cov = (
                            _matrix(batch.column("covariates")[0], nested=True, name="covariates")
                            if kind == "covariates"
                            else np.empty((0, target.shape[-1]))
                        )
                        if target.ndim != 2 or not all(target.shape):
                            raise ValueError("target must have nonempty shape [V,T]")
                        if cov.shape == (0, 0):
                            cov = np.empty((0, target.shape[1]))
                        if cov.ndim != 2 or cov.shape[1] != target.shape[1]:
                            raise ValueError("covariates must be aligned shape [C,T]")
                        budget.check(
                            row_bytes=target.nbytes + cov.nbytes, context=f"{path}:{number}"
                        )
                        for matrix in (target, cov):
                            for part in value_chunks(matrix):
                                if (
                                    np.isinf(part).any()
                                    or np.any(part > np.finfo(np.float32).max)
                                    or np.any(part < -np.finfo(np.float32).max)
                                ):
                                    raise ValueError(
                                        "native value is not finite in trainer float32"
                                    )
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"invalid/ragged native shape at {path}:{number}: {exc}"
                        ) from exc
                    yield dict(
                        row=row,
                        target=target,
                        covariates=cov,
                        shard=path.name,
                        row_index=number,
                        row_group_bytes=metadata.total_byte_size,
                        row_group_values=group_values,
                        row_group_rows=metadata.num_rows,
                    )
                    number += 1
                    # Do not retain the previous row while Arrow decodes the next one.
                    del target, cov, batch
        finally:
            parquet.close()


def _summary(hist):
    total = sum(hist.values())
    if not total:
        return dict(count=0, min=None, max=None, mean=None, histogram={})
    return dict(
        count=total,
        min=min(hist),
        max=max(hist),
        mean=sum(k * v for k, v in hist.items()) / total,
        histogram={str(k): v for k, v in sorted(hist.items())},
    )


def _measure_folder(folder, kind, memory_budget_bytes):
    """Exact streamed counts; first training shard must cover supports and proportions."""
    started = time.monotonic()
    largest_group = largest_group_values = largest_group_rows = longest = largest_row_bytes = 0
    all_hist = {k: Counter() for k in ("variates", "lengths", "frequencies")}
    first_hist = {k: Counter() for k in all_hist}
    train_hist = {k: Counter() for k in all_hist}
    first = min(p.name for p in Path(folder).glob("train-*.parquet"))
    rows = observed = values = cov_observed = cov_values = long_rows = 0
    channels, domains, subdomains, splits = Counter(), Counter(), Counter(), Counter()
    for rec in iter_rows(folder, kind, memory_budget_bytes=memory_budget_bytes):
        row, target, cov = rec["row"], rec["target"], rec["covariates"]
        v, t = target.shape
        largest_group = max(largest_group, rec["row_group_bytes"])
        largest_group_values = max(largest_group_values, rec["row_group_values"])
        largest_group_rows = max(largest_group_rows, rec["row_group_rows"])
        largest_row_bytes = max(largest_row_bytes, target.nbytes + cov.nbytes)
        longest = max(longest, t)
        rows += 1
        long_rows += t >= 16384
        values += target.size
        observed += sum(int(np.isfinite(part).sum()) for part in value_chunks(target))
        cov_values += cov.size
        cov_observed += sum(int(np.isfinite(part).sum()) for part in value_chunks(cov))
        channels[cov.shape[0]] += 1
        domains["present" if row.get("domain") is not None else "absent"] += 1
        subdomains["present" if row.get("subdomain") is not None else "absent"] += 1
        split = "train" if rec["shard"].startswith("train-") else "validation"
        splits[split] += 1
        for key, val in (("variates", v), ("lengths", t), ("frequencies", row["freq"])):
            all_hist[key][val] += 1
            if split == "train":
                train_hist[key][val] += 1
                if rec["shard"] == first:
                    first_hist[key][val] += 1
    if not sum(train_hist["lengths"].values()):
        raise ValueError("folder has no training rows")
    tv, supports = {}, {}
    for key in all_hist:
        a, b = train_hist[key], first_hist[key]
        supports[key] = set(a) <= set(b)
        total_a, total_b = sum(a.values()), sum(b.values())
        tv[key] = sum(abs(a[x] / total_a - b[x] / total_b) for x in a) / 2 if total_b else 1.0
    representative = all(supports.values()) and max(tv.values()) <= 0.1
    return dict(
        rows=rows,
        values=values + cov_values,
        longest_row=longest,
        largest_row_group_bytes=largest_group,
        largest_row_group_values=largest_group_values,
        largest_row_group_rows=largest_group_rows,
        largest_row_bytes=largest_row_bytes,
        measurement=dict(
            version=MEASUREMENT_VERSION,
            strategy="buffered-arrow-rows",
            wall_seconds=time.monotonic() - started,
            peak_rss_bytes=peak_rss_bytes(),
            memory_budget_bytes=memory_budget_bytes,
            hard_row_bytes=memory_budget_bytes,
        ),
        splits=dict(splits),
        variates_per_row=_summary(all_hist["variates"]),
        length_distribution=_summary(all_hist["lengths"]),
        rows_ge_16384=long_rows,
        observed_value_share=observed / values if values else None,
        observed_target_values=observed,
        target_values=values,
        observed_covariate_values=cov_observed,
        covariate_values=cov_values,
        frequency_strings=dict(sorted(all_hist["frequencies"].items())),
        domain=dict(domains),
        subdomain=dict(subdomains),
        covariate_channel_counts={str(k): v for k, v in sorted(channels.items())},
        first_shard_representative=dict(
            shard=first,
            representative=representative,
            support_covered=supports,
            total_variation=tv,
            maximum_total_variation=0.1,
        ),
    )


def measure_folder(folder, kind, *, memory_budget_bytes=DEFAULT_MEMORY_BUDGET_BYTES):
    """Measure in a fresh worker so per-folder RSS is not a prior folder's high-water mark."""
    result = subprocess.run(
        [sys.executable, "-m", __name__, str(folder), kind, str(memory_budget_bytes)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(f"measurement failed for {folder}: {result.stderr.strip()}")
    return json.loads(result.stdout)


if __name__ == "__main__":
    try:
        start_memory_watchdog(int(sys.argv[3]))
        print(json.dumps(_measure_folder(sys.argv[1], sys.argv[2], int(sys.argv[3]))))
    except (ValueError, OSError, OverflowError, MemoryError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
