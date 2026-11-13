"""Streaming native Parquet validation, measurement and integrity primitives."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

MAX_ROW_VALUES = 2_000_000
MAX_ROW_GROUP_BYTES = 512 * 1024**2
MAX_ROW_GROUP_VALUES = 32_000_000


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


def iter_rows(folder, kind):
    """Yield validated [V,T] targets and [C,T] covariates, one Parquet row at a time."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted(Path(folder).glob("*.parquet"))
    if not paths or not any(p.name.startswith("train-") for p in paths):
        raise ValueError("folder has no train shards")
    required = {"item_id", "source", "freq", "target"}
    if kind == "covariates":
        required.add("covariates")
    for path in paths:
        if not path.name.startswith(("train-", "validation-")) or path.is_symlink():
            raise ValueError(f"unsupported shard: {path}")
        parquet = pq.ParquetFile(path)
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
        for group in range(parquet.num_row_groups):
            metadata = parquet.metadata.row_group(group)
            if (
                metadata.total_byte_size > MAX_ROW_GROUP_BYTES
                or sum(metadata.column(i).num_values for i in range(metadata.num_columns))
                > MAX_ROW_GROUP_VALUES
            ):
                raise ValueError(f"row group exceeds memory guard: {path}")
        for number, batch in enumerate(parquet.iter_batches(batch_size=1)):
            row = batch.to_pylist()[0]
            if any(
                not isinstance(row[n], str) or not row[n] for n in ("item_id", "source", "freq")
            ):
                raise ValueError(f"invalid native identity: {path}:{number}")
            try:
                target = np.asarray(row["target"], dtype=np.float64)
                if kind == "covariates":
                    if target.ndim != 1:
                        raise ValueError("covariate target must have shape [T]")
                    target = target[None, :]
                    cov = np.asarray(row["covariates"], dtype=np.float64)
                    if cov.ndim == 1 and cov.size == 0:
                        cov = np.empty((0, target.shape[1]))
                else:
                    cov = np.empty((0, target.shape[-1] if target.ndim else 0))
                if target.ndim != 2 or not all(target.shape):
                    raise ValueError("target must have nonempty shape [V,T]")
                if cov.ndim != 2 or cov.shape[1] != target.shape[1]:
                    raise ValueError("covariates must be aligned shape [C,T]")
                if target.size + cov.size > MAX_ROW_VALUES:
                    raise ValueError("row exceeds memory guard")
                if (
                    np.isinf(target).any()
                    or np.isinf(cov).any()
                    or np.any(np.abs(target) > np.finfo(np.float32).max)
                    or np.any(np.abs(cov) > np.finfo(np.float32).max)
                ):
                    raise ValueError("native value is not finite in trainer float32")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid/ragged native shape at {path}:{number}: {exc}") from exc
            yield dict(row=row, target=target, covariates=cov, shard=path.name, row_index=number)


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


def measure_folder(folder, kind):
    """Exact streamed counts; first training shard must cover supports and proportions."""
    all_hist = {k: Counter() for k in ("variates", "lengths", "frequencies")}
    first_hist = {k: Counter() for k in all_hist}
    train_hist = {k: Counter() for k in all_hist}
    first = min(p.name for p in Path(folder).glob("train-*.parquet"))
    rows = observed = values = cov_observed = cov_values = long_rows = 0
    channels, domains, subdomains, splits = Counter(), Counter(), Counter(), Counter()
    for rec in iter_rows(folder, kind):
        row, target, cov = rec["row"], rec["target"], rec["covariates"]
        v, t = target.shape
        rows += 1
        long_rows += t >= 16384
        values += target.size
        observed += int(np.isfinite(target).sum())
        cov_values += cov.size
        cov_observed += int(np.isfinite(cov).sum())
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
        tv[key] = (
            sum(abs(a[x] / sum(a.values()) - b[x] / sum(b.values())) for x in a) / 2
            if sum(b.values())
            else 1.0
        )
    representative = all(supports.values()) and max(tv.values()) <= 0.1
    return dict(
        rows=rows,
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
