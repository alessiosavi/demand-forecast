"""Reproducible, resumable corpus-v2 content audit against fev dataset columns."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast.bench.specs import (
    FEV_DATASETS_REVISION,
    FevDatasetTask,
    fev_bench_tasks,
    fev_bench_tasks_path,
)
from demand_forecast.pretrain import audit_match as match
from demand_forecast.pretrain.corpus import CORPUS_KINDS, _sha256, read_manifest

DEFAULT_KINDS = ("chronos_real", "gifteval_pretrain")
FREQUENCY_CLASSES = ("15T", "30T", "1H", "1D", "1W", "1M", "1Q", "1Y")
METHOD_VERSION = "run11-corpus-v1"
SCAN_CHUNK_ROWS = 256
SCAN_CHUNK_MIN_GROUPS = 32
SCAN_CHUNKS_PER_WORKER = 4
METHOD = """All four hash families scan every numeric sequence column, including targets
and covariates. Within each kind and frequency class, candidate groups have at
least 3 matched windows or summed window spans of at least 24 points at one
offset. Masked rows expand all four families without multiplicity caps; the
first pass keeps at most 512 candidate groups and the post-pass removes that
candidate limit. The expansion memory guard remains explicit: unresolved rows
are reported and cause exit 2. All passes keep at most 2,048 (series, offset)
groups per row and 128 offsets per candidate series, ranked by descending hit
count with ascending series/offset ties; coverage counters record the drops.
Every retained candidate pair is fitted at each retained qualifying offset,
using merged intervals, window-map regrouping and piecewise refinement.
A copy has at least 24 overlapping finite observations, non-degenerate segments
on both sides, and exact fraction >= 0.99 (isclose atol=0.005, rtol=1e-6, as in
run 11) or maximum affine residual <= max(0.01, 0.005 * index segment range).
The map is corpus = slope * benchmark + intercept; bounds are zero-based and
half-open in the scanned variant. Non-degeneracy requires two levels, range
above 64 float32 ulps of its magnitude, and (after tolerance quantisation) four
off-modal points and seven degrees of freedom: runs-1 + levels-2. Monthly rows
and monthly aggregates also apply the calendar information rule. Daily columns
are scanned natively, as weekly sums at seven anchors, and as full calendar-month
sums and means. Byte-identical float64 rows (with canonical NaNs and matching
calendar dimensions) are scanned once, with counters multiplied by group size.
Copies and near pairs retain all notable segments and all duplicate row IDs;
counters count pairs, while rejection counters count fits and reasons per side.
Target policy exclusions are retained in full; covariate rejection records use
the run-11 per-reason and total record caps within each scan."""
LIMITS = """Known limits of the matcher (specification §3b, accepted and not fixed further).
(1) The piecewise refinement grows pieces with a slack of twice the copy
tolerance, so a splice of three or more maps whose slopes differ by only 1–2%
is merged across its boundaries and missed, and failed pieces are not
re-refined. (2) Above-cap 4-run postings are expanded only for masked rows: a
row that also has an unrelated informative hit elsewhere never reaches the
fallback, so a piecewise-constant copy of 5–7 runs in such a row is examined
only through 8-run fingerprints (which exist only from 8 runs). (3) A copy
whose boundary runs are clipped changes both 4-run fingerprints of a 5-run
segment, and tolerance-level noise can drop a boundary window so a 24-point
copy is found as 23 points; “no candidate group” therefore does not strictly
imply coverage. (4) The tolerance-aware counting is not invariant to shifts
across rounding-bin boundaries, and the calendar predicate as a whole is not
shift-invariant although its local-rate test is. These are limits of a
heuristic value-matching audit; a reviewer can construct a copy that escapes
any such matcher, which is why matching is not proof of source-level
non-overlap. The information thresholds are policies chosen after inspecting
chance matches, not independent evidence of provenance."""


def constants() -> dict:
    """Record every named matcher parameter, including arithmetic thresholds."""
    return {
        name: value.item() if isinstance(value, np.generic) else value
        for name, value in vars(match).items()
        if name.isupper()
        and isinstance(value, (int, float, dict, np.generic))
        and name != "FAMILIES"
    }


def _json_bytes(value) -> bytes:
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(val) for val in item]
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    return (json.dumps(clean(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _fingerprint(value) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json(path: Path, value) -> None:
    """Publish complete scans atomically so interruptions cannot fake completion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _offset(alias):
    # Pandas 3 removed the old aliases still stored in upstream datasets.
    aliases = {"T": "min", "H": "h", "M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}
    value = re.sub(r"^(\d*)(T|H|M|Q|Y|A)(?=-|$)", lambda m: m[1] + aliases[m[2]], str(alias))
    return pd.tseries.frequencies.to_offset(value)


def frequency_class(alias: str) -> str | None:
    """Normalize anchors and equivalent offsets without rounding frequencies."""
    try:
        offset = _offset(alias)
    except (ValueError, TypeError):
        return None
    if offset.n == 1:
        for types, name in (
            ((pd.offsets.BusinessDay,), "1D"),
            ((pd.offsets.Week,), "1W"),
            ((pd.offsets.MonthBegin, pd.offsets.MonthEnd), "1M"),
            ((pd.offsets.QuarterBegin, pd.offsets.QuarterEnd), "1Q"),
            ((pd.offsets.YearBegin, pd.offsets.YearEnd), "1Y"),
        ):
            if isinstance(offset, types):
                return name
    try:
        return {
            15 * 60 * 10**9: "15T",
            30 * 60 * 10**9: "30T",
            3600 * 10**9: "1H",
            86400 * 10**9: "1D",
            7 * 86400 * 10**9: "1W",
        }.get(offset.nanos)
    except ValueError:
        return None


def _timestamp_frequency(timestamps, explicit=None):
    stamps = pd.DatetimeIndex(timestamps)
    if stamps.hasnans or not stamps.is_monotonic_increasing or stamps.has_duplicates:
        raise ValueError("audit timestamps must be finite, increasing and unique")
    alias = explicit or (pd.infer_freq(stamps) if len(stamps) >= 3 else None)
    if alias is None and len(stamps) >= 2:
        delta = pd.Series(stamps[1:] - stamps[:-1]).mode().iloc[0]
        alias = pd.tseries.frequencies.to_offset(delta).freqstr
    return alias, stamps


def load_task_dataset(task: FevDatasetTask, cache_dir: str | Path):
    """Use the same HF train-split loader for pinned remote and local parquet."""
    from datasets import load_dataset

    if task.local_path:
        return load_dataset(
            "parquet",
            data_files={"train": task.local_path},
            split="train",
            cache_dir=str(cache_dir),
        )
    return load_dataset(
        task.dataset_path,
        task.dataset_config,
        split="train",
        revision=FEV_DATASETS_REVISION,
        cache_dir=str(cache_dir),
    )


def seq_columns(dataset) -> list[str]:
    """Numeric one-dimensional sequence features, supporting HF Sequence/List."""
    import pyarrow as pa

    result = []
    for field in dataset.features.arrow_schema:
        dtype = field.type
        if (
            pa.types.is_list(dtype)
            or pa.types.is_large_list(dtype)
            or pa.types.is_fixed_size_list(dtype)
        ):
            value = dtype.value_type
            if (
                pa.types.is_integer(value)
                or pa.types.is_floating(value)
                or pa.types.is_boolean(value)
            ):
                result.append(field.name)
    return result


def _task_metadata(task, cache_dir):
    dataset = load_task_dataset(task, cache_dir)
    columns = seq_columns(dataset)
    if not set(task.targets) <= set(columns):
        raise ValueError(
            f"{task.name}: missing numeric target columns {set(task.targets) - set(columns)}"
        )
    rows = []
    for i, row in enumerate(dataset.select_columns([task.id_column, task.timestamp_column])):
        alias, stamps = _timestamp_frequency(row[task.timestamp_column], task.freq)
        rows.append(
            {
                "row": i,
                "id": str(row[task.id_column]),
                "freq": alias,
                "frequency_class": frequency_class(alias),
                "length": len(stamps),
                "start": stamps[0].isoformat() if len(stamps) else None,
                "end": stamps[-1].isoformat() if len(stamps) else None,
                "daily_regular": bool(
                    alias
                    and frequency_class(alias) == "1D"
                    and stamps.equals(pd.date_range(stamps[0], periods=len(stamps), freq="D"))
                )
                if len(stamps)
                else False,
            }
        )
    # A short row cannot infer its frequency; borrow it only if the dataset is uniform.
    aliases = {r["freq"] for r in rows if r["freq"]}
    if len(aliases) == 1:
        alias = next(iter(aliases))
        for row in rows:
            if not row["freq"]:
                row["freq"], row["frequency_class"] = alias, frequency_class(alias)
    return {
        "task": asdict(task),
        "columns": columns,
        "rows": rows,
        "non_numeric_dynamic_columns": sorted(
            (set(task.known_dynamic_columns) | set(task.past_dynamic_columns)) - set(columns)
        ),
        "range": _range(rows),
        "unsupported_frequency_rows": sum(r["frequency_class"] is None for r in rows),
    }


def _range(rows):
    starts = [r["start"] for r in rows if r.get("start")]
    ends = [r["end"] for r in rows if r.get("end")]
    return {"start": min(starts) if starts else None, "end": max(ends) if ends else None}


class ArrayStore:
    """Read-only views into one disk-backed float64 array; safe across spawn workers."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.rows = json.loads((self.directory / "rows.json").read_text())
        self.values = (
            np.memmap(self.directory / "values.bin", dtype="<f8", mode="r")
            if self.rows
            else np.empty(0)
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if i >= len(self.rows):
            raise IndexError(i)
        row = self.rows[i]
        return self.values[row["offset"] : row["offset"] + row["length"]]


def _spool_corpus(refs, classes, directory):
    """Stream parquet once, retaining metadata and a single on-disk target copy."""
    import pyarrow.parquet as pq

    stores, handles = {}, {}
    skipped = Counter()
    try:
        for ref in refs:
            parquet = pq.ParquetFile(ref.path)
            for batch in parquet.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    if "freq" in row and "start" in row:
                        alias, start = row["freq"], pd.Timestamp(row["start"])
                    elif "timestamp" in row:
                        alias, stamps = _timestamp_frequency(row["timestamp"])
                        start = stamps[0] if len(stamps) else pd.NaT
                    else:
                        raise ValueError(
                            f"{ref.path}: audit needs start/freq or timestamp calendar columns"
                        )
                    cls = frequency_class(alias)
                    if cls not in classes:
                        skipped[f"{ref.kind}::{cls or str(alias)}"] += 1
                        continue
                    key = (ref.kind, cls)
                    if key not in stores:
                        path = directory / ref.kind / cls
                        path.mkdir(parents=True)
                        stores[key] = {"path": path, "rows": [], "points": 0}
                        handles[key] = (path / "values.bin").open("wb")
                    store = stores[key]
                    values = np.asarray(row["target"], dtype="<f8")
                    if values.ndim != 1 or not len(values) or pd.isna(start):
                        raise ValueError(f"invalid corpus series {row['id']}")
                    values.tofile(handles[key])
                    end = start + (len(values) - 1) * _offset(alias)
                    store["rows"].append(
                        {
                            "id": str(row["id"]),
                            "source": row.get("source", ref.kind),
                            "folder": row.get("folder", ref.kind),
                            "shard": ref.path.name,
                            "freq": alias,
                            "start": start.isoformat(),
                            "end": end.isoformat(),
                            "offset": store["points"],
                            "length": len(values),
                        }
                    )
                    store["points"] += len(values)
    finally:
        for handle in handles.values():
            handle.close()
    for store in stores.values():
        _write_json(store["path"] / "rows.json", store.pop("rows"))
    return stores, dict(sorted(skipped.items()))


def _save_index(directory):
    series = ArrayStore(directory)
    index = match.Index(series, [row["id"] for row in series.rows])
    for family, arrays in index.by.items():
        for i, array in enumerate(arrays):
            np.save(Path(directory) / f"{family}-{i}.npy", array)
    coverage = {
        **index.cov,
        "windows": {family: len(arrays[0]) for family, arrays in index.by.items()},
        "range": _range(series.rows),
        "folders": {
            folder: _range([r for r in series.rows if r["folder"] == folder])
            for folder in sorted({r["folder"] for r in series.rows})
        },
    }
    _write_json(Path(directory) / "index.json", coverage)
    return coverage


def _load_index(directory):
    index = match.Index.__new__(match.Index)
    index.series = ArrayStore(directory)
    index.ids = [row["id"] for row in index.series.rows]
    index.by = {
        family: tuple(
            np.load(Path(directory) / f"{family}-{i}.npy", mmap_mode="r") for i in range(4)
        )
        for family in match.FAMILIES
    }
    index.cov = json.loads((Path(directory) / "index.json").read_text())
    return index


_WORKER_INDEX = None


def _init_worker(directory):
    global _WORKER_INDEX
    _WORKER_INDEX = _load_index(directory)


@lru_cache(maxsize=1)
def _worker_dataset(task_json, cache_dir):
    return load_task_dataset(FevDatasetTask(**json.loads(task_json)), cache_dir)


class VariantRows:
    """Lazy column access avoids materialising a dataset or all its variants."""

    def __init__(self, dataset, metadata, column, variant):
        self.dataset, self.metadata = dataset, metadata
        self.column, self.variant = column, variant

    def __len__(self):
        return len(self.metadata)

    def values_and_days(self, i):
        meta = self.metadata[i]
        row = self.dataset[meta["row"]]
        x = np.asarray(row[self.column], np.float64)
        if x.ndim != 1 or len(x) != meta["length"]:
            raise ValueError(
                f"{self.column}: sequence/timestamp length mismatch at row {meta['row']}"
            )
        if self.variant.startswith("weekly@"):
            return match.weekly_sums(x, meta["start"], int(self.variant[-1])), None
        if self.variant.startswith("monthly@"):
            sums, means, days = match.monthly_aggs(x, meta["start"])
            return (sums if self.variant.endswith("sum") else means), days
        days = None
        if meta["frequency_class"] == "1M":
            days = match.days_in_month_of(row[self.dataset_timestamp_column])
        return x, days

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError(i)
        return self.values_and_days(i)[0]


class VariantDays:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, i):
        return self.rows.values_and_days(i)[1]


def _eligible_rows(job, task_rows):
    aggregation = job["variant"] != "native"
    return [
        r
        for r in task_rows
        if r["frequency_class"] == ("1D" if aggregation else job["frequency_class"])
        and (not aggregation or r["daily_regular"])
    ]


def _job_rows(job, *, kept=None):
    """Prepare lazy rows; chunks reuse the grouping job's aggregation filter."""
    task = FevDatasetTask(**job["task"])
    dataset = _worker_dataset(json.dumps(job["task"], sort_keys=True), job["cache_dir"])
    all_meta = json.loads(Path(job["metadata_path"]).read_text())["rows"]
    eligible = _eligible_rows(job, all_meta)
    rows = VariantRows(
        dataset.select_columns([task.timestamp_column, job["column"]]),
        eligible,
        job["column"],
        job["variant"],
    )
    rows.dataset_timestamp_column = task.timestamp_column
    if kept is not None:
        rows.metadata = [eligible[k] for k in kept]
    elif job["variant"] != "native":
        # Match run 11's exclusion of aggregates without eight full finite bins.
        rows.metadata = [r for i, r in enumerate(eligible) if rows[i] is not None]
    dims = VariantDays(rows) if job["frequency_class"] == "1M" else None
    return task, rows, dims, all_meta, eligible


def _scan_job(job):
    index = _WORKER_INDEX
    task, rows, dims, _, _ = _job_rows(job)
    pairs, rejections = [], []
    target = job["column"] in task.targets
    counts = match.scan(
        task.name,
        job["column"],
        rows,
        index,
        {},
        pairs,
        rejections,
        dims=dims,
        all_rejections=target,
    )
    return _finish_job(job, counts, pairs, rejections, rows.metadata, index.series.rows)


def _finish_job(job, counts, pairs, rejections, rows_metadata, corpus_rows):
    """Enrich and publish a complete scan without accessing index arrays."""
    task = FevDatasetTask(**job["task"])
    target = job["column"] in task.targets
    all_meta = json.loads(Path(job["metadata_path"]).read_text())["rows"]
    eligible = _eligible_rows(job, all_meta)
    counts["input_rows"] = len(all_meta)
    counts["aggregation_ineligible_rows"] = len(eligible) - len(rows_metadata)
    counts["irregular_daily_rows_skipped"] = (
        sum(r["frequency_class"] == "1D" and not r["daily_regular"] for r in all_meta)
        if job["variant"] != "native"
        else 0
    )
    for record in pairs + rejections:
        selected = [rows_metadata[i] for i in record["series_indices"]]
        record["series_indices"] = [r["row"] for r in selected]
        record["series_index"] = selected[0]["row"]
        record["benchmark_ids"] = [r["id"] for r in selected]
        record["benchmark_id"] = selected[0]["id"]
        # IDs are not assumed unique between folders; the matcher supplies ordinal.
        corpus = corpus_rows[record.pop("index_series_index")]
        record.update(
            task=task.name,
            kind=job["kind"],
            frequency_class=job["frequency_class"],
            variant=job["variant"],
            role="target" if target else "covariate",
            corpus_id=corpus["id"],
            corpus_folder=corpus["folder"],
            corpus_source=corpus["source"],
            corpus_shard=corpus["shard"],
            corpus_range={"start": corpus["start"], "end": corpus["end"]},
        )
    unresolved = [
        {"row": rows_metadata[i]["row"], "id": rows_metadata[i]["id"]}
        for i in counts["unresolved_ids"]
    ]
    for key in ("uncovered_ids", "unexplained_ids", "unresolved_ids"):
        counts[key] = [rows_metadata[i]["row"] for i in counts[key]]
    report = {
        key: job[key]
        for key in ("fingerprint", "scan_id", "kind", "frequency_class", "column", "variant")
    }
    report.update(
        task=task.name,
        role="target" if target else "covariate",
        counts=counts,
        copies=[p for p in pairs if p["is_copy"]],
        near_pairs=[p for p in pairs if not p["is_copy"]],
        rejections=rejections,
        unresolved=unresolved,
        policy_excluded_targets=[
            r
            for r in rejections
            if target
            and any(
                why in ("off_modal_lt4", "low_dof", "calendar_pattern")
                for why in r["degenerate_reason"]
            )
        ],
    )
    _write_json(Path(job["result_path"]), report)
    return job["result_path"]


def _group_job(job):
    """Deduplicate a whole scan before distributing its representatives."""
    _, rows, dims, _, eligible = _job_rows(job)
    positions = {r["row"]: i for i, r in enumerate(eligible)}
    return {
        "groups": match.group_rows(rows, dims),
        "kept": [positions[r["row"]] for r in rows.metadata],
        "rows_metadata": rows.metadata,
    }


def _chunk_job(payload):
    job = payload["job"]
    task, rows, dims, _, _ = _job_rows(job, kept=payload["kept"])
    return [
        match.scan_group(task.name, job["column"], i, idxs, rows, _WORKER_INDEX, dims=dims)
        for i, idxs in payload["groups"]
    ]


def _finish_chunked_job(job, grouped, chunks, corpus_rows):
    counts = match.new_cov()
    counts.update(
        unique_series=len(grouped["groups"]),
        initial_unresolved_rows=0,
        initial_truncated_rows=0,
        postpass_rows=0,
        postpass_candidates=0,
        unresolved_ids=[],
        rejection_records_omitted=0,
    )
    pairs, rejections = [], []
    match.finish_scan(
        counts,
        chain.from_iterable(chunks),
        pairs,
        rejections,
        all_rejections=job["column"] in FevDatasetTask(**job["task"]).targets,
    )
    return _finish_job(job, counts, pairs, rejections, grouped["rows_metadata"], corpus_rows)


def _timed_scan_job(job):
    """Return worker wall time alongside the path, without changing scan JSON."""
    started = time.monotonic()
    path = _scan_job(job)
    return path, time.monotonic() - started


def _job_sort_key(job):
    return (job["kind"], job["frequency_class"], job["task"]["name"], job["column"], job["variant"])


def _job_work(job, task_rows):
    return sum(row["length"] for row in _eligible_rows(job, task_rows))


def _order_pending(pending, metadata):
    task_rows = {item["task"]["name"]: item["rows"] for item in metadata}
    return sorted(
        pending,
        key=lambda job: (-_job_work(job, task_rows[job["task"]["name"]]), _job_sort_key(job)),
    )


def _parallel_scan_jobs(pool, pending, metadata, corpus_rows, workers):
    """Overlap grouping, whole scans and chunks, merging each scan in row order."""
    task_rows = {item["task"]["name"]: item["rows"] for item in metadata}
    futures = {}
    for job in _order_pending(pending, metadata):
        if len(_eligible_rows(job, task_rows[job["task"]["name"]])) >= SCAN_CHUNK_ROWS:
            state = {"started": time.monotonic()}
            futures[pool.submit(_group_job, job)] = ("group", job, state, None)
        else:
            futures[pool.submit(_timed_scan_job, job)] = ("scan", job, None, None)
    while futures:
        done, _ = wait(futures, return_when=FIRST_COMPLETED)
        for future in done:
            stage, job, state, position = futures.pop(future)
            result = future.result()
            if stage == "scan":
                path, elapsed = result
                yield job, path, elapsed
                continue
            if stage == "group":
                state["grouped"] = result
                groups = result["groups"]
                size = max(
                    SCAN_CHUNK_MIN_GROUPS,
                    math.ceil(len(groups) / (SCAN_CHUNKS_PER_WORKER * workers)),
                )
                state["chunks"] = [None] * math.ceil(len(groups) / size)
                state["remaining"] = len(state["chunks"])
                for position, start in enumerate(range(0, len(groups), size)):
                    payload = {
                        "job": job,
                        "kept": result["kept"],
                        "groups": groups[start : start + size],
                    }
                    futures[pool.submit(_chunk_job, payload)] = ("chunk", job, state, position)
            else:
                state["chunks"][position] = result
                state["remaining"] -= 1
            # Empty aggregate scans finish here without submitting any chunks.
            if not state["remaining"]:
                path = _finish_chunked_job(job, state["grouped"], state["chunks"], corpus_rows)
                yield job, path, time.monotonic() - state["started"]


def _jobs(metadata, kinds, fingerprint, out):
    jobs = []
    for item in metadata:
        task = item["task"]
        classes = sorted({r["frequency_class"] for r in item["rows"] if r["frequency_class"]})
        variants = [(cls, "native") for cls in classes]
        if "1D" in classes:
            variants += [("1W", f"weekly@{anchor}") for anchor in range(7)]
            variants += [("1M", "monthly@sum"), ("1M", "monthly@mean")]
        for kind in kinds:
            for cls, variant in variants:
                for column in item["columns"]:
                    identity = {
                        "task": task,
                        "column": column,
                        "variant": variant,
                        "kind": kind,
                        "frequency_class": cls,
                    }
                    scan_id = _fingerprint(identity)
                    jobs.append(
                        {
                            **identity,
                            "scan_id": scan_id,
                            "fingerprint": _fingerprint({"inputs": fingerprint, "scan": identity}),
                            "metadata_path": item["metadata_path"],
                            "cache_dir": str(out / ".cache" / "datasets"),
                            "result_path": str(out / "scans" / f"{scan_id}.json"),
                        }
                    )
    return sorted(jobs, key=_job_sort_key)


def _resume_result(job):
    try:
        report = json.loads(Path(job["result_path"]).read_text())
        required = (
            "scan_id",
            "task",
            "kind",
            "frequency_class",
            "column",
            "variant",
            "copies",
            "near_pairs",
            "rejections",
            "policy_excluded_targets",
            "unresolved",
        )
        if (
            report["fingerprint"] == job["fingerprint"]
            and report["counts"]["unresolved_rows"] == 0
            and all(key in report for key in required)
        ):
            return report
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def run_audit(
    root: Path,
    out: Path,
    *,
    tasks="fev100",
    kinds=DEFAULT_KINDS,
    workers=None,
    resume=False,
    progress=print,
):
    """Audit all selected shards; copies are results, unresolved rows are failures.

    Only complete scans with matching content and implementation fingerprints
    are reused. Worker count, scheduling and resume do not change audit.json.
    Per-run wall time and generated_at are the only varying report fields.
    """
    global _WORKER_INDEX
    started = time.monotonic()
    root, out = Path(root).resolve(), Path(out).resolve()
    kinds = tuple(sorted(set(kinds)))
    if not kinds or set(kinds) - CORPUS_KINDS.keys():
        raise ValueError(f"unknown or empty kinds: {kinds}")
    workers = max(1, (os.cpu_count() or 2) - 1) if workers is None else workers
    if workers < 1:
        raise ValueError("workers must be at least 1")
    definitions = fev_bench_tasks(tasks)
    refs = [ref for ref in read_manifest(root / "manifest.json") if ref.kind in kinds]
    for kind in kinds:
        if not any(ref.kind == kind for ref in refs):
            raise ValueError(f"no manifest shards for requested kind {kind}")
    # Rehash bytes, not just the manifest. A changed shard must never reuse results.
    seen = set()
    for ref in refs:
        if ref.path in seen:
            raise ValueError(f"duplicate manifest shard: {ref.path}")
        seen.add(ref.path)
        if ref.path.stat().st_size != ref.size or _sha256(ref.path) != ref.sha256:
            raise ValueError(f"corpus shard differs from manifest: {ref.path}")
    # A missing manifest entry is also an incomplete audit, even when its bytes match.
    for kind in kinds:
        actual = {p.resolve() for p in (root / CORPUS_KINDS[kind]).glob("train-*.parquet")}
        if actual != {ref.path for ref in refs if ref.kind == kind}:
            raise ValueError(f"manifest does not enumerate all {kind} shards")
    manifest_sha = _sha256(root / "manifest.json")
    tasks_sha = hashlib.sha256(fev_bench_tasks_path(tasks).read_bytes()).hexdigest()
    method_sha = _fingerprint(
        {name: _sha256(Path(__file__).with_name(name)) for name in ("audit.py", "audit_match.py")}
    )
    from importlib.metadata import version

    versions = {package: version(package) for package in ("numpy", "pandas", "pyarrow", "datasets")}
    inputs = {
        "corpus_manifest_sha256": manifest_sha,
        "tasks_yaml_sha256": tasks_sha,
        "dataset_revision": FEV_DATASETS_REVISION,
        "constants": constants(),
        "method_version": METHOD_VERSION,
        "method_sha256": method_sha,
        "versions": versions,
        "kinds": kinds,
        "local_dataset_sha256": {
            task.name: _sha256(Path(task.local_path)) for task in definitions if task.local_path
        },
    }
    fingerprint = _fingerprint(inputs)
    out.mkdir(parents=True, exist_ok=True)
    metadata = []
    for task in definitions:
        path = (
            out
            / ".cache"
            / f"task-{_fingerprint({'inputs': fingerprint, 'task': asdict(task)})}.json"
        )
        if resume and path.is_file():
            item = json.loads(path.read_text())
        else:
            progress(f"Reading task schema and calendars: {task.name}")
            item = _task_metadata(task, out / ".cache" / "datasets")
            _write_json(path, item)
        metadata.append({**item, "metadata_path": str(path)})
    jobs = _jobs(metadata, kinds, fingerprint, out)
    classes = {job["frequency_class"] for job in jobs}
    scans, indices, skipped = [], {}, {}
    with tempfile.TemporaryDirectory(prefix=".audit-work-", dir=out) as directory:
        stores, skipped = _spool_corpus(refs, classes, Path(directory))
        for (kind, cls), store in sorted(stores.items()):
            applicable = [
                job for job in jobs if job["kind"] == kind and job["frequency_class"] == cls
            ]
            pending = []
            for job in applicable:
                cached = _resume_result(job) if resume else None
                if cached is None:
                    pending.append(job)
                else:
                    scans.append(cached)
                    progress(
                        f"Resume: {job['task']['name']} / {job['column']} / {job['variant']} / {kind}"
                    )
            index_cache = out / ".cache" / f"index-{fingerprint}-{kind}-{cls}.json"
            # The parent needs only metadata to finish chunks, never mapped arrays.
            corpus_rows = json.loads((store["path"] / "rows.json").read_text())
            if pending or not index_cache.is_file():
                progress(f"Indexing {kind} {cls}: {len(corpus_rows)} series")
                coverage = _save_index(store["path"])
                _write_json(index_cache, coverage)
            else:
                coverage = json.loads(index_cache.read_text())
            indices[f"{kind}::{cls}"] = coverage
            if pending and workers == 1:
                try:
                    _init_worker(store["path"])
                    for job in pending:
                        path, elapsed = _timed_scan_job(job)
                        scans.append(json.loads(Path(path).read_text()))
                        progress(
                            f"Scanned {job['task']['name']} / {job['column']} / {job['variant']} / {kind}"
                            f" in {elapsed:.2f}s"
                        )
                finally:
                    _WORKER_INDEX = None
            elif pending:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker,
                    initargs=(str(store["path"]),),
                ) as pool:
                    for job, path, elapsed in _parallel_scan_jobs(
                        pool, pending, metadata, corpus_rows, workers
                    ):
                        scans.append(json.loads(Path(path).read_text()))
                        progress(
                            f"Scanned {job['task']['name']} / {job['column']} / {job['variant']} / {kind}"
                            f" in {elapsed:.2f}s"
                        )
            # Workers have exited and the serial parent has dropped its memory maps.
            del corpus_rows
            shutil.rmtree(store["path"])
            progress(f"Released {kind} {cls} index")
    scans.sort(key=lambda item: item["scan_id"])
    copies = [copy for scan in scans for copy in scan["copies"]]
    near = [pair for scan in scans for pair in scan["near_pairs"]]
    declaration = sorted({copy["task"] for copy in copies})
    summary = match.new_cov()
    for scan in scans:
        # The driver adds counters absent from run 11's base table.
        for key, value in scan["counts"].items():
            if key not in summary:
                summary[key] = [] if isinstance(value, list) else 0
        match._merge(summary, scan["counts"], 1)
    summary.update(
        column_scans=len(scans),
        copy_segments=len(copies),
        near_segments=len(near),
        skipped_corpus_series=sum(skipped.values()),
    )
    task_summaries = []
    for item in metadata:
        for kind in kinds:
            selected = [
                scan
                for scan in scans
                if scan["task"] == item["task"]["name"] and scan["kind"] == kind
            ]
            task_summaries.append(
                {
                    "task": item["task"]["name"],
                    "kind": kind,
                    "scans": len(selected),
                    "rows": sum(s["counts"]["n_series"] for s in selected),
                    "copies": sum(s["counts"]["copies"] for s in selected),
                    "near_pairs": sum(s["counts"]["near_pairs"] for s in selected),
                    "unresolved": sum(s["counts"]["unresolved_rows"] for s in selected),
                    "folders": sorted({c["corpus_folder"] for s in selected for c in s["copies"]}),
                    "task_range": item["range"],
                    "corpus_ranges": {
                        k.split("::")[1]: v["range"]
                        for k, v in indices.items()
                        if k.startswith(kind + "::")
                    },
                }
            )
    corpus_info = (
        json.loads((root / "corpus.json").read_text()) if (root / "corpus.json").is_file() else None
    )
    report = {
        **inputs,
        "input_fingerprint": fingerprint,
        "method": METHOD,
        "limits": LIMITS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "summary": summary,
        "tasks": [{k: v for k, v in item.items() if k != "metadata_path"} for item in metadata],
        "task_summary": task_summaries,
        "indices": indices,
        "skipped_corpus_series": skipped,
        "missing_index_classes": sorted(
            {f"{j['kind']}::{j['frequency_class']}" for j in jobs} - indices.keys()
        ),
        "corpus_totals": corpus_info.get("totals") if corpus_info else None,
        "scans": scans,
        "copies": copies,
        "near_pairs": near,
        "policy_excluded_targets": [r for scan in scans for r in scan["policy_excluded_targets"]],
        "trained_on_datasets": declaration,
    }
    _write_json(out / "audit.json", report)
    _write_json(
        out / "declaration.json",
        {
            "trained_on_datasets": declaration,
            "copies": copies,
            "generated_at": report["generated_at"],
            "corpus_manifest_sha256": manifest_sha,
            "tasks_yaml_sha256": tasks_sha,
            "complete": summary["unresolved_rows"] == 0,
        },
    )
    (out / "audit.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Corpus content audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Status: {'INCOMPLETE — unresolved rows' if report['summary']['unresolved_rows'] else 'complete under the stated operational definition'}.",
        "",
        "## Task × kind summary",
        "",
        "| Task | Kind | Scans | Rows | Copy pairs | Near pairs | Unresolved | Corpus folders with copies |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["task_summary"]:
        lines.append(
            "| "
            + " | ".join(
                cell(row[key])
                for key in ("task", "kind", "scans", "rows", "copies", "near_pairs", "unresolved")
            )
            + " | "
            + cell(", ".join(row["folders"]))
            + " |"
        )
    lines += [
        "",
        "## Declaration",
        "",
        "Use these exact task names in `trained_on_datasets`. A copy in any numeric",
        "target or covariate column declares that task. This list is an empirical",
        "minimum; provenance may require additional declarations.",
        "",
        "```json",
        json.dumps({"trained_on_datasets": report["trained_on_datasets"]}, indent=2),
        "```",
        "",
        "`declaration.json` contains the complete copy records. `audit.json` contains",
        "every scan's coverage counters, copies, near pairs, policy exclusions,",
        "unresolved row IDs, and input fingerprints. Duplicate rows retain every ID.",
        "",
        "## Temporal facts",
        "",
        "These are stored series date ranges, not proof of common origin or temporal",
        "exclusion. Legacy synthetic calendars are dummy coordinates. Value matching",
        "does not require equal dates. Bounds cover entire series, not just fitted segments.",
        "",
        "| Task | Kind | Task range | Corpus ranges by frequency class |",
        "|---|---|---|---|",
    ]
    for row in report["task_summary"]:
        lines.append(
            "| "
            + " | ".join(cell(row[key]) for key in ("task", "kind", "task_range", "corpus_ranges"))
            + " |"
        )
    lines += [
        "",
        "## Coverage",
        "",
        f"Skipped corpus series with no fev class: {report['skipped_corpus_series']}",
        "",
        f"Classes with no corpus index: {report['missing_index_classes']}",
        "",
        "Unsupported benchmark frequencies, nonnumeric dynamic columns and irregular",
        "daily rows excluded from aggregation are recorded explicitly in `audit.json`.",
        "",
        "## Method and constants",
        "",
        report["method"],
        "",
        "```json",
        json.dumps(report["constants"], indent=2, sort_keys=True),
        "```",
        "",
        f"Corpus manifest SHA-256: `{report['corpus_manifest_sha256']}`",
        "",
        f"Tasks YAML SHA-256: `{report['tasks_yaml_sha256']}`",
        "",
        f"fev dataset revision: `{report['dataset_revision']}`",
        "",
        "## Limits",
        "",
        report["limits"],
        "",
    ]
    return "\n".join(lines)
