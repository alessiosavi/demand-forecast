"""Audit native Simons targets and covariates against full, pinned fev histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast.bench.specs import FEV_DATASETS_REVISION, fev_bench_tasks
from demand_forecast.pretrain import audit_match as old
from demand_forecast.pretrain.audit import _timestamp_frequency, load_task_dataset, seq_columns
from demand_forecast.pretrain.audit_controls import run_controls
from scripts.corpus_v4.freeze_manifest import DEFAULT_MANIFEST, read_manifest, validate_manifest
from scripts.corpus_v4.matching import (
    MAX_INDEX_POINTS,
    MAX_PHASES,
    METHOD_VERSION,
    aggregate_fixed,
    chunks,
    find_index_matches,
    find_matches,
    frequency,
    identifying_parent,
)
from scripts.corpus_v4.native import (
    DEFAULT_MEMORY_BUDGET_BYTES,
    VALUE_CHUNK_POINTS,
    MemoryBudget,
    atomic_json,
    digest,
    folder_frequencies,
    folder_path,
    iter_rows,
    memory_budget_arg,
    start_memory_watchdog,
    value_chunks,
)


@dataclass
class BenchmarkSeries:
    """One complete numeric benchmark column; evaluation cutoffs are never applied."""

    dataset: str
    item_id: str
    column: str
    freq: str | None
    values: np.ndarray
    timestamps: object = None
    domain: str = "unknown"


def load_benchmarks(cache):
    """Load all 100 task datasets at the audit's existing pinned Hub revision."""
    tasks = fev_bench_tasks("fev100")
    if len(tasks) != 100:
        raise ValueError("expected the frozen 100 fev tasks")
    for task in tasks:
        dataset = load_task_dataset(task, cache)
        columns = seq_columns(dataset)
        if not set(task.targets) <= set(columns):
            raise ValueError(f"benchmark target missing: {task.name}")
        # A short row cannot infer its frequency; borrow only a uniform dataset alias.
        aliases = {
            _timestamp_frequency(row[task.timestamp_column], task.freq)[0]
            for row in dataset.select_columns([task.timestamp_column])
        }
        aliases.discard(None)
        fallback = next(iter(aliases)) if len(aliases) == 1 else None
        for row in dataset:
            alias, stamps = _timestamp_frequency(row[task.timestamp_column], task.freq)
            alias = alias or fallback
            if alias is not None and frequency(alias) is None:
                raise ValueError(f"unknown benchmark frequency: {task.name}")
            for column in columns:
                values = np.asarray(row[column], np.float64)
                if values.ndim != 1 or len(values) != len(stamps):
                    raise ValueError(f"unaligned benchmark column: {task.name}/{column}")
                yield BenchmarkSeries(
                    task.name, str(row[task.id_column]), column, alias, values, stamps
                )


class BenchmarkStore:
    """SQLite-backed full histories; one series and one bounded index reside in RAM."""

    def __init__(self, path, series, *, memory_budget_bytes=DEFAULT_MEMORY_BUDGET_BYTES):
        self.budget = MemoryBudget(memory_budget_bytes)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("DROP TABLE IF EXISTS series")
        self.db.execute("PRAGMA cache_size=-8192")  # 8 MiB, independent of corpus size.
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute(
            "CREATE TABLE series (meta TEXT, start INTEGER, length INTEGER, timed INTEGER)"
        )
        self.values_path = self.path.with_suffix(".values")
        self.stamps_path = self.path.with_suffix(".stamps")
        with self.values_path.open("wb") as values_file, self.stamps_path.open("wb") as stamps_file:
            h, self.points, self.count = hashlib.sha256(), 0, 0
            self.samples = []
            self.coverage = dict(unknown_frequency_rows=[])
            sampled = set()
            for item in series:
                values = np.asarray(item.values, dtype="<f8")
                if values.ndim != 1:
                    raise ValueError("benchmark series must be one-dimensional")
                self.budget.check(
                    row_bytes=values.nbytes * (2 if item.timestamps is not None else 1),
                    context=f"benchmark {item.dataset}/{item.item_id}",
                )
                if item.freq is None:
                    self.coverage["unknown_frequency_rows"].append(
                        dict(
                            dataset=item.dataset,
                            item_id=item.item_id,
                            column=item.column,
                            length=len(values),
                        )
                    )
                meta = dict(
                    dataset=item.dataset,
                    item_id=item.item_id,
                    column=item.column,
                    freq=item.freq,
                    domain=item.domain,
                )
                encoded = json.dumps(meta, sort_keys=True).encode()
                stamps = (
                    pd.DatetimeIndex(item.timestamps).as_unit("ns").asi8.astype("<i8", copy=False)
                    if item.timestamps is not None
                    else None
                )
                if stamps is not None and len(stamps) != len(values):
                    raise ValueError("benchmark timestamps misaligned")
                h.update(len(encoded).to_bytes(8, "little") + encoded)
                h.update(values.nbytes.to_bytes(8, "little"))
                for part in value_chunks(values):
                    payload = part.tobytes()
                    h.update(payload)
                    values_file.write(payload)
                stamp_bytes = values.nbytes if stamps is not None else 0
                h.update(stamp_bytes.to_bytes(8, "little"))
                if stamps is None:
                    stamps_file.seek(values.nbytes, 1)  # sparse space keeps identical offsets
                else:
                    for part in value_chunks(stamps):
                        payload = part.tobytes()
                        h.update(payload)
                        stamps_file.write(payload)
                self.db.execute(
                    "INSERT INTO series VALUES (?, ?, ?, ?)",
                    (encoded.decode(), self.points * 8, len(values), int(stamps is not None)),
                )
                self.points += len(values)
                self.count += 1
                if (
                    (item.dataset, item.freq) not in sampled
                    and len(self.samples) < 128
                    and len(values) >= 96
                    and _identifying_parent(values, require_identifying=False)
                ):
                    self.samples.append(values[:512].copy())
                    sampled.add((item.dataset, item.freq))
        self.db.commit()
        self.fingerprint = h.hexdigest()
        if not self.count:
            raise ValueError("no benchmark series")

    def __iter__(self):
        for meta, start, length, timed in self.db.execute(
            "SELECT meta, start, length, timed FROM series"
        ):
            item = json.loads(meta)
            values = (
                np.memmap(self.values_path, dtype="<f8", mode="r", offset=start, shape=(length,))
                if length
                else np.empty(0, dtype="<f8")
            )
            times = None
            if timed:
                stamps = (
                    np.memmap(
                        self.stamps_path, dtype="<i8", mode="r", offset=start, shape=(length,)
                    )
                    if length
                    else np.empty(0, dtype="<i8")
                )
                times = pd.to_datetime(stamps)
            yield BenchmarkSeries(**item, values=values, timestamps=times)

    def close(self):
        """Release the on-disk store."""
        self.db.close()


def _calendar_alias(key):
    if key[0] == "fixed":
        return pd.tseries.frequencies.to_offset(pd.Timedelta(key[1], unit="ns")).freqstr
    return key[1]


def _rank(key):
    if key[0] == "fixed":
        return key[1]
    offset = pd.tseries.frequencies.to_offset(key[1])
    for classes, days in (
        (pd.offsets.Week, 7),
        ((pd.offsets.MonthBegin, pd.offsets.MonthEnd), 30),
        ((pd.offsets.QuarterBegin, pd.offsets.QuarterEnd), 91),
        ((pd.offsets.YearBegin, pd.offsets.YearEnd), 365),
    ):
        if isinstance(offset, classes):
            return offset.n * days * 86400 * 10**9
    return None


def _identifying_parent(values, *, require_identifying=True):
    """Same parent gates as matching.identifying_parent, with bounded temporaries.

    Quantized levels span at most ~202 bins under the unchanged tolerance.
    Counts/runs span chunk boundaries; periodicity uses the finite sequence,
    while the strict-ramp check retains original positions and missingness.
    """
    if len(values) <= VALUE_CHUNK_POINTS:
        return identifying_parent(values) if require_identifying else old.nondegenerate(values)[0]
    count, lo, hi = 0, np.inf, -np.inf
    first = last = None
    for start in range(0, len(values), VALUE_CHUNK_POINTS):
        part = np.asarray(values[start : start + VALUE_CHUNK_POINTS], np.float64)
        positions = np.flatnonzero(np.isfinite(part))
        if not len(positions):
            continue
        finite = part[positions]
        count += len(finite)
        lo, hi = min(lo, float(finite.min())), max(hi, float(finite.max()))
        if first is None:
            first = (start + int(positions[0]), float(finite[0]))
        last = (start + int(positions[-1]), float(finite[-1]))
    scale = max(abs(lo), abs(hi), old.PRECISION_SCALE_FLOOR)
    minimum = old.SEG_MIN if require_identifying else 3
    if count < minimum or hi - lo <= old.ULP_FACTOR * old.ULP32 * scale:
        return False
    tolerance = max(old.COPY_ATOL, old.COPY_RTOL * (hi - lo))
    levels, runs, previous = Counter(), 0, None
    periods, tail = set(range(1, min(32, count // 2) + 1)), np.empty(0)
    nonlinear = False
    for start in range(0, len(values), VALUE_CHUNK_POINTS):
        part = np.asarray(values[start : start + VALUE_CHUNK_POINTS], np.float64)
        positions = np.flatnonzero(np.isfinite(part))
        finite = part[positions]
        if not len(finite):
            continue
        q = np.round(finite / tolerance)
        distinct, counts = np.unique(q, return_counts=True)
        levels.update({float(k): int(n) for k, n in zip(distinct, counts, strict=True)})
        runs += int(np.count_nonzero(q[1:] != q[:-1])) + int(previous is None or q[0] != previous)
        previous = q[-1]
        joined = np.concatenate((tail, finite))
        for period in tuple(periods):
            begin = max(len(tail), period)
            if np.any(
                np.abs(joined[begin:] - joined[begin - period : len(joined) - period]) > tolerance
            ):
                periods.remove(period)
        tail = joined[-32:]
        line = first[1] + (last[1] - first[1]) * (
            (positions + start - first[0]) / (last[0] - first[0])
        )
        nonlinear |= bool(np.any(np.abs(finite - line) > 8 * old.ULP32 * scale))
    return (
        len(levels) >= 2
        and count - max(levels.values()) >= old.MIN_OFF_MODAL
        and runs - 1 + len(levels) - 2 >= old.MIN_DOF
        and (not require_identifying or (not periods and nonlinear))
    )


class _FixedAggregate:
    """Lazy exact fixed-bin reductions: chunks keep global phase and bin offsets."""

    def __init__(self, values, ratio, phase, method):
        self.values, self.ratio, self.phase, self.method = values, ratio, phase, method

    def __len__(self):
        return max(0, (len(self.values) - self.phase) // self.ratio)

    def __getitem__(self, part):
        start, stop, step = part.indices(len(self))
        if step != 1:
            raise ValueError("aggregation requires contiguous chunks")
        blocks = self.values[
            self.phase + start * self.ratio : self.phase + stop * self.ratio
        ].reshape(-1, self.ratio)
        # Numeric native storage can be float32. The former Python reader used
        # float64 throughout; retain that arithmetic without a whole-row cast.
        result = blocks.sum(axis=1, dtype=np.float64)
        return result if self.method == "sum" else result / self.ratio


def _variants(values, fine, coarse, timestamps=None, *, fine_alias=None, coarse_alias=None):
    """Yield all supported aggregation phases, or a precise unresolved-coverage reason."""
    if not _identifying_parent(values):
        yield None, dict(uninformative="non-identifying fine-frequency parent")
        return
    if fine[0] == coarse[0] == "fixed":
        if coarse[1] % fine[1]:
            yield None, dict(unresolved="non-integral frequency ratio")
            return
        ratio = coarse[1] // fine[1]
        if ratio > MAX_PHASES:
            yield None, dict(unresolved=f"aggregation phases {ratio} exceed {MAX_PHASES}")
            return
        for phase in range(ratio):
            for method in ("sum", "mean"):
                yield (
                    _FixedAggregate(values, ratio, phase, method),
                    dict(method=method, ratio=ratio, phase=phase, alignment="all fixed-bin phases"),
                )
        return
    if timestamps is None:
        yield (
            None,
            dict(unresolved="calendar aggregation needs absolute origin; native release lacks it"),
        )
        return
    alias = coarse_alias or _calendar_alias(coarse)
    offset = pd.tseries.frequencies.to_offset(alias)
    if coarse[0] == "calendar" and offset.n > 1:
        yield (
            None,
            dict(unresolved="multi-period calendar phase is not established by frequency alone"),
        )
        return
    stamps = pd.DatetimeIndex(timestamps)
    if (
        len(stamps) < 3
        or stamps.hasnans
        or not stamps.is_monotonic_increasing
        or stamps.has_duplicates
    ):
        yield None, dict(unresolved="invalid calendar timestamps")
        return
    regular = pd.date_range(
        stamps[0], periods=len(stamps), freq=fine_alias or _calendar_alias(fine)
    )
    if not stamps.equals(regular):
        yield None, dict(unresolved="irregular calendar parent")
        return
    source = pd.Series(values, index=stamps)
    bins = source.resample(alias)
    # Edge bins may be partial. Internal bins have all expected source timestamps.
    for method in ("sum", "mean"):
        aggregated = bins.sum(min_count=1) if method == "sum" else bins.mean()
        aggregated[bins.count() != bins.size()] = np.nan
        yield (
            aggregated.iloc[1:-1].to_numpy(),
            dict(
                method=method,
                phase="calendar",
                alignment="benchmark timestamps; edge bins discarded",
                _calendar_sizes=bins.size().iloc[1:-1].to_numpy(),
            ),
        )


def controls(samples, opportunities):
    """Run old offline controls plus benchmark-derived plants with the production matcher."""
    legacy = run_controls()
    legacy_ok = all(x["ok"] for key in ("controls", "self_tests") for x in legacy[key].values())
    selected = None
    for candidate in samples:
        if find_matches(candidate, candidate, opportunities=opportunities)[0]:
            selected = candidate
            break
    if selected is None:
        return dict(
            ok=False, legacy=legacy, reason="no identifying benchmark series for planted controls"
        )
    x = selected
    rng = np.random.default_rng(917)
    # Preserve an estimated seasonal template, independently resample innovations.
    finite_x = np.where(np.isfinite(x), x, np.nanmean(x))
    spectral = np.abs(np.fft.rfft(finite_x - finite_x.mean()))
    peak = int(np.argmax(spectral[1:]) + 1)
    period = min(365, max(2, round(len(x) / peak)))
    phases = np.arange(len(x)) % period
    template = np.array([finite_x[phases == phase].mean() for phase in range(period)])
    residual_scale = max(float((finite_x - template[phases]).std()), 1e-3)
    twin = template[phases] + rng.normal(0, residual_scale, len(x))
    plants = {
        "as_is": (x.astype("float32"), x, True),
        "scaled": ((x * 3.71 - 8.32).astype("float32"), x, True),
        "cropped": ((x[len(x) // 8 : 7 * len(x) // 8] * -2.4 + 7).astype("float32"), x, True),
        "seasonal_twin": (twin, x, False),
    }
    results = {}
    for name, (plant, target, expected) in plants.items():
        records, coverage = find_matches(plant, target, opportunities=opportunities)
        got = bool(records)
        results[name] = dict(ok=got == expected, expected=expected, detected=got, coverage=coverage)
    # Plant the benchmark as a fine-frequency parent, including substep innovations.
    noise = rng.normal(size=(len(x), 24))
    noise -= noise.mean(axis=1)[:, None]
    parent = (x[:, None] / 24 + noise).astype("float32").ravel()
    aggregated = aggregate_fixed(parent.astype(float), 24, 0, "sum")
    found = bool(find_matches(aggregated, x, opportunities=opportunities)[0])
    results["aggregated"] = dict(ok=found, expected=True, detected=found)
    return dict(
        ok=legacy_ok and all(x["ok"] for x in results.values()),
        legacy=legacy,
        planted=results,
        origin="full benchmark series",
        benchmark_values=len(x),
    )


def _segment_id(item_id):
    found = re.fullmatch(r"(.+)@(\d+)", item_id)
    return (found[1], int(found[2])) if found else (item_id, 0)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _index_batches(store, benchmark_alias, corpus_alias, direction):
    """Bound combined benchmark indices so corpus reads scale with points, not row count."""
    arrays, metadata, events, size = [], [], [], 0
    for bench in store:
        if bench.freq != benchmark_alias:
            continue
        variants = (
            _variants(
                bench.values,
                frequency(bench.freq),
                frequency(corpus_alias),
                bench.timestamps,
                fine_alias=bench.freq,
                coarse_alias=corpus_alias,
            )
            if direction == "benchmark_to_corpus"
            else [(bench.values, dict(method="identity"))]
        )
        for values, aggregation in variants:
            dimensions = aggregation.pop("_calendar_sizes", None)
            if values is None:
                events.append(aggregation)
                if len(events) >= 128:
                    yield arrays, metadata, events
                    arrays, metadata, events, size = [], [], [], 0
                continue
            for start, part in chunks(values):
                if arrays and (size + len(part) > MAX_INDEX_POINTS or len(arrays) >= 4096):
                    yield arrays, metadata, events
                    arrays, metadata, events, size = [], [], [], 0
                arrays.append(np.array(part, copy=True))
                metadata.append(
                    dict(
                        dataset=bench.dataset,
                        item_id=bench.item_id,
                        column=bench.column,
                        chunk_offset=start,
                        aggregation=aggregation,
                        calendar_sizes=(
                            dimensions[start : start + len(part)].copy()
                            if dimensions is not None
                            else None
                        ),
                    )
                )
                size += len(part)
            if len(events) >= 128:
                yield arrays, metadata, events
                arrays, metadata, events, size = [], [], [], 0
    if arrays or events:
        yield arrays, metadata, events


def _scan_folder(root, entry, store, opportunities, memory_budget_bytes):
    path = folder_path(root, entry)
    matches, coverage, searched, skipped = [], Counter(), {}, {}
    reasons = set()
    aggregation_seen = {}
    kind = entry["folder"].split("/")[0]
    budget = MemoryBudget(memory_budget_bytes)
    aliases = folder_frequencies(path)
    benchmark_groups = {}
    for (encoded,) in store.db.execute("SELECT meta FROM series"):
        meta = json.loads(encoded)
        benchmark_groups.setdefault(meta["freq"], set()).add(meta["dataset"])
    benchmark_aliases = sorted(benchmark_groups, key=lambda alias: alias or "")
    for corpus_alias in aliases:
        cf = frequency(corpus_alias)
        for benchmark_alias in benchmark_aliases:
            bf = frequency(benchmark_alias)
            datasets = sorted(benchmark_groups[benchmark_alias])
            pairs = [
                dict(
                    benchmark=name,
                    benchmark_frequency=benchmark_alias,
                    corpus_frequency=corpus_alias,
                    domains="all pairs (conservative)",
                )
                for name in datasets
            ]
            if cf is None or bf is None:
                failure = "unsupported frequency"
            elif cf == bf:
                failure = None
                rule, direction = "F", "same_frequency"
            elif _rank(cf) is None or _rank(bf) is None:
                failure = "unsupported cross-frequency ordering"
            else:
                failure = None
                rule = "L"
                direction = (
                    "corpus_to_benchmark" if _rank(cf) < _rank(bf) else "benchmark_to_corpus"
                )
            if failure:
                for pair in pairs:
                    skipped[json.dumps(pair, sort_keys=True)] = dict(**pair, reason=failure)
                reasons.add(failure)
                continue
            for pair in pairs:
                pair.update(rule=rule, direction=direction)
                key = (pair["benchmark"], corpus_alias, benchmark_alias)
                searched[key] = dict(**pair, comparisons=0, matched=0, aggregations=[])
                aggregation_seen[key] = set()
            for arrays, metadata, events in _index_batches(
                store, benchmark_alias, corpus_alias, direction
            ):
                for event in events:
                    if "unresolved" in event:
                        reasons.add(event["unresolved"])
                        for pair in pairs:
                            skipped[json.dumps(pair, sort_keys=True)] = dict(
                                **pair, reason=event["unresolved"]
                            )
                    else:
                        coverage["uninformative_aggregation_parents"] += 1
                if not arrays:
                    continue
                batch_counts, batch_transforms = Counter(), {}
                for meta in metadata:
                    name = meta["dataset"]
                    batch_counts[name] += 1
                    encoded = json.dumps(meta["aggregation"], sort_keys=True)
                    batch_transforms.setdefault(name, set()).add(encoded)
                index = old.Index(arrays, [f"benchmark-{i}" for i in range(len(arrays))])
                budget.check(context=f"audit index {entry['folder']}")
                for rec in iter_rows(path, kind, memory_budget_bytes=memory_budget_bytes):
                    row = rec["row"]
                    if row["freq"] != corpus_alias:
                        continue
                    channels = [("target", i, v) for i, v in enumerate(rec["target"])]
                    channels += [("covariates", i, v) for i, v in enumerate(rec["covariates"])]
                    for column, channel, values in channels:
                        variants = (
                            _variants(
                                values,
                                cf,
                                bf,
                                fine_alias=corpus_alias,
                                coarse_alias=benchmark_alias,
                            )
                            if direction == "corpus_to_benchmark"
                            else [(values, dict(method="identity"))]
                        )
                        for query, aggregation in variants:
                            if query is None:
                                if "unresolved" in aggregation:
                                    reasons.add(aggregation["unresolved"])
                                    for pair in pairs:
                                        skipped[json.dumps(pair, sort_keys=True)] = dict(
                                            **pair, reason=aggregation["unresolved"]
                                        )
                                else:
                                    coverage["uninformative_aggregation_parents"] += 1
                                continue
                            for bx, xx in chunks(query):
                                records, counters = find_index_matches(
                                    xx,
                                    index,
                                    opportunities=opportunities,
                                    calendar_dimensions=[m["calendar_sizes"] for m in metadata],
                                )
                                budget.check(context=f"audit query {entry['folder']}")
                                coverage.update(counters)
                                for name, count in batch_counts.items():
                                    key = (name, corpus_alias, benchmark_alias)
                                    searched[key]["comparisons"] += count
                                    transforms = (
                                        {json.dumps(aggregation, sort_keys=True)}
                                        if direction == "corpus_to_benchmark"
                                        else batch_transforms[name]
                                    )
                                    for encoded in sorted(transforms - aggregation_seen[key]):
                                        searched[key]["aggregations"].append(encoded)
                                        aggregation_seen[key].add(encoded)
                                for record in records:
                                    meta = metadata[record["index_series_index"]]
                                    key = (meta["dataset"], corpus_alias, benchmark_alias)
                                    searched[key]["matched"] += 1
                                    base, offset = _segment_id(row["item_id"])
                                    record.update(
                                        rule=rule,
                                        aggregation=(
                                            aggregation
                                            if direction == "corpus_to_benchmark"
                                            else meta["aggregation"]
                                        ),
                                        corpus=dict(
                                            item_id=row["item_id"],
                                            base_id=base,
                                            segment_offset=offset,
                                            source=row["source"],
                                            shard=rec["shard"],
                                            row=rec["row_index"],
                                            column=column,
                                            channel=channel,
                                            chunk_offset=bx,
                                        ),
                                        benchmark={
                                            k: v
                                            for k, v in meta.items()
                                            if k not in {"aggregation", "calendar_sizes"}
                                        },
                                    )
                                    matches.append(record)
                                    if len(matches) >= 10000:
                                        reasons.add("evidence record limit reached")
                                        break
                                if len(matches) >= 10000:
                                    break
                            if len(matches) >= 10000:
                                break
                        if len(matches) >= 10000:
                            break
                    if len(matches) >= 10000:
                        break
                if len(matches) >= 10000:
                    break
            if len(matches) >= 10000:
                break
        if len(matches) >= 10000:
            break
    if any(coverage[k] for k in ("unresolved_rows", "groups_dropped", "offsets_dropped")):
        reasons.add("matcher candidate/resource cap dropped coverage")
    rules = {m["rule"] for m in matches}
    verdict = (
        "rule_f_match"
        if "F" in rules
        else "rule_l_match"
        if "L" in rules
        else "inconclusive"
        if reasons
        else "clean"
    )
    return dict(
        folder=entry["folder"],
        status=entry["status"],
        verdict=verdict,
        reason=sorted(reasons),
        matches=matches,
        coverage=dict(coverage),
        searched_pairs=list(searched.values()),
        skipped_pairs=list(skipped.values()),
        quarantine_action=(
            "releasable by PR; metadata review still required"
            if entry["status"] == "quarantined" and verdict == "clean"
            else "no automatic change"
        ),
    )


def _scan_in_subprocess(root, entry, store, opportunities, memory_budget_bytes):
    request = dict(
        root=str(root),
        entry=entry,
        store=str(store.path),
        opportunities=opportunities,
        memory_budget_bytes=memory_budget_bytes,
    )
    result = subprocess.run(
        [sys.executable, "-m", "scripts.corpus_v4.audit_fev_overlap", "--scan-worker"],
        input=json.dumps(request),
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(f"audit worker exit {result.returncode}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _scan_worker():
    request = json.load(sys.stdin)
    budget = request["memory_budget_bytes"]
    start_memory_watchdog(budget)
    store = BenchmarkStore.__new__(BenchmarkStore)
    store.path = Path(request["store"])
    store.values_path = store.path.with_suffix(".values")
    store.stamps_path = store.path.with_suffix(".stamps")
    store.db = sqlite3.connect(store.path)
    store.db.execute("PRAGMA cache_size=-8192")
    try:
        result = _scan_folder(
            request["root"], request["entry"], store, request["opportunities"], budget
        )
        print(json.dumps(result))
    finally:
        store.close()


def run_audit(
    root,
    out,
    manifest,
    benchmarks=None,
    *,
    state=None,
    control_runner=controls,
    memory_budget_bytes=DEFAULT_MEMORY_BUDGET_BYTES,
):
    """Restart by folder with hashes, benchmark identity, method and policy bound to state."""
    validate_manifest(manifest)
    root, out = Path(root), Path(out)
    state = Path(state or Path(os.environ.get("RUN_DIR", out)) / "state")
    state.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    # Offline datasets loading selects the newest cached config, ignoring revision.
    source = (
        benchmarks
        if benchmarks is not None
        else load_benchmarks(state / "benchmark-cache" / FEV_DATASETS_REVISION)
    )
    store = BenchmarkStore(
        state / "benchmark.sqlite", source, memory_budget_bytes=memory_budget_bytes
    )
    try:
        import pyarrow.parquet as pq

        identities, folder_reasons, input_errors = {}, {}, []
        permitted = {
            (
                Path("quarantine") / e["folder"]
                if e["status"] == "quarantined"
                else Path(e["folder"])
            ).as_posix()
            for e in manifest["folders"]
        }
        for pattern in (
            "univariate/*",
            "covariates/*",
            "quarantine/univariate/*",
            "quarantine/covariates/*",
        ):
            for path in root.glob(pattern):
                if path.relative_to(root).as_posix() not in permitted:
                    input_errors.append(
                        f"unlisted or misplaced corpus folder: {path.relative_to(root)}"
                    )
        marker = root / "corpus-identity.json"
        if marker.exists():
            recorded = json.loads(marker.read_text())
            expected = dict(
                repo_id=manifest["repo_id"],
                revision=manifest["revision"],
                folders=manifest["folders"],
            )
            if recorded != expected:
                input_errors.append("corpus provenance differs from manifest/revision")
        total_values_bound = 0
        for entry in manifest["folders"]:
            identities[entry["folder"]] = {}
            try:
                path = folder_path(root, entry)
                files = sorted(path.glob("*.parquet"))
                if not files:
                    raise ValueError("missing corpus folder/shards")
                receipt_path = path / "receipt.json"
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    expected_names = {Path(f["path"]).name for f in receipt["files"]}
                    if (
                        not expected_names <= {p.name for p in path.iterdir()}
                        or not (path / "stats.json").is_file()
                    ):
                        raise ValueError("incomplete published folder/receipt")
                for shard in files:
                    identities[entry["folder"]][shard.name] = digest(shard)
                    if shard.is_symlink():
                        raise ValueError(f"symlink shard: {shard}")
                    parquet = pq.ParquetFile(shard)
                    total_values_bound += parquet.metadata.serialized_size
                    total_values_bound += sum(
                        parquet.metadata.row_group(i).total_byte_size
                        for i in range(parquet.num_row_groups)
                    )
                    parquet.close()
            except Exception as exc:
                folder_reasons[entry["folder"]] = str(exc)
        opportunities = max(1, total_values_bound) * max(1, store.points) * (2 * MAX_PHASES * 16)
        identity = dict(
            manifest=manifest,
            benchmark=store.fingerprint,
            benchmark_revision=FEV_DATASETS_REVISION,
            method=METHOD_VERSION,
            reader_version="buffered-arrow-1",
            opportunities=opportunities,
        )
        fingerprint = _fingerprint(dict(**identity, folders=identities))
        checks = control_runner(store.samples, opportunities)
        folders = []
        for entry in manifest["folders"]:
            key = _fingerprint(
                dict(
                    **identity,
                    entry=entry,
                    hashes=identities[entry["folder"]],
                    memory_budget_bytes=memory_budget_bytes,
                )
            )
            stable_identity = {k: v for k, v in identity.items() if k != "opportunities"}
            resume_key = _fingerprint(
                dict(
                    **stable_identity,
                    entry=entry,
                    hashes=identities[entry["folder"]],
                    memory_budget_bytes=memory_budget_bytes,
                )
            )
            saved = state / "audit-folders" / f"{entry['folder'].replace('/', '--')}.json"
            report = None
            if saved.exists():
                previous = json.loads(saved.read_text())
                # A clean scan at a lower information threshold stays clean when
                # a repaired/missing folder increases the global opportunity bound.
                compatible_clean = (
                    previous.get("resume_fingerprint") == resume_key
                    and previous.get("opportunities", 0) <= opportunities
                    and previous["report"]["verdict"] == "clean"
                )
                if (previous.get("fingerprint") == key or compatible_clean) and not previous[
                    "report"
                ].get("error"):
                    report = previous["report"]
                    report["resumed"] = True
            if report is None:
                try:
                    if entry["folder"] in folder_reasons:
                        raise ValueError(folder_reasons[entry["folder"]])
                    report = _scan_in_subprocess(
                        root, entry, store, opportunities, memory_budget_bytes
                    )
                except (ValueError, OSError, OverflowError, MemoryError, sqlite3.Error) as exc:
                    report = dict(
                        error=f"{type(exc).__name__}: {exc}",
                        folder=entry["folder"],
                        status=entry["status"],
                        verdict="inconclusive",
                        reason=[str(exc)],
                        matches=[],
                        searched_pairs=[],
                        skipped_pairs=[],
                    )
                report["resumed"] = False
                atomic_json(
                    saved,
                    dict(
                        fingerprint=key,
                        resume_fingerprint=resume_key,
                        opportunities=opportunities,
                        report=report,
                    ),
                )
            folders.append(report)
        failed_folders = [
            f["folder"]
            for f in folders
            if f.get("error") or (f["status"] == "allowed" and f["verdict"] != "clean")
        ]
        valid = bool(checks["ok"]) and not input_errors
        code = 2 if not valid or failed_folders else 0
        result = dict(
            version="4.0",
            attempt_id=os.environ.get("CORPUS_ATTEMPT_ID"),
            repo_sha=os.environ.get("CORPUS_REPO_SHA"),
            method=METHOD_VERSION,
            default_rule="L",
            valid=valid,
            exit_code=code,
            input_fingerprint=fingerprint,
            benchmark_revision=FEV_DATASETS_REVISION,
            benchmark_series=store.count,
            benchmark_points=store.points,
            benchmark_coverage=store.coverage,
            information_threshold=20 + math.log2(opportunities),
            controls=checks,
            input_errors=input_errors,
            folders=folders,
            failed_folders=failed_folders,
            memory_budget_bytes=memory_budget_bytes,
        )
        atomic_json(out / "audit.json", result)
        return result, code
    finally:
        store.close()


def main(argv=None):
    """Run content audit; any inconclusive allowed folder or failed control exits 2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state", type=Path)
    parser.add_argument(
        "--memory-budget-gib",
        type=memory_budget_arg,
        default=DEFAULT_MEMORY_BUDGET_BYTES,
        help="process RSS budget in GiB (default: 32)",
    )
    args = parser.parse_args(argv)
    report, code = run_audit(
        args.root,
        args.out,
        read_manifest(args.manifest),
        state=args.state,
        memory_budget_bytes=args.memory_budget_gib,
    )
    if report["failed_folders"]:
        print("failed audit folders: " + ", ".join(report["failed_folders"]))
    return code


if __name__ == "__main__":
    if sys.argv[1:] == ["--scan-worker"]:
        _scan_worker()
    else:
        raise SystemExit(main())
