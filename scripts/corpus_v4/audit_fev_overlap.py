"""Audit native Simons targets and covariates against full, pinned fev histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
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
    MAX_ROW_VALUES,
    atomic_json,
    digest,
    folder_path,
    iter_rows,
)


@dataclass
class BenchmarkSeries:
    """One complete numeric benchmark column; evaluation cutoffs are never applied."""

    dataset: str
    item_id: str
    column: str
    freq: str
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
        for row in dataset:
            alias, stamps = _timestamp_frequency(row[task.timestamp_column], task.freq)
            if frequency(alias) is None:
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

    def __init__(self, path, series):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("DROP TABLE IF EXISTS series")
        self.db.execute("CREATE TABLE series (meta TEXT, vals BLOB, stamps BLOB)")
        h, self.points, self.count = hashlib.sha256(), 0, 0
        self.samples = []
        sampled = set()
        for item in series:
            values = np.asarray(item.values, dtype="<f8")
            if values.ndim != 1 or len(values) > MAX_ROW_VALUES:
                raise ValueError("benchmark series exceeds memory guard or invalid shape")
            meta = dict(
                dataset=item.dataset,
                item_id=item.item_id,
                column=item.column,
                freq=item.freq,
                domain=item.domain,
            )
            encoded = json.dumps(meta, sort_keys=True).encode()
            stamps = (
                pd.DatetimeIndex(item.timestamps).as_unit("ns").asi8.astype("<i8").tobytes()
                if item.timestamps is not None
                else b""
            )
            if stamps and len(stamps) != values.nbytes:
                raise ValueError("benchmark timestamps misaligned")
            payload = values.tobytes()
            h.update(len(encoded).to_bytes(8, "little") + encoded)
            h.update(len(payload).to_bytes(8, "little") + payload)
            h.update(len(stamps).to_bytes(8, "little") + stamps)
            self.db.execute(
                "INSERT INTO series VALUES (?, ?, ?)", (encoded.decode(), payload, stamps)
            )
            self.points += len(values)
            self.count += 1
            if (
                (item.dataset, item.freq) not in sampled
                and len(self.samples) < 128
                and len(values) >= 96
                and old.nondegenerate(values)[0]
            ):
                self.samples.append(values[:512].copy())
                sampled.add((item.dataset, item.freq))
        self.db.commit()
        self.fingerprint = h.hexdigest()
        if not self.count:
            raise ValueError("no benchmark series")

    def __iter__(self):
        for meta, values, stamps in self.db.execute("SELECT meta, vals, stamps FROM series"):
            item = json.loads(meta)
            times = pd.to_datetime(np.frombuffer(stamps, dtype="<i8")) if stamps else None
            yield BenchmarkSeries(
                **item, values=np.frombuffer(values, dtype="<f8"), timestamps=times
            )

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


def _variants(values, fine, coarse, timestamps=None, *, fine_alias=None, coarse_alias=None):
    """Yield all supported aggregation phases, or a precise unresolved-coverage reason."""
    if not identifying_parent(values):
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
                    aggregate_fixed(values, ratio, phase, method),
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
                continue
            for start, part in chunks(values):
                if arrays and size + len(part) > MAX_INDEX_POINTS:
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


def _scan_folder(root, entry, store, opportunities):
    path = folder_path(root, entry)
    matches, coverage, searched, skipped = [], Counter(), {}, {}
    reasons = set()
    kind = entry["folder"].split("/")[0]
    aliases = sorted({r["row"]["freq"] for r in iter_rows(path, kind)})
    benchmark_aliases = sorted(
        {json.loads(row[0])["freq"] for row in store.db.execute("SELECT meta FROM series")}
    )
    for corpus_alias in aliases:
        cf = frequency(corpus_alias)
        for benchmark_alias in benchmark_aliases:
            bf = frequency(benchmark_alias)
            datasets = sorted(
                {
                    json.loads(row[0])["dataset"]
                    for row in store.db.execute("SELECT meta FROM series")
                    if json.loads(row[0])["freq"] == benchmark_alias
                }
            )
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
                index = old.Index(arrays, [f"benchmark-{i}" for i in range(len(arrays))])
                for rec in iter_rows(path, kind):
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
                                coverage.update(counters)
                                for meta in metadata:
                                    key = (meta["dataset"], corpus_alias, benchmark_alias)
                                    searched[key]["comparisons"] += 1
                                    transform = (
                                        aggregation
                                        if direction == "corpus_to_benchmark"
                                        else meta["aggregation"]
                                    )
                                    encoded = json.dumps(transform, sort_keys=True)
                                    if encoded not in searched[key]["aggregations"]:
                                        searched[key]["aggregations"].append(encoded)
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


def run_audit(root, out, manifest, benchmarks=None, *, state=None, control_runner=controls):
    """Restart by folder with hashes, benchmark identity, method and policy bound to state."""
    validate_manifest(manifest)
    root, out = Path(root), Path(out)
    state = Path(state or Path(os.environ.get("RUN_DIR", out)) / "state")
    state.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    source = benchmarks if benchmarks is not None else load_benchmarks(state / "benchmark-cache")
    store = BenchmarkStore(state / "benchmark.sqlite", source)
    try:
        import pyarrow.parquet as pq

        identities, folder_reasons, input_errors = {}, {}, []
        permitted = {folder_path(root, e).relative_to(root).as_posix() for e in manifest["folders"]}
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
            path = folder_path(root, entry)
            files = sorted(path.glob("*.parquet"))
            identities[entry["folder"]] = {}
            try:
                if not files:
                    raise ValueError("missing corpus folder/shards")
                for shard in files:
                    identities[entry["folder"]][shard.name] = digest(shard)
                    parquet = pq.ParquetFile(shard)
                    total_values_bound += parquet.metadata.serialized_size
                    total_values_bound += sum(
                        parquet.metadata.row_group(i).total_byte_size
                        for i in range(parquet.num_row_groups)
                    )
            except (ValueError, OSError) as exc:
                folder_reasons[entry["folder"]] = str(exc)
        opportunities = max(1, total_values_bound) * max(1, store.points) * (2 * MAX_PHASES * 16)
        identity = dict(
            manifest=manifest,
            benchmark=store.fingerprint,
            benchmark_revision=FEV_DATASETS_REVISION,
            method=METHOD_VERSION,
            opportunities=opportunities,
        )
        fingerprint = _fingerprint(dict(**identity, folders=identities))
        checks = control_runner(store.samples, opportunities)
        folders = []
        for entry in manifest["folders"]:
            key = _fingerprint(dict(**identity, entry=entry, hashes=identities[entry["folder"]]))
            saved = state / "audit-folders" / f"{entry['folder'].replace('/', '--')}.json"
            report = None
            if saved.exists():
                previous = json.loads(saved.read_text())
                if previous.get("fingerprint") == key:
                    report = previous["report"]
                    report["resumed"] = True
            if report is None:
                try:
                    if entry["folder"] in folder_reasons:
                        raise ValueError(folder_reasons[entry["folder"]])
                    report = _scan_folder(root, entry, store, opportunities)
                except (ValueError, OSError, OverflowError) as exc:
                    report = dict(
                        folder=entry["folder"],
                        status=entry["status"],
                        verdict="inconclusive",
                        reason=[str(exc)],
                        matches=[],
                        searched_pairs=[],
                        skipped_pairs=[],
                    )
                report["resumed"] = False
                atomic_json(saved, dict(fingerprint=key, report=report))
            folders.append(report)
        valid = bool(checks["ok"]) and not input_errors
        code = (
            2
            if not valid
            or any(f["status"] == "allowed" and f["verdict"] != "clean" for f in folders)
            else 0
        )
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
            information_threshold=20 + math.log2(opportunities),
            controls=checks,
            input_errors=input_errors,
            folders=folders,
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
    args = parser.parse_args(argv)
    _, code = run_audit(args.root, args.out, read_manifest(args.manifest), state=args.state)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
