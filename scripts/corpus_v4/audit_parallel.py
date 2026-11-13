"""Disposable row-group workers and ordered journals for the native overlap audit.

Scheduling/read order may change; reduction order is always the original
frequency, benchmark batch, shard, row group, row, channel, phase, chunk order.
Only positive queries need journal boundaries: a negative run can sum counters.
"""

from __future__ import annotations

import heapq
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from demand_forecast.pretrain import audit_match as old
from scripts.corpus_v4.matching import QueryHashes, find_index_matches
from scripts.corpus_v4.native import (
    MemoryBudget,
    atomic_json,
    digest,
    folder_frequencies,
    folder_path,
    iter_rows,
    parquet_paths,
    start_memory_watchdog,
    value_chunks,
)

EXECUTION_VERSION = "ordered-row-groups-1"
EVIDENCE_LIMIT = 10000


def publish_checkpoints(state, paths):
    """Cloud jobs acknowledge a unit only after its journal then marker are durable.

    The supervisor uploads serially so gcloud memory does not multiply by N.
    Local/offline invocation has no destination and makes no cloud calls.
    """
    uri = os.environ.get("CORPUS_AUDIT_STATE_URI")
    if not uri:
        return
    for path in paths:
        destination = uri.rstrip("/") + "/" + Path(path).relative_to(state).as_posix()
        result = subprocess.run(
            ["gcloud", "storage", "cp", str(path), destination],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode:
            raise ValueError(
                f"audit checkpoint upload failed ({result.returncode}): {result.stderr.strip()}"
            )


def pair_plans(store, aliases):
    """Retain every pair and the old unsupported-frequency reporting rules."""
    from scripts.corpus_v4 import audit_fev_overlap as a

    groups = {}
    for (encoded,) in store.db.execute("SELECT meta FROM series"):
        meta = json.loads(encoded)
        groups.setdefault(meta["freq"], set()).add(meta["dataset"])
    for ca in aliases:
        cf = a.frequency(ca)
        for ba in sorted(groups, key=lambda alias: alias or ""):
            bf = a.frequency(ba)
            failure = None
            if cf is None or bf is None:
                failure = "unsupported frequency"
            elif cf == bf:
                rule, direction = "F", "same_frequency"
            elif a._rank(cf) is None or a._rank(bf) is None:
                failure = "unsupported cross-frequency ordering"
            else:
                rule = "L"
                direction = (
                    "corpus_to_benchmark" if a._rank(cf) < a._rank(bf) else "benchmark_to_corpus"
                )
            pairs = [
                dict(
                    benchmark=name,
                    benchmark_frequency=ba,
                    corpus_frequency=ca,
                    domains="all pairs (conservative)",
                )
                for name in sorted(groups[ba])
            ]
            if failure is None:
                for pair in pairs:
                    pair.update(rule=rule, direction=direction)
            yield dict(
                ca=ca,
                ba=ba,
                pairs=pairs,
                failure=failure,
                direction=None if failure else direction,
                rule=None if failure else rule,
            )


class DecodedRows:
    """One Arrow decode; later passes map original numeric dtypes from scratch.

    Metadata stays in SQLite, including original shard/row positions. Scratch is
    outside state: it is disposable and never copied to the preemption bucket.
    """

    def __init__(self, directory, root, entry, unit, budget, progress, heartbeat):
        self.db = sqlite3.connect(Path(directory) / "rows.sqlite")
        self.db.execute("PRAGMA cache_size=-1024")
        self.db.execute("CREATE TABLE rows (freq TEXT, meta TEXT)")
        self.path = Path(directory) / "rows.values"
        offset = 0
        try:
            with self.path.open("wb") as stream:
                for rec in iter_rows(
                    folder_path(root, entry),
                    entry["folder"].split("/")[0],
                    memory_budget_bytes=budget,
                    shard=unit["shard"],
                    row_group=unit["group"],
                ):
                    meta = {k: rec[k] for k in ("shard", "row_index")}
                    meta["row"] = {k: rec["row"][k] for k in ("item_id", "source", "freq")}
                    for name in ("target", "covariates"):
                        matrix = rec[name]
                        meta[name] = dict(offset=offset, shape=matrix.shape, dtype=matrix.dtype.str)
                        for part in value_chunks(matrix):
                            stream.write(part.tobytes())
                        offset += matrix.nbytes
                        progress["native_values_decoded"] += int(matrix.size)
                    self.db.execute(
                        "INSERT INTO rows VALUES (?, ?)", (rec["row"]["freq"], json.dumps(meta))
                    )
            self.db.commit()
        except BaseException:
            self.db.close()
            raise

    def rows(self, alias):
        for (encoded,) in self.db.execute("SELECT meta FROM rows WHERE freq=?", (alias,)):
            meta = json.loads(encoded)
            for name in ("target", "covariates"):
                spec = meta[name]
                meta[name] = (
                    np.memmap(
                        self.path,
                        mode="r",
                        dtype=spec["dtype"],
                        offset=spec["offset"],
                        shape=tuple(spec["shape"]),
                    )
                    if all(spec["shape"])
                    else np.empty(spec["shape"])
                )
            yield meta

    def close(self):
        self.db.close()


class BatchJournal:
    """Compact deltas, flushed at each positive query so the evidence cap is exact."""

    def __init__(self, path, number, arrays, metadata, events, plan, first, budget):
        self.number, self.metadata, self.plan = number, metadata, plan
        self.stream = Path(path).open("w")
        self.index = (
            old.Index(arrays, [f"benchmark-{i}" for i in range(len(arrays))]) if arrays else None
        )
        MemoryBudget(budget).check(context="audit index")
        self.stride_indices = {}
        self.batch_counts, self.transforms = Counter(), {}
        for meta in metadata:
            name = meta["dataset"]
            self.batch_counts[name] += 1
            self.transforms.setdefault(name, set()).add(
                json.dumps(meta["aggregation"], sort_keys=True)
            )
        self.seen, self.count, self.done = {}, 0, False
        self.reset()
        if first:
            for event in events:
                self.event(event)

    def reset(self):
        self.delta = dict(
            coverage=Counter(),
            comparisons=Counter(),
            aggregations={},
            reasons=[],
            skipped=None,
            matches=[],
        )

    def event(self, aggregation):
        if "unresolved" in aggregation:
            reason = aggregation["unresolved"]
            if reason not in self.delta["reasons"]:
                self.delta["reasons"].append(reason)
            self.delta["skipped"] = reason
        else:
            self.delta["coverage"]["uninformative_aggregation_parents"] += 1

    def query(self, xx, bx, aggregation, rec, column, channel, opportunities, budget, hashes):
        from scripts.corpus_v4 import audit_fev_overlap as a

        records, counters = find_index_matches(
            xx,
            self.index,
            opportunities=opportunities,
            calendar_dimensions=[m["calendar_sizes"] for m in self.metadata],
            query_hashes=hashes,
            stride_indices=self.stride_indices,
        )
        MemoryBudget(budget).check(context="audit query")
        self.delta["coverage"].update(counters)
        for name, count in self.batch_counts.items():
            self.delta["comparisons"][name] += count
            transforms = (
                {json.dumps(aggregation, sort_keys=True)}
                if self.plan["direction"] == "corpus_to_benchmark"
                else self.transforms[name]
            )
            unseen = sorted(transforms - self.seen.setdefault(name, set()))
            self.delta["aggregations"].setdefault(name, []).extend(unseen)
            self.seen[name].update(unseen)
        for record in records:
            meta = self.metadata[record["index_series_index"]]
            row = rec["row"]
            base, offset = a._segment_id(row["item_id"])
            record.update(
                rule=self.plan["rule"],
                aggregation=(
                    aggregation
                    if self.plan["direction"] == "corpus_to_benchmark"
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
                    k: v for k, v in meta.items() if k not in {"aggregation", "calendar_sizes"}
                },
            )
            self.delta["matches"].append(record)
            self.count += 1
            if self.count >= EVIDENCE_LIMIT:
                self.done = True
                break
        if self.delta["matches"]:
            self.flush()

    def flush(self):
        self.stream.write(json.dumps(self.delta, allow_nan=False) + "\n")
        self.reset()

    def close(self):
        self.flush()
        self.stream.close()


def _probe_group(group, decoded, plan, request, progress, heartbeat, budget):
    # Scope query views, hash caches and active indices to one pass so they are
    # released before the next resident group is constructed.
    from scripts.corpus_v4 import audit_fev_overlap as a

    for rec in decoded.rows(plan["ca"]):
        channels = [("target", i, v) for i, v in enumerate(rec["target"])]
        channels += [("covariates", i, v) for i, v in enumerate(rec["covariates"])]
        for column, channel, values in channels:
            variants = (
                a._variants(
                    values,
                    a.frequency(plan["ca"]),
                    a.frequency(plan["ba"]),
                    fine_alias=plan["ca"],
                    coarse_alias=plan["ba"],
                )
                if plan["direction"] == "corpus_to_benchmark"
                else [(values, dict(method="identity"))]
            )
            for query, aggregation in variants:
                active = [b for b, _ in group if b.index is not None and not b.done]
                if not active:
                    break
                if query is None:
                    for b in active:
                        b.event(aggregation)
                    continue
                for bx, xx in a.chunks(query):
                    hashes = QueryHashes(xx)
                    xx = hashes.values
                    for b in active:
                        if not b.done:
                            b.query(
                                xx,
                                bx,
                                aggregation,
                                rec,
                                column,
                                channel,
                                request["opportunities"],
                                budget,
                                hashes,
                            )
                            progress["values_scanned"] += len(xx)
                    heartbeat()
        if all(b.done or b.index is None for b, _ in group):
            break


def scan_unit(request, journal_path, progress_path):
    """Co-probe bounded groups of unchanged indices; write one sorted unit journal."""
    from scripts.corpus_v4 import audit_fev_overlap as a

    budget = request["memory_budget_bytes"]
    a.MAX_INDEX_POINTS = request["max_index_points"]
    progress = dict(values_scanned=0, native_values_decoded=0)
    last_update = 0

    def heartbeat(force=False):
        nonlocal last_update
        now = time.monotonic()
        if force or now - last_update >= 1:
            atomic_json(progress_path, progress)
            last_update = now

    store = a.BenchmarkStore.__new__(a.BenchmarkStore)
    store.path = Path(request["store"])
    store.values_path = store.path.with_suffix(".values")
    store.stamps_path = store.path.with_suffix(".stamps")
    store.db = sqlite3.connect(store.path)
    store.db.execute("PRAGMA cache_size=-1024")
    scratch = Path(request["scratch"])
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=scratch) as temporary, ExitStack() as resources:
            decoded = None
            # Reserve at most 1/4 of a worker budget for resident index planning.
            # 512 bytes/point covers the four posting families plus construction
            # temporaries; actual RSS is still checked and watched, never assumed.
            group_size = max(1, min(4, budget // (4 * 512 * a.MAX_INDEX_POINTS)))
            local_records = 0
            with Path(journal_path).open("w") as output:
                for pi, plan in enumerate(pair_plans(store, request["aliases"])):
                    if plan["failure"]:
                        continue  # The reducer emits the unchanged skipped-pair reason.
                    batches = iter(
                        a._index_batches(store, plan["ba"], plan["ca"], plan["direction"])
                    )
                    batch_number = 0
                    while True:
                        group = []
                        for _ in range(group_size):
                            batch = next(batches, None)
                            if batch is None:
                                break
                            path = Path(temporary) / f"batch-{batch_number}.jsonl"
                            item = BatchJournal(
                                path,
                                batch_number,
                                *batch,
                                plan,
                                request["unit"]["order"] == 0,
                                budget,
                            )
                            group.append((item, path))
                            batch_number += 1
                        if not group:
                            break
                        del item, batch
                        try:
                            if any(b.index is not None for b, _ in group):
                                if decoded is None:
                                    decoded = DecodedRows(
                                        temporary,
                                        request["root"],
                                        request["entry"],
                                        request["unit"],
                                        budget,
                                        progress,
                                        heartbeat,
                                    )
                                    resources.callback(decoded.close)
                                    heartbeat(True)
                                _probe_group(
                                    group, decoded, plan, request, progress, heartbeat, budget
                                )
                        finally:
                            for b, _ in group:
                                b.close()
                        for b, path in group:
                            with path.open() as stream:
                                for line in stream:
                                    delta = json.loads(line)
                                    delta["matches"] = delta["matches"][
                                        : EVIDENCE_LIMIT - local_records
                                    ]
                                    local_records += len(delta["matches"])
                                    output.write(json.dumps([pi, b.number, delta]) + "\n")
                                    # A unit's traversal is a subsequence of the
                                    # global traversal. After 10,000 local matches
                                    # the global cutoff must already have occurred.
                                    if local_records >= EVIDENCE_LIMIT:
                                        heartbeat(True)
                                        return progress
                            path.unlink()
                        del group, b
                heartbeat(True)
        return progress
    finally:
        store.close()


def unit_worker(request):
    """A checkpoint is published by its worker, before acknowledging completion."""
    start_memory_watchdog(request["memory_budget_bytes"])
    checkpoint = Path(request["checkpoint"])
    journal = checkpoint.with_suffix(".jsonl")
    temporary = journal.with_suffix(".jsonl.tmp")
    started = time.monotonic()
    metrics = scan_unit(request, temporary, checkpoint.with_suffix(".progress"))
    temporary.replace(journal)
    atomic_json(
        checkpoint,
        dict(
            fingerprint=request["fingerprint"],
            memory_budget_bytes=request["memory_budget_bytes"],
            journal_sha256=digest(journal),
            elapsed_seconds=time.monotonic() - started,
            **metrics,
        ),
    )


def _events(path, order):
    """Prime merge heads with keys/offsets only, never a positive payload.

    Files are closed before yielding, so neither memory nor open descriptors
    multiply by the number of units. Advance past a consumed line in bounded
    buffers; unselected payloads are not read at all.
    """
    path, offset, seq = Path(path), 0, 0
    while True:
        with path.open("rb") as stream:
            stream.seek(offset)
            prefix = stream.read(256)
        if not prefix:
            return
        header = re.match(rb"\[\s*(\d+)\s*,\s*(\d+)\s*,", prefix)
        if header is None:
            raise ValueError(f"invalid unit journal header: {path}:{offset}")
        pi, batch = (int(x) for x in header.groups())
        yield (pi, batch, order, seq), (path, offset)
        with path.open("rb") as stream:
            stream.seek(offset)
            while True:
                part = stream.readline(65536)
                if not part or part.endswith(b"\n"):
                    break
            offset = stream.tell()
        seq += 1


def reduce_folder(entry, plans, unit_paths):
    """Replay deltas in legacy traversal order, including the exact cap query."""
    matches, coverage, searched, skipped, seen = [], Counter(), {}, {}, {}
    reasons = set()
    events = heapq.merge(
        *(_events(path, order) for order, path in enumerate(unit_paths)), key=lambda x: x[0]
    )
    current_pair = -1
    for (pi, _batch, _unit, _seq), (path, offset) in events:
        with path.open("rb") as stream:
            stream.seek(offset)
            delta = json.loads(stream.readline())[2]
        while current_pair < pi:
            current_pair += 1
            plan = plans[current_pair]
            for pair in plan["pairs"]:
                if plan["failure"]:
                    skipped[json.dumps(pair, sort_keys=True)] = dict(**pair, reason=plan["failure"])
                    reasons.add(plan["failure"])
                else:
                    key = (pair["benchmark"], plan["ca"], plan["ba"])
                    searched[key] = dict(**pair, comparisons=0, matched=0, aggregations=[])
                    seen[key] = set()
        plan = plans[pi]
        reasons.update(delta["reasons"])
        if delta["skipped"]:
            for pair in plan["pairs"]:
                skipped[json.dumps(pair, sort_keys=True)] = dict(**pair, reason=delta["skipped"])
        coverage.update(delta["coverage"])
        for name, count in delta["comparisons"].items():
            key = (name, plan["ca"], plan["ba"])
            searched[key]["comparisons"] += count
            for transform in delta["aggregations"].get(name, []):
                if transform not in seen[key]:
                    searched[key]["aggregations"].append(transform)
                    seen[key].add(transform)
        for record in delta["matches"]:
            key = (record["benchmark"]["dataset"], plan["ca"], plan["ba"])
            searched[key]["matched"] += 1
            matches.append(record)
            if len(matches) >= EVIDENCE_LIMIT:
                reasons.add("evidence record limit reached")
                break
        if len(matches) >= EVIDENCE_LIMIT:
            break
    else:
        # Empty benchmark batches still initialize pairs; unsupported pairs after
        # the last emitted event retain their original position and reasons.
        for plan in plans[current_pair + 1 :]:
            for pair in plan["pairs"]:
                if plan["failure"]:
                    skipped[json.dumps(pair, sort_keys=True)] = dict(**pair, reason=plan["failure"])
                    reasons.add(plan["failure"])
                else:
                    searched[(pair["benchmark"], plan["ca"], plan["ba"])] = dict(
                        **pair, comparisons=0, matched=0, aggregations=[]
                    )
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


def error_report(entry, exc):
    return dict(
        error=f"{type(exc).__name__}: {exc}",
        folder=entry["folder"],
        status=entry["status"],
        verdict="inconclusive",
        reason=[str(exc)],
        matches=[],
        searched_pairs=[],
        skipped_pairs=[],
    )


def _serial_scan(root, entry, store, opportunities, budget):
    """Recheck failed speculative work against the original traversal/cutoff."""
    from scripts.corpus_v4 import audit_fev_overlap as a

    request = dict(
        serial_fallback=True,
        root=str(root),
        entry=entry,
        store=str(store.path),
        opportunities=opportunities,
        memory_budget_bytes=budget,
        max_index_points=a.MAX_INDEX_POINTS,
    )
    result = subprocess.run(
        [sys.executable, "-m", "scripts.corpus_v4.audit_fev_overlap", "--scan-worker"],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
    )
    if result.returncode:
        raise ValueError(f"audit worker exit {result.returncode}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def scan_folders(
    root, state, manifest, store, identity, identities, folder_reasons, memory_budget_bytes, workers
):
    """Resume completed folders/units; supervise at most N fresh RSS-limited children."""
    from scripts.corpus_v4 import audit_fev_overlap as a

    budget = memory_budget_bytes // workers
    if budget < 1:
        raise ValueError("memory budget divided by workers is less than one byte")
    reports, folders, requests, metrics = {}, {}, [], {}
    for entry in manifest["folders"]:
        name = entry["folder"]
        key = a._fingerprint(
            dict(
                **identity,
                entry=entry,
                hashes=identities[name],
                memory_budget_bytes=memory_budget_bytes,
            )
        )
        stable = {k: v for k, v in identity.items() if k != "opportunities"}
        resume_key = a._fingerprint(
            dict(
                **stable,
                entry=entry,
                hashes=identities[name],
                memory_budget_bytes=memory_budget_bytes,
            )
        )
        saved = state / "audit-folders" / f"{name.replace('/', '--')}.json"
        if saved.exists():
            previous = json.loads(saved.read_text())
            compatible = (
                previous.get("resume_fingerprint") == resume_key
                and previous.get("opportunities", 0) <= identity["opportunities"]
                and previous["report"]["verdict"] == "clean"
            )
            if (previous.get("fingerprint") == key or compatible) and not previous["report"].get(
                "error"
            ):
                reports[name] = dict(previous["report"], resumed=True)
        info = folders[name] = dict(
            entry=entry,
            key=key,
            resume_key=resume_key,
            saved=saved,
            paths=[],
            failed=None,
            plans=[],
            scan_failed=False,
            hard_failure=False,
        )
        try:
            if name in folder_reasons:
                raise ValueError(folder_reasons[name])
            path = folder_path(root, entry)
            aliases = folder_frequencies(path)
            info["plans"] = list(pair_plans(store, aliases))
            import pyarrow.parquet as pq

            for shard in parquet_paths(path):
                with pq.ParquetFile(shard) as parquet:
                    for group in range(parquet.num_row_groups):
                        order = len(info["paths"])
                        unit = dict(shard=shard.name, group=group, order=order)
                        cp = state / "audit-units" / name.replace("/", "--") / f"{order:06}.json"
                        info["paths"].append(cp.with_suffix(".jsonl"))
                        fingerprint = a._fingerprint(
                            dict(folder=key, unit=unit, execution=EXECUTION_VERSION)
                        )
                        req = dict(
                            root=str(root),
                            entry=entry,
                            unit=unit,
                            store=str(store.path),
                            opportunities=identity["opportunities"],
                            memory_budget_bytes=budget,
                            aliases=aliases,
                            max_index_points=a.MAX_INDEX_POINTS,
                            scratch=str(
                                state.parent
                                / "audit-scratch"
                                / name.replace("/", "--")
                                / f"{order:06}"
                            ),
                            checkpoint=str(cp),
                            fingerprint=fingerprint,
                        )
                        previous = None
                        if cp.exists():
                            try:
                                candidate = json.loads(cp.read_text())
                                if candidate["fingerprint"] == fingerprint and candidate[
                                    "journal_sha256"
                                ] == digest(cp.with_suffix(".jsonl")):
                                    previous = candidate
                            except (OSError, ValueError, KeyError):
                                pass
                        uid = str(cp)
                        if name in reports:
                            metrics[uid] = dict(
                                previous or {},
                                status="done",
                                folder=name,
                                values_scanned=(previous or {}).get("values_scanned", 0),
                                native_values_decoded=(previous or {}).get(
                                    "native_values_decoded", 0
                                ),
                            )
                        elif previous is not None:
                            publish_checkpoints(state, [cp.with_suffix(".jsonl"), cp])
                            metrics[uid] = dict(previous, status="done", folder=name)
                        else:
                            cp.with_suffix(".progress").unlink(missing_ok=True)
                            metrics[uid] = dict(
                                status="pending",
                                folder=name,
                                values_scanned=0,
                                native_values_decoded=0,
                            )
                            group_meta = parquet.metadata.row_group(group)
                            req["work_weight"] = 256 * group_meta.num_rows + sum(
                                group_meta.column(i).num_values
                                for i in range(group_meta.num_columns)
                                if group_meta.column(i).path_in_schema.split(".")[0]
                                in {"target", "covariates"}
                            )
                            requests.append(req)
        except (ValueError, OSError) as exc:
            info["failed"] = exc
            reports.pop(name, None)
            # Missing/corrupt folders are themselves retryable failed units.
            uid = f"error:{name}"
            metrics[uid] = dict(
                status="failed", folder=name, values_scanned=0, native_values_decoded=0
            )
    started = time.monotonic()
    active = {}
    queue = iter(sorted(requests, key=lambda request: -request["work_weight"]))
    last_progress = 0

    def progress(force=False):
        nonlocal last_progress
        now = time.monotonic()
        if not force and now - last_progress < 1:
            return
        for req, _proc, _stderr in active.values():
            heartbeat = Path(req["checkpoint"]).with_suffix(".progress")
            if heartbeat.exists():
                try:
                    metrics[req["checkpoint"]].update(json.loads(heartbeat.read_text()))
                except (OSError, ValueError):
                    pass
        counts = Counter(x["status"] for x in metrics.values())
        done, failed, total = counts["done"], counts["failed"], len(metrics)
        elapsed = now - started
        per_folder = {}
        for name in folders:
            statuses = Counter(v["status"] for v in metrics.values() if v["folder"] == name)
            per_folder[name] = dict(
                units_total=sum(statuses.values()),
                units_done=statuses["done"],
                units_failed=statuses["failed"],
                serial_fallback=bool(folders[name].get("serial_fallback")),
                status=(
                    "fallback"
                    if folders[name].get("fallback_running")
                    else "failed"
                    if statuses["failed"]
                    else "done"
                    if statuses["done"] == sum(statuses.values())
                    else "running"
                    if statuses["running"]
                    else "pending"
                ),
            )
        finished = done + failed
        atomic_json(
            state / "audit-progress.json",
            dict(
                version=1,
                units_total=total,
                units_done=done,
                units_failed=failed,
                values_scanned=sum(v["values_scanned"] for v in metrics.values()),
                native_values_decoded=sum(v["native_values_decoded"] for v in metrics.values()),
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=(
                    0
                    if finished == total
                    else elapsed * (total - finished) / finished
                    if finished
                    else None
                ),
                folders=per_folder,
            ),
        )
        last_progress = now

    def interrupted(_signum, _frame):
        raise InterruptedError("audit worker pool interrupted")

    handlers = {}
    import threading

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, interrupted)
    try:
        exhausted = False
        progress(True)
        while active or not exhausted:
            while not exhausted and len(active) < workers:
                req = next(queue, None)
                if req is None:
                    exhausted = True
                    break
                cp = Path(req["checkpoint"])
                cp.parent.mkdir(parents=True, exist_ok=True)
                shutil.rmtree(req["scratch"], ignore_errors=True)
                stderr = cp.with_suffix(".stderr").open("w+")
                env = dict(
                    os.environ,
                    OPENBLAS_NUM_THREADS="1",
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    NUMEXPR_NUM_THREADS="1",
                )
                proc = subprocess.Popen(
                    [sys.executable, "-m", "scripts.corpus_v4.audit_fev_overlap", "--scan-worker"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                    text=True,
                    env=env,
                )
                active[proc.pid] = (req, proc, stderr)
                proc.stdin.write(json.dumps(req))
                proc.stdin.close()
                metrics[str(cp)]["status"] = "running"
            for pid, (req, proc, stderr) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                del active[pid]
                cp = Path(req["checkpoint"])
                if code:
                    stderr.seek(0)
                    reason = stderr.read().strip()
                    exc = ValueError(f"audit worker exit {code}: {reason}")
                    folders[req["entry"]["folder"]]["failed"] = exc
                    folders[req["entry"]["folder"]]["scan_failed"] = True
                    metrics[str(cp)]["status"] = "failed"
                else:
                    try:
                        publish_checkpoints(state, [cp.with_suffix(".jsonl"), cp])
                        metrics[str(cp)].update(json.loads(cp.read_text()), status="done")
                    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                        folders[req["entry"]["folder"]]["failed"] = exc
                        folders[req["entry"]["folder"]]["hard_failure"] = True
                        metrics[str(cp)]["status"] = "failed"
                stderr.close()
                shutil.rmtree(req["scratch"], ignore_errors=True)
            progress()
            if active:
                time.sleep(0.05)
        for name, info in folders.items():
            if name in reports:
                continue
            if info["scan_failed"] and not info["hard_failure"]:
                info["serial_fallback"] = info["fallback_running"] = True
                progress(True)
                try:
                    report = _serial_scan(
                        root, info["entry"], store, identity["opportunities"], budget
                    )
                    for metric in metrics.values():
                        if metric["folder"] == name:
                            metric["status"] = "done"
                except InterruptedError:
                    raise
                except (ValueError, OSError, MemoryError, OverflowError) as exc:
                    report = error_report(info["entry"], exc)
                finally:
                    info["fallback_running"] = False
            else:
                report = (
                    error_report(info["entry"], info["failed"])
                    if info["failed"] is not None
                    else reduce_folder(info["entry"], info["plans"], info["paths"])
                )
            report["resumed"] = False
            atomic_json(
                info["saved"],
                dict(
                    fingerprint=info["key"],
                    resume_fingerprint=info["resume_key"],
                    opportunities=identity["opportunities"],
                    report=report,
                ),
            )
            reports[name] = report
            publish_checkpoints(state, [info["saved"]])
            progress(True)
        return [reports[e["folder"]] for e in manifest["folders"]]

    finally:
        for req, proc, stderr in active.values():
            proc.terminate()
        for req, proc, stderr in active.values():
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            stderr.close()
            shutil.rmtree(req["scratch"], ignore_errors=True)
        progress(True)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
