"""Original-order compatibility scan for failed speculative work.

The folder scanner is retained from 32511ee so a failure beyond its evidence
cutoff cannot erase an earlier valid match. Only failed parallel folders use it.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

from demand_forecast.pretrain import audit_match as old
from scripts.corpus_v4 import audit_fev_overlap as audit
from scripts.corpus_v4.audit_fev_overlap import _index_batches, _rank, _segment_id, _variants
from scripts.corpus_v4.matching import chunks, find_index_matches, frequency
from scripts.corpus_v4.native import (
    MemoryBudget,
    folder_frequencies,
    folder_path,
    iter_rows,
    start_memory_watchdog,
)


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


def serial_worker(request):
    """Run the original traversal in a disposable worker with the same RSS quota."""
    budget = request["memory_budget_bytes"]
    start_memory_watchdog(budget)
    audit.MAX_INDEX_POINTS = request["max_index_points"]
    store = audit.BenchmarkStore.__new__(audit.BenchmarkStore)
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
