"""Provisional endpoint fingerprints: triage evidence, NEVER a release audit.

Only endpoints are indexed. Negative retrieval says nothing about interior crops,
changed masks, rounding boundaries or any resampling/aggregation ancestry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from demand_forecast.bench.specs import FEV_DATASETS_REVISION, fev_bench_tasks
from demand_forecast.pretrain import audit_info
from demand_forecast.pretrain import audit_match as old
from scripts.corpus_v4.audit_fev_overlap import BenchmarkStore, load_benchmarks
from scripts.corpus_v4.freeze_manifest import DEFAULT_MANIFEST, read_manifest, validate_manifest
from scripts.corpus_v4.matching import MAX_PHASES, _repetitive, frequency
from scripts.corpus_v4.native import (
    DEFAULT_MEMORY_BUDGET_BYTES,
    MemoryBudget,
    atomic_json,
    digest,
    folder_path,
    iter_rows,
    memory_budget_arg,
    start_memory_watchdog,
)
from scripts.corpus_v4.verify_published import verify_published

METHOD = "provisional-endpoints-1"
ARTIFACT = "corpus-sketch-triage"
LENGTHS = (24, 64, 256)
POSTING_CAP = 64
TOP = 20
MATCH_SAMPLE = 100
FEATURES = 11
POSTING = np.dtype([("hi", "<u8"), ("lo", "<u8"), ("owner", "<i8"), ("pos", "<i8")])
LIMITATIONS = [
    "Provisional triage only; a negative result is NOT corpus clearance.",
    "Interior crops and copies not preserving a sampled endpoint are not searched.",
    "Changed finite masks and quantisation-boundary cases can escape retrieval.",
    "No fixed-ratio or calendar aggregation is computed; every phase is unsupported.",
    "Common keys are capped and recorded; summary ranking is a bounded heuristic.",
    "Constants, ramps and calendar-only patterns are non-identifying by existing policy.",
]


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _saved_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _token(label, payload):
    return hashlib.blake2b(label.encode() + b"\0" + payload, digest_size=16).hexdigest()


@dataclass(frozen=True)
class SketchRecord:
    """Transient complete-channel statistics plus bounded endpoint keys."""

    length: int
    finite_digest: str
    raw_fingerprint: str
    quantiles: tuple
    zero_fraction: float
    change_fraction: float
    longest_run_fraction: float
    autocorrelation: tuple
    features: tuple
    keys: tuple


@dataclass(frozen=True)
class Candidate:
    """An alignment hypothesis, never a numerical match."""

    offset: int
    reason: str
    distance: float = 0.0


def sketch(channel):
    """One O(n) value pass plus bounded endpoint work; masks retain positions.

    Full raw hashes use canonical little-endian float64 and a positional mask.
    Endpoints use float32 hashes and sign-canonical local z-scores quantised at
    .05, on two shifted grids. These grids improve recall, not guarantee it.
    """
    x = np.asarray(channel, np.float64)
    if x.ndim != 1:
        raise ValueError("sketch requires one complete numeric channel")
    finite = np.isfinite(x)
    mask = np.packbits(finite).tobytes()
    clean = np.where(finite, x, 0).astype("<f8")
    clean[clean == 0] = 0  # canonicalise negative zero
    raw = _token("raw", len(x).to_bytes(8, "little") + mask + clean.tobytes())
    keys = [(raw, 0, "raw")]
    for side in ("prefix", "suffix"):
        for size in LENGTHS:
            if len(x) < size:
                continue
            pos = 0 if side == "prefix" else len(x) - size
            part = clean[pos : pos + size]
            good = finite[pos : pos + size]
            packed = np.packbits(good).tobytes()
            label = f"{side}/{size}"
            keys.append((_token(label + "/f32", packed + part.astype("<f4").tobytes()), pos, label))
            v = part[good]
            if len(v) < old.SEG_MIN or not v.std() > 0:
                continue
            z = (v - v.mean()) / v.std()
            first = np.flatnonzero(np.abs(z) > 1e-8)
            if not len(first):
                continue
            z *= 1 if z[first[0]] > 0 else -1
            for shift in (0.0, 0.5):
                q = np.floor(z / 0.05 + shift).astype("<i4")
                keys.append((_token(label + f"/affine/{shift}", packed + q.tobytes()), pos, label))
    v = clean[finite]
    quantiles = (
        tuple(float(q) for q in np.quantile(v, (0, 0.25, 0.5, 0.75, 1))) if len(v) else (0.0,) * 5
    )
    zero = float(np.mean(v == 0)) if len(v) else 0.0
    # Changes/runs include missing positions; no compaction of the mask.
    changes = (finite[1:] != finite[:-1]) | (clean[1:] != clean[:-1])
    ends = np.r_[0, np.flatnonzero(changes) + 1, len(x)]
    longest = float(np.diff(ends).max() / len(x)) if len(x) else 0.0
    change = float(changes.mean()) if len(changes) else 0.0
    acf = []
    for lag in (1, 7):
        good = finite[lag:] & finite[:-lag]
        a, b = clean[lag:][good], clean[:-lag][good]
        if len(a) > 2 and a.std() > 0 and b.std() > 0:
            acf.append(float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1)))
        else:
            acf.append(0.0)
    span = quantiles[-1] - quantiles[0]
    shape = tuple((q - quantiles[0]) / span if span > 0 else 0.0 for q in quantiles[1:4])
    # Symmetric shape supports negative affine scale for ranking as well.
    shape = min(shape, tuple(1 - q for q in reversed(shape)))
    features = (
        *shape,
        zero,
        change,
        longest,
        *acf,
        float(finite.mean()) if len(x) else 0.0,
        math.log2(max(1, len(x))) / 32,
        float(span > 0),
    )
    bucket = np.floor(np.asarray((*shape, abs(acf[0]))) * 4).astype("<i2").tobytes()
    keys.append((_token("summary", bucket), 0, "summary"))
    return SketchRecord(
        len(x),
        _token("mask", len(x).to_bytes(8, "little") + mask),
        raw,
        quantiles,
        zero,
        change,
        longest,
        tuple(acf),
        tuple(features),
        tuple(keys),
    )


def compare_sketches(query, benchmark):
    """Endpoint/raw retrieval only; deliberately does not search interior offsets."""
    by_key = {token: pos for token, pos, label in benchmark.keys if label != "summary"}
    candidates = {}
    for token, pos, label in query.keys:
        if label != "summary" and token in by_key:
            offset = pos - by_key[token]
            candidates.setdefault(offset, Candidate(offset, label))
    return [candidates[k] for k in sorted(candidates)]


def verify_candidate(x, y, candidate, *, opportunities, dimensions=None):
    """Reuse the original fit and every subsequent numerical identification gate.

    No Index/scan_group retrieval is reused. Original values and original masks
    are refitted over the endpoint alignment's entire overlap.
    """
    off = candidate.offset
    rec = old.fit_segment(x, y, max(0, off), min(len(x), len(y) + off), off)
    reason = "numerical_fit_rejected"
    if rec is not None:
        rec.pop("_info_values", None)
        if rec["is_copy"]:
            a = x[rec["t_start"] : rec["t_end"]]
            b = y[rec["index_start"] : rec["index_end"]]
            finite = np.isfinite(a) & np.isfinite(b)
            a, b = a[finite], b[finite]
            if dimensions is not None and old.calendar_pattern(
                b, dimensions[rec["index_start"] : rec["index_end"]][finite]
            ):
                reason = "calendar_pattern"
            elif _repetitive(a) or _repetitive(b):
                reason = "repetitive_pattern"
            elif rec["info_strict_ramp"]:
                reason = "interpolation_ramp"
            else:
                bits = min(rec["info_bits_target"], rec["info_bits_index"])
                if (
                    bits < audit_info.INFO_MARGIN_BITS
                    or (1 << (bits - audit_info.INFO_MARGIN_BITS)) < opportunities
                ):
                    reason = "low_information"
                else:
                    reason = "verified"
        elif rec["degenerate_reason"]:
            reason = "nonidentifying:" + str(rec["degenerate_reason"])
        # Rejected constant fits carry infinity, which cannot enter JSON.
        rec = {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in rec.items()
        }
    return dict(
        verified=reason == "verified",
        reason=reason,
        proposal=candidate.reason,
        offset=off,
        distance=candidate.distance,
        fit=rec,
        information_threshold=audit_info.INFO_MARGIN_BITS + math.log2(max(1, opportunities)),
    )


def _spool_files(path):
    return [path, path.with_suffix(".values"), path.with_suffix(".stamps")]


def prepare_benchmarks(
    state, source=None, *, memory_budget_bytes=DEFAULT_MEMORY_BUDGET_BYTES, require_pinned=False
):
    """Legacy spools without revision-bound byte identities are rebuilt, never guessed.

    A marker is published only after all three spool files close. Reuse rehashes
    those files, and binds the revision, task YAML and loader implementation.
    """
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    path = state / "benchmark.sqlite"
    marker = state / "benchmark.sketch-identity.json"
    contract = dict(
        revision=FEV_DATASETS_REVISION,
        tasks=[vars(t) for t in fev_bench_tasks("fev100")],
        loader=digest(Path(__file__).with_name("audit_fev_overlap.py")),
    )
    saved = _saved_json(marker)
    injected = source is not None
    if require_pinned and injected:
        raise ValueError("production sketches require the pinned benchmark loader")
    if (
        source is None
        and saved.get("contract") == contract
        and (not require_pinned or saved.get("origin") == "pinned_loader")
        and (
            not require_pinned
            or saved.get("datasets") == sorted(t.name for t in fev_bench_tasks("fev100"))
        )
    ):
        files = _spool_files(path)
        if all(p.is_file() and saved.get("files", {}).get(p.name) == digest(p) for p in files):
            return path, saved
    source = (
        source
        if source is not None
        else load_benchmarks(state / "benchmark-cache" / FEV_DATASETS_REVISION)
    )
    marker.unlink(missing_ok=True)
    # Recreate SQLite from scratch: DROP/CREATE changes SQLite's header counter
    # even for identical logical rows, which would otherwise alter resume keys.
    staging = state / "sketch-benchmark-build"
    staging.mkdir(exist_ok=True)
    temporary = staging / path.name
    for file in _spool_files(temporary):
        file.unlink(missing_ok=True)
    store = BenchmarkStore(temporary, source, memory_budget_bytes=memory_budget_bytes)
    identity = dict(
        contract=contract,
        points=store.points,
        count=store.count,
        fingerprint=store.fingerprint,
        origin="synthetic_injection" if injected else "pinned_loader",
    )
    names = {
        json.loads(encoded)["dataset"] for (encoded,) in store.db.execute("SELECT meta FROM series")
    }
    identity["datasets"] = sorted(names)
    if require_pinned:
        if names != {t.name for t in fev_bench_tasks("fev100")}:
            store.close()
            raise ValueError("pinned benchmark spool does not contain every fev100 dataset")
    store.close()
    for file, destination in zip(_spool_files(temporary), _spool_files(path)):
        file.replace(destination)
    identity["files"] = {p.name: digest(p) for p in _spool_files(path)}
    atomic_json(marker, identity)
    return path, identity


def _open_spool(path):
    db = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    db.execute("PRAGMA cache_size=-8192")
    return db


def _series(db, path, owner):
    encoded, start, length, timed = db.execute(
        "SELECT meta, start, length, timed FROM series WHERE rowid=?", (int(owner),)
    ).fetchone()
    meta = json.loads(encoded)
    values = (
        np.memmap(
            Path(path).with_suffix(".values"), dtype="<f8", mode="r", offset=start, shape=(length,)
        )
        if length
        else np.empty(0)
    )
    dimensions = None
    if timed and frequency(meta["freq"]) == ("calendar", "1ME"):
        import pandas as pd

        stamps = np.memmap(
            Path(path).with_suffix(".stamps"), dtype="<i8", mode="r", offset=start, shape=(length,)
        )
        dimensions = pd.DatetimeIndex(stamps).days_in_month.to_numpy()
    return meta, values, dimensions


def build_index(spool, directory, identity, *, memory_budget_bytes):
    """Packed sorted 128-bit keys + owner/position: 32 bytes per posting.

    Only benchmark sketches are retained. Workers map immutable files rather
    than constructing millions of Python posting objects or copying an index.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "index.json"
    contract = dict(benchmark=identity, method=METHOD, implementation=digest(Path(__file__)))
    saved = _saved_json(marker)
    names = ("postings.bin", "features.bin")
    if saved.get("contract") == contract and all(
        (directory / n).is_file() and saved["files"].get(n) == digest(directory / n) for n in names
    ):
        return saved
    marker.unlink(missing_ok=True)
    budget = MemoryBudget(memory_budget_bytes)
    count = 0
    db = _open_spool(spool)
    try:
        with (
            (directory / names[0]).open("wb") as out,
            (directory / names[1]).open("wb") as features,
        ):
            for (owner,) in db.execute("SELECT rowid FROM series ORDER BY rowid"):
                _, values, _ = _series(db, spool, owner)
                rec = sketch(values)
                rows = [
                    (int(token[:16], 16), int(token[16:], 16), owner, pos)
                    for token, pos, _ in rec.keys
                ]
                out.write(np.asarray(rows, dtype=POSTING).tobytes())
                features.write(np.asarray(rec.features, dtype="<f4").tobytes())
                count += len(rows)
                budget.check(context="sketch benchmark extraction")
        if count:
            rows = np.memmap(directory / names[0], dtype=POSTING, mode="r+")
            rows.sort(order=["hi", "lo", "owner", "pos"], kind="quicksort")
            rows.flush()
            del rows
        budget.check(context="sketch index sort")
        saved = dict(
            contract=contract,
            postings=count,
            series=identity["count"],
            bytes=sum((directory / n).stat().st_size for n in names),
            files={n: digest(directory / n) for n in names},
        )
        atomic_json(marker, saved)
        return saved
    finally:
        db.close()


class SketchIndex:
    """Read-only lookup; oversized equal-key ranges remain explicit diagnostics."""

    def __init__(self, directory, spool):
        directory = Path(directory)
        self.rows = np.memmap(directory / "postings.bin", dtype=POSTING, mode="r")
        self.keys = self.rows[["hi", "lo"]]
        self.features = np.memmap(directory / "features.bin", dtype="<f4", mode="r").reshape(
            -1, FEATURES
        )
        self.spool = Path(spool)
        self.db = _open_spool(spool)

    @lru_cache(maxsize=64)
    def series(self, owner):
        return _series(self.db, self.spool, owner)

    def lookup(self, rec):
        targets = np.zeros(len(rec.keys), dtype=self.keys.dtype)
        targets["hi"] = [int(k[:16], 16) for k, _, _ in rec.keys]
        targets["lo"] = [int(k[16:], 16) for k, _, _ in rec.keys]
        left = np.searchsorted(self.keys, targets, side="left")
        right = np.searchsorted(self.keys, targets, side="right")
        candidates, summaries, heavy = {}, [], []
        for (token, pos, label), a, b in zip(rec.keys, left, right):
            if b - a > POSTING_CAP:
                heavy.append(
                    dict(
                        key=token,
                        family=label,
                        postings=int(b - a),
                        retained=POSTING_CAP,
                        reason="posting_cap",
                    )
                )
            rows = self.rows[a : min(b, a + POSTING_CAP)]
            if label == "summary":
                summaries.extend(int(o) for o in rows["owner"])
            else:
                for row in rows:
                    owner, off = int(row["owner"]), pos - int(row["pos"])
                    candidates.setdefault((owner, off), Candidate(off, label))
        return sorted(candidates.items()), sorted(set(summaries)), heavy

    def close(self):
        self.series.cache_clear()
        self.db.close()


def _words(text):
    import re

    return set(re.findall(r"[a-z0-9]+", str(text).lower())) - {
        "unknown",
        "data",
        "dataset",
        "the",
        "univariate",
        "covariates",
        "1d",
        "1h",
        "1w",
        "1m",
        "1q",
        "1y",
    }


def routing(entry, datasets, aliases, sources):
    """Metadata similarities route investigation and NEVER exclude value pairs."""
    own = _words(entry["folder"] + " " + entry.get("lineage", "") + " " + " ".join(sources))
    canonical = {frequency(a) for a in aliases} - {None}
    result = []
    for dataset, meta in sorted(datasets.items()):
        shared = sorted(own & _words(dataset))
        domain = entry.get("domain", "unknown")
        known_domains = {d.lower() for d in meta["domains"]} - {"unknown"}
        shared_domain = (
            domain.lower() in known_domains
            if domain.lower() != "unknown" and known_domains
            else None
        )
        shared_freq = bool(canonical & {frequency(a) for a in meta["frequencies"]} - {None})
        result.append(
            dict(
                dataset=dataset,
                benchmark_frequencies=sorted(meta["frequencies"], key=str),
                benchmark_domains=sorted(meta["domains"]),
                shared_source=bool(shared),
                source_tokens=shared,
                shared_domain=shared_domain,
                shared_frequency=shared_freq,
            )
        )
    return result


def _journal_name(entry):
    return entry["folder"].replace("/", "--")


def _rank_key(evidence):
    return (
        evidence["distance"],
        json.dumps(evidence["corpus"], sort_keys=True),
        json.dumps(evidence["benchmark"], sort_keys=True),
        evidence["offset"],
    )


def scan_folder(request):
    """Disposable worker: decode each native row once, publish evidence before marker."""
    budget = request["budget"]
    start_memory_watchdog(budget)
    entry = request["entry"]
    name = _journal_name(entry)
    directory = Path(request["directory"])
    path = directory / f"{name}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    index = SketchIndex(request["index"], request["spool"])
    counts, reasons = Counter(), Counter()
    matches, ranked, control_evidence = [], [], {}
    aliases, sources = set(), set()
    try:
        with tmp.open("w") as stream:

            def emit(record):
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

            for rec in iter_rows(
                request["folder_path"], entry["folder"].split("/")[0], memory_budget_bytes=budget
            ):
                row = rec["row"]
                aliases.add(row["freq"])
                sources.add(row["source"])
                for column in ("target", "covariates"):
                    for channel, values in enumerate(rec[column]):
                        counts["channels"] += 1
                        counts["values"] += len(values)
                        computed = sketch(values)
                        pairs, summary, heavy = index.lookup(computed)
                        corpus = dict(
                            item_id=row["item_id"],
                            source=row["source"],
                            freq=row["freq"],
                            shard=rec["shard"],
                            row=rec["row_index"],
                            column=column,
                            channel=channel,
                        )
                        for event in heavy:
                            counts["unresolved_buckets"] += 1
                            reasons[event["reason"]] += 1
                            emit(dict(corpus=corpus, unresolved=event))
                        seen = set()

                        def verify(owner, proposal):
                            seen.add((owner, proposal.offset))
                            meta, benchmark, dims = index.series(owner)
                            evidence = verify_candidate(
                                values,
                                benchmark,
                                proposal,
                                opportunities=request["opportunities"],
                                dimensions=dims,
                            )
                            evidence.update(
                                corpus=corpus,
                                benchmark=meta,
                                transform=dict(
                                    method="identity",
                                    aggregation="unsupported",
                                    affine_query_to_benchmark=(evidence["fit"] or {}).get("coef"),
                                ),
                            )
                            counts["proposed_pairs"] += 1
                            counts[
                                "verified_matches"
                                if evidence["verified"]
                                else "unverified_candidates"
                            ] += 1
                            emit(evidence)
                            if evidence["verified"]:
                                if len(matches) < MATCH_SAMPLE:
                                    matches.append(evidence)
                                if request.get("control"):
                                    control_evidence.setdefault(row["item_id"], []).append(evidence)
                            else:
                                reasons[evidence["reason"]] += 1
                                ranked.append(evidence)
                                ranked.sort(key=_rank_key)
                                del ranked[TOP:]
                            return evidence

                        for (owner, _), proposal in pairs:
                            distance = float(
                                np.linalg.norm(
                                    np.asarray(computed.features) - index.features[owner - 1]
                                )
                            )
                            verify(owner, Candidate(proposal.offset, proposal.reason, distance))
                        # Summary proposals compete for a folder's bounded review list.
                        # They never open an interior-crop search or all-pairs expansion.
                        if summary:
                            distances = np.linalg.norm(
                                index.features[np.asarray(summary) - 1]
                                - np.asarray(computed.features),
                                axis=1,
                            )
                            best = min(
                                range(len(summary)), key=lambda i: (distances[i], summary[i])
                            )
                            owner, distance = summary[best], float(distances[best])
                            if (owner, 0) not in seen and (
                                len(ranked) < TOP or distance < ranked[-1]["distance"]
                            ):
                                verify(owner, Candidate(0, "summary_shortlist", distance))
                            else:
                                counts["summary_pairs_not_proposed"] += 1
                        MemoryBudget(budget).check(context="sketch corpus channel")
        tmp.replace(path)
        result = dict(
            folder=entry["folder"],
            status=entry["status"],
            counts=dict(sorted(counts.items())),
            reasons=dict(sorted(reasons.items())),
            matches=matches,
            ranked_unverified=ranked,
            evidence=dict(file=path.name, sha256=digest(path)),
            match_sample_limit=MATCH_SAMPLE,
            corpus_frequencies=sorted(aliases),
            corpus_sources=sorted(sources),
            lineage=entry.get("lineage", "unknown"),
            domain=entry.get("domain", "unknown"),
        )
        if request.get("control"):
            result["control_evidence"] = control_evidence
        return result
    finally:
        index.close()


def _control_shards(spool, directory, opportunities):
    """Plants selected numerically, before retrieval, with no self-search screening."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    db = _open_spool(spool)
    rows, cases, selected = [], [], set()
    available = set()
    try:
        for (owner,) in db.execute("SELECT rowid FROM series ORDER BY rowid"):
            meta, values, dimensions = _series(db, spool, owner)
            available.add(meta["dataset"])
            if meta["dataset"] in selected or not 96 <= len(values) <= 65536:
                continue
            eligible = True
            for start, end in (
                (0, len(values)),
                (0, len(values) * 2 // 3),
                (len(values) // 3, len(values)),
            ):
                segment = values[start:end]
                finite = np.isfinite(segment)
                v = segment[finite]
                score = audit_info.info_bits(v, np.flatnonzero(finite))
                if (
                    len(v) < old.SEG_MIN
                    or not old.nondegenerate(v)[0]
                    or _repetitive(v)
                    or (
                        dimensions is not None
                        and old.calendar_pattern(v, dimensions[start:end][finite])
                    )
                    or score.strict_ramp
                    or score.bits < audit_info.INFO_MARGIN_BITS + math.log2(max(1, opportunities))
                ):
                    eligible = False
                    break
            if not eligible:
                continue
            selected.add(meta["dataset"])
            cut = len(values) // 3
            variants = dict(
                whole=values,
                affine_positive=values * 2.5 + 7,
                affine_negative=values * -1.75 - 3,
                shift=values + 19,
                float32=values.astype("float32").astype("float64"),
                prefix=values[: 2 * cut],
                suffix=values[cut:],
                interior_miss=values[cut : 2 * cut],
            )
            for label, planted in variants.items():
                item = f"plant-{owner}-{label}"
                rows.append(
                    dict(
                        item_id=item,
                        source="benchmark-sidecar-control",
                        freq=meta["freq"] or "unknown",
                        target=[planted.tolist()],
                    )
                )
                cases.append(
                    dict(
                        item_id=item,
                        **{"class": label},
                        benchmark=meta,
                        expected=label != "interior_miss",
                        index_start=cut if label == "suffix" else 0,
                        index_end=len(values) if label == "suffix" else len(planted),
                        length=len(planted),
                    )
                )
        for label, values in (
            ("constant", np.ones(512)),
            ("ramp", np.arange(512.0)),
            (
                "calendar_only",
                np.tile(
                    [31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0], 43
                ),
            ),
        ):
            rows.append(
                dict(item_id=label, source="negative-control", freq="h", target=[values.tolist()])
            )
            cases.append(dict(item_id=label, **{"class": label}, expected=False, benchmark=None))
        path = directory / "train-controls.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        return cases, sorted(available - selected)
    finally:
        db.close()


def _check_controls(cases, result, unplantable):
    evidence = result.get("control_evidence", {})
    checked = []
    for case in cases:
        records = evidence.get(case["item_id"], [])
        records = [
            r for r in records if case["benchmark"] is None or r["benchmark"] == case["benchmark"]
        ]
        exact = [
            r
            for r in records
            if r["fit"]["index_start"] == case.get("index_start")
            and r["fit"]["index_end"] == case.get("index_end")
            and r["fit"]["t_start"] == 0
            and r["fit"]["t_end"] == case.get("length")
        ]
        ok = bool(exact) if case["expected"] else not records
        checked.append(dict(**case, ok=ok, verified=len(records)))
    return dict(
        ok=any(c["expected"] for c in cases) and all(c["ok"] for c in checked),
        cases=checked,
        unplantable_datasets=unplantable,
        unsupported=[
            "all aggregation phases",
            "changed masks",
            "interior crops (negative control)",
        ],
        design="sidecar Parquet through the same reader/index/worker/verifier as all folders; full benchmark index",
    )


def _execute(request):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.corpus_v4.audit_sketch", "--worker"],
        input=json.dumps(request),
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"sketch worker exit {result.returncode}: {result.stderr.strip()[-2000:]}"
        )
    return json.loads(result.stdout)


def _checkpoint(request, identity):
    directory = Path(request["directory"])
    name = _journal_name(request["entry"])
    marker = directory / f"{name}.json"
    saved = _saved_json(marker)
    if saved.get("identity") == identity and saved.get("result_digest") == _fingerprint(
        saved.get("result")
    ):
        result = saved["result"]
        path = directory / result["evidence"]["file"]
        if path.is_file() and digest(path) == result["evidence"]["sha256"]:
            return result
    result = _execute(request)
    atomic_json(marker, dict(identity=identity, result=result, result_digest=_fingerprint(result)))
    return result


def prediction(index_bytes, benchmark_series):
    """Conditional planning arithmetic, explicitly NOT a measured cloud forecast."""
    return dict(
        machine="n2d-standard-32 (128 GiB), 16 workers initially",
        measured=False,
        values=11_400_000_000,
        channels=13_800_000,
        benchmark_values=752_000_000,
        seconds=dict(
            decode_hash_and_statistics=2400,
            endpoint_probes=1200,
            benchmark_build=600,
            verification_and_transfer=1800,
        ),
        predicted_wall_seconds=6000,
        target_seconds=7200,
        assumptions=[
            "whole-machine 5M values/s including Arrow and statistics",
            "whole-machine 12k channels/s for endpoint keys and lookups",
            "benchmark extraction/build in 10 minutes; verification + transfers in 30 minutes",
            "no dominant serial folder, huge common-key region, or slow cold benchmark download",
        ],
        index_bytes=index_bytes,
        benchmark_series=benchmark_series,
        memory="index mapped/shared + at most 6 GiB RSS per worker + 8 GiB reserve; budget admission reduces workers",
        disk="recommend 200 GiB persistent disk for corpus, benchmark cache, index and journals",
        cloud_must_confirm=[
            "full-index build time/bytes",
            "largest-folder throughput and decoder peak RSS",
            "candidate/verification rate and capped buckets",
            "controls through full crowded index",
            "GCS and cold benchmark transfer time",
            "whole pass <=2h; otherwise revise forecast",
        ],
    )


def _markdown(report):
    lines = [
        "# PROVISIONAL corpus sketch triage — NOT clearance",
        "",
        "This report cannot satisfy the training/release audit gate. A negative is not clearance.",
        "",
        f"Execution and controls passed: {report['execution_ok']}. Release eligible: false.",
        "",
        "| Folder | Policy | Channels | Verified | Unverified |",
        "|---|---|---:|---:|---:|",
    ]
    for folder in report["folders"]:
        counts = folder.get("counts", {})
        lines.append(
            f"| {folder['folder']} | {folder['status']} | {counts.get('channels', 0)} | {counts.get('verified_matches', 0)} | {counts.get('unverified_candidates', 0)} |"
        )
    lines.extend(
        [
            "",
            *LIMITATIONS,
            "",
            "Conditional forecast: 100 minutes on n2d-standard-32; not measured. See JSON assumptions.",
            "Every proposed pair was refitted on original values. Numerical similarity is evidence, not proof of provenance.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_sketch(
    root,
    out,
    manifest,
    benchmarks=None,
    *,
    state=None,
    workers=1,
    memory_budget_bytes=DEFAULT_MEMORY_BUDGET_BYTES,
    require_pinned=False,
):
    """Run triage. Exit zero means execution/controls only, NEVER release permission.

    Public CLI verifies published bytes before entering this function. Direct
    calls support offline synthetic trees and still bind all shard bytes.
    """
    import pyarrow.parquet as pq

    validate_manifest(manifest)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    MemoryBudget(memory_budget_bytes)
    root, out = Path(root), Path(out)
    state = Path(state or out / "state")
    for destination in (state, out):
        if (
            root.resolve() == destination.resolve()
            or root.resolve() in destination.resolve().parents
        ):
            raise ValueError("sidecars, state and reports must stay outside the corpus tree")
    state.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    # Remove stale final reports even if preparation or a worker fails.
    for name in ("sketch-report.json", "sketch-report.md"):
        (out / name).unlink(missing_ok=True)
    spool, benchmark_identity = prepare_benchmarks(
        state, benchmarks, memory_budget_bytes=memory_budget_bytes, require_pinned=require_pinned
    )
    index_dir = state / "sketch-index"
    built = build_index(
        spool, index_dir, benchmark_identity, memory_budget_bytes=memory_budget_bytes
    )
    identities, corpus_bytes = {}, 0
    for entry in manifest["folders"]:
        files = sorted(folder_path(root, entry).glob("*.parquet"))
        if not files:
            raise ValueError(f"missing corpus shards: {entry['folder']}")
        identities[entry["folder"]] = {p.name: digest(p) for p in files}
        for path in files:
            parquet = pq.ParquetFile(path)
            corpus_bytes += parquet.metadata.serialized_size + sum(
                parquet.metadata.row_group(i).total_byte_size for i in range(parquet.num_row_groups)
            )
            parquet.close()
    opportunities = (
        max(1, corpus_bytes) * max(1, benchmark_identity["points"]) * (2 * MAX_PHASES * 16)
    )
    identity = dict(
        manifest=manifest,
        shards=identities,
        benchmark=benchmark_identity,
        index=built["files"],
        method=METHOD,
        opportunities=opportunities,
        implementation=dict(
            sketch=digest(Path(__file__)),
            verifier=digest(Path(old.__file__)),
            information=digest(Path(audit_info.__file__)),
            matching=digest(Path(__file__).with_name("matching.py")),
            reader=digest(Path(__file__).with_name("native.py")),
        ),
    )
    key = _fingerprint(identity)
    sidecars = state / "sketch-controls"
    # Controls are deliberately rerun on EVERY invocation, including resumes.
    cases, unplantable = _control_shards(spool, sidecars, opportunities)
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    # Conservative admission: include the shared mapping in every worker quota.
    reserve = min(8 * 1024**3, memory_budget_bytes // 4)
    minimum = built["bytes"] + 256 * 1024**2
    active = min(workers, max(1, (memory_budget_bytes - reserve) // minimum))
    quota = (memory_budget_bytes - reserve) // active
    if quota < minimum:
        raise ValueError("memory budget cannot admit one sketch worker and the mapped index")
    common = dict(
        directory=str(evidence_dir),
        index=str(index_dir),
        spool=str(spool),
        opportunities=opportunities,
        budget=quota,
    )
    control_request = dict(
        **common,
        entry=dict(folder="univariate/controls", status="sidecar"),
        folder_path=str(sidecars),
        control=True,
    )
    controls = _check_controls(cases, _execute(control_request), unplantable)
    db = _open_spool(spool)
    datasets = {}
    try:
        for (encoded,) in db.execute("SELECT meta FROM series"):
            meta = json.loads(encoded)
            group = datasets.setdefault(meta["dataset"], dict(frequencies=set(), domains=set()))
            group["frequencies"].add(meta["freq"])
            group["domains"].add(meta["domain"])
    finally:
        db.close()
    entries = sorted(manifest["folders"], key=lambda e: e["folder"])

    def run(entry):
        request = dict(**common, entry=entry, folder_path=str(folder_path(root, entry)))
        try:
            result = _checkpoint(request, key)
            result["routing"] = routing(
                entry, datasets, result["corpus_frequencies"], result["corpus_sources"]
            )
            return result
        except (RuntimeError, OSError, ValueError) as exc:
            return dict(folder=entry["folder"], status=entry["status"], error=str(exc))

    with ThreadPoolExecutor(max_workers=active) as pool:
        folders = list(pool.map(run, entries))
    totals = {}
    for status in ("allowed", "quarantined"):
        counts = Counter()
        for folder in folders:
            if folder["status"] == status:
                counts.update(folder.get("counts", {}))
        totals[status] = dict(sorted(counts.items()))
    execution_ok = controls["ok"] and all("error" not in f for f in folders)
    report = dict(
        artifact_type=ARTIFACT,
        schema_version=1,
        method=METHOD,
        provisional=True,
        release_eligible=False,
        execution_ok=execution_ok,
        notice=LIMITATIONS[0],
        limitations=LIMITATIONS,
        input_fingerprint=key,
        benchmark_revision=FEV_DATASETS_REVISION,
        benchmark_series=benchmark_identity["count"],
        benchmark_points=benchmark_identity["points"],
        index_postings=built["postings"],
        controls=controls,
        folders=folders,
        totals=totals,
        prediction=prediction(built["bytes"], benchmark_identity["count"]),
        sketch_definition=dict(
            endpoints=list(LENGTHS),
            grids=[0.0, 0.5],
            step=0.05,
            statistics="length, positional mask/raw digests, quantiles, zeros, changes, longest run, acf(1,7)",
            posting_cap=POSTING_CAP,
            ranked_limit=TOP,
        ),
    )
    atomic_json(out / "sketch-report.json", report)
    (out / "sketch-report.md").write_text(_markdown(report))
    return report, 0 if execution_ok else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--memory-budget-gib", type=memory_budget_arg, default=DEFAULT_MEMORY_BUDGET_BYTES
    )
    args = parser.parse_args(argv)
    manifest = read_manifest(args.manifest)
    verify_published(args.root, manifest)
    started = time.monotonic()
    report, code = run_sketch(
        args.root,
        args.out,
        manifest,
        state=args.state,
        workers=args.workers,
        memory_budget_bytes=args.memory_budget_gib,
        require_pinned=True,
    )
    print(
        f"PROVISIONAL triage: execution_ok={report['execution_ok']}; elapsed={time.monotonic() - started:.1f}s; NOT clearance"
    )
    return code


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(scan_folder(json.load(sys.stdin)), allow_nan=False))
    else:
        raise SystemExit(main())
