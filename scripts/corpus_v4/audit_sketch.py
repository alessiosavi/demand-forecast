"""Provisional endpoint fingerprints: triage evidence, NEVER a release audit.

Only endpoints are indexed. Negative retrieval says nothing about interior crops,
changed masks, rounding boundaries or any resampling/aggregation ancestry.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
    digest,
    folder_path,
    iter_rows,
    memory_budget_arg,
    start_memory_watchdog,
)
from scripts.corpus_v4.verify_published import verify_published

METHOD = "provisional-endpoints-fidelity-2"
ARTIFACT = "corpus-sketch-triage"
LENGTHS = (24, 64, 256)
POSTING_CAP = 64
TOP = 20
MATCH_SAMPLE = 100
FIDELITY_RTOL = 0.005
FIDELITY_FRACTION = 0.99
CHECKPOINT_ROWS = 512
CHECKPOINT_VALUES = 2_000_000
UNIT_VALUES = 16_000_000
UNIT_ROWS = 8192
FEATURES = 11
SERIES_CACHE_BYTES = 64 * 1024**2
SERIES_READ_BYTES = 8 * 1024**2
SERIES_CACHE_ENTRIES = 1024
POSTING = np.dtype([("hi", "<u8"), ("lo", "<u8"), ("owner", "<i8"), ("pos", "<i8")])
LIMITATIONS = [
    "Provisional triage only; a negative result is NOT corpus clearance.",
    "Interior crops and copies not preserving a sampled endpoint are not searched.",
    "Changed finite masks and quantisation-boundary cases can escape retrieval.",
    "No fixed-ratio or calendar aggregation is computed; every phase is unsupported.",
    "Common keys are capped and recorded; summary ranking is a bounded heuristic.",
    "Constants, ramps and calendar-only patterns are non-identifying by existing policy.",
    "Fidelity requires benchmark variation, including outside the retrieval key; lost rare events can reject an otherwise 99%-equal pair.",
    "Range-relative fidelity can miss small steps dwarfed by larger excursions; numerical conditioning and tiny modal groups can reject copies.",
    "Numerical fidelity is not provenance; unchanged retrieval and information gates still exclude some true copies.",
]


def _workspace(state):
    path = Path(state).parent / ("." + Path(state).name + "-sketch-work")
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path, value, *, scratch=None):
    """Growing bytes live outside the periodically copied state tree."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(scratch) if scratch else _workspace(path.parent)
    scratch.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=scratch)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


class _Ranking:
    """Cache each evidence key once, retaining the original stable ordering."""

    def __init__(self):
        self.keys, self.records = [], []

    def add(self, record):
        key = _rank_key(record)
        pos = bisect.bisect_right(self.keys, key)
        if pos < TOP:
            self.keys.insert(pos, key)
            self.records.insert(pos, record)
            del self.keys[TOP:]
            del self.records[TOP:]

    def admits(self, distance):
        return len(self.records) < TOP or distance < self.records[-1]["distance"]


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


def _variation_fidelity(a, b, positions, overlap_length, proposal, fit):
    """One global map must reproduce the benchmark's variation in every region.

    Positions refer to the unmasked overlap. No absolute tolerance or r2 floor
    enters this decision. Identity preserves the original 99%-agreement branch;
    the modal median map resists edits at otherwise uninformative positions.
    """
    levels, counts = np.unique(b, return_counts=True)
    mode = float(levels[counts.argmax()])
    span = float(np.ptp(b))
    events = np.abs(b - mode) > FIDELITY_RTOL * span
    window = None
    inside = np.zeros(len(b), dtype=bool)
    side, _, width = proposal.partition("/")
    if side in ("prefix", "suffix") and width.isdigit():
        width = min(int(width), overlap_length)
        window = [0, width] if side == "prefix" else [overlap_length - width, overlap_length]
        inside = (positions >= window[0]) & (positions < window[1])
    result = dict(
        passed=False,
        relative_tolerance=FIDELITY_RTOL,
        minimum_fraction=FIDELITY_FRACTION,
        benchmark_mode=mode,
        benchmark_range=span,
        retrieval_window=window,
        off_modal_points=int(events.sum()),
        off_modal_inside=int((events & inside).sum()) if window else None,
        off_modal_outside=int((events & ~inside).sum()) if window else None,
        maps=[],
    )
    if not span > 0 or not math.isfinite(span):
        return result

    def assess(method, predicted, parameters):
        with np.errstate(over="ignore", invalid="ignore"):
            reproduced = np.isfinite(predicted) & (np.abs(b - predicted) / span <= FIDELITY_RTOL)

        def fraction(mask):
            return float(reproduced[mask].mean()) if mask.any() else None

        rec = dict(
            method=method,
            parameters=parameters,
            reproduced_fraction=float(reproduced.mean()),
            off_modal_fraction=fraction(events),
            inside_off_modal_fraction=fraction(events & inside) if window else None,
            outside_off_modal_fraction=fraction(events & ~inside) if window else None,
        )
        rec["passed"] = all(
            value is None or value >= FIDELITY_FRACTION
            for key, value in rec.items()
            if key.endswith("fraction")
        )
        result["maps"].append(rec)
        result["passed"] |= rec["passed"]

    coef = fit["coef"]
    with np.errstate(over="ignore", invalid="ignore"):
        assess("least_squares", coef[0] * a + coef[1], dict(coef=coef))
    if not result["passed"] and fit["exact_frac"] >= old.EXACT_FRACTION:
        assess("identity", a, dict(coef=[1.0, 0.0]))
    if not result["passed"] and events.any():
        anchor = float(np.median(a[b == mode]))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            inverse_slope = float(np.median((a[events] - anchor) / (b[events] - mode)))
            if math.isfinite(inverse_slope) and inverse_slope != 0:
                # Anchored evaluation avoids cancellation from a large intercept.
                assess(
                    "modal_median",
                    (a - anchor) / inverse_slope + mode,
                    dict(corpus_anchor=anchor, benchmark_anchor=mode, inverse_slope=inverse_slope),
                )
    return result


def verify_candidate(
    x,
    y,
    candidate,
    *,
    opportunities,
    dimensions=None,
    nonidentifying_shortcut=False,
    cache=None,
    benchmark_owner=None,
    benchmark_cache=None,
):
    """Reuse every original gate, then require benchmark variation fidelity.

    No Index/scan_group retrieval is reused. Original values and original masks
    are refitted over the endpoint alignment's entire overlap.
    """
    off = candidate.offset
    if nonidentifying_shortcut:
        start, end = max(0, off), min(len(x), len(y) + off)
        if end - start >= old.WIN:
            a, b = x[start:end], y[start - off : end - off]
            both = np.isfinite(a) & np.isfinite(b)
            if both.sum() >= old.WIN:
                # Same alignment and joint mask as fit_segment. Whole-channel
                # degeneracy is not generally hereditary under cropping/masking.
                mask = np.packbits(both).tobytes()
                memo = {} if cache is None or len(mask) > 8192 else cache
                qa = ("query", start, end, mask)
                qb = ("benchmark", benchmark_owner, start - off, end - off, mask)
                if qa not in memo:
                    memo[qa] = old.nondegenerate(a[both])
                benchmark_memo = (
                    memo if benchmark_cache is None or len(mask) > 8192 else benchmark_cache
                )
                if benchmark_memo is not memo and len(benchmark_memo) >= 1024:
                    benchmark_memo.clear()
                if qb not in benchmark_memo:
                    benchmark_memo[qb] = old.nondegenerate(b[both])
                ok_a, why_a = memo[qa]
                ok_b, why_b = benchmark_memo[qb]
                if not (ok_a and ok_b):
                    return dict(
                        verified=False,
                        reason="nonidentifying:" + str([why_a, why_b]),
                        proposal=candidate.reason,
                        offset=off,
                        distance=candidate.distance,
                        fit=None,
                        fidelity=None,
                        old_rule_accepted=False,
                        shortcut="verifier_nondegenerate_on_joint_overlap",
                        information_threshold=audit_info.INFO_MARGIN_BITS
                        + math.log2(max(1, opportunities)),
                    )
    rec = old.fit_segment(x, y, max(0, off), min(len(x), len(y) + off), off)
    reason = "numerical_fit_rejected"
    fidelity = None
    if rec is not None:
        rec.pop("_info_values", None)
        a = x[rec["t_start"] : rec["t_end"]]
        b = y[rec["index_start"] : rec["index_end"]]
        finite = np.isfinite(a) & np.isfinite(b)
        positions = np.flatnonzero(finite)
        a, b = a[finite], b[finite]
        fidelity = _variation_fidelity(a, b, positions, len(finite), candidate.reason, rec)
        if rec["is_copy"]:
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
    old_rule_accepted = reason == "verified"
    if old_rule_accepted and not fidelity["passed"]:
        reason = "variation_fidelity_rejected"
    return dict(
        verified=reason == "verified",
        reason=reason,
        proposal=candidate.reason,
        offset=off,
        distance=candidate.distance,
        fit=rec,
        fidelity=fidelity,
        old_rule_accepted=old_rule_accepted,
        information_threshold=audit_info.INFO_MARGIN_BITS + math.log2(max(1, opportunities)),
    )


def _spool_files(path):
    return [path, path.with_suffix(".values"), path.with_suffix(".stamps")]


def _benchmark_sources():
    # Local code executed by preparation, task loading, frequency normalization,
    # spool writing and BenchmarkStore's sample eligibility checks. Keep this
    # list with the call graph when adding loader/writer dependencies.
    root = Path(__file__).resolve().parents[2]
    names = (
        "scripts/corpus_v4/audit_sketch.py",
        "scripts/corpus_v4/audit_fev_overlap.py",
        "scripts/corpus_v4/native.py",
        "scripts/corpus_v4/matching.py",
        "demand_forecast/bench/specs.py",
        "demand_forecast/pretrain/audit.py",
        "demand_forecast/pretrain/audit_match.py",
        "demand_forecast/pretrain/audit_info.py",
    )
    return {name: digest(root / name) for name in names}


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
        sources=_benchmark_sources(),
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
        else load_benchmarks(_workspace(state) / "benchmark-cache" / FEV_DATASETS_REVISION)
    )
    marker.unlink(missing_ok=True)
    # Recreate SQLite from scratch: DROP/CREATE changes SQLite's header counter
    # even for identical logical rows, which would otherwise alter resume keys.
    staging = _workspace(state) / "sketch-benchmark-build"
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


def _series(db, path, owner, *, calendar_dimensions=True, values_file=None, stamps_file=None):
    encoded, start, length, timed = db.execute(
        "SELECT meta, start, length, timed FROM series WHERE rowid=?", (int(owner),)
    ).fetchone()
    meta = json.loads(encoded)

    def read(suffix, dtype, stream):
        if not length:
            return np.empty(0, dtype=dtype)
        if stream is not None and length * 8 <= SERIES_READ_BYTES:
            stream.seek(start)
            array = np.fromfile(stream, dtype=dtype, count=length)
            if len(array) != length:
                raise ValueError("truncated benchmark spool")
            array.flags.writeable = False
            return array
        return np.memmap(
            Path(path).with_suffix(suffix), dtype=dtype, mode="r", offset=start, shape=(length,)
        )

    values = read(".values", "<f8", values_file)
    dimensions = None
    if calendar_dimensions and timed and frequency(meta["freq"]) == ("calendar", "1ME"):
        import pandas as pd

        stamps = read(".stamps", "<i8", stamps_file)
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
    destination = directory
    directory = _workspace(directory.parent) / "index-build"
    directory.mkdir(parents=True, exist_ok=True)
    budget = MemoryBudget(memory_budget_bytes)
    count = 0
    db = _open_spool(spool)
    try:
        with (
            (directory / names[0]).open("wb") as out,
            (directory / names[1]).open("wb") as features,
        ):
            for (owner,) in db.execute("SELECT rowid FROM series ORDER BY rowid"):
                _, values, _ = _series(db, spool, owner, calendar_dimensions=False)
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
        for name in names:
            (directory / name).replace(destination / name)
        atomic_json(marker, saved, scratch=directory)
        return saved
    finally:
        db.close()


class SketchIndex:
    """Read-only lookup; oversized equal-key ranges remain explicit diagnostics."""

    def __init__(self, directory, spool):
        directory = Path(directory)
        self.rows = np.memmap(directory / "postings.bin", dtype=POSTING, mode="r")
        self.highs, self.lows = self.rows["hi"], self.rows["lo"]
        self.features = np.memmap(directory / "features.bin", dtype="<f4", mode="r").reshape(
            -1, FEATURES
        )
        self.degeneracy = {}
        self.spool = Path(spool)
        self.db = _open_spool(spool)
        # Read short histories through persistent descriptors rather than
        # opening/mapping/closing a file on every miss of a 64-owner cache.
        # Oversized histories still use a disposable read-only mapping.
        self.values_file = self.spool.with_suffix(".values").open("rb", buffering=0)
        self.stamps_file = self.spool.with_suffix(".stamps").open("rb", buffering=0)
        self.series_cache, self.series_cache_bytes = OrderedDict(), 0

    def series(self, owner):
        if owner in self.series_cache:
            self.series_cache.move_to_end(owner)
            return self.series_cache[owner][0]
        result = _series(
            self.db, self.spool, owner, values_file=self.values_file, stamps_file=self.stamps_file
        )
        meta, values, dimensions = result
        if isinstance(values, np.memmap):
            return result
        size = (
            sys.getsizeof(meta)
            + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in meta.items())
            + sys.getsizeof(values)
            + (0 if dimensions is None else sys.getsizeof(dimensions) + dimensions.nbytes)
            + 512  # owner, tuple, size integer and OrderedDict node allowance
        )
        if size <= SERIES_CACHE_BYTES:
            while self.series_cache and (
                self.series_cache_bytes + size > SERIES_CACHE_BYTES
                or len(self.series_cache) >= SERIES_CACHE_ENTRIES
            ):
                _, (_, removed) = self.series_cache.popitem(last=False)
                self.series_cache_bytes -= removed
            self.series_cache[owner] = result, size
            self.series_cache_bytes += size
        return result

    def lookup(self, rec):
        # Selecting hi/lo together leaves 16 bytes of tail padding in each
        # 32-byte posting. NumPy packs that structured view on EVERY search,
        # copying the full index twice per channel. Search scalar-word views
        # instead: first the high-word range, then the low word within it.
        # The file is sorted lexicographically by (hi, lo, owner, pos), so these
        # are exactly the same 128-bit equal-key bounds, without an index copy.
        highs = np.asarray([int(k[:16], 16) for k, _, _ in rec.keys], dtype="<u8")
        lows = np.asarray([int(k[16:], 16) for k, _, _ in rec.keys], dtype="<u8")
        left = np.searchsorted(self.highs, highs, side="left")
        right = np.searchsorted(self.highs, highs, side="right")
        candidates, summaries, heavy = {}, [], []
        for (token, pos, label), low, a, b in zip(rec.keys, lows, left, right):
            if a != b:
                low_words = self.lows[a:b]
                a, b = (
                    a + np.searchsorted(low_words, low, side="left"),
                    a + np.searchsorted(low_words, low, side="right"),
                )
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
        self.series_cache.clear()
        self.series_cache_bytes = 0
        self.values_file.close()
        self.stamps_file.close()
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


def _assemble(entry, units, directory, *, scratch, control=False, control_benchmarks=None):
    """Replay native order, including the original adaptive summary admission."""
    directory = Path(directory)
    final = directory / (_journal_name(entry) + ".jsonl")
    fd, name = tempfile.mkstemp(dir=scratch)
    counts, reasons = Counter(), Counter()
    ranked, matches, control_evidence = _Ranking(), [], {}
    fidelity_rejections = []
    control_diagnostics = {}
    control_benchmarks = control_benchmarks or {}
    aliases, sources = set(), set()

    def diagnostic(evidence, reason):
        if control:
            item = evidence["corpus"]["item_id"]
            control_diagnostics.setdefault(item, Counter())[reason] += 1

    try:
        with os.fdopen(fd, "w") as output:
            for unit in units:
                counts.update(unit["counts"])
                aliases.update(unit["corpus_frequencies"])
                sources.update(unit["corpus_sources"])
                output.write(json.dumps(unit["evidence"], sort_keys=True) + "\n")
                journal_path = directory / unit["evidence"]["file"]
                with (
                    gzip.open(journal_path, "rt")
                    if journal_path.suffix == ".gz"
                    else journal_path.open()
                ) as stream:
                    for line in stream:
                        evidence = json.loads(line)
                        if evidence.pop("summary_not_proposed", False):
                            counts["summary_pairs_not_proposed"] += 1
                            diagnostic(evidence, "summary_not_proposed")
                            continue
                        if evidence.pop("deferred_summary", False) and not ranked.admits(
                            evidence["distance"]
                        ):
                            counts["summary_pairs_not_proposed"] += 1
                            diagnostic(evidence, "summary_shortlist_pruned")
                            continue
                        if "unresolved" in evidence:
                            counts["unresolved_buckets"] += 1
                            reasons[evidence["unresolved"]["reason"]] += 1
                            diagnostic(evidence, evidence["unresolved"]["reason"])
                            continue
                        counts["proposed_pairs"] += 1
                        counts[
                            "verified_matches" if evidence["verified"] else "unverified_candidates"
                        ] += 1
                        if evidence.get("shortcut"):
                            counts["nonidentifying_pairs_skipped"] += 1
                        if control and (
                            evidence["verified"]
                            or evidence["benchmark"]
                            == control_benchmarks.get(evidence["corpus"]["item_id"])
                        ):
                            # Retain rejections only for the planted parent. The
                            # full crowded index can produce many unrelated fits.
                            control_evidence.setdefault(evidence["corpus"]["item_id"], []).append(
                                evidence
                            )
                        if evidence["verified"]:
                            if len(matches) < MATCH_SAMPLE:
                                matches.append(evidence)
                        else:
                            reasons[evidence["reason"]] += 1
                            if evidence["reason"] == "variation_fidelity_rejected":
                                counts["fidelity_rejections"] += 1
                                if len(fidelity_rejections) < MATCH_SAMPLE:
                                    fidelity_rejections.append(evidence)
                            ranked.add(evidence)
        Path(name).replace(final)
    finally:
        Path(name).unlink(missing_ok=True)
    result = dict(
        folder=entry["folder"],
        status=entry["status"],
        counts=dict(sorted(counts.items())),
        reasons=dict(sorted(reasons.items())),
        matches=matches,
        fidelity_rejections=fidelity_rejections,
        ranked_unverified=ranked.records,
        evidence=dict(
            file=final.name,
            sha256=digest(final),
            units=[u["evidence"] for u in units],
            format="unit-reference-jsonl",
        ),
        match_sample_limit=MATCH_SAMPLE,
        corpus_frequencies=sorted(aliases),
        corpus_sources=sorted(sources),
        lineage=entry.get("lineage", "unknown"),
        domain=entry.get("domain", "unknown"),
    )
    if control:
        result["control_evidence"] = control_evidence
        result["control_diagnostics"] = control_diagnostics
    return result


class _UnitJournal:
    """Bound lost sketch work within a large row group; never split a row.

    Resume re-decodes the completed prefix with the unchanged buffered reader,
    but skips its fingerprints, lookups and verification. Closed fragments are
    immutable. Their boundaries do not enter the final journal or summary.
    """

    def __init__(self, request, path, scratch):
        self.path, self.scratch = path, scratch
        self.unit = request.get("unit")
        self.identity = request.get("checkpoint_identity")
        self.counts, self.aliases, self.sources = Counter(), set(), set()
        self.parts, self.resume_row = [], -1
        if self.unit and self.identity:
            while True:
                marker = self._marker(len(self.parts))
                saved = _saved_result(marker, self.identity, path.parent, last_row=True)
                if saved is None:
                    break
                self.parts.append(saved)
                self.counts.update(saved["counts"])
                self.aliases.update(saved["corpus_frequencies"])
                self.sources.update(saved["corpus_sources"])
                self.resume_row = saved["last_row"]
        self.committed = self.counts.copy()
        self.rows = 0
        self._open()

    def _marker(self, number):
        return self.path.with_name(
            self.path.name.removesuffix(".jsonl.gz") + f"-part{number:06d}.json"
        )

    def _open(self):
        fd, name = tempfile.mkstemp(dir=self.scratch)
        self.temporary = Path(name)
        self.raw = os.fdopen(fd, "wb")
        self.stream = io.TextIOWrapper(
            gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0, compresslevel=1)
        )

    def _close_stream(self):
        self.stream.close()
        self.raw.close()

    def write(self, record):
        self.stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def after_row(self, row):
        self.resume_row = row
        self.rows += 1
        if self.unit and (
            self.rows >= CHECKPOINT_ROWS
            or self.counts["values"] - self.committed["values"] >= CHECKPOINT_VALUES
        ):
            self._commit()
            self._open()

    def _commit(self):
        self._close_stream()
        if not self.rows:
            self.temporary.unlink(missing_ok=True)
            return
        marker = self._marker(len(self.parts))
        evidence = marker.with_suffix(".jsonl.gz")
        self.temporary.replace(evidence)
        result = dict(
            counts=dict(sorted((self.counts - self.committed).items())),
            corpus_frequencies=sorted(self.aliases),
            corpus_sources=sorted(self.sources),
            last_row=self.resume_row,
            evidence=dict(file=evidence.name, sha256=digest(evidence)),
        )
        atomic_json(
            marker,
            dict(identity=self.identity, result=result, result_digest=_fingerprint(result)),
            scratch=self.scratch,
        )
        self.parts.append(result)
        self.committed = self.counts.copy()
        self.rows = 0

    def finish(self):
        if self.unit:
            self._commit()
            fd, name = tempfile.mkstemp(dir=self.scratch)
            self.temporary = Path(name)
            with os.fdopen(fd, "wb") as output:
                for part in self.parts:
                    with (self.path.parent / part["evidence"]["file"]).open("rb") as source:
                        shutil.copyfileobj(source, output)
            self.temporary.replace(self.path)
        else:
            self._close_stream()
            self.temporary.replace(self.path)
        result = dict(
            counts=dict(sorted(self.counts.items())),
            corpus_frequencies=sorted(self.aliases),
            corpus_sources=sorted(self.sources),
            evidence=dict(file=self.path.name, sha256=digest(self.path)),
        )
        if self.unit:
            result["last_row"] = self.resume_row
        return result

    def close(self):
        self._close_stream()
        self.temporary.unlink(missing_ok=True)


def scan_folder(request):
    """Disposable row-group worker; only closed journals enter the copied tree."""
    budget = request["budget"]
    start_memory_watchdog(budget)
    entry = request["entry"]
    directory = Path(request["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    scratch = Path(request.get("scratch") or _workspace(directory.parent))
    scratch.mkdir(parents=True, exist_ok=True)
    unit = request.get("unit")
    suffix = "--" + unit["id"] if unit else "--scan"
    path = directory / (_journal_name(entry) + suffix + ".jsonl.gz")
    index = SketchIndex(request["index"], request["spool"])
    journal = _UnitJournal(request, path, scratch)
    counts, aliases, sources = journal.counts, journal.aliases, journal.sources
    try:

        def emit(record):
            journal.write(record)

        for rec in iter_rows(
            request["folder_path"],
            entry["folder"].split("/")[0],
            memory_budget_bytes=budget,
            **({k: unit[k] for k in ("shard", "row_group")} if unit else {}),
        ):
            if unit and rec["row_index"] >= unit.get("row_end", math.inf):
                break
            if unit and rec["row_index"] < unit.get("row_start", 0):
                continue
            if rec["row_index"] <= journal.resume_row and journal.rows == 0:
                continue
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
                        emit(dict(corpus=corpus, unresolved=event))
                    seen, cache = set(), {}

                    def verify(owner, proposal, deferred=False):
                        seen.add((owner, proposal.offset))
                        meta, benchmark, dims = index.series(owner)
                        evidence = verify_candidate(
                            values,
                            benchmark,
                            proposal,
                            opportunities=request["opportunities"],
                            dimensions=dims,
                            nonidentifying_shortcut=request.get("nonidentifying_shortcut", True),
                            cache=cache,
                            benchmark_owner=owner,
                            benchmark_cache=index.degeneracy,
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
                        if deferred:
                            evidence["deferred_summary"] = True
                        emit(evidence)

                    for (owner, _), proposal in pairs:
                        distance = float(
                            np.linalg.norm(
                                np.asarray(computed.features) - index.features[owner - 1]
                            )
                        )
                        verify(owner, Candidate(proposal.offset, proposal.reason, distance))
                    if summary:
                        distances = np.linalg.norm(
                            index.features[np.asarray(summary) - 1] - np.asarray(computed.features),
                            axis=1,
                        )
                        best = min(range(len(summary)), key=lambda i: (distances[i], summary[i]))
                        owner, distance = summary[best], float(distances[best])
                        if (owner, 0) not in seen:
                            verify(
                                owner,
                                Candidate(0, "summary_shortlist", distance),
                                deferred=True,
                            )
                        else:
                            event = dict(summary_not_proposed=True)
                            if request.get("control"):
                                event["corpus"] = corpus
                            emit(event)
                    MemoryBudget(budget).check(context="sketch corpus channel")
            journal.after_row(rec["row_index"])
            if unit and rec["row_index"] + 1 >= unit.get("row_end", math.inf):
                break
        result = journal.finish()
        if unit:
            return result
        return _assemble(
            entry,
            [result],
            directory,
            scratch=scratch,
            control=request.get("control", False),
            control_benchmarks=request.get("control_benchmarks"),
        )
    finally:
        journal.close()
        index.close()


def _fidelity_plants(values):
    """Shape-specific plants; eligibility is checked by the unchanged old gates."""
    finite = np.isfinite(values)
    levels, counts = np.unique(values[finite], return_counts=True)
    if len(levels) != 2:
        return None, {}
    mode = levels[counts.argmax()]
    events = finite & (values != mode)
    if np.array_equal(levels, [0, 1]) and events.sum() <= len(values) * 0.005:
        kind = "sparse_flag"
    elif np.isclose(levels[1] - levels[0], 0.01, rtol=0.01, atol=0):
        kind = "cent_step"
    else:
        return None, {}
    # Preserve both endpoints and every benchmark event in the edited plants.
    modal = np.flatnonzero(finite & ~events)
    eligible = modal[(modal >= 256) & (modal < len(values) - 256)]
    n_edits = len(values) // 100
    if not n_edits or len(eligible) < n_edits:
        return kind, {}
    span = float(levels[1] - levels[0])
    edited = values.copy()
    edited[eligible[:n_edits]] += 20 * span
    affine_edited = values * 2.5 + 7
    affine_edited[eligible[:n_edits]] += (0.001 if kind == "sparse_flag" else 2) * span
    variants = dict(
        exact=(values, True),
        affine_positive=(values * 2.5 + 7, True),
        affine_negative=(values * -1.75 - 3, True),
        edited_1pct=(edited, True),
        affine_edited_1pct=(affine_edited, True),
    )
    negative = values.copy()
    if kind == "sparse_flag":
        widths = [w for w in LENGTHS if events[:w].sum() == 1]
        if widths and len(eligible) >= 3:
            width = widths[0]
            negative[width:] = np.where(finite[width:], mode, np.nan)
            negative[eligible[-3:]] = mode + span * np.array([2, 3, 4])
            variants["coincidence"] = (negative, False)
    else:
        # Same run pattern in the key, independent run pattern afterwards.
        rng = np.random.default_rng(418)
        runs = rng.integers(4, 32, len(values))
        negative[256:] = np.repeat(levels[np.arange(len(values)) % 2], runs)[: len(values) - 256]
        negative[~finite] = np.nan
        variants["coincidence"] = (negative, False)
    return kind, variants


def _control_shards(spool, directory, opportunities):
    """Select positive plants numerically; exclude endpoint-preserving negatives."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    db = _open_spool(spool)
    rows, cases, skipped, selected = [], [], [], set()
    fidelity_selected = set()
    available = set()

    def add_case(owner, meta, label, planted, expected, index_start=0):
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
                expected=expected,
                index_start=index_start,
                index_end=index_start + len(planted),
                length=len(planted),
            )
        )

    try:
        for (owner,) in db.execute("SELECT rowid FROM series ORDER BY rowid"):
            meta, values, dimensions = _series(db, spool, owner)
            available.add(meta["dataset"])
            if not 96 <= len(values) <= 65536:
                continue
            if len(fidelity_selected) < 2:
                kind, variants = _fidelity_plants(values)
                if variants and kind not in fidelity_selected:
                    admitted = verify_candidate(
                        values,
                        values,
                        Candidate(0, "raw"),
                        opportunities=opportunities,
                        dimensions=dimensions,
                    )["old_rule_accepted"]
                    if admitted:
                        fidelity_selected.add(kind)
                        for label, (planted, expected) in variants.items():
                            add_case(owner, meta, f"fidelity_{kind}_{label}", planted, expected)
            if meta["dataset"] in selected:
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
                if label == "interior_miss" and compare_sketches(sketch(planted), sketch(values)):
                    # A periodic interior can be an actual prefix/suffix copy.
                    # Key screening excludes exactly the unsupported negative
                    # premise, without an O(n * period) periodicity search or
                    # screening any positive plant for retrieval success.
                    skipped.append(
                        dict(
                            item_id=item,
                            **{"class": label},
                            benchmark=meta,
                            reason="interior_endpoint_keys_match_parent",
                        )
                    )
                    continue
                add_case(
                    owner,
                    meta,
                    label,
                    planted,
                    label != "interior_miss",
                    cut if label == "suffix" else 0,
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
        return cases, sorted(available - selected), skipped
    finally:
        db.close()


def _check_controls(cases, result, unplantable, skipped=()):
    evidence = result.get("control_evidence", {})
    checked = []
    for case in cases:
        records = evidence.get(case["item_id"], [])
        records = [
            r for r in records if case["benchmark"] is None or r["benchmark"] == case["benchmark"]
        ]
        verified = [r for r in records if r["verified"]]
        exact = [
            r
            for r in verified
            if r["fit"]["index_start"] == case.get("index_start")
            and r["fit"]["index_end"] == case.get("index_end")
            and r["fit"]["t_start"] == 0
            and r["fit"]["t_end"] == case.get("length")
        ]
        ok = bool(exact) if case["expected"] else not verified
        rejections = Counter(r["reason"] for r in records if not r["verified"])
        if ok:
            reason = "verified" if case["expected"] else "expected_miss"
        elif not case["expected"]:
            reason = "unexpected_verified_match"
        elif verified:
            reason = "expected_overlap_not_verified"
        elif rejections:
            reason = next(iter(rejections)) if len(rejections) == 1 else "verifier_rejected"
        else:
            reason = "expected_benchmark_not_proposed"
        checked.append(
            dict(
                **case,
                ok=ok,
                verified=len(verified),
                reason=reason,
                rejection_reasons=dict(sorted(rejections.items())),
                retrieval_reasons=dict(
                    sorted(result.get("control_diagnostics", {}).get(case["item_id"], {}).items())
                ),
            )
        )
    return dict(
        ok=any(c["expected"] for c in cases) and all(c["ok"] for c in checked),
        cases=checked,
        failures=[c for c in checked if not c["ok"]],
        skipped_cases=list(skipped),
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


def _valid_result(result, *, last_row=False):
    """Validate restored JSON before any consumer uses counts or evidence."""
    if not isinstance(result, dict) or not isinstance(result.get("counts"), dict):
        return False
    if any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in result["counts"].items()
    ):
        return False
    for field in ("corpus_frequencies", "corpus_sources"):
        items = result.get(field)
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            return False
    if last_row and (type(result.get("last_row")) is not int or result["last_row"] < -1):
        return False
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        return False
    units = evidence.get("units", [])
    if not isinstance(units, list):
        return False
    for item in [evidence, *units]:
        if not isinstance(item, dict):
            return False
        name = item.get("file")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(item.get("sha256"), str)
        ):
            return False
    return True


def _saved_result(marker, identity, directory, *, last_row=False):
    saved = _saved_json(marker)
    result = saved.get("result")
    if (
        not isinstance(saved.get("identity"), str)
        or not saved["identity"]
        or saved["identity"] != identity
        or not _valid_result(result, last_row=last_row)
        or saved.get("result_digest") != _fingerprint(result)
    ):
        return None
    evidence = result["evidence"]
    for item in [evidence, *evidence.get("units", [])]:
        name = item.get("file", "")
        path = Path(directory) / name
        if not path.is_file() or digest(path) != item.get("sha256"):
            return None
    return result


def _checkpoint(request, identity):
    directory = Path(request["directory"])
    name = _journal_name(request["entry"])
    if request.get("unit"):
        name += "--" + request["unit"]["id"]
    marker = directory / (name + ".json")
    result = _saved_result(marker, identity, directory, last_row=bool(request.get("unit")))
    if result is None:
        result = _execute(dict(request, checkpoint_identity=identity))
        atomic_json(
            marker,
            dict(identity=identity, result=result, result_digest=_fingerprint(result)),
            scratch=request.get("scratch"),
        )
    return result


class _Progress:
    def __init__(self, state):
        self.path = Path(state) / "sketch-progress.json"
        self.scratch = _workspace(state)
        self.started = time.monotonic()
        self.pending = {}
        self.reused_values = 0
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.record = dict(
            phase="setup", units_done=0, units_total=0, channels_done=0, values_done=0, folders={}
        )
        self.publish()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def publish(self):
        with self.lock:
            elapsed = max(1e-9, time.monotonic() - self.started)
            snapshot = json.loads(json.dumps(self.record))
            for (folder, _), (directory, prefix, identity) in self.pending.items():
                for marker in Path(directory).glob(prefix + "-part*.json"):
                    saved = _saved_json(marker)
                    result = saved.get("result", {})
                    if (
                        saved.get("identity") != identity
                        or not _valid_result(result, last_row=True)
                        or saved.get("result_digest") != _fingerprint(result)
                    ):
                        continue
                    for metric in ("values", "channels"):
                        amount = result.get("counts", {}).get(metric, 0)
                        snapshot[metric + "_done"] += amount
                        snapshot["folders"][folder][metric + "_done"] += amount
            values = snapshot["values_done"]
            newly_done = max(0, values - self.reused_values)
            done, total = snapshot["units_done"], snapshot["units_total"]
            remaining = sum(
                max(0, f["values_estimated"] - f["values_done"])
                for f in snapshot["folders"].values()
            )
            snapshot.update(
                elapsed_seconds=elapsed,
                updated_unix=time.time(),
                values_reused=self.reused_values,
                values_per_minute=newly_done * 60 / elapsed,
                estimated_seconds_left=0
                if done == total and snapshot["phase"] == "complete"
                else (remaining * elapsed / newly_done if newly_done else None),
            )
            atomic_json(self.path, snapshot, scratch=self.scratch)

    def _heartbeat(self):
        while not self.stop.wait(30):
            self.publish()

    def close(self):
        self.stop.set()
        self.thread.join()
        self.publish()


def prediction(index_bytes, benchmark_series):
    """Empirically anchored planning ranges; no full-cloud speedup is claimed."""
    return dict(
        measured=False,
        values=11_082_499_857,
        channels=13_773_320,
        observed_old_values_per_worker_minute=1_000_000,
        assumed_overall_speedup=[1, 2],
        setup_transfer_hours=[0.5, 1],
        sizing=[
            dict(machine="n2d-standard-32", workers=16, memory_budget_gib=104, hours=[6.3, 12.7]),
            dict(machine="n2d-standard-64", workers=32, memory_budget_gib=208, hours=[3.4, 6.8]),
            dict(machine="n2d-standard-96", workers=48, memory_budget_gib=320, hours=[2.4, 4.9]),
            dict(machine="c3d-standard-16", workers=8, memory_budget_gib=52, hours=[12.2, 24.3]),
        ],
        index_bytes=index_bytes,
        benchmark_series=benchmark_series,
        assumptions=[
            "1M values per worker-minute from sketch1; 1–2x overall speedup is conditional",
            "synthetic gains do not establish full-index, full-corpus throughput",
            "independent row-group scheduling keeps physical cores busy; cloud CPU scaling unmeasured",
            "C3D quota 24 admits standard-16; standard-30 exceeds quota",
        ],
        memory="mapped index charged to each worker; 8 GiB parent reserve; unchanged whole-row reader/watchdog",
        disk="500 GiB pending measured compressed evidence volume",
        cloud_must_confirm=[
            "full-index and controls time",
            "per-shape candidate rate and throughput",
            "evidence volume and transfer rate",
            "RSS and largest-group tail",
            "preemption recovery and status upload",
            "four-hour finish is not established",
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
        "| Folder | Policy | Channels | Verified | Unverified | Fidelity rejected |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for folder in report["folders"]:
        counts = folder.get("counts", {})
        lines.append(
            f"| {folder['folder']} | {folder['status']} | {counts.get('channels', 0)} | {counts.get('verified_matches', 0)} | {counts.get('unverified_candidates', 0)} | {counts.get('fidelity_rejections', 0)} |"
        )
    for case in report["controls"]["failures"]:
        benchmark = case["benchmark"] or {}
        parent = "/".join(str(benchmark.get(k, "")) for k in ("dataset", "item_id", "column"))
        lines.append(
            f"\nFailed control {case['item_id']} ({case['class']}, {parent}): {case['reason']}; "
            f"retrieval={case['retrieval_reasons']}; rejections={case['rejection_reasons']}."
        )
    lines.extend(
        [
            "",
            *LIMITATIONS,
            "",
            "Conditional estimate: 2.4–4.9 hours on n2d-standard-96; full-cloud throughput unverified. See JSON assumptions.",
            "Pairs certain to fail the verifier non-identifying gate are counted explicitly; other admitted pairs are refitted on original values. Numerical similarity is evidence, not proof of provenance.",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_units(
    root,
    out,
    manifest,
    benchmarks,
    *,
    state,
    workers,
    memory_budget_bytes,
    require_pinned,
    nonidentifying_shortcut,
    only_folders,
    skip_finished,
    progress,
):
    import pyarrow.parquet as pq

    spool, benchmark_identity = prepare_benchmarks(
        state, benchmarks, memory_budget_bytes=memory_budget_bytes, require_pinned=require_pinned
    )
    index_dir = state / "sketch-index"
    built = build_index(
        spool, index_dir, benchmark_identity, memory_budget_bytes=memory_budget_bytes
    )
    identities, corpus_bytes, plans = {}, 0, {}
    entries = sorted(manifest["folders"], key=lambda e: e["folder"])
    selected = set(only_folders) if only_folders is not None else {e["folder"] for e in entries}
    unknown = selected - {e["folder"] for e in entries}
    if unknown:
        raise ValueError(f"unknown only-folders: {sorted(unknown)}")
    for entry in entries:
        files = sorted(folder_path(root, entry).glob("*.parquet"))
        if not files:
            raise ValueError(f"missing corpus shards: {entry['folder']}")
        identities[entry["folder"]] = {p.name: digest(p) for p in files}
        units = []
        for shard in files:
            row_offset = 0
            with pq.ParquetFile(shard) as parquet:
                corpus_bytes += parquet.metadata.serialized_size
                for group in range(parquet.num_row_groups):
                    meta = parquet.metadata.row_group(group)
                    corpus_bytes += meta.total_byte_size
                    values = sum(
                        meta.column(i).num_values
                        for i in range(meta.num_columns)
                        if meta.column(i).path_in_schema.split(".")[0] in {"target", "covariates"}
                    )
                    # Bound the scheduling tail as well as the checkpoint loss.
                    # Native reading stays unchanged; a range may re-decode its
                    # group's prefix, but never fingerprints/verifies that prefix.
                    per_row = max(1, math.ceil(values / max(1, meta.num_rows)))
                    step = max(1, min(UNIT_ROWS, UNIT_VALUES // per_row))
                    for start in range(0, meta.num_rows, step):
                        end = min(meta.num_rows, start + step)
                        name = f"{shard.stem}-rg{group:06d}"
                        if start or end != meta.num_rows:
                            name += f"-r{start:09d}-{end:09d}"
                        units.append(
                            dict(
                                id=name,
                                shard=shard.name,
                                row_group=group,
                                row_start=row_offset + start,
                                row_end=row_offset + end,
                                rows=end - start,
                                values=per_row * (end - start),
                            )
                        )
                    row_offset += meta.num_rows
        plans[entry["folder"]] = units
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
    identity["nonidentifying_shortcut"] = nonidentifying_shortcut
    key = _fingerprint(identity)
    atomic_json(
        state / "sketch-input.json",
        dict(identity=identity, fingerprint=key),
        scratch=_workspace(state),
    )
    evidence_dir, unit_dir = out / "evidence", out / "evidence/units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    scratch = _workspace(state)
    reserve = min(8 * 1024**3, memory_budget_bytes // 4)
    minimum = built["bytes"] + 256 * 1024**2 + SERIES_CACHE_BYTES + 3 * SERIES_READ_BYTES
    active = min(workers, max(1, (memory_budget_bytes - reserve) // minimum))
    quota = (memory_budget_bytes - reserve) // active
    if quota < minimum:
        raise ValueError("memory budget cannot admit one sketch worker and the mapped index")
    common = dict(
        directory=str(evidence_dir),
        scratch=str(scratch),
        index=str(index_dir),
        spool=str(spool),
        opportunities=opportunities,
        budget=quota,
        nonidentifying_shortcut=nonidentifying_shortcut,
    )
    controls_marker = state / "sketch-controls.json"
    controls_key = _fingerprint(dict(identity=identity, shortcut=nonidentifying_shortcut))
    cached_controls = _saved_json(controls_marker)
    control_result = _saved_result(controls_marker, controls_key, evidence_dir)
    if control_result is not None and cached_controls.get("controls_digest") == _fingerprint(
        cached_controls.get("controls")
    ):
        controls = cached_controls["controls"]
    else:
        sidecars = scratch / "controls"
        cases, unplantable, skipped = _control_shards(spool, sidecars, opportunities)
        request = dict(
            **common,
            entry=dict(folder="univariate/controls", status="sidecar"),
            folder_path=str(sidecars),
            control=True,
            control_benchmarks={c["item_id"]: c["benchmark"] for c in cases},
        )
        control_result = _execute(request)
        controls = _check_controls(cases, control_result, unplantable, skipped)
        # Failed controls are evidence, never reusable proof of execution.
        if controls["ok"]:
            atomic_json(
                controls_marker,
                dict(
                    identity=controls_key,
                    result=control_result,
                    result_digest=_fingerprint(control_result),
                    controls=controls,
                    controls_digest=_fingerprint(controls),
                ),
                scratch=scratch,
            )
    datasets = {}
    db = _open_spool(spool)
    try:
        for (encoded,) in db.execute("SELECT meta FROM series"):
            meta = json.loads(encoded)
            group = datasets.setdefault(meta["dataset"], dict(frequencies=set(), domains=set()))
            group["frequencies"].add(meta["freq"])
            group["domains"].add(meta["domain"])
    finally:
        db.close()
    completed, requests, results = {}, [], {}
    with progress.lock:
        progress.record.update(
            phase="scanning",
            units_total=sum(len(u) for u in plans.values()),
            folders={
                e["folder"]: dict(
                    units_done=0,
                    units_total=len(plans[e["folder"]]),
                    values_done=0,
                    channels_done=0,
                    values_estimated=sum(u["values"] for u in plans[e["folder"]]),
                )
                for e in entries
            },
        )

    def advance(folder, result, units=1, unit_id=None):
        with progress.lock:
            if unit_id is not None:
                progress.pending.pop((folder, unit_id), None)
            f = progress.record["folders"][folder]
            f["units_done"] += units
            for name in ("values", "channels"):
                amount = result["counts"].get(name, 0)
                f[name + "_done"] += amount
                progress.record[name + "_done"] += amount
            progress.record["units_done"] += units

    for entry in entries:
        folder = entry["folder"]
        marker = evidence_dir / (_journal_name(entry) + ".json")
        finished = _saved_result(marker, key, evidence_dir)
        if finished is not None and (
            only_folders is None or skip_finished or folder not in selected
        ):
            completed[folder] = finished
            advance(folder, finished, len(plans[folder]))
            continue
        if folder not in selected:
            raise ValueError(
                f"unselected folder {folder}: no reusable finished summary with matching identity/evidence"
            )
        if finished is not None:
            # Explicitly rescan a completed selection, including its fragments.
            # Incomplete folders retain bounded-unit resume behavior.
            marker.unlink(missing_ok=True)
            for unit in plans[folder]:
                prefix = _journal_name(entry) + "--" + unit["id"]
                (unit_dir / (prefix + ".json")).unlink(missing_ok=True)
                for part in unit_dir.glob(prefix + "-part*.json"):
                    part.unlink()
        results[folder] = {}
        for unit in plans[folder]:
            request = dict(
                common,
                directory=str(unit_dir),
                entry=entry,
                folder_path=str(folder_path(root, entry)),
                unit=unit,
            )
            unit_key = _fingerprint(
                dict(identity=identity, unit=unit, shortcut=nonidentifying_shortcut)
            )
            saved = _saved_result(
                unit_dir / (_journal_name(entry) + "--" + unit["id"] + ".json"),
                unit_key,
                unit_dir,
                last_row=True,
            )
            if saved is not None:
                results[folder][unit["id"]] = saved
                advance(folder, saved)
            else:
                requests.append((request, unit_key))
    requests.sort(
        key=lambda pair: (
            -pair[0]["unit"]["values"],
            pair[0]["entry"]["folder"],
            pair[0]["unit"]["id"],
        )
    )
    errors = {}
    progress.reused_values = progress.record["values_done"]

    def dispatch(request, unit_key):
        unit_id = request["unit"]["id"]
        prefix = _journal_name(request["entry"]) + "--" + unit_id
        # Apply the worker's exact validation, outside the progress lock.
        # Corrupt/missing fragments will be recomputed and must count as new.
        number, reused = 0, 0
        while True:
            marker = Path(request["directory"]) / (prefix + f"-part{number:06d}.json")
            saved = _saved_result(marker, unit_key, request["directory"], last_row=True)
            if saved is None:
                break
            reused += saved["counts"].get("values", 0)
            number += 1
        with progress.lock:
            progress.pending[(request["entry"]["folder"], unit_id)] = (
                request["directory"],
                prefix,
                unit_key,
            )
            progress.reused_values += reused
        return _checkpoint(request, unit_key)

    with ThreadPoolExecutor(max_workers=active) as pool:
        futures = {
            pool.submit(dispatch, request, unit_key): request for request, unit_key in requests
        }
        for future in as_completed(futures):
            request = futures[future]
            folder = request["entry"]["folder"]
            try:
                result = future.result()
                results[folder][request["unit"]["id"]] = result
                advance(folder, result, unit_id=request["unit"]["id"])
            except (RuntimeError, OSError, ValueError) as exc:
                errors.setdefault(folder, []).append(str(exc))
    folders = []
    for entry in entries:
        folder = entry["folder"]
        if folder in errors:
            folders.append(
                dict(folder=folder, status=entry["status"], error="; ".join(sorted(errors[folder])))
            )
            continue
        if folder in completed:
            result = completed[folder]
        else:
            units = []
            for unit in plans[folder]:
                rec = results[folder][unit["id"]]
                rec = dict(
                    rec, evidence=dict(rec["evidence"], file="units/" + rec["evidence"]["file"])
                )
                units.append(rec)
            result = _assemble(entry, units, evidence_dir, scratch=scratch)
            atomic_json(
                evidence_dir / (_journal_name(entry) + ".json"),
                dict(identity=key, result=result, result_digest=_fingerprint(result)),
                scratch=scratch,
            )
        result["routing"] = routing(
            entry, datasets, result["corpus_frequencies"], result["corpus_sources"]
        )
        folders.append(result)
    return folders, controls, built, benchmark_identity, key


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
    nonidentifying_shortcut=True,
    only_folders=None,
    skip_finished=False,
):
    """Run triage. Exit zero means execution/controls only, NEVER release permission.

    Public CLI verifies published bytes before entering this function. Direct
    calls support offline synthetic trees and still bind all shard bytes.
    """
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
    scratch = _workspace(state)
    evidence = out / "evidence"
    units = evidence / "units"
    units.mkdir(parents=True, exist_ok=True)
    if len({path.stat().st_dev for path in (state, out, scratch, evidence, units)}) != 1:
        raise ValueError("sketch scratch, state and out/evidence must share the same filesystem")
    # Remove stale final reports even if preparation or a worker fails.
    for name in ("sketch-report.json", "sketch-report.md"):
        (out / name).unlink(missing_ok=True)
    progress = _Progress(state)
    try:
        folders, controls, built, benchmark_identity, key = _run_units(
            root,
            out,
            manifest,
            benchmarks,
            state=state,
            workers=workers,
            memory_budget_bytes=memory_budget_bytes,
            require_pinned=require_pinned,
            nonidentifying_shortcut=nonidentifying_shortcut,
            only_folders=only_folders,
            skip_finished=skip_finished,
            progress=progress,
        )
        with progress.lock:
            progress.record["phase"] = (
                "complete" if all("error" not in f for f in folders) else "failed"
            )
    except BaseException:
        with progress.lock:
            progress.record["phase"] = "failed"
        raise
    finally:
        progress.close()
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
            fidelity=dict(
                relative_tolerance=FIDELITY_RTOL,
                minimum_fraction=FIDELITY_FRACTION,
                maps=["least_squares", "identity", "modal_median"],
                regions=["whole", "inside_endpoint", "outside_endpoint"],
            ),
        ),
    )
    atomic_json(out / "sketch-report.json", report, scratch=_workspace(state))
    markdown = _workspace(state) / "sketch-report.md"
    markdown.write_text(_markdown(report))
    markdown.replace(out / "sketch-report.md")
    return report, 0 if execution_ok else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--only-folders",
        nargs="+",
        help="native folder paths; other folders require verified completed summaries",
    )
    parser.add_argument("--skip-finished", action="store_true")
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
        only_folders=args.only_folders,
        skip_finished=args.skip_finished,
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
