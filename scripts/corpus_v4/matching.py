"""Bounded value matching with unchanged run-12 tolerances and explicit coverage loss."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from demand_forecast.pretrain import audit_info
from demand_forecast.pretrain import audit_match as old
from demand_forecast.pretrain.audit import _offset

CHUNK_POINTS = 65536
OVERLAP = 512
MAX_PHASES = 4096
METHOD_VERSION = "corpus-v4-native-3"


def frequency(alias):
    """Canonical exact offset; unlike v2, do not merge business days with daily data."""
    if alias is None:
        return None
    try:
        offset = _offset(alias)
    except (TypeError, ValueError):
        return None
    if isinstance(offset, pd.offsets.Day):
        return ("fixed", offset.n * 86400 * 10**9)
    if isinstance(offset, pd.offsets.Week):
        return ("fixed", offset.n * 7 * 86400 * 10**9)
    if isinstance(offset, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)):
        return ("calendar", f"{offset.n}ME")
    if isinstance(offset, (pd.offsets.QuarterBegin, pd.offsets.QuarterEnd)):
        return ("calendar", f"{offset.n}QE-DEC")
    if isinstance(offset, (pd.offsets.YearBegin, pd.offsets.YearEnd)):
        return ("calendar", f"{offset.n}YE-DEC")
    if isinstance(offset, pd.offsets.Tick):
        return ("fixed", offset.nanos)
    return ("calendar", offset.freqstr)


def aggregate_fixed(values, ratio, phase, method):
    """Keep only complete bins; a missing input makes the whole bin missing."""
    n = (len(values) - phase) // ratio
    if n <= 0:
        return np.empty(0)
    blocks = np.asarray(values[phase : phase + n * ratio]).reshape(n, ratio)
    return blocks.sum(axis=1) if method == "sum" else blocks.mean(axis=1)


def chunks(values):
    """Overlapping views preserve short identifying segments at chunk boundaries."""
    for start in range(0, len(values), CHUNK_POINTS - OVERLAP):
        part = values[start : start + CHUNK_POINTS]
        if len(part):
            yield start, part
        if start + CHUNK_POINTS >= len(values):
            break


def _repetitive(values):
    values = np.asarray(values)
    tolerance = max(old.COPY_ATOL, old.COPY_RTOL * float(np.ptp(values)))
    for period in range(1, min(32, len(values) // 2) + 1):
        if np.all(np.abs(values[period:] - values[:-period]) <= tolerance):
            return True
    return False


MAX_POSTINGS = 1_000_000
MAX_INDEX_POINTS = 262144


def identifying_parent(values):
    """Constants, short motifs and strict ramps cannot identify aggregated ancestry."""
    x = np.asarray(values, np.float64)
    finite = np.isfinite(x)
    v = x[finite]
    if len(v) < old.SEG_MIN or not old.nondegenerate(v)[0] or _repetitive(v):
        return False
    positions = np.flatnonzero(finite)
    line = v[0] + (v[-1] - v[0]) * ((positions - positions[0]) / (positions[-1] - positions[0]))
    tolerance = 8 * old.ULP32 * max(float(np.max(np.abs(v))), old.PRECISION_SCALE_FLOOR)
    return bool(np.max(np.abs(v - line)) > tolerance)


def _posting_budget(x, index):
    """Count ALL postings before the old Python expansion, including fallback hits."""
    total = 0
    for family, fn in old.FAMILIES.items():
        hashes, _, _ = fn(x)
        indexed = index.by[family][0]
        total += int(
            (
                np.searchsorted(indexed, hashes, side="right")
                - np.searchsorted(indexed, hashes, side="left")
            ).sum()
        )
        if total > MAX_POSTINGS:
            return False
    return True


def find_index_matches(x, index, *, opportunities=1 << 60, calendar_dimensions=None):
    """Search a bounded multi-series index with positional missingness seed strides."""
    x = np.asarray(x, np.float64)
    coverage = dict(
        unresolved_rows=0,
        groups_dropped=0,
        offsets_dropped=0,
        low_information=0,
        repetitive_rejected=0,
        posting_budget_exceeded=0,
    )
    if np.isfinite(x).sum() < old.SEG_MIN or not old.nondegenerate(x)[0]:
        return [], coverage
    matches, seen = [], set()
    missing = not np.isfinite(x).all() or any(not np.isfinite(y).all() for y in index.series)
    for stride in (1, 2, 3, 4) if missing else (1,):
        for yp in range(stride):
            idx = (
                index
                if stride == 1
                else old.Index([y[yp::stride] for y in index.series], index.ids)
            )
            for xp in range(stride):
                xx = x[xp::stride]
                if not _posting_budget(xx, idx):
                    coverage["unresolved_rows"] += 1
                    coverage["posting_budget_exceeded"] += 1
                    continue
                result = old.scan_group("native", "values", 0, [0], [xx], idx)
                counts = result["counters"]
                for key in ("unresolved_rows", "groups_dropped", "offsets_dropped"):
                    coverage[key] += counts[key]
                for candidate in result["pairs"]:
                    if not candidate["is_copy"]:
                        continue
                    owner = candidate["index_series_index"]
                    y = index.series[owner]
                    off = xp + stride * candidate["offset"] - yp
                    a0 = xp + stride * candidate["t_start"]
                    a1 = min(len(x), xp + stride * (candidate["t_end"] - 1) + 1)
                    rec = old.fit_segment(x, y, a0, a1, off)
                    if rec is None or not rec["is_copy"]:
                        continue
                    rec.pop("_info_values", None)
                    key = (owner, off, rec["t_start"], rec["t_end"])
                    if key in seen:
                        continue
                    seen.add(key)
                    finite = np.isfinite(x[rec["t_start"] : rec["t_end"]]) & np.isfinite(
                        y[rec["index_start"] : rec["index_end"]]
                    )
                    a = x[rec["t_start"] : rec["t_end"]][finite]
                    b = y[rec["index_start"] : rec["index_end"]][finite]
                    dimensions = None if calendar_dimensions is None else calendar_dimensions[owner]
                    if dimensions is not None and old.calendar_pattern(
                        b, dimensions[rec["index_start"] : rec["index_end"]][finite]
                    ):
                        coverage["repetitive_rejected"] += 1
                        continue
                    if _repetitive(a) or _repetitive(b):
                        coverage["repetitive_rejected"] += 1
                        continue
                    bits = min(rec["info_bits_target"], rec["info_bits_index"])
                    if (
                        rec["info_strict_ramp"]
                        or bits < audit_info.INFO_MARGIN_BITS
                        or (1 << (bits - audit_info.INFO_MARGIN_BITS)) < opportunities
                    ):
                        coverage["low_information"] += 1
                        continue
                    rec.update(
                        offset=off,
                        index_series_index=owner,
                        seed_stride=stride,
                        info_threshold=audit_info.INFO_MARGIN_BITS
                        + math.log2(max(1, opportunities)),
                    )
                    matches.append(rec)
    return matches, coverage


def find_matches(x, y, *, opportunities=1 << 60, index=None):
    """Single-series convenience interface used by planted controls and tests."""
    idx = index if index is not None else old.Index([np.asarray(y, np.float64)], ["benchmark"])
    return find_index_matches(x, idx, opportunities=opportunities)
