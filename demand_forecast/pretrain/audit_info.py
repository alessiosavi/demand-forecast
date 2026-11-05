"""Conditional description lengths for copy identification (score C, version 1).

This is a reproducible description-length diagnostic conditional on the affine
map, length, quantised range and finite mask, not a probability or a provenance
test. Quantisation deliberately preserves the matcher's bin origin and absolute
floor; the score is not affine invariant, so both actual sides must be scored.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

INFO_VERSION = 1
INFO_MARGIN_BITS = 20
RAMP_ULP_FACTOR = 8
STATUSES = ("identifying", "low_information", "interpolation_ramp")
_LOG2 = math.log(2)


@dataclass(frozen=True)
class InfoScore:
    """Winning conditional code (ties use the documented six-code order)."""

    bits: float
    code: str
    strict_ramp: bool
    n: int
    levels: int
    runs: int


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return math.inf
    if k == 0 or k == n:
        return 0.0
    return max(0.0, (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / _LOG2)


def _log_size(n: int) -> float:
    return math.log2(max(1, int(n)))


def _ceil_bits(bits: float) -> int:
    # Only a floating-roundoff guard, not a relaxed coding threshold.
    return math.ceil(max(0.0, bits) - 1e-10)


def _size_bits(n: int) -> int:
    return _ceil_bits(_log_size(n))


@lru_cache(maxsize=50000)
def _level_cost(levels: tuple[int, ...]) -> float:
    """Encode interior levels, or an exactly reconstructible digital lattice.

    Extrema are supplied. The lattice charges denominator, rounding phase and
    occupied positions; integer-valued input never gets free level spacing.
    """
    k, extent = len(levels), levels[-1]
    if k <= 2:
        return 0.0
    best = _ceil_bits(_log_choose(extent - 1, k - 2))
    values = np.asarray(levels)
    for denominator in range(k - 1, extent + 1):
        cost = (
            _size_bits(extent)
            + _size_bits(denominator + 2)
            + _ceil_bits(_log_choose(denominator - 1, k - 2))
        )
        if cost >= best:
            continue
        positions = np.rint(values * denominator / extent).astype(int)
        if len(set(positions)) != k:
            continue
        raw = positions * extent / denominator
        lo = max(-0.5, max(values - raw - 0.5))
        hi = min(0.5, min(values - raw + 0.5))
        if lo < hi and np.array_equal(np.rint(raw + (lo + hi) / 2).astype(int), values):
            best = cost
    return 1.0 + best


def info_bits(values: ArrayLike, positions: ArrayLike | None = None) -> InfoScore:
    """Score the finite sequence with six exact conditional codes.

    ``positions`` are the original aligned indices, used only for the strict
    endpoint-line test when a joint finite mask has gaps. Other codes describe
    the compressed finite sequence conditional on that mask. Nonfinite input
    is removed with its positions. Empty and single-level sequences score zero.
    """
    # Local import avoids a cycle: the matcher calls this diagnostic after fitting.
    from .audit_match import COPY_ATOL, COPY_RTOL, PRECISION_SCALE_FLOOR, ULP32

    v = np.asarray(values, dtype=np.float64)
    pos = np.arange(len(v)) if positions is None else np.asarray(positions)
    if v.ndim != 1 or pos.shape != v.shape:
        raise ValueError("values and positions must be aligned one-dimensional sequences")
    finite = np.isfinite(v)
    v, pos = v[finite], pos[finite]
    n = len(v)
    if not n:
        return InfoScore(0.0, "empty", False, 0, 0, 0)
    if not np.isfinite(pos).all() or np.any(np.diff(pos) <= 0):
        raise ValueError("positions must be finite and strictly increasing")
    tolerance = max(COPY_ATOL, COPY_RTOL * float(np.ptp(v)))
    q = np.rint(v / tolerance).astype(np.int64)
    q -= q.min()
    levels, inverse, counts = np.unique(q, return_inverse=True, return_counts=True)
    k, extent = len(levels), int(q.max())
    runs = 1 + int(np.count_nonzero(np.diff(q)))
    if k < 2:
        return InfoScore(0.0, "constant", True, n, k, runs)

    # Use linspace for contiguous data to reproduce the numerical reference
    # exactly, and original positions for a mask with internal holes.
    if np.all(np.diff(pos) == 1):
        line = np.linspace(v[0], v[-1], n)
    else:
        line = v[0] + (v[-1] - v[0]) * ((pos - pos[0]) / (pos[-1] - pos[0]))
    ramp_tolerance = min(
        tolerance, RAMP_ULP_FACTOR * ULP32 * max(float(np.max(np.abs(v))), PRECISION_SCALE_FLOOR)
    )
    if float(np.max(np.abs(v - line))) <= ramp_tolerance:
        return InfoScore(0.0, "strict_ramp", True, n, k, runs)

    level_bits = _level_cost(tuple(levels.tolist()))
    arrangement = max(
        0.0,
        (math.lgamma(sum(counts) + 1) - sum(math.lgamma(int(c) + 1) for c in counts)) / _LOG2,
    )
    enumerative = (
        _size_bits(n) + level_bits + _ceil_bits(_log_choose(n - 1, k - 1)) + _ceil_bits(arrangement)
    )
    run = (
        2 * _size_bits(n)
        + level_bits
        + _ceil_bits(_log_choose(n - 1, runs - 1))
        + _size_bits(k)
        + (runs - 1) * _size_bits(k - 1)
    )
    knots = int(np.count_nonzero(np.diff(q, n=2)))
    piecewise = (
        _size_bits(n - 1)
        + _ceil_bits(_log_choose(n - 2, knots))
        + (knots + 2) * _size_bits(extent + 1)
    )
    # KT mixture includes uncertainty in every transition row, including self
    # transitions. Free fitted transition counts would erase genuine information.
    transitions = np.zeros((k, k), int)
    np.add.at(transitions, (inverse[:-1], inverse[1:]), 1)
    kt = _log_size(k) + sum(
        (
            math.lgamma(int(row.sum()) + k / 2)
            - math.lgamma(k / 2)
            - sum(math.lgamma(int(c) + 0.5) - math.lgamma(0.5) for c in row)
        )
        / _LOG2
        for row in transitions
    )
    markov = _size_bits(n) + level_bits + _ceil_bits(kt) + 1
    period = next((p for p in range(1, n // 2 + 1) if np.array_equal(q[p:], q[:-p])), None)
    periodic = (
        _size_bits(n) + _size_bits(n // 2) + level_bits + period * _size_bits(k)
        if period is not None
        else math.inf
    )
    # Signed Elias-gamma residuals preserve every deviation from the digital line.
    prediction = np.rint(np.linspace(q[0], q[-1], n)).astype(int)
    residuals = q - prediction
    nonzero = int(np.count_nonzero(residuals))
    endpoints = 2 if {int(q[0]), int(q[-1])} == {0, extent} else 1 + 2 * _size_bits(extent + 1)
    digital_line = (
        endpoints
        + _size_bits(n + 1)
        + _ceil_bits(_log_choose(n, nonzero))
        + sum(2 + 2 * int(math.floor(_log_size(abs(e)))) for e in residuals if e)
    )
    codes = {
        "enumerative": enumerative,
        "run": run,
        "piecewise_linear": piecewise,
        "kt_markov": markov,
        "periodic_motif": periodic,
        "digital_line": digital_line,
    }
    code = min(codes, key=codes.__getitem__)
    return InfoScore(float(3 + codes[code]), code, False, n, k, runs)


def finish_identification(
    counts: dict[str, Any], records: list[dict[str, Any]], index_windows: Mapping[str, int]
) -> dict[str, int | float]:
    """Label copies once, after ordered whole-scan merging and final counters.

    New counters count segment records, not duplicate-expanded copy pairs.
    The top twenty low-information records use stable record-order ties.
    """
    target_total, index_total = sum(counts["windows"].values()), sum(index_windows.values())
    search_bits = math.log2(max(1, target_total)) + math.log2(max(1, index_total))
    threshold = search_bits + INFO_MARGIN_BITS
    low = []
    for status in STATUSES:
        counts[f"{status}_copies"] = 0
    for i, record in enumerate(records):
        if not record["is_copy"]:
            continue
        bits = min(record["info_bits_target"], record["info_bits_index"])
        status = (
            "interpolation_ramp"
            if record["info_strict_ramp"]
            else "identifying"
            if bits >= threshold
            else "low_information"
        )
        record.update(identification=status, info_search_bits=search_bits, info_threshold=threshold)
        counts[f"{status}_copies"] += 1
        if status == "low_information":
            low.append((i, bits))
    keep_low = {i for i, _ in sorted(low, key=lambda item: -item[1])[:20]}
    for i, record in enumerate(records):
        values = record.pop("_info_values", None)
        if values is not None and (record["identification"] == "identifying" or i in keep_low):
            record["values_target"], record["values_index"] = values
    return {
        "version": INFO_VERSION,
        "margin_bits": INFO_MARGIN_BITS,
        "target_windows": target_total,
        "index_windows": index_total,
        "search_bits": search_bits,
    }
