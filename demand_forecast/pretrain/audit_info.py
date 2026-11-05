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
_CODE_ORDER = (
    "enumerative",
    "run",
    "piecewise_linear",
    "kt_markov",
    "periodic_motif",
    "digital_line",
)


@dataclass(frozen=True)
class InfoScore:
    """Winning conditional code (ties use the documented six-code order)."""

    bits: int
    code: str
    strict_ramp: bool
    n: int
    levels: int
    runs: int


def _size_bits(n: int) -> int:
    """Exact ceil(log2(n)), with zero cost for empty/singleton choices."""
    return (max(1, n) - 1).bit_length()


def _choose_bits(n: int, k: int) -> int:
    return _size_bits(math.comb(n, k))


def _ratio_bits(numerator: int, denominator: int) -> int:
    """Exact nonnegative ceil(log2(numerator / denominator))."""
    shift = max(0, numerator.bit_length() - denominator.bit_length())
    return shift + (numerator > denominator << shift)


def _multinomial(counts: list[int]) -> int:
    # Balanced products avoid dividing two enormous, almost equal factorials.
    if len(counts) <= 1:
        return 1
    mid = len(counts) // 2
    left, right = counts[:mid], counts[mid:]
    return math.comb(sum(counts), sum(left)) * _multinomial(left) * _multinomial(right)


def _odd_factorial(n: int) -> int:
    """1 * 3 * ... * (2n-1), including the empty product."""
    return math.factorial(2 * n) // (math.factorial(n) << n)


def _kt_bits(transitions: np.ndarray) -> int:
    # Cancel the powers of two in (k/2)_N / product((1/2)_c).
    # Fold the first-symbol k into the ratio BEFORE taking the single ceiling.
    k = len(transitions)
    numerators, denominators = [k], []
    for row in transitions:
        counts = [int(c) for c in row if c]
        total = sum(counts)
        if not total:
            continue
        if k % 2:
            offset = (k - 1) // 2
            numerator = _odd_factorial(total + offset) // _odd_factorial(offset)
        else:
            offset = k // 2 - 1
            numerator = (math.factorial(total + offset) // math.factorial(offset)) << total
        denominator = math.prod(_odd_factorial(c) for c in counts)
        common = math.gcd(numerator, denominator)
        numerators.append(numerator // common)
        denominators.append(denominator // common)
    return _ratio_bits(math.prod(numerators), math.prod(denominators))


def _smallest_period(values: np.ndarray) -> int | None:
    """Linear prefix function; a final partial motif is allowed."""
    q = values.tolist()
    n = len(q)
    borders = [0] * n
    border = 0
    for i in range(1, n):
        while border and q[i] != q[border]:
            border = borders[border - 1]
        if q[i] == q[border]:
            border += 1
        borders[i] = border
    period = n - border
    return period if 0 < period <= n // 2 else None


def _round_ratio(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return quotient + (
        2 * remainder > denominator or (2 * remainder == denominator and quotient % 2)
    )


@lru_cache(maxsize=256)
def _level_cost(levels: tuple[int, ...]) -> int:
    """Encode interior levels, or an exactly reconstructible digital lattice.

    Extrema are supplied. The lattice charges denominator, rounding phase and
    occupied positions; integer-valued input never gets free level spacing.
    The 256-entry cache bounds diverse 150-level keys to about 0.4 MB (measured),
    while retaining the 24-key working set of the reference fixture.
    """
    k, extent = len(levels), levels[-1]
    if k <= 2:
        return 0
    best = _choose_bits(extent - 1, k - 2)
    for denominator in range(k - 1, extent + 1):
        cost = (
            _size_bits(extent) + _size_bits(denominator + 2) + _choose_bits(denominator - 1, k - 2)
        )
        if cost >= best:
            continue
        positions = [_round_ratio(v * denominator, extent) for v in levels]
        if len(set(positions)) != k:
            continue
        # Phase bounds in units of 1/(2*denominator). A nonempty open
        # interval reconstructs every level without any rounding-boundary ties.
        errors = [2 * (v * denominator - j * extent) for v, j in zip(levels, positions)]
        lo = max(-denominator, max(errors) - denominator)
        hi = min(denominator, min(errors) + denominator)
        if lo < hi:
            best = cost
    return 1 + best


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
        return InfoScore(0, "empty", False, 0, 0, 0)
    if not np.isfinite(pos).all() or np.any(np.diff(pos) <= 0):
        raise ValueError("positions must be finite and strictly increasing")
    tolerance = max(COPY_ATOL, COPY_RTOL * float(np.ptp(v)))
    q = np.rint(v / tolerance).astype(np.int64)
    q -= q.min()
    levels, inverse, counts = np.unique(q, return_inverse=True, return_counts=True)
    k, extent = len(levels), int(q.max())
    runs = 1 + int(np.count_nonzero(np.diff(q)))
    if k < 2:
        return InfoScore(0, "constant", True, n, k, runs)

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
        return InfoScore(0, "strict_ramp", True, n, k, runs)

    level_bits = _level_cost(tuple(levels.tolist()))
    arrangement = _multinomial([int(c) for c in counts])
    enumerative = _size_bits(n) + level_bits + _choose_bits(n - 1, k - 1) + _size_bits(arrangement)
    run = (
        2 * _size_bits(n)
        + level_bits
        + _choose_bits(n - 1, runs - 1)
        + _size_bits(k)
        + (runs - 1) * _size_bits(k - 1)
    )
    knots = int(np.count_nonzero(np.diff(q, n=2)))
    piecewise = (
        _size_bits(n - 1) + _choose_bits(n - 2, knots) + (knots + 2) * _size_bits(extent + 1)
    )
    # KT mixture includes uncertainty in every transition row, including self
    # transitions. Free fitted transition counts would erase genuine information.
    transitions = np.zeros((k, k), int)
    np.add.at(transitions, (inverse[:-1], inverse[1:]), 1)
    markov = _size_bits(n) + level_bits + _kt_bits(transitions) + 1
    period = _smallest_period(q)
    periodic = (
        _size_bits(n) + _size_bits(n // 2) + level_bits + period * _size_bits(k)
        if period is not None
        else math.inf
    )
    # Signed Elias-gamma residuals preserve every deviation from the digital line.
    start, step, span = int(q[0]) * (n - 1), int(q[-1] - q[0]), n - 1
    prediction = np.fromiter(
        (_round_ratio(start + step * i, span) for i in range(n)), dtype=np.int64, count=n
    )
    residuals = q - prediction
    nonzero = int(np.count_nonzero(residuals))
    endpoints = 2 if {int(q[0]), int(q[-1])} == {0, extent} else 1 + 2 * _size_bits(extent + 1)
    digital_line = (
        endpoints
        + _size_bits(n + 1)
        + _choose_bits(n, nonzero)
        + sum(2 * abs(int(e)).bit_length() for e in residuals if e)
    )
    codes = {
        "enumerative": enumerative,
        "run": run,
        "piecewise_linear": piecewise,
        "kt_markov": markov,
        "periodic_motif": periodic,
        "digital_line": digital_line,
    }
    code = min(_CODE_ORDER, key=codes.__getitem__)
    return InfoScore(3 + codes[code], code, False, n, k, runs)


def finish_identification(
    counts: dict[str, Any], records: list[dict[str, Any]], index_windows: Mapping[str, int]
) -> dict[str, int | float]:
    """Label copies once, after ordered whole-scan merging and final counters.

    New counters count segment records, not duplicate-expanded copy pairs.
    The top twenty low-information records use stable record-order ties.
    Status uses integer arithmetic; info_search_bits and info_threshold are
    floating-point display fields only and never decide a declaration.
    """
    target_total, index_total = sum(counts["windows"].values()), sum(index_windows.values())
    search_bits = math.log2(max(1, target_total)) + math.log2(max(1, index_total))
    threshold = search_bits + INFO_MARGIN_BITS
    opportunities = max(1, target_total) * max(1, index_total)
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
            if bits >= INFO_MARGIN_BITS and (1 << (bits - INFO_MARGIN_BITS)) >= opportunities
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
            record["values_target"], record["values_index"] = (v.tolist() for v in values)
    return {
        "version": INFO_VERSION,
        "margin_bits": INFO_MARGIN_BITS,
        "target_windows": target_total,
        "index_windows": index_total,
        "search_bits": search_bits,
    }
