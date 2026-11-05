"""Run-11 value matcher, ported from the first leakage audit, specification §3b.

Index values are y, benchmark values are x; maps are y = slope*x + intercept.
Bounds are zero-based, half-open. All four families apply to every row. This is
an empirical heuristic, not a proof of provenance or absence of overlap.
IO and scheduling live in audit.py. Controls live in audit_controls.py.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from . import audit_info

WIN, RWIN, RWIN8, MIN_HITS, SEG_MIN = (8, 4, 8, 3, 24)
CAP = {"exact": 1024, "affine": 1024, "runs": 64, "runs8": 8192}
FALLBACK_TOPK, FALLBACK_CAP, MAX_RECORDS, MAX_RECORDS_PER_REASON = (512, 1 << 40, 60000, 3000)
ROW_GROUP_CAP = 2048
VERIFY_OFFSET_CAP = 128
ULP32, ULP_FACTOR = (2.0 ** (-23), 64.0)
MOD = np.uint64(1 << 61) - np.uint64(1)
COPY_ATOL = 0.01
COPY_RTOL = 0.005
EXACT_ATOL = 0.005
EXACT_RTOL = 1e-06
EXACT_FRACTION = 0.99
NEAR_R2 = 0.999
REFINE_SLACK = 2.0
EXACT_SCALE = 100.0
AFFINE_SCALE = 10.0
RUN_LENGTH_CLIP = 1000000
HASH_BASE = 1000003
HASH_BIAS = 7
AFFINE_QUANT_BIAS = 64
CANONICAL_EPSILON = 1e-9
EXACT_MIN_LEVELS = 2
AFFINE_MIN_LEVELS = 3
MAP_KEY_DECIMALS = 1
R2_DENOMINATOR_FLOOR = 1e-12
PRECISION_SCALE_FLOOR = 1e-300
DUPLICATE_EXACT_DECIMALS = 6
DUPLICATE_AFFINE_DECIMALS = 3


def polyhash(w):
    h = np.zeros(w.shape[0], dtype=np.uint64)
    for k in range(w.shape[1]):
        h = (h * np.uint64(HASH_BASE) + w[:, k] + np.uint64(HASH_BIAS)) % MOD
    return h


def _canon_quant(win):
    mean = win.mean(axis=1, keepdims=True)
    std = win.std(axis=1, keepdims=True)
    z = (win - mean) / std
    first = np.argmax(np.abs(z) > CANONICAL_EPSILON, axis=1)
    sign = np.sign(z[np.arange(len(z)), first])
    sign[sign == 0] = 1
    z = z * sign[:, None]
    return (np.rint(z * AFFINE_SCALE).astype(np.int64) + AFFINE_QUANT_BIAS).astype(np.uint64)


def exact_hashes(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x) - WIN + 1
    if n <= 0:
        return (np.empty(0, np.uint64), np.empty(0, np.int64), np.empty(0, np.int64))
    q = np.rint(x * EXACT_SCALE)
    finite = np.isfinite(q)
    qi = np.where(finite, q, 0).astype(np.int64).astype(np.uint64)
    win = np.lib.stride_tricks.sliding_window_view(qi, WIN)
    fin = np.lib.stride_tricks.sliding_window_view(finite, WIN).all(axis=1)
    distinct = 1 + (np.diff(np.sort(win, axis=1), axis=1) != 0).sum(axis=1)
    pos = np.nonzero(fin & (distinct >= EXACT_MIN_LEVELS))[0]
    return (polyhash(win[pos]), pos, np.full(len(pos), WIN, np.int64))


def affine_hashes(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x) - WIN + 1
    if n <= 0:
        return (np.empty(0, np.uint64), np.empty(0, np.int64), np.empty(0, np.int64))
    finite = np.isfinite(x)
    win = np.lib.stride_tricks.sliding_window_view(np.where(finite, x, 0.0), WIN)
    fin = np.lib.stride_tricks.sliding_window_view(finite, WIN).all(axis=1)
    std = win.std(axis=1)
    distinct = 1 + (np.diff(np.sort(win, axis=1), axis=1) != 0).sum(axis=1)
    pos = np.nonzero(fin & (std > 0) & (distinct >= AFFINE_MIN_LEVELS))[0]
    if len(pos) == 0:
        return (np.empty(0, np.uint64), pos, np.empty(0, np.int64))
    return (polyhash(_canon_quant(win[pos])), pos, np.full(len(pos), WIN, np.int64))


def runs_hashes(x, rw=None):
    rw = RWIN if rw is None else rw
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < rw:
        return (np.empty(0, np.uint64), np.empty(0, np.int64), np.empty(0, np.int64))
    finite = np.isfinite(x)
    change = np.ones(n, bool)
    change[1:] = (x[1:] != x[:-1]) | (finite[1:] != finite[:-1])
    starts = np.flatnonzero(change)
    lengths = np.diff(np.append(starts, n))
    levels = x[starts]
    fin_runs = finite[starts]
    m = len(starts) - rw + 1
    if m <= 0:
        return (np.empty(0, np.uint64), np.empty(0, np.int64), np.empty(0, np.int64))
    span_length = np.lib.stride_tricks.sliding_window_view(lengths, rw)
    values = np.lib.stride_tricks.sliding_window_view(levels, rw)
    finite_runs = np.lib.stride_tricks.sliding_window_view(fin_runs, rw).all(axis=1)
    std = values.std(axis=1)
    ok = finite_runs & (std > 0)
    pos = np.nonzero(ok)[0]
    if len(pos) == 0:
        return (np.empty(0, np.uint64), pos, np.empty(0, np.int64))
    q = np.concatenate(
        [
            np.minimum(span_length[pos], RUN_LENGTH_CLIP).astype(np.uint64),
            _canon_quant(values[pos]),
        ],
        axis=1,
    )
    span = span_length[pos].sum(axis=1)
    return (polyhash(q), starts[pos], span)


FAMILIES = {
    "exact": exact_hashes,
    "affine": affine_hashes,
    "runs": runs_hashes,
    "runs8": lambda x: runs_hashes(x, RWIN8),
}


def weekly_sums(x, start, anchor):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    dow0 = pd.Timestamp(start).day_of_week
    first = (anchor - dow0) % 7 + 1
    m = (n - first) // 7
    if m < WIN:
        return None
    blk = x[first : first + 7 * m].reshape(m, 7)
    s = blk.sum(axis=1)
    s[~np.isfinite(blk).all(axis=1)] = np.nan
    return s


def monthly_aggs(x, start):
    x = np.asarray(x, dtype=np.float64)
    idx = pd.date_range(pd.Timestamp(start), periods=len(x), freq="D")
    s = pd.Series(x, index=idx)
    g = s.groupby([idx.year, idx.month])
    days = g.size()
    full = (
        days.values
        == pd.PeriodIndex([pd.Period(f"{y}-{m:02d}", "M") for y, m in days.index]).days_in_month
    )
    cnt = g.count().values
    sums = g.sum(min_count=1).values
    means = g.mean().values
    ok = full & (cnt == days.values)
    sums = np.where(ok, sums, np.nan)
    means = np.where(ok, means, np.nan)
    dim = np.array(
        [pd.Period(f"{y}-{m:02d}", "M").days_in_month for y, m in days.index], np.float64
    )
    if ok.sum() < WIN:
        return (None, None, None)
    return (sums, means, dim)


def days_in_month_of(timestamps):
    return np.array([pd.Timestamp(t).days_in_month for t in timestamps], np.float64)


class Index:
    def __init__(self, series_list, ids):
        self.series, self.ids, self.by = (series_list, ids, {})
        self.cov = {
            "n": len(series_list),
            "nan_values": int(sum(np.isnan(s).sum() for s in series_list)),
            "no_window": {},
            "degenerate": 0,
        }
        anyw = np.zeros(len(series_list), bool)
        for fam, fn in FAMILIES.items():
            hs, owner, pos, per_series = ([], [], [], [])
            for i, s in enumerate(series_list):
                h, p, _ = fn(s)
                hs.append(h)
                owner.append(np.full(len(h), i, np.int64))
                pos.append(p)
                per_series.append(h)
                if len(h):
                    anyw[i] = True
            hs = np.concatenate(hs) if hs else np.empty(0, np.uint64)
            owner = np.concatenate(owner) if owner else np.empty(0, np.int64)
            pos = np.concatenate(pos) if pos else np.empty(0, np.int64)
            o = np.argsort(hs, kind="stable")
            hs, owner, pos = (hs[o], owner[o], pos[o])
            _, inv, cnt = np.unique(hs, return_inverse=True, return_counts=True)
            mult = cnt[inv]
            self.by[fam] = (hs, owner, pos, mult)
            self.cov["no_window"][fam] = int(sum(1 for h in per_series if len(h) == 0))
            if fam in ("exact", "affine"):
                self.cov[f"series_all_{fam}_windows_above_cap"] = int(
                    sum(
                        1
                        for h in per_series
                        if len(h) and bool((mult[np.searchsorted(hs, h)] > CAP[fam]).all())
                    )
                )
            self.cov[f"windows_above_cap_share_{fam}"] = (
                float((mult > CAP[fam]).mean()) if len(mult) else 0.0
            )
            self.cov[f"max_multiplicity_{fam}"] = int(mult.max()) if len(mult) else 0
        self.cov["windows"] = {fam: len(arrays[0]) for fam, arrays in self.by.items()}
        self.cov["no_window_any_family"] = [ids[i] for i in np.flatnonzero(~anyw)]
        self.cov["degenerate"] = int(
            sum(1 for s in series_list if len(np.unique(s[np.isfinite(s)])) < 3)
        )
        ex = defaultdict(int)
        af = defaultdict(int)
        for s in series_list:
            f = s[np.isfinite(s)]
            ex[hashlib.sha1(np.round(f, DUPLICATE_EXACT_DECIMALS).tobytes()).hexdigest()] += 1
            if len(f) and f.std() > 0:
                z = (f - f.mean()) / f.std()
                z = z * (1 if z[np.argmax(np.abs(z) > CANONICAL_EPSILON)] >= 0 else -1)
                af[hashlib.sha1(np.round(z, DUPLICATE_AFFINE_DECIMALS).tobytes()).hexdigest()] += 1
        self.cov["duplicate_series_exact"] = int(sum(c - 1 for c in ex.values() if c > 1))
        self.cov["duplicate_series_affine"] = int(sum(c - 1 for c in af.values() if c > 1))

    def lookup(self, fam, h, cap):
        """(positions into h, positions into the index) for informative hits; dropped count; raw hit count."""
        hs, owner, pos, mult = self.by[fam]
        if len(h) == 0 or len(hs) == 0:
            return (np.empty(0, np.int64), np.empty(0, np.int64), 0, 0)
        i = np.searchsorted(hs, h)
        i[i >= len(hs)] = 0
        hit = hs[i] == h
        informative = hit & (mult[i] <= cap)
        return (
            np.nonzero(informative)[0],
            i[informative],
            int((hit & ~informative).sum()),
            int(hit.sum()),
        )


MIN_OFF_MODAL, MIN_DOF = (4, 7)


def nondegenerate(v):
    """(ok, reason). Precision rule: the segment's range must exceed 64 float32 ulps of its magnitude
    (float32 noise around a constant is ~1 ulp). Information rules (policy, run 7): at least MIN_OFF_MODAL points
    must lie off the segment's modal level (a constant with <= 3 excursions carries <= 3 step positions and at most one
    free gap ratio; run 5 matched eight such segments by chance across three unrelated datasets), and the segment must
    carry >= MIN_DOF degrees of freedom beyond the affine map: (runs - 1) change positions + (distinct - 2) level ratios."""
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return (False, "short")
    if len(np.unique(v)) < 2:
        return (False, "constant")
    rng = float(np.ptp(v))
    scale = float(np.abs(v).max())
    if rng <= ULP_FACTOR * ULP32 * max(scale, PRECISION_SCALE_FLOOR):
        return (False, "precision")
    q = np.round(v / max(COPY_ATOL, COPY_RTOL * rng))
    vals, counts = np.unique(q, return_counts=True)
    distinct = len(vals)
    if distinct < 2:
        return (False, "precision")
    if len(q) - int(counts.max()) < MIN_OFF_MODAL:
        return (False, "off_modal_lt4")
    runs = 1 + int((q[1:] != q[:-1]).sum())
    if runs - 1 + (distinct - 2) < MIN_DOF:
        return (False, "low_dof")
    return (True, None)


def _info_degenerate(q):
    """Off-modal and degrees-of-freedom rules on an already quantised series (True = non-identifying)."""
    if np.ptp(q) == 0:
        return True
    vals, counts = np.unique(q, return_counts=True)
    runs = 1 + int((q[1:] != q[:-1]).sum())
    return len(q) - int(counts.max()) < MIN_OFF_MODAL or runs - 1 + (len(vals) - 2) < MIN_DOF


def calendar_pattern(a, d):
    """True when the target segment `a` carries no information beyond the calendar: both the residual from the affine fit on
    days-in-month `d` and the per-day rate a / d, quantised to the copy tolerance, are checked; if either is degenerate under the
    off-modal / degrees-of-freedom rules the segment is a calendar pattern (plus at most a few steps)."""
    if d is None or len(a) < 3 or np.ptp(d) == 0 or (np.ptp(a) == 0):
        return False
    tol = max(COPY_ATOL, COPY_RTOL * float(np.ptp(a)))
    c = np.polyfit(d, a, 1)
    resid = a - (c[0] * d + c[1])
    if _info_degenerate(np.round(resid / tol)):
        return True
    for base in (0.0, float(c[1])):
        rate = (a - base) / d
        if _info_degenerate(np.round(rate / (tol / float(d.mean())))):
            return True
    dd = np.diff(d)
    da = np.diff(a)
    m = dd != 0
    if m.sum() >= 3 and _info_degenerate(np.round(da[m] / dd[m] / tol)):
        return True
    return False


def affine_key(t, s, tp, sp, span_length, fam, scale):
    """Coarse key of the affine map implied by one matched window (groups hits with the same map)."""
    if fam == "exact":
        return ("exact",)
    a = t[tp : tp + span_length]
    b = s[sp : sp + span_length]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0:
        return None
    c = np.polyfit(a[m], b[m], 1)
    ka = (
        (int(np.sign(c[0])), round(float(np.log10(abs(c[0]))), MAP_KEY_DECIMALS))
        if c[0] != 0
        else (0, 0.0)
    )
    return (ka, round(float(c[1]) / scale, MAP_KEY_DECIMALS))


def intervals_of(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def merge_spans(spans: Iterable[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    """Union half-open spans in start order, with the same bounds as mask slices.

    Overlapping and touching spans form one interval. Slice normalization also
    clips overruns and discards empty spans without allocating a length-sized mask.
    """
    clipped = []
    for start, end in spans:
        start, end, _ = slice(start, end).indices(length)
        if start < end:
            clipped.append((start, end))
    merged = []
    for start, end in sorted(clipped):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def fit_segment(t, s, a0, a1, off, dim=None):
    b0, b1 = (a0 - off, a1 - off)
    if b0 < 0:
        a0, b0 = (a0 - b0, 0)
    if b1 > len(s):
        a1, b1 = (a1 - (b1 - len(s)), len(s))
    if a1 - a0 < WIN:
        return None
    a, b = (t[a0:a1], s[b0:b1])
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < WIN:
        return None
    a, b = (a[both], b[both])
    n = int(both.sum())
    exact_frac = float(np.isclose(a, b, atol=EXACT_ATOL, rtol=EXACT_RTOL).mean())
    if a.std() > 0 and b.std() > 0:
        coef = np.polyfit(a, b, 1)
        resid = b - (coef[0] * a + coef[1])
        r2 = float(1 - (resid**2).sum() / max(((b - b.mean()) ** 2).sum(), R2_DENOMINATOR_FLOOR))
        maxres = float(np.abs(resid).max())
    else:
        coef = [0.0, 0.0]
        r2 = 0.0
        maxres = float("inf")
    rng = float(np.ptp(b)) if len(b) else 0.0
    nd_a, why_a = nondegenerate(a)
    nd_b, why_b = nondegenerate(b)
    nd = nd_a and nd_b
    if nd and dim is not None and calendar_pattern(a, dim[a0:a1][both]):
        nd, why_a, why_b = (False, "calendar_pattern", "calendar_pattern")
    fits = n >= SEG_MIN and (
        exact_frac >= EXACT_FRACTION or maxres <= max(COPY_ATOL, COPY_RTOL * rng)
    )
    result = {
        "segment": n,
        "r2": r2,
        "max_resid": maxres,
        "index_range": rng,
        "exact_frac": exact_frac,
        "coef": [float(c) for c in coef],
        "is_copy": bool(fits and nd),
        "nondegenerate": bool(nd),
        "degenerate_copy_rejected": bool(fits and (not nd)),
        "degenerate_reason": None if nd else [why_a, why_b],
        "target_distinct": int(len(np.unique(a))),
        "index_distinct": int(len(np.unique(b))),
        "t_start": int(a0),
        "t_end": int(a1),
        "index_start": int(b0),
        "index_end": int(b1),
    }

    if result["is_copy"]:
        positions = np.flatnonzero(both)
        target_info = audit_info.info_bits(a, positions)
        index_info = audit_info.info_bits(b, positions)
        result.update(
            info_bits_target=target_info.bits,
            info_bits_index=index_info.bits,
            info_code_target=target_info.code,
            info_code_index=index_info.code,
            info_strict_ramp=target_info.strict_ramp or index_info.strict_ramp,
            info_n=target_info.n,
            info_levels_target=target_info.levels,
            info_levels_index=index_info.levels,
            info_runs_target=target_info.runs,
            info_runs_index=index_info.runs,
        )
        if not result["info_strict_ramp"]:
            # Own bounded buffers: views would retain the complete aligned arrays.
            # Lists are materialised only for records selected after the scan.
            result["_info_values"] = (
                a[:512].astype(np.float64, copy=True),
                b[:512].astype(np.float64, copy=True),
            )
    return result


def refine_pieces(t, s, a0, a1, off, slack):
    """Left-to-right piecewise-affine growth over [a0, a1): maximal pieces that a single affine map fits within `slack`
    (incremental least squares; the initial 8-point window must itself fit). Finds spliced copies whose maps share a coarse key."""
    pieces = []
    i = a0
    while a1 - i >= WIN:
        idx = []
        for k in range(i, a1):
            if np.isfinite(t[k]) and np.isfinite(s[k - off]):
                idx.append(k)
            if len(idx) == WIN:
                break
        if len(idx) < WIN:
            break
        a = t[idx]
        b = s[np.asarray(idx) - off]
        if a.std() == 0:
            i = idx[0] + 1
            continue
        n = float(len(a))
        sum_a, sum_b, sum_aa, sum_ab = (a.sum(), b.sum(), (a * a).sum(), (a * b).sum())

        def coef():
            den = n * sum_aa - sum_a * sum_a
            return (
                ((n * sum_ab - sum_a * sum_b) / den, (sum_b * sum_aa - sum_a * sum_ab) / den)
                if den > 0
                else (0.0, sum_b / n)
            )

        c0, c1 = coef()
        if np.abs(b - (c0 * a + c1)).max() > slack:
            i = idx[0] + 1
            continue
        j = idx[-1] + 1
        while j < a1:
            tj, sj = (t[j], s[j - off])
            if not (np.isfinite(tj) and np.isfinite(sj)):
                j += 1
                continue
            if abs(sj - (c0 * tj + c1)) > slack:
                break
            n += 1
            sum_a += tj
            sum_b += sj
            sum_aa += tj * tj
            sum_ab += tj * sj
            c0, c1 = coef()
            j += 1
        pieces.append((idx[0], j))
        i = j
    return pieces


def verify(t, s, hits, dim=None, stats=None):
    """hits: list of (fam, t_pos, s_pos, span). Returns (best record, notable records)."""
    offsets = defaultdict(list)
    for fam, tp, sp, span_length in hits:
        offsets[int(tp - sp)].append((int(tp), int(span_length), fam, int(sp)))
    if len(offsets) > VERIFY_OFFSET_CAP:
        if stats is not None:
            stats["offsets_capped_candidates"] = stats.get("offsets_capped_candidates", 0) + 1
            stats["offsets_dropped"] = (
                stats.get("offsets_dropped", 0) + len(offsets) - VERIFY_OFFSET_CAP
            )
        offsets = dict(
            sorted(offsets.items(), key=lambda item: (-len(item[1]), item[0]))[:VERIFY_OFFSET_CAP]
        )
    best, notable = (None, [])
    fin_s = s[np.isfinite(s)]
    scale = max(1.0, float(np.abs(fin_s).max())) if len(fin_s) else 1.0
    for off, lst in offsets.items():
        if len(lst) < MIN_HITS and sum((span_length for _, span_length, _, _ in lst)) < SEG_MIN:
            continue
        fams = sorted({f for _, _, f, _ in lst})
        todo = [
            ("merged", a0, a1)
            for a0, a1 in merge_spans(
                ((tp, tp + span_length) for tp, span_length, _, _ in lst), len(t)
            )
        ]
        groups = defaultdict(list)
        for tp, span_length, fam, sp in lst:
            k = affine_key(t, s, tp, sp, span_length, fam, scale)
            if k is not None:
                groups[k].append((tp, span_length))
        if len(groups) > 1:
            for items in groups.values():
                todo.extend(
                    ("grouped", a0, a1)
                    for a0, a1 in merge_spans(
                        ((tp, tp + span_length) for tp, span_length in items), len(t)
                    )
                )
        seen = set()
        queue = list(todo)
        while queue:
            mode, a0, a1 = queue.pop(0)
            if (a0, a1) in seen:
                continue
            seen.add((a0, a1))
            rec = fit_segment(t, s, a0, a1, off, dim)
            if rec is None:
                continue
            if not rec["is_copy"] and mode in ("merged", "grouped") and (a1 - a0 >= 2 * WIN):
                b_fin = s[max(0, a0 - off) : max(0, a1 - off)]
                b_fin = b_fin[np.isfinite(b_fin)]
                slack = (
                    REFINE_SLACK * max(COPY_ATOL, COPY_RTOL * float(np.ptp(b_fin)))
                    if len(b_fin) > 1
                    else REFINE_SLACK * COPY_ATOL
                )
                for p0, p1 in refine_pieces(t, s, a0, a1, off, slack):
                    if p1 - p0 >= WIN and (p0, p1) != (a0, a1):
                        queue.append(("piece", p0, p1))
            rec.update(
                {"offset": off, "families": fams, "hits_at_offset": len(lst), "fit_mode": mode}
            )
            key = (
                rec["is_copy"],
                rec["segment"] if rec["is_copy"] else 0,
                rec["r2"] * rec["segment"] if rec["nondegenerate"] else -1.0,
            )
            if best is None or key > best[0]:
                best = (key, rec)
            if (
                rec["degenerate_copy_rejected"]
                or rec["is_copy"]
                or (rec["nondegenerate"] and rec["segment"] >= SEG_MIN and (rec["r2"] >= NEAR_R2))
            ):
                notable.append(rec)
    return (best[1] if best else None, notable)


def new_cov():
    return {
        "n_series": 0,
        "len_lt8": 0,
        "no_8_consecutive_finite": 0,
        "no_window_any_family": 0,
        "degenerate_lt3_levels": 0,
        "no_window_unexplained": 0,
        "unexplained_ids": [],
        "with": {f: 0 for f in FAMILIES},
        "windows": {f: 0 for f in FAMILIES},
        "raw_hits": {f: 0 for f in FAMILIES},
        "hits": {f: 0 for f in FAMILIES},
        "hits_uninformative": {f: 0 for f in FAMILIES},
        "rows_with_informative_hits": 0,
        "masked_rows": 0,
        "fallback_rows": 0,
        "fallback_candidates": 0,
        "unresolved_rows": 0,
        "truncated_rows": 0,
        "fallback_no_candidate_rows": 0,
        "fallback_oversize_rows": 0,
        "groups_capped_rows": 0,
        "groups_dropped": 0,
        "offsets_capped_candidates": 0,
        "offsets_dropped": 0,
        "candidates": 0,
        "verified": 0,
        "copies": 0,
        "degenerate_rejections": 0,
        "rejection_reasons": {
            "precision": 0,
            "off_modal_lt4": 0,
            "low_dof": 0,
            "calendar_pattern": 0,
            "constant": 0,
            "short": 0,
        },
        "near_pairs": 0,
        "max_r2_segment24": 0.0,
        "uncovered_ids": [],
    }


def _expand_arrays(index, fam, pos, span, hp, ip):
    """Vectorised expansion of every posting of every hit: arrays (key, tp, sp, L) with key = owner * 2^32 + (offset + 2^31)."""
    hs, owner, mpos, _ = index.by[fam]
    hv = hs[ip]
    left = np.searchsorted(hs, hv, side="left")
    right = np.searchsorted(hs, hv, side="right")
    counts = right - left
    tot = int(counts.sum())
    if tot == 0:
        return None
    starts = np.repeat(left, counts)
    within = np.arange(tot) - np.repeat(np.cumsum(counts) - counts, counts)
    idx = starts + within
    tp = np.repeat(pos[hp], counts).astype(np.int64)
    span_length = np.repeat(span[hp], counts).astype(np.int64)
    own = owner[idx].astype(np.int64)
    sp = mpos[idx].astype(np.int64)
    return (own * (1 << 32) + (tp - sp + (1 << 31)), tp, sp, span_length)


FALLBACK_MAX_EXPANSIONS = 50000000


def _expand(groups, index, fam, pos, span, hp, ip):
    hs, owner, mpos, _ = index.by[fam]
    for p_, span_length, j in zip(pos[hp], span[hp], ip):
        k = j
        while k < len(hs) and hs[k] == hs[j]:
            groups[int(owner[k]), int(p_) - int(mpos[k])].append(
                (fam, int(p_), int(mpos[k]), int(span_length))
            )
            k += 1


def _row_key(x):
    """Float64 bytes with canonical NaNs; keep infinities and signed zeros distinct."""
    values = np.array(x, dtype="<f8", copy=True)
    values[np.isnan(values)] = np.nan
    return hashlib.sha256(values.tobytes()).digest() + len(x).to_bytes(8, "little")


def _merge(dst, src, mult):
    """Add the single-row counters `src` into `dst` for `mult` identical rows (max for the r2 statistic; lists handled by the caller)."""
    for k, v in src.items():
        if isinstance(v, dict):
            _merge(dst[k], v, mult)
        elif isinstance(v, list):
            continue
        elif k == "max_r2_segment24":
            dst[k] = max(dst[k], v)
        else:
            dst[k] += v * mult


def _scan_row(name, column, i, x, index, c, pairs, records, extra, dim=None, *, uncapped=False):
    """Scan one row `x` (float64), counting into `c`; uncapped lifts only FALLBACK_TOPK."""
    c["n_series"] += 1
    fin = np.isfinite(x)
    if len(x) < WIN:
        c["len_lt8"] += 1
    runs8 = (
        np.lib.stride_tricks.sliding_window_view(fin, WIN).all(axis=1).any()
        if len(x) >= WIN
        else False
    )
    if not runs8:
        c["no_8_consecutive_finite"] += 1
    n_distinct = len(np.unique(x[fin]))
    if n_distinct < 3:
        c["degenerate_lt3_levels"] += 1
    anyw = False
    groups = defaultdict(list)
    raw_total = informative_total = 0
    hashes = {}
    for fam in FAMILIES:
        h, pos, span = FAMILIES[fam](x)
        hashes[fam] = (h, pos, span)
        if len(h):
            c["with"][fam] += 1
            c["windows"][fam] += len(h)
            anyw = True
        hp, ip, dropped, raw = index.lookup(fam, h, CAP[fam])
        c["hits_uninformative"][fam] += dropped
        c["raw_hits"][fam] += raw
        raw_total += raw
        informative_total += len(hp)
        if len(hp):
            c["hits"][fam] += len(hp)
            _expand(groups, index, fam, pos, span, hp, ip)
    if informative_total:
        c["rows_with_informative_hits"] += 1
    if raw_total and (not informative_total):
        c["masked_rows"] += 1
        lookups = []
        tot = 0
        for fi, fam in enumerate(FAMILIES):
            h, pos, span = hashes[fam]
            hp, ip, _, _ = index.lookup(fam, h, FALLBACK_CAP)
            hs = index.by[fam][0]
            if len(hp):
                hv = hs[ip]
                tot += int(
                    (
                        np.searchsorted(hs, hv, side="right") - np.searchsorted(hs, hv, side="left")
                    ).sum()
                )
                lookups.append((fi, fam, pos, span, hp, ip))
        if tot > FALLBACK_MAX_EXPANSIONS:
            c["fallback_oversize_rows"] += 1
            c["unresolved_rows"] += 1
        elif tot:
            parts = []
            for fi, fam, pos, span, hp, ip in lookups:
                arr = _expand_arrays(index, fam, pos, span, hp, ip)
                parts.append(arr + (np.full(len(arr[0]), fi, np.int8),))
            keys = np.concatenate([p_[0] for p_ in parts])
            tps = np.concatenate([p_[1] for p_ in parts])
            sps = np.concatenate([p_[2] for p_ in parts])
            spans_array = np.concatenate([p_[3] for p_ in parts])
            fis = np.concatenate([p_[4] for p_ in parts])
            uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
            spans = np.bincount(inv, weights=spans_array, minlength=len(uk))
            cand = np.flatnonzero((cnt >= MIN_HITS) | (spans >= SEG_MIN))
            if len(cand) == 0:
                c["fallback_no_candidate_rows"] += 1
            else:
                c["fallback_rows"] += 1
                if not uncapped and len(cand) > FALLBACK_TOPK:
                    c["truncated_rows"] += 1
                    c["unresolved_rows"] += 1
                    cand = cand[np.argsort(-cnt[cand], kind="stable")[:FALLBACK_TOPK]]
                sel = np.isin(inv, cand)
                fam_names = list(FAMILIES)
                groups = defaultdict(list)
                for k_, tp_, sp_, span_value, fi_ in zip(
                    keys[sel].tolist(),
                    tps[sel].tolist(),
                    sps[sel].tolist(),
                    spans_array[sel].tolist(),
                    fis[sel].tolist(),
                ):
                    groups[k_ >> 32, (k_ & 4294967295) - (1 << 31)].append(
                        (fam_names[fi_], tp_, sp_, span_value)
                    )
                c["fallback_candidates"] += len(cand)
        else:
            c["fallback_no_candidate_rows"] += 1
    if not anyw:
        c["no_window_any_family"] += 1
        c["uncovered_ids"].append(i)
        if n_distinct >= AFFINE_MIN_LEVELS and runs8:
            c["no_window_unexplained"] += 1
            c["unexplained_ids"].append(i)
    if len(groups) > ROW_GROUP_CAP:
        c["groups_capped_rows"] += 1
        c["groups_dropped"] += len(groups) - ROW_GROUP_CAP
        groups = dict(
            sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:ROW_GROUP_CAP]
        )
    by_index = defaultdict(list)
    for (m, off), hits in groups.items():
        if len(hits) >= MIN_HITS or sum((span_length for _, _, _, span_length in hits)) >= SEG_MIN:
            by_index[m].extend(hits)
    for m, hits in by_index.items():
        c["candidates"] += 1
        rec, notable = verify(x, index.series[m], hits, dim, stats=c)
        if rec is None:
            continue
        c["verified"] += 1
        if rec["nondegenerate"] and rec["segment"] >= SEG_MIN:
            c["max_r2_segment24"] = max(c["max_r2_segment24"], rec["r2"])
        ids = {
            "config": name,
            "column": column,
            "series_index": i,
            "index_id": index.ids[m],
            "index_series_index": m,
        }
        if extra:
            ids.update(extra)
        copies = [r for r in notable if r["is_copy"]]
        near = [
            r
            for r in notable
            if not r["is_copy"]
            and r["nondegenerate"]
            and (r["segment"] >= SEG_MIN)
            and (r["r2"] >= NEAR_R2)
        ]
        c["copies"] += bool(copies)
        c["near_pairs"] += bool(near)
        for r in copies + near:
            pairs.append({**r, **ids})
        for r in notable:
            if r["degenerate_copy_rejected"]:
                c["degenerate_rejections"] += 1
                for why in r["degenerate_reason"]:
                    if why:
                        c["rejection_reasons"][why] += 1
                records.append({**r, **ids})


def group_rows(series_list, dims=None) -> list[tuple[int, list[int]]]:
    """Group byte-identical rows and calendar dimensions in first-occurrence order."""
    groups = {}
    for i, x in enumerate(series_list):
        x = np.asarray(x, dtype=np.float64)
        d = None if dims is None else np.asarray(dims[i], dtype=np.float64)
        key = _row_key(x) + (b"" if d is None else d.astype("<i2").tobytes())
        groups.setdefault(key, (i, []))[1].append(i)
    return list(groups.values())


def scan_group(name, column, i, idxs, series_list, index, extra=None, dims=None, *, resolve=True):
    """Scan one representative, retaining first-pass and final counters separately."""
    x = np.asarray(series_list[i], dtype=np.float64)
    d = None if dims is None else np.asarray(dims[i], dtype=np.float64)
    mult = len(idxs)
    cr = new_cov()
    lp, lr = ([], [])
    _scan_row(name, column, i, x, index, cr, lp, lr, extra, d)
    result = {
        "i": i,
        "idxs": idxs,
        "mult": mult,
        "initial_unresolved_rows": cr["unresolved_rows"] * mult,
        "initial_truncated_rows": cr["truncated_rows"] * mult,
        "postpass_rows": 0,
        "postpass_candidates": 0,
    }
    if resolve and cr["unresolved_rows"]:
        cr = new_cov()
        lp, lr = ([], [])
        _scan_row(name, column, i, x, index, cr, lp, lr, extra, d, uncapped=True)
        result["postpass_rows"] = mult
        result["postpass_candidates"] = cr["candidates"] * mult
    result.update(counters=cr, pairs=lp, records=lr)
    return result


def finish_scan(c, results, pairs, records, *, index_windows, scan_info=None, all_rejections=False):
    """Merge group results in group order, applying rejection caps across the scan."""
    rec_count = defaultdict(int)
    for result in results:
        idxs, mult, cr = result["idxs"], result["mult"], result["counters"]
        for key in (
            "initial_unresolved_rows",
            "initial_truncated_rows",
            "postpass_rows",
            "postpass_candidates",
        ):
            c[key] += result[key]
        _merge(c, cr, mult)
        if cr["unresolved_rows"]:
            c["unresolved_ids"].extend(idxs)
        for lst in ("uncovered_ids", "unexplained_ids"):
            if cr[lst]:
                c[lst].extend(idxs)
        for p in result["pairs"]:
            p.update(duplicate_rows=mult, series_indices=idxs)
            pairs.append(p)
        for r in result["records"]:
            key = tuple(r["degenerate_reason"] or [])
            if all_rejections or (
                len(records) < MAX_RECORDS and rec_count[key] < MAX_RECORDS_PER_REASON
            ):
                rec_count[key] += 1
                r.update(duplicate_rows=mult, series_indices=idxs)
                records.append(r)
            else:
                c["rejection_records_omitted"] += mult
    info = audit_info.finish_identification(c, pairs, index_windows)
    if scan_info is not None:
        scan_info.update(info)
    return c


def scan(
    name,
    column,
    series_list,
    index,
    cov,
    pairs,
    records,
    extra=None,
    dims=None,
    *,
    resolve=True,
    all_rejections=False,
    scan_info=None,
):
    """Scan byte-identical rows once, then re-scan truncated rows without top-K.

    Final counters replace the first pass for resolved rows (no double counting).
    initial_unresolved_rows / initial_truncated_rows retain first-pass accounting.
    Target policy rejections are never capped. Other reasons use run-11 caps.
    """
    c = new_cov()
    c.update(
        unique_series=0,
        initial_unresolved_rows=0,
        initial_truncated_rows=0,
        postpass_rows=0,
        postpass_candidates=0,
        unresolved_ids=[],
        rejection_records_omitted=0,
    )
    cov[f"{name}::{column}"] = c
    groups = group_rows(series_list, dims)
    c["unique_series"] = len(groups)
    return finish_scan(
        c,
        (
            scan_group(name, column, i, idxs, series_list, index, extra, dims, resolve=resolve)
            for i, idxs in groups
        ),
        pairs,
        records,
        all_rejections=all_rejections,
        index_windows=index.cov["windows"],
        scan_info=scan_info,
    )
