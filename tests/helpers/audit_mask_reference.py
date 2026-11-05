"""Mask-based verify from main 0c76082, retained as a differential oracle.

Only imports are qualified; keep the control flow frozen so ordering changes
in the optimized matcher are detected as well as interval changes.
"""

from collections import defaultdict

import numpy as np

from demand_forecast.pretrain import audit_match as match


def verify(t, s, hits, dim=None):
    """hits: list of (fam, t_pos, s_pos, span). Returns (best record, notable records)."""
    offsets = defaultdict(list)
    for fam, tp, sp, span_length in hits:
        offsets[int(tp - sp)].append((int(tp), int(span_length), fam, int(sp)))
    best, notable = (None, [])
    fin_s = s[np.isfinite(s)]
    scale = max(1.0, float(np.abs(fin_s).max())) if len(fin_s) else 1.0
    for off, lst in offsets.items():
        if (
            len(lst) < match.MIN_HITS
            and sum((span_length for _, span_length, _, _ in lst)) < match.SEG_MIN
        ):
            continue
        covered = np.zeros(len(t), bool)
        for tp, span_length, _, _ in lst:
            covered[tp : tp + span_length] = True
        fams = sorted({f for _, _, f, _ in lst})
        todo = [("merged", a0, a1) for a0, a1 in match.intervals_of(covered)]
        groups = defaultdict(list)
        for tp, span_length, fam, sp in lst:
            k = match.affine_key(t, s, tp, sp, span_length, fam, scale)
            if k is not None:
                groups[k].append((tp, span_length))
        if len(groups) > 1:
            for items in groups.values():
                cg = np.zeros(len(t), bool)
                for tp, span_length in items:
                    cg[tp : tp + span_length] = True
                todo.extend((("grouped", a0, a1) for a0, a1 in match.intervals_of(cg)))
        seen = set()
        queue = list(todo)
        while queue:
            mode, a0, a1 = queue.pop(0)
            if (a0, a1) in seen:
                continue
            seen.add((a0, a1))
            rec = match.fit_segment(t, s, a0, a1, off, dim)
            if rec is None:
                continue
            if not rec["is_copy"] and mode in ("merged", "grouped") and (a1 - a0 >= 2 * match.WIN):
                b_fin = s[max(0, a0 - off) : max(0, a1 - off)]
                b_fin = b_fin[np.isfinite(b_fin)]
                slack = (
                    match.REFINE_SLACK
                    * max(match.COPY_ATOL, match.COPY_RTOL * float(np.ptp(b_fin)))
                    if len(b_fin) > 1
                    else match.REFINE_SLACK * match.COPY_ATOL
                )
                for p0, p1 in match.refine_pieces(t, s, a0, a1, off, slack):
                    if p1 - p0 >= match.WIN and (p0, p1) != (a0, a1):
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
                or (
                    rec["nondegenerate"]
                    and rec["segment"] >= match.SEG_MIN
                    and (rec["r2"] >= match.NEAR_R2)
                )
            ):
                notable.append(rec)
    return (best[1] if best else None, notable)
