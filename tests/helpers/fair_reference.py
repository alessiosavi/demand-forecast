# ruff: noqa: N803, N806
"""Frozen F13 round 4 and 5 static category score reference, formatted for the suite.

components(S, codes, k): today's uncalibrated component scores, two per summary entry (mean share, spread share).
fair_rank(...): the observed components and those of P label permutations form 1 + P exchangeable elements under
the null. Each element's component is compared with the largest value of that component among ALL OTHER elements;
an element's statistic M is its largest calibrated component; the score compares the observed M with the largest
M among the permutations. Under the null the score is positive with probability at most 1 / (1 + P).
"""

import numpy as np


def adjusted(values, codes, k):
    n = len(values)
    counts = np.bincount(codes, minlength=k)
    sums = np.bincount(codes, weights=values, minlength=k)
    between = np.sum(sums**2 / np.maximum(counts, 1)) - values.sum() ** 2 / n
    total = np.sum((values - values.mean()) ** 2)
    if k < 2 or n <= k or total <= 0:
        return 0.0
    return float(np.clip(1 - (1 - between / total) * (n - 1) / (n - k), 0, 1))


def components(S, codes, k):
    out = []
    for j in range(S.shape[1]):
        v = S[:, j] - S[:, j].mean()
        means = np.bincount(codes, weights=v, minlength=k) / np.maximum(
            np.bincount(codes, minlength=k), 1
        )
        out += [adjusted(v, codes, k), adjusted(np.abs(v - means[codes]), codes, k)]
    return np.array(out)


def fair_rank(S, codes, k, perms=32, seed=0):
    rng = np.random.default_rng([seed, len(codes), k])
    allc = np.array(
        [components(S, codes, k)]
        + [components(S, codes[rng.permutation(len(codes))], k) for _ in range(perms)]
    )
    order = np.sort(allc, axis=0)
    top, second = order[-1], order[-2]
    ref = np.where(allc == top, second, top)  # per component: the largest among the other elements
    live = 1 - ref > 1e-9
    c = np.where(live, np.maximum(0, (allc - ref) / np.where(live, 1 - ref, 1)), 0.0)
    M = c.max(1)
    q = np.max(M[1:])
    return max(0.0, (M[0] - q) / (1 - q)) if 1 - q > 1e-9 else 0.0


def fair_rank_masked(S, valid, codes, perms=32, seed=0):
    """As fair_rank, with a validity mask per summary entry (round 5).

    S: items x entries; valid: boolean items x entries; codes: label code per item (single-label).
    For entry j only its valid items count. A permutation acts WITHIN the entry's valid items: they keep their
    values and receive the label cells of the valid items in the order induced by one shared permutation of all
    items. The label multiset of every entry's valid set is thereby preserved in every element.
    """
    n = len(codes)
    rng = np.random.default_rng([seed, n, len(np.unique(codes))])
    orders = [np.arange(n)] + [rng.permutation(n) for _ in range(perms)]
    rows = []
    for order in orders:
        row = []
        for j in range(S.shape[1]):
            idx = np.flatnonzero(valid[:, j])
            if len(idx) < 2:
                row += [0.0, 0.0]
                continue
            induced = idx[
                np.argsort(np.argsort(order[idx]))
            ]  # valid items reordered by the shared permutation
            _, local = np.unique(codes[induced], return_inverse=True)
            row += list(components(S[idx, j : j + 1], local, local.max() + 1))
        rows.append(row)
    allc = np.array(rows)
    srt = np.sort(allc, axis=0)
    top, second = srt[-1], srt[-2]
    ref = np.where(allc == top, second, top)
    live = 1 - ref > 1e-9
    c = np.where(live, np.maximum(0, (allc - ref) / np.where(live, 1 - ref, 1)), 0.0)
    M = c.max(1)
    q = np.max(M[1:])
    return max(0.0, (M[0] - q) / (1 - q)) if 1 - q > 1e-9 else 0.0
