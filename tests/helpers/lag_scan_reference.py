"""Frozen D2 calibration rule and cases from the plan 2c handoffs.

This is a frozen copy of the plan's reference rule and must never be edited to make a test pass.
"""

import numpy as np


def adjust(v, season):
    n = len(v)
    t = np.arange(n) - (n - 1) / 2
    v = v - v.mean()
    v = v - t * (t @ v) / (t @ t)
    if season > 1 and n >= 8 * season:
        ph = np.arange(n) % season
        v = v - (np.bincount(ph, v, season) / np.bincount(ph, minlength=season))[ph]
    return v


def acf(v, acf_lags):
    v = v - v.mean()
    d = v @ v
    return np.array([1.0] + [(v[k:] @ v[:-k]) / d for k in range(1, acf_lags + 1)])


def scan_delay(x, y, season, k=3.5, floor_min=0.2, max_delay=64, window=512):
    """Finite, equally long series ending just before the cutoff. Returns (delay or 0, score, score at lag 0)."""
    x = adjust(np.asarray(x[-window:], float), season)
    y = adjust(np.asarray(y[-window:], float), season)
    n0 = len(x)
    max_lag = min(max_delay, n0 // 3)
    if max_lag < 8:
        return 0, 0.0, 0.0
    b = y[max_lag:]
    n = len(b)
    if n < 32:
        return 0, 0.0, 0.0
    r = np.array([abs(np.corrcoef(x[max_lag - d : n0 - d], b)[0, 1]) for d in range(max_lag + 1)])
    acf_lags = min(max_lag, n // 4)
    infl = max(1.0, 1 + 2 * float(acf(x, acf_lags)[1:] @ acf(y, acf_lags)[1:]))
    d = int(np.argmax(r[1:])) + 1
    dof = n - 2 - (season if season > 1 and n0 >= 8 * season else 0)
    if dof < 24:
        return 0, 0.0, 0.0
    ok = r[d] >= max(floor_min, k * np.sqrt(infl / dof)) and r[d] > r[0]
    return (d if ok else 0), float(r[d]), float(r[0])


def ar(rng, n, coef, burn=200):
    e = rng.normal(size=n + burn)
    x = np.zeros(n + burn)
    for t in range(len(coef), n + burn):
        x[t] = sum(c * x[t - 1 - i] for i, c in enumerate(coef)) + e[t]
    return x[burn:]


def case(name, rng, n, season):
    s = np.sin(2 * np.pi * np.arange(n) / max(season, 2)) if season > 1 else np.zeros(n)
    if name == "white":
        return rng.normal(size=n), rng.normal(size=n), 0
    if name == "ar2":
        return ar(rng, n, (1.6, -0.64)), ar(rng, n, (1.6, -0.64)), 0
    if name == "ar1":
        return ar(rng, n, (0.7,)), ar(rng, n, (0.9,)), 0
    if name == "season":
        return 2 * np.roll(s, 2) + ar(rng, n, (0.5,)), 2 * s + ar(rng, n, (0.5,)), 0
    if name == "contemp":
        x = ar(rng, n, (0.7,))
        return x, 0.7 * x / x.std() + ar(rng, n, (0.5,)), 0
    if name.startswith("gen"):
        d = int(name[3:])
        x = ar(rng, n + d, (0.7,))
        beta = rng.uniform(0.4, 0.9)
        xs = (x - x.mean()) / x.std()
        return x[d:], 1.5 * s + beta * xs[:n] * 1.0 + ar(rng, n, (0.5,)) * 0.6, d
    if name.startswith("smooth"):
        d = int(name[6:])
        x = ar(rng, n + d, (1.6, -0.64))
        xs = (x - x.mean()) / x.std()
        return x[d:], 1.5 * s + 0.7 * xs[:n] + ar(rng, n, (0.5,)) * 0.6, d
    raise ValueError(name)


CASES = (
    "white",
    "ar1",
    "ar2",
    "season",
    "contemp",
    "gen1",
    "gen3",
    "gen7",
    "gen10",
    "gen14",
    "gen30",
    "smooth3",
)
