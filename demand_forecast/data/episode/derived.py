"""Optional causal presentations of episode inputs."""

import numpy as np
import pandas as pd

from .kinds import labels_of, observed_mask
from .scale import Scale, finite_scale_values, spread_location, to_scaled
from .schema import NAIVE_OPTIONS, PATCH, QUANTILES

LAG_SCAN_WINDOW = 512
_ONSET_AGES = 32
NAIVE_LIMIT = 1e6


class NaiveHistory:
    """Index the kept target's scaled history once for context and all origins."""

    def __init__(self, scaled: np.ndarray, *, season: int, long_lag: int | None, end: int) -> None:
        self.season = season
        self.long_lag = long_lag if long_lag is not None and long_lag > season else None
        self.width = season * ((8 + season - 1) // season)
        # Position zero denotes a missing source; position j + 1 stores cell j.
        scaled = scaled[: max(0, end)]
        self.values = np.r_[np.nan, scaled]
        self.finite = np.isfinite(self.values)
        self.previous = np.maximum.accumulate(np.where(self.finite, np.arange(len(self.values)), 0))
        size = len(scaled)
        self.context_end = min(max(0, end), size + 4 * max(self.width, self.long_lag or 0))
        pairs_needed = (self.width + 1) // 2
        self.slope = np.full(1, np.nan)
        if size - season >= pairs_needed:
            drift_width = season * ((64 + season - 1) // season)
            # Beyond the item's end, the rolling window loses older pairs without adding any.
            padding = min(drift_width, max(0, end - size))
            differences = np.full(size + padding, np.nan)
            paired = self.finite[1 + season :] & self.finite[1:-season]
            with np.errstate(over="ignore"):
                np.subtract(
                    scaled[season:], scaled[:-season], out=differences[season:size], where=paired
                )
            differences /= season
            means = pd.Series(differences).rolling(drift_width, min_periods=pairs_needed).mean()
            self.slope = np.r_[np.nan, means.to_numpy()]

    def level(self, origin: int) -> float:
        """Mean of the local window ending just before this origin."""
        if origin <= 0:
            return np.nan
        window = self.values[max(0, origin - self.width) + 1 : origin + 1]
        observed = window[np.isfinite(window)]
        if len(observed) < (self.width + 1) // 2:
            return np.nan
        with np.errstate(over="ignore", invalid="ignore"):
            return float(observed.mean())

    def _same_phase(
        self, cells: np.ndarray, bounds: np.ndarray, upper: np.ndarray, lag: int
    ) -> np.ndarray:
        latest = upper - (upper - 1 - cells) % lag
        positions = np.maximum(latest[:, None] - lag * np.arange(4), 0)
        previous = np.where(self.finite[positions], positions, 0).max(axis=1)
        return np.where(previous > bounds - 4 * lag, previous, 0)

    def series(
        self, *, left: int, right: int, origin: int | None
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        cells = np.arange(left, right)
        bounds = cells if origin is None else np.minimum(cells, origin)
        upper = np.clip(bounds, 0, len(self.values) - 1)
        previous = self.previous[upper]
        last = np.where(previous > bounds - 4 * self.width, previous, 0)
        base = last
        rows = {"last": self.values[last]}
        if self.season > 1:
            base = self._same_phase(cells, bounds, upper, self.season)
            rows["season"] = self.values[base]
        if self.long_lag is not None:
            source = self._same_phase(cells, bounds, upper, self.long_lag)
            rows["season2"] = self.values[source]

        slope_index = np.where((bounds > 0) & (bounds < len(self.slope)), bounds, 0)
        with np.errstate(over="ignore", invalid="ignore"):
            rows["drift"] = self.values[base] + self.slope[slope_index] * (cells - base + 1)
        result = {}
        for name, values in rows.items():
            observed = np.isfinite(values)
            result[name] = np.where(observed, values, 0.0), observed
        return result

    def backtest(
        self,
        raw: np.ndarray,
        scale: Scale,
        *,
        origin: int,
        horizon: int,
        kind: str = "seasonal",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pool causal, signed residuals in each patch of up to four earlier horizons."""
        patches = (horizon + PATCH - 1) // PATCH
        residuals = np.full((4, patches * PATCH, len(NAIVE_OPTIONS)), np.nan)
        for m in range(1, 5):
            start, stop = origin - m * horizon, origin - (m - 1) * horizon
            if start < 0:
                break
            if stop >= len(self.values) or np.isfinite(raw[start:stop]).sum() < (horizon + 1) // 2:
                continue
            outcomes = self.values[start + 1 : stop + 1]
            paths = self.series(left=start, right=stop, origin=start)
            finite = finite_scale_values(raw, start, kind=kind)
            with np.errstate(over="ignore", invalid="ignore"):
                if finite.size:
                    loc = spread_location(finite) if kind == "spread" else np.median(finite)
                    median = to_scaled(loc, scale)
                    paths["none"] = median, np.isfinite(median)
                level = self.level(start)
                paths["level"] = level, np.isfinite(level)
                if self.long_lag is not None and start < self.long_lag:
                    paths.pop("season2")
                for j, name in enumerate(NAIVE_OPTIONS):
                    if name in paths:
                        path, available = paths[name]
                        difference = outcomes - path
                        valid = np.isfinite(difference) & available & (np.abs(path) <= NAIVE_LIMIT)
                        residuals[m - 1, :horizon, j] = np.where(valid, difference, np.nan)

        # Pool pairs across windows before applying the minimum count in each patch.
        pairs = residuals.reshape(4, patches, PATCH, -1).transpose(1, 3, 0, 2)
        pairs = pairs.reshape(patches, len(NAIVE_OPTIONS), 4 * PATCH)
        count = np.isfinite(pairs).sum(axis=-1)
        patch_size = np.minimum(PATCH, horizon - np.arange(patches) * PATCH)
        with np.errstate(over="ignore"):
            error = np.nansum(np.abs(pairs), axis=-1) / np.maximum(count, 1)
        seen = (count >= (patch_size[:, None] + 1) // 2) & (error <= NAIVE_LIMIT)
        resq = np.zeros((*seen.shape, len(QUANTILES)), dtype=np.float32)
        # Sort the viable columns once, with missing pairs at the end.
        ordered = np.sort(pairs[seen], axis=-1)
        ranks = (count[seen][:, None] - 1) * np.asarray(QUANTILES)
        lower = np.floor(ranks).astype(int)
        low = np.take_along_axis(ordered, lower, axis=-1)
        high = np.take_along_axis(ordered, np.ceil(ranks).astype(int), axis=-1)
        resq[seen] = low + (ranks - lower) * (high - low)
        return np.where(seen, error, 0).astype(np.float32), seen, resq


def naive_series(
    scaled: np.ndarray,
    *,
    season: int,
    long_lag: int | None,
    left: int,
    right: int,
    origin: int | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return naive values and observed flags for item-local cells in [left, right).

    A cell uses its own bound up to the origin, and the origin's bound thereafter.
    Pass a long lag only when the request has enough history at its first origin.
    """
    end = right - 1 if origin is None else min(right - 1, origin)
    history = NaiveHistory(scaled, season=season, long_lag=long_lag, end=end)
    return history.series(left=left, right=right, origin=origin)


def _adjust(values: np.ndarray, observed: np.ndarray, phases: int) -> np.ndarray:
    """Remove a linear trend, then phase means, fitting only observed cells."""
    time = np.arange(len(values)) - (len(values) - 1) / 2
    time -= time[observed].mean()
    centered = values - values[observed].mean()
    fitted_time = time[observed]
    residual = centered - time * (fitted_time @ centered[observed]) / (fitted_time @ fitted_time)
    if phases:
        phase = np.arange(len(values)) % phases
        count = np.bincount(phase[observed], minlength=phases)
        total = np.bincount(phase[observed], residual[observed], minlength=phases)
        residual -= (total / np.maximum(count, 1))[phase]
    return residual


def _acf(values: np.ndarray, observed: np.ndarray, lags: int) -> np.ndarray:
    centered = values - values[observed].mean()
    if observed.all():
        # Preserve the frozen calibration on complete series.
        energy = centered @ centered
        if energy == 0:
            return np.zeros(lags)
        return np.array([(centered[k:] @ centered[:-k]) / energy for k in range(1, lags + 1)])

    acf = np.zeros(lags)
    for lag in range(1, lags + 1):
        paired = observed[lag:] & observed[:-lag]
        left, right = centered[:-lag][paired], centered[lag:][paired]
        norm = np.linalg.norm(left) * np.linalg.norm(right)
        if norm:
            acf[lag - 1] = (right @ left) / norm
    return acf


def _pearson(x: np.ndarray, y: np.ndarray, has_missing: bool) -> tuple[float, int]:
    """Absolute correlation and the number of complete pairs it rests on."""
    if has_missing:
        paired = np.isfinite(x) & np.isfinite(y)
        if paired.sum() < 2:
            return 0.0, int(paired.sum())
        x, y = x[paired], y[paired]
    score = float(abs(np.corrcoef(x, y)[0, 1]))
    return (0.0 if has_missing and np.isnan(score) else score), len(x)


def scan_delay(
    x: np.ndarray, y: np.ndarray, *, season: int, no_evidence: float
) -> tuple[int, float, float]:
    """Return the admitted delay (or zero), best positive-lag score and lag-zero score."""
    x = np.asarray(x[-LAG_SCAN_WINDOW:], dtype=np.float64)
    y = np.asarray(y[-LAG_SCAN_WINDOW:], dtype=np.float64)
    width = len(x)
    max_delay = min(64, width // 3)
    common = width - max_delay
    phases = season if season > 1 and width >= 8 * season else 0
    dof = common - 2 - phases
    if max_delay < 8 or common < 32 or dof < 24:
        return 0, 0.0, 0.0

    x_observed, y_observed = np.isfinite(x), np.isfinite(y)
    observed = min(x_observed.sum(), y_observed.sum())
    if 5 * observed < 4 * width:
        return 0, 0.0, 0.0
    has_missing = observed < width
    with np.errstate(invalid="ignore", divide="ignore"):
        x = _adjust(x, x_observed, phases)
        y = _adjust(y, y_observed, phases)
        scores, pairs = zip(
            *(
                _pearson(x[max_delay - delay : width - delay], y[max_delay:], has_missing)
                for delay in range(max_delay + 1)
            ),
            strict=True,
        )
        acf_lags = min(max_delay, common // 4)
        inflation = max(
            1.0, 1 + 2 * float(_acf(x, x_observed, acf_lags) @ _acf(y, y_observed, acf_lags))
        )

    # argmax returns the first of any exact ties.
    delay = int(np.argmax(scores[1:])) + 1
    score, contemporaneous = scores[delay], scores[0]
    # Missing cells shorten the evidence: the threshold counts the chosen delay's complete pairs.
    dof = pairs[delay] - 2 - phases
    if pairs[delay] < 32 or dof < 24:
        return 0, score, contemporaneous
    # The absolute floor was calibrated on complete windows: it rises as missing cells remove pairs.
    floor = 0.2 * np.sqrt(common / pairs[delay])
    threshold = max(floor, 3.5 * np.sqrt(inflation / dof))
    admitted = score >= threshold and score > contemporaneous and score > no_evidence
    return (delay if admitted else 0), score, contemporaneous


def increment_spread(
    labels: np.ndarray,
    target: np.ndarray,
    *,
    left: int,
    right: int,
    frozen_at: int | None,
) -> np.ndarray:
    """Return the RMS increment divided by sqrt(2) for each active label."""
    labels = np.asarray(labels, dtype=object)
    target = np.asarray(target)
    if labels.ndim != 1 or target.ndim != 1:
        raise ValueError("labels and target must be one-dimensional")
    if right < left:
        raise ValueError("right must be at least left")
    values = np.zeros(right - left, dtype=np.float32)
    start, stop = max(0, left), min(right, len(labels))
    bound = stop - 1 if frozen_at is None else min(stop - 1, frozen_at)
    end = min(bound, len(target))
    if start >= stop or end <= 1:
        return values

    labels = labels[:stop]
    names = labels_of(labels)
    names[~observed_mask(labels)] = None
    codes, unique = pd.factorize(names, sort=False)
    # Group by label while keeping each group's original time order.
    order = np.argsort(codes, kind="stable")
    edges = np.searchsorted(codes[order], np.arange(len(unique) + 1))
    target = target[:end].astype(np.float64, copy=False)
    for code in range(len(unique)):
        occurrences = order[edges[code] : edges[code + 1]]
        selected = occurrences[np.searchsorted(occurrences, start) :]
        if not len(selected):
            continue
        completed = occurrences[1:][np.diff(occurrences) == 1]
        completed = completed[completed < end]
        finite = np.isfinite(target[completed]) & np.isfinite(target[completed - 1])
        completed = completed[finite]
        ends = selected if frozen_at is None else np.minimum(selected, frozen_at)
        count = np.searchsorted(completed, ends, side="left")
        with np.errstate(over="ignore", invalid="ignore"):
            difference = target[completed] - target[completed - 1]
            squares = np.r_[0.0, (difference**2).cumsum()][count]
            values[selected - left] = np.sqrt(squares / np.maximum(count, 1) / 2)
    return np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def onset_profile(
    labels: np.ndarray,
    target: np.ndarray,
    *,
    left: int,
    right: int,
    frozen_at: int | None,
    season: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the shrunk mean response at each cell's label and age since onset."""
    labels, target = np.asarray(labels), np.asarray(target)
    if labels.ndim != 1 or target.ndim != 1:
        raise ValueError("labels and target must be one-dimensional")
    if right < left:
        raise ValueError("right must be at least left")
    values = np.zeros(right - left, dtype=np.float64)
    observed = np.zeros(right - left, dtype=bool)
    start, stop = max(0, left), min(right, len(labels))
    end = min(stop - 1, len(target))
    if frozen_at is not None:
        end = min(end, frozen_at)
    if start >= stop or end <= 1:
        return values, observed

    target = target[:end].astype(np.float64, copy=False)
    finite = np.isfinite(target)
    labels = labels[:stop].astype(object, copy=False)
    present = observed_mask(labels)
    names = np.empty(stop, dtype=object)
    # Bound the work of labels_of's numeric deduplication even on high-cardinality input.
    for offset in range(0, stop, 1024):
        block = slice(offset, offset + 1024)
        names[block] = labels_of(labels[block])

    history: dict[tuple[str, int], tuple[float, int]] = {}
    previous = onset = base = None
    window = min(season, 8)
    for t, (name, valid) in enumerate(zip(names, present, strict=True)):
        if not valid:
            previous = onset = None
            continue
        if previous is not None and name != previous:
            onset, base = t, None
            if t < end:
                before = slice(max(0, t - window), t)
                prefix = target[before][finite[before]]
                if len(prefix):
                    base = float(prefix.mean())
        previous = name
        if onset is None or t - onset >= _ONSET_AGES:
            continue

        key = name, t - onset
        total, count = history.get(key, (0.0, 0))
        if t >= start and count:
            values[t - left] = (count / (count + 2)) * (total / count)
            observed[t - left] = True
        # Read before updating. After end, the horizon can change the age but never the history.
        if t < end and finite[t] and base is not None and np.isfinite(base):
            history[key] = total + (float(target[t]) - base), count + 1

    return values, observed
