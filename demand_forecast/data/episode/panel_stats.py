"""Cutoff statistics and leave-one-out typed context for one panel target."""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .bank import summary_vectors
from .kinds import infer_static_kind, is_missing, label_of
from .scale import Scale, compress, fit_scale, to_scaled
from .schema import BANK_WINDOW, N_SUMMARY, N_VALUE_TOKEN, Panel


@dataclass
class _Grid:
    left: int
    total: np.ndarray
    support: np.ndarray
    filled: np.ndarray


class PanelStats:
    """Statistics shared by all items at a target's fixed statistics cutoff."""

    SUMMARY_CHUNK = 2048
    SUMMARY_CHUNK_CELLS = 4_000_000

    def __init__(self, panel: Panel, target: str, cutoff: int) -> None:
        self.panel = panel
        self.target = target
        self.cutoff = cutoff
        self.screen_cache: dict = {}
        self.scales: list[Scale] = []
        self._eligible = np.zeros(len(panel.items), dtype=bool)
        for i, item in enumerate(panel.items):
            values = item.targets.get(target, np.empty(0, dtype=np.float64))
            end = max(0, cutoff - item.start)
            self.scales.append(fit_scale(values, end, panel.season))
            self._eligible[i] = np.isfinite(values[:end]).any()

        self.summaries = np.zeros((len(panel.items), N_SUMMARY), dtype=np.float32)
        self.summary_flags = np.zeros(self.summaries.shape, dtype=bool)
        for group_first in range(0, len(panel.items), self.SUMMARY_CHUNK):
            group = panel.items[group_first : group_first + self.SUMMARY_CHUNK]
            length = min(
                BANK_WINDOW,
                max(
                    (max(0, cutoff - item.start) for item in group if target in item.targets),
                    default=0,
                ),
            )
            # Preserve the original group's padding width and reduction order.
            rows = max(1, self.SUMMARY_CHUNK_CELLS // max(1, length))
            for offset in range(0, len(group), rows):
                first = group_first + offset
                items = group[offset : offset + rows]
                raw = np.full((len(items), length), np.nan, dtype=np.float64)
                for row, item in enumerate(items):
                    values = item.targets.get(target)
                    if values is None:
                        continue
                    left = max(item.start, cutoff - length)
                    right = min(cutoff, item.start + len(values))
                    if left < right:
                        raw[row, left - cutoff + length : right - cutoff + length] = values[
                            left - item.start : right - item.start
                        ]
                scales = self.scales[first : first + len(items)]
                loc = np.array([s.loc for s in scales])[:, None]
                scale = np.array([s.scale for s in scales])[:, None]
                summaries, flags = summary_vectors(
                    (raw - loc) / scale,
                    raw,
                    season=panel.season,
                    long_lag=panel.long_lag,
                    end=cutoff,
                )
                self.summaries[first : first + len(items)] = summaries
                self.summary_flags[first : first + len(items)] = flags

        self.static_kinds: dict[str, str] = {}
        self._labels: dict[str, list[tuple[str, ...]]] = {}
        self._members: dict[tuple[str, str], np.ndarray] = {}
        self._numbers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._moments: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._grids: dict[tuple[str, str] | None, _Grid] = {}
        self._all_members = np.flatnonzero(self._eligible)
        attributes = sorted({name for item in panel.items for name in item.static})
        for attribute in attributes:
            cells = [item.static.get(attribute) for item in panel.items]
            kind = infer_static_kind(cells)
            self.static_kinds[attribute] = kind
            if kind == "number":
                numbers = np.array(
                    [np.nan if is_missing(value) else float(value) for value in cells],
                    dtype=np.float64,
                )
                finite = np.isfinite(numbers)
                observed = numbers[finite]
                normalised = np.zeros(len(cells), dtype=np.float32)
                edges = np.empty(0)
                if len(observed):
                    std = observed.std()
                    normalised[finite] = (observed - observed.mean()) / (std if std > 0 else 1.0)
                    quantiles = np.linspace(0, 1, min(8, len(observed)) + 1)
                    edges = np.unique(np.quantile(observed, quantiles))[1:-1]
                self._numbers[attribute] = normalised, finite
                bins = np.searchsorted(edges, numbers, side="right")
                labels = [(f"bin{b}",) if valid else () for b, valid in zip(bins, finite)]
            else:
                labels = []
                for value in cells:
                    if is_missing(value):
                        labels.append(())
                    elif kind == "multilabel" and isinstance(value, (list, tuple)):
                        labels.append(
                            tuple(sorted({label_of(v) for v in value if not is_missing(v)}))
                        )
                    else:
                        labels.append((label_of(value),))
            self._labels[attribute] = labels
            groups: dict[str, list[int]] = {}
            # Item records and their label lists are ragged; numerical reductions stay batched.
            for i, item_labels in enumerate(labels):
                if self._eligible[i]:
                    for label in item_labels:
                        groups.setdefault(label, []).append(i)
            for label, members in groups.items():
                self._members[attribute, label] = np.asarray(members, dtype=np.int64)

    def labels(self, attribute: str, item_index: int) -> tuple[str, ...]:
        """Return the item's canonical, deduplicated labels, including number bins."""
        if not 0 <= item_index < len(self.panel.items):
            raise IndexError("item index outside the panel")
        labels = self._labels.get(attribute)
        return () if labels is None else labels[item_index]

    def members(self, attribute: str, label: str) -> np.ndarray:
        """Return sorted indices with observed target history and the given label."""
        return self._members.get((attribute, label), np.empty(0, dtype=np.int64)).copy()

    @staticmethod
    def _contains(members: np.ndarray, item_index: int) -> bool:
        position = np.searchsorted(members, item_index)
        return position < len(members) and members[position] == item_index

    def value_token(
        self, attribute: str, label: str, exclude: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Pool valid summary entries, subtracting the query from cached moments."""
        key = attribute, label
        members = self._members.get(key, np.empty(0, dtype=np.int64))
        subtract = self._contains(members, exclude)
        support = len(members) - int(subtract)
        token = np.zeros(N_VALUE_TOKEN, dtype=np.float32)
        flags = np.zeros(N_SUMMARY, dtype=bool)
        if not support:
            return token, flags, 0
        if key not in self._moments:
            valid = self.summary_flags[members]
            values = np.where(valid, self.summaries[members], 0.0).astype(np.float64)
            self._moments[key] = values.sum(axis=0), (values**2).sum(axis=0), valid.sum(axis=0)
        total, squares, counts = (a.copy() for a in self._moments[key])
        if subtract:
            valid = self.summary_flags[exclude]
            own = np.where(valid, self.summaries[exclude], 0.0).astype(np.float64)
            total -= own
            squares -= own**2
            counts -= valid
        flags = counts > 0
        mean = total / np.maximum(counts, 1)
        variance = np.maximum(0.0, squares / np.maximum(counts, 1) - mean**2)
        token[:N_SUMMARY] = np.where(flags, mean, 0.0)
        token[N_SUMMARY : 2 * N_SUMMARY] = np.where(flags, np.sqrt(variance), 0.0)
        token[-1] = np.log1p(support)
        return token, flags, support

    def _contribution(
        self, item_index: int, left: int, right: int
    ) -> tuple[slice, np.ndarray, np.ndarray] | None:
        item = self.panel.items[item_index]
        target = item.targets[self.target]
        start = max(left, item.start)
        end = min(right, item.start + len(target))
        if start >= end:
            return None
        raw = target[start - item.start : end - item.start]
        observed = np.isfinite(raw)
        values = np.where(observed, compress(to_scaled(raw, self.scales[item_index])), 0.0)
        return slice(start - left, end - left), values, observed

    def _aggregate(
        self, members: np.ndarray, left: int, right: int
    ) -> tuple[np.ndarray, np.ndarray]:
        total = np.zeros(right - left, dtype=np.float64)
        support = np.zeros(right - left, dtype=np.int64)
        for i in members:
            contribution = self._contribution(int(i), left, right)
            if contribution is not None:
                span, values, observed = contribution
                total[span] += values
                support[span] += observed
        return total, support

    def _cached_grid(
        self, key: tuple[str, str] | None, members: np.ndarray, left: int, right: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if key not in self._grids:
            self._grids[key] = _Grid(
                left,
                np.zeros(right - left, dtype=np.float64),
                np.zeros(right - left, dtype=np.int64),
                np.zeros(right - left, dtype=bool),
            )
        grid = self._grids[key]
        padding = max(0, grid.left - left), max(0, right - grid.left - len(grid.total))
        if any(padding):
            grid.total = np.pad(grid.total, padding)
            grid.support = np.pad(grid.support, padding)
            grid.filled = np.pad(grid.filled, padding)
            grid.left -= padding[0]
        span = slice(left - grid.left, right - grid.left)
        missing = ~grid.filled[span]
        transitions = np.diff(np.r_[False, missing, False].astype(np.int8))
        starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
        for start, end in zip(starts, ends):
            total, support = self._aggregate(members, left + int(start), left + int(end))
            destination = slice(left + start - grid.left, left + end - grid.left)
            grid.total[destination] = total
            grid.support[destination] = support
            grid.filled[destination] = True
        return grid.total[span].copy(), grid.support[span].copy()

    def _row(
        self,
        key: tuple[str, str] | None,
        members: np.ndarray,
        exclude: int,
        left: int,
        right: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if right < left:
            raise ValueError("right must be at least left")
        if not self.panel.aligned or not len(members) or right == left:
            return np.zeros(right - left, dtype=np.float32), np.zeros(right - left, dtype=np.int64)
        total, support = self._cached_grid(key, members, left, right)
        if self._contains(members, exclude):
            contribution = self._contribution(exclude, left, right)
            if contribution is not None:
                span, values, observed = contribution
                total[span] -= values
                support[span] -= observed
        values = np.where(support > 0, total / np.maximum(support, 1), 0.0)
        return values.astype(np.float32), support

    def peer_row(
        self, attribute: str, label: str, exclude: int, left: int, right: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean compressed, individually scaled peers on exactly the requested grid."""
        key = attribute, label
        members = self._members.get(key, np.empty(0, dtype=np.int64))
        return self._row(key, members, exclude, left, right)

    def panel_row(self, exclude: int, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
        """Whole-panel peer mean, leaving out the query and ineligible histories."""
        return self._row(None, self._all_members, exclude, left, right)

    def static_numbers(self, item_index: int) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Standardised numeric attributes in name order; missing is zero and false."""
        if not 0 <= item_index < len(self.panel.items):
            raise IndexError("item index outside the panel")
        names = sorted(self._numbers)
        values = np.array([self._numbers[name][0][item_index] for name in names], dtype=np.float32)
        flags = np.array([self._numbers[name][1][item_index] for name in names], dtype=bool)
        return names, values, flags


class PanelStatsProvider:
    """Cache panel statistics by target and cutoff, optionally bounding retention."""

    def __init__(self, panel: Panel, *, limit: int | None = 16) -> None:
        if limit is not None and limit < 0:
            raise ValueError("statistics cache limit must be nonnegative")
        self.panel = panel
        self.builds = 0
        self._limit = limit
        self._cache: OrderedDict[tuple[str, int], PanelStats] = OrderedDict()

    def get(self, target: str, cutoff: int) -> PanelStats:
        key = target, cutoff
        stats = self._cache.get(key)
        if stats is None:
            stats = PanelStats(self.panel, target, cutoff)
            self.builds += 1
            self._cache[key] = stats
        self._cache.move_to_end(key)
        if self._limit is not None and len(self._cache) > self._limit:
            self._cache.popitem(last=False)
        return stats

    def release(self, cutoff: int) -> None:
        """Discard all cached targets at the given cutoff."""
        for key in [key for key in self._cache if key[1] == cutoff]:
            del self._cache[key]
