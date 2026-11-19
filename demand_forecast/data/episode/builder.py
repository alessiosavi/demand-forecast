"""Build cutoff-bounded, typed numpy episodes on an absolute panel grid."""

from dataclasses import dataclass, replace

import numpy as np

from .bank import step_features, summary_vectors
from .calendar import calendar_arrays
from .kinds import flag_codes, is_missing, label_of
from .panel_stats import PanelStats, PanelStatsProvider
from .scale import Scale, compress, fit_scale, to_scaled
from .schema import (
    BANK_WINDOW,
    CH_OBSERVED,
    CH_STEP,
    CH_STEP_FLAG,
    CH_SUPPORT,
    CH_VALUE,
    CH_VALUE_STATS,
    KIND_CATEGORY,
    KIND_FLAG,
    KIND_NUMBER,
    MAX_CONTEXT,
    MAX_CONTEXT_ROWS,
    MAX_FEATURE_ROWS,
    MAX_FORECAST_ROWS,
    MAX_HORIZON,
    N_CHANNELS,
    N_PHASES,
    N_SUMMARY,
    N_VALUE_TOKEN,
    PATCH,
    ROLE_FORECAST,
    ROLE_KNOWN,
    ROLE_PANEL,
    ROLE_PAST,
    ROLE_PEER,
    ROLE_SIBLING,
    STEP_FAMILIES,
    SUMMARY_FEATURES,
    Panel,
)
from .screening import Candidate, ScreeningConfig, relevance_number, screen, select
from .value_stats import category_step_stats

DROPPABLE = (
    "features",
    "siblings",
    "peers",
    "panel_row",
    "attributes",
    "static_numbers",
    "calendar",
    "rolling",
    "lags",
    "zeros",
    "value_stats",
    "value_effects",
)


@dataclass(frozen=True)
class EpisodeRequest:
    """Origins share a patch anchor and one statistics cutoff."""

    item_index: int
    origins: tuple[int, ...]
    horizon: int
    context: int = MAX_CONTEXT
    target: str | None = None
    cutoff: int | None = None
    drop: frozenset[str] = frozenset()
    keep_columns: frozenset[str] | None = None


@dataclass
class Episode:
    """Model inputs, supervision and reporting metadata for one item.

    ``row_keys`` has one entry per row in the final row order: the
    ``(source, name)`` of the screening candidate that produced a sibling or
    dynamic row (every label row and the effect row of an expanded category
    carry the category's key), and ``None`` for forecast, peer and whole-panel
    rows.
    """

    x: np.ndarray
    patch_valid: np.ndarray
    row_kind: np.ndarray
    row_role: np.ndarray
    row_names: list[str]
    row_summary: np.ndarray
    row_summary_flags: np.ndarray
    row_score: np.ndarray
    row_cand: np.ndarray
    forecast_rows: np.ndarray
    keep: np.ndarray
    labels: np.ndarray
    label_mask: np.ndarray
    loc: np.ndarray
    scale: np.ndarray
    fallback: np.ndarray
    known_rows: np.ndarray
    xh: np.ndarray
    cal_phases: np.ndarray
    cal_ids: np.ndarray
    calh_phases: np.ndarray
    calh_ids: np.ndarray
    cand_names: list[str]
    cand_kind: np.ndarray
    cand_summary: np.ndarray
    cand_flags: np.ndarray
    cand_score: np.ndarray
    attr_stats: np.ndarray
    attr_flags: np.ndarray
    attr_cand: np.ndarray
    attr_names: list[str]
    static_num: np.ndarray
    static_num_flags: np.ndarray
    static_num_cand: np.ndarray
    freq_class: int
    horizon: int
    origin_patches: np.ndarray
    context_target: str | None
    trace: tuple[Candidate, ...] = ()
    row_keys: tuple[tuple[str, str] | None, ...] = ()


def apply_summary_off(episode: Episode, names: tuple[str, ...]) -> Episode:
    """Clear named summaries and their flags in place and return the episode."""
    for name in names:
        j = SUMMARY_FEATURES.index(name)
        episode.row_summary[..., j] = 0
        episode.row_summary_flags[..., j] = False
        episode.cand_summary[..., j] = 0
        episode.cand_flags[..., j] = False
        episode.attr_stats[..., j] = 0
        episode.attr_stats[..., N_SUMMARY + j] = 0
    return episode


@dataclass
class _Row:
    name: str
    kind: int
    role: int
    score: float
    raw: np.ndarray
    scaled: np.ndarray
    scale: Scale
    summary: np.ndarray
    flags: np.ndarray
    mode: str = "number"
    cand: int = -1
    support: np.ndarray | None = None
    candidate_key: tuple[str, str] | None = None


def _array(values: object, dtype: type, *shape: int) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).reshape(shape)
    if np.issubdtype(result.dtype, np.floating):
        np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def _window(values: np.ndarray, left: int, right: int) -> np.ndarray:
    """Read a local grid window without negative indexing or wrapping."""
    result = np.full(right - left, np.nan, dtype=np.float64)
    start, stop = max(0, left), min(len(values), right)
    if start < stop:
        result[start - left : stop - left] = values[start:stop]
    return result


class EpisodeBuilder:
    """Reuse cutoff statistics across planning, screening and episode assembly."""

    def __init__(
        self,
        panel: Panel,
        config: ScreeningConfig = ScreeningConfig(),
        *,
        stats: PanelStatsProvider | None = None,
    ) -> None:
        panel.validate()
        if stats is not None and stats.panel is not panel:
            raise ValueError("statistics provider must use the builder's panel")
        self.panel = panel
        self.config = config
        self.stats = PanelStatsProvider(panel) if stats is None else stats

    def _stats(self, target: str, cutoff: int) -> PanelStats:
        return self.stats.get(target, cutoff)

    def _validate(self, request: EpisodeRequest) -> int:
        origins = request.origins
        if not 0 <= request.item_index < len(self.panel.items):
            raise ValueError("item index outside the panel")
        if not origins or any(b <= a for a, b in zip(origins, origins[1:])):
            raise ValueError("origins must be nonempty and strictly ascending")
        if not PATCH <= request.context <= MAX_CONTEXT or request.context % PATCH:
            raise ValueError("context must be a multiple of PATCH in [16, 2048]")
        if not 1 <= request.horizon <= MAX_HORIZON:
            raise ValueError("horizon outside [1, 288]")
        if any((origins[-1] - origin) % PATCH for origin in origins):
            raise ValueError("origins must share a patch anchor")
        if origins[0] < origins[-1] - request.context + PATCH:
            raise ValueError("earliest origin must see at least one context patch")
        cutoff = origins[0] if request.cutoff is None else request.cutoff
        if cutoff > origins[0]:
            raise ValueError("statistics cutoff follows the earliest origin")
        if set(request.drop) - set(DROPPABLE):
            raise ValueError("unknown drop switch")
        targets = self.panel.items[request.item_index].targets
        if not targets or (request.target is not None and request.target not in targets):
            raise ValueError("a forecast target must exist")
        if request.target is None and len(targets) > MAX_FORECAST_ROWS:
            raise ValueError("more than 16 targets requires a designated target")
        return cutoff

    @staticmethod
    def _restricted(candidates: list[Candidate], request: EpisodeRequest) -> list[Candidate]:
        restricted = []
        for candidate in candidates:
            reason = candidate.removed
            if reason is None:
                if candidate.source in ("known", "past") and "features" in request.drop:
                    reason = "features"
                elif candidate.source == "sibling" and "siblings" in request.drop:
                    reason = "siblings"
                elif (
                    candidate.source != "static"
                    and request.keep_columns is not None
                    and candidate.name not in request.keep_columns
                ):
                    reason = "keep_columns"
            restricted.append(
                candidate if reason == candidate.removed else replace(candidate, removed=reason)
            )
        return restricted

    @staticmethod
    def _typed(candidates: list[Candidate]) -> bool:
        return any(
            c.removed is None and (c.source == "static" or c.kind == KIND_CATEGORY)
            for c in candidates
        )

    def plan_requests(
        self,
        item_index: int,
        origins: tuple[int, ...],
        horizon: int,
        *,
        context: int = MAX_CONTEXT,
        cutoff: int | None = None,
        drop: frozenset[str] = frozenset(),
        keep_columns: frozenset[str] | None = None,
    ) -> list[EpisodeRequest]:
        """Split multi-target items when typed context needs a designated column."""
        if not 0 <= item_index < len(self.panel.items):
            raise ValueError("item index outside the panel")
        targets = sorted(self.panel.items[item_index].targets)
        if not targets:
            raise ValueError("a forecast target must exist")
        probe = EpisodeRequest(
            item_index, origins, horizon, context, targets[0], cutoff, drop, keep_columns
        )
        end = self._validate(probe)
        candidates = screen(
            self.panel,
            self._stats(targets[0], end),
            item_index,
            (targets[0],),
            end,
            self.config,
            provider=self.stats,
        )
        split = len(targets) > MAX_FORECAST_ROWS or self._typed(self._restricted(candidates, probe))
        return [
            EpisodeRequest(
                item_index, origins, horizon, context, target, cutoff, drop, keep_columns
            )
            for target in (targets if split and len(targets) > 1 else [None])
        ]

    def _summary(
        self, scaled: np.ndarray, raw: np.ndarray, end: int, *, cutoff: int
    ) -> tuple[np.ndarray, np.ndarray]:
        left = max(0, end - BANK_WINDOW)
        values, flags = summary_vectors(
            _window(scaled, left, max(0, end))[None],
            _window(raw, left, max(0, end))[None],
            season=self.panel.season,
            long_lag=self.panel.long_lag,
            end=cutoff,
        )
        return values[0], flags[0]

    def _row(
        self,
        name: str,
        raw: np.ndarray,
        kind: int,
        role: int,
        score: float,
        end: int,
        *,
        cutoff: int,
    ) -> _Row:
        if kind == KIND_CATEGORY:
            return _Row(
                name,
                kind,
                role,
                score,
                raw,
                np.empty(0),
                Scale(0, 1),
                np.zeros(N_SUMMARY, np.float32),
                np.zeros(N_SUMMARY, bool),
                "category",
            )
        scale = (
            fit_scale(raw, max(0, end), self.panel.season) if kind == KIND_NUMBER else Scale(0, 1)
        )
        scaled = to_scaled(raw, scale) if kind == KIND_NUMBER else flag_codes(raw, end)
        summary, flags = self._summary(
            scaled, raw if kind == KIND_NUMBER else scaled, end, cutoff=cutoff
        )
        return _Row(
            name,
            kind,
            role,
            score,
            raw,
            scaled,
            scale,
            summary,
            flags,
            "number" if kind == KIND_NUMBER else "flag",
        )

    def _encode(
        self,
        row: _Row,
        left: int,
        right: int,
        target: np.ndarray,
        *,
        frozen_at: int | None = None,
        horizon: bool = False,
    ) -> np.ndarray:
        result = np.zeros((right - left, N_CHANNELS), dtype=np.float32)
        start, stop = max(0, left), min(right, len(row.raw))
        if row.mode in ("category", "effect"):
            stats, _ = category_step_stats(
                row.raw, target, left=left, right=right, frozen_at=frozen_at
            )
            observed = np.zeros(right - left, dtype=bool)
            if start < stop:
                observed[start - left : stop - left] = ~np.frompyfunc(is_missing, 1, 1)(
                    row.raw[start:stop]
                ).astype(bool)
            result[:, CH_OBSERVED] = observed
            if row.mode == "category":
                result[:, CH_VALUE_STATS:] = stats
            else:
                result[:, CH_VALUE] = stats[:, 0]
        else:
            values = _window(row.scaled, left, right)
            observed = np.isfinite(values)
            result[:, CH_VALUE] = np.where(
                observed, compress(values) if row.mode == "number" else values, 0.0
            )
            result[:, CH_OBSERVED] = observed
        if row.mode in ("number", "effect") and not horizon and start < stop:
            scaled, raw = row.scaled, row.raw
            if row.mode == "effect":
                effects, _ = category_step_stats(row.raw, target, left=0, right=stop)
                present = ~np.frompyfunc(is_missing, 1, 1)(row.raw[:stop]).astype(bool)
                # Effects already inhabit the target's scaled frame.
                scaled = raw = np.where(present, effects[:, 0], np.nan).astype(np.float64)
            values, flags = step_features(
                scaled,
                raw,
                season=self.panel.season,
                long_lag=self.panel.long_lag,
                left=start,
                right=stop,
            )
            result[start - left : stop - left, CH_STEP:CH_STEP_FLAG] = values
            result[start - left : stop - left, CH_STEP_FLAG:CH_VALUE_STATS] = flags
        result[~observed] = 0.0
        np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def _expand(
        self, row: _Row, end: int, target: np.ndarray, *, cutoff: int, include_effect: bool
    ) -> list[_Row]:
        prefix = row.raw[: max(0, end)]
        observed = ~np.frompyfunc(is_missing, 1, 1)(prefix).astype(bool)
        names = np.frompyfunc(label_of, 1, 1)(prefix[observed])
        unique, first, counts = np.unique(names, return_index=True, return_counts=True)
        order = np.lexsort((first, -counts))[:4]
        present = ~np.frompyfunc(is_missing, 1, 1)(row.raw).astype(bool)
        labels = np.frompyfunc(label_of, 1, 1)(row.raw)
        rows = []
        for label in unique[order]:
            codes = np.where(present, (labels == label).astype(np.float64), np.nan)
            summary, flags = self._summary(codes, codes, end, cutoff=cutoff)
            rows.append(
                _Row(
                    f"{row.name}={label}",
                    KIND_FLAG,
                    row.role,
                    row.score,
                    codes,
                    codes,
                    Scale(0, 1),
                    summary,
                    flags,
                    "flag",
                )
            )
        if not include_effect:
            return rows
        effects, _ = category_step_stats(
            row.raw, target, left=0, right=max(0, min(end, len(row.raw)))
        )
        raw = np.where(present[: len(effects)], effects[:, 0], np.nan)
        summary, flags = self._summary(raw, raw, end, cutoff=cutoff)
        rows.append(
            _Row(
                f"{row.name}~effect",
                KIND_NUMBER,
                row.role,
                row.score,
                row.raw,
                np.empty(0),
                Scale(0, 1),
                summary,
                flags,
                "effect",
            )
        )
        return rows

    def build(self, request: EpisodeRequest) -> Episode:
        """Materialise one episode, keeping unavailable input cells identically zero."""
        cutoff = self._validate(request)
        panel, item = self.panel, self.panel.items[request.item_index]
        targets = (request.target,) if request.target is not None else tuple(sorted(item.targets))
        stats = self._stats(targets[0], cutoff)
        candidates = self._restricted(
            screen(
                panel, stats, request.item_index, targets, cutoff, self.config, provider=self.stats
            ),
            request,
        )
        if len(targets) > 1 and self._typed(candidates):
            raise ValueError(
                "typed context requires one designated forecast column; use plan_requests"
            )
        context_target = targets[0] if len(targets) == 1 else None
        selected = select(candidates, feature_budget=len(candidates))
        end = cutoff - item.start
        left, right = request.origins[-1] - request.context, request.origins[-1]
        npatches, nh = request.context // PATCH, (request.horizon + PATCH - 1) // PATCH
        target_scaled = to_scaled(item.targets[targets[0]], stats.scales[request.item_index])
        rows = [
            self._row(name, item.targets[name], KIND_NUMBER, ROLE_FORECAST, 1.0, end, cutoff=cutoff)
            for name in targets
        ]
        for c in selected:
            if c.source == "sibling":
                row = self._row(
                    c.name,
                    item.targets[c.name],
                    KIND_NUMBER,
                    ROLE_SIBLING,
                    c.score,
                    end,
                    cutoff=cutoff,
                )
                row.candidate_key = c.source, c.name
                rows.append(row)
        dispositions: dict[tuple[str, str], str | None] = {
            (c.source, c.name): None for c in selected if c.source in ("sibling", "static")
        }
        features = []
        for c in selected:
            if c.source not in ("known", "past"):
                continue
            row = self._row(
                c.name,
                item.dynamic[c.name],
                int(c.kind),
                ROLE_KNOWN if c.source == "known" else ROLE_PAST,
                c.score,
                end,
                cutoff=cutoff,
            )
            expanded = (
                self._expand(
                    row,
                    end,
                    target_scaled,
                    cutoff=cutoff,
                    include_effect="value_effects" not in request.drop,
                )
                if row.kind == KIND_CATEGORY and "value_stats" in request.drop
                else [row]
            )
            key = c.source, c.name
            if not expanded:
                # A known category can have labels only after the statistics cutoff.
                dispositions[key] = None
            features.extend((part, key) for part in expanded)
        features.sort(key=lambda pair: (pair[0].role, -pair[0].score, pair[0].name))
        for row, key in features[:MAX_FEATURE_ROWS]:
            row.candidate_key = key
            rows.append(row)
            dispositions[key] = None

        trace = []
        for candidate in candidates:
            reason = candidate.removed or dispositions.get(
                (candidate.source, candidate.name), "budget"
            )
            trace.append(
                candidate if reason == candidate.removed else replace(candidate, removed=reason)
            )

        statics = (
            sorted((c for c in selected if c.source == "static"), key=lambda c: c.name)
            if context_target is not None
            else []
        )
        cand_names, cand_kind, cand_summary, cand_flags, cand_score = [], [], [], [], []
        attr_stats, attr_flags, attr_cand, attr_names = [], [], [], []
        num_values, num_flags, num_cand = [], [], []
        number_names, numbers, number_flags = stats.static_numbers(request.item_index)
        number_index = {name: i for i, name in enumerate(number_names)}
        peers = []
        for c in statics:
            ci = len(cand_names)
            cand_names.append(c.name)
            cand_kind.append(KIND_NUMBER if c.kind == "number" else KIND_CATEGORY)
            cand_score.append(c.score)
            tokens, flags = [], []
            for label in stats.labels(c.name, request.item_index):
                token, valid, support = stats.value_token(c.name, label, request.item_index)
                tokens.append(token[:N_SUMMARY])
                flags.append(valid)
                name = f"{c.name}={label}"
                if "attributes" not in request.drop:
                    attr_stats.append(token)
                    attr_flags.append(support > 0)
                    attr_cand.append(ci)
                    attr_names.append(name)
                if "peers" not in request.drop and panel.aligned:
                    values, counts = stats.peer_row(c.name, label, request.item_index, left, right)
                    if (counts > 0).any():
                        peers.append(
                            _Row(
                                name,
                                KIND_NUMBER,
                                ROLE_PEER,
                                c.score,
                                values,
                                values,
                                Scale(0, 1),
                                np.zeros(N_SUMMARY, np.float32),
                                np.zeros(N_SUMMARY, bool),
                                "peer",
                                ci,
                                counts,
                            )
                        )
            cand_summary.append(np.mean(tokens, axis=0) if tokens else np.zeros(N_SUMMARY))
            cand_flags.append(np.any(flags, axis=0) if flags else np.zeros(N_SUMMARY, bool))
            if c.kind == "number" and "static_numbers" not in request.drop:
                ni = number_index[c.name]
                num_values.append(numbers[ni])
                num_flags.append(number_flags[ni])
                num_cand.append(ci)

        panel_rows = []
        if context_target is not None and panel.aligned and "panel_row" not in request.drop:
            values, support = stats.panel_row(request.item_index, left, right)
            if (support > 0).any():
                score_left = max(item.start, cutoff - BANK_WINDOW)
                prefix, count = stats.panel_row(
                    request.item_index, score_left, max(score_left, cutoff)
                )
                own = _window(
                    target_scaled, score_left - item.start, max(score_left, cutoff) - item.start
                )
                score = relevance_number(
                    np.where(count > 0, prefix, np.nan), own, (0,), self.config.min_steps
                )
                panel_rows.append(
                    _Row(
                        "__panel__",
                        KIND_NUMBER,
                        ROLE_PANEL,
                        score,
                        values,
                        values,
                        Scale(0, 1),
                        np.zeros(N_SUMMARY, np.float32),
                        np.zeros(N_SUMMARY, bool),
                        "peer",
                        -1,
                        support,
                    )
                )
        rows.extend(
            sorted(peers, key=lambda row: (-row.score, row.name))[
                : MAX_CONTEXT_ROWS - len(panel_rows)
            ]
        )
        rows.extend(panel_rows)
        rows.sort(key=lambda row: (row.role, row.name))

        x, patch_valid = [], []
        for row in rows:
            if row.mode == "peer":
                data = np.zeros((request.context, N_CHANNELS), np.float32)
                data[:, CH_VALUE] = row.scaled
                data[:, CH_OBSERVED] = row.support > 0
                data[:, CH_SUPPORT] = np.log1p(row.support)
                valid = (row.support.reshape(npatches, PATCH) > 0).any(axis=1)
            else:
                data = self._encode(row, left - item.start, right - item.start, target_scaled)
                steps = np.arange(left, right).reshape(npatches, PATCH)
                valid = ((steps >= item.start) & (steps < item.start + len(row.raw))).any(axis=1)
            for family, span in STEP_FAMILIES.items():
                if family in request.drop:
                    data[:, CH_STEP + span.start : CH_STEP + span.stop] = 0
                    data[:, CH_STEP_FLAG + span.start : CH_STEP_FLAG + span.stop] = 0
            x.append(data.reshape(npatches, PATCH, N_CHANNELS))
            patch_valid.append(valid)
        forecast_rows = [i for i, r in enumerate(rows) if r.role in (ROLE_FORECAST, ROLE_SIBLING)]
        known_rows = [i for i, r in enumerate(rows) if r.role == ROLE_KNOWN]
        labels = np.zeros((len(request.origins), len(forecast_rows), nh * PATCH), np.float32)
        label_mask = np.zeros(labels.shape, bool)
        xh = np.zeros((len(request.origins), len(known_rows), nh * PATCH, N_CHANNELS), np.float32)
        for oi, origin in enumerate(request.origins):
            local = origin - item.start
            for fi, ri in enumerate(forecast_rows):
                raw = _window(rows[ri].raw, local, local + request.horizon)
                valid = np.isfinite(raw)
                labels[oi, fi, : request.horizon] = np.where(
                    valid, to_scaled(raw, rows[ri].scale), 0
                )
                label_mask[oi, fi, : request.horizon] = valid
            for ki, ri in enumerate(known_rows):
                xh[oi, ki, : request.horizon] = self._encode(
                    rows[ri],
                    local,
                    local + request.horizon,
                    target_scaled,
                    frozen_at=local,
                    horizon=True,
                )

        cal_phases = np.zeros((request.context, N_PHASES), np.float32)
        cal_ids = np.full((request.context, 6), -1, np.int16)
        calh_phases = np.zeros((len(request.origins), nh * PATCH, N_PHASES), np.float32)
        calh_ids = np.full((len(request.origins), nh * PATCH, 6), -1, np.int16)
        if panel.epoch is not None and "calendar" not in request.drop:
            cal_phases[:], cal_ids[:] = calendar_arrays(panel, left, request.context)
            for oi, origin in enumerate(request.origins):
                phases, ids = calendar_arrays(panel, origin, request.horizon)
                calh_phases[oi, : request.horizon] = phases
                calh_ids[oi, : request.horizon] = ids
        fallback = []
        for ri in forecast_rows:
            observed = rows[ri].raw[: max(0, end)]
            observed = observed[np.isfinite(observed)]
            fallback.append(not len(observed) or np.all(observed == observed[0]))
        r, g, a, k, o = (
            len(rows),
            len(cand_names),
            len(attr_stats),
            len(num_values),
            len(request.origins),
        )
        return Episode(
            x=_array(x, np.float32, r, npatches, PATCH, N_CHANNELS),
            patch_valid=_array(patch_valid, bool, r, npatches),
            row_kind=_array([r.kind for r in rows], np.int8, r),
            row_role=_array([r.role for r in rows], np.int8, r),
            row_names=[r.name for r in rows],
            row_summary=_array([r.summary for r in rows], np.float32, r, N_SUMMARY),
            row_summary_flags=_array([r.flags for r in rows], bool, r, N_SUMMARY),
            row_score=_array([r.score for r in rows], np.float32, r),
            row_cand=_array([r.cand for r in rows], np.int16, r),
            forecast_rows=_array(forecast_rows, np.int16, len(forecast_rows)),
            keep=_array(
                [rows[i].role == ROLE_FORECAST for i in forecast_rows], bool, len(forecast_rows)
            ),
            labels=labels,
            label_mask=label_mask,
            loc=_array([rows[i].scale.loc for i in forecast_rows], np.float64, len(forecast_rows)),
            scale=_array(
                [rows[i].scale.scale for i in forecast_rows], np.float64, len(forecast_rows)
            ),
            fallback=_array(fallback, bool, len(forecast_rows)),
            known_rows=_array(known_rows, np.int16, len(known_rows)),
            xh=xh.reshape(o, len(known_rows), nh, PATCH, N_CHANNELS),
            cal_phases=cal_phases.reshape(npatches, PATCH, N_PHASES),
            cal_ids=cal_ids.reshape(npatches, PATCH, 6),
            calh_phases=calh_phases.reshape(o, nh, PATCH, N_PHASES),
            calh_ids=calh_ids.reshape(o, nh, PATCH, 6),
            cand_names=cand_names,
            cand_kind=_array(cand_kind, np.int8, g),
            cand_summary=_array(cand_summary, np.float32, g, N_SUMMARY),
            cand_flags=_array(cand_flags, bool, g, N_SUMMARY),
            cand_score=_array(cand_score, np.float32, g),
            attr_stats=_array(attr_stats, np.float32, a, N_VALUE_TOKEN),
            attr_flags=_array(attr_flags, bool, a),
            attr_cand=_array(attr_cand, np.int16, a),
            attr_names=attr_names,
            static_num=_array(num_values, np.float32, k),
            static_num_flags=_array(num_flags, bool, k),
            static_num_cand=_array(num_cand, np.int16, k),
            freq_class=panel.freq_class,
            horizon=request.horizon,
            origin_patches=_array(
                [npatches - (right - origin) // PATCH for origin in request.origins], np.int16, o
            ),
            context_target=context_target,
            trace=tuple(trace),
            row_keys=tuple(row.candidate_key for row in rows),
        )
