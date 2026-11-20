"""Absolute v5 forecasts on fixed generated series, with classical controls."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import isfinite

import numpy as np
import pandas as pd

from demand_forecast.data.episode import EpisodeBatch, Item, Panel, ScreeningConfig
from demand_forecast.data.episode.collate import (
    EpisodeBatchWithNaive,
    EpisodeBatchWithNaiveEvidence,
    EpisodeBatchWithRowKeys,
)
from demand_forecast.data.episode.derived import naive_series
from demand_forecast.data.episode.schema import NAIVE_OPTIONS
from demand_forecast.models.v5 import V5Model, V5Output, forecast_items
from demand_forecast.models.v5.export import _screening

_ORIGIN = 2208
_HORIZON = 96
_CONTEXT = 512
_LENGTH = 2400
_ITEMS = 24
_BATCH_EPISODES = 16
_SHAPE = ("seasonal_h", "seasonal_h2", "seasonal_d", "trend", "seasonal_trend")
_FAMILIES = (*_SHAPE, "noise", "intermittent", "level_shift")
_NAIVE_SUFFIXES = ("~last", "~season", "~season2", "~drift")


@dataclass(frozen=True)
class ProbeSettings:
    """Scoring presentation shared by a checkpoint and its export."""

    screening: ScreeningConfig = ScreeningConfig()
    derive: frozenset[str] = frozenset()

    @classmethod
    def from_training(cls, training: dict) -> "ProbeSettings":
        return cls(_screening(training), frozenset(training["sampler"].get("derive", ())))


def probe_panels() -> dict[str, Panel]:
    """Generate the eight fixed panels, using one fixed NumPy seed per family."""
    panels = {}
    time = np.arange(_LENGTH)
    epoch = pd.Timestamp("2024-01-01")
    for seed, name in enumerate(_FAMILIES):
        rng = np.random.default_rng(seed)
        hourly = name in ("seasonal_h", "seasonal_h2")
        season = 24 if hourly else 7
        items = []
        for phase in range(_ITEMS):
            if name == "intermittent":
                values = np.where(
                    rng.random(_LENGTH) < 0.7, 0.0, 1.0 + rng.exponential(4.0, _LENGTH)
                )
            else:
                sd = {"noise": 5, "level_shift": 2}.get(name, 1)
                values = 50.0 + rng.normal(0, sd, _LENGTH)
                if name.startswith("seasonal"):
                    values += 10 * np.sin(2 * np.pi * time / season + phase)
                if name == "seasonal_h2":
                    values += 10 * np.sin(2 * np.pi * time / 168 + 2 * phase)
                if name in ("trend", "seasonal_trend"):
                    values += 0.2 * time
                if name == "level_shift":
                    values[2008:] += 30
            items.append(Item(f"{name}-{phase}", 0, {"y": values}, {}, {}))
        panels[name] = Panel(name, tuple(items), {}, "h" if hourly else "D", epoch, True)
    return panels


def _naive(history: np.ndarray, lag: int) -> np.ndarray:
    return history[:, -lag:][:, np.arange(_HORIZON) % lag]


def _reference_path(panel: Panel, history: np.ndarray) -> np.ndarray:
    if panel.name in ("seasonal_h", "seasonal_d"):
        season = panel.season
        phases = history[:, -4 * season :].reshape(-1, 4, season).mean(axis=1)
        return _naive(phases, season)
    if panel.name == "seasonal_h2":
        return _naive(history, 168)
    if panel.name in ("trend", "seasonal_trend"):
        origin = history.shape[1]
        return np.stack(
            [
                naive_series(
                    item,
                    season=panel.season,
                    long_lag=None,
                    left=origin,
                    right=origin + _HORIZON,
                    origin=origin,
                )["drift"][0]
                for item in history
            ]
        )
    if panel.name == "level_shift":
        level = history[:, -28:].mean(axis=1)
    else:
        level = np.median(history, axis=1)
    return np.broadcast_to(level[:, None], (len(history), _HORIZON))


def _mase_scale(history: np.ndarray, season: int) -> np.ndarray:
    return np.maximum(1e-8, np.abs(history[:, season:] - history[:, :-season]).mean(axis=1))


def _series(panel: Panel) -> np.ndarray:
    return np.stack([item.targets["y"] for item in panel.items])


def flat(panel: Panel, levels: Sequence[float]) -> np.ndarray:
    """Repeat each item's empirical history quantiles at every horizon cell."""
    quantiles = np.quantile(_series(panel)[:, :_ORIGIN], levels, axis=1).T
    return np.broadcast_to(quantiles[:, None, :], (len(panel.items), _HORIZON, len(levels)))


def reference(panel: Panel, levels: Sequence[float]) -> np.ndarray:
    """Add empirical horizon-specific backtest error quantiles to the reference path."""
    history = _series(panel)[:, :_ORIGIN]
    path = _reference_path(panel, history)
    errors = []
    for k in range(1, 9):
        origin = _ORIGIN - _HORIZON * k
        past = history[:, :origin]
        truth = history[:, origin : origin + _HORIZON]
        errors.append(
            (truth - _reference_path(panel, past)) / _mase_scale(past, panel.season)[:, None]
        )
    offsets = np.quantile(np.concatenate(errors, axis=0), levels, axis=0).T
    return path[..., None] + _mase_scale(history, panel.season)[:, None, None] * offsets


def _finite(value: float) -> float | None:
    return float(value) if isfinite(value) else None


def _mase(truth: np.ndarray, path: np.ndarray, scale: np.ndarray) -> float | None:
    return _finite((np.abs(truth - path).mean(axis=1) / scale).mean())


def _point_figures(truth: np.ndarray, scale: np.ndarray, path: np.ndarray) -> dict:
    centred_truth = truth - truth.mean(axis=1, keepdims=True)
    centred_path = path - path.mean(axis=1, keepdims=True)
    truth_sd, path_sd = truth.std(axis=1), path.std(axis=1)
    denominator = truth_sd * path_sd
    correlation = np.divide(
        (centred_truth * centred_path).mean(axis=1),
        denominator,
        out=np.zeros(len(truth)),
        where=denominator != 0,
    )
    correlation[np.ptp(path, axis=1) == 0] = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = (path_sd / truth_sd).mean()
    return {
        "mase": _mase(truth, path, scale),
        "correlation": _finite(correlation.mean()),
        "spread_ratio": _finite(spread),
    }


def _forecast_figures(
    truth: np.ndarray, scale: np.ndarray, quantiles: np.ndarray, levels: Sequence[float]
) -> dict:
    levels = list(levels)
    if 0.5 in levels:
        figures = _point_figures(truth, scale, quantiles[..., levels.index(0.5)])
    else:
        figures = {"mase": None, "correlation": None, "spread_ratio": None}
    coverage = []
    for knot in np.moveaxis(quantiles, -1, 0):
        if np.isfinite(knot).all() and np.isfinite(truth).all():
            coverage.append(
                {"lt": float((truth < knot).mean()), "le": float((truth <= knot).mean())}
            )
        else:
            coverage.append({"lt": None, "le": None})
    figures["coverage"] = coverage
    if len(levels) > 1:
        figures["min_mean_gap"] = _finite(np.diff(quantiles, axis=-1).mean(axis=(0, 1)).min())
    return figures


def score_panel(panel: Panel, quantiles: np.ndarray, levels: Sequence[float]) -> dict:
    """Score one family's forecast, with quantiles in item, horizon, knot order."""
    series = _series(panel)
    history, truth = series[:, :_ORIGIN], series[:, _ORIGIN : _ORIGIN + _HORIZON]
    season = panel.season
    scale = _mase_scale(history, season)
    return {
        **_forecast_figures(truth, scale, quantiles, levels),
        "naive_mase": _mase(truth, _naive(history, season), scale),
        "reference_mase": _mase(truth, _reference_path(panel, history), scale),
        "level_mase": _mase(truth, np.median(history, axis=1, keepdims=True), scale),
    }


def zero_naive_rows(batch: EpisodeBatch) -> EpisodeBatch:
    """Zero known naive horizon values and flags while retaining their rows."""
    if not isinstance(batch, EpisodeBatchWithRowKeys):
        # A silent pass-through would report the ablated figures as equal to the model's own.
        raise TypeError("zero_naive_rows needs a batch with row keys (a forecast_items hook)")
    xh = batch.xh.clone()
    for i, (names, keys, rows, valid) in enumerate(
        zip(
            batch.row_names,
            batch.row_keys,
            batch.known_rows.tolist(),
            batch.known_mask.tolist(),
            strict=True,
        )
    ):
        slots = [
            slot
            for slot, (row, known) in enumerate(zip(rows, valid, strict=True))
            if known
            and keys[row] == ("derived", names[row])
            and names[row].endswith(_NAIVE_SUFFIXES)
        ]
        xh[i, :, slots] = 0
    return replace(batch, xh=xh)


def anchor_off(batch: EpisodeBatch) -> EpisodeBatch:
    """Clear path availability while preserving the decoder's inputs."""
    if isinstance(batch, EpisodeBatchWithNaive):
        return replace(batch, naive_seen=batch.naive_seen.new_zeros(batch.naive_seen.shape))
    return batch


def fan_off(batch: EpisodeBatch) -> EpisodeBatch:
    """Zero residual quantiles while retaining the mix prior and head gain."""
    if isinstance(batch, EpisodeBatchWithNaiveEvidence):
        return replace(batch, naive_resq=batch.naive_resq.new_zeros(batch.naive_resq.shape))
    return batch


def evidence_off(batch: EpisodeBatch) -> EpisodeBatch:
    """Clear the evidence mask so the forecast uses the original mix and head."""
    if isinstance(batch, EpisodeBatchWithNaiveEvidence):
        return replace(
            batch, naive_error_seen=batch.naive_error_seen.new_zeros(batch.naive_error_seen.shape)
        )
    return batch


def _score_model(
    model: V5Model,
    panel: Panel,
    settings: ProbeSettings,
    *,
    batch_hook: Callable[[EpisodeBatch], EpisodeBatch] | None = None,
    observe_mix: bool = False,
) -> dict:
    anchors, weights, label_masks, evidence_masks = [], [], [], []

    def observe(batch: EpisodeBatch, output: V5Output) -> None:
        # Probe jobs have one origin and one kept target per item, in item order.
        keep = batch.keep & batch.forecast_mask & batch.origin_mask[:, :1]
        label_masks.append(batch.label_mask[:, 0][keep][:, :_HORIZON].cpu().numpy())
        if output.evidence is not None:
            evidence_masks.append(output.evidence[:, 0][keep][:, :_HORIZON].cpu().numpy())
        if not observe_mix:
            return
        loc = batch.loc[keep][:, None]
        if output.anchor is None:
            anchors.append(loc.expand(-1, _HORIZON).cpu().numpy())
            weights.append(np.zeros((len(loc), _HORIZON, len(NAIVE_OPTIONS))))
            return
        anchor = loc + batch.scale[keep][:, None] * output.anchor[:, 0][keep][:, :_HORIZON]
        anchor = anchor.where(~batch.fallback[keep][:, None], loc)
        anchors.append(anchor.cpu().numpy())
        weights.append(output.mix[:, 0][keep][:, :_HORIZON].cpu().numpy())

    jobs = [(i, _ORIGIN) for i in range(len(panel.items))]
    forecasts = forecast_items(
        model,
        panel,
        jobs,
        _HORIZON,
        context=_CONTEXT,
        screening=settings.screening,
        derive=settings.derive,
        batch_episodes=_BATCH_EPISODES,
        batch_hook=batch_hook,
        output_hook=observe if observe_mix or model.config.mix_evidence else None,
    )
    quantiles = np.stack([forecasts.values[i, "y"] for i, _ in jobs])
    levels = model.config.quantiles
    figures = score_panel(panel, quantiles, levels)
    if evidence_masks:
        scored = np.concatenate(evidence_masks)[np.concatenate(label_masks)]
        figures["evidence_share"] = float(scored.mean()) if len(scored) else None
    if observe_mix:
        quantiles[..., levels.index(0.5)] = np.concatenate(anchors)
        figures["anchor_only"] = score_panel(panel, quantiles, levels)
        mix = np.concatenate(weights)[np.concatenate(label_masks)]
        if len(mix):
            mean_weights = mix.mean(axis=0, dtype=np.float64).tolist()
            no_seen_share = float((~mix.any(axis=-1)).mean())
        else:
            mean_weights = [None] * len(NAIVE_OPTIONS)
            no_seen_share = None
        figures["mix_mean"] = dict(zip(NAIVE_OPTIONS, mean_weights, strict=True))
        figures["mix_no_seen_share"] = no_seen_share
    return figures


def run_probe(model: V5Model, settings: ProbeSettings, *, device: str = "cpu") -> dict:
    """Measure the probe in eval mode through the ordinary batch scoring path."""
    original_device = next(model.parameters()).device
    families = {}
    try:
        model.to(device)
        for name, panel in probe_panels().items():
            figures = _score_model(model, panel, settings, observe_mix=model.config.naive_mix)
            if "naive_rows" in settings.derive:
                figures["zeroed_naive_rows"] = _score_model(
                    model, panel, settings, batch_hook=zero_naive_rows
                )
            if model.config.naive_mix:
                figures["anchor_off"] = _score_model(model, panel, settings, batch_hook=anchor_off)
            if model.config.mix_evidence:
                figures["fan_off"] = _score_model(model, panel, settings, batch_hook=fan_off)
                figures["evidence_off"] = _score_model(
                    model, panel, settings, batch_hook=evidence_off
                )
            families[name] = figures
    finally:
        model.to(original_device)
    return {"levels": list(model.config.quantiles), "families": families}


def _defined(value: float | None) -> bool:
    return value is not None and isfinite(value)


def _failure(name: str, figures: dict, levels: Sequence[float]) -> str | None:
    for key in ("mase", "naive_mase", "reference_mase", "level_mase"):
        if not _defined(figures.get(key)):
            return key
    naive, level = figures["naive_mase"], figures["level_mase"]
    if name in _SHAPE:
        limit = naive
    elif name == "noise":
        limit = 1.05 * level
    else:
        limit = 1.05 * min(naive, level)
    if figures["mase"] > limit:
        return "mase"
    correlation = figures.get("correlation")
    if not _defined(correlation) or (name in _SHAPE and correlation < 0.5):
        return "correlation"
    coverage = figures.get("coverage", ())
    if not levels or len(coverage) != len(levels):
        return "coverage"
    for level, knot in zip(levels, coverage, strict=True):
        lt, le = knot.get("lt"), knot.get("le")
        if not all(map(_defined, (level, lt, le))) or not lt - 0.15 <= level <= le + 0.15:
            return "coverage"
    if len(levels) > 1:
        gap = figures.get("min_mean_gap")
        if not _defined(gap) or (name != "intermittent" and gap <= 0):
            return "quantile_gap"
    if not _defined(figures.get("spread_ratio")):
        return "spread_ratio"
    return None


def gate(report: dict) -> dict:
    """Record the first failing criterion per family and the overall result."""
    families = {}
    for name in _FAMILIES:
        criterion = _failure(name, report["families"].get(name, {}), report["levels"])
        families[name] = {"passed": criterion is None, "criterion": criterion}
    return {"families": families, "passed": all(family["passed"] for family in families.values())}
