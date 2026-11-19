"""Masked v5 training loss and original-unit panel forecasts with gate reports."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from demand_forecast.data.episode import (
    EpisodeBatch,
    EpisodeBuilder,
    EpisodeRequest,
    Panel,
    collate,
)
from demand_forecast.data.episode.schema import PATCH, STEP_FAMILIES
from demand_forecast.models.losses import pinball_loss

from .config import V5Config
from .model import V5Model, V5Output, prune


def _full_precision(value: Tensor) -> Tensor:
    """Promote half precision to float32; leave float32 and float64 as they are."""
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def v5_loss(
    output: V5Output, batch: EpisodeBatch, config: V5Config, *, label_cap: float | None = None
) -> tuple[Tensor, dict[str, float | int]]:
    """Supervise real non-fallback rows and add gate and balance costs.

    When set, ``label_cap`` excludes and counts cells beyond the absolute scaled limit.
    Dense expert mode supplies a zero balance cost through ``output.aux``.
    """
    eligible = (
        batch.origin_mask[:, :, None]
        & batch.forecast_mask[:, None, :]
        & ~batch.fallback[:, None, :]
    )
    horizon = output.quantiles.shape[-2]
    steps = torch.arange(horizon, device=batch.labels.device)
    real_steps = steps[None, :] < batch.horizon[:, None]
    real_steps = real_steps & batch.horizon_mask.repeat_interleave(PATCH, dim=-1)
    mask = batch.label_mask & real_steps[:, None, None, :]
    capped = 0
    with torch.autocast(output.quantiles.device.type, enabled=False):
        predictions = _full_precision(output.quantiles).flatten(0, 2)
        labels = _full_precision(batch.labels)
        if label_cap is not None:
            selected = mask & eligible[..., None]
            # Compare in float64: exact for float32 and float64 labels and for any finite cap.
            over_cap = labels.double().abs() > label_cap
            capped = int((selected & over_cap).sum().item())
            mask = selected & ~over_cap
        if label_cap is not None and not mask.any():
            # The empty sum stays connected to predictions and never reads a NaN.
            pinball, count = predictions[:0].sum(), 0
        else:
            pinball, count = pinball_loss(
                predictions,
                labels.flatten(0, 2),
                mask.flatten(0, 2),
                eligible.flatten(),
                predictions.new_tensor(config.quantiles),
            )
        sparsity = _full_precision(output.gates.expected_l0)
        balance = _full_precision(output.aux["balance"])
        loss = pinball + config.sparsity_weight * sparsity + config.balance_weight * balance
    parts = {
        "pinball": float(pinball.detach()),
        "sparsity": float(sparsity.detach()),
        "balance": float(balance.detach()),
        "total": float(loss.detach()),
        "n_labels": float(count),
    }
    if label_cap is not None:
        parts["capped"] = capped
    return loss, parts


@dataclass
class ForecastResult:
    """Kept forecasts in original units and every applied gate, reported by name."""

    quantiles: np.ndarray
    names: list[str]
    gates: dict[str, float]


@torch.no_grad()
def forecast(
    model: V5Model,
    panel: Panel,
    request: EpisodeRequest,
    *,
    builder: EpisodeBuilder | None = None,
    gates: str = "learned",
    pruned: bool = True,
) -> ForecastResult:
    """Build one episode, select/prune inputs, and trim kept forecasts to the horizon."""
    builder = EpisodeBuilder(panel) if builder is None else builder
    batch = collate([builder.build(request)]).to(next(model.parameters()).device)
    training = model.training
    try:
        model.eval()
        selected = model.gates(batch, mode=gates)
        report = {
            f"row:{name}": float(selected.row[0, i]) for i, name in enumerate(batch.row_names[0])
        }
        report.update(
            {
                f"candidate:{name}": float(selected.cand[0, i])
                for i, name in enumerate(batch.cand_names[0])
            }
        )
        report.update(
            {f"family:{name}": float(selected.family[0, i]) for i, name in enumerate(STEP_FAMILIES)}
        )
        if pruned:
            batch, selected = prune(batch, selected)
        output = model(batch, selected)
        quantiles = output.quantiles[0]
        quantiles = batch.loc[0, :, None, None] + batch.scale[0, :, None, None] * quantiles
        quantiles = torch.where(
            batch.fallback[0, :, None, None], batch.loc[0, :, None, None], quantiles
        )
        keep = batch.keep[0] & batch.forecast_mask[0]
        names = [batch.row_names[0][row] for row in batch.forecast_rows[0, keep].tolist()]
        return ForecastResult(quantiles[:, keep, : request.horizon].cpu().numpy(), names, report)
    finally:
        model.train(training)
