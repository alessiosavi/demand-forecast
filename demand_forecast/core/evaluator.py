"""Model evaluation for demand forecasting."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from demand_forecast.models.losses import compute_loss
from demand_forecast.utils.visualization import (
    plot_prediction_quality,
    plot_validation_results,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of model validation."""

    predictions: list[np.ndarray]
    actuals: list[np.ndarray]
    flatten_predictions: list[float]
    flatten_actuals: list[float]
    skus: list[int]
    avg_loss: float
    mse: float
    mae: float
    flatten_mse: float
    flatten_mae: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class Evaluator:
    """Evaluator for demand forecasting models.

    Handles validation on test data and metric computation.

    Attributes:
        model: The model to evaluate.
        criterion: Loss function.
        batch_size: Batch size.
        total_examples: Total number of test examples.
        flatten_loss: Whether to use flattened loss.
        metrics: Dictionary of torchmetrics to compute.
        plot_dir: Directory to save plots.
    """

    model: nn.Module
    criterion: nn.Module
    batch_size: int
    total_examples: int
    flatten_loss: bool = True
    metrics: dict[str, Any] = field(default_factory=dict)
    plot_dir: Path | None = None

    def validate(
        self,
        dataloader: Any,
        plot: bool = False,
        epoch: int | None = None,
    ) -> ValidationResult:
        """Validate model on test data.

        Args:
            dataloader: Test dataloader (RoundRobinIterator).
            plot: Whether to plot results.
            epoch: Current epoch number (for plot filenames).

        Returns:
            ValidationResult with predictions and metrics.
        """
        self.model.eval()
        total_loss = 0.0

        flatten_predictions: list[float] = []
        flatten_actuals: list[float] = []
        predictions: list[np.ndarray] = []
        actuals: list[np.ndarray] = []
        skus: list[int] = []

        total_steps = self.total_examples // self.batch_size

        with torch.no_grad():
            for item in tqdm(dataloader, total=total_steps, leave=False, desc="Validating"):
                model_idx = list(item.keys())[0]
                batch = item[model_idx]

                # Forward pass
                qty = batch["qty"]
                past_time = batch["past_time"]
                future_time = batch["future_time"]
                sku = batch["sku"]
                cats = {key: value.to(dtype=torch.int32) for key, value in batch["cats"].items()}
                targets = batch["y"]

                outputs = self.model(model_idx, qty, past_time, future_time, sku, cats)
                loss, flatten_outputs, flatten_targets = compute_loss(
                    outputs, targets, self.criterion, self.flatten_loss
                )
                total_loss += loss.item()

                # Store predictions and actuals
                flatten_predictions.extend(
                    flatten_outputs.squeeze().detach().cpu().numpy().tolist()
                )
                flatten_actuals.extend(flatten_targets.detach().cpu().numpy().tolist())
                predictions.extend(outputs.squeeze().detach().cpu().numpy())
                actuals.extend(targets.detach().cpu().numpy())
                skus.extend(batch["sku"].detach().cpu().numpy().tolist())

        avg_loss = total_loss / total_steps

        # Calculate metrics
        _actuals = np.array(actuals)
        _predictions = np.array(predictions)
        residuals = _actuals - _predictions

        _flatten_actuals = np.array(flatten_actuals)
        _flatten_predictions = np.array(flatten_predictions)
        flatten_residuals = _flatten_actuals - _flatten_predictions

        mse = float(np.mean(residuals**2))
        mae = float(np.mean(np.abs(residuals)))
        flatten_mse = float(np.mean(flatten_residuals**2))
        flatten_mae = float(np.mean(np.abs(flatten_residuals)))

        # Log results
        logger.info(
            f"Validation: Loss={avg_loss:.4f}, MSE={mse:.4f}, MAE={mae:.4f}, "
            f"Flatten_MSE={flatten_mse:.4f}, Flatten_MAE={flatten_mae:.4f}"
        )

        # Compute additional metrics
        _res_metric: dict[str, float] = {}
        if self.metrics:
            _p = torch.as_tensor(flatten_predictions)
            _a = torch.as_tensor(flatten_actuals)

            for metric_name, metric in self.metrics.items():
                try:
                    _res_metric[metric_name] = metric(_p, _a).item()
                except Exception:
                    logger.debug(f"Skipping metric {metric_name}")

        # Plot if requested
        if plot:
            # Determine save path
            save_path = None
            if self.plot_dir is not None:
                self.plot_dir.mkdir(parents=True, exist_ok=True)
                epoch_suffix = f"_epoch{epoch}" if epoch is not None else ""
                save_path = self.plot_dir / f"validation{epoch_suffix}.png"

            # Simple validation plot
            plot_validation_results(
                flatten_actuals,
                flatten_predictions,
                title=f"Validation Results{f' - Epoch {epoch}' if epoch else ''}",
                save_path=save_path,
                show=self.plot_dir is None,  # Only show if not saving
            )

            # Comprehensive quality plot
            if self.plot_dir is not None:
                quality_path = self.plot_dir / f"quality{epoch_suffix}.png"
                plot_prediction_quality(
                    actuals=_flatten_actuals,
                    predictions=_flatten_predictions,
                    title=f"Prediction Quality{f' - Epoch {epoch}' if epoch else ''}",
                    save_path=quality_path,
                    show=False,
                )

        return ValidationResult(
            predictions=predictions,
            actuals=actuals,
            flatten_predictions=flatten_predictions,
            flatten_actuals=flatten_actuals,
            skus=skus,
            avg_loss=avg_loss,
            mse=mse,
            mae=mae,
            flatten_mse=flatten_mse,
            flatten_mae=flatten_mae,
            metrics=_res_metric,
        )
