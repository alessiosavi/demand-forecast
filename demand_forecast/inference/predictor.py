"""Prediction wrapper for demand forecasting models."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from demand_forecast.data.preprocessor import ScalerManager
from demand_forecast.inference.confidence import calculate_confidence_intervals
from demand_forecast.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of prediction."""

    predictions: np.ndarray
    flatten_predictions: np.ndarray
    skus: list[str]
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    confidence: float = 0.95


@dataclass
class Predictor:
    """Predictor for generating demand forecasts.

    Handles inference, inverse scaling, and confidence intervals.

    Attributes:
        model: Trained model.
        scaler_manager: Scalers for inverse transformation.
        sku_index_to_name: Mapping from indices to SKU names.
        device: Computation device.
    """

    model: nn.Module
    scaler_manager: ScalerManager | None = None
    sku_index_to_name: dict[int, str] | None = None
    device: torch.device | None = None

    def __post_init__(self):
        """Initialize predictor."""
        if self.device is None:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(
        self,
        batch: dict[str, torch.Tensor],
        cluster_id: str,
        return_confidence: bool = True,
        confidence_level: float = 0.95,
        validation_targets: torch.Tensor | None = None,
    ) -> PredictionResult:
        """Generate predictions for a batch.

        Args:
            batch: Batch of input data.
            cluster_id: Cluster model to use.
            return_confidence: Whether to compute confidence intervals.
            confidence_level: Confidence level for intervals.
            validation_targets: Optional targets for confidence calculation.

        Returns:
            PredictionResult with predictions and optional intervals.
        """
        qty = batch["qty"].to(self.device)
        past_time = batch["past_time"].to(self.device)
        future_time = batch["future_time"].to(self.device)
        sku = batch["sku"].to(self.device)
        cats = {
            key: value.to(dtype=torch.int32, device=self.device)
            for key, value in batch["cats"].items()
        }

        with torch.no_grad():
            outputs = self.model(cluster_id, qty, past_time, future_time, sku, cats)

        predictions = outputs.cpu().numpy()
        flatten_predictions = predictions.sum(axis=-1) if predictions.ndim > 1 else predictions

        # Get SKU names
        sku_indices = batch["sku"].cpu().numpy()
        if self.sku_index_to_name:
            sku_names = [self.sku_index_to_name.get(idx, f"sku_{idx}") for idx in sku_indices]
        else:
            sku_names = [f"sku_{idx}" for idx in sku_indices]

        # Calculate confidence intervals if requested
        lower_bounds = None
        upper_bounds = None

        if return_confidence and validation_targets is not None:
            targets = validation_targets.cpu().numpy()
            flatten_targets = targets.sum(axis=-1) if targets.ndim > 1 else targets

            lower_bounds, upper_bounds = calculate_confidence_intervals(
                flatten_predictions, flatten_targets, confidence_level
            )
            lower_bounds = lower_bounds.numpy()
            upper_bounds = upper_bounds.numpy()

        return PredictionResult(
            predictions=predictions,
            flatten_predictions=flatten_predictions,
            skus=sku_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            confidence=confidence_level,
        )

    def predict_dataloader(
        self,
        dataloader: Any,
        confidence_level: float = 0.95,
    ) -> PredictionResult:
        """Generate predictions for an entire dataloader.

        Args:
            dataloader: RoundRobin iterator of batches.
            confidence_level: Confidence level for intervals.

        Returns:
            Combined PredictionResult for all batches.
        """
        all_predictions = []
        all_flatten_predictions = []
        all_skus = []
        all_targets = []

        for item in dataloader:
            cluster_id = list(item.keys())[0]
            batch = item[cluster_id]

            result = self.predict_batch(batch, cluster_id, return_confidence=False)

            all_predictions.extend(result.predictions)
            all_flatten_predictions.extend(result.flatten_predictions)
            all_skus.extend(result.skus)

            if "y" in batch:
                all_targets.extend(batch["y"].cpu().numpy())

        # Combine results
        predictions = np.array(all_predictions)
        flatten_predictions = np.array(all_flatten_predictions)

        # Calculate confidence intervals
        lower_bounds = None
        upper_bounds = None

        if all_targets:
            targets = np.array(all_targets)
            flatten_targets = targets.sum(axis=-1) if targets.ndim > 1 else targets

            lower_bounds, upper_bounds = calculate_confidence_intervals(
                torch.from_numpy(flatten_predictions),
                torch.from_numpy(flatten_targets),
                confidence_level,
            )
            lower_bounds = lower_bounds.numpy()
            upper_bounds = upper_bounds.numpy()

        return PredictionResult(
            predictions=predictions,
            flatten_predictions=flatten_predictions,
            skus=all_skus,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            confidence=confidence_level,
        )

    def inverse_scale(
        self,
        predictions: np.ndarray,
        cluster_id: str,
    ) -> np.ndarray:
        """Inverse scale predictions to original values.

        Args:
            predictions: Scaled predictions.
            cluster_id: Cluster ID for scaler lookup.

        Returns:
            Predictions in original scale.
        """
        if self.scaler_manager is None:
            logger.warning("No scaler manager available, returning raw predictions")
            return predictions

        return self.scaler_manager.inverse_transform(cluster_id, predictions)

    def to_dataframe(
        self,
        result: PredictionResult,
        include_confidence: bool = True,
    ) -> pd.DataFrame:
        """Convert prediction result to DataFrame.

        Args:
            result: Prediction result.
            include_confidence: Whether to include confidence bounds.

        Returns:
            DataFrame with predictions.
        """
        data = {
            "sku": result.skus,
            "prediction": result.flatten_predictions,
        }

        if include_confidence and result.lower_bounds is not None:
            data["lower_bound"] = result.lower_bounds
            data["upper_bound"] = result.upper_bounds
            data["confidence"] = result.confidence

        return pd.DataFrame(data)

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Path,
        scaler_path: Path | None = None,
        device: torch.device | None = None,
    ) -> "Predictor":
        """Load predictor from checkpoint.

        Args:
            model_path: Path to saved model.
            scaler_path: Path to saved scalers.
            device: Computation device.

        Returns:
            Initialized Predictor.
        """
        if device is None:
            device = torch.device("cpu")

        model = load_checkpoint(model_path, map_location=device)

        scaler_manager = None
        if scaler_path and scaler_path.exists():
            scaler_manager = ScalerManager.load(scaler_path)

        return cls(
            model=model,
            scaler_manager=scaler_manager,
            device=device,
        )
