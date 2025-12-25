"""Confidence interval calculations for predictions."""

import numpy as np
import torch
from scipy.stats import norm


def calculate_confidence_intervals(
    predictions: np.ndarray | torch.Tensor,
    actuals: np.ndarray | torch.Tensor,
    confidence: float = 0.95,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Calculate confidence intervals for predictions.

    Uses the standard deviation of prediction errors to compute
    confidence bands around predictions.

    Args:
        predictions: Predicted values.
        actuals: Actual values.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bounds, upper_bounds).

    Raises:
        ValueError: If predictions and actuals have different shapes.

    Example:
        >>> preds = np.array([1.0, 2.0, 3.0])
        >>> actuals = np.array([1.1, 1.9, 3.2])
        >>> lower, upper = calculate_confidence_intervals(preds, actuals)
    """
    # Convert to tensors if needed
    if not isinstance(actuals, torch.Tensor):
        actuals = torch.from_numpy(actuals)
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.from_numpy(predictions)

    if predictions.shape != actuals.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs actuals {actuals.shape}"
        )

    # Compute errors
    errors = actuals - predictions

    # Compute standard deviation
    if errors.dim() > 1:
        std_devs = errors.std(dim=0)
    else:
        std_dev = torch.std(errors)
        std_devs = torch.full_like(predictions, std_dev)

    # Z-score for confidence level
    z = norm.ppf(1 - (1 - confidence) / 2)

    lower_bounds = predictions - (z * std_devs)
    upper_bounds = predictions + (z * std_devs)

    return lower_bounds, upper_bounds


def calculate_prediction_intervals(
    predictions: np.ndarray,
    residuals: np.ndarray,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate prediction intervals using historical residuals.

    Args:
        predictions: Predicted values.
        residuals: Historical prediction residuals (actuals - predictions).
        confidence: Confidence level.

    Returns:
        Tuple of (lower_bounds, upper_bounds).
    """
    # Calculate residual statistics
    std = np.std(residuals)
    z = norm.ppf(1 - (1 - confidence) / 2)

    lower_bounds = predictions - (z * std)
    upper_bounds = predictions + (z * std)

    return lower_bounds, upper_bounds


def calculate_quantile_predictions(
    residuals: np.ndarray,
    predictions: np.ndarray,
    quantiles: list = None,
) -> dict:
    """Calculate quantile-based prediction bounds.

    Args:
        residuals: Historical prediction residuals.
        predictions: Point predictions.
        quantiles: List of quantiles (default [0.1, 0.25, 0.5, 0.75, 0.9]).

    Returns:
        Dictionary mapping quantile names to prediction arrays.
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    result = {"point": predictions}

    for q in quantiles:
        residual_quantile = np.quantile(residuals, q)
        result[f"q{int(q * 100)}"] = predictions + residual_quantile

    return result
