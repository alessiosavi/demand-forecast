"""Loss functions for demand forecasting."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def compute_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    flatten: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute loss with optional flattening.

    Args:
        outputs: Model predictions, shape (batch_size, n_out).
        targets: Ground truth, shape (batch_size, n_out).
        criterion: Loss function (e.g., MSELoss).
        flatten: If True, sum across time dimension before computing loss.

    Returns:
        Tuple of (loss, processed_outputs, processed_targets).
    """
    if flatten:
        # Sum across time dimension for "global batch distance"
        flatten_outputs = torch.sum(outputs, dim=-1)
        flatten_targets = torch.sum(targets, dim=-1)
    else:
        flatten_outputs = outputs
        flatten_targets = targets

    loss = criterion(flatten_outputs, flatten_targets)

    return loss, flatten_outputs, flatten_targets


class WeightedMSELoss(nn.Module):
    """MSE loss with time step weighting.

    Allows giving more weight to certain time steps (e.g., near-term predictions).
    """

    def __init__(self, weights: torch.Tensor = None, n_out: int = 16):
        """Initialize weighted MSE loss.

        Args:
            weights: Tensor of weights per time step. If None, uses uniform weights.
            n_out: Number of output time steps (used if weights is None).
        """
        super().__init__()
        if weights is None:
            weights = torch.ones(n_out)
        self.register_buffer("weights", weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss.

        Args:
            pred: Predictions, shape (batch_size, n_out).
            target: Targets, shape (batch_size, n_out).

        Returns:
            Scalar loss value.
        """
        mse = (pred - target) ** 2
        weighted_mse = mse * self.weights
        return weighted_mse.mean()


class QuantileLoss(nn.Module):
    """Quantile loss for probabilistic forecasting.

    Useful for generating prediction intervals.
    """

    def __init__(self, quantiles: list = None):
        """Initialize quantile loss.

        Args:
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
        """
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.register_buffer("quantiles", torch.tensor(quantiles))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss.

        Args:
            pred: Predictions, shape (batch_size, n_out, n_quantiles).
            target: Targets, shape (batch_size, n_out).

        Returns:
            Scalar loss value.
        """
        target = target.unsqueeze(-1)  # (batch_size, n_out, 1)
        errors = target - pred
        losses = torch.max(
            (self.quantiles - 1) * errors,
            self.quantiles * errors,
        )
        return losses.mean()


class CombinedForecastLoss(nn.Module):
    """Combined loss for probabilistic forecasting with decomposition.

    Combines multiple loss components:
    - Point loss: Huber loss for robust point prediction
    - Quantile loss: Pinball loss for prediction intervals
    - Decomposition consistency: Ensures trend + seasonality = prediction

    This loss is designed for the AdvancedDemandForecastModelV2 which outputs
    point predictions, quantiles, and optional decomposition components.
    """

    def __init__(
        self,
        quantiles: list[float] = None,
        point_weight: float = 1.0,
        quantile_weight: float = 0.5,
        decomposition_weight: float = 0.1,
        huber_delta: float = 1.0,
    ):
        """Initialize combined loss.

        Args:
            quantiles: List of quantiles for probabilistic prediction.
            point_weight: Weight for point prediction loss.
            quantile_weight: Weight for quantile loss.
            decomposition_weight: Weight for decomposition consistency loss.
            huber_delta: Delta parameter for Huber loss.
        """
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.register_buffer("quantiles", torch.tensor(quantiles))

        self.point_weight = point_weight
        self.quantile_weight = quantile_weight
        self.decomposition_weight = decomposition_weight
        self.huber_delta = huber_delta

    def _huber_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss (smooth L1).

        Args:
            pred: Predictions of shape (batch, n_out).
            target: Targets of shape (batch, n_out).

        Returns:
            Scalar loss value.
        """
        return F.smooth_l1_loss(pred, target, beta=self.huber_delta)

    def _quantile_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute quantile (pinball) loss.

        Args:
            pred: Quantile predictions of shape (batch, n_out, n_quantiles).
            target: Targets of shape (batch, n_out).

        Returns:
            Scalar loss value.
        """
        target = target.unsqueeze(-1)  # (batch, n_out, 1)
        errors = target - pred
        losses = torch.max(
            (self.quantiles - 1) * errors,
            self.quantiles * errors,
        )
        return losses.mean()

    def _decomposition_consistency_loss(
        self,
        prediction: torch.Tensor,
        trend: torch.Tensor,
        seasonality: torch.Tensor,
    ) -> torch.Tensor:
        """Compute decomposition consistency loss.

        Ensures that trend + seasonality approximately equals the prediction.

        Args:
            prediction: Point predictions of shape (batch, n_out).
            trend: Trend component of shape (batch, n_out).
            seasonality: Seasonality component of shape (batch, n_out).

        Returns:
            Scalar loss value.
        """
        reconstructed = trend + seasonality
        return F.mse_loss(reconstructed, prediction)

    def forward(
        self,
        outputs: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined loss.

        Args:
            outputs: Model outputs. Either a tensor of shape (batch, n_out) for
                simple models, or a dict with keys:
                - 'prediction': Point predictions (batch, n_out)
                - 'quantiles': Quantile predictions (batch, n_out, n_quantiles)
                - 'trend': Optional trend component (batch, n_out)
                - 'seasonality': Optional seasonality component (batch, n_out)
            targets: Ground truth of shape (batch, n_out).

        Returns:
            Tuple of (total_loss, loss_components_dict).
            loss_components_dict contains individual loss values for logging.
        """
        loss_components = {}

        # Handle simple tensor output (backward compatibility)
        if isinstance(outputs, torch.Tensor):
            point_loss = self._huber_loss(outputs, targets)
            loss_components["point_loss"] = point_loss
            return point_loss, loss_components

        # Handle dict output from advanced model
        prediction = outputs.get("prediction")
        quantiles = outputs.get("quantiles")
        trend = outputs.get("trend")
        seasonality = outputs.get("seasonality")

        total_loss = torch.tensor(0.0, device=targets.device)

        # Point prediction loss
        if prediction is not None:
            point_loss = self._huber_loss(prediction, targets)
            loss_components["point_loss"] = point_loss
            total_loss = total_loss + self.point_weight * point_loss

        # Quantile loss
        if quantiles is not None:
            quantile_loss = self._quantile_loss(quantiles, targets)
            loss_components["quantile_loss"] = quantile_loss
            total_loss = total_loss + self.quantile_weight * quantile_loss

        # Decomposition consistency loss
        if trend is not None and seasonality is not None and prediction is not None:
            decomp_loss = self._decomposition_consistency_loss(prediction, trend, seasonality)
            loss_components["decomposition_loss"] = decomp_loss
            total_loss = total_loss + self.decomposition_weight * decomp_loss

        loss_components["total_loss"] = total_loss
        return total_loss, loss_components


class SMAPELoss(nn.Module):
    """Symmetric Mean Absolute Percentage Error loss.

    SMAPE is scale-independent and bounded between 0 and 200%,
    making it useful for comparing across different SKUs.
    """

    def __init__(self, epsilon: float = 1e-8):
        """Initialize SMAPE loss.

        Args:
            epsilon: Small constant for numerical stability.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SMAPE loss.

        Args:
            pred: Predictions of shape (batch, n_out).
            target: Targets of shape (batch, n_out).

        Returns:
            Scalar loss value (percentage).
        """
        numerator = torch.abs(pred - target)
        denominator = torch.abs(pred) + torch.abs(target) + self.epsilon
        smape = 200.0 * numerator / denominator
        return smape.mean()


class MASELoss(nn.Module):
    """Mean Absolute Scaled Error loss.

    MASE normalizes errors by the in-sample naive forecast error,
    making it scale-independent and suitable for intermittent demand.
    """

    def __init__(self, seasonality: int = 1):
        """Initialize MASE loss.

        Args:
            seasonality: Seasonal period for naive forecast (1 = non-seasonal).
        """
        super().__init__()
        self.seasonality = seasonality

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MASE loss.

        Args:
            pred: Predictions of shape (batch, n_out).
            target: Targets of shape (batch, n_out).
            history: Historical values for scaling (batch, history_len).
                If None, uses target mean as scale.

        Returns:
            Scalar loss value.
        """
        mae = torch.abs(pred - target).mean(dim=-1)

        if history is not None and history.shape[-1] > self.seasonality:
            # In-sample naive forecast error
            naive_errors = torch.abs(
                history[:, self.seasonality :] - history[:, : -self.seasonality]
            )
            scale = naive_errors.mean(dim=-1) + 1e-8
        else:
            # Fallback: use target mean as scale
            scale = torch.abs(target).mean(dim=-1) + 1e-8

        mase = mae / scale
        return mase.mean()
