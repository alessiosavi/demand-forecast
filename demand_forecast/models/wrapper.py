"""Model wrapper for multi-cluster forecasting."""

from typing import Literal

import torch
import torch.nn as nn

from demand_forecast.models.lightweight import LightweightDemandModel, LightweightMixerModel
from demand_forecast.models.transformer import AdvancedDemandForecastModel
from demand_forecast.models.transformer_v2 import AdvancedDemandForecastModelV2

# Type alias for all supported model types
ModelType = (
    AdvancedDemandForecastModel
    | AdvancedDemandForecastModelV2
    | LightweightDemandModel
    | LightweightMixerModel
)

# Mapping from model type string to class
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "standard": AdvancedDemandForecastModel,
    "advanced": AdvancedDemandForecastModelV2,
    "lightweight": LightweightDemandModel,
    "lightweight_tcn": LightweightDemandModel,
    "lightweight_mixer": LightweightMixerModel,
}


def create_model(
    model_type: str,
    **kwargs,
) -> nn.Module:
    """Factory function to create a model by type.

    Args:
        model_type: One of "standard", "advanced", "lightweight", "lightweight_mixer".
        **kwargs: Model-specific arguments.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type not in MODEL_REGISTRY:
        valid_types = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_type '{model_type}'. Valid options: {valid_types}")

    model_class = MODEL_REGISTRY[model_type]
    return model_class(**kwargs)


class ModelWrapper(nn.Module):
    """Wrapper for multiple cluster-specific models.

    Maintains a dictionary of model instances, one per cluster/bin.
    Routes forward passes to the appropriate model based on cluster ID.
    Supports multiple model architectures through the model_type parameter.

    Attributes:
        models: ModuleDict mapping cluster IDs to models.
        model_type: Type of model architecture being used.
    """

    def __init__(
        self,
        n: int,
        model_type: Literal[
            "standard", "advanced", "lightweight", "lightweight_mixer"
        ] = "standard",
        **kwargs,
    ):
        """Initialize model wrapper.

        Args:
            n: Number of clusters/models to create.
            model_type: Type of model to use for each cluster.
                - "standard": AdvancedDemandForecastModel (original transformer)
                - "advanced": AdvancedDemandForecastModelV2 (research-grade with TFT/PatchTST)
                - "lightweight": LightweightDemandModel (TCN-based, CPU-optimized)
                - "lightweight_mixer": LightweightMixerModel (MLP-Mixer based)
            **kwargs: Arguments passed to each model instance.
        """
        super().__init__()

        self.model_type = model_type
        self._model_kwargs = kwargs

        self.models = nn.ModuleDict({f"{i}": create_model(model_type, **kwargs) for i in range(n)})

    def forward(
        self,
        n: str,
        qty: torch.Tensor,
        past_time: torch.Tensor,
        future_time: torch.Tensor,
        sku: torch.Tensor,
        cats: dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass through the appropriate cluster model.

        Args:
            n: Cluster ID as string.
            qty: Past quantity values.
            past_time: Past time features.
            future_time: Future time features.
            sku: SKU indices.
            cats: Categorical features.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Model predictions. For standard/lightweight models, returns a tensor.
            For advanced model, returns a dict with 'prediction', 'quantiles', etc.
        """
        model = self.models[n]
        return model(qty, past_time, future_time, sku, cats, **kwargs)

    def get_model(self, n: str) -> ModelType:
        """Get a specific cluster model.

        Args:
            n: Cluster ID as string.

        Returns:
            The model for the specified cluster.
        """
        return self.models[n]

    @property
    def num_models(self) -> int:
        """Return the number of cluster models."""
        return len(self.models)

    def count_parameters(self) -> int:
        """Count total trainable parameters across all models."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> dict[str, any]:
        """Get information about the wrapped models.

        Returns:
            Dict with model type, parameter count, and configuration.
        """
        return {
            "model_type": self.model_type,
            "num_clusters": self.num_models,
            "total_parameters": self.count_parameters(),
            "parameters_per_model": self.count_parameters() // max(self.num_models, 1),
            "config": self._model_kwargs,
        }


class EnsembleWrapper(nn.Module):
    """Ensemble wrapper for combining multiple model types.

    Supports weighted averaging of predictions from different model architectures.
    """

    def __init__(
        self,
        models: dict[str, nn.Module],
        weights: dict[str, float] | None = None,
    ):
        """Initialize ensemble wrapper.

        Args:
            models: Dict mapping model names to model instances.
            weights: Optional weights for each model. If None, uses uniform weights.
        """
        super().__init__()

        self.model_names = list(models.keys())
        self.models = nn.ModuleDict(models)

        if weights is None:
            weights = {name: 1.0 / len(models) for name in models}
        self.register_buffer("weights", torch.tensor([weights[name] for name in self.model_names]))

    def forward(
        self,
        qty: torch.Tensor,
        past_time: torch.Tensor,
        future_time: torch.Tensor,
        sku: torch.Tensor,
        cats: dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with weighted ensemble.

        Args:
            qty: Past quantity values.
            past_time: Past time features.
            future_time: Future time features.
            sku: SKU indices.
            cats: Categorical features.

        Returns:
            Weighted average of model predictions.
        """
        predictions = []

        for name in self.model_names:
            model = self.models[name]
            output = model(qty, past_time, future_time, sku, cats, **kwargs)

            # Handle dict outputs from advanced model
            if isinstance(output, dict):
                output = output["prediction"]

            predictions.append(output)

        # Stack and compute weighted average
        stacked = torch.stack(predictions, dim=0)  # (num_models, batch, n_out)
        weights = self.weights.view(-1, 1, 1)  # (num_models, 1, 1)

        return (stacked * weights).sum(dim=0)
