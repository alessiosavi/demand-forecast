"""Neural network models for demand forecasting."""

# Components
from demand_forecast.models.components import (
    FiLMConditioning,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    PatchEmbedding,
    PositionalEncoding,
    RotaryPositionEmbedding,
    SeriesDecomposition,
    StochasticDepth,
    TemporalBlock,
    TemporalConvNet,
    VariableSelectionNetwork,
    apply_rotary_pos_emb,
)
from demand_forecast.models.lightweight import (
    LightweightDemandModel,
    LightweightMixerModel,
)

# Loss functions
from demand_forecast.models.losses import (
    CombinedForecastLoss,
    MASELoss,
    QuantileLoss,
    SMAPELoss,
    WeightedMSELoss,
    compute_loss,
)

# Models
from demand_forecast.models.transformer import AdvancedDemandForecastModel
from demand_forecast.models.transformer_v2 import AdvancedDemandForecastModelV2
from demand_forecast.models.wrapper import MODEL_REGISTRY, ModelWrapper, create_model

__all__ = [
    # Components
    "PositionalEncoding",
    "RotaryPositionEmbedding",
    "apply_rotary_pos_emb",
    "GatedResidualNetwork",
    "VariableSelectionNetwork",
    "PatchEmbedding",
    "SeriesDecomposition",
    "FiLMConditioning",
    "StochasticDepth",
    "InterpretableMultiHeadAttention",
    "TemporalBlock",
    "TemporalConvNet",
    # Losses
    "compute_loss",
    "WeightedMSELoss",
    "QuantileLoss",
    "CombinedForecastLoss",
    "SMAPELoss",
    "MASELoss",
    # Models
    "AdvancedDemandForecastModel",
    "AdvancedDemandForecastModelV2",
    "LightweightDemandModel",
    "LightweightMixerModel",
    "ModelWrapper",
    "MODEL_REGISTRY",
    "create_model",
]
