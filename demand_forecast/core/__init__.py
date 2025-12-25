"""Core module for training, evaluation, and orchestration."""

# Only import exceptions here to avoid circular imports
# Other modules should be imported directly:
#   from demand_forecast.core.pipeline import ForecastPipeline
#   from demand_forecast.core.trainer import Trainer
#   from demand_forecast.core.evaluator import Evaluator
from demand_forecast.core.exceptions import (
    ConfigurationError,
    DataValidationError,
    DemandForecastError,
    InsufficientDataError,
    ModelNotFoundError,
    TrainingError,
)

__all__ = [
    "DemandForecastError",
    "DataValidationError",
    "ConfigurationError",
    "ModelNotFoundError",
    "InsufficientDataError",
    "TrainingError",
]


def __getattr__(name: str):
    """Lazy import for heavy modules to avoid circular imports."""
    if name == "ForecastPipeline":
        from demand_forecast.core.pipeline import ForecastPipeline

        return ForecastPipeline
    elif name == "Trainer":
        from demand_forecast.core.trainer import Trainer

        return Trainer
    elif name == "TrainingState":
        from demand_forecast.core.trainer import TrainingState

        return TrainingState
    elif name == "EarlyStopConfig":
        from demand_forecast.core.trainer import EarlyStopConfig

        return EarlyStopConfig
    elif name == "Evaluator":
        from demand_forecast.core.evaluator import Evaluator

        return Evaluator
    elif name == "ValidationResult":
        from demand_forecast.core.evaluator import ValidationResult

        return ValidationResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
