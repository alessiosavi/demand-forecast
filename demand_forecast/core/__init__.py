"""Core module for training, evaluation, and orchestration."""

# Only import exceptions here to avoid circular imports
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
    if name == "HyperparameterTuner":
        from demand_forecast.core.tuning import HyperparameterTuner

        return HyperparameterTuner
    elif name == "TuningConfig":
        from demand_forecast.core.tuning import TuningConfig

        return TuningConfig
    elif name == "SearchSpace":
        from demand_forecast.core.tuning import SearchSpace

        return SearchSpace
    elif name == "quick_tune":
        from demand_forecast.core.tuning import quick_tune

        return quick_tune
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
