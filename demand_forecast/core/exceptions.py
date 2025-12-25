"""Custom exceptions for demand forecasting."""


class DemandForecastError(Exception):
    """Base exception for demand forecasting errors."""

    pass


class DataValidationError(DemandForecastError):
    """Raised when input data fails validation.

    Examples:
        - Missing required columns
        - Invalid data types
        - Empty dataframes
    """

    pass


class ConfigurationError(DemandForecastError):
    """Raised when configuration is invalid.

    Examples:
        - Invalid parameter values
        - Missing required configuration
        - Incompatible settings
    """

    pass


class ModelNotFoundError(DemandForecastError):
    """Raised when attempting to load a non-existent model.

    Examples:
        - Model file not found
        - Invalid model path
    """

    pass


class InsufficientDataError(DemandForecastError):
    """Raised when there is not enough data for the requested operation.

    Examples:
        - Time series too short for window size
        - Not enough samples for train/test split
    """

    pass


class TrainingError(DemandForecastError):
    """Raised when training fails.

    Examples:
        - NaN loss values
        - Gradient explosion
        - CUDA out of memory
    """

    pass
