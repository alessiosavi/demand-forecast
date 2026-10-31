"""Data loading, preprocessing, and dataset management."""

# Use lazy imports to avoid circular import issues

__all__ = [
    "extract_metafeatures",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name == "extract_metafeatures":
        from demand_forecast.data.feature_engineering import extract_metafeatures

        return extract_metafeatures
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
