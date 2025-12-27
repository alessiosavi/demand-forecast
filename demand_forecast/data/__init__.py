"""Data loading, preprocessing, and dataset management."""

# Use lazy imports to avoid circular import issues
# Import directly from submodules:
#   from demand_forecast.data.loader import load_sales_data
#   from demand_forecast.data.dataloader import create_dataloaders

__all__ = [
    "load_sales_data",
    "resample_series",
    "filter_skus",
    "ScalerManager",
    "CategoricalEncoder",
    "extract_metafeatures",
    "DemandDataset",
    "create_time_series_data",
    "create_dataloaders",
    "get_round_robin_iterators",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name == "load_sales_data":
        from demand_forecast.data.loader import load_sales_data

        return load_sales_data
    elif name in ("create_dataloaders", "get_round_robin_iterators"):
        from demand_forecast.data import dataloader

        return getattr(dataloader, name)
    elif name in ("DemandDataset", "create_time_series_data"):
        from demand_forecast.data import dataset

        return getattr(dataset, name)
    elif name in ("CategoricalEncoder", "extract_metafeatures"):
        from demand_forecast.data import feature_engineering

        return getattr(feature_engineering, name)
    elif name in ("ScalerManager", "filter_skus", "resample_series"):
        from demand_forecast.data import preprocessor

        return getattr(preprocessor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
