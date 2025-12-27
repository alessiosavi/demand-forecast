"""Utility modules for demand forecasting."""

# Use lazy imports to avoid circular import issues
# Import directly from submodules:
#   from demand_forecast.utils.checkpoint import load_checkpoint
#   from demand_forecast.utils.clustering import find_best_k

__all__ = [
    "calculate_time_features",
    "load_checkpoint",
    "remove_outliers",
    "find_best_k",
    "ClusterResult",
    "collect_garbage",
    "init_metrics",
    "create_timeseries",
    "plot_model",
    "plot_clustering",
    "plot_predictions",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name == "load_checkpoint":
        from demand_forecast.utils.checkpoint import load_checkpoint

        return load_checkpoint
    elif name in ("find_best_k", "ClusterResult"):
        from demand_forecast.utils import clustering

        return getattr(clustering, name)
    elif name == "collect_garbage":
        from demand_forecast.utils.memory import collect_garbage

        return collect_garbage
    elif name == "init_metrics":
        from demand_forecast.utils.metrics import init_metrics

        return init_metrics
    elif name == "remove_outliers":
        from demand_forecast.utils.outliers import remove_outliers

        return remove_outliers
    elif name == "calculate_time_features":
        from demand_forecast.utils.time_features import calculate_time_features

        return calculate_time_features
    elif name == "create_timeseries":
        from demand_forecast.utils.timeseries import create_timeseries

        return create_timeseries
    elif name in ("plot_model", "plot_clustering", "plot_predictions"):
        from demand_forecast.utils import visualization

        return getattr(visualization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
