"""Task specifications for the local fev fixture."""

from pathlib import Path

from demand_forecast.bench.specs import FevTaskSpec


def fixture_spec(parquet: Path, **overrides) -> FevTaskSpec:
    """Match ``fixture_task`` without importing optional benchmark packages."""
    values = {
        "name": "fixture",
        "dataset_path": str(parquet),
        "dataset_config": None,
        "horizon": 4,
        "num_windows": 2,
        "seasonality": 1,
        "quantile_levels": [i / 10 for i in range(1, 10)],
        "eval_metric": "SQL",
        "extra_metrics": [
            "MASE",
            {"name": "WAPE", "epsilon": 1.0},
            {"name": "WQL", "epsilon": 1.0},
        ],
        "known_dynamic": ["promo"],
        "past_dynamic": ["oil"],
        "static_columns": ["cat"],
        "categorical_statics": ["cat"],
        "target_support": "nonnegative",
    }
    values.update(overrides)
    return FevTaskSpec(**values)
