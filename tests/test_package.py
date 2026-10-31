"""Package exports stay importable after the legacy pipeline is retired."""

import importlib.util


def test_public_lazy_exports_resolve_and_legacy_modules_are_gone():
    import demand_forecast.core as core
    import demand_forecast.data as data
    import demand_forecast.inference as inference
    import demand_forecast.utils as utils

    for module in (core, data, utils, inference):
        for name in module.__all__:
            assert getattr(module, name) is not None

    for module_name in (
        "core.pipeline",
        "core.forecast_run",
        "core.benchmark",
        "core.trainer",
        "core.evaluator",
        "data.loader",
        "data.preprocessor",
        "data.dataset",
        "data.dataloader",
        "utils.time_features",
        "utils.metrics",
        "inference.confidence",
    ):
        assert importlib.util.find_spec(f"demand_forecast.{module_name}") is None
