"""Construct forecasting families behind the shared skeleton contract."""

from collections.abc import Callable

from demand_forecast.models.base import Backbone, ForecastModel, ModelSpec
from demand_forecast.models.v2 import V2Backbone, V2Config


def _build_v2(spec: ModelSpec, kwargs: dict) -> V2Backbone:
    options = {"d_model": spec.d_model, **kwargs}
    if options["d_model"] != spec.d_model:
        raise ValueError("backbone d_model must equal ModelSpec.d_model")
    return V2Backbone(V2Config(**options))


FAMILIES: dict[str, Callable[[ModelSpec, dict], Backbone]] = {"v2": _build_v2}


def build_model(family: str, spec: ModelSpec, backbone_kwargs: dict) -> ForecastModel:
    """Build a family at the skeleton's width, rejecting unknown family names."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown model family: {family!r}")
    backbone = FAMILIES[family](spec, backbone_kwargs)
    if backbone.d_model != spec.d_model:
        raise ValueError("backbone d_model must equal ModelSpec.d_model")
    return ForecastModel(spec, backbone)
