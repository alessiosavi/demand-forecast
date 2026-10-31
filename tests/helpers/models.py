"""Small V2 models for CPU-only tests, without bench extras."""

from demand_forecast.models.base import ForecastModel, ModelSpec
from demand_forecast.models.registry import build_model


def build_v2(spec: ModelSpec, **kw) -> ForecastModel:
    """Build the tiny reference backbone with individually overridable options."""
    options = dict(
        d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32
    )
    options.update(kw)
    return build_model("v2", spec, options)
