"""Small panels and normalised batches for the shared forecasting skeleton."""

import numpy as np
import pytest

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.sampler import make_batch
from demand_forecast.models.base import ModelSpec
from tests.helpers.panels import weekly_panel


@pytest.fixture
def panel29():
    return weekly_panel([29] * 3)


@pytest.fixture
def encoder29(panel29):
    return StaticEncoder().fit(panel29, [series.series_id for series in panel29.series])


@pytest.fixture
def spec29(panel29, encoder29):
    return ModelSpec.from_panel(panel29, encoder29, ForecastConfig(horizon=13), 16, 4)


@pytest.fixture
def batch29(panel29, encoder29):
    return make_batch(panel29, [(0, 29), (1, 29)], 13, 64, encoder29)


@pytest.fixture
def batch_with_empty_and_constant(encoder29):
    panel = weekly_panel([29] * 2)
    panel.series[0].target[:] = np.nan
    panel.series[1].target[:] = 5.0
    return make_batch(panel, [(0, 29), (1, 29)], 13, 64, encoder29)
