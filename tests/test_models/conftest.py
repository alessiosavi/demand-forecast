"""Small panels and normalised batches for the shared forecasting skeleton."""

from copy import deepcopy

import numpy as np
import pytest

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.sampler import make_batch
from demand_forecast.models.base import ModelSpec
from tests.helpers.panels import weekly_panel
from tests.test_data.conftest import encoder_future_nan, panel_with_future_nan  # noqa: F401


@pytest.fixture
def panel29():
    panel = weekly_panel([29] * 3)
    # Future promo must be observed for the bidirectional decoder test; targets
    # and past-only covariates still end at week 29.
    for i, series in enumerate(panel.series):
        series.dynamic["promo"] = ((np.arange(29 + 52) + i) % 4 == 0).astype(np.float32)
    return panel


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


@pytest.fixture
def spec29_for(panel29, encoder29):
    def build(horizon):
        return ModelSpec.from_panel(panel29, encoder29, ForecastConfig(horizon=horizon), 16, 4)

    return build


@pytest.fixture
def batch29_for(panel29, encoder29):
    def build(horizon):
        return make_batch(panel29, [(0, 29), (1, 29)], horizon, 64, encoder29)

    return build


@pytest.fixture
def batch_short_items():
    return [(0, 7), (1, 29)]


@pytest.fixture
def batch_short(panel29, encoder29, batch_short_items):
    return make_batch(panel29, batch_short_items, 13, 64, encoder29)


@pytest.fixture
def batch_future_nan(panel_with_future_nan, encoder_future_nan):  # noqa: F811
    return make_batch(panel_with_future_nan, [(0, 20)], 13, 64, encoder_future_nan)


@pytest.fixture
def panel29_with_future_oil(panel29):
    panel = deepcopy(panel29)
    for series in panel.series:
        oil = np.concatenate([series.dynamic["oil"], np.full(13, 1000, dtype=np.float32)])
        oil[20:] += 1000  # Change the entire forecast span at origin 20 as well.
        series.dynamic["oil"] = oil
    return panel
