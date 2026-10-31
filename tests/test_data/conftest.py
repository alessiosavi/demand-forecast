"""Shared panels and fitted preprocessing for data-core tests."""

import numpy as np
import pytest

from tests.helpers.panels import weekly_panel


@pytest.fixture
def panel29():
    return weekly_panel([29] * 3)


@pytest.fixture
def encoder29(panel29):
    from demand_forecast.data.encoders import StaticEncoder

    return StaticEncoder().fit(panel29, [series.series_id for series in panel29.series])


@pytest.fixture
def panel8():
    return weekly_panel([8] * 2)


@pytest.fixture
def panel_mixed():
    return weekly_panel([60, 20])


@pytest.fixture
def panel_long():
    return weekly_panel([80] * 20)


@pytest.fixture
def encoder8(panel8):
    from demand_forecast.data.encoders import StaticEncoder

    return StaticEncoder().fit(panel8, [series.series_id for series in panel8.series])


@pytest.fixture
def encoder_long(panel_long):
    from demand_forecast.data.encoders import StaticEncoder

    return StaticEncoder().fit(panel_long, [series.series_id for series in panel_long.series])


@pytest.fixture
def panel_with_future_nan():
    panel = weekly_panel([20])
    panel.series[0].dynamic["promo"] = np.concatenate(
        [panel.series[0].dynamic["promo"], [1, 0, np.nan, 1]]
    ).astype(np.float32)
    return panel


@pytest.fixture
def encoder_future_nan(panel_with_future_nan):
    from demand_forecast.data.encoders import StaticEncoder

    return StaticEncoder().fit(panel_with_future_nan, ["a"])


@pytest.fixture
def panel_constant():
    panel = weekly_panel([20])
    panel.series[0].target[:] = 6
    return panel


@pytest.fixture
def encoder_constant(panel_constant):
    from demand_forecast.data.encoders import StaticEncoder

    return StaticEncoder().fit(panel_constant, ["a"])
