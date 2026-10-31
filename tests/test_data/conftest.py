"""Shared panels and fitted preprocessing for data-core tests."""

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
