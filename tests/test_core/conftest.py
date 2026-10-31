"""Small panels and forecast origins for the new core training path."""

from dataclasses import replace

import pytest

from demand_forecast.data.frequency import to_offset
from tests.helpers.panels import weekly_panel


@pytest.fixture
def panel29():
    panel = weekly_panel([29] * 3)
    return replace(panel, freq=to_offset(panel.freq))


@pytest.fixture
def origin29(panel29):
    return panel29.timestamps(0)[-1] + panel29.freq


@pytest.fixture
def panel8():
    panel = weekly_panel([8] * 2)
    return replace(panel, freq=to_offset(panel.freq))


@pytest.fixture
def origin8(panel8):
    return panel8.timestamps(0)[-1] + panel8.freq


@pytest.fixture
def panel_constant_long():
    panel = weekly_panel([60] * 2)
    for series in panel.series:
        series.target[:] = 6
    return replace(panel, freq=to_offset(panel.freq))


@pytest.fixture
def origin_cl(panel_constant_long):
    return panel_constant_long.timestamps(0)[-1] + panel_constant_long.freq


@pytest.fixture
def panel_mixed():
    panel = weekly_panel([60, 20])
    return replace(panel, freq=to_offset(panel.freq))


@pytest.fixture
def origin_mixed(panel_mixed):
    return panel_mixed.timestamps(0)[-1] + panel_mixed.freq
