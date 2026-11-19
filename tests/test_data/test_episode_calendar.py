"""Regression coverage for dated episodes on the scoring environment's pandas."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from demand_forecast.bench.v5_panels import time_axis
from demand_forecast.data.episode import (
    CALENDAR_FIELDS,
    MAX_CONTEXT,
    N_PHASES,
    EpisodeBuilder,
    EpisodeRequest,
    Item,
    Panel,
)
from demand_forecast.data.frequency import calendar_features, grid_range, steps_between, to_offset


@pytest.mark.parametrize(
    "freq,epoch,origin,first,first_ns_step,quarter_id",
    [
        ("QS", "1959-01-01", 106, (1473, 7, 1), -1125, 2),
        ("QS-DEC", "1959-03-01", 106, (1473, 9, 1), -1125, 2),
        ("QE-DEC", "1959-03-31", 106, (1473, 9, 30), -1126, 2),
        ("YS-JUL", "1959-07-01", 16, (-73, 7, 1), -281, -1),
        ("ME", "1959-01-31", 106, (1797, 3, 31), -1942, 0),
        ("2ME", "1959-01-31", 106, (1635, 5, 31), -1688, 1),
    ],
)
def test_long_episode_calendar(freq, epoch, origin, first, first_ns_step, quarter_id):
    horizon = 4
    item = Item("item", 0, {"y": np.arange(origin + horizon, dtype=np.float64)}, {}, {})
    # A nanosecond epoch also exercises overflow when applying the negative offset.
    panel = Panel("long", (item,), {}, freq, pd.Timestamp(epoch).as_unit("ns"), True)
    episode = EpisodeBuilder(panel).build(EpisodeRequest(0, (origin,), horizon, MAX_CONTEXT))

    left = origin - MAX_CONTEXT
    stamps = panel.timestamps(left, MAX_CONTEXT + horizon)
    assert (stamps[0].year, stamps[0].month, stamps[0].day) == first
    assert stamps[-left] == panel.epoch
    ids = np.concatenate(
        (episode.cal_ids.reshape(-1, len(CALENDAR_FIELDS)), episode.calh_ids[0, 0, :horizon])
    )
    phases = np.concatenate(
        (episode.cal_phases.reshape(-1, N_PHASES), episode.calh_phases[0, 0, :horizon])
    )
    np.testing.assert_array_equal(ids[0], [-1, -1, -1, first[1] - 1, -1, quarter_id])

    offset = to_offset(freq)
    reference = pd.date_range(
        panel.epoch + first_ns_step * offset,
        periods=origin + horizon - first_ns_step,
        freq=offset,
        unit="ns",
    )
    in_range = slice(first_ns_step - left, None)
    assert (stamps[in_range] == reference).all()
    expected_ids = np.full((len(reference), len(CALENDAR_FIELDS)), -1, dtype=np.int16)
    expected_ids[:, CALENDAR_FIELDS.index("month")] = reference.month - 1
    if quarter_id != -1:
        expected_ids[:, CALENDAR_FIELDS.index("quarter")] = reference.quarter - 1
    np.testing.assert_array_equal(ids[in_range], expected_ids)
    expected_phases = calendar_features(reference, freq, schema="canonical")
    assert phases[in_range].tobytes() == expected_phases.tobytes()


@pytest.mark.parametrize("tz", [None, timezone.utc])
def test_scoring_time_axis_accepts_early_observations(tz):
    history = np.array([datetime(1500, month, 1, tzinfo=tz) for month in (3, 6)])
    future = np.array([datetime(1500, 9, 1, tzinfo=tz)])
    axis = time_axis(history, "QS-DEC", future=future, horizon=1)
    assert axis.on_grid
    assert axis.first == pd.Timestamp(history[0])


@pytest.mark.parametrize(
    "freq,start,end,steps",
    [
        ("D", "1500-01-01", "2020-01-01", 189926),
        ("h", "1500-01-01", "2020-01-01 03:00:00.000000000", 4558227),
        ("B", "1500-03-01", "1500-03-05", 2),
    ],
)
def test_steps_between_early_observations(freq, start, end, steps):
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    assert steps_between(first, last, freq) == steps
    assert steps_between(last, first, freq) == -steps
    if freq == "B":
        assert grid_range(first, steps + 1, freq)[-1] == last
