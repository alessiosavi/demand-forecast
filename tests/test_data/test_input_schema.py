"""Named calendar slots and leak-free target lags share the target affine frame."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import (
    CANONICAL_CALENDAR_SCHEMA,
    calendar_feature_mask,
    calendar_feature_names,
    calendar_features,
)
from demand_forecast.data.sampler import BatchBuilder
from demand_forecast.data.scaling import sample_frame
from tests.helpers.panels import synthetic_panel


@pytest.mark.parametrize(
    "freq,available",
    [
        ("W-SUN", ("woy", "month")),
        ("D", ("dow", "dom", "doy", "woy", "month")),
        ("5min", ("tod", "hour", "dow", "dom", "doy", "woy", "month")),
        ("h", ("tod", "hour", "dow", "dom", "doy", "woy", "month")),
        ("MS", ("month", "quarter")),
        ("ME", ("month", "quarter")),
        ("QS", ("month", "quarter")),
        (pd.offsets.QuarterEnd(startingMonth=3), ("month", "quarter")),
        ("YS", ("month",)),
    ],
)
def test_canonical_calendar_names_values_masks_and_padding(freq, available):
    expected_names = [
        f"{name}_{function}"
        for name in ("tod", "hour", "dow", "dom", "doy", "woy", "month", "quarter")
        for function in ("sin", "cos")
    ]
    assert list(CANONICAL_CALENDAR_SCHEMA) == expected_names
    assert calendar_feature_names(freq, "canonical") == expected_names
    mask = np.array([name.split("_")[0] in available for name in expected_names])
    np.testing.assert_array_equal(calendar_feature_mask(freq, "canonical"), mask)
    panel = synthetic_panel(1, 6, freq, seed=0)
    encoder = StaticEncoder().fit(panel, ["0"])
    builder = BatchBuilder(panel, 3, 8, encoder, calendar_schema="canonical")
    batch = builder.build([(0, 6), (0, 2), (0, 0)])
    assert batch.enc_cov.shape == (3, 6, 16) and batch.dec_cov.shape == (3, 3, 16)
    np.testing.assert_array_equal(batch.enc_cov_mask[0], np.tile(mask, (6, 1)))
    np.testing.assert_array_equal(batch.dec_cov_mask, np.tile(mask, (3, 3, 1)))
    assert not batch.enc_cov_mask[1, :4].any() and not batch.enc_cov_mask[2].any()
    assert (batch.enc_cov[..., ~mask] == 0).all() and (batch.dec_cov[..., ~mask] == 0).all()
    grid = panel.timestamps(0, 9)
    legacy_names = calendar_feature_names(freq)
    legacy = calendar_features(grid, freq)
    for column, name in enumerate(legacy_names):
        np.testing.assert_array_equal(
            batch.enc_cov[0, :, expected_names.index(name)], legacy[:6, column]
        )
        np.testing.assert_array_equal(
            batch.dec_cov[0, :, expected_names.index(name)], legacy[6:, column]
        )
    if "quarter" in available:
        angle = grid.quarter.to_numpy() * (2 * np.pi / 4)
        np.testing.assert_array_equal(batch.enc_cov[0, :, 14], np.sin(angle[:6]).astype("f4"))
    off = BatchBuilder(panel, 3, 8, encoder, calendar_schema="canonical", calendar=False).build(
        [(0, 6)]
    )
    assert not off.enc_cov.any() and not off.dec_cov.any()
    assert not off.enc_cov_mask.any() and not off.dec_cov_mask.any()


def test_tod_uses_local_seconds_including_subsecond_resolution():
    index = pd.DatetimeIndex(
        ["2024-03-31 01:15:30.125", "2024-03-31 03:15:30.125"], tz="Europe/Rome"
    )
    canonical = calendar_features(index, "5min", "canonical")
    angles = np.array([4530.125, 11730.125]) * (2 * np.pi / 86400)
    np.testing.assert_array_equal(canonical[:, 0], np.sin(angles).astype("f4"))
    np.testing.assert_array_equal(canonical[:, 1], np.cos(angles).astype("f4"))
    assert canonical[0, 0] != canonical[1, 0]


def test_cycle_lags_use_target_frame_full_history_and_no_future_targets():
    panel = synthetic_panel(1, 400, "D", seed=1)
    panel.series[0].target = np.arange(400, dtype=np.float64) ** 1.5
    panel.series[0].target[[3, 375]] = np.nan
    encoder = StaticEncoder().fit(panel, ["0"])
    builder = BatchBuilder(panel, 10, 400, encoder, target_lags="cycle")
    batch = builder.build([(0, 380), (0, 4), (0, 0)])
    assert builder.lag_schema == ("lag_period", "lag_long") and builder.lags == (7, 364)
    assert batch.enc_cov.shape[-1] == batch.dec_cov.shape[-1] == 12
    frame = sample_frame(panel.series[0].target[:380], panel.series[0].target[:380], 7)
    for slot, lag in enumerate((7, 364), start=10):
        for values, masks, positions in (
            (batch.enc_cov[0, :, slot], batch.enc_cov_mask[0, :, slot], range(380)),
            (batch.dec_cov[0, :, slot], batch.dec_cov_mask[0, :, slot], range(380, 390)),
        ):
            for value, mask, position in zip(values, masks, positions):
                source = position - lag
                valid = 0 <= source < 380 and np.isfinite(panel.series[0].target[source])
                assert bool(mask) == valid
                expected = (
                    (panel.series[0].target[source] - frame.loc) / frame.scale if valid else 0
                )
                assert value.item() == np.float32(expected)
    assert not batch.enc_cov_mask[0, :7, 10].any()
    assert not batch.enc_cov_mask[0, 10, 10]  # A historical NaN at position 3.
    assert batch.dec_cov_mask[0, :, 10].tolist() == [
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert not batch.enc_cov_mask[0, :364, 11].any()
    assert not batch.enc_cov_mask[1:, :, 10:].any()  # Short/empty rows and left padding.
    panel.series[0].target[380:] = 1e10
    changed = builder.build([(0, 380)])
    assert torch.equal(changed.enc_cov[0], batch.enc_cov[0])
    assert torch.equal(changed.dec_cov[0], batch.dec_cov[0])
    assert torch.equal(changed.dec_cov_mask[0], batch.dec_cov_mask[0])


@pytest.mark.parametrize("freq,period,long_lag", [("5min", 288, 2016), ("min", 1440, 1440)])
def test_lag_longer_than_context_is_read_outside_the_crop(freq, period, long_lag):
    origin = long_lag + 100
    panel = replace(synthetic_panel(1, origin + 10, freq, seed=0), seasonal_period=period)
    panel.series[0].target = np.arange(origin + 10, dtype=np.float64) ** 1.5
    encoder = StaticEncoder().fit(panel, ["0"])
    batch = BatchBuilder(panel, 10, 512, encoder, target_lags="cycle").build([(0, origin)])
    frame = sample_frame(
        panel.series[0].target[origin - 512 : origin], panel.series[0].target[:origin], period
    )
    assert batch.target.shape == (1, 512)
    # Both reads precede the earliest model context position.
    assert 0 <= origin - long_lag < origin - 512
    expected = (
        panel.series[0].target[origin - long_lag : origin - long_lag + 10] - frame.loc
    ) / frame.scale
    assert batch.dec_cov_mask[0, :, 13].all()
    np.testing.assert_array_equal(batch.dec_cov[0, :, 13], expected.astype("f4"))
    first = 512 - 100
    assert not batch.enc_cov_mask[0, :first, 13].any()
    assert batch.enc_cov_mask[0, first:, 13].all()


def test_period_one_is_disabled_while_long_lag_remains_available():
    panel = replace(synthetic_panel(1, 400, "D", seed=0), seasonal_period=1)
    encoder = StaticEncoder().fit(panel, ["0"])
    batch = BatchBuilder(panel, 370, 512, encoder, target_lags="cycle").build([(0, 400)])
    assert not batch.enc_cov_mask[..., 10].any() and not batch.dec_cov_mask[..., 10].any()
    assert not batch.enc_cov[..., 10].any() and not batch.dec_cov[..., 10].any()
    assert batch.dec_cov_mask[0, :364, 11].all()
    assert not batch.dec_cov_mask[0, 364:, 11].any()


@pytest.mark.parametrize("kwargs", [{"calendar_schema": "typo"}, {"target_lags": "typo"}])
def test_invalid_schema_knobs_are_rejected(kwargs):
    panel = synthetic_panel(1, 10, "D", seed=0)
    with pytest.raises(ValueError):
        BatchBuilder(panel, 3, 8, StaticEncoder().fit(panel, ["0"]), **kwargs)
