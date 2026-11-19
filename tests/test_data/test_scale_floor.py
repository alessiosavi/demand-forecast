"""Prefix-only relative scale floors for training and forecast episodes."""

from dataclasses import fields, replace
from functools import wraps

import numpy as np
import pytest

from demand_forecast.data.episode import (
    EpisodeBuilder,
    EpisodeRequest,
    Item,
    Panel,
    PanelStatsProvider,
    ScreeningConfig,
    builder,
    panel_stats,
    screening,
)
from demand_forecast.data.episode.scale import Scale, fit_scale, to_scaled

CUTOFF = 400
SEASON = 7
FLOOR = 0.01


def _shifted_series():
    prefix = 1000 + 1e-9 * np.random.default_rng(19).normal(size=CUTOFF)
    return np.r_[prefix, 1100.0]


def _panel():
    values = _shifted_series()
    item = Item("item", 0, {"y": values, "z": 2 * values}, {"plan": 3 * values}, {})
    return Panel("scale-floor", (item,), {"plan": "known"}, "D", None, True)


def _watch_scales(monkeypatch):
    calls = {module: [] for module in (builder, panel_stats, screening)}

    def watch(records):
        @wraps(fit_scale)
        def record(values, cutoff, season, **kwargs):
            scale = fit_scale(values, cutoff, season, **kwargs)
            records.append((kwargs, scale))
            return scale

        return record

    for module, records in calls.items():
        monkeypatch.setattr(module, "fit_scale", watch(records))
    return calls


def test_floor_none_is_bitwise_today(monkeypatch):
    seasonal = np.tile(np.arange(SEASON, dtype=float), 10) + np.arange(70) / 100
    series = (
        seasonal,
        np.full(20, 24.0),
        np.r_[np.full(12, np.nan), 1000.0],
        np.where(np.arange(70) % 4 == 0, np.nan, seasonal),
        np.arange(4, dtype=float),
        np.empty(0),
    )
    for values in series:
        cutoff = max(0, len(values) - 1)
        assert fit_scale(values, cutoff, SEASON, floor=None) == fit_scale(values, cutoff, SEASON)

    calls = _watch_scales(monkeypatch)
    panel = _panel()
    request = EpisodeRequest(0, (CUTOFF,), 1, context=32)
    default = EpisodeBuilder(panel, ScreeningConfig()).build(request)
    explicit = EpisodeBuilder(panel, ScreeningConfig(scale_floor=None)).build(request)
    for field in fields(default):
        expected, actual = (getattr(episode, field.name) for episode in (default, explicit))
        if isinstance(expected, np.ndarray):
            assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape), field.name
            assert actual.tobytes() == expected.tobytes(), field.name
        else:
            assert actual == expected, field.name
    for records in calls.values():
        assert records and all(not kwargs for kwargs, _ in records)


def test_constant_history_with_a_shift_is_order_one():
    values = _shifted_series()
    before = fit_scale(values, CUTOFF, SEASON)
    after = fit_scale(values, CUTOFF, SEASON, floor=FLOOR)
    assert to_scaled(values[CUTOFF:], before)[0] > 1e9
    assert after.scale == pytest.approx(FLOOR * np.abs(values[:CUTOFF]).mean(), rel=1e-12, abs=0)
    assert after.loc == before.loc
    assert 5 < to_scaled(values[CUTOFF:], after)[0] < 20


def test_floor_only_raises():
    rng = np.random.default_rng(32)
    for values in rng.normal(loc=10, size=(20, 80)):
        before = fit_scale(values, 64, SEASON)
        for fraction in (1e-4, FLOOR, 0.2):
            after = fit_scale(values, 64, SEASON, floor=fraction)
            assert after.scale == pytest.approx(
                max(before.scale, fraction * np.abs(values[:64]).mean())
            )

    ordinary = 100 + np.tile(np.arange(SEASON), 10) + rng.normal(size=70)
    assert fit_scale(ordinary, 64, SEASON, floor=1e-4) == fit_scale(ordinary, 64, SEASON)
    assert fit_scale(np.full(12, 1000.0), 12, SEASON, floor=FLOOR) == Scale(1000.0, 10.0)
    assert fit_scale(np.full(12, np.nan), 12, SEASON, floor=FLOOR) == Scale(0.0, 1.0)


def test_unit_invariance_with_floor():
    ordinary = np.random.default_rng(42).normal(loc=5, size=CUTOFF + 1)
    for values in (ordinary, _shifted_series()):
        expected = to_scaled(values, fit_scale(values, CUTOFF, SEASON, floor=FLOOR))
        for multiplier in (1e-6, 1e6):
            changed = multiplier * values
            actual = to_scaled(changed, fit_scale(changed, CUTOFF, SEASON, floor=FLOOR))
            # Subtracting nearly equal levels introduces roundoff in residuals near zero.
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=5e-14)


def test_floor_uses_only_the_prefix():
    values = _shifted_series()
    values[4:6] = [np.nan, -np.inf]
    changed = values.copy()
    changed[CUTOFF:] = 1e25
    expected = fit_scale(values, CUTOFF, SEASON, floor=FLOOR)
    assert fit_scale(changed, CUTOFF, SEASON, floor=FLOOR) == expected
    finite = values[:CUTOFF][np.isfinite(values[:CUTOFF])]
    assert expected.scale == pytest.approx(FLOOR * np.abs(finite).mean(), rel=1e-12, abs=0)


def test_every_scale_site_reads_the_floor(monkeypatch):
    calls = _watch_scales(monkeypatch)
    panel = _panel()
    config = ScreeningConfig(scale_floor=FLOOR)
    episode_builder = EpisodeBuilder(panel, config)
    request = EpisodeRequest(0, (CUTOFF,), 1, context=32, target="y")
    episode = episode_builder.build(request)
    item = panel.items[0]
    expected = {
        name: FLOOR * np.abs(values[:CUTOFF]).mean()
        for name, values in (item.targets | item.dynamic).items()
    }
    assert episode.row_names == ["y", "z", "plan"]
    np.testing.assert_allclose(episode.scale, [expected["y"], expected["z"]], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        [scale.scale for _, scale in calls[builder]],
        [expected[name] for name in episode.row_names],
        rtol=1e-12,
        atol=0,
    )
    labels = episode.labels[episode.label_mask]
    assert ((labels > 5) & (labels < 20)).all()

    # Direct screening of both targets with static context needs a second PanelStats.
    typed = replace(panel, items=(replace(item, static={"group": "a"}),))
    stats = panel_stats.PanelStats(typed, "y", CUTOFF, scale_floor=FLOOR)
    screening.screen(typed, stats, 0, ("y", "z"), CUTOFF, config)
    other = screening._standalone_provider(stats).get("z", CUTOFF)
    scales = [episode_builder.stats.get("y", CUTOFF).scales[0], stats.scales[0], other.scales[0]]
    np.testing.assert_allclose(
        [scale.scale for scale in scales],
        [expected["y"], expected["y"], expected["z"]],
        rtol=1e-12,
        atol=0,
    )
    assert [scale.scale for _, scale in calls[screening]] == pytest.approx([expected["z"]])
    assert {module: len(records) for module, records in calls.items()} == {
        builder: 3,
        panel_stats: 3,
        screening: 1,
    }
    assert all(kwargs == {"floor": FLOOR} for records in calls.values() for kwargs, _ in records)


def test_shared_provider_refuses_another_floor():
    panel = _panel()
    config = ScreeningConfig(scale_floor=FLOOR)
    default = PanelStatsProvider(panel)
    floored = PanelStatsProvider(panel, scale_floor=FLOOR)
    with pytest.raises(ValueError, match="scale_floor"):
        EpisodeBuilder(panel, config, stats=default)
    with pytest.raises(ValueError, match="scale_floor"):
        EpisodeBuilder(panel, stats=floored)

    assert EpisodeBuilder(panel, stats=default).stats is default
    request = EpisodeRequest(0, (CUTOFF,), 1, context=32)
    for _ in range(2):
        episode_builder = EpisodeBuilder(panel, config, stats=floored)
        assert episode_builder.stats is floored
        episode_builder.build(request)
    assert floored.builds == 1


def test_screen_refuses_statistics_of_another_floor():
    panel = _panel()
    unfloored = panel_stats.PanelStats(panel, "y", CUTOFF)
    floored = panel_stats.PanelStats(panel, "y", CUTOFF, scale_floor=FLOOR)
    with pytest.raises(ValueError, match="scale_floor"):
        screening.screen(panel, unfloored, 0, ("y", "z"), CUTOFF, ScreeningConfig(scale_floor=FLOOR))
    with pytest.raises(ValueError, match="scale_floor"):
        screening.screen(panel, floored, 0, ("y", "z"), CUTOFF, ScreeningConfig())
    screening.screen(panel, floored, 0, ("y", "z"), CUTOFF, ScreeningConfig(scale_floor=FLOOR))
    screening.screen(panel, unfloored, 0, ("y", "z"), CUTOFF, ScreeningConfig())
