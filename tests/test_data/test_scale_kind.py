"""Closed-form and integration contracts for the windowed spread scale."""

from dataclasses import fields

import numpy as np
import pytest

from demand_forecast.data.episode import (
    EpisodeBuilder,
    EpisodeRequest,
    Item,
    Panel,
    PanelStatsProvider,
    ScreeningConfig,
    screening,
)
from demand_forecast.data.episode.derived import NaiveHistory
from demand_forecast.data.episode.scale import Scale, fit_scale
from demand_forecast.data.episode.schema import MAX_CONTEXT


@pytest.mark.parametrize(
    ("prefix", "loc", "scale"),
    [
        ([1, 3, 9], 3, np.sqrt(40 / 3)),
        ([1, np.nan, 3, np.inf, 9], 3, np.sqrt(40 / 3)),
        ([8] * 4, 8, 1),
        ([0] * 4, 0, 1),
        ([np.nan, np.inf, -np.inf], 0, 1),
        ([0, 0, 0, 3, 4], 0, np.sqrt(5)),
        ([0] * (MAX_CONTEXT - 1) + [1], 0, 1 / np.sqrt(MAX_CONTEXT)),
        ([101] * MAX_CONTEXT + [2, 4] * (MAX_CONTEXT // 2), 3, 1),
        ([1, 3, 9] + [np.nan] * MAX_CONTEXT, 3, np.sqrt(40 / 3)),
    ],
    ids=[
        "dense",
        "gappy",
        "constant",
        "all-zero",
        "all-missing",
        "intermittent",
        "pulse",
        "long",
        "empty-window",
    ],
)
def test_spread_closed_form(prefix, loc, scale):
    raw = np.r_[prefix, [1e20, -1e20]]
    actual = fit_scale(raw, len(prefix), 7, kind="spread")
    assert (actual.loc, actual.scale) == pytest.approx((loc, scale))


def test_spread_floor_only_reads_the_fitted_cells():
    raw = np.r_[[1e9] * MAX_CONTEXT, [99, 101] * (MAX_CONTEXT // 2)]
    scale = fit_scale(raw, len(raw), 7, kind="spread", floor=0.2)
    assert (scale.loc, scale.scale) == pytest.approx((100, 20))
    empty = fit_scale(raw, -1, 7, kind="spread")
    assert (empty.loc, empty.scale) == pytest.approx((0, 1))


@pytest.mark.parametrize(
    ("values", "loc", "scale", "floor"),
    [
        (np.full(MAX_CONTEXT, 1e308), 1e308, 1, None),
        (np.full(MAX_CONTEXT, 1e308), 1e308, 1e307, 0.1),
        (np.full(MAX_CONTEXT, 1e308), 1e308, np.finfo(float).max, 2.0),
        # The rounded quotient max_float / floor rounds upward at this boundary.
        (
            np.full(MAX_CONTEXT, 9.914093862011287e307),
            9.914093862011287e307,
            np.finfo(float).max,
            1.8132702392002724,
        ),
        (np.array([-1e155, 0, 1e155]), 0, 1e155 * np.sqrt(2 / 3), None),
        (np.array([-1e-170, 0, 1e-170]), 0, 1e-170 * np.sqrt(2 / 3), None),
        (np.array([-1.7e308] * 3 + [1.7e308] * 4), 1.7e308, np.finfo(float).max, None),
    ],
)
def test_spread_reductions_are_finite_at_extreme_magnitudes(values, loc, scale, floor):
    with np.errstate(all="raise"):
        actual = fit_scale(values, len(values), 7, kind="spread", floor=floor)
    assert (actual.loc, actual.scale) == pytest.approx((loc, scale), abs=0, rel=1e-12)


def _window_panel():
    tail = np.repeat([2.0, 4.0], MAX_CONTEXT // 2)
    short = np.r_[np.full(MAX_CONTEXT, 100.0), tail]
    longer = np.r_[short, tail]
    return Panel(
        "window",
        (
            Item("short", 0, {"a": short, "b": short * 3}, {}, {}),
            Item("long", 0, {"a": longer, "b": longer * 3}, {}, {}),
        ),
        {},
        "D",
        None,
        True,
    )


def test_joint_builder_uses_spread_for_all_rows_at_a_clipped_cutoff(monkeypatch):
    panel = _window_panel()
    cutoff = len(panel.items[1].targets["a"]) - 16
    config = ScreeningConfig(scale_kind="spread")
    builder = EpisodeBuilder(panel, config)
    scales = []
    fit = screening.fit_scale

    def record(*args, **kwargs):
        actual = fit(*args, **kwargs)
        scales.append(actual)
        return actual

    monkeypatch.setattr(screening, "fit_scale", record)
    episode = builder.build(EpisodeRequest(0, (cutoff,), 3, context=32))
    assert episode.row_names[:2] == ["a", "b"]
    assert episode.loc == pytest.approx([3, 9])
    assert episode.scale == pytest.approx([1, 3])
    np.testing.assert_allclose([(scale.loc, scale.scale) for scale in scales], [(9, 3)])
    for target, loc, scale in (("a", 3, 1), ("b", 9, 3)):
        stats = builder.stats.get(target, cutoff)
        assert stats.scale_kind == "spread"
        np.testing.assert_allclose([(s.loc, s.scale) for s in stats.scales], [(loc, scale)] * 2)


def test_screening_standalone_provider_retains_kind_and_rejects_mismatch():
    panel = _window_panel()
    cutoff = 2 * MAX_CONTEXT
    config = ScreeningConfig(scale_kind="spread")
    spread = PanelStatsProvider(panel, scale_kind="spread")
    seasonal = PanelStatsProvider(panel)
    with pytest.raises(ValueError, match="scale_kind"):
        EpisodeBuilder(panel, config, stats=seasonal)
    with pytest.raises(ValueError, match="scale_kind"):
        screening.screen(panel, seasonal.get("a", cutoff), 0, ("a", "b"), cutoff, config)

    stats = spread.get("a", cutoff)
    provider = screening._standalone_provider(stats)
    assert provider is screening._standalone_provider(stats)
    secondary = provider.get("b", cutoff)
    np.testing.assert_allclose([(s.loc, s.scale) for s in secondary.scales], [(9, 3)] * 2)


def test_spread_backtest_none_uses_the_same_window_median_as_fit():
    horizon = 16
    origin = 2 * MAX_CONTEXT + 4 * horizon
    raw = np.r_[np.zeros(MAX_CONTEXT), np.full(MAX_CONTEXT, 4.0), np.full(4 * horizon, 7.0)]
    hist = NaiveHistory(raw, season=7, long_lag=None, end=origin)
    error, seen, residuals = hist.backtest(
        raw, Scale(0, 1), origin=origin, horizon=horizon, kind="spread"
    )
    assert seen[0, 0]
    for m in range(1, 5):
        start = origin - m * horizon
        assert fit_scale(raw, start, 7, kind="spread").loc == pytest.approx(4)
    assert error[0, 0] == pytest.approx(3)
    assert residuals[0, 0] == pytest.approx([3] * 9)


def test_spread_constant_window_is_fallback_even_with_older_variation():
    origin = 2 * MAX_CONTEXT
    raw = np.r_[np.arange(MAX_CONTEXT, dtype=float), [5] * MAX_CONTEXT, [99] * 16]
    panel = Panel("fallback", (Item("i", 0, {"y": raw}, {}, {}),), {}, "D", None, True)
    req = EpisodeRequest(
        0, (origin,), 16, context=32, derive=frozenset({"naive_anchor", "naive_evidence"})
    )
    seasonal = EpisodeBuilder(panel).build(req)
    explicit = EpisodeBuilder(panel, ScreeningConfig(scale_kind=None)).build(req)
    for field in fields(seasonal):
        old, new = (getattr(e, field.name) for e in (seasonal, explicit))
        if isinstance(old, np.ndarray):
            assert old.dtype == new.dtype and old.shape == new.shape
            assert old.tobytes() == new.tobytes(), field.name
        else:
            assert old == new, field.name
    spread = EpisodeBuilder(panel, ScreeningConfig(scale_kind="spread")).build(req)
    assert not seasonal.fallback[0]
    assert spread.fallback[0]
    assert spread.loc[0] == pytest.approx(5)
    assert spread.scale[0] == pytest.approx(1)
    assert spread.naive_error_seen[0, 0, 0, 0]
    np.testing.assert_allclose(spread.naive_resq[0, 0, 0, 0], 0, atol=1e-7)
    assert not np.allclose(seasonal.naive_resq[0, 0, 0, 0], 0)


def test_screening_config_rejects_an_unknown_scale_kind():
    assert ScreeningConfig(scale_kind="seasonal").scale_kind == "seasonal"
    with pytest.raises(ValueError, match="scale_kind"):
        ScreeningConfig(scale_kind="spreadd")


def test_fit_scale_rejects_an_unknown_kind():
    values = np.array([1.0, 3.0, 9.0])
    assert fit_scale(values, 3, 7, kind=None) == fit_scale(values, 3, 7, kind="seasonal")
    with pytest.raises(ValueError, match="scale_kind"):
        fit_scale(values, 3, 7, kind="spreadd")
    with pytest.raises(ValueError, match="scale_kind"):
        PanelStatsProvider(_window_panel(), scale_kind="spreadd").get("a", 2 * MAX_CONTEXT)
