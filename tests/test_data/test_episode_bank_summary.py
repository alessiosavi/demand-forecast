"""Summary definitions, ragged grids and the test-only tsfresh reference."""

from types import ModuleType

import numpy as np
import pytest
from tsfresh.feature_extraction import feature_calculators as fc

from demand_forecast.data.episode.bank import TSFRESH_PARITY, summary_vectors
from demand_forecast.data.episode.scale import fit_scale, to_scaled
from demand_forecast.data.episode.schema import BANK_WINDOW, SUMMARY_FEATURES


def reference_value(
    calculators: ModuleType,
    calc: str,
    kwargs: dict,
    x: np.ndarray,
    *,
    span: int | None = None,
) -> float:
    """Apply the plan's transforms after the calculator sees observed cells."""
    n = len(x) if span is None else span
    result = getattr(calculators, calc)(x, **kwargs)
    if calc == "linear_trend":
        result = next(iter(result))[1]
        result = result * n if kwargs["param"][0]["attr"] == "slope" else result**2
    elif calc == "cid_ce":
        result /= np.sqrt(n)
    elif calc.startswith("longest_strike_"):
        result /= n
    return float(result)


def test_parity_with_tsfresh():
    rng = np.random.default_rng(2)
    for length in (64, 200, 512):
        x = rng.normal(size=length).cumsum()
        v, f = summary_vectors(x[None], x[None], season=7, long_lag=30, end=length)
        for name, (calc, kwargs) in TSFRESH_PARITY.items():
            ref = reference_value(fc, calc, kwargs, x)
            i = SUMMARY_FEATURES.index(name)
            assert f[0, i] == np.isfinite(ref)
            if f[0, i]:
                assert abs(v[0, i] - np.clip(ref, -10, 10)) < 1e-4, name


def test_parity_table_covers_every_reference_entry():
    assert set(TSFRESH_PARITY) == {
        "skewness",
        "kurtosis",
        "trend_slope",
        "trend_strength",
        "acf_1",
        "acf_season",
        "acf_long",
        "mean_abs_change",
        "cid_ce",
        "beyond_2_sigma",
        "longest_above",
        "longest_below",
    }


def test_constant_short_and_empty_rows_are_finite():
    rows = np.full((3, 64), np.nan)
    rows[0, :] = 0.377191
    rows[1, -3:] = [1.0, 2.0, 4.0]
    v, f = summary_vectors(rows, rows, season=7, long_lag=None, end=64)
    assert v.shape == f.shape == (3, 28)
    assert v.dtype == np.float32 and f.dtype == np.bool_
    assert np.isfinite(v).all() and not f[0, SUMMARY_FEATURES.index("acf_1")]
    assert not f[2].any() and (v[2] == 0).all()
    for name in (
        "acf_1",
        "acf_season",
        "acf_long",
        "trend_slope",
        "trend_strength",
        "spec_period",
        "spec_strength",
    ):
        assert not f[0, SUMMARY_FEATURES.index(name)], name
    for name in ("skewness", "kurtosis", "acf_season", "acf_long", "spec_period"):
        assert not f[1, SUMMARY_FEATURES.index(name)], name
    assert np.all(v[~f] == 0) and np.all(np.abs(v) <= 10)


@pytest.mark.parametrize("level", [0.377191, 1.636365, -3.3, 1e6 / 3])
@pytest.mark.parametrize("length", [1, 4, 16, 64, 100, 513])
@pytest.mark.parametrize("left_padding", [0, 23])
@pytest.mark.parametrize("with_gaps", [False, True])
def test_constant_rows_have_invalid_trends_correlations_and_spectra(
    level, length, left_padding, with_gaps
):
    row = np.full((1, left_padding + length), level)
    row[:, :left_padding] = np.nan
    if with_gaps:
        row[:, left_padding + 1 :: 3] = np.nan
    v, f = summary_vectors(row, row, season=7, long_lag=30, end=row.shape[1])
    for name in (
        "trend_slope",
        "trend_strength",
        "acf_1",
        "acf_season",
        "acf_long",
        "spec_period",
        "spec_strength",
    ):
        i = SUMMARY_FEATURES.index(name)
        assert not f[0, i] and v[0, i] == 0.0, name
    for name in ("beyond_2_sigma", "longest_above", "longest_below"):
        i = SUMMARY_FEATURES.index(name)
        assert f[0, i] and v[0, i] == 0.0, name


@pytest.mark.parametrize("length", [4, 16, 64, 100, 513])
@pytest.mark.parametrize("left_padding", [0, 23])
@pytest.mark.parametrize("with_gaps", [False, True])
def test_near_constant_rows_keep_ordinary_trend_and_spectrum_flags(length, left_padding, with_gaps):
    row = np.full((1, left_padding + length), 0.377191)
    row[:, :left_padding] = np.nan
    if with_gaps:
        row[:, left_padding + 1 :: 3] = np.nan
    last_observed = np.flatnonzero(np.isfinite(row[0]))[-1]
    row[0, last_observed] += 1e-9
    v, f = summary_vectors(row, row, season=7, long_lag=30, end=row.shape[1])
    count = np.isfinite(row).sum()
    for name in ("trend_slope", "trend_strength", "spec_period", "spec_strength"):
        minimum = 3 if name.startswith("trend_") else 16
        assert f[0, SUMMARY_FEATURES.index(name)] == (count >= minimum), name
    assert np.isfinite(v).all() and np.all(v[~f] == 0)


def test_left_padding_changes_nothing():
    row = np.random.default_rng(3).normal(size=(1, 113))
    row[:, [5, 40, 112]] = np.nan
    padded = np.pad(row, ((0, 0), (100, 0)), constant_values=np.nan)
    v, f = summary_vectors(row, row, season=12, long_lag=30, end=200)
    pv, pf = summary_vectors(padded, padded, season=12, long_lag=30, end=200)
    np.testing.assert_allclose(pv, v, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(pf, f)


def test_rows_are_independent():
    rows = np.random.default_rng(4).normal(size=(7, 96))
    rows[0] = np.nan
    rows[1, :80] = np.nan
    rows[2, ::3] = np.nan
    rows[3, :95] = np.nan
    rows[4] = 2
    v, f = summary_vectors(rows, rows, season=12, long_lag=30, end=103)
    for i, row in enumerate(rows):
        sv, sf = summary_vectors(row[None], row[None], season=12, long_lag=30, end=103)
        np.testing.assert_allclose(sv[0], v[i], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(sf[0], f[i])
    order = [5, 0, 4, 1, 6, 2, 3]
    pv, pf = summary_vectors(rows[order], rows[order], season=12, long_lag=30, end=103)
    np.testing.assert_array_equal(pv, v[order])
    np.testing.assert_array_equal(pf, f[order])


def test_unit_free():
    raw = np.random.default_rng(5).lognormal(size=128)
    raw[::4] = 0
    raw[::13] = np.nan
    scaled = to_scaled(raw, fit_scale(raw, len(raw), 7))
    larger = raw * 1000
    rescaled = to_scaled(larger, fit_scale(larger, len(larger), 7))
    v, f = summary_vectors(scaled[None], raw[None], season=7, long_lag=30, end=128)
    uv, uf = summary_vectors(rescaled[None], larger[None], season=7, long_lag=30, end=128)
    np.testing.assert_allclose(uv, v, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(uf, f)


def test_gaps():
    row = np.random.default_rng(6).normal(size=105).cumsum()
    row[[4, 5, 9, 25, 50, 102, 104]] = np.nan
    observed = row[np.isfinite(row)]
    v, f = summary_vectors(row[None], row[None], season=7, long_lag=30, end=117)
    for name, (calc, kwargs) in TSFRESH_PARITY.items():
        if name in ("acf_season", "acf_long"):
            continue  # Seasonal lags retain grid gaps; the other calculators see observed cells.
        ref = reference_value(fc, calc, kwargs, observed, span=len(row))
        i = SUMMARY_FEATURES.index(name)
        assert f[0, i] == np.isfinite(ref), name
        if f[0, i]:
            assert abs(v[0, i] - np.clip(ref, -10, 10)) < 1e-4, name
    for name, lag in (("acf_season", 7), ("acf_long", 30)):
        pairs = np.isfinite(row[:-lag]) & np.isfinite(row[lag:])
        ref = (
            np.mean((row[:-lag][pairs] - observed.mean()) * (row[lag:][pairs] - observed.mean()))
            / observed.var()
        )
        i = SUMMARY_FEATURES.index(name)
        assert f[0, i] and np.isclose(v[0, i], np.clip(ref, -10, 10))
    shifted, flags = summary_vectors(row[None], row[None], season=7, long_lag=30, end=124)
    np.testing.assert_array_equal(shifted[:, 18:26], v[:, 18:26])
    np.testing.assert_array_equal(flags[:, 18:26], f[:, 18:26])
    assert np.isclose(v[0, 26], np.log(len(row)) / np.log(BANK_WINDOW))
    assert np.isclose(v[0, 27], np.isnan(row).mean())


def test_raw_ratios_and_intermittency():
    raw = np.array([[np.nan, 0, 2, np.nan, 0, 4, 0, 8]], dtype=float)
    scaled = raw - 2
    v, f = summary_vectors(scaled, raw, season=1, long_lag=None, end=8)
    observed = raw[np.isfinite(raw)]
    nonzero = observed[observed != 0]
    expected = {
        "spread_over_level": observed.std() / np.abs(observed).mean(),
        "zero_share": 0.5,
        "adi": np.log(np.diff(np.flatnonzero(observed)).mean()),
        "cv2_nonzero": (nonzero.std() / nonzero.mean()) ** 2,
        "log_length": np.log(7) / np.log(BANK_WINDOW),
        "missing_share": 1 / 7,
    }
    for name, ref in expected.items():
        i = SUMMARY_FEATURES.index(name)
        assert f[0, i] and np.isclose(v[0, i], ref), name


@pytest.mark.parametrize("season", [2, 7, 8, 12, 24])
def test_profiles_use_absolute_grid_phases(season):
    end = 53
    row = np.arange(35, dtype=float)
    row[[0, 3, 7, 22, 34]] = np.nan
    v, f = summary_vectors(row[None], row[None], season=season, long_lag=None, end=end)
    bins = ((np.arange(len(row)) + end - len(row)) % season) * min(8, season) // season
    for b in range(8):
        cells = row[(bins == b) & np.isfinite(row)]
        assert f[0, 18 + b] == (len(cells) > 0)
        assert np.isclose(v[0, 18 + b], np.clip(cells.mean(), -10, 10) if len(cells) else 0)


def test_spectrum_uses_the_span_and_missing_as_zero():
    row = np.sin(2 * np.pi * np.arange(64) / 8)
    row[[3, 5, 13, 20, 63]] = np.nan
    observed = row[np.isfinite(row)]
    centered = np.where(np.isfinite(row), row - observed.mean(), 0)
    power = np.abs(np.fft.rfft(centered)) ** 2
    k = 1 + np.argmax(power[1:])
    padded = np.pad(row, (17, 0), constant_values=np.nan)
    v, f = summary_vectors(padded[None], padded[None], season=8, long_lag=None, end=100)
    assert f[0, 11] and f[0, 12]
    assert np.isclose(v[0, 11], np.log(len(row) / k) / np.log(len(row)))
    assert np.isclose(v[0, 12], power[k] / power[1:].sum())


@pytest.mark.parametrize("observations", [0, 1, 2, 3, 4, 15, 16])
def test_minimum_support(observations):
    row = np.full((1, 32), np.nan)
    row[0, 32 - observations :] = np.arange(1, observations + 1)
    v, f = summary_vectors(row, row, season=1, long_lag=None, end=32)
    minimum = {
        "skewness": 4,
        "kurtosis": 4,
        "trend_slope": 3,
        "trend_strength": 3,
        "spec_period": 16,
        "spec_strength": 16,
        "mean_abs_change": 2,
        "cid_ce": 2,
        "beyond_2_sigma": 1,
        "adi": 2,
        "cv2_nonzero": 2,
    }
    for name, needed in minimum.items():
        assert f[0, SUMMARY_FEATURES.index(name)] == (observations >= needed), name
    assert not f[0, 9:11].any() and not f[0, 18:26].any()
    assert np.isfinite(v).all() and np.all(v[~f] == 0)


def test_no_observed_seasonal_pairs_are_invalid():
    row = np.array([[1, np.nan, 2, np.nan, 4, np.nan, 8, np.nan]], dtype=float)
    v, f = summary_vectors(row, row, season=3, long_lag=20, end=8)
    assert f[0, 8] and not f[0, 9:11].any()
    assert (v[0, 9:11] == 0).all()


def test_zero_and_zero_mean_nonzero_histories():
    rows = np.array([[0, 0, 0, 0], [-1, 1, -1, 1]], dtype=float)
    v, f = summary_vectors(rows, rows, season=2, long_lag=None, end=4)
    assert not f[0, 2] and f[0, 3] and v[0, 3] == 1
    assert not f[0, 4:6].any() and not f[1, 5]
    assert np.isfinite(v).all() and np.all(v[~f] == 0)


@pytest.mark.parametrize("shape", [(0, 64), (3, 0), (0, 0)])
def test_empty_dimensions(shape):
    rows = np.empty(shape)
    v, f = summary_vectors(rows, rows, season=7, long_lag=None, end=0)
    assert v.shape == f.shape == (shape[0], 28)
    assert v.dtype == np.float32 and f.dtype == np.bool_
    assert (v == 0).all() and not f.any()
