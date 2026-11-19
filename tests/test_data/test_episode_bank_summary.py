"""Summary definitions, ragged grids and the test-only tsfresh reference."""

from types import ModuleType

import numpy as np
import pytest
from tsfresh.feature_extraction import feature_calculators as fc

from demand_forecast.data.episode.bank import (
    TSFRESH_PARITY,
    _sample_shape_statistics,
    summary_vectors,
)
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


@pytest.mark.parametrize(
    ("row", "values", "flags"),
    [
        ([0.377191] * 5, [0, 0], [False, False]),
        ([1, 1, 1, 1 + 1e-15, np.nan], [2, 4], [True, True]),
        ([np.nan] * 5, [0, 0], [False, False]),
        ([1, np.nan, np.nan, np.nan, np.nan], [0, 0], [False, False]),
        ([1, 2, np.nan, np.nan, np.nan], [0, 0], [False, False]),
        ([1, 2, 4, np.nan, np.nan], [0, 0], [False, False]),
        ([1, 2, 3, 4, np.nan], [0, -1.2], [True, True]),
        ([1, 2, np.inf, 3, 4], [0, -1.2], [True, True]),
    ],
    ids=["constant", "near-constant", "missing", "one", "two", "three", "four", "infinity"],
)
def test_summary_shape_statistics_for_degenerate_rows(row, values, flags):
    history = np.array([row], dtype=np.float64)
    actual, available = summary_vectors(history, history, season=1, long_lag=None, end=5)
    columns = [SUMMARY_FEATURES.index(name) for name in ("skewness", "kurtosis")]
    np.testing.assert_array_equal(actual[0, columns], np.array(values, dtype=np.float32))
    np.testing.assert_array_equal(available[0, columns], flags)


def test_sample_shape_statistics_match_pandas_with_missing_values():
    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(27)
    history = rng.lognormal(size=(64, 53))
    observed = rng.random(history.shape) > 0.25
    count = observed.sum(axis=1)
    mean = np.where(observed, history, 0.0).sum(axis=1) / count
    centered = np.where(observed, history - mean[:, None], 0.0)
    skewness, kurtosis = _sample_shape_statistics(centered, (centered**2).sum(axis=1), count)
    frame = pandas.DataFrame(np.where(observed, history, np.nan))
    # pandas 3.x is the training reference. Older releases only agree on nondegenerate rows.
    rtol = 1e-12 if pandas.__version__.startswith("3.") else 1e-9
    np.testing.assert_allclose(skewness, frame.skew(axis=1), rtol=rtol, atol=0)
    np.testing.assert_allclose(kurtosis, frame.kurt(axis=1), rtol=rtol, atol=0)


@pytest.mark.parametrize("level", [1000.0, 0.1, 123456.789, 3e7, 1e-3])
@pytest.mark.parametrize("length", [16, 40, 96])
def test_shape_statistics_after_scaling_rounding_noise(level, length):
    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(3)
    raw = np.full((3, length), level)
    raw[0, ::5] = level / 3 * 3
    raw[1] = level * (1 + rng.integers(-2, 3, size=length) * 2.2e-16)
    # Keep the shorter fitting prefix exactly constant in this variant.
    raw[2, length // 2 :: 2] = level / 3 * 3
    columns = [SUMMARY_FEATURES.index(name) for name in ("skewness", "kurtosis")]

    for cutoff in (length, length // 2):
        scaled = np.stack([to_scaled(row, fit_scale(row, cutoff, 7)) for row in raw])
        values, flags = summary_vectors(scaled, raw, season=7, long_lag=None, end=length)
        # pandas 2.x handles tiny moments and constant rows differently, so check only
        # the expected flags and finiteness there.
        expected_flags = np.ptp(scaled, axis=1) > 0
        np.testing.assert_array_equal(
            flags[:, columns], np.repeat(expected_flags[:, None], 2, axis=1)
        )
        assert np.isfinite(values[:, columns]).all()
        if pandas.__version__.startswith("3."):
            frame = pandas.DataFrame(scaled)
            expected = np.column_stack((frame.skew(axis=1), frame.kurt(axis=1)))
            valid = np.isfinite(expected)
            np.testing.assert_array_equal(flags[:, columns], valid)
            np.testing.assert_array_equal(
                values[:, columns], np.where(valid, expected.clip(-10, 10), 0).astype(np.float32)
            )


def test_shape_statistics_pin_rounding_noise_at_a_large_offset():
    pattern = np.array([0, 1, 2, -1, 0, 3, -2, 1, 0, -1])
    row = 1000 * (1 + pattern * 1e-16)
    values, flags = summary_vectors(row[None], row[None], season=7, long_lag=None, end=len(row))
    columns = [SUMMARY_FEATURES.index(name) for name in ("skewness", "kurtosis")]
    # These literals pin determinism across NumPy versions. pandas can differ on this row.
    np.testing.assert_array_equal(flags[0, columns], [True, True])
    np.testing.assert_array_equal(
        values[0, columns], np.array([0.42952698, 0.17055394], dtype=np.float32)
    )


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
