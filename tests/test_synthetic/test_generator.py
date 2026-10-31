"""Contract and statistical checks for catalogue-based synthetic sales."""

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from demand_forecast.synthetic import generate_product_sales, generate_sales_data
from demand_forecast.synthetic.generator import (
    _promo_schedule,
    _season_profile,
    generate_long_panel,
    inject_missing,
)

START = "2024-10-17"
END = "2026-10-16"
DATES = pd.date_range(START, END)


@pytest.fixture(scope="module")
def products():
    """Cover all profiles at ten different catalogue prices each."""
    rows = []
    for profile, category, subcategory in [
        ("summer", "clothing", "shorts"),
        ("winter", "outerwear", "jacket"),
        ("neutral", "bags", "backpack"),
    ]:
        for i, price in enumerate(np.geomspace(10.99, 899, 10)):
            rows.append(
                {
                    "id": f"catalogue:{profile}:{i}",
                    "category": category,
                    "subcategory": subcategory,
                    "price": round(price, 2),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def sales(products):
    """Generate one shared two-year sample."""
    return generate_product_sales(products, START, END)


@pytest.fixture(scope="module")
def calendar(products, sales):
    """Restore omitted zero-sale days before computing demand statistics."""
    index = pd.MultiIndex.from_product([products["id"], DATES], names=["product_id", "date"])
    return (
        sales.set_index(["product_id", "date"])["qty"].reindex(index, fill_value=0).unstack("date")
    )


def test_schema_and_daily_contract(products, sales):
    assert list(sales.columns) == ["date", "product_id", "qty", "unit_price"]
    assert sales["date"].dtype == np.dtype("datetime64[ns]")
    assert sales["product_id"].map(type).eq(str).all()
    assert sales["qty"].dtype == np.dtype("int64")
    assert sales["unit_price"].dtype == np.dtype("float64")
    assert sales["qty"].ge(0).all()
    assert sales["qty"].eq(0).any()
    assert sales["date"].between(pd.Timestamp(START), pd.Timestamp(END)).all()
    assert sales["date"].eq(sales["date"].dt.normalize()).all()
    assert not sales.duplicated(["date", "product_id"]).any()
    assert set(sales["product_id"]) == set(products["id"])
    np.testing.assert_allclose(sales["unit_price"] * 100, (sales["unit_price"] * 100).round())
    assert_frame_equal(sales, sales.sort_values(["date", "product_id"], ignore_index=True))


def test_product_sales_emit_zero_rows_by_default(products):
    sales = generate_product_sales(products, START, END, seed=0)
    assert len(sales) == len(products) * len(DATES)
    assert sales["qty"].ge(0).all() and sales["qty"].eq(0).any()
    positive = generate_product_sales(products, START, END, seed=0, emit_zero_rows=False)
    assert positive["qty"].ge(1).all()
    assert_frame_equal(positive, sales.loc[sales["qty"] > 0].reset_index(drop=True))


def test_inject_missing_and_long_panel_schema():
    long, static = generate_long_panel(
        4, 30, "W-SUN", "2023-01-01", ("promo",), ("oil",), ("cat",), 0
    )
    assert list(long.columns) == ["id", "timestamp", "target", "promo", "oil"]
    assert list(static.columns) == ["id", "cat"]
    assert len(long) == 120 and len(static) == 4
    assert static["id"].is_unique and set(static["id"]) == set(long["id"])
    assert set(static["cat"]) <= {"A", "B"}
    assert long["target"].ge(0).all() and np.isfinite(long["target"]).all()
    assert set(long["promo"]) == {0, 1}
    assert np.isfinite(long["oil"]).all() and long["oil"].nunique() > 2
    assert not long.duplicated(["id", "timestamp"]).any()
    for _, series in long.groupby("id"):
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(series["timestamp"]),
            pd.date_range("2023-01-01", periods=30, freq="W-SUN"),
            check_names=False,
        )
    assert len(inject_missing(long, 0.2, 0)) == pytest.approx(len(long) * 0.8, abs=6)
    repeated = generate_long_panel(4, 30, "W-SUN", "2023-01-01", ("promo",), ("oil",), ("cat",), 0)
    assert_frame_equal(long, repeated[0])
    assert_frame_equal(static, repeated[1])


def test_long_panel_seasonality_and_promo_uplift():
    args = (2, 70, "D", "2023-01-01", ("promo", "price"), ("oil",), ("cat", "size"), 3)
    baseline, static = generate_long_panel(*args, promo_uplift=0)
    promoted, promoted_static = generate_long_panel(*args, promo_uplift=2.5)
    assert_frame_equal(baseline.drop(columns="target"), promoted.drop(columns="target"))
    assert_frame_equal(static, promoted_static)
    np.testing.assert_allclose(promoted["target"] - baseline["target"], 2.5 * baseline["promo"])
    for _, series in baseline.groupby("id"):
        target = series["target"].to_numpy()
        assert np.ptp(target) > 0
        np.testing.assert_allclose(target[:-7], target[7:])
    different, _ = generate_long_panel(*args[:-1], seed=4)
    assert not different.equals(promoted)


@pytest.mark.parametrize(
    "freq,start", [("D", "2023-01-01"), ("H", "2023-01-01 03:00"), ("M", "2023-01-31")]
)
def test_long_panel_without_covariates(freq, start):
    long, static = generate_long_panel(1, 10, freq, start, (), (), (), 0)
    assert list(long.columns) == ["id", "timestamp", "target"]
    assert list(static.columns) == ["id"]
    assert len(long) == 10 and len(static) == 1
    assert long["timestamp"].iloc[0] == pd.Timestamp(start)
    assert long["timestamp"].is_monotonic_increasing


@pytest.mark.parametrize(
    "override,match",
    [
        ({"n_series": 0}, "n_series"),
        ({"length": 0}, "length"),
        ({"freq": "invalid"}, "Invalid frequency"),
        ({"freq": "0D"}, "positive"),
        ({"start": "2023-01-02"}, "off.*grid"),
        ({"start": "2023-01-01 12:00"}, "off.*grid"),
        ({"known": ("target",)}, "columns"),
        ({"past": ("promo",)}, "columns"),
        ({"promo_uplift": np.nan}, "promo_uplift"),
        ({"promo_uplift": -1}, "promo_uplift"),
    ],
)
def test_long_panel_invalid_inputs(override, match):
    args = dict(
        n_series=2,
        length=10,
        freq="W-SUN",
        start="2023-01-01",
        known=("promo",),
        past=("oil",),
        statics=("cat",),
        seed=0,
    )
    with pytest.raises(ValueError, match=match):
        generate_long_panel(**(args | override))


def test_inject_missing_preserves_input_order_and_duplicate_indexes():
    frame = pd.DataFrame({"target": np.arange(100), "label": "A"}, index=[0] * 100)
    original = frame.copy(deep=True)
    result = inject_missing(frame, 0.2, 0)
    assert len(result) == 80 and result["target"].is_monotonic_increasing
    assert_frame_equal(result, inject_missing(frame, 0.2, 0))
    assert not result.equals(inject_missing(frame, 0.2, 1))
    assert_frame_equal(inject_missing(frame, 0), frame)
    assert_frame_equal(inject_missing(frame, 1), frame.iloc[:0])
    assert_frame_equal(inject_missing(frame.iloc[:0], 0.5), frame.iloc[:0])
    result.iloc[0, 0] = -1
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.nan, np.inf])
def test_inject_missing_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        inject_missing(pd.DataFrame({"target": [1, 2]}), fraction)


def test_new_helpers_preserve_global_rng():
    np.random.seed(100)
    expected = np.random.random(4)
    np.random.seed(100)
    long, _ = generate_long_panel(2, 30, "D", "2023-01-01", ("promo",), ("oil",), ("cat",), 0)
    inject_missing(long, 0.2, 0)
    np.testing.assert_array_equal(np.random.random(4), expected)


def test_seed_and_input_are_preserved(products, sales):
    original = products.copy(deep=True)
    assert_frame_equal(
        sales, generate_product_sales(products, date(2024, 10, 17), date(2026, 10, 16))
    )
    assert not sales.equals(generate_product_sales(products, START, END, seed=43))
    assert_frame_equal(products, original)
    # The new API must not reset or consume the process-global NumPy RNG.
    np.random.seed(100)
    expected = np.random.random(4)
    np.random.seed(100)
    generate_product_sales(products.iloc[:1], START, END)
    np.testing.assert_array_equal(np.random.random(4), expected)


@pytest.mark.parametrize("profile", ["summer", "winter"])
def test_seasonality_includes_zero_days(calendar, profile):
    demand = calendar.loc[calendar.index.str.contains(f":{profile}:")]
    winter = demand.loc[:, DATES.month.isin([12, 1, 2])].mean(axis=1)
    summer = demand.loc[:, DATES.month.isin([6, 7, 8])].mean(axis=1)
    peak, trough = (winter, summer) if profile == "winter" else (summer, winter)
    assert (peak >= 1.3 * trough).all()


def test_inverse_price_volume_relation(products, calendar):
    mean_qty = calendar.mean(axis=1).reindex(products["id"]).to_numpy()
    correlation = products["price"].corr(pd.Series(mean_qty), method="spearman")
    assert correlation < 0


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_promo_calendar_share_and_events(seed):
    discounts, uplifts = _promo_schedule(np.random.default_rng(seed), len(DATES), 0.1)
    promo = discounts > 0
    assert 0.08 <= promo.mean() <= 0.12
    assert ((discounts[promo] >= 0.15) & (discounts[promo] <= 0.35)).all()
    assert ((uplifts[promo] >= 1.6) & (uplifts[promo] <= 2.4)).all()
    assert (discounts[~promo] == 0).all()
    assert (uplifts[~promo] == 1).all()
    changes = np.diff(np.r_[False, promo, False].astype(int))
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    assert ((lengths >= 3) & (lengths <= 7)).all()
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True):
        assert (discounts[start:end] == discounts[start]).all()
        assert (uplifts[start:end] == uplifts[start]).all()
    repeated = _promo_schedule(np.random.default_rng(seed), len(DATES), 0.1)
    np.testing.assert_array_equal(discounts, repeated[0])
    np.testing.assert_array_equal(uplifts, repeated[1])


@pytest.mark.parametrize("n_days,share", [(1, 0), (1, 1), (2, 1), (730, 0), (730, 1), (730, 0.99)])
def test_promo_schedule_boundaries(n_days, share):
    discounts, uplifts = _promo_schedule(np.random.default_rng(42), n_days, share)
    assert len(discounts) == len(uplifts) == n_days
    assert abs(np.count_nonzero(discounts) - n_days * share) <= 2


def test_promos_use_catalogue_price(products, sales, monkeypatch):
    schedules = []

    def record_schedule(rng, n_days, promo_share):
        schedule = _promo_schedule(rng, n_days, promo_share)
        schedules.append(schedule)
        return schedule

    monkeypatch.setattr("demand_forecast.synthetic.generator._promo_schedule", record_schedule)
    generated = generate_product_sales(products, START, END)
    assert_frame_equal(generated, sales)
    for product, (discounts, _) in zip(products.itertuples(), schedules, strict=True):
        rows = generated.loc[generated["product_id"] == product.id].set_index("date")
        day_indices = DATES.get_indexer(rows.index)
        expected = np.round(product.price * (1 - discounts[day_indices]), 2)
        np.testing.assert_array_equal(rows["unit_price"], expected)
        non_promo = discounts[day_indices] == 0
        assert rows["unit_price"].to_numpy()[non_promo].size > 0
        assert (rows["unit_price"].to_numpy()[non_promo] == product.price).all()
    assert 0.08 <= np.mean([np.mean(discounts > 0) for discounts, _ in schedules]) <= 0.12


@pytest.mark.parametrize(
    "subcategory,category,expected",
    [
        ("sun_hat", "accessories", "summer"),
        ("bucket hat", "accessories", "summer"),
        ("tank_top", "outerwear", "summer"),
        ("long_johns", "clothing", "winter"),
        ("base layer", "clothing", "winter"),
        ("coats & jackets", "clothing", "winter"),
        ("Running Socks", "clothing", "winter"),
        ("other", "outerwear", "winter"),
        ("other", "shorts", "summer"),
        ("tshirt", "clothing", "neutral"),
        ("hatchet", "office_supplies", "neutral"),
        ("shortbread", "home", "neutral"),
        ("backpack", "bags", "neutral"),
    ],
)
def test_profile_matching(subcategory, category, expected):
    assert _season_profile(subcategory, category) == expected


@pytest.mark.parametrize("price", [0, -1, np.nan, np.inf, 0.001])
def test_invalid_prices(products, price):
    invalid = products.iloc[:1].assign(price=price)
    with pytest.raises(ValueError, match="prices"):
        generate_product_sales(invalid, START, END)


@pytest.mark.parametrize("share", [-0.01, 1.01, np.nan])
def test_invalid_promo_share(products, share):
    with pytest.raises(ValueError, match="promo_share"):
        generate_product_sales(products, START, END, promo_share=share)


def test_invalid_catalogue_and_dates(products):
    with pytest.raises(ValueError, match="Missing product columns"):
        generate_product_sales(products.drop(columns="price"), START, END)
    with pytest.raises(ValueError, match="unique"):
        generate_product_sales(pd.concat([products, products.iloc[:1]]), START, END)
    with pytest.raises(ValueError, match="non-null"):
        generate_product_sales(products.iloc[:1].assign(id=None), START, END)
    with pytest.raises(ValueError, match="start must be on or before end"):
        generate_product_sales(products, END, START)


def test_empty_catalogue_has_typed_schema(products):
    empty = generate_product_sales(products.iloc[:0], START, END)
    assert empty.empty
    assert list(empty.columns) == ["date", "product_id", "qty", "unit_price"]
    assert empty["date"].dtype == np.dtype("datetime64[ns]")
    assert empty["qty"].dtype == np.dtype("int64")


def test_old_api_uses_one_phase_per_series():
    """A constant baseline must repeat exactly every 30 days after the phase fix."""

    def choose(options):
        return 30 if options[0] is False else options[0]

    with (
        patch("demand_forecast.synthetic.generator.random.choice", side_effect=choose),
        patch("demand_forecast.synthetic.generator.np.random.rand", return_value=0.5),
        patch(
            "demand_forecast.synthetic.generator.make_regression",
            return_value=(np.ones((120, 3)), np.full(120, 100.0)),
        ),
        patch(
            "demand_forecast.synthetic.generator.np.random.uniform", wraps=np.random.uniform
        ) as phase_draw,
    ):
        old_sales = generate_sales_data(num_days=120, num_products=1, num_stores=1)
    assert phase_draw.call_count == 1
    qty = old_sales["sales_qty"].to_numpy()
    assert np.ptp(qty) > 0
    np.testing.assert_array_equal(qty[:-30], qty[30:])
    assert list(old_sales.columns) == [
        "date",
        "store_id",
        "product_id",
        "sales_qty",
        "price",
        "stock",
        "discount",
        "is_promo_day",
        "color",
        "size",
        "category",
        "subcategory",
    ]
