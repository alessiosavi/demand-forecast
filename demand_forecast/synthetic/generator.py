"""Synthetic sales data generator for testing and development."""

import logging
import random
import re
from collections import Counter
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from tqdm.auto import tqdm

from demand_forecast.data.frequency import grid_range, validate_on_grid

logger = logging.getLogger(__name__)

PRODUCT_COLUMNS = ["id", "category", "subcategory", "price"]
_SUMMER_TERMS = {
    "short",
    "shorts",
    "swim",
    "swimwear",
    "trunk",
    "trunks",
    "tank",
    "tank top",
    "sun hat",
    "bucket hat",
    "sandal",
    "sandals",
    "skort",
    "boardshort",
}
_WINTER_TERMS = {
    "jacket",
    "puffer",
    "parka",
    "coat",
    "coats",
    "anorak",
    "snow",
    "bibs",
    "insulated",
    "fleece",
    "sweater",
    "hoodie",
    "hoody",
    "sweatshirt",
    "pullover",
    "long johns",
    "base layer",
    "baselayer",
    "midlayer",
    "sock",
    "socks",
    "hat",
    "hats",
    "beanie",
    "glove",
    "gloves",
    "mitten",
    "mittens",
    "mitts",
    "scarf",
    "headband",
    "accessories",
}


def _season_profile(subcategory: str, category: str) -> str:
    """Match whole tokens and compound terms, with summer taking precedence."""
    labels = [
        " " + re.sub(r"[_\s&]+", " ", str(value).strip().lower()) + " "
        for value in (subcategory, category)
    ]
    if any(f" {term} " in label for term in _SUMMER_TERMS for label in labels):
        return "summer"
    if str(category).strip().lower() == "outerwear" or any(
        f" {term} " in label for term in _WINTER_TERMS for label in labels
    ):
        return "winter"
    return "neutral"


def _promo_schedule(
    rng: np.random.Generator, n_days: int, promo_share: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return full-calendar discounts and uplifts in nonoverlapping 3–7 day events.

    Coverage is rounded to whole days. Targets below three days are rounded to
    zero or three; calendars shorter than three days can have one shorter event.
    """
    if n_days < 1 or not 0 <= promo_share <= 1:
        raise ValueError("n_days must be positive and promo_share must be between 0 and 1")
    discounts = np.zeros(n_days, dtype=np.float64)
    uplifts = np.ones(n_days, dtype=np.float64)
    target = round(n_days * promo_share)
    if target < 3 and n_days >= 3:
        target = 0 if target < 2 else 3
    lengths = []
    remaining = target
    while remaining > 7:
        length = int(rng.integers(3, 8))
        if remaining - length < 3:
            length = remaining - 3
        lengths.append(length)
        remaining -= length
    if remaining:
        lengths.append(remaining)
    if not lengths:
        return discounts, uplifts

    # Random gaps place complete events without rejection loops or overlaps.
    gaps = np.zeros(len(lengths) + 1, dtype=int)
    if n_days - target >= len(lengths) - 1:
        gaps[1:-1] = 1
    gaps += rng.multinomial(n_days - target - gaps.sum(), rng.dirichlet(np.ones(len(gaps))))
    offset = 0
    for i, length in enumerate(lengths):
        offset += int(gaps[i])
        discounts[offset : offset + length] = rng.uniform(0.15, 0.35)
        uplifts[offset : offset + length] = rng.uniform(1.6, 2.4)
        offset += length
    return discounts, uplifts


def generate_product_sales(
    products: pd.DataFrame,
    start: str | date,
    end: str | date,
    seed: int = 42,
    promo_share: float = 0.10,
    emit_zero_rows: bool = True,
) -> pd.DataFrame:
    """Generate non-negative daily sales for real catalogue IDs and prices.

    The inclusive calendar uses product-specific annual seasons, weekday demand,
    inverse price-volume rates and multi-day promotions. Every calendar day is
    emitted unless ``emit_zero_rows=False``, which omits zero-quantity days.
    All randomness comes from a local seeded NumPy generator.
    """
    missing = set(PRODUCT_COLUMNS) - set(products.columns)
    if missing:
        raise ValueError(f"Missing product columns: {', '.join(sorted(missing))}")
    if products["id"].isna().any():
        raise ValueError("Product IDs must be non-null and unique")
    product_ids = products["id"].astype(str)
    if product_ids.duplicated().any() or product_ids.str.strip().eq("").any():
        raise ValueError("Product IDs must be non-empty and unique")
    prices = products["price"].to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Product prices must be finite and positive")
    prices = prices.round(2)
    if (prices <= 0).any():
        raise ValueError("Product prices must round to at least one cent")
    if not 0 <= promo_share <= 1:
        raise ValueError("promo_share must be between 0 and 1")
    start_day, end_day = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start_day) or pd.isna(end_day) or start_day.tz or end_day.tz:
        raise ValueError("start and end must be valid dates without a timezone")
    dates = pd.date_range(start_day.normalize(), end_day.normalize(), freq="D").as_unit("ns")
    if dates.empty:
        raise ValueError("start must be on or before end")

    rng = np.random.default_rng(seed)
    profiles = [
        _season_profile(subcategory, category)
        for subcategory, category in products[["subcategory", "category"]].itertuples(
            index=False, name=None
        )
    ]
    logger.info("Season profiles (products): %s", dict(Counter(profiles)))
    day_of_year = dates.day_of_year.to_numpy()
    weekday = np.where(dates.day_of_week < 4, 0.9, 1.25) / ((4 * 0.9 + 3 * 1.25) / 7)
    columns: dict[str, list[np.ndarray]] = {
        name: [] for name in ("date", "product_id", "qty", "unit_price")
    }
    profile_totals: Counter = Counter()
    promo_days = 0
    for product_id, price, profile in zip(product_ids, prices, profiles, strict=True):
        peak = {"summer": 196.0, "winter": 15.0}.get(profile)
        if peak is None:
            peak = rng.uniform(0, 365.25)
        peak += rng.uniform(-10, 10)
        amplitude = 0.15 if profile == "neutral" else 0.6
        season = 1 + amplitude * np.cos(2 * np.pi * (day_of_year - peak) / 365.25)
        base_rate = np.clip(30 / price**0.7, 0.15, 12) * rng.lognormal(0, 0.35)
        discounts, uplift = _promo_schedule(rng, len(dates), promo_share)
        qty = rng.poisson(base_rate * season * weekday * uplift).astype(np.int64)
        keep = np.ones(len(dates), dtype=bool) if emit_zero_rows else qty > 0
        columns["date"].append(dates.to_numpy()[keep])
        columns["product_id"].append(np.full(keep.sum(), product_id, dtype=object))
        columns["qty"].append(qty[keep])
        columns["unit_price"].append(np.round(price * (1 - discounts[keep]), 2))
        profile_totals[profile] += int(qty.sum())
        promo_days += int(np.count_nonzero(discounts))

    dtypes = {
        "date": "datetime64[ns]",
        "product_id": "object",
        "qty": "int64",
        "unit_price": "float64",
    }
    sales = pd.DataFrame(
        {
            name: np.concatenate(values) if values else np.array([], dtype=dtypes[name])
            for name, values in columns.items()
        }
    ).sort_values(["date", "product_id"], ignore_index=True)
    calendar_days = len(products) * len(dates)
    logger.info(
        "Generated %d rows; products=%d/%d; calendar=%s..%s; promo share=%.4f (target=%.4f)",
        len(sales),
        sales["product_id"].nunique(),
        len(products),
        dates[0].date(),
        dates[-1].date(),
        promo_days / calendar_days if calendar_days else 0.0,
        promo_share,
    )
    logger.info(
        "Mean daily qty by profile (including zero days): %s",
        {
            profile: round(profile_totals[profile] / (count * len(dates)), 4)
            for profile, count in Counter(profiles).items()
        },
    )
    return sales


def inject_missing(frame: pd.DataFrame, fraction: float, seed: int = 0) -> pd.DataFrame:
    """Drop ``round(len(frame) * fraction)`` rows uniformly without replacement.

    Return a copy preserving the surviving rows' order, index and dtypes. The
    input and process-global random state are unchanged.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    drop = rng.choice(len(frame), size=round(len(frame) * fraction), replace=False)
    keep = np.ones(len(frame), dtype=bool)
    keep[drop] = False
    return frame.iloc[keep].copy()


def generate_long_panel(
    n_series: int,
    length: int,
    freq: str,
    start: str,
    known: tuple[str, ...],
    past: tuple[str, ...],
    statics: tuple[str, ...],
    seed: int,
    promo_uplift: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a complete non-negative panel and a static table on a pinned grid.

    Each series has its own level and phase in a seven-step seasonal cycle
    (weekly on a daily grid). A dynamic column named ``promo`` is a binary
    indicator with 10% probability and adds ``promo_uplift`` to the target.
    Other dynamics are continuous; statics are A/B categories except ``size``,
    which is numeric. Randomness is local to ``seed``.
    """
    if n_series < 1 or length < 1:
        raise ValueError("n_series and length must be positive")
    names = ["id", "timestamp", "target", *known, *past, *statics]
    if len(set(names)) != len(names):
        raise ValueError("Role columns must be unique and cannot use id, timestamp or target")
    if not np.isfinite(promo_uplift) or promo_uplift < 0:
        raise ValueError("promo_uplift must be finite and non-negative")
    start_time = pd.Timestamp(start)
    validate_on_grid(pd.DatetimeIndex([start_time]), freq, "series_0")
    dates = grid_range(start_time, length, freq)

    rng = np.random.default_rng(seed)
    ids = np.array([f"series_{i}" for i in range(n_series)])
    level = rng.uniform(5, 20, size=(n_series, 1))
    phase = rng.uniform(0, 2 * np.pi, size=(n_series, 1))
    target = (level + 2 * np.sin(2 * np.pi * np.arange(length) / 7 + phase)).ravel()
    dynamic = {}
    for name in (*known, *past):
        if name == "promo":
            dynamic[name] = rng.binomial(1, 0.1, size=n_series * length)
            target += promo_uplift * dynamic[name]
        else:
            dynamic[name] = rng.normal(size=n_series * length)

    long = pd.DataFrame(
        {
            "id": np.repeat(ids, length),
            "timestamp": np.tile(dates.to_numpy(), n_series),
            "target": target,
            **dynamic,
        }
    )
    static = pd.DataFrame(
        {
            "id": ids,
            **{
                name: rng.uniform(1, 10, size=n_series)
                if name == "size"
                else rng.choice(["A", "B"], size=n_series)
                for name in statics
            },
        }
    )
    return long, static


def generate_sales_data(
    num_days: int = 365 * 4,
    num_products: int = 100,
    num_stores: int = 20,
    start_date: str = "2020-01-01",
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic sales data.

    Creates sales data with:
    - Multiple products across multiple stores
    - Seasonal patterns (various cycle lengths)
    - Promotional effects
    - Store-level demand multipliers
    - Categorical product attributes
    - Realistic zero-sales patterns

    Args:
        num_days: Number of days to generate.
        num_products: Number of unique products.
        num_stores: Number of stores.
        start_date: Start date for the data.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with generated sales data.

    Example:
        >>> df = generate_sales_data(num_days=365, num_products=10, seed=42)
        >>> print(df.columns.tolist())
        ['date', 'store_id', 'product_id', 'sales_qty', 'price', 'stock', ...]
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Store configuration
    store_ids = [f"store_{i}" for i in range(num_stores)]
    store_factors = {sid: round(random.uniform(0.5, 2.0), 2) for sid in store_ids}

    # Product configuration
    seasonalities = {
        pid: random.choice([False, 30, 30 * 4, 30 * 8, 365]) for pid in range(num_products)
    }
    volumes = {pid: random.choice([5, 20, 100]) for pid in range(num_products)}
    zero_sales_ratio = {
        pid: random.choice([round(0.1 * i, 1) for i in range(0, 10)]) for pid in range(num_products)
    }

    # Categorical options
    categorical_choices = {
        "color": ["red", "blue", "green", "black", "white"],
        "size": ["S", "M", "L", "XL"],
        "category": ["shoes", "dress", "suit", "socks", "bag", "costume"],
        "subcategory": ["A", "B", "C", "D", "E", "F"],
    }

    # Category factors
    category_factor = {
        "shoes": 1.2,
        "dress": 1.4,
        "suit": 1.0,
        "socks": 0.8,
        "bag": 0.8,
        "costume": 0.5,
    }
    subcategory_factor = {"A": 1.2, "B": 1.5, "C": 1, "D": 0.8, "E": 0.6, "F": 0.7}
    size_factor = {"S": 0.8, "M": 1.0, "L": 1.1, "XL": 1.2}
    color_bias = {"red": 1.0, "blue": 1.0, "green": 0.95, "black": 1.05, "white": 1.0}

    def is_promo() -> bool:
        """10% chance of promo day."""
        return np.random.rand() < 0.1

    start = datetime.strptime(start_date, "%Y-%m-%d")

    all_rows: list[dict] = []

    logger.info(f"Generating data for {num_products} products across {num_stores} stores")

    for pid in tqdm(range(num_products), desc="Generating products"):
        product_id = f"product_{pid}"
        product_cats = {k: random.choice(v) for k, v in categorical_choices.items()}
        volume_scale = volumes[pid]

        # Generate base features per product
        x, y = make_regression(n_samples=num_days, n_features=3, noise=0.1)
        base_price = np.clip(x[:, 0] * 10 + 50, 5, 150)
        base_stock = np.clip(x[:, 1] * 100 + 500, 0, 2000)
        base_discount = np.clip(np.abs(x[:, 2] * 0.05), 0, 0.3)

        # Calculate category factor
        cat_factor = (
            category_factor[product_cats["category"]]
            * size_factor[product_cats["size"]]
            * color_bias[product_cats["color"]]
            * subcategory_factor[product_cats["subcategory"]]
        )

        # Generate for each store
        selected_stores = random.choices(store_ids, k=min(num_stores, len(store_ids)))

        for store_id in selected_stores:
            store_factor = store_factors[store_id]
            seasonality = seasonalities[pid]
            phase = np.random.uniform(0, 2 * np.pi) if seasonality else 0.0

            for i in range(num_days):
                date = start + timedelta(days=i)
                promo = is_promo()

                # Seasonality
                seasonal_factor = 1.0
                if seasonality:
                    seasonal_factor += np.sin(2 * np.pi * i / seasonality + phase)

                # Demand scaling
                demand_multiplier = (
                    seasonal_factor
                    * store_factor
                    * cat_factor
                    * (round(random.uniform(1.5, 2.0), 2) if promo else 1.0)
                )

                # Generate sales quantity
                if np.random.rand() < zero_sales_ratio[pid]:
                    sales_qty = 0
                else:
                    raw_sales = abs(y[i]) * 0.1 * demand_multiplier * volume_scale
                    sales_qty = int(np.clip(raw_sales, 0, None))

                row = {
                    "date": date,
                    "store_id": store_id,
                    "product_id": product_id,
                    "sales_qty": sales_qty,
                    "price": round(base_price[i], 2),
                    "stock": int(base_stock[i]),
                    "discount": round(base_discount[i], 2),
                    "is_promo_day": promo,
                    **product_cats,
                }
                all_rows.append(row)

    df = pd.DataFrame(all_rows).sort_values("date")

    logger.info(f"Generated {len(df)} rows of sales data")

    return df
