"""Synthetic sales data generator for testing and development."""

import logging
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


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

            for i in range(num_days):
                date = start + timedelta(days=i)
                promo = is_promo()

                # Seasonality
                seasonal_factor = 1.0
                seasonality = seasonalities[pid]
                if seasonality:
                    seasonal_factor += np.sin(
                        2 * np.pi * i / seasonality + np.random.uniform(0, 2 * np.pi)
                    )

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
