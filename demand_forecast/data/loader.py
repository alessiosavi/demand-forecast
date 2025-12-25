"""Data loading utilities."""

import logging
from pathlib import Path

import pandas as pd

from demand_forecast.core.exceptions import DataValidationError

logger = logging.getLogger(__name__)


def load_sales_data(
    path: Path,
    date_column: str = "date",
    sku_column: str = "sku",
    quantity_column: str = "qty",
    store_column: str = "store_id",
    product_id_column: str | None = "product_id",
    sales_qty_column: str | None = "sales_qty",
) -> pd.DataFrame:
    """Load sales data from CSV or Parquet file.

    Args:
        path: Path to the data file (CSV or Parquet).
        date_column: Name of the date column in the output.
        sku_column: Name of the SKU column in the output.
        quantity_column: Name of the quantity column in the output.
        store_column: Name of the store column.
        product_id_column: Original column name for product ID (will be renamed to sku_column).
        sales_qty_column: Original column name for sales quantity (will be renamed to quantity_column).

    Returns:
        DataFrame with standardized column names and DatetimeIndex.

    Raises:
        DataValidationError: If required columns are missing or file cannot be loaded.
    """
    if not path.exists():
        raise DataValidationError(f"Data file not found: {path}")

    logger.info(f"Loading data from {path}")

    # Load based on file extension
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise DataValidationError(f"Unsupported file format: {path.suffix}")

    # Rename columns if needed
    rename_map = {}
    if product_id_column and product_id_column in df.columns:
        rename_map[product_id_column] = sku_column
    if sales_qty_column and sales_qty_column in df.columns:
        rename_map[sales_qty_column] = quantity_column

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Validate required columns
    required_columns = [date_column, sku_column, quantity_column, store_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")

    # Convert date column
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)

    # Convert dtypes
    df = df.convert_dtypes()

    logger.info(f"Loaded {len(df)} rows with {df[sku_column].nunique()} unique SKUs")

    return df
