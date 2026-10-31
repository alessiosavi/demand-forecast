"""Feature engineering utilities."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_metafeatures(
    df: pd.DataFrame,
    sku_code_column: str = "sku_code",
    store_column: str = "store_id",
    quantity_column: str = "qty",
    date_column: str = "date",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Extract time series meta-features using TSFresh.

    Args:
        df: DataFrame with time series data.
        sku_code_column: Column with SKU codes.
        store_column: Column with store IDs.
        quantity_column: Column with quantity values.
        date_column: Column with date/time values for sorting.
        cache_path: Optional path to cache/load features.

    Returns:
        DataFrame with extracted meta-features.
    """
    import tsfresh
    from tsfresh.feature_extraction import MinimalFCParameters

    logger.info("Extracting TSFresh meta-features")

    # One extraction call, no worker pool per SKU; cache is output-only, never blindly reused.
    keys = [sku_code_column, store_column]
    data = df.reset_index().copy()
    data["_series_id"] = data.groupby(keys, sort=True).ngroup()
    identities = data.drop_duplicates("_series_id").set_index("_series_id")[keys]
    features = tsfresh.extract_features(
        data[["_series_id", date_column, quantity_column]],
        column_id="_series_id",
        column_value=quantity_column,
        column_sort=date_column,
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=0,
        disable_progressbar=True,
    )
    feature_df = identities.join(features).reset_index(drop=True)

    if cache_path:
        feature_df.to_csv(cache_path, index=False)
        logger.info(f"Cached metafeatures to {cache_path}")

    return feature_df


def get_categorical_columns(
    df: pd.DataFrame,
    exclude_patterns: list[str] | None = None,
) -> list[str]:
    """Identify categorical columns in a DataFrame.

    Args:
        df: DataFrame to analyze.
        exclude_patterns: List of patterns to exclude from categorical columns.

    Returns:
        List of categorical column names.
    """
    if exclude_patterns is None:
        exclude_patterns = ["sku"]

    numeric_cols = set(df.select_dtypes(np.number).columns)

    categorical_cols = [
        c
        for c in df.columns
        if c not in numeric_cols and not any(pattern in c for pattern in exclude_patterns)
    ]

    return categorical_cols
