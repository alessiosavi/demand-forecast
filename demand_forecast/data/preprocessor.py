"""Data preprocessing utilities."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ScalerManager:
    """Manages scalers per group without global state.

    This class replaces the global `scalers` dict pattern with a proper
    encapsulated data structure that can be serialized and loaded.
    """

    scalers: dict[str, StandardScaler] = field(default_factory=dict)

    def fit_transform(self, group_name: str, data: np.ndarray) -> np.ndarray:
        """Fit a scaler on data and transform it.

        Args:
            group_name: Identifier for this group (e.g., cluster bin).
            data: 1D array of values to scale.

        Returns:
            Scaled values as 1D array.
        """
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data.reshape(-1, 1))
        self.scalers[group_name] = scaler
        return scaled.flatten()

    def transform(self, group_name: str, data: np.ndarray) -> np.ndarray:
        """Transform data using a previously fitted scaler.

        Args:
            group_name: Identifier for the group.
            data: 1D array of values to scale.

        Returns:
            Scaled values as 1D array.

        Raises:
            KeyError: If no scaler exists for this group.
        """
        if group_name not in self.scalers:
            raise KeyError(f"No scaler found for group: {group_name}")
        return self.scalers[group_name].transform(data.reshape(-1, 1)).flatten()

    def inverse_transform(self, group_name: str, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale.

        Args:
            group_name: Identifier for the group.
            data: 1D array of scaled values.

        Returns:
            Original scale values as 1D array.

        Raises:
            KeyError: If no scaler exists for this group.
        """
        if group_name not in self.scalers:
            raise KeyError(f"No scaler found for group: {group_name}")
        return self.scalers[group_name].inverse_transform(data.reshape(-1, 1)).flatten()

    def save(self, path: Path) -> None:
        """Save scalers to file.

        Args:
            path: Path to save the scalers.
        """
        joblib.dump(self.scalers, path)
        logger.info(f"Saved {len(self.scalers)} scalers to {path}")

    @classmethod
    def load(cls, path: Path) -> "ScalerManager":
        """Load scalers from file.

        Args:
            path: Path to load scalers from.

        Returns:
            ScalerManager instance with loaded scalers.
        """
        instance = cls()
        instance.scalers = joblib.load(path)
        logger.info(f"Loaded {len(instance.scalers)} scalers from {path}")
        return instance


def resample_series(
    df: pd.DataFrame,
    resample_period: str = "1W",
    sku_column: str = "sku",
    store_column: str = "store_id",
    quantity_column: str = "qty",
) -> pd.DataFrame:
    """Resample time series data to a specified period.

    Args:
        df: DataFrame with DatetimeIndex.
        resample_period: Pandas resample period string (e.g., "1W", "1D").
        sku_column: Name of SKU column.
        store_column: Name of store column.
        quantity_column: Name of quantity column.

    Returns:
        Resampled DataFrame with aggregated quantities.
    """
    logger.info(f"Resampling {len(df)} rows to {resample_period} periods")

    series_list = []
    grouped = df.groupby([sku_column, store_column])

    for _, _df in tqdm(grouped, desc="Resampling"):
        # Build aggregation: sum for quantity, last for other columns
        agg = {col: "last" for col in _df.columns}
        agg[quantity_column] = "sum"
        _series = _df.resample(resample_period).agg(agg)
        series_list.append(_series)

    series = pd.concat(series_list).sort_index()

    logger.info(
        f"Resampled {len(df)} entries ({df[sku_column].nunique()} SKUs) into {len(series)} entries"
    )

    return series


def filter_skus(
    df: pd.DataFrame,
    window: int,
    n_out: int,
    max_zeros_ratio: float = 0.7,
    sku_column: str = "sku",
    store_column: str = "store_id",
    quantity_column: str = "qty",
) -> pd.DataFrame:
    """Filter SKUs based on data quality criteria.

    Removes SKUs that:
    - Have too few data points for the time series window
    - Have too many zero values

    Args:
        df: DataFrame to filter.
        window: Lookback window size.
        n_out: Forecast horizon.
        max_zeros_ratio: Maximum allowed ratio of zero values.
        sku_column: Name of SKU column.
        store_column: Name of store column.
        quantity_column: Name of quantity column.

    Returns:
        Filtered DataFrame.
    """
    min_length = window + n_out + 2

    def filter_group(group: pd.DataFrame) -> bool:
        """Filter a single SKU-store group."""
        if len(group) < min_length:
            return False
        zero_ratio = (group[quantity_column] == 0).sum() / len(group)
        return zero_ratio < max_zeros_ratio

    filter_fields = [sku_column, store_column]

    # First filter out groups with null categorical data
    mask = df.select_dtypes("string").isna().any(axis=1)
    if mask.sum() > 0:
        logger.info(f"Removing {mask.sum()} rows with missing categorical data")
        df = df[~mask].copy()

    # Filter by quality criteria
    entries_to_retain = (
        df.groupby(filter_fields)
        .filter(filter_group)[filter_fields]
        .reset_index(drop=True)
        .drop_duplicates()
    )

    original_skus = df[sku_column].nunique()
    index = df.index
    df = df.merge(entries_to_retain, on=filter_fields, how="inner")
    df.index = index

    filtered_skus = df[sku_column].nunique()
    logger.info(f"Filtered from {original_skus} to {filtered_skus} SKUs")

    return df


def create_sku_index(df: pd.DataFrame, sku_column: str = "sku") -> dict[str, int]:
    """Create mapping from SKU names to indices.

    Args:
        df: DataFrame with SKU column.
        sku_column: Name of SKU column.

    Returns:
        Dictionary mapping SKU names to integer indices.
    """
    unique_skus = df[sku_column].unique()
    return {sku: idx for idx, sku in enumerate(unique_skus)}


def scale_by_group(
    df: pd.DataFrame,
    scaler_manager: ScalerManager,
    group_column: str = "bins",
    quantity_column: str = "qty",
    scaled_column: str = "qty_scaled",
) -> pd.DataFrame:
    """Scale quantity values by group using StandardScaler.

    Args:
        df: DataFrame to scale.
        scaler_manager: ScalerManager instance to store fitted scalers.
        group_column: Column to group by for scaling.
        quantity_column: Column containing values to scale.
        scaled_column: Name for the new scaled column.

    Returns:
        DataFrame with added scaled column.
    """

    def scale_group(group: pd.Series) -> np.ndarray:
        return scaler_manager.fit_transform(str(group.name), group.values)

    df[scaled_column] = df.groupby(group_column)[quantity_column].transform(scale_group)

    return df
