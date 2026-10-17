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
        options = {"closed": "left", "label": "left"} if resample_period == "W-MON" else {}
        _series = _df.resample(resample_period, **options).agg(agg)
        _series[sku_column] = _df[sku_column].iloc[0]
        _series[store_column] = _df[store_column].iloc[0]
        series_list.append(_series)

    series = pd.concat(series_list).sort_index()
    if resample_period == "W-MON":
        first, last = complete_week_bounds(df.index)
        series = series.loc[(series.index >= first) & (series.index <= last)]

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

    original_skus = df[sku_column].nunique()
    df = df.groupby(filter_fields, group_keys=False).filter(filter_group)

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


def complete_week_bounds(dates: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return Monday labels wholly covered by the observed daily date range."""
    start, end = dates.min().normalize(), dates.max().normalize()
    first = start + pd.Timedelta(days=(-start.weekday()) % 7)
    last = end - pd.Timedelta(days=end.weekday())
    if end.weekday() != 6:
        last -= pd.Timedelta(weeks=1)
    if first > last:
        raise ValueError("Sales history contains no complete Monday weeks")
    return first, last


def catalogue_weekly_grid(sales: pd.DataFrame, products: pd.DataFrame, cfg) -> pd.DataFrame:
    """Zero-fill every catalogue SKU using global daily coverage, then attach static data."""
    first, last = complete_week_bounds(sales.index)
    weeks = pd.date_range(first, last, freq="W-MON", name=cfg.date_column)
    ids = products[cfg.product_key_column]
    index = pd.MultiIndex.from_product([ids, weeks], names=[cfg.sku_column, cfg.date_column])
    # Keys must share a dtype with the catalogue ids (str), or nothing matches and the grid is all zeros.
    sales = sales.assign(**{cfg.sku_column: sales[cfg.sku_column].astype(str)})
    selected = sales[sales[cfg.sku_column].isin(ids)]
    if selected.empty:
        raise ValueError("No sales rows match the catalogue ids; check product_id values and types")
    qty = selected.groupby([cfg.sku_column, pd.Grouper(freq="W-MON", closed="left", label="left")])[
        cfg.quantity_column
    ].sum()
    grid = qty.reindex(index, fill_value=0).reset_index()
    static = products.rename(columns={cfg.product_key_column: cfg.sku_column})
    grid = grid.merge(static, on=cfg.sku_column, how="left", validate="many_to_one")
    grid[cfg.store_column or "store_id"] = "store"
    return grid.set_index(cfg.date_column)
