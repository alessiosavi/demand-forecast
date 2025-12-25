"""PyTorch Dataset classes for demand forecasting."""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from demand_forecast.utils.timeseries import create_timeseries

logger = logging.getLogger(__name__)


class DemandDataset(Dataset):
    """PyTorch Dataset for demand forecasting time series.

    Attributes:
        raw_dataset: Array of time series sequences.
        cat_dataset: Array of categorical data per sequence.
        y: Array of target values.
        encoded_categorical_features: List of encoded categorical column names.
    """

    def __init__(
        self,
        raw_dataset: np.ndarray,
        cat_dataset: np.ndarray,
        y: np.ndarray,
        encoded_categorical_features: list[str],
    ):
        """Initialize the dataset.

        Args:
            raw_dataset: Array of input sequences, shape (N, window, features).
            cat_dataset: Array of categorical data, shape (N,).
            y: Array of target sequences, shape (N, n_out).
            encoded_categorical_features: List of encoded categorical column names.
        """
        self.raw_dataset = raw_dataset
        self.cat_dataset = cat_dataset
        self.encoded_categorical_features = encoded_categorical_features
        self.y = y

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with sequence, target, and categorical data.
        """
        return {
            "sequence": self.raw_dataset[idx],
            "y": self.y[idx],
            "categorical_data": self.cat_dataset[idx],
        }

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Custom collate function for batching.

        Args:
            batch: List of samples from __getitem__.

        Returns:
            Batched tensors ready for model input.
        """
        # Quantity features
        qty = torch.as_tensor(
            np.asarray([x["sequence"][:, 0] for x in batch], dtype=np.float32),
            dtype=torch.float32,
        ).unsqueeze(-1)

        # SKU indices
        skus = torch.as_tensor(
            np.asarray([x["sequence"][:, 1][0] for x in batch], dtype=np.int32),
            dtype=torch.int32,
        )

        # Past time features (columns 2-5)
        past_time = torch.as_tensor(
            np.asarray([x["sequence"][:, 2:6] for x in batch], dtype=np.float32),
            dtype=torch.float32,
        )

        # Future time features (columns 6+, first 16 steps)
        future_time = torch.as_tensor(
            np.asarray([x["sequence"][:, 6:][:16] for x in batch], dtype=np.float32),
            dtype=torch.float32,
        )

        # Target values
        y = torch.as_tensor(
            np.asarray([x["y"] for x in batch], dtype=np.float32),
            dtype=torch.float32,
        )

        # Categorical features
        cats: dict[str, list] = defaultdict(list)
        for entry in batch:
            v = dict(zip(self.encoded_categorical_features, entry["categorical_data"]))
            for k, val in v.items():
                cats[k].append(val)

        # Convert to tensors - embeddings require long integer indices
        cats_tensor = {}
        for k, v in cats.items():
            arr = np.asarray(v)
            # Convert to int for embedding lookup (handles bool, float, int)
            if arr.dtype == np.bool_ or arr.dtype == bool:
                arr = arr.astype(np.int64)
            elif not np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.int64)
            cats_tensor[k] = torch.as_tensor(arr, dtype=torch.long)

        return {
            "qty": qty,
            "sku": skus,
            "past_time": past_time,
            "future_time": future_time,
            "y": y,
            "cats": cats_tensor,
        }


def create_time_series_data(
    series: pd.DataFrame,
    series_features: list[str],
    encoded_categorical_features: list[str],
    test_size: float = 0.2,
    window: int = 52,
    n_out: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create time series datasets from grouped data.

    Args:
        series: DataFrame with time series data, grouped by SKU and store.
        series_features: List of feature column names.
        encoded_categorical_features: List of encoded categorical column names.
        test_size: Fraction of data to use for testing.
        window: Lookback window size.
        n_out: Forecast horizon.

    Returns:
        Tuple of (train_x, train_y, train_cat, test_x, test_y, test_cat) arrays.
    """
    ts_train_x, ts_train_cat, ts_train_y = [], [], []
    ts_test_x, ts_test_cat, ts_test_y = [], [], []

    grouped = series.groupby(["sku_code", "store_id"])

    def process_group(_series: pd.DataFrame, window: int, n_out: int) -> tuple[np.ndarray, ...]:
        """Process a single SKU-store group."""
        # Categorical data are the same for all timesteps, take first row
        categorical_data = (
            _series[encoded_categorical_features]
            .iloc[0]
            .apply(lambda x: np.asarray(x, dtype=np.bool_))
            .values
        )

        _ts, _cat, _y = create_timeseries(
            _series[series_features].values,
            [categorical_data] * len(_series),
            _series["qty_scaled"].values,
            window=window,
            n_out=n_out,
        )

        return train_test_split(_ts, _cat, _y, test_size=test_size, shuffle=False)

    # Process groups in parallel
    results = []
    with ThreadPoolExecutor() as executor:
        for idx, (sku, _series) in enumerate(grouped):
            results.append(executor.submit(process_group, _series, window, n_out))

    # Collect results
    for future in results:
        _ts_train, _ts_test, _cat_train, _cat_test, _y_train, _y_test = future.result()
        ts_train_x.extend(_ts_train)
        ts_test_x.extend(_ts_test)
        ts_train_cat.extend(_cat_train)
        ts_test_cat.extend(_cat_test)
        ts_train_y.extend(_y_train)
        ts_test_y.extend(_y_test)

    logger.info(f"Created time series: {len(ts_train_x)} train, {len(ts_test_x)} test samples")

    return (
        np.asarray(ts_train_x),
        np.asarray(ts_train_y),
        np.asarray(ts_train_cat),
        np.asarray(ts_test_x),
        np.asarray(ts_test_y),
        np.asarray(ts_test_cat),
    )
