"""Explicit dated forecast windows and PyTorch batches."""

from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from demand_forecast.utils.time_features import calculate_time_features


class WindowData(NamedTuple):
    x: np.ndarray
    future_time: np.ndarray
    cat_codes: np.ndarray
    cont: np.ndarray
    y: np.ndarray
    origins: np.ndarray

    def select(self, mask):
        return WindowData(*(value[mask] for value in self))


class DemandDataset(Dataset):
    """Inputs and targets with calendars explicitly supplied from target dates."""

    def __init__(
        self,
        raw_dataset,
        cat_dataset,
        y,
        encoded_categorical_features,
        n_out=None,
        future_time=None,
        cont=None,
    ):
        self.raw_dataset = raw_dataset
        self.cat_dataset = cat_dataset
        self.y = y
        self.encoded_categorical_features = encoded_categorical_features
        self.n_out = n_out or y.shape[1]
        if future_time is None:
            raise ValueError("future_time from the target dates is required")
        self.future_time = future_time
        self.cont = cont if cont is not None else np.empty((len(y), 0), dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "sequence": self.raw_dataset[idx],
            "categorical_data": self.cat_dataset[idx],
            "y": self.y[idx],
            "future_time": self.future_time[idx],
            "cont": self.cont[idx],
        }

    def collate_fn(self, batch):
        x = torch.as_tensor(np.asarray([row["sequence"] for row in batch]), dtype=torch.float32)
        cats = np.asarray([row["categorical_data"] for row in batch])
        return {
            "qty": x[:, :, :1],
            "sku": x[:, 0, 1].long(),
            "past_time": x[:, :, 2:6],
            "future_time": torch.as_tensor(
                np.asarray([row["future_time"] for row in batch]), dtype=torch.float32
            ),
            "cats": {
                name: torch.as_tensor(cats[:, i], dtype=torch.long)
                for i, name in enumerate(self.encoded_categorical_features)
            },
            "cont": torch.as_tensor(
                np.asarray([row["cont"] for row in batch]), dtype=torch.float32
            ),
            "y": torch.as_tensor(np.asarray([row["y"] for row in batch]), dtype=torch.float32),
        }


def create_time_series_data(
    series,
    series_features,
    encoded_categorical_features,
    window=52,
    n_out=16,
    store_column="store_id",
    target_end_max=None,
    continuous_features=(),
):
    """Create all complete windows with last target <= target_end_max (inclusive).

    Returns x, target calendars, integer categories, continuous statics, y and origins.
    There is deliberately no random or overlapping-window train/test split here.
    """
    xs, futures, cats, conts, ys, origins = [], [], [], [], [], []
    for _, group in series.groupby(["sku_code", store_column], sort=True):
        group = group.sort_index()
        if group.index.has_duplicates:
            raise ValueError("Duplicate dates in a SKU/store series")
        values = group[series_features].to_numpy(dtype=np.float32)
        targets = group["qty_scaled"].to_numpy(dtype=np.float32)
        calendar = calculate_time_features(group.index).astype(np.float32)
        categories = group[list(encoded_categorical_features)].to_numpy(dtype=np.int64)
        continuous = group[list(continuous_features)].to_numpy(dtype=np.float32)
        for i in range(len(group) - window - n_out + 1):
            start, end = i + window, i + window + n_out
            if target_end_max is not None and group.index[end - 1] > pd.Timestamp(target_end_max):
                continue
            xs.append(values[i:start])
            futures.append(calendar[start:end])
            # Static attributes are taken from the last input, never the target block.
            cats.append(categories[start - 1])
            conts.append(continuous[start - 1])
            ys.append(targets[start:end])
            origins.append(group.index[start].to_datetime64())
    count = len(xs)
    return WindowData(
        np.asarray(xs, dtype=np.float32).reshape(count, window, len(series_features)),
        np.asarray(futures, dtype=np.float32).reshape(count, n_out, 4),
        np.asarray(cats, dtype=np.int64).reshape(count, len(encoded_categorical_features)),
        np.asarray(conts, dtype=np.float32).reshape(count, len(continuous_features)),
        np.asarray(ys, dtype=np.float32).reshape(count, n_out),
        np.asarray(origins, dtype="datetime64[ns]"),
    )
