import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchtnt.utils.data import CudaDataPrefetcher
from torchtnt.utils.data.iterators import (RoundRobin, RoundRobinIterator,
                                           StoppingMechanism)

import utils


def create_time_series_data(
    series: DataFrame,
    series_features: List[str],
    encoded_categorical_features: List[str],
    test_size=0.2,
    window=52,
    n_out=16,
):
    ts_train_x_dataset = []
    ts_train_cat_dataset = []
    ts_train_y_dataset = []
    ts_test_x_dataset = []
    ts_test_cat_dataset = []
    ts_test_y_dataset = []
    grouped = series.groupby(["sku_code", "store_id"])

    def process_group(_series, window, n_out):
        # Categorical data are the same for all the given series, so we take only the first row
        categorical_data = (
            _series[encoded_categorical_features]
            .iloc[0]
            .apply(lambda x: np.asarray(x, dtype=np.bool_))
            .values
        )
        _ts, _cat, _y = utils.create_timeseries(
            _series[series_features].values,
            [categorical_data] * len(_series),
            _series["qty_scaled"].values,
            window=window,
            n_out=n_out,
        )
        return train_test_split(_ts, _cat, _y, test_size=test_size, shuffle=False)

    results = []
    with ThreadPoolExecutor() as executor:
        for idx, (sku, _series) in enumerate(grouped):
            results.append(executor.submit(process_group, _series, window, n_out))

    for future in results:
        _ts_train, _ts_test, _cat_train, _cat_test, _y_train, _y_test = future.result()
        ts_train_x_dataset.extend(_ts_train)
        ts_test_x_dataset.extend(_ts_test)
        ts_train_cat_dataset.extend(_cat_train)
        ts_test_cat_dataset.extend(_cat_test)
        ts_train_y_dataset.extend(_y_train)
        ts_test_y_dataset.extend(_y_test)

    return (
        np.asarray(ts_train_x_dataset),
        np.asarray(ts_train_y_dataset),
        np.asarray(ts_train_cat_dataset),
        np.asarray(ts_test_x_dataset),
        np.asarray(ts_test_y_dataset),
        np.asarray(ts_test_cat_dataset),
    )


class DemandDataset(Dataset):
    def __init__(self, raw_dataset, cat_dataset, y, encoded_categorical_features):
        self.raw_dataset = raw_dataset
        self.cat_dataset = cat_dataset
        self.encoded_categorical_features = encoded_categorical_features
        self.y = y

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        return {
            "sequence": self.raw_dataset[idx],
            "y": self.y[idx],
            "categorical_data": self.cat_dataset[idx],
        }

    def collate_fn(self, batch):
        qty = torch.as_tensor(
            np.asarray([x["sequence"][:, 0] for x in batch], dtype=np.float32),
            dtype=torch.float32,
        ).unsqueeze(-1)
        skus = torch.as_tensor(
            np.asarray([x["sequence"][:, 1][0] for x in batch], dtype=np.int32),
            dtype=torch.int32,
        )
        past_time = torch.as_tensor(
            np.asarray([x["sequence"][:, 2:6] for x in batch], dtype=np.float32),
            dtype=torch.float32,
        )
        future_time = torch.as_tensor(
            np.asarray([x["sequence"][:, 6:][:16] for x in batch], dtype=np.float32),
            dtype=torch.float32,
        )
        y = torch.as_tensor(
            np.asarray([x["y"] for x in batch], dtype=np.float32),
            dtype=torch.float32,
        )

        cats = defaultdict(list)
        for entry in batch:
            v = dict(zip(self.encoded_categorical_features, entry["categorical_data"]))
            for k, v in v.items():
                cats[k].append(v)
        for k in cats:
            cats[k] = torch.as_tensor(np.asarray(cats[k]), dtype=torch.bool)

        return {
            "qty": qty,
            "sku": skus,
            "past_time": past_time,
            "future_time": future_time,
            "y": y,
            "cats": cats,
        }


def init_ds(
    x,
    cat,
    y,
    encoded_categorical_features,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    seed=42,
    device=torch.device("cpu"),
):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    pin_memory = device.type == "cuda"

    ds = DemandDataset(x, cat, y, encoded_categorical_features)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ds.collate_fn,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True,
    )
    if device.type == "cuda":
        dl = CudaDataPrefetcher(dl, device, 16)

    return dl, ds


def get_data(
    dls_train,
    dls_test,
    stopping_mechanism=StoppingMechanism.ALL_DATASETS_EXHAUSTED,
):
    ds_train = RoundRobinIterator(
        individual_dataloaders=dls_train,
        iteration_strategy=RoundRobin(stopping_mechanism=stopping_mechanism),
    )
    ds_test = RoundRobinIterator(
        individual_dataloaders=dls_test,
        iteration_strategy=RoundRobin(
            stopping_mechanism=StoppingMechanism.ALL_DATASETS_EXHAUSTED
        ),
    )
    utils._collect()
    return ds_train, ds_test
