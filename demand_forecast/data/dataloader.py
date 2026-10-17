"""DataLoader factory functions."""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchtnt.utils.data.iterators import (
    RoundRobin,
    RoundRobinIterator,
    StoppingMechanism,
)

from demand_forecast.data.dataset import DemandDataset
from demand_forecast.utils.memory import collect_garbage

logger = logging.getLogger(__name__)

# def seed_worker(worker_id: int) -> None:
#     """Set random seed for each worker."""
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


def create_dataloader(
    x: np.ndarray,
    cat: np.ndarray,
    y: np.ndarray,
    encoded_categorical_features: list[str],
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 42,
    device: torch.device | None = None,
    future_time: np.ndarray | None = None,
    cont: np.ndarray | None = None,
) -> tuple[DataLoader, DemandDataset]:
    """Create a DataLoader for demand forecasting data.

    Args:
        x: Input sequences array.
        cat: Categorical data array.
        y: Target sequences array.
        encoded_categorical_features: List of encoded categorical column names.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers.
        shuffle: Whether to shuffle the data.
        seed: Random seed.
        device: Target device (for CUDA prefetching).

    Returns:
        Tuple of (DataLoader, DemandDataset).
    """
    if device is None:
        device = torch.device("cpu")

    g = torch.Generator()
    g.manual_seed(seed)
    pin_memory = device.type == "cuda"

    ds = DemandDataset(x, cat, y, encoded_categorical_features, future_time=future_time, cont=cont)

    # Build DataLoader kwargs - persistent_workers requires num_workers > 0
    dl_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": ds.collate_fn,
        "pin_memory": pin_memory,
        "generator": g,
    }

    # Only add worker-related args if using multiple workers
    # if num_workers > 0:
    #     dl_kwargs["worker_init_fn"] = seed_worker
    #     dl_kwargs["persistent_workers"] = True

    dl = DataLoader(ds, **dl_kwargs)

    return dl, ds


def create_dataloaders(
    raw_datasets: dict[int, tuple[np.ndarray, ...]],
    encoded_categorical_features: list[str],
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    device: torch.device | None = None,
    max_total_workers: int = 8,
) -> tuple[
    dict[str, DataLoader], dict[str, DataLoader], dict[str, DemandDataset], dict[str, DemandDataset]
]:
    """Create train and test DataLoaders for all clusters.

    Args:
        raw_datasets: Dictionary mapping cluster bin to dataset tuple.
        encoded_categorical_features: List of encoded categorical column names.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers per dataloader.
            If 0 (default), data loading happens in the main process.
        seed: Random seed.
        device: Target device.
        max_total_workers: Maximum total workers across all dataloaders.
            Workers are distributed evenly if num_workers would exceed this.

    Returns:
        Tuple of (train_dataloaders, test_dataloaders, train_datasets, test_datasets).
    """
    train_dls: dict[str, DataLoader] = {}
    test_dls: dict[str, DataLoader] = {}
    train_dss: dict[str, DemandDataset] = {}
    test_dss: dict[str, DemandDataset] = {}

    n_clusters = len(raw_datasets)
    n_dataloaders = n_clusters * 2  # train + test per cluster

    # Limit workers to avoid "too many open files" error
    if num_workers > 0:
        total_workers = num_workers * n_dataloaders
        if total_workers > max_total_workers:
            num_workers = max(0, max_total_workers // n_dataloaders)
            logger.warning(
                f"Reduced num_workers to {num_workers} per dataloader "
                f"to stay within {max_total_workers} total workers"
            )

    for _bin, (train, val) in raw_datasets.items():
        for data, loaders, datasets, shuffle in (
            (train, train_dls, train_dss, True),
            (val, test_dls, test_dss, False),
        ):
            if data is None or not len(data.y):
                continue
            loader, dataset = create_dataloader(
                x=data.x,
                cat=data.cat_codes,
                y=data.y,
                future_time=data.future_time,
                cont=data.cont,
                encoded_categorical_features=encoded_categorical_features,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                seed=seed,
                device=device,
            )
            loaders[str(_bin)] = loader
            datasets[str(_bin)] = dataset

    logger.info(f"Created DataLoaders for {n_clusters} clusters (num_workers={num_workers})")

    return train_dls, test_dls, train_dss, test_dss


def get_round_robin_iterators(
    train_dls: dict[str, DataLoader],
    test_dls: dict[str, DataLoader],
    stopping_mechanism: StoppingMechanism = StoppingMechanism.ALL_DATASETS_EXHAUSTED,
) -> tuple[RoundRobinIterator, RoundRobinIterator]:
    """Create round-robin iterators for balanced sampling across clusters.

    Args:
        train_dls: Dictionary of training DataLoaders.
        test_dls: Dictionary of test DataLoaders.
        stopping_mechanism: When to stop iteration.

    Returns:
        Tuple of (train_iterator, test_iterator).
    """
    ds_train = RoundRobinIterator(
        individual_dataloaders=train_dls,
        iteration_strategy=RoundRobin(stopping_mechanism=stopping_mechanism),
    )

    ds_test = RoundRobinIterator(
        individual_dataloaders=test_dls,
        iteration_strategy=RoundRobin(stopping_mechanism=StoppingMechanism.ALL_DATASETS_EXHAUSTED),
    )

    collect_garbage()

    return ds_train, ds_test
