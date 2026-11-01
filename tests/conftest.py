"""Pytest fixtures for demand_forecast tests."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Fix for macOS + PyTorch threading segfault issue
# This must be done before any PyTorch operations
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

from demand_forecast.config import (  # noqa: E402
    DatasetConfig,
    ForecastConfig,
    ModelConfig,
    Settings,
    TrainingConfig,
)


@pytest.fixture
def sample_sales_data() -> pd.DataFrame:
    """Generate sample sales data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=200, freq="D")

    data = []
    for sku in ["SKU001", "SKU002", "SKU003"]:
        for store in ["store_1", "store_2"]:
            for date in dates:
                data.append(
                    {
                        "date": date,
                        "sku": sku,
                        "store_id": store,
                        "qty": np.random.randint(0, 100),
                        "category": "shoes" if sku == "SKU001" else "clothing",
                        "color": np.random.choice(["red", "blue", "green"]),
                        "size": np.random.choice(["S", "M", "L"]),
                    }
                )

    return pd.DataFrame(data)


@pytest.fixture
def sample_config(tmp_path: Path) -> Settings:
    """Create sample configuration for testing."""
    return Settings(
        data=DatasetConfig(
            path=tmp_path / "test_data.csv",
            id_columns=["id"],
            target="target",
            target_support="nonnegative",
            freq="W-SUN",
        ),
        forecast=ForecastConfig(horizon=4, max_context=64),
        model=ModelConfig(
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            hidden_continuous_size=16,
            cat_emb_dim=4,
        ),
        training=TrainingConfig(base_steps=10, batch_size=8, amp=False),
        seed=42,
        device="cpu",
    )


@pytest.fixture
def sample_tensor_data() -> dict[str, torch.Tensor]:
    """Generate sample tensor data for model testing."""
    batch_size = 4
    window = 10
    n_out = 4

    return {
        "qty": torch.randn(batch_size, window, 1),
        "past_time": torch.randn(batch_size, window, 4),
        "future_time": torch.randn(batch_size, n_out, 4),
        "sku": torch.randint(0, 10, (batch_size,)),
        "cats": {
            "category": torch.randint(0, 5, (batch_size,)),
            "color": torch.randint(0, 3, (batch_size,)),
        },
    }


@pytest.fixture
def temp_csv_file(tmp_path: Path, sample_sales_data: pd.DataFrame) -> Path:
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "sales.csv"
    sample_sales_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_parquet_file(tmp_path: Path, sample_sales_data: pd.DataFrame) -> Path:
    """Create a temporary Parquet file with sample data."""
    parquet_path = tmp_path / "sales.parquet"
    sample_sales_data.to_parquet(parquet_path, index=False)
    return parquet_path


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility.

    Note: We use seed 123 instead of 42 because torch.manual_seed(42)
    can cause segmentation faults on certain macOS + PyTorch configurations.
    """
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)


@pytest.fixture(scope="session")
def tiny_pretrained_bundle_dir(tmp_path_factory):
    pytest.importorskip("pyarrow")
    from demand_forecast.pretrain.config import PretrainConfig
    from demand_forecast.pretrain.corpus import corpus_manifest, write_manifest
    from demand_forecast.pretrain.train import pretrain
    from tests.helpers.corpus_fixture import write_corpus_fixture
    from tests.helpers.settings import tiny_settings

    directory = tmp_path_factory.mktemp("transfer-pretrain")
    root = directory / "corpus"
    for kind in ("tsmixup", "kernel_synth"):
        for index in range(2):
            write_corpus_fixture(root, kind=kind, index=index, seed=index)
    write_manifest(corpus_manifest(root), root / "manifest.json")
    cfg = PretrainConfig(
        corpus=dict(
            root=root,
            tsmixup_shards=[0],
            kernel_synth_shards=[0],
            holdout_tsmixup=[1],
            holdout_kernel_synth=[1],
            val_pairs_per_kind=8,
            external_validation=None,
        ),
        forecast=dict(horizon=4, max_context=16, min_context=8),
        model=tiny_settings().model,
        training=dict(
            max_steps=12,
            batch_size=4,
            eval_every=6,
            checkpoint_every=3,
            amp=False,
            warmup_steps=2,
        ),
        device="cpu",
    )
    result = pretrain(cfg, directory / "out")
    assert result.steps == 12
    return directory / "out/bundle"


@pytest.fixture(scope="session")
def pretrained_univariate_bundle(tiny_pretrained_bundle_dir):
    from demand_forecast.core.bundle import load_bundle

    return load_bundle(tiny_pretrained_bundle_dir)
