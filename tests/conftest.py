"""Pytest fixtures for demand_forecast tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from demand_forecast.config import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    Settings,
    TimeSeriesConfig,
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
        data=DataConfig(
            input_path=tmp_path / "test_data.csv",
            resample_period="1W",
            max_zeros_ratio=0.8,
        ),
        timeseries=TimeSeriesConfig(
            window=10,
            n_out=4,
            test_size=0.2,
        ),
        model=ModelConfig(
            d_model=64,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=128,
            dropout=0.1,
        ),
        training=TrainingConfig(
            num_epochs=2,
            batch_size=16,
            learning_rate=1e-4,
        ),
        output=OutputConfig(
            model_dir=tmp_path / "models",
            cache_dir=tmp_path / "cache",
        ),
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
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
