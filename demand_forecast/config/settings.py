"""Pydantic configuration models for demand forecasting."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""

    input_path: Path = Field(..., description="Path to input CSV/Parquet file")
    resample_period: str = Field("1W", pattern=r"^\d*[DWMY]$", description="Resampling period")
    max_zeros_ratio: float = Field(
        0.7, ge=0.0, le=1.0, description="Maximum ratio of zero values per SKU"
    )
    date_column: str = Field("date", description="Name of date column")
    sku_column: str = Field("sku", description="Name of SKU column")
    quantity_column: str = Field("qty", description="Name of quantity column")
    store_column: str = Field("store_id", description="Name of store column")

    @field_validator("input_path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Validate that input path exists."""
        if not v.exists():
            raise ValueError(f"Input path does not exist: {v}")
        return v


class TimeSeriesConfig(BaseModel):
    """Time series window configuration."""

    window: int = Field(52, ge=1, description="Lookback window in periods")
    n_out: int = Field(16, ge=1, description="Forecast horizon in periods")
    test_size: float = Field(0.2, ge=0.0, lt=1.0, description="Test set ratio")


class ModelConfig(BaseModel):
    """Neural network architecture configuration."""

    sku_emb_dim: int = Field(32, ge=1, description="SKU embedding dimension")
    cat_emb_dims: int = Field(32, ge=1, description="Categorical embedding dimension")
    d_model: int = Field(256, ge=1, description="Transformer model dimension")
    nhead: int = Field(8, ge=1, description="Number of attention heads")
    num_encoder_layers: int = Field(4, ge=1, description="Number of encoder layers")
    num_decoder_layers: int = Field(4, ge=1, description="Number of decoder layers")
    dim_feedforward: int = Field(2048, ge=1, description="Feedforward dimension")
    dropout: float = Field(0.3, ge=0.0, lt=1.0, description="Dropout rate")

    @field_validator("nhead")
    @classmethod
    def validate_nhead(cls, v: int, info) -> int:
        """Validate that d_model is divisible by nhead."""
        d_model = info.data.get("d_model", 256)
        if d_model % v != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({v})")
        return v


class TrainingConfig(BaseModel):
    """Training loop configuration."""

    num_epochs: int = Field(10, ge=1, description="Number of training epochs")
    batch_size: int = Field(128, ge=1, description="Batch size")
    learning_rate: float = Field(1e-5, gt=0, description="Learning rate")
    weight_decay: float = Field(1e-2, ge=0, description="Weight decay for AdamW")
    early_stop_patience: int = Field(3, ge=1, description="Early stopping patience")
    early_stop_min_delta: float = Field(1.0, ge=0, description="Minimum improvement for early stop")
    num_workers: int = Field(0, ge=0, description="DataLoader workers (0=main process)")
    pin_memory: bool = Field(True, description="Pin memory for CUDA")
    flatten_loss: bool = Field(True, description="Use flattened loss (sum reduction)")


class OutputConfig(BaseModel):
    """Output paths configuration."""

    model_dir: Path = Field(Path("models"), description="Model output directory")
    cache_dir: Path = Field(Path("dataset"), description="Dataset cache directory")
    metafeatures_path: Path = Field(
        Path("metafeatures_minimal.csv"), description="Metafeatures cache path"
    )


class Settings(BaseModel):
    """Main application settings."""

    data: DataConfig
    timeseries: TimeSeriesConfig = Field(default_factory=TimeSeriesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    device: str | None = Field(None, description="Device (cpu, cuda, mps). Auto-detect if None")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Logging level"
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load settings from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Settings instance.
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save settings to YAML file.

        Args:
            path: Path to save YAML configuration.
        """
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)
