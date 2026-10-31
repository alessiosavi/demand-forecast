"""Pydantic configuration for long panels and the shared forecasting trainer."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetConfig(BaseModel):
    """Dataset source, column roles, and preprocessing configuration."""

    source: Literal["long", "fev"] = "long"
    path: Path | None = None
    id_columns: list[str]
    timestamp_column: str = "timestamp"
    target: str
    target_support: Literal["nonnegative", "real"]
    known_dynamic: list[str] = Field(default_factory=list)
    past_dynamic: list[str] = Field(default_factory=list)
    static_path: Path | None = None
    static_categorical: list[str] = Field(default_factory=list)
    static_numeric: list[str] = Field(default_factory=list)
    freq: str | None = None
    missing_target: Literal["zero", "nan"] = "zero"
    duplicates: Literal["error", "sum"] = "error"
    resample: str | None = None


class ForecastConfig(BaseModel):
    """Forecast horizon, quantile levels, and context configuration."""

    horizon: int = Field(ge=1)
    quantiles: list[float] = Field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    max_context: int = 512
    min_context: int = 1

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, v: list[float]) -> list[float]:
        """Require increasing quantiles inside (0, 1), including the median."""
        if any(not 0 < q < 1 for q in v):
            raise ValueError("Quantiles must be strictly between 0 and 1")
        if any(left >= right for left, right in zip(v, v[1:])):
            raise ValueError("Quantiles must be strictly increasing")
        if 0.5 not in v:
            raise ValueError("Quantiles must contain 0.5")
        return v

    @property
    def median_index(self) -> int:
        """Return the index of the median quantile."""
        return self.quantiles.index(0.5)


class ModelConfig(BaseModel):
    """Standard v2 architecture; embeddings belong to the shared model skeleton."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["v2"] = "v2"
    d_model: int = Field(64, ge=1)
    nhead: int = Field(4, ge=1)
    num_encoder_layers: int = Field(2, ge=1)
    num_decoder_layers: int = Field(2, ge=1)
    dim_feedforward: int = Field(256, ge=1)
    dropout: float = Field(0.1, ge=0, le=1)
    cat_emb_dim: int = Field(16, ge=1)
    hidden_continuous_size: int = Field(64, ge=1)
    stochastic_depth_rate: float = Field(0.0, ge=0, lt=1)
    use_rope: bool = True
    use_patch_embedding: bool = False
    use_decomposition: bool = False

    @model_validator(mode="after")
    def validate_architecture(self) -> "ModelConfig":
        if self.use_patch_embedding:
            raise ValueError("v2 does not support patch embedding")
        if self.use_decomposition:
            raise ValueError("v2 does not support decomposition")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if self.use_rope and (self.d_model // self.nhead) % 2:
            raise ValueError("RoPE requires an even dimension per attention head")
        return self

    def backbone_kwargs(self) -> dict:
        """Return only arguments accepted by the v2 temporal backbone."""
        return self.model_dump(
            exclude={"family", "cat_emb_dim", "use_patch_embedding", "use_decomposition"}
        )


class TrainingConfig(BaseModel):
    """Trainer controls plus the inner-validation and refit policy."""

    model_config = ConfigDict(extra="forbid")

    base_steps: int = Field(3000, ge=1)
    batch_size: int = Field(128, ge=1)
    learning_rate: float = Field(1e-3, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(1e-2, ge=0, allow_inf_nan=False)
    warmup_steps: int = Field(100, ge=0)
    grad_clip: float = Field(1.0, gt=0, allow_inf_nan=False)
    eval_every: int = Field(200, ge=1)
    patience: int = Field(5, ge=1)
    rel_tol: float = Field(0.002, ge=0, lt=1)
    amp: bool = True
    checkpoint_every: int = Field(200, ge=1)
    refit: Literal["none", "full"] = "none"
    min_train_history: int | None = Field(None, ge=1)


class TuningConfig(BaseModel):
    """Hyperparameter tuning configuration."""

    enabled: bool = Field(False, description="Enable hyperparameter tuning")
    n_trials: int = Field(50, ge=0, description="Number of Optuna trials")
    timeout: int | None = Field(None, ge=1, description="Timeout in seconds (None = no limit)")
    direction: Literal["minimize", "maximize"] = Field(
        "minimize", description="Optimization direction"
    )
    metric: Literal["mse", "mae", "loss"] = Field("mse", description="Metric to optimize")
    pruner: Literal["median", "hyperband", "none"] = Field(
        "median", description="Optuna pruner type"
    )
    sampler: Literal["tpe", "random", "cmaes"] = Field("tpe", description="Optuna sampler type")
    study_name: str = Field("demand_forecast_tuning", description="Optuna study name")
    storage: str | None = Field(None, description="SQLite path for persistence")
    n_jobs: int = Field(1, ge=-1, description="Parallel jobs (-1 = all cores)")
    epochs_per_trial: int = Field(5, ge=1, description="Epochs per trial")
    early_stop_patience: int = Field(2, ge=1, description="Early stop patience per trial")


class Settings(BaseModel):
    """Application configuration; paths are resolved from the working directory."""

    model_config = ConfigDict(extra="forbid")

    data: DatasetConfig
    forecast: ForecastConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    seed: int = Field(0, ge=0, le=2**32 - 1)
    device: str | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load and validate a YAML mapping."""
        import yaml

        with Path(path).open(encoding="utf-8") as stream:
            return cls.model_validate(yaml.safe_load(stream))

    def to_yaml(self, path: Path) -> None:
        """Write JSON-compatible values, including paths, as plain YAML."""
        import yaml

        with Path(path).open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.model_dump(mode="json"), stream, sort_keys=False)
