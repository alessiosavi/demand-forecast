# Architecture Guide

This document describes the architecture of the Demand Forecast system, including component design, data flow, and key design decisions.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI / API Layer                                │
│                                  cli.py                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Configuration Layer                              │
│                           config/settings.py                                │
│                     (Pydantic models with validation)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Orchestration Layer                               │
│                            core/pipeline.py                                 │
│                          (ForecastPipeline class)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│    Data Layer       │   │    Model Layer      │   │   Training Layer    │
│                     │   │                     │   │                     │
│ data/loader.py      │   │ models/transformer  │   │ core/trainer.py     │
│ data/preprocessor   │   │ models/wrapper.py   │   │ core/evaluator.py   │
│ data/dataset.py     │   │ models/components   │   │                     │
│ data/dataloader.py  │   │ models/losses.py    │   │                     │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Inference Layer                                  │
│                    inference/predictor.py, confidence.py                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Utilities Layer                                 │
│          utils/clustering.py, metrics.py, time_features.py, etc.           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Configuration Layer (`config/`)

#### `settings.py`

Pydantic-based configuration with full validation:

```python
class Settings(BaseModel):
    data: DataConfig           # Input paths, column names
    timeseries: TimeSeriesConfig  # Window, horizon, test size
    model: ModelConfig         # Architecture hyperparameters
    training: TrainingConfig   # Epochs, batch size, LR
    output: OutputConfig       # Model/cache directories
    seed: int
    device: Optional[str]
    log_level: str
```

**Design Decisions:**

- Uses Pydantic for automatic validation and type coercion
- YAML file support via `Settings.from_yaml()`
- Environment variable overrides possible via `pydantic-settings`

### 2. Data Layer (`data/`)

#### Data Flow

```
Raw CSV/Parquet
      │
      ▼ loader.py
DataFrame (standardized columns)
      │
      ▼ preprocessor.py
Resampled + Filtered + Scaled DataFrame
      │
      ▼ feature_engineering.py
Encoded Categoricals + Time Features
      │
      ▼ dataset.py
PyTorch Dataset with sliding windows
      │
      ▼ dataloader.py
DataLoaders with RoundRobin iteration
```

#### `loader.py`

- Loads CSV/Parquet files
- Standardizes column names
- Validates required columns exist
- Sets DatetimeIndex

#### `preprocessor.py`

Key classes:

```python
@dataclass
class ScalerManager:
    """Manages StandardScaler instances per cluster."""
    scalers: Dict[str, StandardScaler]

    def fit_transform(self, group_name: str, data: np.ndarray) -> np.ndarray
    def inverse_transform(self, group_name: str, data: np.ndarray) -> np.ndarray
    def save(self, path: Path) -> None
    def load(cls, path: Path) -> "ScalerManager"
```

**Design Decision:** Replaced global `scalers` dict with `ScalerManager` class for:

- Encapsulation and testability
- Serialization/deserialization
- No global state

#### `feature_engineering.py`

```python
@dataclass
class CategoricalEncoder:
    """Encapsulates categorical encoding logic."""
    encoders: Dict[str, Union[MultiLabelBinarizer, OneHotEncoder]]

    def fit_transform(df, categorical_columns, onehot_columns) -> DataFrame
    def transform(df) -> DataFrame
    def save(path) / load(path)
```

#### `dataset.py`

```python
class DemandDataset(Dataset):
    """PyTorch Dataset for demand forecasting."""

    def __getitem__(self, idx) -> Dict[str, Any]
    def collate_fn(self, batch) -> Dict[str, Tensor]
```

The custom `collate_fn` organizes data into model-expected format:

- `qty`: Past quantities `[batch, seq_len, 1]`
- `past_time`: Past time features `[batch, seq_len, 4]`
- `future_time`: Future time features `[batch, horizon, 4]`
- `sku`: SKU indices `[batch]`
- `cats`: Categorical tensors `{name: [batch, vocab_size]}`

#### `dataloader.py`

Uses `torchtnt.utils.data.RoundRobinIterator` to balance sampling across cluster-specific dataloaders.

### 3. Model Layer (`models/`)

#### Architecture Diagram

```
                    ┌─────────────────────┐
                    │   Input Features    │
                    └─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ SKU Embedding │    │  Cat Embeds   │    │  Past Series  │
│   [B, d_sku]  │    │  [B, d_cat]   │    │ [B, L, d_in]  │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────┬───────────┘                     │
                  ▼                                 │
          ┌───────────────┐                         │
          │  Static Proj  │                         │
          │  [B, d_model] │                         │
          └───────────────┘                         │
                  │                                 │
                  │    ┌────────────────────────────┘
                  │    ▼
                  │ ┌───────────────┐
                  │ │   Past Proj   │
                  │ │[B, L, d_model]│
                  │ └───────────────┘
                  │         │
                  └────┬────┘
                       ▼
              ┌─────────────────┐
              │ + Pos Encoding  │
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Transformer   │
              │     Encoder     │
              │   (4 layers)    │
              └─────────────────┘
                       │
                       ▼ (memory)
              ┌─────────────────┐         ┌─────────────────┐
              │   Transformer   │◄────────│  Future Time    │
              │     Decoder     │         │  [B, M, d_in]   │
              │   (4 layers)    │         │  + Pos Encoding │
              └─────────────────┘         └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Output Head   │
              │   [B, n_out]    │
              └─────────────────┘
```

#### `transformer.py` - AdvancedDemandForecastModel

Key features:

- **Multi-head attention** (8 heads by default)
- **Positional encoding** for temporal information
- **Causal masking** in decoder (future can't attend to future)
- **GELU activation** for better gradient flow
- **Dropout regularization** (0.3 default)

#### `wrapper.py` - ModelWrapper

```python
class ModelWrapper(nn.Module):
    """Wrapper for multiple cluster-specific models."""

    def __init__(self, n: int, **kwargs):
        self.models = nn.ModuleDict({
            f"{i}": AdvancedDemandForecastModel(**kwargs)
            for i in range(n)
        })

    def forward(self, n, qty, past_time, future_time, sku, cats):
        return self.models[n](qty, past_time, future_time, sku, cats)
```

**Design Decision:** One model per cluster because:

- Different demand patterns require different parameters
- Clustering groups similar time series
- Improves accuracy for heterogeneous product catalogs

### 4. Training Layer (`core/`)

#### `trainer.py`

```python
@dataclass
class Trainer:
    model: nn.Module
    train_dataloaders: Dict[str, DataLoader]
    test_dataloaders: Dict[str, DataLoader]
    num_epochs: int
    batch_size: int
    # ... more config

    def train(self, metrics, plot_every_n_epochs) -> None
    def save_checkpoint(self, path) -> None
    def load_checkpoint(self, path) -> None
```

Features:

- **Early stopping** with configurable patience
- **Cosine annealing** learning rate schedule
- **Per-cluster optimizers** (AdamW)
- **Callback protocol** for extensibility
- **Checkpoint save/load**

#### `evaluator.py`

```python
@dataclass
class ValidationResult:
    predictions: List[np.ndarray]
    actuals: List[np.ndarray]
    flatten_predictions: List[float]
    flatten_actuals: List[float]
    skus: List[int]
    avg_loss: float
    mse: float
    mae: float
    flatten_mse: float
    flatten_mae: float
    metrics: Dict[str, float]
```

### 5. Inference Layer (`inference/`)

#### `predictor.py`

```python
@dataclass
class Predictor:
    model: nn.Module
    scaler_manager: Optional[ScalerManager]
    sku_index_to_name: Optional[Dict[int, str]]

    def predict_batch(batch, cluster_id) -> PredictionResult
    def predict_dataloader(dataloader) -> PredictionResult
    def inverse_scale(predictions, cluster_id) -> np.ndarray
    def to_dataframe(result) -> pd.DataFrame
```

#### `confidence.py`

```python
def calculate_confidence_intervals(
    predictions: np.ndarray,
    actuals: np.ndarray,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate CI using historical residuals."""
```

### 6. Pipeline Orchestration (`core/pipeline.py`)

The `ForecastPipeline` class orchestrates the entire workflow:

```python
class ForecastPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.scaler_manager = ScalerManager()
        self.categorical_encoder = CategoricalEncoder()
        # ...

    def load_and_preprocess(self) -> pd.DataFrame
    def create_datasets(self) -> Dict[int, Tuple]
    def build_model(self) -> ModelWrapper
    def train(self, plot: bool = False, plot_dir: Path = None) -> None
    def evaluate(self, plot: bool = False, plot_dir: Path = None) -> dict
    def predict(self, confidence: float = 0.95, plot: bool = False) -> pd.DataFrame
    def save(self, path: Path) -> None
    def load(self, path: Path) -> None
```

#### Evaluation Method

The `evaluate()` method computes comprehensive metrics on test data:

```python
metrics = pipeline.evaluate(plot=True, plot_dir=Path("evaluation"))
# Returns: {
#     "mse": 0.1234,
#     "rmse": 0.3513,
#     "mae": 0.2567,
#     "mape": 12.34,        # Mean Absolute Percentage Error
#     "r_squared": 0.8765,
#     "correlation": 0.9365,
#     "total_samples": 1000,
# }
```

### 7. Visualization Layer (`utils/visualization.py`)

Comprehensive plotting utilities for model analysis:

| Function | Purpose |
|----------|---------|
| `plot_prediction_quality()` | 4-panel quality analysis (scatter, residuals, comparison, error by SKU) |
| `plot_validation_results()` | Line plot of actual vs predicted |
| `plot_training_history()` | Training/validation loss curves |
| `plot_forecast_horizon()` | MAE and correlation by forecast step |
| `plot_clustering()` | Elbow method and Davies-Bouldin visualization |
| `save_prediction_report()` | Complete report with summary, distribution, and CI plots |

Generated plots are saved to configurable directories and optionally displayed interactively.

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PHASE                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CSV File ─────► load_sales_data() ─────► resample_series()                 │
│                                                  │                           │
│                                                  ▼                           │
│                                          filter_skus()                       │
│                                                  │                           │
│                                                  ▼                           │
│                                      extract_metafeatures()                  │
│                                                  │                           │
│                                                  ▼                           │
│                                          find_best_k()                       │
│                                                  │                           │
│                                                  ▼                           │
│                                       scale_by_group() ◄──── ScalerManager  │
│                                                  │                           │
│                                                  ▼                           │
│  CategoricalEncoder ────► fit_transform() ◄───────                          │
│                                                  │                           │
│                                                  ▼                           │
│                                   calculate_time_features()                  │
│                                                  │                           │
│                                                  ▼                           │
│                                   create_time_series_data()                  │
│                                                  │                           │
│                                                  ▼                           │
│                                      create_dataloaders()                    │
│                                                  │                           │
│                                                  ▼                           │
│  ModelWrapper ◄────────────────────────── Trainer.train()                   │
│       │                                                                      │
│       ▼                                                                      │
│  torch.save() ─────► model.pt + scalers.joblib + encoders.joblib            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE PHASE                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  model.pt ─────► Predictor.from_checkpoint()                                │
│                           │                                                  │
│  Input Data ──────────────┼────► predict_dataloader()                       │
│                           │              │                                   │
│                           ▼              ▼                                   │
│                   PredictionResult ◄──────                                   │
│                           │                                                  │
│                           ▼                                                  │
│              calculate_confidence_intervals()                                │
│                           │                                                  │
│                           ▼                                                  │
│                    to_dataframe() ─────► predictions.csv                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Design Patterns Used

### 1. Dependency Injection

- Configuration passed to components rather than using globals
- `ScalerManager`, `CategoricalEncoder` injected into pipeline

### 2. Factory Pattern

- `create_dataloaders()` creates configured DataLoader instances
- `Predictor.from_checkpoint()` factory method

### 3. Strategy Pattern

- Loss modes (point vs flattened) selectable via config
- Callback protocol for training extensibility

### 4. Data Transfer Objects

- `ValidationResult`, `PredictionResult`, `ClusterResult` for structured returns

### 5. Builder Pattern (implicit)

- `ForecastPipeline` builds complex objects step by step

## Error Handling

Custom exception hierarchy in `core/exceptions.py`:

```
DemandForecastError (base)
├── DataValidationError    # Invalid input data
├── ConfigurationError     # Invalid configuration
├── ModelNotFoundError     # Missing model file
├── InsufficientDataError  # Not enough data
└── TrainingError          # Training failures
```

## Logging

Centralized logging configuration in `__init__.py`:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```

Each module uses `logger = logging.getLogger(__name__)` for namespaced logging.

## Thread Safety

- `ScalerManager` and `CategoricalEncoder` are not thread-safe by design
- For multi-threaded inference, create separate instances per thread
- DataLoader uses `persistent_workers=True` with proper seed management

## Memory Management

- `collect_garbage()` utility for explicit memory cleanup
- CUDA cache clearing between epochs
- Gradient checkpointing can be enabled for large models (future)

## Future Architecture Considerations

1. **Distributed Training**: Add PyTorch DDP support
2. **Model Registry**: MLflow integration for versioning
3. **Feature Store**: External feature store integration
4. **Streaming Inference**: gRPC or async HTTP support
5. **Model Serving**: ONNX export for optimized inference
