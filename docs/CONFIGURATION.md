# Configuration Guide

Complete reference for all configuration options in the demand forecasting system.

## Configuration File Format

Configuration is specified in YAML format. Create a `config.yaml` file:

```yaml
data:
  input_path: "data/sales.csv"
  # ... other data options

timeseries:
  window: 52
  # ... other time series options

model:
  model_type: "standard"
  d_model: 256
  # ... other model options

training:
  num_epochs: 10
  # ... other training options

tuning:
  enabled: false
  # ... tuning options

output:
  model_dir: "models"
  # ... other output options

seed: 42
device: null
log_level: "INFO"
```

## Data Configuration

### `data.input_path` (required)

**Type:** `Path`

Path to input data file (CSV or Parquet).

```yaml
data:
  input_path: "data/sales.csv"
  # or
  input_path: "data/sales.parquet"
```

### `data.resample_period`

**Type:** `str` | **Default:** `"1W"`

Pandas resample period string for time aggregation.

| Value | Description |
|-------|-------------|
| `"1D"` | Daily |
| `"1W"` | Weekly (default) |
| `"2W"` | Bi-weekly |
| `"1M"` | Monthly |

```yaml
data:
  resample_period: "1W"
```

### `data.max_zeros_ratio`

**Type:** `float` | **Default:** `0.7` | **Range:** `[0.0, 1.0]`

Maximum ratio of zero-sales weeks allowed per SKU. SKUs with more zeros are filtered out.

```yaml
data:
  max_zeros_ratio: 0.7  # Keep SKUs with <70% zero weeks
```

### `data.date_column`

**Type:** `str` | **Default:** `"date"`

Name of the date column in input data.

### `data.sku_column`

**Type:** `str` | **Default:** `"sku"`

Name of the SKU/product identifier column.

### `data.quantity_column`

**Type:** `str` | **Default:** `"qty"`

Name of the sales quantity column.

### `data.store_column`

**Type:** `str` | **Default:** `"store_id"`

Name of the store identifier column.

## Time Series Configuration

### `timeseries.window`

**Type:** `int` | **Default:** `52` | **Min:** `1`

Number of historical periods (lookback window) used as model input.

```yaml
timeseries:
  window: 52  # Use 52 weeks of history
```

**Recommendations:**

- Weekly data: 52 (1 year) or 104 (2 years)
- Daily data: 365 or 730
- Monthly data: 12 or 24

### `timeseries.n_out`

**Type:** `int` | **Default:** `16` | **Min:** `1`

Number of future periods to forecast (forecast horizon).

```yaml
timeseries:
  n_out: 16  # Forecast 16 weeks ahead
```

### `timeseries.test_size`

**Type:** `float` | **Default:** `0.2` | **Range:** `[0.0, 1.0)`

Fraction of data to use for validation/testing.

```yaml
timeseries:
  test_size: 0.2  # 20% for testing
```

## Model Configuration

### `model.model_type`

**Type:** `str` | **Default:** `"standard"` | **Options:** `"standard"`, `"advanced"`, `"lightweight"`

Model architecture to use.

| Type | Description | Use Case |
|------|-------------|----------|
| `standard` | Transformer encoder-decoder | General purpose, balanced performance |
| `advanced` | TFT/PatchTST-inspired model | Research, best accuracy |
| `lightweight` | TCN or MLP-Mixer | CPU deployment, low latency |

```yaml
model:
  model_type: "standard"  # or "advanced" or "lightweight"
```

### Common Model Parameters

#### `model.sku_emb_dim`

**Type:** `int` | **Default:** `32` | **Min:** `1`

Embedding dimension for SKU identifiers.

#### `model.cat_emb_dims`

**Type:** `int` | **Default:** `32` | **Min:** `1`

Embedding dimension for categorical features (color, size, etc.).

#### `model.d_model`

**Type:** `int` | **Default:** `256` | **Min:** `1`

Transformer model dimension. Must be divisible by `nhead`.

```yaml
model:
  d_model: 256  # Standard size
  # d_model: 128  # Smaller/faster
  # d_model: 512  # Larger/more capacity
```

#### `model.nhead`

**Type:** `int` | **Default:** `8` | **Min:** `1`

Number of attention heads. `d_model` must be divisible by `nhead`.

```yaml
model:
  d_model: 256
  nhead: 8  # 256/8 = 32 dim per head
```

#### `model.num_encoder_layers`

**Type:** `int` | **Default:** `4` | **Min:** `1`

Number of Transformer encoder layers.

#### `model.num_decoder_layers`

**Type:** `int` | **Default:** `4` | **Min:** `1`

Number of Transformer decoder layers.

#### `model.dim_feedforward`

**Type:** `int` | **Default:** `2048` | **Min:** `1`

Dimension of feedforward network in Transformer layers. Typically 4x `d_model`.

#### `model.dropout`

**Type:** `float` | **Default:** `0.3` | **Range:** `[0.0, 1.0)`

Dropout rate for regularization.

```yaml
model:
  dropout: 0.3  # Standard
  # dropout: 0.1  # Less regularization (larger datasets)
  # dropout: 0.5  # More regularization (smaller datasets)
```

### Standard Model Improvements

These options enhance the standard Transformer model with modern techniques.

#### `model.use_rope`

**Type:** `bool` | **Default:** `false`

Enable Rotary Position Embeddings (RoPE) for better sequence modeling.

```yaml
model:
  use_rope: true
```

#### `model.use_pre_layernorm`

**Type:** `bool` | **Default:** `false`

Use Pre-LayerNorm instead of Post-LayerNorm for improved training stability.

```yaml
model:
  use_pre_layernorm: true
```

#### `model.use_film_conditioning`

**Type:** `bool` | **Default:** `false`

Enable FiLM (Feature-wise Linear Modulation) conditioning on static features.

```yaml
model:
  use_film_conditioning: true
```

#### `model.use_improved_head`

**Type:** `bool` | **Default:** `false`

Use enhanced output head with GELU activation.

```yaml
model:
  use_improved_head: true
```

#### `model.stochastic_depth_rate`

**Type:** `float` | **Default:** `0.0` | **Range:** `[0.0, 1.0)`

Stochastic depth drop rate for regularization. Higher values = more regularization.

```yaml
model:
  stochastic_depth_rate: 0.1  # 10% layer drop probability
```

### Advanced Model Parameters

These options are specific to `model_type: "advanced"`.

#### `model.use_quantiles`

**Type:** `bool` | **Default:** `false`

Enable quantile output heads for probabilistic forecasting.

```yaml
model:
  model_type: "advanced"
  use_quantiles: true
  num_quantiles: 3
  quantiles: [0.1, 0.5, 0.9]  # Optional specific quantiles
```

#### `model.num_quantiles`

**Type:** `int` | **Default:** `3` | **Min:** `1`

Number of quantile outputs when `use_quantiles` is enabled.

#### `model.quantiles`

**Type:** `list[float] | null` | **Default:** `null`

Specific quantile values. If null, uses evenly spaced quantiles.

```yaml
model:
  quantiles: [0.1, 0.25, 0.5, 0.75, 0.9]
```

#### `model.use_decomposition`

**Type:** `bool` | **Default:** `false`

Enable Autoformer-style trend/seasonality decomposition.

```yaml
model:
  use_decomposition: true
  decomposition_kernel: 25
```

#### `model.decomposition_kernel`

**Type:** `int` | **Default:** `25` | **Min:** `3`

Kernel size for moving average decomposition.

#### `model.patch_size`

**Type:** `int` | **Default:** `4` | **Min:** `1`

Patch size for PatchTST-style time series tokenization.

```yaml
model:
  patch_size: 8  # Larger patches for longer sequences
```

#### `model.use_patch_embedding`

**Type:** `bool` | **Default:** `true`

Enable patch embedding for the advanced model.

#### `model.hidden_continuous_size`

**Type:** `int` | **Default:** `64` | **Min:** `1`

Hidden size for continuous feature processing in VSN.

### Lightweight Model Parameters

These options are specific to `model_type: "lightweight"`.

#### `model.lightweight_variant`

**Type:** `str` | **Default:** `"tcn"` | **Options:** `"tcn"`, `"mixer"`

Lightweight model architecture variant.

| Variant | Description | Parameters |
|---------|-------------|------------|
| `tcn` | Temporal Convolutional Network | ~500K-1M |
| `mixer` | MLP-Mixer architecture | ~200K-500K |

```yaml
model:
  model_type: "lightweight"
  lightweight_variant: "tcn"
```

#### `model.tcn_channels`

**Type:** `list[int]` | **Default:** `[32, 64, 64]`

Channel sizes for each TCN layer.

```yaml
model:
  tcn_channels: [32, 64, 128]  # Increasing channels
```

#### `model.tcn_kernel_size`

**Type:** `int` | **Default:** `3` | **Min:** `2`

Convolution kernel size for TCN.

```yaml
model:
  tcn_kernel_size: 5  # Larger receptive field
```

## Training Configuration

### `training.num_epochs`

**Type:** `int` | **Default:** `10` | **Min:** `1`

Number of training epochs.

```yaml
training:
  num_epochs: 10  # Quick training
  # num_epochs: 50  # Thorough training
  # num_epochs: 100  # Extended training
```

### `training.batch_size`

**Type:** `int` | **Default:** `128` | **Min:** `1`

Training batch size. Larger = faster training, more memory.

```yaml
training:
  batch_size: 128  # Standard
  # batch_size: 32   # Low memory
  # batch_size: 256  # High memory, GPU
```

### `training.learning_rate`

**Type:** `float` | **Default:** `1e-5` | **Min:** `>0`

Initial learning rate for AdamW optimizer.

```yaml
training:
  learning_rate: 0.00001  # Conservative (default)
  # learning_rate: 0.0001   # Faster convergence
  # learning_rate: 0.001    # Aggressive (may diverge)
```

### `training.weight_decay`

**Type:** `float` | **Default:** `1e-2` | **Min:** `>=0`

Weight decay (L2 regularization) for AdamW optimizer.

### `training.early_stop_patience`

**Type:** `int` | **Default:** `3` | **Min:** `1`

Number of epochs without improvement before early stopping.

### `training.early_stop_min_delta`

**Type:** `float` | **Default:** `1.0` | **Min:** `>=0`

Minimum improvement required to reset early stopping counter.

### `training.num_workers`

**Type:** `int` | **Default:** `0` | **Min:** `>=0`

Number of DataLoader worker processes. Set to 0 for single-process data loading (recommended to avoid file descriptor issues). Increase for faster data loading on systems with high ulimit.

### `training.pin_memory`

**Type:** `bool` | **Default:** `true`

Pin memory for faster GPU transfer. Disable if running out of memory.

### `training.flatten_loss`

**Type:** `bool` | **Default:** `true`

If true, use sum-reduced loss (total volume accuracy). If false, use point-by-point loss.

```yaml
training:
  flatten_loss: true   # Optimize total volume
  # flatten_loss: false  # Optimize per-timestep accuracy
```

## Tuning Configuration

Configuration for Optuna hyperparameter tuning.

### `tuning.enabled`

**Type:** `bool` | **Default:** `false`

Enable hyperparameter tuning.

```yaml
tuning:
  enabled: true
```

### `tuning.n_trials`

**Type:** `int` | **Default:** `50` | **Min:** `1`

Number of Optuna trials to run.

```yaml
tuning:
  n_trials: 100  # More trials for better results
```

### `tuning.timeout`

**Type:** `int | null` | **Default:** `null` | **Min:** `1`

Maximum time in seconds for tuning. If null, no timeout.

```yaml
tuning:
  timeout: 3600  # 1 hour timeout
```

### `tuning.direction`

**Type:** `str` | **Default:** `"minimize"` | **Options:** `"minimize"`, `"maximize"`

Optimization direction.

### `tuning.metric`

**Type:** `str` | **Default:** `"mse"` | **Options:** `"mse"`, `"mae"`, `"loss"`

Metric to optimize.

```yaml
tuning:
  metric: "mae"  # Optimize MAE instead of MSE
```

### `tuning.pruner`

**Type:** `str` | **Default:** `"median"` | **Options:** `"median"`, `"hyperband"`, `"none"`

Optuna pruner for early stopping unpromising trials.

| Pruner | Description |
|--------|-------------|
| `median` | Prune based on median intermediate values |
| `hyperband` | Hyperband pruning algorithm |
| `none` | No pruning |

```yaml
tuning:
  pruner: "hyperband"
```

### `tuning.sampler`

**Type:** `str` | **Default:** `"tpe"` | **Options:** `"tpe"`, `"random"`, `"cmaes"`

Optuna sampler for hyperparameter suggestions.

| Sampler | Description |
|---------|-------------|
| `tpe` | Tree-structured Parzen Estimator (recommended) |
| `random` | Random sampling |
| `cmaes` | CMA-ES algorithm |

```yaml
tuning:
  sampler: "tpe"
```

### `tuning.study_name`

**Type:** `str` | **Default:** `"demand_forecast_tuning"`

Name for the Optuna study (useful for persistence).

### `tuning.storage`

**Type:** `str | null` | **Default:** `null`

SQLite database path for study persistence. Enables resuming interrupted tuning.

```yaml
tuning:
  storage: "sqlite:///tuning.db"
```

### `tuning.n_jobs`

**Type:** `int` | **Default:** `1` | **Min:** `-1`

Number of parallel jobs. Use -1 for all CPU cores.

```yaml
tuning:
  n_jobs: 4  # 4 parallel trials
```

### `tuning.epochs_per_trial`

**Type:** `int` | **Default:** `5` | **Min:** `1`

Number of training epochs per trial.

### `tuning.early_stop_patience`

**Type:** `int` | **Default:** `2` | **Min:** `1`

Early stopping patience for each trial.

## Output Configuration

### `output.model_dir`

**Type:** `Path` | **Default:** `"models"`

Directory to save trained models.

### `output.cache_dir`

**Type:** `Path` | **Default:** `"dataset"`

Directory for cached preprocessed data.

### `output.metafeatures_path`

**Type:** `Path` | **Default:** `"metafeatures_minimal.csv"`

Path to cache TSFresh metafeatures.

## Global Settings

### `seed`

**Type:** `int` | **Default:** `42` | **Min:** `>=0`

Random seed for reproducibility.

### `device`

**Type:** `str | null` | **Default:** `null`

Computation device. If null, auto-detects best available.

```yaml
device: null    # Auto-detect
# device: "cpu"   # Force CPU
# device: "cuda"  # Force CUDA GPU
# device: "mps"   # Force Apple Silicon GPU
```

### `log_level`

**Type:** `str` | **Default:** `"INFO"` | **Options:** `DEBUG, INFO, WARNING, ERROR`

Logging verbosity level.

## Complete Examples

### Standard Training Configuration

```yaml
# config.yaml - Standard model training

data:
  input_path: "data/sales.parquet"
  resample_period: "1W"
  max_zeros_ratio: 0.7

timeseries:
  window: 52
  n_out: 16
  test_size: 0.15

model:
  model_type: "standard"
  d_model: 256
  nhead: 8
  num_encoder_layers: 4
  num_decoder_layers: 4
  dropout: 0.3

training:
  num_epochs: 50
  batch_size: 128
  learning_rate: 0.00001
  early_stop_patience: 5

output:
  model_dir: "models/standard"

seed: 42
device: "cuda"
log_level: "INFO"
```

### Advanced Model with Improvements

```yaml
# config.yaml - Advanced model with all improvements

data:
  input_path: "data/sales.parquet"
  resample_period: "1W"

timeseries:
  window: 52
  n_out: 16

model:
  model_type: "advanced"
  d_model: 256
  nhead: 8
  num_encoder_layers: 4
  num_decoder_layers: 4
  dropout: 0.3
  # Advanced features
  use_quantiles: true
  quantiles: [0.1, 0.5, 0.9]
  use_decomposition: true
  patch_size: 4

training:
  num_epochs: 100
  batch_size: 64
  learning_rate: 0.0001

output:
  model_dir: "models/advanced"

seed: 42
device: "cuda"
```

### Lightweight Model for Deployment

```yaml
# config.yaml - Lightweight model for CPU deployment

data:
  input_path: "data/sales.parquet"
  resample_period: "1W"

timeseries:
  window: 26  # Shorter window for faster inference
  n_out: 8

model:
  model_type: "lightweight"
  lightweight_variant: "tcn"
  tcn_channels: [32, 64, 64]
  tcn_kernel_size: 3
  d_model: 64  # Smaller for efficiency
  dropout: 0.2

training:
  num_epochs: 30
  batch_size: 256
  learning_rate: 0.001

output:
  model_dir: "models/lightweight"

device: "cpu"
```

### Hyperparameter Tuning Configuration

```yaml
# config.yaml - Hyperparameter tuning setup

data:
  input_path: "data/sales.parquet"
  resample_period: "1W"

timeseries:
  window: 52
  n_out: 16

model:
  model_type: "standard"
  d_model: 256  # Will be tuned
  nhead: 8
  dropout: 0.3  # Will be tuned

training:
  num_epochs: 10  # Override by tuning.epochs_per_trial
  batch_size: 128
  learning_rate: 0.00001  # Will be tuned

tuning:
  enabled: true
  n_trials: 50
  timeout: 7200  # 2 hours
  metric: "mse"
  direction: "minimize"
  sampler: "tpe"
  pruner: "median"
  storage: "sqlite:///tuning_study.db"
  n_jobs: 2
  epochs_per_trial: 5
  early_stop_patience: 2

output:
  model_dir: "models/tuned"

seed: 42
device: "cuda"
```

## Environment Variables

Configuration can be overridden with environment variables:

```bash
export DEMAND_DATA__INPUT_PATH=/path/to/data.csv
export DEMAND_TRAINING__NUM_EPOCHS=20
export DEMAND_DEVICE=cuda
export DEMAND_TUNING__ENABLED=true
```

Pattern: `DEMAND_<SECTION>__<OPTION>`

## Configuration Validation

All configuration is validated on load:

```python
from demand_forecast.config import Settings

try:
    settings = Settings.from_yaml(Path("config.yaml"))
except ValidationError as e:
    print(f"Configuration error: {e}")
```

Common validation errors:

- `d_model` must be divisible by `nhead`
- Numeric values must be in valid ranges
- `quantiles` must be between 0 and 1
- `model_type` must be one of: "standard", "advanced", "lightweight"
