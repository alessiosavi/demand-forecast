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
  d_model: 256
  # ... other model options

training:
  num_epochs: 10
  # ... other training options

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

### `model.sku_emb_dim`

**Type:** `int` | **Default:** `32` | **Min:** `1`

Embedding dimension for SKU identifiers.

### `model.cat_emb_dims`

**Type:** `int` | **Default:** `32` | **Min:** `1`

Embedding dimension for categorical features (color, size, etc.).

### `model.d_model`

**Type:** `int` | **Default:** `256` | **Min:** `1`

Transformer model dimension. Must be divisible by `nhead`.

```yaml
model:
  d_model: 256  # Standard size
  # d_model: 128  # Smaller/faster
  # d_model: 512  # Larger/more capacity
```

### `model.nhead`

**Type:** `int` | **Default:** `8` | **Min:** `1`

Number of attention heads. `d_model` must be divisible by `nhead`.

```yaml
model:
  d_model: 256
  nhead: 8  # 256/8 = 32 dim per head
```

### `model.num_encoder_layers`

**Type:** `int` | **Default:** `4` | **Min:** `1`

Number of Transformer encoder layers.

### `model.num_decoder_layers`

**Type:** `int` | **Default:** `4` | **Min:** `1`

Number of Transformer decoder layers.

### `model.dim_feedforward`

**Type:** `int` | **Default:** `2048` | **Min:** `1`

Dimension of feedforward network in Transformer layers. Typically 4x `d_model`.

### `model.dropout`

**Type:** `float` | **Default:** `0.3` | **Range:** `[0.0, 1.0)`

Dropout rate for regularization.

```yaml
model:
  dropout: 0.3  # Standard
  # dropout: 0.1  # Less regularization (larger datasets)
  # dropout: 0.5  # More regularization (smaller datasets)
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

## Complete Example

```yaml
# config.yaml - Production configuration

data:
  input_path: "data/production_sales.parquet"
  resample_period: "1W"
  max_zeros_ratio: 0.7
  date_column: "transaction_date"
  sku_column: "product_id"
  quantity_column: "units_sold"
  store_column: "store_code"

timeseries:
  window: 52
  n_out: 16
  test_size: 0.15

model:
  sku_emb_dim: 64
  cat_emb_dims: 32
  d_model: 256
  nhead: 8
  num_encoder_layers: 4
  num_decoder_layers: 4
  dim_feedforward: 1024
  dropout: 0.3

training:
  num_epochs: 50
  batch_size: 128
  learning_rate: 0.00001
  weight_decay: 0.01
  early_stop_patience: 5
  early_stop_min_delta: 0.5
  num_workers: 8
  pin_memory: true
  flatten_loss: true

output:
  model_dir: "models/production"
  cache_dir: "cache/datasets"
  metafeatures_path: "cache/metafeatures.csv"

seed: 42
device: "cuda"
log_level: "INFO"
```

## Environment Variables

Configuration can be overridden with environment variables:

```bash
export DEMAND_DATA__INPUT_PATH=/path/to/data.csv
export DEMAND_TRAINING__NUM_EPOCHS=20
export DEMAND_DEVICE=cuda
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

- `input_path` must exist
- `d_model` must be divisible by `nhead`
- Numeric values must be in valid ranges
