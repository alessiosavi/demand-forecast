# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Visualization Suite**
  - `plot_prediction_quality()` - 4-panel quality analysis (scatter, residuals, comparison, error by SKU)
  - `plot_validation_results()` - Enhanced with save_path and epoch-based naming
  - `plot_training_history()` - Training/validation loss curves
  - `plot_forecast_horizon()` - MAE and correlation by forecast step
  - `save_prediction_report()` - Complete report generator with summary, distribution, and CI plots

- **Pipeline Evaluate Method**
  - `ForecastPipeline.evaluate()` - Full evaluation with metrics (MSE, RMSE, MAE, MAPE, R², correlation)
  - JSON output support for metrics export
  - Optional plot generation during evaluation

- **CLI Enhancements**
  - `--plot` flag for train, evaluate, and predict commands
  - `--plot-dir` flag to customize plot output directory
  - `--show` flag to display plots interactively
  - Full implementation of `evaluate` command (was a stub)

- **Test Suite**
  - Tests for visualization functions
  - Tests for CLI commands with mocking
  - Tests for pipeline evaluate method
  - Tests for evaluator class

### Changed

- `training.num_workers` default changed from 4 to 0 to avoid file descriptor issues
- Predictions DataFrame now includes `actual` column alongside predictions
- Evaluator now accepts `plot_dir` and `epoch` parameters
- Trainer now accepts `plot_dir` parameter for validation plots

### Fixed

- Circular import issues with lazy imports using `__getattr__`
- `seed_worker` pickle error with DataLoader multiprocessing
- NumPy 2.x compatibility issues (pinned to `numpy>=1.24,<2.0`)
- Embedding dtype error (ensured categorical features use `torch.long`)
- Clustering error with small datasets (added sample count validation)
- Too many open files error (capped workers with `max_total_workers`)

---

## [1.0.0] - 2024-12-25

### Added

#### Core Features

- Transformer-based demand forecasting model (`AdvancedDemandForecastModel`)
- Multi-cluster model support via `ModelWrapper`
- Automatic time series clustering using K-means on TSFresh features
- Confidence interval calculation for predictions

#### Configuration

- Pydantic-based configuration system with validation
- YAML configuration file support
- Environment variable overrides

#### CLI

- `demand-forecast train` - Train forecasting models
- `demand-forecast evaluate` - Evaluate model performance
- `demand-forecast predict` - Generate predictions with confidence intervals
- `demand-forecast generate-data` - Generate synthetic sales data
- `demand-forecast preprocess` - Preprocess raw data to Parquet
- `demand-forecast version` - Show version information

#### Data Processing

- CSV and Parquet data loading with validation
- Configurable time resampling (daily, weekly, monthly)
- Categorical feature encoding (MultiLabelBinarizer, OneHotEncoder)
- Cyclical time feature generation (sin/cos encoding)
- Z-score based outlier detection and capping
- SKU filtering by data quality

#### Training

- AdamW optimizer with weight decay
- Cosine annealing learning rate schedule
- Early stopping with configurable patience
- Per-cluster optimization
- Checkpoint save/load
- Training callbacks protocol

#### Inference

- Batch and single-sample prediction
- Confidence interval calculation using historical residuals
- DataFrame output format
- Model loading from checkpoint

#### Utilities

- Memory management (garbage collection, CUDA cache)
- Visualization utilities (clustering plots, prediction plots)
- Metrics initialization (torchmetrics integration)

### Changed

- Complete refactoring from notebook-style code to production package
- Replaced global state with dependency injection pattern
- Replaced print statements with proper logging

### Removed

- Jupyter notebook artifacts and cell markers
- Global `scalers` and `label_encoders` dictionaries
- Magic commands (`%matplotlib inline`, etc.)

---

## [0.0.1] - 2024-12-20

### Added

- Initial prototype implementation
- Basic LSTM/Transformer model
- Notebook-based workflow

---

## Migration Guide

### From 0.0.1 to 1.0.0

The 1.0.0 release is a complete rewrite. Key changes:

#### Configuration

**Before (0.0.1):**

```python
window = 52
n_out = 16
batch_size = 128
```

**After (1.0.0):**

```yaml
# config.yaml
timeseries:
  window: 52
  n_out: 16
training:
  batch_size: 128
```

#### Data Processing

**Before (0.0.1):**

```python
scalers = {}  # Global dict
def scale_data(group):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(...)
    scalers[group.name] = scaler
    return scaled_data
```

**After (1.0.0):**

```python
from demand_forecast.data import ScalerManager

manager = ScalerManager()
scaled = manager.fit_transform("cluster_0", data)
manager.save(Path("scalers.joblib"))
```

#### Training

**Before (0.0.1):**

```python
models.train_model(
    models=model,
    dataloader_train=train_dls,
    ...
)
```

**After (1.0.0):**

```python
from demand_forecast.core import Trainer

trainer = Trainer(
    model=model,
    train_dataloaders=train_dls,
    ...
)
trainer.train()
```

#### CLI Usage

**Before (0.0.1):**

```bash
python sales_v2.py  # Edit file to change parameters
```

**After (1.0.0):**

```bash
demand-forecast train --config config.yaml
demand-forecast predict model.pt data.csv --output predictions.csv
```

---

[Unreleased]: https://github.com/yourusername/demand-forecast/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/demand-forecast/compare/v0.0.1...v1.0.0
[0.0.1]: https://github.com/yourusername/demand-forecast/releases/tag/v0.0.1
