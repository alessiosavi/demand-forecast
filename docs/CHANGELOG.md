# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Debug Scripts**
  - `debug_notebook.py` - Comprehensive debug script with cell markers for Jupyter conversion
  - `debug_quick.py` - Minimal quick test script (~1-2 minute runtime)
  - Covers data generation, training, tuning, evaluation, and inference

- **Extended DataConfig Fields** (`config/settings.py`)
  - `product_id_column` - Source column name for product ID (renamed to sku_column)
  - `sales_qty_column` - Source column name for sales quantity (renamed to quantity_column)
  - `categorical_columns` - Explicit list of columns to encode (auto-detected if null)
  - `onehot_columns` - Subset of categorical columns for OneHot encoding

- **Model Factory Exports** (`models/__init__.py`)
  - Exported `create_model()` factory function
  - Exported `MODEL_REGISTRY` for model type lookup

- **Advanced Model Architecture (V2)**
  - `AdvancedDemandForecastModelV2` - Research-grade model combining TFT, PatchTST, and Autoformer concepts
  - Variable Selection Networks (VSN) for learnable feature importance
  - Gated Residual Networks (GRN) for enhanced information flow
  - Patch embedding for PatchTST-style time series tokenization
  - Series decomposition for trend/seasonality separation
  - Quantile output heads for probabilistic forecasting
  - Interpretable multi-head attention with feature weights

- **Lightweight Models for CPU Deployment**
  - `LightweightDemandModel` - TCN-based model with FiLM conditioning (< 1M parameters)
  - `LightweightMixerModel` - MLP-Mixer architecture (< 500K parameters)
  - ONNX export support for optimized inference
  - TorchScript compilation support
  - INT8 quantization utilities

- **Standard Model Improvements**
  - `use_rope` - Rotary Position Embeddings for better sequence modeling
  - `use_pre_layernorm` - Pre-LayerNorm for improved training stability
  - `use_film_conditioning` - FiLM (Feature-wise Linear Modulation) for static features
  - `use_improved_head` - Enhanced output head with GELU activation
  - `stochastic_depth_rate` - Stochastic depth regularization

- **New Model Components** (`models/components.py`)
  - `RotaryPositionEmbedding` and `apply_rotary_pos_emb` - RoPE implementation
  - `GatedResidualNetwork` - TFT-style GRN with optional context
  - `VariableSelectionNetwork` - Feature importance learning
  - `PatchEmbedding` - Time series patch tokenization
  - `SeriesDecomposition` - Moving average decomposition
  - `FiLMConditioning` - Feature-wise linear modulation layer
  - `StochasticDepth` - Layer dropping regularization
  - `InterpretableMultiHeadAttention` - Attention with interpretable weights
  - `TemporalBlock` and `TemporalConvNet` - Dilated causal convolutions

- **New Loss Functions** (`models/losses.py`)
  - `CombinedForecastLoss` - Huber + quantile + decomposition loss
  - `SMAPELoss` - Symmetric Mean Absolute Percentage Error
  - `MASELoss` - Mean Absolute Scaled Error

- **Hyperparameter Tuning** (`core/tuning.py`)
  - `HyperparameterTuner` - Optuna-based hyperparameter optimization
  - `TuningConfig` - Configuration dataclass for tuning parameters
  - `SearchSpace` - Flexible search space definition
  - `quick_tune()` - Convenience function for quick tuning
  - Support for TPE, Random, and CMA-ES samplers
  - Median and Hyperband pruners for early stopping
  - SQLite persistence for study resumption

- **Model Factory and Registry** (`models/wrapper.py`)
  - `MODEL_REGISTRY` - Central registry for model types
  - `create_model()` - Factory function for model instantiation
  - `EnsembleWrapper` - Wrapper for model ensembles with averaging
  - `model_type` parameter in `ModelWrapper`

- **Configuration Extensions** (`config/settings.py`)
  - `TuningConfig` - Hyperparameter tuning configuration
  - Extended `ModelConfig` with all new model parameters
  - `model_type` field: "standard", "advanced", or "lightweight"
  - Lightweight model parameters: `tcn_channels`, `tcn_kernel_size`, `lightweight_variant`
  - Advanced model parameters: `use_quantiles`, `num_quantiles`, `use_decomposition`, etc.

- **Comprehensive Test Suite** (`tests/test_models/`)
  - `test_model_forward.py` - Forward pass tests for all model architectures
  - `test_training_flow.py` - End-to-end training integration tests
  - `test_tuning.py` - Hyperparameter tuning tests

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

- **Dependencies**
  - Added `optuna>=3.0` as core dependency for hyperparameter tuning
  - Added optional `deploy` extras: `onnx>=1.14`, `onnxruntime>=1.15`

### Changed

- **Configuration Propagation Improvements**
  - `extract_metafeatures()` now accepts `date_column` parameter (was hardcoded to "date")
  - `extract_metafeatures()` now receives `store_column` from config
  - `create_time_series_data()` now accepts `store_column` parameter
  - Pipeline merge operations use `cfg.store_column` instead of hardcoded "store_id"
  - Pipeline index restoration uses `cfg.date_column` instead of hardcoded "date"
  - `_encode_categoricals()` uses config fields or auto-detects categorical columns

- **Configuration Files**
  - `config.example.yaml` - Comprehensive template with all 125+ options documented
  - `config.yaml` - Minimal working configuration aligned with example

- `training.num_workers` default changed from 4 to 0 to avoid file descriptor issues
- Predictions DataFrame now includes `actual` column alongside predictions
- Evaluator now accepts `plot_dir` and `epoch` parameters
- Trainer now accepts `plot_dir` parameter for validation plots
- `ModelWrapper` now supports `model_type` parameter for architecture selection
- `calculate_time_features()` now accepts both `DatetimeIndex` (returns array) and `DataFrame` (modifies in-place)
- Test random seed changed from 42 to 123 to avoid macOS + PyTorch threading segfaults
- Added single-threading configuration in tests for stability

### Fixed

- **Pipeline Bugs**
  - `scale_by_group()` not receiving `quantity_column` parameter
  - `load_sales_data()` not receiving `product_id_column` and `sales_qty_column` parameters
  - Clustering merge losing DatetimeIndex (changed to `how="left"` with proper index restoration)
  - Model wrapper creating models with consecutive keys (0,1,2,3) instead of actual bin labels
  - Trainer creating optimizers with wrong keys (now uses `train_dataloaders.keys()`)
  - `n_out=1` hardcoded in pipeline model creation (now uses `ts_cfg.n_out`)

- **Dataset Bugs**
  - `future_time` slicing using wrong indices (changed to `[-self.n_out:, 6:]`)
  - Added `n_out` parameter to `DemandDataset.__init__` for correct slicing

- **Visualization Bugs**
  - `save_prediction_report()` error bar calculation could produce negative values (now uses `np.maximum(0, ...)`)

- Circular import issues with lazy imports using `__getattr__`
- `seed_worker` pickle error with DataLoader multiprocessing
- NumPy 2.x compatibility issues (pinned to `numpy>=1.24,<2.0`)
- Embedding dtype error (ensured categorical features use `torch.long`)
- Clustering error with small datasets (added sample count validation)
- Too many open files error (capped workers with `max_total_workers`)
- Output shape bug in standard model simple head (was returning 3D instead of 2D)
- macOS + PyTorch threading segfault in tests (added single-threading configuration)
- DataConfig path validation (deferred to runtime for test compatibility)

---

## [1.0.0] - 2025-12-25

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

## [0.0.1] - 2025-12-20

### Added

- Initial prototype implementation
- Basic LSTM/Transformer model
- Notebook-based workflow

---

## Migration Guide

### From 1.0.0 to Unreleased

#### Model Selection

**Before (1.0.0):**

```python
from demand_forecast.models import ModelWrapper

wrapper = ModelWrapper(n=5, **kwargs)
```

**After (Unreleased):**

```python
from demand_forecast.models import create_model, ModelWrapper

# Using factory function (recommended)
model = create_model(
    model_type="advanced",  # "standard", "advanced", or "lightweight"
    n_clusters=5,
    **kwargs
)

# Or with ModelWrapper directly
wrapper = ModelWrapper(
    n=5,
    model_type="advanced",
    **kwargs
)
```

#### Hyperparameter Tuning

**New in Unreleased:**

```python
from demand_forecast.core.tuning import quick_tune, HyperparameterTuner

# Quick tuning
best_params = quick_tune(train_dl, val_dl, n_trials=20)

# Full control
tuner = HyperparameterTuner(config, search_space)
best_params = tuner.tune(train_data, val_data)
```

#### Configuration Updates

**Before (1.0.0):**

```yaml
model:
  d_model: 256
  nhead: 8
```

**After (Unreleased):**

```yaml
model:
  model_type: "standard"  # NEW: Choose architecture
  d_model: 256
  nhead: 8
  # NEW: Optional improvements
  use_rope: false
  use_pre_layernorm: false
  use_film_conditioning: false
  stochastic_depth_rate: 0.0

# NEW: Tuning configuration
tuning:
  enabled: false
  n_trials: 50
  metric: "mse"
```

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

[Unreleased]: https://github.com/alessiosavi/demand-forecast/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/alessiosavi/demand-forecast/compare/v0.0.1...v1.0.0
[0.0.1]: https://github.com/alessiosavi/demand-forecast/releases/tag/v0.0.1
