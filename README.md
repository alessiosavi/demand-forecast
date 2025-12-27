# Demand Forecast

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A production-ready demand forecasting system using state-of-the-art Transformer neural networks. Predicts SKU-level demand with confidence intervals, supporting multiple time series clustering, categorical feature embeddings, and advanced architectures including TFT, PatchTST, and lightweight models for CPU deployment.

## Features

- **Multiple Model Architectures**:
  - **Standard**: Transformer encoder-decoder with optional improvements (RoPE, Pre-LN, FiLM, stochastic depth)
  - **Advanced (V2)**: Research-grade model with TFT-style variable selection, PatchTST embeddings, and quantile forecasting
  - **Lightweight**: CPU-optimized models (TCN, MLP-Mixer) for edge deployment with ONNX export
- **Hyperparameter Tuning**: Optuna integration for automated hyperparameter optimization
- **Multi-Cluster Training**: Automatic time series clustering with separate models per cluster
- **Categorical Embeddings**: Dynamic embedding layers for product attributes (color, size, category, etc.)
- **Uncertainty Quantification**: Quantile forecasting and confidence intervals
- **CLI Interface**: Easy-to-use command-line tools for training, evaluation, tuning, and prediction
- **Configurable Pipeline**: YAML-based configuration for all hyperparameters
- **Production Ready**: Proper logging, error handling, type hints, and comprehensive test suite

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/alessiosavi/demand-forecast.git
cd demand-forecast

# Install in development mode
pip install -e ".[dev]"

# For ONNX export support (lightweight models)
pip install -e ".[deploy]"
```

### Generate Sample Data

```bash
demand-forecast generate-data sample_data.csv --products 50 --stores 10 --days 730
```

### Train a Model

```bash
# Copy and customize the example configuration
cp config.example.yaml config.yaml

# Train the standard model
demand-forecast train --config config.yaml

# Train with advanced model
demand-forecast train --config config.yaml --model-type advanced

# Train with lightweight model for CPU deployment
demand-forecast train --config config.yaml --model-type lightweight
```

### Hyperparameter Tuning

```bash
# Run Optuna hyperparameter tuning
demand-forecast tune --config config.yaml --n-trials 50 --timeout 3600
```

### Generate Predictions

```bash
demand-forecast predict models/model.pt input_data.csv --config config.yaml --output predictions.csv
```

## Model Architectures

### Standard Model (`AdvancedDemandForecastModel`)

Classic Transformer encoder-decoder with optional modern improvements:

| Feature | Description |
|---------|-------------|
| **RoPE** | Rotary Position Embeddings for better sequence modeling |
| **Pre-LayerNorm** | Improved training stability |
| **FiLM Conditioning** | Feature-wise Linear Modulation for static features |
| **Stochastic Depth** | Regularization through random layer dropping |
| **Improved Head** | GELU activation in output projection |

### Advanced Model V2 (`AdvancedDemandForecastModelV2`)

Research-grade architecture combining state-of-the-art techniques:

- **Variable Selection Networks (VSN)**: TFT-style feature importance learning
- **Gated Residual Networks (GRN)**: Enhanced information flow
- **Patch Embedding**: PatchTST-style time series tokenization
- **Series Decomposition**: Autoformer-style trend/seasonality separation
- **Quantile Output**: Probabilistic forecasting with uncertainty

### Lightweight Models

CPU-optimized architectures for edge deployment:

| Model | Description | Parameters |
|-------|-------------|------------|
| `LightweightDemandModel` | TCN + FiLM conditioning | < 1M |
| `LightweightMixerModel` | MLP-Mixer architecture | < 500K |

Features:
- ONNX export for optimized inference
- TorchScript compilation
- INT8 quantization support

## Architecture Overview

```text
demand_forecast/
├── cli.py                 # Typer CLI commands
├── config/settings.py     # Pydantic configuration models
├── core/
│   ├── pipeline.py        # Main orchestration
│   ├── trainer.py         # Training loop with callbacks
│   ├── evaluator.py       # Validation and metrics
│   └── tuning.py          # Optuna hyperparameter tuning
├── data/
│   ├── loader.py          # Data loading with validation
│   ├── preprocessor.py    # Scaling, filtering, resampling
│   ├── feature_engineering.py  # Categorical encoding
│   └── dataset.py         # PyTorch Dataset
├── models/
│   ├── transformer.py     # Standard model with improvements
│   ├── transformer_v2.py  # Advanced research-grade model
│   ├── lightweight.py     # CPU-optimized models
│   ├── components.py      # Reusable building blocks
│   ├── losses.py          # Loss functions (Huber, quantile, SMAPE)
│   └── wrapper.py         # Multi-cluster ModelWrapper + factory
├── inference/
│   ├── predictor.py       # Prediction wrapper
│   └── confidence.py      # Confidence interval calculation
└── utils/                 # Utilities (clustering, metrics, etc.)
```

## Configuration

The system is configured via YAML files. See `config.example.yaml` for all options:

```yaml
data:
  input_path: "sales_data.csv"
  resample_period: "1W"        # Weekly aggregation
  max_zeros_ratio: 0.7         # Filter sparse SKUs

timeseries:
  window: 52                   # 52-week lookback
  n_out: 16                    # 16-week forecast horizon

model:
  model_type: "standard"       # standard, advanced, or lightweight
  d_model: 256                 # Transformer dimension
  nhead: 8                     # Attention heads
  num_encoder_layers: 4
  num_decoder_layers: 4
  dropout: 0.3
  # Optional improvements
  use_rope: false              # Rotary Position Embeddings
  use_pre_layernorm: false     # Pre-LN for stability
  use_film_conditioning: false # FiLM for static features
  stochastic_depth_rate: 0.0   # Stochastic depth regularization

training:
  num_epochs: 10
  batch_size: 128
  learning_rate: 0.00001
  early_stop_patience: 3

tuning:
  enabled: false
  n_trials: 50
  timeout: 3600                # 1 hour timeout
  metric: "mse"                # Optimize MSE
  sampler: "tpe"               # TPE sampler
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `train` | Train the demand forecasting model |
| `evaluate` | Evaluate a trained model on test data |
| `predict` | Generate predictions with confidence intervals |
| `tune` | Run Optuna hyperparameter tuning |
| `generate-data` | Generate synthetic sales data for testing |
| `preprocess` | Preprocess raw data and save as Parquet |
| `version` | Show version information |

### Common Options

All commands support `--verbose` (`-v`) for debug logging.

**Training with visualization:**
```bash
demand-forecast train --config config.yaml --plot --plot-dir plots/training
```

**Hyperparameter tuning:**
```bash
demand-forecast tune --config config.yaml --n-trials 100 --metric mae
```

**Evaluation with metrics export:**
```bash
demand-forecast evaluate models/model.pt data.csv --config config.yaml \
    --output metrics.json --plot
```

**Prediction with confidence intervals and plots:**
```bash
demand-forecast predict models/model.pt data.csv --config config.yaml \
    --output predictions.csv --confidence 0.95 --plot
```

Run `demand-forecast --help` for detailed usage information.

## Hyperparameter Tuning

The system includes Optuna integration for automated hyperparameter optimization:

```python
from demand_forecast.core.tuning import HyperparameterTuner, TuningConfig, SearchSpace

# Define search space
search_space = SearchSpace(
    d_model=[64, 128, 256],
    nhead=[4, 8],
    num_layers=(1, 6),
    dropout=(0.1, 0.5),
    learning_rate=(1e-5, 1e-3),
)

# Configure tuning
config = TuningConfig(
    n_trials=50,
    timeout=3600,
    metric="mse",
    direction="minimize",
)

# Run tuning
tuner = HyperparameterTuner(config, search_space)
best_params = tuner.tune(train_data, val_data)
```

Or use the convenience function:

```python
from demand_forecast.core.tuning import quick_tune

best_params = quick_tune(
    train_dataloader=train_dl,
    val_dataloader=val_dl,
    n_trials=20,
)
```

## Model Details

### Transformer Encoder-Decoder

The model uses a Transformer architecture optimized for time series:

1. **Static Embeddings**: SKU and categorical features are embedded and projected
2. **Encoder**: Processes historical sales with positional encoding
3. **Decoder**: Generates forecasts with causal masking and cross-attention
4. **Multi-Cluster**: Separate model per cluster for heterogeneous demand patterns

### New Components

| Component | Description |
|-----------|-------------|
| `RotaryPositionEmbedding` | RoPE for better long-range dependencies |
| `GatedResidualNetwork` | TFT-style gated residual connections |
| `VariableSelectionNetwork` | Learnable feature importance |
| `PatchEmbedding` | PatchTST-style time series tokenization |
| `SeriesDecomposition` | Autoformer trend/seasonality separation |
| `FiLMConditioning` | Feature-wise Linear Modulation |
| `TemporalConvNet` | Dilated causal convolutions |
| `InterpretableMultiHeadAttention` | TFT-style interpretable attention |

### Loss Functions

- **HuberLoss**: Robust to outliers (default)
- **QuantileLoss**: For probabilistic forecasting
- **CombinedForecastLoss**: Huber + quantile + decomposition
- **SMAPELoss**: Symmetric MAPE for percentage errors
- **MASELoss**: Mean Absolute Scaled Error

### Clustering

Time series are clustered using K-means on TSFresh meta-features. The optimal K is selected via the elbow method with Davies-Bouldin Index validation.

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
ruff check .
mypy demand_forecast/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=demand_forecast

# Run specific test categories
pytest tests/test_models/           # Model tests
pytest tests/test_models/test_tuning.py  # Tuning tests
```

### Project Structure

```
demand-forecast/
├── demand_forecast/       # Main package
├── tests/                 # Test suite
│   ├── test_models/       # Model and tuning tests
│   ├── test_core/         # Core module tests
│   └── test_utils/        # Utility tests
├── docs/                  # Documentation
├── config.example.yaml    # Example configuration
├── pyproject.toml         # Project metadata
└── Makefile              # Development commands
```

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and component details
- [API Reference](docs/API.md) - Module and class documentation
- [Configuration Guide](docs/CONFIGURATION.md) - All configuration options
- [Quick Start Guide](docs/QUICKSTART.md) - Step-by-step tutorial
- [Development Guide](docs/DEVELOPMENT.md) - Contributing and development setup

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Optuna 3.0+ (for hyperparameter tuning)
- CUDA (optional, for GPU acceleration)

See `pyproject.toml` for complete dependency list.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{demand_forecast,
  title = {Demand Forecast: Transformer-based Time Series Forecasting},
  year = {2025},
  url = {https://github.com/alessiosavi/demand-forecast}
}
```

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Hyperparameter tuning with [Optuna](https://optuna.org/)
- Configuration via [Pydantic](https://docs.pydantic.dev/)
- CLI powered by [Typer](https://typer.tiangolo.com/)
- Feature extraction with [TSFresh](https://tsfresh.readthedocs.io/)
