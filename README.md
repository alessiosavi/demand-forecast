# Demand Forecast

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A production-ready demand forecasting system using Transformer neural networks. Predicts SKU-level demand with confidence intervals, supporting multiple time series clustering and categorical feature embeddings.

## Features

- **Transformer Architecture**: Encoder-decoder model with attention mechanisms for sequence-to-sequence forecasting
- **Multi-Cluster Training**: Automatic time series clustering with separate models per cluster
- **Categorical Embeddings**: Dynamic embedding layers for product attributes (color, size, category, etc.)
- **Confidence Intervals**: Statistical confidence bands on predictions
- **CLI Interface**: Easy-to-use command-line tools for training, evaluation, and prediction
- **Configurable Pipeline**: YAML-based configuration for all hyperparameters
- **Production Ready**: Proper logging, error handling, and type hints throughout

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/demand-forecast.git
cd demand-forecast

# Install in development mode
pip install -e ".[dev]"
```

### Generate Sample Data

```bash
demand-forecast generate-data sample_data.csv --products 50 --stores 10 --days 730
```

### Train a Model

```bash
# Copy and customize the example configuration
cp config.example.yaml config.yaml

# Train the model
demand-forecast train --config config.yaml
```

### Generate Predictions

```bash
demand-forecast predict models/model.pt input_data.csv --config config.yaml --output predictions.csv
```

## Architecture Overview

```text
demand_forecast/
├── cli.py                 # Typer CLI commands
├── config/settings.py     # Pydantic configuration models
├── core/
│   ├── pipeline.py        # Main orchestration
│   ├── trainer.py         # Training loop with callbacks
│   └── evaluator.py       # Validation and metrics
├── data/
│   ├── loader.py          # Data loading with validation
│   ├── preprocessor.py    # Scaling, filtering, resampling
│   ├── feature_engineering.py  # Categorical encoding
│   └── dataset.py         # PyTorch Dataset
├── models/
│   ├── transformer.py     # AdvancedDemandForecastModel
│   └── wrapper.py         # Multi-cluster ModelWrapper
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
  d_model: 256                 # Transformer dimension
  nhead: 8                     # Attention heads
  num_encoder_layers: 4
  num_decoder_layers: 4
  dropout: 0.3

training:
  num_epochs: 10
  batch_size: 128
  learning_rate: 0.00001
  early_stop_patience: 3
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `train` | Train the demand forecasting model |
| `evaluate` | Evaluate a trained model on test data |
| `predict` | Generate predictions with confidence intervals |
| `generate-data` | Generate synthetic sales data for testing |
| `preprocess` | Preprocess raw data and save as Parquet |
| `version` | Show version information |

### Common Options

All commands support `--verbose` (`-v`) for debug logging.

**Training with visualization:**
```bash
demand-forecast train --config config.yaml --plot --plot-dir plots/training
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

## Model Details

### Transformer Encoder-Decoder

The model uses a Transformer architecture optimized for time series:

1. **Static Embeddings**: SKU and categorical features are embedded and projected
2. **Encoder**: Processes historical sales with positional encoding
3. **Decoder**: Generates forecasts with causal masking and cross-attention
4. **Multi-Cluster**: Separate model per cluster for heterogeneous demand patterns

### Loss Functions

- **Point-by-Point**: Standard MSE on each forecast timestep
- **Flattened**: Sum-reduced MSE for total volume accuracy

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

### Project Structure

```
demand-forecast/
├── demand_forecast/       # Main package
├── tests/                 # Test suite
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
- CUDA (optional, for GPU acceleration)

See `pyproject.toml` for complete dependency list.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{demand_forecast,
  title = {Demand Forecast: Transformer-based Time Series Forecasting},
  year = {2024},
  url = {https://github.com/yourusername/demand-forecast}
}
```

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Configuration via [Pydantic](https://docs.pydantic.dev/)
- CLI powered by [Typer](https://typer.tiangolo.com/)
- Feature extraction with [TSFresh](https://tsfresh.readthedocs.io/)
