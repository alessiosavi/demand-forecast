# Demand Forecast

A production-ready demand forecasting system using state-of-the-art Transformer neural networks.

## Overview

Demand Forecast predicts SKU-level demand with confidence intervals, supporting multiple time series clustering, categorical feature embeddings, and advanced architectures including TFT, PatchTST, and lightweight models for CPU deployment.

## Key Features

- **Multiple Model Architectures**
    - **Standard**: Transformer encoder-decoder with optional improvements (RoPE, Pre-LN, FiLM, stochastic depth)
    - **Advanced (V2)**: Research-grade model with TFT-style variable selection, PatchTST embeddings, and quantile forecasting
    - **Lightweight**: CPU-optimized models (TCN, MLP-Mixer) for edge deployment with ONNX export

- **Hyperparameter Tuning**: Optuna integration for automated hyperparameter optimization

- **Multi-Cluster Training**: Automatic time series clustering with separate models per cluster

- **Categorical Embeddings**: Dynamic embedding layers for product attributes

- **Uncertainty Quantification**: Quantile forecasting and confidence intervals

- **CLI Interface**: Easy-to-use command-line tools for training, evaluation, tuning, and prediction

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Generate sample data
demand-forecast generate-data sample_data.csv --products 50 --stores 10 --days 730

# Train a model
demand-forecast train --config config.yaml

# Generate predictions
demand-forecast predict models/model.pt sample_data.csv --output predictions.csv
```

## Model Architectures

| Model | Description | Parameters | Use Case |
|-------|-------------|------------|----------|
| Standard | Transformer with optional improvements | ~2-5M | General purpose |
| Advanced V2 | TFT/PatchTST research model | ~5-10M | High accuracy |
| Lightweight | TCN/MLP-Mixer | < 1M | CPU/Edge deployment |

## Documentation

- [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes
- [Configuration Guide](CONFIGURATION.md) - All configuration options
- [Architecture Guide](ARCHITECTURE.md) - System design and components
- [API Reference](API.md) - Complete API documentation
- [Development Guide](DEVELOPMENT.md) - Contributing and development setup

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Optuna 3.0+ (for hyperparameter tuning)
- CUDA (optional, for GPU acceleration)

## License

MIT License - see [LICENSE](https://github.com/alessiosavi/demand-forecast/blob/main/LICENSE) for details.
