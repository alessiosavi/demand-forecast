# Quick Start Guide

This guide walks you through training your first demand forecasting model.

## Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- (Optional) CUDA-compatible GPU for faster training

## Installation

### Option 1: Install as Package (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/demand-forecast.git
cd demand-forecast

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

After installation, use commands like:

```bash
demand-forecast train --config config.yaml
```

### Option 2: Run Without Installing

If you prefer not to install the package, you can run directly using Python:

```bash
# Clone and enter the directory
git clone https://github.com/yourusername/demand-forecast.git
cd demand-forecast

# Install dependencies only
pip install -r requirements.txt
# Or manually: pip install torch pandas numpy scikit-learn pydantic typer tqdm matplotlib torchmetrics tsfresh scipy pyyaml joblib dill

# Run using python -m
python -m demand_forecast --help

# Or run the CLI directly
python demand_forecast/cli.py --help
```

### Verify Setup

```bash
# If installed:
demand-forecast version

# If not installed:
python -m demand_forecast version
# or
python demand_forecast/cli.py version
```

## Complete Workflow Example

Below is a complete example you can follow step-by-step. We'll use the "without installing" method, but you can replace `python demand_forecast/cli.py` with `demand-forecast` if you installed the package.

### Step 1: Generate Sample Data

Generate synthetic sales data for testing:

```bash
python demand_forecast/cli.py generate-data \
    sample_sales.csv \
    --products 50 \
    --stores 10 \
    --days 730 \
    --seed 42
```

This creates `sample_sales.csv` with:

- 50 products across 10 stores
- 2 years (730 days) of daily sales
- Realistic patterns: seasonality, promotions, zero-sales periods
- Categorical attributes: color, size, category, subcategory

For a quick test with smaller data:

```bash
python demand_forecast/cli.py generate-data \
    sample_data_small.csv \
    --products 10 \
    --stores 3 \
    --days 730 \
    --seed 42
```

### Step 2: Create Configuration File

Copy and edit the example configuration:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml`:

```yaml
data:
  input_path: "sample_sales.csv"      # Your data file
  resample_period: "1W"               # Weekly aggregation
  max_zeros_ratio: 0.7                # Keep SKUs with <70% zero sales

timeseries:
  window: 52                          # 1 year lookback (52 weeks)
  n_out: 16                           # 16-week forecast horizon
  test_size: 0.2                      # 20% for validation

model:
  d_model: 128                        # Model dimension (smaller = faster)
  nhead: 4                            # Attention heads
  num_encoder_layers: 2               # Encoder layers
  num_decoder_layers: 2               # Decoder layers
  dropout: 0.3                        # Regularization

training:
  num_epochs: 5                       # Training epochs
  batch_size: 64                      # Batch size
  learning_rate: 0.0001               # Learning rate
  num_workers: 0                      # Data loading workers (0 = main process)

output:
  model_dir: "models"                 # Where to save the model

seed: 42
device: null                          # Auto-detect (cuda/mps/cpu)
log_level: "INFO"
```

### Step 3: Train the Model

```bash
python demand_forecast/cli.py train --config config.yaml
```

You'll see output like:

```
2024-01-15 10:00:00 [INFO] - Loading configuration from config.yaml
2024-01-15 10:00:00 [INFO] - Loading data from sample_sales.csv
2024-01-15 10:00:01 [INFO] - Resampled 365000 entries into 52000 entries
2024-01-15 10:00:02 [INFO] - Filtered from 500 to 420 SKUs
2024-01-15 10:00:05 [INFO] - Finding optimal K in range 2-29 (n_samples=420)
2024-01-15 10:00:10 [INFO] - Built model with 5 cluster models
2024-01-15 10:00:10 [INFO] - Starting training...
Epoch [1/5] - Loss: 12.3456: 100%|████████████| 100/100 [00:30<00:00]
...
2024-01-15 10:05:00 [INFO] - Training complete. Model saved to models/model.pt
```

The trained model is saved to:

- `models/model.pt` - The model weights
- `models/scalers.joblib` - Data scalers
- `models/encoders.joblib` - Categorical encoders

### Step 4: Generate Predictions

Once training completes, generate predictions:

```bash
python demand_forecast/cli.py predict \
    models/model.pt \
    sample_sales.csv \
    --config config.yaml \
    --output predictions.csv \
    --confidence 0.95
```

The output `predictions.csv` contains:

| Column | Description |
|--------|-------------|
| `sku` | Product identifier |
| `cluster` | Cluster the SKU belongs to |
| `prediction` | Forecasted quantity (sum over forecast horizon) |
| `lower_bound` | 95% confidence interval lower bound |
| `upper_bound` | 95% confidence interval upper bound |

Example output:

```csv
sku,cluster,prediction,lower_bound,upper_bound
product_1_store_1,0,1234.56,1100.23,1368.89
product_2_store_1,0,567.89,433.56,702.22
...
```

### Step 5: Evaluate the Model

Evaluate the model performance on test data:

```bash
python demand_forecast/cli.py evaluate \
    models/model.pt \
    sample_sales.csv \
    --config config.yaml \
    --output metrics.json \
    --plot
```

This outputs evaluation metrics:

```
==================================================
EVALUATION RESULTS
==================================================
  MSE:         12.3456
  RMSE:        3.5136
  MAE:         2.1234
  MAPE:        15.67%
  R-squared:   0.8542
  Correlation: 0.9245
  Samples:     1000
==================================================
```

The `metrics.json` file contains all metrics in JSON format for programmatic access.

### Step 6: View Predictions

```bash
# View first few predictions
head predictions.csv

# Or in Python
python -c "import pandas as pd; print(pd.read_csv('predictions.csv').head(10))"
```

## Using the Python API

For more control, use the Python API directly:

```python
from pathlib import Path
from demand_forecast.config.settings import Settings
from demand_forecast.core.pipeline import ForecastPipeline

# Load configuration
settings = Settings.from_yaml(Path("config.yaml"))

# Create pipeline
pipeline = ForecastPipeline(settings)

# Step 1: Load and preprocess data
df = pipeline.load_and_preprocess()
print(f"Loaded {len(df)} samples")

# Step 2: Create datasets (includes clustering)
datasets = pipeline.create_datasets()
print(f"Created {len(datasets)} cluster datasets")

# Step 3: Build model
model = pipeline.build_model()
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

# Step 4: Train
pipeline.train()

# Step 5: Save
pipeline.save(Path("models/model.pt"))

# Step 6: Generate predictions
predictions_df = pipeline.predict(confidence=0.95)
predictions_df.to_csv("predictions.csv", index=False)
print(f"Saved {len(predictions_df)} predictions")
```

## Common Issues

### Out of Memory

If you run out of GPU/CPU memory:

1. Reduce batch size: `batch_size: 32`
2. Reduce model size: `d_model: 64`, `num_encoder_layers: 1`
3. Force CPU: `device: "cpu"`

### Too Many Open Files Error

If you see `OSError: [Errno 24] Too many open files`:

1. Keep `num_workers: 0` (default, uses main process)
2. Or increase system limit: `ulimit -n 4096`

### No SKUs After Filtering

If all SKUs are filtered out:

1. Increase `max_zeros_ratio: 0.9`
2. Use more data (more days)
3. Check your data has enough history (at least `window + n_out` weeks)

### Clustering Error with Small Data

If you see `ValueError: Number of labels is X. Valid values are 2 to n_samples - 1`:

1. Generate more data: `--products 50 --stores 10`
2. The clustering automatically adapts to sample count

### Poor Accuracy

To improve accuracy:

1. More epochs: `num_epochs: 20`
2. Larger model: `d_model: 256`, `num_encoder_layers: 4`
3. Lower learning rate: `learning_rate: 0.00001`
4. More training data

## Visualization

The CLI supports generating quality plots during training and prediction.

### Training Plots

Generate validation plots during training with `--plot`:

```bash
python demand_forecast/cli.py train \
    --config config.yaml \
    --plot \
    --plot-dir training_plots
```

This creates validation and quality analysis plots every 3 epochs in the specified directory:

- `validation_epoch3.png` - Actual vs predicted comparison
- `quality_epoch3.png` - Comprehensive quality analysis (scatter, residuals, error distribution)

### Prediction Plots

Generate prediction quality plots with `--plot`:

```bash
python demand_forecast/cli.py predict \
    models/model.pt \
    sample_sales.csv \
    --config config.yaml \
    --output predictions.csv \
    --plot \
    --plot-dir prediction_plots
```

This creates:

- `prediction_quality.png` - 4-panel quality analysis
- `predictions_distribution.png` - Prediction distribution histograms
- `predictions_confidence_intervals.png` - Predictions with confidence bands
- `predictions_summary.txt` - Summary statistics

### Python API

You can also generate plots programmatically:

```python
from demand_forecast.utils.visualization import (
    plot_prediction_quality,
    save_prediction_report,
)

# After running predictions
predictions_df = pipeline.predict(
    confidence=0.95,
    plot=True,
    plot_dir="my_plots",
)

# Or call visualization directly
plot_prediction_quality(
    actuals=actuals_array,
    predictions=predictions_array,
    skus=sku_list,
    save_path="quality_analysis.png",
)
```

## Quick Test Commands

Here's a minimal test you can run in under 5 minutes:

```bash
# 1. Generate small dataset
python demand_forecast/cli.py generate-data test_data.csv --products 20 --stores 5 --days 400

# 2. Create minimal config
cat > test_config.yaml << 'EOF'
data:
  input_path: "test_data.csv"
  resample_period: "1W"
  max_zeros_ratio: 0.8
timeseries:
  window: 26
  n_out: 8
  test_size: 0.2
model:
  d_model: 64
  nhead: 4
  num_encoder_layers: 1
  num_decoder_layers: 1
training:
  num_epochs: 2
  batch_size: 32
output:
  model_dir: "test_models"
seed: 42
EOF

# 3. Train (with plots)
python demand_forecast/cli.py train --config test_config.yaml --plot

# 4. Evaluate the model
python demand_forecast/cli.py evaluate \
    test_models/model.pt \
    test_data.csv \
    --config test_config.yaml \
    --output test_metrics.json \
    --plot

# 5. Predict (with plots)
python demand_forecast/cli.py predict \
    test_models/model.pt \
    test_data.csv \
    --config test_config.yaml \
    --output test_predictions.csv \
    --plot

# 6. View results
cat test_predictions.csv
cat test_metrics.json

# 7. View generated plots
ls test_models/training_plots/
ls test_models/evaluation/
ls test_models/plots/
```

## Next Steps

- Read the [Configuration Guide](CONFIGURATION.md) for all options
- See the [Architecture Guide](ARCHITECTURE.md) for system design
- Check the [API Reference](API.md) for programmatic usage
- Read the [Development Guide](DEVELOPMENT.md) to contribute
