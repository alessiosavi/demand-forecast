# API Reference

Complete API documentation for the `demand_forecast` package.

## Table of Contents

- [Configuration](#configuration)
- [Data](#data)
- [Models](#models)
- [Training](#training)
- [Inference](#inference)
- [Utilities](#utilities)

---

## Configuration

### `demand_forecast.config.Settings`

Main configuration class that holds all settings.

```python
from demand_forecast.config import Settings

# Load from YAML
settings = Settings.from_yaml(Path("config.yaml"))

# Create programmatically
settings = Settings(
    data=DataConfig(input_path=Path("data.csv")),
    timeseries=TimeSeriesConfig(window=52, n_out=16),
    model=ModelConfig(d_model=256, nhead=8),
    training=TrainingConfig(num_epochs=10),
    output=OutputConfig(model_dir=Path("models")),
)

# Save to YAML
settings.to_yaml(Path("config_output.yaml"))
```

#### Nested Configurations

| Class | Key Fields | Description |
|-------|------------|-------------|
| `DataConfig` | `input_path`, `resample_period`, `max_zeros_ratio` | Data loading settings |
| `TimeSeriesConfig` | `window`, `n_out`, `test_size` | Time series parameters |
| `ModelConfig` | `d_model`, `nhead`, `dropout` | Model architecture |
| `TrainingConfig` | `num_epochs`, `batch_size`, `learning_rate` | Training hyperparameters |
| `OutputConfig` | `model_dir`, `cache_dir` | Output paths |

---

## Data

### `demand_forecast.data.load_sales_data`

Load sales data from CSV or Parquet file.

```python
from demand_forecast.data import load_sales_data

df = load_sales_data(
    path=Path("sales.csv"),
    date_column="date",
    sku_column="sku",
    quantity_column="qty",
    store_column="store_id",
    product_id_column="product_id",  # Optional: rename to sku_column
    sales_qty_column="sales_qty",    # Optional: rename to quantity_column
)
```

**Returns:** `pd.DataFrame` with DatetimeIndex

**Raises:** `DataValidationError` if required columns missing

---

### `demand_forecast.data.ScalerManager`

Manages StandardScaler instances per group.

```python
from demand_forecast.data import ScalerManager

manager = ScalerManager()

# Fit and transform
scaled = manager.fit_transform("cluster_0", raw_data)

# Transform with existing scaler
scaled = manager.transform("cluster_0", new_data)

# Inverse transform
original = manager.inverse_transform("cluster_0", scaled)

# Save/Load
manager.save(Path("scalers.joblib"))
manager = ScalerManager.load(Path("scalers.joblib"))
```

---

### `demand_forecast.data.CategoricalEncoder`

Encapsulates categorical encoding logic.

```python
from demand_forecast.data import CategoricalEncoder

encoder = CategoricalEncoder()

# Fit and transform
df = encoder.fit_transform(
    df,
    categorical_columns=["color", "size", "category"],
    onehot_columns=["is_promo", "store_id"],  # Use OneHotEncoder
)

# Get encoded column names
encoded_cols = encoder.get_encoded_columns()
# ['encoded_color', 'encoded_size', 'encoded_category', ...]

# Save/Load
encoder.save(Path("encoders.joblib"))
encoder = CategoricalEncoder.load(Path("encoders.joblib"))
```

---

### `demand_forecast.data.DemandDataset`

PyTorch Dataset for demand forecasting.

```python
from demand_forecast.data import DemandDataset

dataset = DemandDataset(
    raw_dataset=x_train,      # np.ndarray [N, window, features]
    cat_dataset=cat_train,    # np.ndarray [N, num_cats]
    y=y_train,                # np.ndarray [N, n_out]
    encoded_categorical_features=["encoded_color", "encoded_size"],
)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)
```

---

### `demand_forecast.data.create_time_series_data`

Create sliding window datasets from grouped data.

```python
from demand_forecast.data import create_time_series_data

x_train, y_train, cat_train, x_test, y_test, cat_test = create_time_series_data(
    series=df,
    series_features=["qty_scaled", "sku_code", "p_t_sin", "p_t_cos"],
    encoded_categorical_features=["encoded_color", "encoded_size"],
    test_size=0.2,
    window=52,
    n_out=16,
)
```

---

### `demand_forecast.data.create_dataloaders`

Create train/test DataLoaders for all clusters.

```python
from demand_forecast.data import create_dataloaders

train_dls, test_dls, train_dss, test_dss = create_dataloaders(
    raw_datasets=raw_ds,  # Dict[int, Tuple]
    encoded_categorical_features=encoded_cols,
    batch_size=128,
    num_workers=4,
    seed=42,
    device=torch.device("cuda"),
)
```

---

## Models

### `demand_forecast.models.AdvancedDemandForecastModel`

Transformer encoder-decoder for demand forecasting.

```python
from demand_forecast.models import AdvancedDemandForecastModel

model = AdvancedDemandForecastModel(
    sku_vocab_size=1000,
    sku_emb_dim=32,
    cat_features_dim={"color": 10, "size": 5},
    cat_emb_dims=32,
    past_time_features_dim=5,
    future_time_features_dim=4,
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=2048,
    dropout=0.3,
    n_out=1,
    max_past_len=52,
    max_future_len=16,
)

# Forward pass
output = model(qty, past_time, future_time, sku, cats)
# output: [batch_size, n_out]
```

---

### `demand_forecast.models.ModelWrapper`

Wrapper for multiple cluster-specific models.

```python
from demand_forecast.models import ModelWrapper

wrapper = ModelWrapper(
    n=5,  # Number of clusters
    sku_vocab_size=1000,
    # ... same kwargs as AdvancedDemandForecastModel
)

# Forward pass with cluster ID
output = wrapper("0", qty, past_time, future_time, sku, cats)

# Get specific model
model = wrapper.get_model("0")

# Number of models
print(wrapper.num_models)  # 5
```

---

### `demand_forecast.models.PositionalEncoding`

Sinusoidal positional encoding for Transformers.

```python
from demand_forecast.models import PositionalEncoding

pos_enc = PositionalEncoding(d_model=256, max_len=1000)
encoded = pos_enc(x)  # x: [batch, seq_len, d_model]
```

---

## Training

### `demand_forecast.core.Trainer`

Training loop with early stopping and callbacks.

```python
from demand_forecast.core import Trainer, EarlyStopConfig

trainer = Trainer(
    model=model,
    train_dataloaders=train_dls,
    test_dataloaders=test_dls,
    num_epochs=10,
    batch_size=128,
    learning_rate=1e-5,
    weight_decay=1e-2,
    early_stop=EarlyStopConfig(patience=3, min_delta=1.0),
    flatten_loss=True,
    device=torch.device("cuda"),
    callbacks=[],  # Optional TrainingCallback implementations
)

trainer.set_example_counts(train_count=10000, test_count=2000)
trainer.train(metrics=metrics_dict, plot_every_n_epochs=3)

# Save/load checkpoints
trainer.save_checkpoint("checkpoint.pt")
trainer.load_checkpoint("checkpoint.pt")
```

---

### `demand_forecast.core.Evaluator`

Model evaluation with metrics.

```python
from demand_forecast.core import Evaluator

evaluator = Evaluator(
    model=model,
    criterion=nn.MSELoss(),
    batch_size=128,
    total_examples=2000,
    flatten_loss=True,
    metrics=metrics_dict,
)

result = evaluator.validate(dataloader, plot=True)
# ValidationResult with predictions, metrics, etc.
```

---

### `demand_forecast.core.ForecastPipeline`

End-to-end pipeline orchestration.

```python
from demand_forecast.core import ForecastPipeline
from demand_forecast.config import Settings

settings = Settings.from_yaml(Path("config.yaml"))
pipeline = ForecastPipeline(settings)

# Step by step
df = pipeline.load_and_preprocess()
datasets = pipeline.create_datasets()
model = pipeline.build_model()
pipeline.train(plot=True, plot_dir=Path("plots/training"))

# Save
pipeline.save(Path("models/model.pt"))

# Load and evaluate
pipeline.load(Path("models/model.pt"))
metrics = pipeline.evaluate(plot=True, plot_dir=Path("plots/evaluation"))
# metrics: {"mse": 0.1, "rmse": 0.316, "mae": 0.2, "mape": 15.0, "r_squared": 0.85, ...}

# Generate predictions with visualization
predictions_df = pipeline.predict(
    confidence=0.95,
    plot=True,
    plot_dir=Path("plots/predictions"),
)
# predictions_df columns: sku, cluster, prediction, actual, lower_bound, upper_bound
```

#### Pipeline Methods

| Method | Description |
|--------|-------------|
| `load_and_preprocess()` | Load data, resample, filter, scale, encode features |
| `create_datasets()` | Create time series datasets per cluster |
| `build_model()` | Initialize ModelWrapper with cluster models |
| `train(plot, plot_dir)` | Train with optional validation plots |
| `evaluate(plot, plot_dir, show_plots)` | Compute metrics (MSE, RMSE, MAE, MAPE, R², correlation) |
| `predict(confidence, plot, plot_dir, show_plots)` | Generate predictions with confidence intervals |
| `save(path)` | Save model, scalers, and encoders |
| `load(path)` | Load model and artifacts |

---

## Inference

### `demand_forecast.inference.Predictor`

Prediction wrapper for trained models.

```python
from demand_forecast.inference import Predictor

# From checkpoint
predictor = Predictor.from_checkpoint(
    model_path=Path("model.pt"),
    scaler_path=Path("scalers.joblib"),
    device=torch.device("cpu"),
)

# Or from existing model
predictor = Predictor(
    model=model,
    scaler_manager=scaler_manager,
    sku_index_to_name=sku_index_to_name,
    device=device,
)

# Predict single batch
result = predictor.predict_batch(
    batch=batch,
    cluster_id="0",
    return_confidence=True,
    confidence_level=0.95,
    validation_targets=targets,  # Optional for CI calculation
)

# Predict entire dataloader
result = predictor.predict_dataloader(dataloader, confidence_level=0.95)

# Convert to DataFrame
df = predictor.to_dataframe(result, include_confidence=True)
```

---

### `demand_forecast.inference.calculate_confidence_intervals`

Calculate confidence intervals from predictions and actuals.

```python
from demand_forecast.inference import calculate_confidence_intervals

lower, upper = calculate_confidence_intervals(
    predictions=predictions,  # np.ndarray or torch.Tensor
    actuals=actuals,
    confidence=0.95,
)
```

---

## Utilities

### `demand_forecast.utils.find_best_k`

Find optimal number of clusters using elbow method.

```python
from demand_forecast.utils import find_best_k

result = find_best_k(
    series=feature_matrix,  # 2D array
    k_range=range(2, 20),
    random_state=42,
    plot=True,
)

print(f"Optimal K: {result.best_k}")
kmeans = result.kmeans
labels = kmeans.fit_predict(feature_matrix)
```

---

### `demand_forecast.utils.calculate_time_features`

Add cyclical time features to DataFrame.

```python
from demand_forecast.utils import calculate_time_features

# Present time features (p_t_sin, p_t_cos, p_m_sin, p_m_cos)
calculate_time_features(df, label="present")

# Future time features (f_t_sin, f_t_cos, f_m_sin, f_m_cos)
calculate_time_features(df, label="future", window=16)
```

---

### `demand_forecast.utils.remove_outliers`

Remove outliers using Z-score method.

```python
from demand_forecast.utils import remove_outliers

clipped_data, had_outliers = remove_outliers(
    data=series,  # pd.Series
    n=3,          # Z-score threshold
)
```

---

### `demand_forecast.utils.create_timeseries`

Create sliding window sequences.

```python
from demand_forecast.utils import create_timeseries

x_seq, cat_seq, y_seq = create_timeseries(
    X=features,      # [T, F] or [T]
    cat=categories,  # [T]
    y=targets,       # [T]
    window=52,
    n_out=16,
    shift=0,
)
```

---

### `demand_forecast.utils.init_metrics`

Initialize torchmetrics regression metrics.

```python
from demand_forecast.utils import init_metrics

metrics = init_metrics()
# {'MeanSquaredError': <Metric>, 'MeanAbsoluteError': <Metric>, ...}
```

---

### `demand_forecast.utils.collect_garbage`

Force garbage collection and CUDA cache clearing.

```python
from demand_forecast.utils import collect_garbage

collected = collect_garbage()
print(f"Freed {collected} objects")
```

---

## Visualization

### `demand_forecast.utils.visualization.plot_prediction_quality`

Generate comprehensive 4-panel quality analysis plot.

```python
from demand_forecast.utils.visualization import plot_prediction_quality

plot_prediction_quality(
    actuals=np.array([1.0, 2.0, 3.0]),
    predictions=np.array([1.1, 2.0, 2.9]),
    skus=["SKU1", "SKU2", "SKU3"],  # Optional
    title="Prediction Quality Analysis",
    save_path=Path("quality.png"),
    show=False,
    max_samples=100,  # Limit samples in line plot
)
```

**Panels:**
1. Actual vs Predicted scatter with metrics (MSE, MAE, MAPE, Correlation)
2. Residual distribution histogram
3. Time series comparison (limited samples)
4. Error by SKU (top 20) or error percentiles

---

### `demand_forecast.utils.visualization.plot_validation_results`

Simple validation line plot.

```python
from demand_forecast.utils.visualization import plot_validation_results

plot_validation_results(
    flatten_actuals=[1.0, 2.0, 3.0],
    flatten_predictions=[1.1, 2.0, 2.9],
    title="Validation Results - Epoch 5",
    save_path=Path("validation.png"),
    show=False,
)
```

---

### `demand_forecast.utils.visualization.plot_training_history`

Plot training and validation loss curves.

```python
from demand_forecast.utils.visualization import plot_training_history

plot_training_history(
    train_losses=[1.0, 0.8, 0.6, 0.5],
    val_losses=[1.1, 0.9, 0.7, 0.6],  # Optional
    title="Training History",
    save_path=Path("history.png"),
    show=False,
)
```

---

### `demand_forecast.utils.visualization.save_prediction_report`

Generate complete prediction report with multiple files.

```python
from demand_forecast.utils.visualization import save_prediction_report

saved_files = save_prediction_report(
    predictions_df=df,  # DataFrame with: sku, prediction, lower_bound, upper_bound
    output_dir=Path("report"),
    prefix="predictions",
)
# saved_files: {
#     "summary": Path("report/predictions_summary.txt"),
#     "distribution": Path("report/predictions_distribution.png"),
#     "confidence_intervals": Path("report/predictions_confidence_intervals.png"),
# }
```

---

## Exceptions

All custom exceptions inherit from `DemandForecastError`:

```python
from demand_forecast.core import (
    DemandForecastError,
    DataValidationError,
    ConfigurationError,
    ModelNotFoundError,
    InsufficientDataError,
    TrainingError,
)

try:
    df = load_sales_data(Path("missing.csv"))
except DataValidationError as e:
    print(f"Data error: {e}")
except DemandForecastError as e:
    print(f"General error: {e}")
```

---

## Synthetic Data

### `demand_forecast.synthetic.generate_sales_data`

Generate realistic synthetic sales data.

```python
from demand_forecast.synthetic import generate_sales_data

df = generate_sales_data(
    num_days=365 * 4,
    num_products=100,
    num_stores=20,
    start_date="2020-01-01",
    seed=42,
)

# Columns: date, store_id, product_id, sales_qty, price, stock,
#          discount, is_promo_day, color, size, category, subcategory
```
