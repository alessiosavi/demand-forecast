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

Standard Transformer encoder-decoder for demand forecasting with optional improvements.

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
    # Optional improvements
    use_rope=False,              # Rotary Position Embeddings
    use_pre_layernorm=False,     # Pre-LayerNorm for stability
    use_film_conditioning=False, # FiLM for static features
    use_improved_head=False,     # GELU activation in output head
    stochastic_depth_rate=0.0,   # Stochastic depth regularization
)

# Forward pass
output = model(qty, past_time, future_time, sku, cats)
# output: [batch_size, n_out]
```

---

### `demand_forecast.models.AdvancedDemandForecastModelV2`

Research-grade model combining TFT, PatchTST, and Autoformer concepts.

```python
from demand_forecast.models import AdvancedDemandForecastModelV2

model = AdvancedDemandForecastModelV2(
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
    n_out=16,
    max_past_len=52,
    max_future_len=16,
    # V2-specific parameters
    use_quantiles=True,          # Probabilistic forecasting
    num_quantiles=3,             # Number of quantile outputs (e.g., 0.1, 0.5, 0.9)
    use_decomposition=True,      # Autoformer-style trend/seasonality separation
    patch_len=4,                 # PatchTST patch length
    use_vsn=True,                # Variable Selection Networks
    use_grn=True,                # Gated Residual Networks
)

# Forward pass
output = model(qty, past_time, future_time, sku, cats)
# output: [batch_size, n_out] or [batch_size, n_out, num_quantiles] if use_quantiles=True
```

**Key Features:**
- **Variable Selection Networks (VSN)**: TFT-style learnable feature importance
- **Gated Residual Networks (GRN)**: Enhanced information flow
- **Patch Embedding**: PatchTST-style time series tokenization
- **Series Decomposition**: Autoformer-style trend/seasonality separation
- **Quantile Output**: Probabilistic forecasting with uncertainty

---

### `demand_forecast.models.LightweightDemandModel`

CPU-optimized TCN-based model for edge deployment (< 1M parameters).

```python
from demand_forecast.models import LightweightDemandModel

model = LightweightDemandModel(
    sku_vocab_size=1000,
    sku_emb_dim=16,
    cat_features_dim={"color": 10, "size": 5},
    cat_emb_dims=16,
    past_time_features_dim=5,
    future_time_features_dim=4,
    n_out=16,
    tcn_channels=[32, 64, 128],  # TCN channel progression
    tcn_kernel_size=3,           # Convolution kernel size
    use_film=True,               # FiLM conditioning for static features
    dropout=0.2,
)

# Forward pass
output = model(qty, past_time, future_time, sku, cats)
# output: [batch_size, n_out]

# ONNX export
model.export_onnx("model.onnx", example_inputs)

# TorchScript export
scripted = model.to_torchscript()
```

---

### `demand_forecast.models.LightweightMixerModel`

MLP-Mixer architecture for ultra-lightweight deployment (< 500K parameters).

```python
from demand_forecast.models import LightweightMixerModel

model = LightweightMixerModel(
    sku_vocab_size=1000,
    sku_emb_dim=16,
    cat_features_dim={"color": 10, "size": 5},
    cat_emb_dims=16,
    past_time_features_dim=5,
    future_time_features_dim=4,
    n_out=16,
    hidden_dim=128,
    num_mixer_layers=4,
    dropout=0.1,
)

# Forward pass
output = model(qty, past_time, future_time, sku, cats)
```

---

### `demand_forecast.models.create_model`

Factory function for creating models by type.

```python
from demand_forecast.models import create_model

# Create standard model
model = create_model(
    model_type="standard",
    n_clusters=5,
    sku_vocab_size=1000,
    # ... other kwargs
)

# Create advanced model
model = create_model(
    model_type="advanced",
    n_clusters=5,
    use_quantiles=True,
    # ... other kwargs
)

# Create lightweight model
model = create_model(
    model_type="lightweight",
    n_clusters=5,
    tcn_channels=[32, 64],
    # ... other kwargs
)
```

---

### `demand_forecast.models.ModelWrapper`

Wrapper for multiple cluster-specific models.

```python
from demand_forecast.models import ModelWrapper

wrapper = ModelWrapper(
    n=5,  # Number of clusters
    sku_vocab_size=1000,
    model_type="standard",  # "standard", "advanced", or "lightweight"
    # ... same kwargs as the selected model type
)

# Forward pass with cluster ID
output = wrapper("0", qty, past_time, future_time, sku, cats)

# Get specific model
model = wrapper.get_model("0")

# Number of models
print(wrapper.num_models)  # 5
```

#### Model Types

| Type | Model Class | Description |
|------|-------------|-------------|
| `"standard"` | `AdvancedDemandForecastModel` | Transformer with optional improvements |
| `"advanced"` | `AdvancedDemandForecastModelV2` | TFT/PatchTST research model |
| `"lightweight"` | `LightweightDemandModel` | TCN for CPU deployment |

---

### `demand_forecast.models.EnsembleWrapper`

Wrapper for model ensembles with prediction averaging.

```python
from demand_forecast.models import EnsembleWrapper

# Create ensemble from multiple models
ensemble = EnsembleWrapper(models=[model1, model2, model3])

# Forward pass (averages predictions)
output = ensemble(qty, past_time, future_time, sku, cats)
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

## Model Components

### `demand_forecast.models.components`

Reusable building blocks for model architectures.

| Component | Description |
|-----------|-------------|
| `RotaryPositionEmbedding` | RoPE implementation for better long-range dependencies |
| `GatedResidualNetwork` | TFT-style GRN with optional context |
| `VariableSelectionNetwork` | Learnable feature importance |
| `PatchEmbedding` | PatchTST-style time series tokenization |
| `SeriesDecomposition` | Moving average decomposition |
| `FiLMConditioning` | Feature-wise Linear Modulation layer |
| `StochasticDepth` | Layer dropping regularization |
| `InterpretableMultiHeadAttention` | TFT-style interpretable attention |
| `TemporalBlock` | Dilated causal convolution block |
| `TemporalConvNet` | Full TCN architecture |

```python
from demand_forecast.models.components import (
    RotaryPositionEmbedding,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    FiLMConditioning,
    TemporalConvNet,
)

# RoPE
rope = RotaryPositionEmbedding(dim=64, max_seq_len=512)
q_rotated, k_rotated = rope(query, key)

# GRN
grn = GatedResidualNetwork(input_dim=256, hidden_dim=128, output_dim=256, dropout=0.1)
output = grn(x, context=static_features)

# VSN
vsn = VariableSelectionNetwork(input_dim=256, num_inputs=10, hidden_dim=128)
selected, weights = vsn(inputs)  # weights are interpretable

# FiLM
film = FiLMConditioning(feature_dim=256, condition_dim=64)
modulated = film(features, condition)

# TCN
tcn = TemporalConvNet(num_inputs=1, num_channels=[32, 64, 128], kernel_size=3)
output = tcn(sequence)  # [batch, channels, seq_len]
```

---

## Loss Functions

### `demand_forecast.models.losses`

Specialized loss functions for forecasting.

```python
from demand_forecast.models.losses import (
    CombinedForecastLoss,
    SMAPELoss,
    MASELoss,
)

# Combined loss: Huber + quantile + decomposition
loss_fn = CombinedForecastLoss(
    huber_weight=1.0,
    quantile_weight=0.5,
    decomposition_weight=0.1,
    quantiles=[0.1, 0.5, 0.9],
)
loss = loss_fn(predictions, targets, trend=trend, seasonal=seasonal)

# SMAPE loss
smape = SMAPELoss()
loss = smape(predictions, targets)

# MASE loss
mase = MASELoss(seasonality=52)
loss = mase(predictions, targets, historical=past_values)
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

## Hyperparameter Tuning

### `demand_forecast.core.tuning.HyperparameterTuner`

Optuna-based hyperparameter optimization.

```python
from demand_forecast.core.tuning import HyperparameterTuner, TuningConfig, SearchSpace

# Define search space
search_space = SearchSpace(
    d_model=[64, 128, 256],          # Categorical choices
    nhead=[4, 8],
    num_layers=(1, 6),               # Integer range
    dropout=(0.1, 0.5),              # Float range
    learning_rate=(1e-5, 1e-3),
    batch_size=[32, 64, 128, 256],
)

# Configure tuning
config = TuningConfig(
    n_trials=50,
    timeout=3600,                    # 1 hour max
    metric="mse",                    # Optimize MSE
    direction="minimize",
    sampler="tpe",                   # TPE, Random, or CMA-ES
    pruner="median",                 # Median or Hyperband
    study_name="demand_forecast_study",
    storage="sqlite:///optuna.db",   # For persistence
)

# Run tuning
tuner = HyperparameterTuner(config, search_space)
best_params = tuner.tune(train_dataloader, val_dataloader)

# Access study results
print(f"Best params: {best_params}")
print(f"Best value: {tuner.study.best_value}")
```

---

### `demand_forecast.core.tuning.quick_tune`

Convenience function for quick hyperparameter tuning.

```python
from demand_forecast.core.tuning import quick_tune

best_params = quick_tune(
    train_dataloader=train_dl,
    val_dataloader=val_dl,
    n_trials=20,
    timeout=1800,
    metric="mae",
)

print(f"Best learning rate: {best_params['learning_rate']}")
print(f"Best d_model: {best_params['d_model']}")
```

---

### `demand_forecast.core.tuning.TuningConfig`

Configuration dataclass for tuning parameters.

```python
from demand_forecast.core.tuning import TuningConfig

config = TuningConfig(
    n_trials=100,                    # Number of trials
    timeout=7200,                    # Max time in seconds
    metric="mse",                    # Metric to optimize
    direction="minimize",            # "minimize" or "maximize"
    sampler="tpe",                   # "tpe", "random", or "cmaes"
    pruner="hyperband",              # "median", "hyperband", or None
    study_name="my_study",
    storage=None,                    # SQLite URL for persistence
)
```

---

### `demand_forecast.core.tuning.SearchSpace`

Flexible search space definition.

```python
from demand_forecast.core.tuning import SearchSpace

# Define search space with different parameter types
search_space = SearchSpace(
    # Categorical (list of choices)
    d_model=[64, 128, 256, 512],
    nhead=[4, 8, 16],

    # Integer range (tuple)
    num_encoder_layers=(1, 8),
    num_decoder_layers=(1, 8),

    # Float range (tuple)
    dropout=(0.0, 0.5),
    learning_rate=(1e-6, 1e-3),
    weight_decay=(1e-6, 1e-2),

    # Boolean
    use_rope=[True, False],
    use_pre_layernorm=[True, False],
)

# Access parameters
print(search_space.get_param_names())
```

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
