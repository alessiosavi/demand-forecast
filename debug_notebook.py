# [ ] Markdown
# # Demand Forecast - Debug & Development Notebook
#
# This notebook provides a starting point for debugging and testing the demand forecast system.
# It covers:
# 1. **Data Generation** - Create synthetic sales data
# 2. **Data Loading & Preprocessing** - Load and prepare data for training
# 3. **Model Architecture** - Explore different model types
# 4. **Training** - Train models with visualization
# 5. **Hyperparameter Tuning** - Optuna-based optimization
# 6. **Inference** - Generate predictions with confidence intervals
# 7. **Evaluation** - Compute metrics and visualize results

# [ ] cell
# === Setup & Imports ===
import os
import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path if needed
PROJECT_ROOT = Path(__file__).parent if "__file__" in dir() else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set environment variables for reproducibility and performance
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

print(f"Project root: {PROJECT_ROOT}")

# [ ] cell
# === Core Imports ===
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device configuration - default to CPU for compatibility
# Set FORCE_CUDA=1 or FORCE_MPS=1 environment variable to use GPU
if os.environ.get("FORCE_CUDA") and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif os.environ.get("FORCE_MPS") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")

# [ ] Markdown
# ## 1. Data Generation
#
# Generate synthetic sales data for testing. Adjust parameters as needed.

# [ ] cell
# === Generate Synthetic Data ===
from demand_forecast.synthetic import generate_sales_data  # noqa: E402

# Parameters for data generation
NUM_PRODUCTS = 30
NUM_STORES = 5
NUM_DAYS = 400  # ~1.1 years of data
START_DATE = "2022-01-01"

# Generate data
print("Generating synthetic sales data...")
df_raw = generate_sales_data(
    num_days=NUM_DAYS,
    num_products=NUM_PRODUCTS,
    num_stores=NUM_STORES,
    start_date=START_DATE,
    seed=SEED,
)

print(f"Generated {len(df_raw):,} rows")
print(f"Date range: {df_raw['date'].min()} to {df_raw['date'].max()}")
print(f"Unique SKUs: {df_raw['product_id'].nunique() * df_raw['store_id'].nunique()}")
print(f"\nColumns: {list(df_raw.columns)}")
df_raw.head()

# [ ] cell
# === Explore Generated Data ===
print("Data statistics:")
print(df_raw.describe())

print("\nCategorical columns:")
for col in ["color", "size", "category", "subcategory"]:
    if col in df_raw.columns:
        print(f"  {col}: {df_raw[col].nunique()} unique values - {df_raw[col].unique()[:5]}")

# [ ] cell
# === Visualize Sample Time Series ===
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Get a few sample SKUs
sample_skus = df_raw.groupby(["product_id", "store_id"]).size().head(4).index.tolist()

for idx, (product_id, store_id) in enumerate(sample_skus):
    ax = axes[idx // 2, idx % 2]
    mask = (df_raw["product_id"] == product_id) & (df_raw["store_id"] == store_id)
    sku_data = df_raw[mask].set_index("date")["sales_qty"]
    ax.plot(sku_data.index, sku_data.values, alpha=0.8)
    ax.set_title(f"Product {product_id} - Store {store_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales Quantity")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# [ ] Markdown
# ## 2. Configuration
#
# Set up the configuration for training. You can modify these values to experiment.

# [ ] cell
# === Create Configuration ===
from demand_forecast.config.settings import (  # noqa: E402
    DataConfig,
    ModelConfig,
    OutputConfig,
    Settings,
    TimeSeriesConfig,
    TrainingConfig,
    TuningConfig,
)

# Save generated data to a temp file
DATA_PATH = PROJECT_ROOT / "debug_data.csv"
df_raw.to_csv(DATA_PATH, index=False)
print(f"Saved data to: {DATA_PATH}")

# Create configuration
# Note: The data loader renames columns from the source names to the target names:
#   product_id_column -> sku_column (e.g., "product_id" -> "sku")
#   sales_qty_column -> quantity_column (e.g., "sales_qty" -> "qty")
settings = Settings(
    data=DataConfig(
        input_path=DATA_PATH,
        date_column="date",
        sku_column="sku",  # Target column name after renaming
        quantity_column="qty",  # Target column name after renaming
        store_column="store_id",
        product_id_column="product_id",  # Source column name in raw data
        sales_qty_column="sales_qty",  # Source column name in raw data
        resample_period="1W",  # Weekly aggregation
        max_zeros_ratio=0.8,  # Allow SKUs with up to 80% zeros
        categorical_columns=["color", "size", "category"],
        onehot_columns=["is_promo_day", "store_id"],
    ),
    timeseries=TimeSeriesConfig(
        window=26,  # 26 weeks lookback (~6 months)
        n_out=8,  # 8 weeks forecast horizon (~2 months)
        test_size=0.2,
    ),
    model=ModelConfig(
        model_type="standard",  # "standard", "advanced", or "lightweight"
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        # Optional improvements (for standard model)
        use_rope=False,
        use_pre_layernorm=False,
        use_film_conditioning=False,
        stochastic_depth_rate=0.0,
    ),
    training=TrainingConfig(
        num_epochs=5,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-4,
        early_stop_patience=3,
        num_workers=0,  # Use 0 to avoid multiprocessing issues
    ),
    output=OutputConfig(
        model_dir=PROJECT_ROOT / "debug_models",
        cache_dir=PROJECT_ROOT / "debug_cache",
    ),
    tuning=TuningConfig(
        enabled=False,
        n_trials=10,
        timeout=600,
        metric="mse",
    ),
    seed=SEED,
    device=str(DEVICE),
    log_level="INFO",
)

print("Configuration created:")
print(f"  Model type: {settings.model.model_type}")
print(f"  Window: {settings.timeseries.window} weeks")
print(f"  Forecast horizon: {settings.timeseries.n_out} weeks")
print(f"  Batch size: {settings.training.batch_size}")
print(f"  Learning rate: {settings.training.learning_rate}")

# [ ] Markdown
# ## 3. Pipeline Initialization
#
# Create the forecast pipeline and load/preprocess data.

# [ ] cell
# === Initialize Pipeline ===
from demand_forecast.core.pipeline import ForecastPipeline  # noqa: E402

pipeline = ForecastPipeline(settings)
print(f"Pipeline initialized with device: {pipeline.device}")

# [ ] cell
# === Load and Preprocess Data ===
print("Loading and preprocessing data...")
df_processed = pipeline.load_and_preprocess()

print(f"\nProcessed data shape: {df_processed.shape}")
print(f"Columns: {list(df_processed.columns)}")
print("\nSample of processed data:")
df_processed.head()

# [ ] cell
# === Create Datasets ===
print("Creating time series datasets...")
datasets = pipeline.create_datasets()

print(f"\nNumber of clusters: {len(datasets)}")
for cluster_id, data in datasets.items():
    x_train, y_train, cat_train, x_test, y_test, cat_test = data
    print(f"  Cluster {cluster_id}:")
    print(f"    Train: X={x_train.shape}, y={y_train.shape}, cats={cat_train.shape}")
    print(f"    Test:  X={x_test.shape}, y={y_test.shape}, cats={cat_test.shape}")

# [ ] Markdown
# ## 4. Model Architecture
#
# Explore and build the model architecture.

# [ ] cell
# === Build Model ===
print("Building model...")
model = pipeline.build_model()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel type: {settings.model.model_type}")
print(f"Number of cluster models: {model.num_models}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# [ ] cell
# === Inspect Model Architecture ===
# Print architecture of first cluster model
print("Model architecture (Cluster 0):")
print(model.get_model("0"))

# [ ] cell
# === Test Forward Pass ===
print("Testing forward pass...")

# Get a sample batch from the first cluster
from demand_forecast.data import create_dataloaders  # noqa: E402

train_dls, test_dls, _, _ = create_dataloaders(
    raw_datasets=pipeline._raw_datasets,
    encoded_categorical_features=pipeline.categorical_encoder.get_encoded_columns(),
    batch_size=settings.training.batch_size,
    num_workers=0,
    seed=SEED,
    device=pipeline.device,
)

# Get first batch
first_cluster = list(train_dls.keys())[0]
batch = next(iter(train_dls[first_cluster]))

print(f"Batch keys: {batch.keys()}")
print(f"  qty shape: {batch['qty'].shape}")
print(f"  past_time shape: {batch['past_time'].shape}")
print(f"  future_time shape: {batch['future_time'].shape}")
print(f"  sku shape: {batch['sku'].shape}")
print(f"  cats keys: {list(batch['cats'].keys())}")

# Forward pass
model.eval()
with torch.no_grad():
    output = model(
        first_cluster,
        batch["qty"].to(pipeline.device),
        batch["past_time"].to(pipeline.device),
        batch["future_time"].to(pipeline.device),
        batch["sku"].to(pipeline.device),
        {k: v.to(pipeline.device) for k, v in batch["cats"].items()},
    )

print(f"\nOutput shape: {output.shape}")
print(f"Output sample: {output[0, :5]}")

# [ ] Markdown
# ## 5. Training
#
# Train the model with optional visualization.

# [ ] cell
# === Train Model ===
print("Starting training...")
print(f"  Epochs: {settings.training.num_epochs}")
print(f"  Batch size: {settings.training.batch_size}")
print(f"  Learning rate: {settings.training.learning_rate}")
print()

# Create plot directory
plot_dir = PROJECT_ROOT / "debug_plots" / "training"
plot_dir.mkdir(parents=True, exist_ok=True)

# Train with plots
pipeline.train(plot=True, plot_dir=plot_dir)

print(f"\nTraining complete! Plots saved to: {plot_dir}")

# [ ] cell
# === Save Model ===
model_path = settings.output.model_dir / "debug_model.pt"
settings.output.model_dir.mkdir(parents=True, exist_ok=True)

pipeline.save(model_path)
print(f"Model saved to: {model_path}")

# [ ] Markdown
# ## 6. Evaluation
#
# Evaluate the trained model on test data.

# [ ] cell
# === Evaluate Model ===
print("Evaluating model...")

eval_plot_dir = PROJECT_ROOT / "debug_plots" / "evaluation"
eval_plot_dir.mkdir(parents=True, exist_ok=True)

metrics = pipeline.evaluate(plot=True, plot_dir=eval_plot_dir)

print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
for metric_name, value in metrics.items():
    if isinstance(value, float):
        print(f"  {metric_name:15s}: {value:.4f}")
    else:
        print(f"  {metric_name:15s}: {value}")
print("=" * 50)

# [ ] Markdown
# ## 7. Inference
#
# Generate predictions with confidence intervals.

# [ ] cell
# === Generate Predictions ===
print("Generating predictions...")

pred_plot_dir = PROJECT_ROOT / "debug_plots" / "predictions"
pred_plot_dir.mkdir(parents=True, exist_ok=True)

predictions_df = pipeline.predict(
    confidence=0.95,
    plot=True,
    plot_dir=pred_plot_dir,
)

print(f"\nPredictions shape: {predictions_df.shape}")
print(f"Columns: {list(predictions_df.columns)}")
print("\nSample predictions:")
predictions_df.head(10)

# [ ] cell
# === Analyze Predictions ===
print("Prediction statistics:")
print(predictions_df.describe())

# Plot prediction distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(predictions_df["prediction"], bins=30, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Predicted Value")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Predictions")

if "actual" in predictions_df.columns:
    axes[1].scatter(predictions_df["actual"], predictions_df["prediction"], alpha=0.5)
    axes[1].plot(
        [predictions_df["actual"].min(), predictions_df["actual"].max()],
        [predictions_df["actual"].min(), predictions_df["actual"].max()],
        "r--",
        label="Perfect prediction",
    )
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("Actual vs Predicted")
    axes[1].legend()

plt.tight_layout()
plt.show()

# [ ] Markdown
# ## 8. Hyperparameter Tuning (Optional)
#
# Use Optuna for hyperparameter optimization. This can take a while!

# [ ] cell
# === Hyperparameter Tuning ===
# Set RUN_TUNING = True to enable tuning
RUN_TUNING = False

if RUN_TUNING:
    from demand_forecast.core.tuning import (
        TuningConfig,
        quick_tune,
    )

    print("Starting hyperparameter tuning...")
    print("This may take several minutes...")

    # Quick tune with a small number of trials
    best_params = quick_tune(
        train_dataloader=train_dls[first_cluster],
        val_dataloader=test_dls[first_cluster],
        n_trials=5,
        timeout=300,  # 5 minute timeout
        metric="mae",
    )

    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
else:
    print("Tuning skipped. Set RUN_TUNING = True to enable.")

# [ ] Markdown
# ## 9. Advanced: Custom Model Types
#
# Explore different model architectures.

# [ ] cell
# === Test Different Model Types ===
from demand_forecast.models import MODEL_REGISTRY, create_model  # noqa: E402

print("Available model types:")
for name, cls in MODEL_REGISTRY.items():
    print(f"  {name}: {cls.__name__}")

# [ ] cell
# === Create Lightweight Model ===
# Create a lightweight model for comparison
print("\nCreating lightweight model...")

lightweight_settings = settings.model_copy(deep=True)
lightweight_settings.model.model_type = "lightweight"
lightweight_settings.model.tcn_channels = [32, 64]
lightweight_settings.model.tcn_kernel_size = 3

# Get dimensions from first cluster
sample_x, sample_y, sample_cats, _, _, _ = list(datasets.values())[0]
n_skus = len(pipeline.sku_to_index)
cat_dims = {col: df_processed[col].nunique() for col in settings.data.categorical_columns}

lightweight_model = create_model(
    model_type="lightweight",
    n_clusters=len(datasets),
    sku_vocab_size=n_skus + 1,
    sku_emb_dim=16,
    cat_features_dim=cat_dims,
    cat_emb_dims=16,
    past_time_features_dim=sample_x.shape[-1] - 1,  # Exclude qty
    future_time_features_dim=4,
    n_out=settings.timeseries.n_out,
    tcn_channels=[32, 64],
    tcn_kernel_size=3,
)

lightweight_params = sum(p.numel() for p in lightweight_model.parameters())
print(f"Lightweight model parameters: {lightweight_params:,}")
print(f"Comparison: Standard model has {total_params:,} parameters")
print(f"Reduction: {(1 - lightweight_params / total_params) * 100:.1f}%")

# [ ] Markdown
# ## 10. Visualization Utilities
#
# Additional visualization functions for analysis.

# [ ] cell
# === Visualization Utilities ===
from demand_forecast.utils.visualization import (  # noqa: E402
    plot_prediction_quality,
)

# If we have actuals, create quality plots
if "actual" in predictions_df.columns:
    actuals = predictions_df["actual"].values
    preds = predictions_df["prediction"].values
    skus = predictions_df["sku"].values if "sku" in predictions_df.columns else None

    # Create quality analysis plot
    fig = plot_prediction_quality(
        actuals=actuals,
        predictions=preds,
        skus=skus,
        title="Prediction Quality Analysis",
        save_path=pred_plot_dir / "quality_analysis.png",
        show=True,
    )

# [ ] Markdown
# ## 11. Cleanup
#
# Clean up temporary files and resources.

# [ ] cell
# === Cleanup ===
import gc  # noqa: E402

# Collect garbage
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Cleanup complete!")

# Optionally remove debug files
CLEANUP_FILES = False

if CLEANUP_FILES:
    import shutil

    # Remove generated files
    if DATA_PATH.exists():
        DATA_PATH.unlink()
        print(f"Removed: {DATA_PATH}")

    # Remove model directory
    if settings.output.model_dir.exists():
        shutil.rmtree(settings.output.model_dir)
        print(f"Removed: {settings.output.model_dir}")

    # Remove cache directory
    if settings.output.cache_dir.exists():
        shutil.rmtree(settings.output.cache_dir)
        print(f"Removed: {settings.output.cache_dir}")

    # Remove plot directory
    plot_root = PROJECT_ROOT / "debug_plots"
    if plot_root.exists():
        shutil.rmtree(plot_root)
        print(f"Removed: {plot_root}")
else:
    print("Debug files preserved. Set CLEANUP_FILES = True to remove.")

# [ ] Markdown
# ## Summary
#
# This notebook covered:
#
# 1. **Data Generation** - Created synthetic sales data with realistic patterns
# 2. **Configuration** - Set up model and training parameters
# 3. **Pipeline** - Loaded, preprocessed, and created time series datasets
# 4. **Model** - Built and inspected the transformer architecture
# 5. **Training** - Trained the model with visualization
# 6. **Evaluation** - Computed metrics (MSE, RMSE, MAE, MAPE, R², correlation)
# 7. **Inference** - Generated predictions with confidence intervals
# 8. **Tuning** - Demonstrated Optuna-based hyperparameter optimization
# 9. **Advanced** - Explored different model architectures
#
# ### Next Steps
#
# - Experiment with different `model_type` values: "standard", "advanced", "lightweight"
# - Enable optional improvements: `use_rope`, `use_pre_layernorm`, `use_film_conditioning`
# - Run hyperparameter tuning with more trials
# - Try different data configurations (window size, forecast horizon)
# - Export lightweight models to ONNX for deployment

# [ ] cell
# === Final Summary ===
print("\n" + "=" * 60)
print("DEBUG SESSION SUMMARY")
print("=" * 60)
print(f"Data: {NUM_PRODUCTS} products × {NUM_STORES} stores × {NUM_DAYS} days")
print(f"Model: {settings.model.model_type} with {total_params:,} parameters")
print(f"Training: {settings.training.num_epochs} epochs, batch_size={settings.training.batch_size}")
if metrics:
    print(f"Final MAE: {metrics.get('mae', 'N/A'):.4f}")
    print(f"Final R²: {metrics.get('r_squared', 'N/A'):.4f}")
print(f"Predictions: {len(predictions_df)} samples")
print("=" * 60)
