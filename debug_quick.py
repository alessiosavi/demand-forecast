#!/usr/bin/env python3
"""
Quick Debug Script for Demand Forecast
=======================================

A minimal script to quickly test that everything works.
Run with: python debug_quick.py

This script:
1. Generates a small synthetic dataset
2. Trains a minimal model for 2 epochs
3. Generates predictions
4. Prints metrics

Expected runtime: ~1-2 minutes
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Imports
import numpy as np  # noqa: E402
import torch  # noqa: E402

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path(__file__).parent if "__file__" in dir() else Path.cwd()


def main():
    print("=" * 60)
    print("DEMAND FORECAST - QUICK DEBUG")
    print("=" * 60)

    # Device - use CPU for compatibility (MPS has embedding issues)
    # Set FORCE_CUDA=1 or FORCE_MPS=1 to override
    if os.environ.get("FORCE_CUDA") and torch.cuda.is_available():
        device = torch.device("cuda")
    elif os.environ.get("FORCE_MPS") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Step 1: Generate Data
    print("\n[1/5] Generating synthetic data...")
    from demand_forecast.synthetic import generate_sales_data

    df = generate_sales_data(
        num_days=200,
        num_products=10,
        num_stores=3,
        seed=SEED,
    )
    data_path = PROJECT_ROOT / "quick_debug_data.csv"
    df.to_csv(data_path, index=False)
    print(f"  Generated {len(df):,} rows")

    # Step 2: Create Config
    print("\n[2/5] Creating configuration...")
    from demand_forecast.config.settings import (
        DataConfig,
        ModelConfig,
        OutputConfig,
        Settings,
        TimeSeriesConfig,
        TrainingConfig,
    )

    settings = Settings(
        data=DataConfig(
            input_path=data_path,
            resample_period="1W",
            max_zeros_ratio=0.9,
        ),
        timeseries=TimeSeriesConfig(
            window=16,
            n_out=4,
            test_size=0.2,
        ),
        model=ModelConfig(
            model_type="standard",
            d_model=32,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.1,
            use_rope=False,
            use_pre_layernorm=False,
            use_film_conditioning=False,
            stochastic_depth_rate=0.0,
        ),
        training=TrainingConfig(
            num_epochs=2,
            batch_size=16,
            learning_rate=1e-3,
            num_workers=0,
        ),
        output=OutputConfig(
            model_dir=PROJECT_ROOT / "quick_debug_models",
        ),
        seed=SEED,
        device=str(device),
    )
    print(f"  Model: {settings.model.model_type}, d_model={settings.model.d_model}")

    # Step 3: Initialize Pipeline
    print("\n[3/5] Initializing pipeline...")
    from demand_forecast.core.pipeline import ForecastPipeline

    pipeline = ForecastPipeline(settings)
    pipeline.load_and_preprocess()
    pipeline.create_datasets()
    model = pipeline.build_model()

    params = sum(p.numel() for p in model.parameters())
    print(f"  Clusters: {model.num_models}")
    print(f"  Parameters: {params:,}")

    # Step 4: Train
    print("\n[4/5] Training model...")
    pipeline.train(plot=False)
    print("  Training complete!")

    # Step 5: Evaluate & Predict
    print("\n[5/5] Evaluating and predicting...")
    metrics = pipeline.evaluate(plot=False)
    predictions = pipeline.predict(confidence=0.95, plot=False)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  MSE:         {metrics.get('mse', 0):.4f}")
    print(f"  RMSE:        {metrics.get('rmse', 0):.4f}")
    print(f"  MAE:         {metrics.get('mae', 0):.4f}")
    print(f"  R-squared:   {metrics.get('r_squared', 0):.4f}")
    print(f"  Predictions: {len(predictions)}")
    print("=" * 60)

    # Cleanup
    print("\nCleaning up...")
    data_path.unlink(missing_ok=True)
    import shutil

    if settings.output.model_dir.exists():
        shutil.rmtree(settings.output.model_dir)

    print("\nSUCCESS! All components working correctly.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
