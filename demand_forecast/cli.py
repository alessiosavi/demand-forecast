"""Command-line interface for demand forecasting."""

import logging
from pathlib import Path

import typer

from demand_forecast import configure_logging

app = typer.Typer(
    name="demand-forecast",
    help="Demand Forecasting with Transformer Neural Networks",
    add_completion=False,
)


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML configuration file"),
    data_path: Path | None = typer.Option(
        None, "--data", "-d", help="Override input data path from config"
    ),
    epochs: int | None = typer.Option(
        None, "--epochs", "-e", help="Override number of epochs from config"
    ),
    device: str | None = typer.Option(None, "--device", help="Device to use (cpu, cuda, mps)"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Model output directory"),
    plot: bool = typer.Option(
        False, "--plot", "-p", help="Generate validation plots during training"
    ),
    plot_dir: Path | None = typer.Option(
        None, "--plot-dir", help="Directory to save training plots"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Train the demand forecasting model."""
    configure_logging("DEBUG" if verbose else "INFO")
    logger = logging.getLogger(__name__)

    from demand_forecast.config.settings import Settings
    from demand_forecast.core.pipeline import ForecastPipeline

    logger.info(f"Loading configuration from {config}")
    settings = Settings.from_yaml(config)

    # Apply overrides
    if data_path:
        settings.data.input_path = data_path
    if epochs:
        settings.training.num_epochs = epochs
    if device:
        settings.device = device
    if output_dir:
        settings.output.model_dir = output_dir
    settings.seed = seed

    # Run pipeline
    pipeline = ForecastPipeline(settings)

    logger.info("Loading and preprocessing data...")
    pipeline.load_and_preprocess()

    logger.info("Creating datasets...")
    pipeline.create_datasets()

    logger.info("Building model...")
    pipeline.build_model()

    logger.info("Starting training...")
    pipeline.train(plot=plot, plot_dir=plot_dir)

    # Save model
    output_path = settings.output.model_dir / "model.pt"
    settings.output.model_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save(output_path)

    logger.info(f"Training complete. Model saved to {output_path}")

    if plot:
        actual_plot_dir = plot_dir or settings.output.model_dir / "training_plots"
        logger.info(f"Training plots saved to {actual_plot_dir}")


@app.command()
def evaluate(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    data_path: Path = typer.Argument(..., help="Path to evaluation data"),
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML configuration file"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output metrics file (JSON)"),
    plot: bool = typer.Option(False, "--plot", "-p", help="Generate evaluation plots"),
    plot_dir: Path | None = typer.Option(
        None, "--plot-dir", help="Directory to save evaluation plots"
    ),
    show_plots: bool = typer.Option(
        False, "--show", help="Display plots interactively (requires display)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Evaluate a trained model on test data."""
    configure_logging("DEBUG" if verbose else "INFO")
    logger = logging.getLogger(__name__)

    import json

    from demand_forecast.config.settings import Settings
    from demand_forecast.core.pipeline import ForecastPipeline

    logger.info(f"Loading configuration from {config}")
    settings = Settings.from_yaml(config)
    settings.data.input_path = data_path

    # Use the pipeline to handle preprocessing and evaluation
    logger.info("Initializing pipeline...")
    pipeline = ForecastPipeline(settings)

    # Load model and artifacts
    logger.info(f"Loading model from {model_path}")
    pipeline.load(model_path)

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    pipeline.load_and_preprocess()

    # Create datasets for evaluation
    logger.info("Creating datasets...")
    pipeline.create_datasets()

    # Run evaluation
    logger.info("Running evaluation...")
    metrics = pipeline.evaluate(
        plot=plot,
        plot_dir=plot_dir,
        show_plots=show_plots,
    )

    # Print metrics summary
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"  MSE:         {metrics['mse']:.4f}")
    logger.info(f"  RMSE:        {metrics['rmse']:.4f}")
    logger.info(f"  MAE:         {metrics['mae']:.4f}")
    logger.info(f"  MAPE:        {metrics['mape']:.2f}%")
    logger.info(f"  R-squared:   {metrics['r_squared']:.4f}")
    logger.info(f"  Correlation: {metrics['correlation']:.4f}")
    logger.info(f"  Samples:     {metrics['total_samples']}")
    logger.info("=" * 50)

    # Save metrics to file
    if output:
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {output}")

    if plot:
        actual_plot_dir = plot_dir or settings.output.model_dir / "evaluation"
        logger.info(f"Evaluation plots saved to {actual_plot_dir}")


@app.command()
def predict(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    data_path: Path = typer.Argument(..., help="Path to input data"),
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML configuration file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output predictions file (CSV)"),
    confidence: float = typer.Option(0.95, "--confidence", help="Confidence interval level"),
    plot: bool = typer.Option(False, "--plot", "-p", help="Generate prediction quality plots"),
    plot_dir: Path | None = typer.Option(
        None, "--plot-dir", help="Directory to save plots (default: model_dir/plots)"
    ),
    show_plots: bool = typer.Option(
        False, "--show", help="Display plots interactively (requires display)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Generate predictions with confidence intervals."""
    configure_logging("DEBUG" if verbose else "INFO")
    logger = logging.getLogger(__name__)

    from demand_forecast.config.settings import Settings
    from demand_forecast.core.pipeline import ForecastPipeline

    logger.info(f"Loading configuration from {config}")
    settings = Settings.from_yaml(config)
    settings.data.input_path = data_path

    # Use the pipeline to handle preprocessing and prediction
    logger.info("Initializing pipeline...")
    pipeline = ForecastPipeline(settings)

    # Load model and artifacts
    logger.info(f"Loading model from {model_path}")
    pipeline.load(model_path)

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    pipeline.load_and_preprocess()

    # Create datasets for prediction
    logger.info("Creating datasets...")
    pipeline.create_datasets()

    # Run predictions
    logger.info("Generating predictions...")
    predictions_df = pipeline.predict(
        confidence=confidence,
        plot=plot,
        plot_dir=plot_dir,
        show_plots=show_plots,
    )

    # Save predictions
    predictions_df.to_csv(output, index=False)
    logger.info(f"Predictions saved to {output} ({len(predictions_df)} rows)")

    if plot:
        actual_plot_dir = plot_dir or settings.output.model_dir / "plots"
        logger.info(f"Plots saved to {actual_plot_dir}")


@app.command("generate-data")
def generate_data(
    output: Path = typer.Argument(..., help="Output CSV path"),
    num_days: int = typer.Option(365 * 4, "--days", help="Number of days to generate"),
    num_products: int = typer.Option(100, "--products", help="Number of products"),
    num_stores: int = typer.Option(20, "--stores", help="Number of stores"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Generate synthetic sales data for testing."""
    configure_logging("DEBUG" if verbose else "INFO")
    logger = logging.getLogger(__name__)

    from demand_forecast.synthetic.generator import generate_sales_data

    logger.info(
        f"Generating {num_days} days of data for {num_products} products across {num_stores} stores"
    )

    df = generate_sales_data(
        num_days=num_days,
        num_products=num_products,
        num_stores=num_stores,
        seed=seed,
    )

    df.to_csv(output, index=False)
    logger.info(f"Generated {len(df)} rows, saved to {output}")


@app.command()
def preprocess(
    input_path: Path = typer.Argument(..., help="Input data path"),
    output_path: Path = typer.Argument(..., help="Output parquet path"),
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML configuration file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Preprocess raw data and save as Parquet."""
    configure_logging("DEBUG" if verbose else "INFO")
    logger = logging.getLogger(__name__)

    from demand_forecast.config.settings import Settings
    from demand_forecast.data.loader import load_sales_data
    from demand_forecast.data.preprocessor import filter_skus, resample_series

    logger.info(f"Loading configuration from {config}")
    settings = Settings.from_yaml(config)
    settings.data.input_path = input_path

    cfg = settings.data
    ts_cfg = settings.timeseries

    # Load data
    df = load_sales_data(
        path=input_path,
        date_column=cfg.date_column,
        sku_column=cfg.sku_column,
        quantity_column=cfg.quantity_column,
        store_column=cfg.store_column,
    )

    # Resample
    series = resample_series(
        df,
        resample_period=cfg.resample_period,
        sku_column=cfg.sku_column,
        store_column=cfg.store_column,
        quantity_column=cfg.quantity_column,
    )

    # Filter
    series = filter_skus(
        series,
        window=ts_cfg.window,
        n_out=ts_cfg.n_out,
        max_zeros_ratio=cfg.max_zeros_ratio,
    )

    # Save
    series.to_parquet(output_path, compression="brotli")
    logger.info(f"Preprocessed data saved to {output_path}")


@app.command()
def version():
    """Show version information."""
    from demand_forecast import __version__

    typer.echo(f"demand-forecast version {__version__}")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
