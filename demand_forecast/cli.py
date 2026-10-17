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
def forecast(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    products: Path = typer.Option(..., "--products", exists=True, dir_okay=False),
    sales: Path = typer.Option(..., "--sales", exists=True, dir_okay=False),
    out_dir: Path = typer.Option(..., "--out-dir"),
    holdout_from: str | None = typer.Option(None, "--holdout-from"),
    trials: int | None = typer.Option(
        None, "--trials", min=0, help="Override YAML tuning.n_trials (hackathon.yaml: 12)"
    ),
    timeout: int | None = typer.Option(
        None, "--timeout", min=1, help="Override YAML tuning.timeout (hackathon.yaml: 900 seconds)"
    ),
    max_epochs: int | None = typer.Option(
        None, "--max-epochs", min=1, help="Override YAML training.num_epochs (hackathon.yaml: 60)"
    ),
    budget_seconds: float | None = typer.Option(
        None, "--budget-seconds", min=0.001, help="Override YAML budget_seconds (default: 2700)"
    ),
    k: int | None = typer.Option(
        None, "--k", min=1, help="Override YAML data.k (hackathon.yaml: 1)"
    ),
    device: str | None = typer.Option(None, "--device", help="cpu or cuda (also ROCm)"),
    max_products: int | None = typer.Option(
        None, "--max-products", min=1, help="Deterministic debug catalogue subset"
    ),
    seed: int | None = typer.Option(
        None, "--seed", min=0, help="Override YAML seed (hackathon.yaml: 42)"
    ),
    expect_origins: str | None = typer.Option(
        None,
        "--expect-origins",
        metavar="HOLDOUT,FUTURE",
        help="Require these holdout and future origins (comma-separated ISO dates)",
    ),
    preprocess_only: bool = typer.Option(
        False, "--preprocess-only", help="Time both preprocessing stages without training"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Tune, select, refit and export honest weekly store forecasts."""
    import time
    from datetime import date

    started = time.monotonic()
    configure_logging("DEBUG" if verbose else "INFO")
    from demand_forecast.config.settings import Settings
    from demand_forecast.core.forecast_run import run_forecast

    try:
        settings = Settings.from_yaml(config)
        if device not in (None, "cpu", "cuda"):
            raise ValueError("--device must be cpu or cuda")
        settings.data.products_path = products
        settings.data.input_path = sales
        if k is not None:
            settings.data.k = k
        settings.device = device or settings.device
        if seed is not None:
            settings.seed = seed
        if holdout_from:
            settings.timeseries.holdout_from = date.fromisoformat(holdout_from)
        if trials is not None:
            settings.tuning.n_trials = trials
        if timeout is not None:
            settings.tuning.timeout = timeout
        if max_epochs is not None:
            settings.training.num_epochs = max_epochs
        if budget_seconds is not None:
            settings.budget_seconds = budget_seconds
        expected = None
        if expect_origins is not None:
            try:
                holdout, future = expect_origins.split(",")
                expected = (date.fromisoformat(holdout), date.fromisoformat(future))
            except ValueError as exc:
                raise ValueError("--expect-origins requires HOLDOUT,FUTURE as ISO dates") from exc
        run_forecast(
            settings,
            out_dir,
            max_products=max_products,
            expect_origins=expected,
            preprocess_only=preprocess_only,
            started=started,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def benchmark(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    products: Path = typer.Option(..., "--products", exists=True, dir_okay=False),
    sales: Path = typer.Option(..., "--sales", exists=True, dir_okay=False),
    out_dir: Path = typer.Option(..., "--out-dir"),
    models: str = typer.Option("standard,advanced,lightweight_tcn,lightweight_mixer", "--models"),
    device: str | None = typer.Option(None, "--device", help="cpu or cuda"),
    trials: int | None = typer.Option(None, "--trials", min=0),
    timeout: int | None = typer.Option(None, "--timeout", min=1),
    max_epochs: int | None = typer.Option(
        None, "--max-epochs", min=1, help="Cap selection and trial epochs"
    ),
    budget_seconds: float | None = typer.Option(
        None, "--budget-seconds", min=0.001, help="Full budget for EACH model"
    ),
    max_products: int | None = typer.Option(None, "--max-products", min=1),
    seed: int | None = typer.Option(None, "--seed", min=0),
    holdout_from: str | None = typer.Option(None, "--holdout-from"),
    k: int | None = typer.Option(None, "--k", min=1),
    expect_origins: str | None = typer.Option(None, "--expect-origins", metavar="HOLDOUT,FUTURE"),
    compare_only: bool = typer.Option(
        False,
        "--compare-only",
        help="Rebuild reports from existing CSVs without training; repeat original options",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Compare four neural families and two baselines on the same data with equal caps."""
    from datetime import date

    from demand_forecast.config.settings import Settings
    from demand_forecast.core.benchmark import run_benchmark

    configure_logging("DEBUG" if verbose else "INFO")
    try:
        settings = Settings.from_yaml(config)
        if device not in (None, "cpu", "cuda"):
            raise ValueError("--device must be cpu or cuda")
        settings.data.products_path = products
        settings.data.input_path = sales
        settings.device = device or settings.device
        if seed is not None:
            settings.seed = seed
        if k is not None:
            settings.data.k = k
        if holdout_from:
            settings.timeseries.holdout_from = date.fromisoformat(holdout_from)
        if trials is not None:
            settings.tuning.n_trials = trials
        if timeout is not None:
            settings.tuning.timeout = timeout
        if max_epochs is not None:
            settings.training.num_epochs = max_epochs
            settings.tuning.epochs_per_trial = min(settings.tuning.epochs_per_trial, max_epochs)
        if budget_seconds is not None:
            settings.budget_seconds = budget_seconds
        expected = None
        if expect_origins is not None:
            try:
                holdout, future = expect_origins.split(",")
                expected = (date.fromisoformat(holdout), date.fromisoformat(future))
            except ValueError as exc:
                raise ValueError("--expect-origins requires HOLDOUT,FUTURE as ISO dates") from exc
        report = run_benchmark(
            settings,
            out_dir,
            models=tuple(name.strip() for name in models.split(",")),
            max_products=max_products,
            expect_origins=expected,
            compare_only=compare_only,
        )
        typer.echo(f"Comparison saved to {out_dir / 'benchmark.md'}")
        if any(r["status"] != "ok" for r in report["results"].values()):
            typer.echo("Some models failed or are incomplete; see benchmark.json.", err=True)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc


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
    """Generate weekly point forecasts in original quantity units."""
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


@app.command("generate-sales")
def generate_sales(
    products: Path = typer.Option(
        ..., "--products", exists=True, dir_okay=False, help="Product CSV or CSV.gz"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output sales CSV"),
    start: str = typer.Option("2024-10-17", "--start", help="First sales date (YYYY-MM-DD)"),
    end: str = typer.Option("2026-10-16", "--end", help="Last sales date (YYYY-MM-DD)"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    promo_share: float = typer.Option(
        0.10, "--promo-share", min=0.0, max=1.0, help="Share of calendar days on promotion"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Generate daily synthetic sales using real catalogue IDs and prices."""
    configure_logging("DEBUG" if verbose else "INFO")
    logger = logging.getLogger(__name__)

    import pandas as pd

    from demand_forecast.synthetic.generator import PRODUCT_COLUMNS, generate_product_sales

    try:
        catalogue = pd.read_csv(products, usecols=PRODUCT_COLUMNS, dtype={"id": str})
        sales = generate_product_sales(catalogue, start, end, seed, promo_share)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    sales.to_csv(output, index=False, date_format="%Y-%m-%d", float_format="%.2f")
    logger.info("Saved %d sales rows to %s", len(sales), output)


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
        store_column=cfg.store_column or "store_id",
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
