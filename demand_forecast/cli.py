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
fev_app = typer.Typer(help="Run fev benchmarks and compare evaluation summaries.")
app.add_typer(fev_app, name="fev")


@app.command("train")
def train(
    config: Path = typer.Option(
        ..., "--config", exists=True, dir_okay=False, help="Long-panel settings YAML"
    ),
    out: Path = typer.Option(..., "--out", help="Run directory for bundle and checkpoints"),
    resume: bool = typer.Option(False, "--resume", help="Resume a matching run"),
):
    """Train a quantile model and save its bundle and run report."""
    import yaml

    from demand_forecast.config import Settings
    from demand_forecast.core.run import train_from_settings
    from demand_forecast.core.training import InterruptedRun

    try:
        settings = Settings.from_yaml(config)
        configure_logging(settings.log_level)
        bundle = train_from_settings(settings, out, resume=resume)
    except (ValueError, OSError, ImportError, NotImplementedError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except InterruptedRun as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Bundle saved to {bundle}")


@app.command("forecast")
def forecast(
    bundle: Path = typer.Option(
        ..., "--bundle", exists=True, file_okay=False, help="Trained bundle directory"
    ),
    config: Path = typer.Option(
        ..., "--config", exists=True, dir_okay=False, help="Long-panel settings YAML"
    ),
    origin: str = typer.Option(..., "--origin", help="First forecast timestamp (ISO format)"),
    out: Path = typer.Option(..., "--out", help="Output forecast CSV"),
):
    """Forecast from a saved bundle using the declared long-panel inputs."""
    import pandas as pd
    import yaml

    from demand_forecast.config import Settings
    from demand_forecast.core.run import forecast_from_bundle

    try:
        settings = Settings.from_yaml(config)
        configure_logging(settings.log_level)
        forecast_from_bundle(bundle, settings, pd.Timestamp(origin), out)
    except (ValueError, OSError, ImportError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Forecasts saved to {out}")


@fev_app.command("run")
def fev_run(
    tasks: str = typer.Option(
        "weekly", "--tasks", help="weekly, comma-separated names, or task YAML"
    ),
    config: Path = typer.Option(
        ..., "--config", exists=True, dir_okay=False, help="Model and training settings YAML"
    ),
    policy: str = typer.Option("first_window", "--policy", help="first_window or per_window"),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory for bundles and summaries.csv"),
    profiles: Path | None = typer.Option(
        None, "--profiles", exists=True, dir_okay=False, help="Override task role profiles YAML"
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume matching runs and missing windows"),
    model_name: str | None = typer.Option(None, "--model-name", help="Model label in summaries"),
    stop_after: int | None = typer.Option(
        None, "--stop-after", min=1, help="Interrupt training at this step, saving a checkpoint"
    ),
):
    """Evaluate model forecasts on fev tasks, reusing each task's first-window bundle."""
    import yaml

    from demand_forecast.bench.run import run_fev
    from demand_forecast.bench.specs import load_task_specs
    from demand_forecast.config import Settings
    from demand_forecast.core.training import InterruptedRun

    try:
        settings = Settings.from_yaml(config)
        configure_logging(settings.log_level)
        run_fev(
            load_task_specs(tasks, profiles),
            settings,
            out_dir,
            policy=policy,
            resume=resume,
            model_name=model_name,
            stop_after=stop_after,
        )
    except (ValueError, OSError, ImportError, NotImplementedError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except InterruptedRun as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Summaries saved to {out_dir / 'summaries.csv'}")


@fev_app.command("baseline")
def fev_baseline(
    tasks: str = typer.Option(
        "weekly", "--tasks", help="weekly, comma-separated names, or task YAML"
    ),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory for resumable summaries.csv"),
    profiles: Path | None = typer.Option(
        None, "--profiles", exists=True, dir_okay=False, help="Override task role profiles YAML"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Evaluate seasonal naive, skipping tasks already saved in the output directory."""
    from demand_forecast.bench.run import run_baseline
    from demand_forecast.bench.specs import load_task_specs

    configure_logging("DEBUG" if verbose else "INFO")
    try:
        run_baseline(load_task_specs(tasks, profiles), out_dir)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Summaries saved to {out_dir / 'summaries.csv'}")


@fev_app.command("leaderboard")
def fev_leaderboard(
    summaries_csv: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="CSV of fev evaluation summaries"
    ),
    baseline: str = typer.Option(
        "seasonal_naive", "--baseline", help="Reference model for relative skill scores"
    ),
):
    """Show the official fev leaderboard with bootstrap confidence intervals."""
    from demand_forecast.bench.run import leaderboard

    try:
        board = leaderboard(summaries_csv, baseline=baseline)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(board.to_string())


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
    output: Path = typer.Argument(..., help="Output CSV path, or directory for long format"),
    num_days: int = typer.Option(
        365 * 4, "--days", help="Number of days (grid steps in long format)"
    ),
    num_products: int = typer.Option(
        100, "--products", help="Number of products (series in long format)"
    ),
    num_stores: int = typer.Option(20, "--stores", help="Number of stores (sales format only)"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    output_format: str = typer.Option("sales", "--format", help="Output format: sales or long"),
    freq: str = typer.Option("D", "--freq", help="Grid frequency (long format only)"),
    start: str = typer.Option(
        "2020-01-01", "--start", help="First timestamp on the grid (long format only)"
    ),
):
    """Generate synthetic sales, or long.csv/static.csv with promo, oil and cat fields."""
    configure_logging("DEBUG" if verbose else "INFO")
    logger = logging.getLogger(__name__)

    from demand_forecast.synthetic.generator import generate_long_panel, generate_sales_data

    if output_format not in {"sales", "long"}:
        raise typer.BadParameter("must be sales or long", param_hint="--format")
    if output_format == "long":
        try:
            long, static = generate_long_panel(
                n_series=num_products,
                length=num_days,
                freq=freq,
                start=start,
                known=("promo",),
                past=("oil",),
                statics=("cat",),
                seed=seed,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        output.mkdir(parents=True, exist_ok=True)
        long.to_csv(output / "long.csv", index=False)
        static.to_csv(output / "static.csv", index=False)
        logger.info("Generated %d rows, saved long.csv and static.csv to %s", len(long), output)
        return

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
def version():
    """Show version information."""
    from demand_forecast import __version__

    typer.echo(f"demand-forecast version {__version__}")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
