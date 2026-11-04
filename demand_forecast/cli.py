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
pretrain_app = typer.Typer(help="Pretrain on local corpus shards and measure run costs.")
app.add_typer(pretrain_app, name="pretrain")
corpus_app = typer.Typer(help="Build and verify a pinned pretraining corpus.")
pretrain_app.add_typer(corpus_app, name="corpus")


@corpus_app.command("build")
def pretrain_corpus_build(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    root: Path = typer.Option(..., "--root"),
    source: list[str] | None = typer.Option(None, "--source", help="Select a source; repeatable"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    keep_downloads: bool = typer.Option(False, "--keep-downloads"),
):
    """Convert selected pinned sources, or list their files without downloading."""
    import yaml

    from demand_forecast.pretrain.build import build_corpus, select_sources, source_files
    from demand_forecast.pretrain.build_config import CorpusBuildConfig

    try:
        cfg = CorpusBuildConfig.from_yaml(config)
        if dry_run:
            for selected in select_sources(cfg, source or ()):
                files = source_files(selected)
                for item in files:
                    typer.echo(f"{selected.name}\t{item['path']}\t{item['size']} bytes")
                typer.echo(
                    f"{selected.name}: {len(files)} files, {sum(f['size'] for f in files)} bytes"
                )
            return
        report = build_corpus(config, root, names=source or (), keep_downloads=keep_downloads)
    except (ValueError, OSError, ImportError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"Built {report['totals']['series_kept']} series; manifest: {root / 'manifest.json'}"
    )


@corpus_app.command("verify")
def pretrain_corpus_verify(root: Path = typer.Option(..., "--root")):
    """Rehash every manifest shard and fail on changed or missing files."""
    from demand_forecast.pretrain.corpus import verify_manifest

    try:
        counts = verify_manifest(root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    for kind, (valid, failed) in sorted(counts.items()):
        typer.echo(f"{kind}: {valid} verified, {failed} failed")
    if any(failed for _, failed in counts.values()):
        raise typer.Exit(code=1)


@corpus_app.command("audit")
def pretrain_corpus_audit(
    root: Path | None = typer.Option(None, "--root"),
    out: Path | None = typer.Option(None, "--out"),
    tasks: str = typer.Option("fev100", "--tasks"),
    kinds: str = typer.Option("chronos_real,gifteval_pretrain", "--kinds"),
    workers: int | None = typer.Option(None, "--workers", min=1),
    resume: bool = typer.Option(False, "--resume"),
    controls: bool = typer.Option(False, "--controls"),
    control_constants: Path | None = typer.Option(
        None,
        "--control-constants",
        exists=True,
        dir_okay=False,
        help="JSON constant overrides for --controls only; never changes a corpus audit.",
    ),
):
    """Match corpus values against every fev target and numeric covariate."""
    import json

    from demand_forecast.pretrain import audit_match
    from demand_forecast.pretrain.audit import _write_json, constants, run_audit

    if control_constants is not None and not controls:
        raise typer.BadParameter("--control-constants requires --controls")
    if controls:
        from demand_forecast.pretrain.audit_controls import run_controls

        original = constants()
        try:
            if control_constants is not None:
                overrides = json.loads(control_constants.read_text())
                if not isinstance(overrides, dict) or set(overrides) - original.keys():
                    raise typer.BadParameter("control overrides must name matcher constants")
                for key, value in overrides.items():
                    previous = original[key]
                    if isinstance(previous, dict):
                        valid = (
                            isinstance(value, dict)
                            and value.keys() == previous.keys()
                            and all(isinstance(v, int) and v >= 0 for v in value.values())
                        )
                    else:
                        valid = (
                            isinstance(value, (int, float)) and value > 0 and value < float("inf")
                        )
                        if isinstance(previous, int):
                            valid = valid and isinstance(value, int)
                    if not valid:
                        raise typer.BadParameter(f"invalid value for {key}")
                    setattr(audit_match, key, value)
            result = run_controls()
            result["constants"] = constants()
            if out is not None:
                _write_json(out / "controls.json", result)
        finally:
            for key, value in original.items():
                setattr(audit_match, key, value)
        typer.echo(
            f"{result['positive_passed']}/26 positive detected, "
            f"{result['negative_passed']}/14 negative rejected, "
            f"{result['self_tests_passed']}/10 self-tests"
        )
        if (result["positive_passed"], result["negative_passed"], result["self_tests_passed"]) != (
            26,
            14,
            10,
        ):
            raise typer.Exit(code=1)
        return
    if root is None or out is None:
        raise typer.BadParameter("--root and --out are required unless running --controls")
    try:
        result = run_audit(
            root,
            out,
            tasks=tasks,
            kinds=[kind.strip() for kind in kinds.split(",") if kind.strip()],
            workers=workers,
            resume=resume,
            progress=typer.echo,
        )
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Audit: {result['summary']['copies']} copy pairs; "
        f"{result['summary']['unresolved_rows']} unresolved rows; {out / 'audit.md'}"
    )
    if result["summary"]["unresolved_rows"]:
        raise typer.Exit(code=2)


def _training_settings(config: Path, init_from: Path | None):
    """Merge a CLI bundle override before validating the settings YAML."""
    import yaml

    from demand_forecast.config import Settings

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise typer.BadParameter("config root must be a mapping")
    training = raw.setdefault("training", {})
    if not isinstance(training, dict):
        raise typer.BadParameter("training must be a mapping")
    if init_from is not None:
        training["init_from"] = str(init_from)
    return Settings.model_validate(raw)


@pretrain_app.command("run")
def pretrain_run(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    out: Path = typer.Option(..., "--out", help="Run directory for checkpoints and bundle"),
    resume: bool = typer.Option(False, "--resume"),
    stop_after: int | None = typer.Option(None, "--stop-after", min=1),
):
    """Train to the fixed pretraining budget, or resume a matching checkpoint."""
    import yaml

    from demand_forecast.pretrain.config import PretrainConfig
    from demand_forecast.pretrain.train import pretrain

    try:
        cfg = PretrainConfig.from_yaml(config)
        configure_logging(cfg.log_level)
        pretrain(cfg, out, resume=resume, stop_after=stop_after)
    except (ValueError, OSError, ImportError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Bundle saved to {out / 'bundle'}")


@pretrain_app.command("probe")
def pretrain_probe_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    out: Path = typer.Option(..., "--out", help="Probe JSON path"),
    steps: int = typer.Option(50, "--steps", min=1),
):
    """Measure corpus training, validation, checkpoint and shard overheads."""
    import yaml

    from demand_forecast.pretrain.config import PretrainConfig
    from demand_forecast.pretrain.probe import pretrain_probe

    try:
        cfg = PretrainConfig.from_yaml(config)
        configure_logging(cfg.log_level)
        pretrain_probe(cfg, steps=steps, out=out)
    except (ValueError, OSError, ImportError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Probe saved to {out}")


@pretrain_app.command("probe-task")
def pretrain_probe_task(
    tasks: str = typer.Option("weekly", "--tasks", help="weekly, heldout, or task YAML"),
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    out: Path = typer.Option(..., "--out", help="Directory for task probe JSONs"),
    steps: int = typer.Option(30, "--steps", min=1),
    profiles: Path | None = typer.Option(None, "--profiles", exists=True, dir_okay=False),
    max_series: int | None = typer.Option(
        None, "--max-series", min=1, help="Limit series for testing"
    ),
):
    """Measure training and last-window inference with real task shapes."""
    import yaml

    from demand_forecast.bench.specs import load_task_specs
    from demand_forecast.config import Settings
    from demand_forecast.pretrain.probe import task_probe

    try:
        settings = Settings.from_yaml(config)
        configure_logging(settings.log_level)
        task_probe(
            load_task_specs(tasks, profiles), settings, steps=steps, out=out, max_series=max_series
        )
    except (ValueError, OSError, ImportError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Task probes saved to {out}")


@app.command("train")
def train(
    config: Path = typer.Option(
        ..., "--config", exists=True, dir_okay=False, help="Long-panel settings YAML"
    ),
    out: Path = typer.Option(..., "--out", help="Run directory for bundle and checkpoints"),
    resume: bool = typer.Option(False, "--resume", help="Resume a matching run"),
    init_from: Path | None = typer.Option(
        None, "--init-from", exists=True, file_okay=False, help="Pretrained bundle directory"
    ),
):
    """Train a quantile model and save its bundle and run report."""
    import yaml

    from demand_forecast.core.run import train_from_settings
    from demand_forecast.core.training import InterruptedRun, NonFiniteTraining

    try:
        settings = _training_settings(config, init_from)
        configure_logging(settings.log_level)
        bundle = train_from_settings(settings, out, resume=resume)
    except (ValueError, OSError, ImportError, NotImplementedError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except (InterruptedRun, NonFiniteTraining) as exc:
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
        "weekly", "--tasks", help="weekly, heldout, comma-separated weekly names, or task YAML"
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
    init_from: Path | None = typer.Option(
        None, "--init-from", exists=True, file_okay=False, help="Pretrained bundle directory"
    ),
):
    """Evaluate model forecasts on fev tasks, reusing each task's first-window bundle."""
    import yaml

    from demand_forecast.bench.run import run_fev
    from demand_forecast.bench.specs import load_task_specs
    from demand_forecast.core.training import InterruptedRun, NonFiniteTraining

    try:
        settings = _training_settings(config, init_from)
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
    except (InterruptedRun, NonFiniteTraining) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Summaries saved to {out_dir / 'summaries.csv'}")


@fev_app.command("tune")
def fev_tune(
    tasks: str = typer.Option(
        "weekly", "--tasks", help="weekly, heldout, comma-separated weekly names, or task YAML"
    ),
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    init_from: Path = typer.Option(
        ..., "--init-from", exists=True, file_okay=False, help="Pretrained bundle directory"
    ),
    out_dir: Path = typer.Option(..., "--out-dir", help="Study database, trial reports and recipe"),
    trials: int = typer.Option(..., "--trials", min=1, help="Total trials, including resumed ones"),
    timeout_s: float | None = typer.Option(None, "--timeout-s", min=0),
    seed: int = typer.Option(0, "--seed", min=0, max=2**32 - 1, help="Optuna sampler seed"),
    resume: bool = typer.Option(False, "--resume", help="Continue an existing study"),
):
    """Tune one shared recipe on inner validation only; never evaluate fev test windows."""
    import yaml

    from demand_forecast.bench.specs import load_task_specs
    from demand_forecast.bench.tune import tune_finetune
    from demand_forecast.config import Settings

    try:
        settings = Settings.from_yaml(config)
        configure_logging(settings.log_level)
        tune_finetune(
            load_task_specs(tasks),
            settings,
            init_from,
            out_dir,
            n_trials=trials,
            timeout_s=timeout_s,
            seed=seed,
            resume=resume,
        )
    except (ValueError, OSError, ImportError, NotImplementedError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Study saved to {out_dir / 'tune.json'}")


@fev_app.command("baseline")
def fev_baseline(
    tasks: str = typer.Option(
        "weekly", "--tasks", help="weekly, heldout, comma-separated weekly names, or task YAML"
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
