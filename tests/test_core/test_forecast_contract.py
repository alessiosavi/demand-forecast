"""Regression checks for cutoff isolation, weekly units, and the store command."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal
from torch import nn
from typer.testing import CliRunner

from demand_forecast.cli import app
from demand_forecast.config.settings import Settings
from demand_forecast.core.evaluator import Evaluator
from demand_forecast.core.forecast_run import (
    package_commit,
    package_source_identity,
    run_forecast,
    score_forecast,
    validate_forecast_contract,
)
from demand_forecast.core.pipeline import TIME_COLUMNS, ForecastPipeline
from demand_forecast.core.trainer import EarlyStopConfig, Trainer
from demand_forecast.core.tuning import HyperparameterTuner, SearchSpace, suggest_hyperparameters
from demand_forecast.data.dataloader import create_dataloader
from demand_forecast.data.dataset import create_time_series_data
from demand_forecast.data.feature_engineering import CategoricalEncoder
from demand_forecast.data.loader import load_sales_data
from demand_forecast.data.preprocessor import catalogue_weekly_grid, filter_skus, resample_series
from demand_forecast.utils.metrics import wape
from demand_forecast.utils.time_features import calculate_time_features
from demand_forecast.utils.timeseries import create_timeseries


@pytest.fixture
def weekly_pipeline():
    settings = Settings.from_yaml(Path("configs/hackathon.yaml"))
    settings.device = "cpu"
    settings.model.d_model = 16
    settings.model.num_encoder_layers = 1
    settings.model.dim_feedforward = 32
    settings.training.batch_size = 8
    pipeline = ForecastPipeline(settings)
    weeks = pd.date_range(end="2026-10-05", periods=110, freq="W-MON")
    products = pd.DataFrame(
        {
            "id": ["a", "b", "zero"],
            "category": ["tops", "shoes", "tops"],
            "subcategory": ["shirt", "boots", "shirt"],
            "gender": ["F", "M", "U"],
            "brand": ["one", "two", "new"],
            "price": [10.0, 100.0, 25.0],
        }
    )
    rows = []
    for i, product in products.iterrows():
        for j, week in enumerate(weeks):
            rows.append(
                {
                    "date": week,
                    "product_id": product.id,
                    "store_id": "store",
                    "qty": float((i + 1) * (1 + j % 5)) if i < 2 else 0.0,
                    **product.drop("id").to_dict(),
                }
            )
    pipeline.products = products
    pipeline._weekly = pd.DataFrame(rows).set_index("date")
    return pipeline


@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("scaling,clipping", [("sku", False), ("cluster", True)])
def test_preprocessing_and_training_are_cutoff_invariant(weekly_pipeline, stage, scaling, clipping):
    pipeline = weekly_pipeline
    pipeline.settings.data.scaling_group = scaling
    pipeline.settings.data.clip_outliers = clipping
    cutoff = pipeline.cutoffs()[stage]
    pipeline.fit_preprocessing(cutoff)
    data = pipeline.create_datasets(refit=True)[0][0]
    scalers = {
        k: (v.mean_.copy(), v.scale_.copy()) for k, v in pipeline.scaler_manager.scalers.items()
    }
    encoder = pipeline.categorical_encoder
    sku_map = pipeline.sku_to_index.copy()
    mask = pipeline._weekly.index >= cutoff
    qty = pipeline._weekly.loc[mask, "qty"]
    pipeline._weekly.loc[mask, "qty"] = np.where(qty == 0, 1e6, 0)
    pipeline._weekly.loc[mask, "category"] = "FUTURE_ONLY"
    pipeline.fit_preprocessing(cutoff)
    changed = pipeline.create_datasets(refit=True)[0][0]
    assert pipeline.sku_to_index == sku_map
    assert pipeline.categorical_encoder == encoder
    for key, scaler in pipeline.scaler_manager.scalers.items():
        np.testing.assert_array_equal(scaler.mean_, scalers[key][0])
        np.testing.assert_array_equal(scaler.scale_, scalers[key][1])
    for before, after in zip(data, changed):
        np.testing.assert_array_equal(before, after)
    assert (data.origins + np.timedelta64(15 * 7, "D") < cutoff.to_datetime64()).all()
    # Zero-only catalogue products remain members in both stages.
    assert "zero" in sku_map


def test_windows_use_target_calendar_and_include_last_window(weekly_pipeline):
    p = weekly_pipeline
    val, holdout, _ = p.cutoffs()
    p.fit_preprocessing(val)
    train, validation = p.create_datasets(validation_origin=val)[0]
    assert len(validation.y) == 3
    assert np.all(validation.origins == val.to_datetime64())
    assert (train.origins + np.timedelta64(105, "D") < val.to_datetime64()).all()
    expected = calculate_time_features(pd.date_range(val, periods=16, freq="W-MON")).astype(
        np.float32
    )
    np.testing.assert_allclose(validation.future_time[0], expected)
    assert validation.cat_codes.dtype == np.int64
    assert validation.cont.dtype == np.float32
    all_windows = create_time_series_data(
        p._series,
        ["qty_scaled", "sku_code", *TIME_COLUMNS],
        p.categorical_encoder.encoded_columns,
        target_end_max=holdout - pd.Timedelta(weeks=1),
        continuous_features=p.continuous_features,
    )
    assert all_windows.origins.max() == (holdout - pd.Timedelta(weeks=16)).to_datetime64()
    x, _, y = create_timeseries(np.arange(7), np.arange(7), np.arange(7), window=5, n_out=2)
    assert x.shape == (1, 5)
    np.testing.assert_array_equal(y, [[5, 6]])


def test_filter_skus_preserves_datetime_index():
    dates = pd.date_range("2024-01-01", periods=12, freq="W-MON")
    df = pd.DataFrame(
        {"sku": ["keep"] * 12 + ["drop"] * 12, "store_id": "s", "qty": [1] * 12 + [0] * 12},
        index=dates.append(dates),
    )
    result = filter_skus(df, window=4, n_out=2)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.equals(dates)
    assert result.sku.unique().tolist() == ["keep"]


def test_encoder_codes_unknown_and_save_load(tmp_path):
    frame = pd.DataFrame({"category": ["shoe", "shirt", "shoe"], "price": [0, 10, 100]})
    encoder = CategoricalEncoder().fit(frame, ["category"], ["price"])
    assert encoder.vocab_sizes == {"encoded_category": 3}
    encoded = encoder.transform(frame)
    assert encoded.encoded_category.tolist() == [2, 1, 2]
    assert (
        encoder.transform(
            pd.DataFrame({"category": ["unknown"], "price": [10]})
        ).encoded_category.iloc[0]
        == 0
    )
    encoder.save(tmp_path / "encoders.joblib")
    assert_frame_equal(
        encoded, CategoricalEncoder.load(tmp_path / "encoders.joblib").transform(frame)
    )
    assert np.isclose(encoded.continuous_price.mean(), 0)


def test_collate_model_forward_and_static_features(weekly_pipeline):
    p = weekly_pipeline
    p.fit_preprocessing(p.cutoffs()[0])
    p.create_datasets()
    model = p.build_model().eval()
    batch = next(iter(p.dataloaders()[0]["0"]))
    assert all(
        codes.shape == (8,) and codes.dtype == torch.long for codes in batch["cats"].values()
    )
    assert batch["cont"].shape == (8, 1)
    with torch.no_grad():
        output = model("0", **{k: v for k, v in batch.items() if k != "y"})
        changed = {k: v for k, v in batch.items() if k != "y"}
        changed["cats"] = {k: torch.zeros_like(v) for k, v in batch["cats"].items()}
        changed["cont"] = batch["cont"] + 4
        other = model("0", **changed)
    assert output.shape == (8, 16)
    assert torch.isfinite(output).all()
    assert not torch.allclose(output, other)


class ConstantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, n, qty, past_time, future_time, sku, cats, cont=None):
        return self.value.expand(len(qty), future_time.shape[1])


def test_forecast_inverse_units_and_artifact_reload(weekly_pipeline, tmp_path):
    p = weekly_pipeline
    p.fit_preprocessing(p.cutoffs()[1])
    p.model = ConstantModel()
    origin = p.cutoffs()[1]
    result = p.forecast(origin)
    validate_forecast_contract(result, ["a", "b", "zero"], origin)
    assert len(result) == 3 * 16
    for sku, group in result.groupby("product_id"):
        assert np.all(group.qty == np.round(p.scaler_manager.scalers[sku].mean_[0], 2))
    # Forecast must use transform only; input history changes, fitted stats do not.
    means = {k: v.mean_.copy() for k, v in p.scaler_manager.scalers.items()}
    p.forecast(p.cutoffs()[2])
    for k, v in means.items():
        np.testing.assert_equal(p.scaler_manager.scalers[k].mean_, v)
    p.create_datasets(refit=True)
    p.build_model()
    expected = p.forecast(origin)
    p.save(tmp_path / "model.pt")
    reloaded = ForecastPipeline(p.settings)
    reloaded.load(tmp_path / "model.pt")
    reloaded._weekly = p._weekly
    reloaded._series = reloaded._transform(p._weekly)
    assert_frame_equal(expected, reloaded.forecast(origin))


def test_wape():
    assert wape([0, 0], [1, 2]) is None
    assert wape([1, 3], [2, 2]) == 0.5
    assert wape([1, 3], [1, 3]) == 0
    with pytest.raises(ValueError):
        wape([1], [1, 2])


def test_monday_resampling_and_missing_catalogue_rows(tmp_path, weekly_pipeline):
    dates = pd.date_range("2024-10-17", "2024-11-05")
    sales = pd.DataFrame({"sku": "a", "store_id": "store", "qty": 1}, index=dates)
    weekly = resample_series(sales, "W-MON")
    assert weekly.index.tolist() == [pd.Timestamp("2024-10-21"), pd.Timestamp("2024-10-28")]
    assert weekly.qty.tolist() == [7, 7]
    cfg = weekly_pipeline.settings.data
    sales = sales.rename(columns={"sku": "product_id"})
    grid = catalogue_weekly_grid(sales, weekly_pipeline.products, cfg)
    assert len(grid) == 6
    assert grid[grid.product_id == "zero"].qty.sum() == 0
    assert grid[grid.product_id == "zero"].price.eq(25).all()
    sales.drop(columns="store_id").rename_axis("date").reset_index().to_csv(
        tmp_path / "sales.csv", index=False
    )
    loaded = load_sales_data(tmp_path / "sales.csv", store_column=None)
    assert loaded.store_id.eq("store").all()


@pytest.mark.parametrize("mutation", ["week", "missing", "origin", "duplicate", "negative", "nan"])
def test_contract_rejects_bad_exports(weekly_pipeline, mutation):
    p = weekly_pipeline
    p.fit_preprocessing(p.cutoffs()[1])
    p.model = ConstantModel()
    frame = p.forecast(p.cutoffs()[1])
    if mutation == "week":
        frame.loc[0, "week_start"] = "2026-06-23"
    elif mutation == "missing":
        frame = frame[frame.product_id != "zero"]
    elif mutation == "origin":
        frame.loc[0, "origin"] = "2026-06-29"
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame.loc[0, "qty"] = -1 if mutation == "negative" else np.nan
    with pytest.raises(ValueError):
        validate_forecast_contract(frame, ["a", "b", "zero"], p.cutoffs()[1])


def test_hackathon_search_resolves_choices_and_disables_unsearched_fields():
    study = optuna.create_study()
    for _ in range(4):
        trial = study.ask()
        params = suggest_hyperparameters(trial, SearchSpace.hackathon(), fixed_nhead=4)
        assert set(params) == {"d_model", "learning_rate", "weight_decay", "dropout"}
        assert params["d_model"] in [64, 128]
        assert "d_model_base" not in trial.params
    with pytest.raises(ValueError, match="divisible"):
        suggest_hyperparameters(study.ask(), SearchSpace(d_model_choices=[63], nhead=[4]))


def make_tuner(p, **kwargs):
    p.fit_preprocessing(p.cutoffs()[0])
    p.create_datasets()
    train, val, _, _ = p.dataloaders()
    return HyperparameterTuner(
        "standard",
        train["0"],
        val["0"],
        p.model_kwargs(),
        search_space=SearchSpace.hackathon(),
        num_epochs_per_trial=1,
        **kwargs,
    )


def test_zero_completed_trials_fallback(weekly_pipeline):
    tuner = make_tuner(weekly_pipeline)
    params = tuner.tune(n_trials=0, show_progress_bar=False)
    assert params["d_model"] == 16
    assert params["nhead"] == 4
    assert tuner.fallback_reason
    assert tuner.study.trials == []


def test_tuner_deadline_prunes_cleanly(weekly_pipeline):
    tuner = make_tuner(weekly_pipeline, deadline=time.monotonic() - 1)
    params = tuner.tune(n_trials=2, show_progress_bar=False)
    assert params["d_model"] == 16
    assert tuner.fallback_reason
    assert len(tuner.study.trials) == 1
    assert tuner.study.trials[0].state == optuna.trial.TrialState.PRUNED


def test_tuner_resolved_parameters_and_best_epoch(weekly_pipeline):
    tuner = make_tuner(weekly_pipeline)
    # Keep the architecture tiny here; the dedicated search test checks the run's choices.
    tuner.search_space.d_model_choices = [16]
    params = tuner.tune(n_trials=1, show_progress_bar=False)
    assert params["d_model"] == 16
    assert params["nhead"] == 4
    assert params == tuner.study.best_trial.user_attrs["resolved_params"]
    assert tuner.study.best_trial.user_attrs["best_epoch"] == 1


def small_loader():
    loader, _ = create_dataloader(
        np.zeros((3, 5, 6), dtype=np.float32),
        np.empty((3, 0), dtype=np.int64),
        np.ones((3, 2), dtype=np.float32),
        [],
        batch_size=8,
        future_time=np.zeros((3, 2, 4), dtype=np.float32),
    )
    return loader


def test_trainer_restores_best_weights_on_exhaustion_and_ignores_test(monkeypatch):
    snapshots = []

    def validate(evaluator, *args, **kwargs):
        snapshots.append(evaluator.model.value.detach().clone())
        return SimpleNamespace(avg_loss=float(len(snapshots)), metrics={})

    monkeypatch.setattr(Evaluator, "validate", validate)
    model = ConstantModel()
    loader = small_loader()
    trainer = Trainer(
        model,
        {"0": loader},
        val_dataloaders={"0": loader},
        test_dataloaders={"forbidden": object()},
        num_epochs=3,
        learning_rate=0.1,
        early_stop=EarlyStopConfig(patience=4, min_delta=0),
    )
    trainer.train()
    assert trainer.best_epoch == 1
    assert len(trainer.history["train_loss"]) == 3
    assert torch.equal(model.value, snapshots[0])
    assert not torch.equal(snapshots[0], snapshots[-1])
    fixed = Trainer(
        ConstantModel(), {"0": loader}, test_dataloaders={"forbidden": object()}, num_epochs=2
    )
    fixed.train()
    assert len(fixed.history["train_loss"]) == 2
    assert fixed.history["val_loss"] == []


def test_evaluator_quality_plot_marks_standardized_targets(tmp_path, monkeypatch):
    loader = small_loader()
    calls = []
    monkeypatch.setattr(
        "demand_forecast.core.evaluator.plot_prediction_quality",
        lambda **kwargs: calls.append(kwargs),
    )
    evaluator = Evaluator(
        ConstantModel(), nn.MSELoss(), batch_size=8, total_examples=3, plot_dir=tmp_path
    )
    result = evaluator.validate(({"0": batch} for batch in loader), plot=True, epoch=1)
    assert len(calls) == 1
    assert calls[0]["standardized"] is True
    assert calls[0]["save_path"] == tmp_path / "quality_epoch_1.png"
    np.testing.assert_array_equal(calls[0]["actuals"], result.flatten_actuals)
    np.testing.assert_array_equal(calls[0]["predictions"], result.flatten_predictions)
    assert len(result.flatten_predictions) == 6


def test_trainer_deadline_and_evaluator_weekly_plot(tmp_path):
    loader = small_loader()
    model = ConstantModel()
    trainer = Trainer(model, {"0": loader}, num_epochs=3, deadline=time.monotonic() - 1)
    trainer.train()
    assert trainer.budget_exceeded
    assert len(trainer.history["train_loss"]) == 1
    evaluator = Evaluator(model, nn.MSELoss(), batch_size=8, total_examples=3, plot_dir=tmp_path)
    result = evaluator.validate(({"0": batch} for batch in loader), plot=True, epoch=1)
    assert np.asarray(result.predictions).shape == (3, 2)
    assert len(result.flatten_predictions) == 6
    assert (tmp_path / "validation_epoch_1.png").is_file()
    assert result.avg_loss == pytest.approx(result.mse)


def test_trainer_plots_restored_best_weights_after_early_stop(tmp_path, monkeypatch):
    loader = small_loader()
    snapshots = []
    original_validate = Evaluator.validate

    def validate(evaluator, *args, **kwargs):
        snapshots.append(evaluator.model.value.detach().clone())
        result = original_validate(evaluator, *args, **kwargs)
        # Deterministically stop at epoch 2, before the first periodic plot at epoch 5.
        result.avg_loss = float(len(snapshots))
        return result

    monkeypatch.setattr(Evaluator, "validate", validate)
    trainer = Trainer(
        ConstantModel(),
        {"0": loader},
        val_dataloaders={"0": loader},
        num_epochs=10,
        learning_rate=0.1,
        early_stop=EarlyStopConfig(patience=1, min_delta=0),
        plot_dir=tmp_path,
    )
    trainer.train(plot_every_n_epochs=5)
    assert trainer.best_epoch == 1
    assert len(trainer.history["train_loss"]) == len(trainer.history["val_loss"]) == 2
    assert len(snapshots) == 3
    assert torch.equal(snapshots[-1], snapshots[0])
    assert not torch.equal(snapshots[1], snapshots[0])
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "validation_epoch_best.png",
        "quality_epoch_best.png",
    }


def test_refit_learning_rates_follow_selection_horizon():
    loader = small_loader()
    selection = Trainer(ConstantModel(), {"0": loader}, num_epochs=5, learning_rate=0.1)
    refit = Trainer(
        ConstantModel(),
        {"0": loader},
        num_epochs=2,
        learning_rate=0.1,
        scheduler_epochs=selection.num_epochs,
    )
    assert selection._schedulers["0"].T_max == refit._schedulers["0"].T_max == 5
    rates = []
    for trainer in (selection, refit):
        used = []
        trainer.callbacks = [
            SimpleNamespace(
                on_epoch_start=lambda *args: used.append(
                    trainer._optimizers["0"].param_groups[0]["lr"]
                ),
                on_epoch_end=lambda *args: None,
                on_batch_end=lambda *args: None,
            )
        ]
        trainer.train()
        rates.append(used)
    assert rates[1] == rates[0][:2]


@pytest.mark.parametrize("override", [False, True])
def test_forecast_cli_preserves_yaml_unless_overridden(tmp_path, monkeypatch, override):
    settings = Settings.from_yaml(Path("configs/hackathon.yaml"))
    settings.tuning.n_trials = 7
    settings.tuning.timeout = 123
    settings.training.num_epochs = 37
    settings.data.k = 3
    settings.seed = 19
    settings.budget_seconds = 4321
    config = tmp_path / "config.yaml"
    settings.to_yaml(config)
    captured = []
    monkeypatch.setattr(
        "demand_forecast.core.forecast_run.run_forecast",
        lambda settings, *args, **kwargs: captured.append((settings, kwargs)),
    )
    args = [
        "forecast",
        "--config",
        str(config),
        "--products",
        str(config),
        "--sales",
        str(config),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    if override:
        args += [
            "--trials",
            "0",
            "--timeout",
            "45",
            "--max-epochs",
            "2",
            "--k",
            "1",
            "--seed",
            "0",
            "--budget-seconds",
            "120",
            "--expect-origins",
            "2026-06-22,2026-10-12",
        ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    actual, kwargs = captured[0]
    values = (
        actual.tuning.n_trials,
        actual.tuning.timeout,
        actual.training.num_epochs,
        actual.data.k,
        actual.seed,
        actual.budget_seconds,
    )
    assert values == ((0, 45, 2, 1, 0, 120) if override else (7, 123, 37, 3, 19, 4321))
    assert kwargs["expect_origins"] == (
        (pd.Timestamp("2026-06-22").date(), pd.Timestamp("2026-10-12").date()) if override else None
    )


@pytest.mark.parametrize(
    "value", ["2026-06-22", "2026-06-22,invalid", "2026-06-22,2026-10-12,extra"]
)
def test_forecast_cli_rejects_malformed_expected_origins(tmp_path, value):
    config = Path("configs/hackathon.yaml")
    result = CliRunner().invoke(
        app,
        [
            "forecast",
            "--config",
            str(config),
            "--products",
            str(config),
            "--sales",
            str(config),
            "--out-dir",
            str(tmp_path / "out"),
            "--expect-origins",
            value,
        ],
    )
    assert result.exit_code != 0
    assert "requires HOLDOUT,FUTURE as ISO dates" in result.output
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("expected", [("2026-06-29", "2026-10-12"), ("2026-06-22", "2026-10-19")])
def test_forecast_rejects_unexpected_origins_before_training(
    weekly_pipeline, tmp_path, monkeypatch, expected
):
    monkeypatch.setattr(
        "demand_forecast.core.forecast_run.ForecastPipeline", lambda _: weekly_pipeline
    )
    monkeypatch.setattr(weekly_pipeline, "load_data", lambda **kwargs: None)
    monkeypatch.setattr(
        "demand_forecast.core.forecast_run.diagnostic_clustering",
        lambda *args: pytest.fail("Must reject origins before preprocessing/training"),
    )
    with pytest.raises(ValueError, match="Forecast origins differ"):
        run_forecast(weekly_pipeline.settings, tmp_path, expect_origins=expected)


@pytest.mark.parametrize(
    "checkout,status,env,expected,dirty",
    [
        (True, "", None, "abc123", False),
        (True, " M module.py\n", None, "abc123-dirty", True),
        (True, "?? new.py\n", "env-tag", "abc123-dirty", True),
        (False, "", "env-tag-dirty", "env-tag-dirty", False),
        (False, "", None, "unknown", False),
    ],
)
def test_package_source_identity(monkeypatch, checkout, status, env, expected, dirty):
    from demand_forecast.core import forecast_run

    if env is None:
        monkeypatch.delenv("DEMAND_FORECAST_COMMIT", raising=False)
    else:
        monkeypatch.setenv("DEMAND_FORECAST_COMMIT", env)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["cwd"] == Path(forecast_run.__file__).resolve().parents[2]
        if args == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0 if checkout else 128, stdout="abc123\n")
        assert args == ["git", "status", "--porcelain"]
        return SimpleNamespace(returncode=0, stdout=status)

    monkeypatch.setattr(forecast_run.subprocess, "run", run)
    assert package_commit() == expected
    assert package_source_identity() == (expected, dirty)
    assert len(calls) == (4 if checkout else 2)


def test_validation_score_uses_raw_units_and_pre_validation_baseline(weekly_pipeline):
    p = weekly_pipeline
    val, holdout, _ = p.cutoffs()
    # Validation actuals differ sharply from the eight preceding weeks.
    p._weekly.loc[p._weekly.index < val, "qty"] = 2.0
    p._weekly.loc[(p._weekly.index >= val) & (p._weekly.index < holdout), "qty"] = 10.0
    p._weekly.loc[p._weekly.index >= holdout, "qty"] = 10000.0
    p.fit_preprocessing(val)
    p.model = ConstantModel()
    predictions = p.forecast(val)
    _, score, baseline = score_forecast(p, predictions, val)
    assert predictions.qty.eq(2).all()
    assert score == baseline == 80.0
    p._weekly.loc[(p._weekly.index >= val) & (p._weekly.index < holdout), "qty"] = 0.0
    _, score, baseline = score_forecast(p, predictions, val)
    assert score is baseline is None


@pytest.mark.parametrize("trials,budget", [(1, "2700"), (0, "0.001")])
def test_tiny_real_forecast_command(tmp_path, monkeypatch, trials, budget):
    # Eight products x 110 complete weeks, with a real CSV -> CLI -> files round trip.
    monkeypatch.setattr("demand_forecast.core.forecast_run.os.cpu_count", lambda: 1)
    trainers, forecasts = [], []
    original_train, original_forecast = ForecastPipeline.train, ForecastPipeline.forecast

    def train(pipeline, *args, **kwargs):
        trainer = original_train(pipeline, *args, **kwargs)
        trainers.append(trainer)
        return trainer

    def forecast(pipeline, origin):
        forecasts.append((origin, pipeline.cutoff, len(trainers)))
        return original_forecast(pipeline, origin)

    monkeypatch.setattr(ForecastPipeline, "train", train)
    monkeypatch.setattr(ForecastPipeline, "forecast", forecast)
    products = pd.DataFrame(
        {
            "id": [f"sku-{i}" for i in range(8)],
            "category": ["a", "b"] * 4,
            "subcategory": "s",
            "gender": "U",
            "brand": "b",
            "price": np.arange(8) + 10,
        }
    )
    products_path = tmp_path / "products.csv.gz"
    products.to_csv(products_path, index=False)
    days = pd.date_range(end="2026-10-11", periods=110 * 7)
    sales = pd.MultiIndex.from_product([products.id, days], names=["product_id", "date"]).to_frame(
        index=False
    )
    sales["qty"] = np.arange(len(sales)) % 5 + 1
    sales["unit_price"] = 10
    sales_path = tmp_path / "sales.csv"
    sales.to_csv(sales_path, index=False)
    settings = Settings.from_yaml(Path("configs/hackathon.yaml"))
    settings.tuning.epochs_per_trial = 1
    config = tmp_path / "config.yaml"
    settings.to_yaml(config)
    out = tmp_path / "run"
    result = CliRunner().invoke(
        app,
        [
            "forecast",
            "--config",
            str(config),
            "--products",
            str(products_path),
            "--sales",
            str(sales_path),
            "--out-dir",
            str(out),
            "--trials",
            str(trials),
            "--budget-seconds",
            budget,
            "--max-epochs",
            "3",
            "--device",
            "cpu",
            "--expect-origins",
            "2026-06-22,2026-10-12",
        ],
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    for filename, origin in [
        ("artifacts/validation_predictions.csv", "2026-03-02"),
        ("holdout_predictions.csv", "2026-06-22"),
        ("forecasts.csv", "2026-10-12"),
    ]:
        frame = pd.read_csv(out / filename)
        assert len(frame) == 128
        validate_forecast_contract(frame, products.id, origin)
    report = json.loads((out / "run.json").read_text())
    assert report["optuna_trials"] == trials
    assert report["best_epoch"] == report["refit_epochs_completed"]
    selection, refit = trainers
    assert refit.num_epochs == selection.best_epoch
    assert refit._schedulers["0"].T_max == selection.num_epochs == 3
    if trials == 0:
        assert refit.num_epochs == 1
    assert forecasts == [
        (pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-02"), 1),
        (pd.Timestamp("2026-06-22"), pd.Timestamp("2026-06-22"), 2),
        (pd.Timestamp("2026-10-12"), pd.Timestamp("2026-06-22"), 2),
    ]
    for key in ("wape_validation", "wape_validation_naive_last8"):
        assert key in report
        assert report[key] is None or np.isfinite(report[key])
    assert isinstance(report["source_dirty"], bool)
    assert report["wape_holdout"] >= 0
    assert report["budget_exceeded"] == (trials == 0)
    assert bool(report["tuning_fallback"]) == (trials == 0)
    assert report["device"]["device"] == "cpu"
    assert set(report["plots"]) == {str(p.relative_to(out)) for p in (out / "plots").glob("*.png")}
    assert (out / "plots/holdout_summary.txt").exists()
    assert (out / "plots/validation_epoch_best.png").exists()
    assert (out / "plots/quality_epoch_best.png").exists()
    assert any("fewer than 2" in note for note in report["notes"])
    assert {p.name for p in (out / "artifacts").iterdir()} >= {
        "model.pt",
        "scalers.joblib",
        "encoders.joblib",
        "optuna_trials.csv",
    }


def test_tuner_applies_sampled_batch_size(weekly_pipeline, monkeypatch):
    sizes = []

    class RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(0.0))

        def forward(self, qty, past_time, future_time, sku, cats, cont=None):
            if self.training:
                sizes.append(len(qty))
            return self.value.expand(len(qty), future_time.shape[1])

    monkeypatch.setattr(
        "demand_forecast.core.tuning.create_model", lambda *a, **kw: RecordingModel()
    )
    tuner = make_tuner(weekly_pipeline)
    tuner.search_space.batch_size = [2]
    params = tuner.tune(n_trials=1, show_progress_bar=False)
    assert params["batch_size"] == 2
    assert sizes and max(sizes) == 2


def test_catalogue_grid_coerces_numeric_sales_ids_and_rejects_no_match():
    from demand_forecast.config.settings import DataConfig
    from demand_forecast.data.preprocessor import catalogue_weekly_grid

    cfg = DataConfig(input_path="x.csv", sku_column="product_id", store_column=None)
    days = pd.date_range("2026-01-05", periods=21, freq="D")
    sales = pd.DataFrame({"product_id": 7, "qty": 1}, index=days)
    sales.index.name = "date"
    products = pd.DataFrame({"id": ["7"], "category": ["c"], "price": [1.0]})
    grid = catalogue_weekly_grid(sales, products, cfg)
    assert grid["qty"].sum() == 21 and set(grid["product_id"]) == {"7"}
    with pytest.raises(ValueError, match="No sales rows match"):
        catalogue_weekly_grid(sales, products.assign(id=["8"]), cfg)
