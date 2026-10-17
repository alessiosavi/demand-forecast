"""Tiny synthetic checks for the same-data comparison (no real catalogue required)."""

import json
import re
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal
from typer.testing import CliRunner

from demand_forecast.cli import app
from demand_forecast.config.settings import ModelConfig, Settings
from demand_forecast.core.benchmark import (
    MODEL_TYPES,
    PLOT_NAMES,
    aligned_scores,
    baseline_forecast,
    compare,
    comparison_history,
    example_products,
    model_settings,
    performance,
    run_benchmark,
)
from demand_forecast.core.forecast_run import (
    network_latency,
    pooled_loader,
    validate_forecast_contract,
)
from demand_forecast.core.pipeline import ForecastPipeline
from demand_forecast.core.tuning import (
    HyperparameterTuner,
    SearchSpace,
    TuningConfig,
    suggest_hyperparameters,
)
from demand_forecast.models.losses import point_forecast


@pytest.fixture
def tiny_settings(tmp_path, monkeypatch):
    # Three products, including a zero-demand category; 110 weeks on the contract grid.
    # One CPU thread, one epoch, one batch, width 8. Never load repository datasets.
    monkeypatch.setattr("demand_forecast.core.forecast_run.os.cpu_count", lambda: 1)
    products = pd.DataFrame(
        {
            "id": ["sku-a", "sku-b", "zero"],
            "category": ["tops", "shoes", "zero"],
            "subcategory": "s",
            "gender": "U",
            "brand": "b",
            "price": [10, 20, 30],
        }
    )
    days = pd.date_range(end="2026-10-11", periods=110 * 7)
    sales = pd.MultiIndex.from_product([products.id, days], names=["product_id", "date"]).to_frame(
        index=False
    )
    sales["qty"] = np.where(sales.product_id == "zero", 0, np.arange(len(sales)) % 7 + 1)
    products.to_csv(tmp_path / "products.csv", index=False)
    sales.to_csv(tmp_path / "sales.csv", index=False)
    settings = Settings.from_yaml(Path("configs/benchmark-w26.yaml"))
    settings.data.products_path = tmp_path / "products.csv"
    settings.data.input_path = tmp_path / "sales.csv"
    settings.device = "cpu"
    settings.model.d_model = 8
    settings.model.nhead = 2
    settings.model.sku_emb_dim = 4
    settings.model.cat_emb_dims = 4
    settings.model.hidden_continuous_size = 8
    settings.model.hidden_dim = 8
    settings.model.tcn_channels = [4, 8]
    settings.model.num_encoder_layers = 1
    settings.model.num_decoder_layers = 1
    settings.model.dim_feedforward = 16
    settings.model.dropout = 0
    settings.training.num_epochs = 1
    settings.training.batch_size = 128
    settings.tuning.n_trials = 1
    settings.tuning.epochs_per_trial = 1
    return settings


def prepare(settings, family):
    pipeline = ForecastPipeline(model_settings(settings, family))
    pipeline.load_data()
    pipeline.fit_preprocessing(pipeline.cutoffs()[0])
    pipeline.create_datasets()
    return pipeline


def fixed_space():
    return SearchSpace(**{f.name: None for f in fields(SearchSpace)})


@pytest.mark.parametrize("family", MODEL_TYPES)
def test_model_static_paths_and_gradients(tiny_settings, family):
    pipeline = prepare(tiny_settings, family)
    wrapper = pipeline.build_model().eval()
    batch = next(iter(pipeline.dataloaders()[1]["0"]))
    inputs = {key: value for key, value in batch.items() if key != "y"}
    output = point_forecast(wrapper("0", **inputs))
    assert output.shape == (3, 16)
    assert batch["qty"].shape[1] == 26
    loss = torch.nn.functional.mse_loss(output, batch["y"])
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in wrapper.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert wrapper.models["0"].cont_proj.weight.grad.abs().sum() > 0
    with torch.no_grad():
        # Use the same inference fast path on both sides of the legacy comparison.
        output = point_forecast(wrapper("0", **inputs))
        changed = point_forecast(wrapper("0", **{**inputs, "cont": inputs["cont"] + 3}))
        assert not torch.allclose(output, changed)
        legacy = {key: value[:, None].repeat(1, 2) for key, value in inputs["cats"].items()}
        assert torch.allclose(output, point_forecast(wrapper("0", **{**inputs, "cats": legacy})))
    with pytest.raises(ValueError, match="Continuous static"):
        wrapper("0", **{key: value for key, value in inputs.items() if key != "cont"})


@pytest.mark.parametrize("family", MODEL_TYPES)
def test_family_tuner_trainer_evaluator_forecast_reload(tiny_settings, tmp_path, family):
    pipeline = prepare(tiny_settings, family)
    train, validation, _, _ = pipeline.dataloaders()
    tuner = HyperparameterTuner(
        family,
        pooled_loader(train, 128, True, 42),
        pooled_loader(validation, 128, False, 42),
        pipeline.model_kwargs(),
        config=TuningConfig(n_trials=1),
        search_space=fixed_space(),
        num_epochs_per_trial=1,
    )
    resolved = tuner.tune(show_progress_bar=False)
    assert len(tuner.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) == 1
    assert np.isfinite(tuner.study.best_value)
    assert resolved["cont_features_dim"] == 1
    pipeline.build_model()
    trainer = pipeline.train()
    assert trainer.best_epoch == 1
    assert all(np.isfinite(trainer.history[key]).all() for key in ("train_loss", "val_loss"))
    prediction = pipeline.forecast(pipeline.cutoffs()[1])
    validate_forecast_contract(prediction, pipeline.products.id, pipeline.cutoffs()[1])
    path = tmp_path / family / "model.pt"
    pipeline.save(path)
    restored = ForecastPipeline(pipeline.settings.model_copy(deep=True))
    restored.load(path)
    restored.load_and_preprocess()
    assert restored.model.model_type == family
    assert_frame_equal(prediction, restored.forecast(restored.cutoffs()[1]))


def test_point_forecast_and_standard_constructor_unchanged(tiny_settings):
    value = torch.ones(2, 16)
    assert point_forecast(value) is value
    assert point_forecast({"prediction": value, "quantiles": torch.zeros(2, 16, 3)}) is value
    pipeline = prepare(tiny_settings, "standard")
    assert pipeline.model_kwargs() == {
        "sku_vocab_size": 3,
        "cat_features_dim": pipeline.categorical_encoder.vocab_sizes,
        "sku_emb_dim": 4,
        "cat_emb_dims": 4,
        "d_model": 8,
        "nhead": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "use_rope": False,
        "use_pre_layernorm": False,
        "use_film_conditioning": False,
        "use_improved_head": False,
        "stochastic_depth_rate": 0.0,
        "cont_features_dim": 1,
        "past_time_features_dim": 5,
        "future_time_features_dim": 4,
        "n_out": 16,
        "max_past_len": 26,
        "max_future_len": 16,
    }
    original = Settings.from_yaml(Path("configs/hackathon-w26.yaml"))
    benchmark = Settings.from_yaml(Path("configs/benchmark-w26.yaml"))
    first, second = ForecastPipeline(original), ForecastPipeline(benchmark)
    assert first.model_kwargs() == second.model_kwargs()
    original.model = benchmark.model
    assert original == benchmark


@pytest.mark.parametrize(
    "family,knob,choices",
    [
        ("standard", "d_model", [64, 128]),
        ("advanced", "d_model", [64, 128]),
        ("lightweight_tcn", "tcn_width", [32, 64]),
        ("lightweight_mixer", "hidden_dim", [64, 128]),
    ],
)
def test_equal_search_caps_and_family_capacity(family, knob, choices):
    study = optuna.create_study()
    trial = study.ask()
    space = SearchSpace.hackathon(family)
    params = suggest_hyperparameters(
        trial, space, family, fixed_nhead=4 if family in ("standard", "advanced") else 3
    )
    assert set(trial.params) == {knob, "learning_rate", "dropout", "weight_decay"}
    assert list(trial.distributions[knob].choices) == choices
    assert space.learning_rate == (1e-4, 3e-3)
    assert space.dropout == (0.05, 0.3)
    assert space.weight_decay == (1e-5, 1e-2)
    if family == "lightweight_tcn":
        width = trial.params[knob]
        assert params["tcn_channels"] == [width, 2 * width, 4 * width]
    if family.startswith("lightweight"):
        assert "d_model" not in params and "nhead" not in params
        ModelConfig(model_type="lightweight", d_model=7, nhead=3)
    else:
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(model_type=family, d_model=7, nhead=3)


def test_construction_failure_is_failed_trial(tiny_settings, monkeypatch):
    pipeline = prepare(tiny_settings, "advanced")
    train, val, _, _ = pipeline.dataloaders()

    def fail(*args, **kwargs):
        raise ValueError("deliberate constructor failure")

    monkeypatch.setattr("demand_forecast.core.tuning.create_model", fail)
    tuner = HyperparameterTuner(
        "advanced",
        train["0"],
        val["0"],
        pipeline.model_kwargs(),
        search_space=fixed_space(),
        num_epochs_per_trial=1,
    )
    tuner.tune(n_trials=1, show_progress_bar=False)
    assert tuner.study.trials[0].state == optuna.trial.TrialState.FAIL
    assert tuner.study.trials[0].user_attrs["error"] == "deliberate constructor failure"
    assert tuner.fallback_reason


def test_grouped_scores_and_exact_seasonal_week(tiny_settings):
    pipeline = ForecastPipeline(tiny_settings)
    pipeline.load_data()
    weekly = comparison_history(pipeline)
    origin = pipeline.cutoffs()[1]
    ids = pipeline.products.id.tolist()
    frame = baseline_forecast(weekly, ids, origin, "seasonal_naive")
    expected = weekly.set_index(["product_id", "week_start"]).actual.loc[
        ("sku-a", origin - pd.Timedelta(weeks=52))
    ]
    assert frame.iloc[0].qty == expected
    short = weekly[weekly.week_start >= origin - pd.Timedelta(weeks=51)]
    with pytest.raises(ValueError, match="finite"):
        baseline_forecast(short, ids, origin, "seasonal_naive")
    scored = aligned_scores(frame, weekly)
    scored.loc[scored.product_id == "sku-a", "qty"] += 10
    metrics = performance(scored, scored)
    assert metrics["wape_holdout"] == pytest.approx(100 * 160 / scored.actual.sum())
    assert metrics["holdout_bias_percent"] == pytest.approx(metrics["wape_holdout"])
    assert metrics["wape_by_category"]["zero"] is None
    assert set(metrics["wape_by_horizon"]) == {str(i) for i in range(1, 17)}
    before = example_products(weekly, origin)
    weekly.loc[weekly.week_start >= origin, "actual"] = 1e9
    assert example_products(weekly, origin) == before
    scored["actual"] = 0
    metrics = performance(scored, scored)
    assert metrics["wape_holdout"] is metrics["holdout_bias_percent"] is None


def test_latency_fixed_batch_and_warmup():
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, cluster, **batch):
            self.calls.append((cluster, batch["qty"].shape, batch["qty"][:4, 0, 0].tolist()))
            return torch.zeros(512, 16)

    model = Network()
    batch = {
        "qty": torch.arange(3).reshape(3, 1, 1),
        "cats": {"a": torch.zeros(3)},
        "y": torch.zeros(3, 16),
    }
    pipeline = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        forecast_batches=lambda *a, **k: iter([("0", [], [batch])]),
    )
    result = network_latency(pipeline, "2026-06-22")
    assert result["network_latency_ms"] >= 0
    assert result["network_latency_protocol"]["unique_inputs"] == 3
    assert len(model.calls) == 25
    assert all(shape == (512, 1, 1) and values == [0, 1, 2, 0] for _, shape, values in model.calls)


def test_tiny_benchmark_cli_roundtrip_and_provenance(tiny_settings, tmp_path, monkeypatch):
    # Exercise real tuning/training/export/plots. Latency mechanics are tested separately
    # with a trivial network so the local test never runs 25 large neural batches.
    monkeypatch.setattr(
        SearchSpace, "hackathon", classmethod(lambda cls, model_type="standard": fixed_space())
    )
    monkeypatch.setattr(
        "demand_forecast.core.forecast_run.network_latency", lambda *a: {"network_latency_ms": 0.01}
    )
    config = tmp_path / "config.yaml"
    tiny_settings.to_yaml(config)
    out = tmp_path / "benchmark"
    args = [
        "benchmark",
        "--config",
        str(config),
        "--products",
        str(tiny_settings.data.products_path),
        "--sales",
        str(tiny_settings.data.input_path),
        "--out-dir",
        str(out),
        "--models",
        "standard,lightweight_mixer",
        "--device",
        "cpu",
        "--trials",
        "1",
        "--max-epochs",
        "1",
        "--expect-origins",
        "2026-06-22,2026-10-12",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    report = json.loads((out / "benchmark.json").read_text())
    assert (out / "benchmark.md").is_file()
    assert all((out / "plots" / f"{name}.png").stat().st_size > 0 for name in PLOT_NAMES)
    for name in ("standard", "lightweight_mixer"):
        r = report["results"][name]
        assert r["status"] == "ok", r
        assert r["trial_counts"] == {"completed": 1, "pruned": 0, "failed": 0}
        assert r["best_epoch"] == r["refit_epochs_completed"] == 1
        assert r["train_seconds"] > 0 and r["forecast_pipeline_seconds"] > 0
        for filename, origin in (
            ("validation_predictions.csv", "2026-03-02"),
            ("holdout_predictions.csv", "2026-06-22"),
            ("forecasts.csv", "2026-10-12"),
        ):
            validate_forecast_contract(
                pd.read_csv(out / "models" / name / filename), report["product_ids"], origin
            )
    metadata_path = out / "models/standard/run.json"
    original = json.loads(metadata_path.read_text())
    changed = {**original, "wape_holdout": -12345, "wape_validation": -12345}
    metadata_path.write_text(json.dumps(changed))
    monkeypatch.setattr(
        "demand_forecast.core.benchmark.run_forecast",
        lambda *a, **k: pytest.fail("compare-only trained"),
    )
    second = CliRunner().invoke(app, args + ["--compare-only"])
    assert second.exit_code == 0, f"{second.output}\n{second.exception}"
    assert json.loads((out / "benchmark.json").read_text()) == report
    # Different provenance and malformed/missing CSV keys must fail closed.
    for key, bad in (
        ("seed", 999),
        ("product_ids", ["other"]),
        ("inputs_sha256", {}),
        ("holdout_from", "2026-06-29"),
        ("protocol", {}),
    ):
        metadata_path.write_text(json.dumps({**original, key: bad}))
        with pytest.raises(ValueError, match=key):
            compare(tiny_settings, out, ("standard", "lightweight_mixer"))
    metadata_path.write_text(json.dumps(original))
    csv = out / "models/standard/holdout_predictions.csv"
    pd.read_csv(csv).iloc[1:].to_csv(csv, index=False)
    with pytest.raises(ValueError, match="keys"):
        compare(tiny_settings, out, ("standard", "lightweight_mixer"))


def test_settings_isolation_failure_and_incomplete(tiny_settings, tmp_path, monkeypatch):
    # A stub isolates orchestration; the two-family test above exercises real artifacts.
    pristine = tiny_settings.model_dump(mode="json")
    calls = []
    from demand_forecast.core.forecast_run import forecast_provenance

    def fake_run(settings, directory, **kwargs):
        calls.append(
            (
                settings.model.model_type,
                settings.model.lightweight_variant,
                settings.training.num_epochs,
                kwargs["budget_seconds"],
            )
        )
        assert settings.training.num_epochs == 1
        if settings.model.model_type == "advanced":
            settings.training.num_epochs = 99
            raise ValueError("forced model failure")
        pipeline = ForecastPipeline(settings)
        pipeline.load_data()
        provenance = forecast_provenance(pipeline, settings.budget_seconds)
        requested = settings.model_dump(mode="json")
        weekly = comparison_history(pipeline)
        directory.mkdir(parents=True)
        for filename, origin in (
            ("validation_predictions.csv", "validation_from"),
            ("holdout_predictions.csv", "holdout_from"),
            ("forecasts.csv", "origin_future"),
        ):
            frame = baseline_forecast(
                weekly, provenance["product_ids"], provenance[origin], "seasonal_naive"
            )
            frame.to_csv(directory / filename, index=False)
        (directory / "run.json").write_text(
            json.dumps(
                {
                    **provenance,
                    "model_type": pipeline.resolved_model_type(),
                    "requested_settings": requested,
                    "status": "ok",
                    "best_epoch": 2,
                    "refit_epochs_completed": 1,
                    "trial_counts": {"completed": 1, "pruned": 0, "failed": 0},
                }
            )
        )
        settings.training.num_epochs = 99

    monkeypatch.setattr("demand_forecast.core.benchmark.run_forecast", fake_run)
    monkeypatch.setattr("demand_forecast.core.benchmark.comparison_plots", lambda *a: None)
    out = tmp_path / "isolation"
    report = run_benchmark(tiny_settings, out, ("advanced", "lightweight_tcn", "lightweight_mixer"))
    assert tiny_settings.model_dump(mode="json") == pristine
    assert len(calls) == 3
    assert all(call[3] == tiny_settings.budget_seconds for call in calls)
    assert report["results"]["advanced"]["status"] == "error"
    assert report["results"]["lightweight_tcn"]["status"] == "incomplete"
    assert report["best_neural_model"] is None
    assert all(name not in report["ranking"] for name in MODEL_TYPES)
    path = out / "models/lightweight_mixer/run.json"
    metadata = json.loads(path.read_text())
    metadata["refit_epochs_completed"] = 2
    metadata["trial_counts"]["completed"] = 0
    path.write_text(json.dumps(metadata))
    report = compare(tiny_settings, out, ("lightweight_mixer",))
    assert report["results"]["lightweight_mixer"]["status"] == "incomplete"


def test_benchmark_help():
    result = CliRunner().invoke(
        app, ["benchmark", "--help"], env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"}
    )
    assert result.exit_code == 0
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.output)
    for option in (
        "--models",
        "--compare-only",
        "--budget-seconds",
        "--max-epochs",
        "--expect-origins",
    ):
        assert option in output
