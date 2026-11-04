"""Probe the real CLI paths with only local fixture data."""

import json
import math
import time
import weakref
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from demand_forecast.cli import app

pyarrow = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.bench


def test_cli_pretrain_probe_reports_runtime_error(tmp_path, tiny_pretrain_yaml, monkeypatch):
    from demand_forecast.pretrain import probe

    def fail_probe(*args, **kwargs):
        raise RuntimeError("probe failed: nonfinite_losses=1")

    monkeypatch.setattr(probe, "pretrain_probe", fail_probe)
    result = CliRunner().invoke(
        app,
        [
            "pretrain",
            "probe",
            "--config",
            str(tiny_pretrain_yaml),
            "--out",
            str(tmp_path / "probe.json"),
        ],
    )
    assert result.exit_code == 1
    assert result.stderr == "probe failed: nonfinite_losses=1\n"
    assert result.stdout == ""


def test_cli_probe_task_reports_runtime_error(tmp_path, tiny_experiment_yaml, monkeypatch):
    from demand_forecast.pretrain import probe

    def fail_probe(*args, **kwargs):
        raise RuntimeError("task probe failed")

    monkeypatch.setattr(probe, "task_probe", fail_probe)
    result = CliRunner().invoke(
        app,
        [
            "pretrain",
            "probe-task",
            "--config",
            str(tiny_experiment_yaml),
            "--out",
            str(tmp_path / "task"),
        ],
    )
    assert result.exit_code == 1
    assert result.stderr == "task probe failed\n"
    assert result.stdout == ""


def test_cli_pretrain_run_and_probe(tmp_path, tiny_pretrain_yaml):
    out = tmp_path / "o"
    probe = tmp_path / "probe.json"
    runner = CliRunner()
    args = ["pretrain", "run", "--config", str(tiny_pretrain_yaml), "--out", str(out)]
    interrupted = runner.invoke(app, [*args, "--stop-after", "5"])
    assert interrupted.exit_code == 1, interrupted.output
    result = runner.invoke(app, [*args, "--resume"])
    assert result.exit_code == 0, (result.output, result.exception)
    result = runner.invoke(
        app,
        [
            "pretrain",
            "probe",
            "--config",
            str(tiny_pretrain_yaml),
            "--out",
            str(probe),
            "--steps",
            "3",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    report = json.loads(probe.read_text())
    for key in (
        "s_per_step",
        "batch_build_s_per_step",
        "init_s",
        "shard_load_s",
        "checkpoint_s",
        "eval_s",
        "peak_rss_bytes",
        "params",
    ):
        assert report[key] > 0, key
    assert report["peak_cuda_mem_bytes"] is report["eval_peak_cuda_mem_bytes"] is None
    assert report["nonfinite_losses"] == 0
    assert math.isfinite(report["val_sql"])


@pytest.mark.parametrize("nan_step", [1, 11], ids=["warmup", "measurement"])
def test_pretrain_probe_rejects_nonfinite_losses(
    tmp_path, tiny_pretrain_cfg, monkeypatch, nan_step
):
    from demand_forecast.pretrain import train
    from demand_forecast.pretrain.probe import pretrain_probe

    pinball_loss = train.pinball_loss
    calls = 0

    def fail_one_loss(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == nan_step:
            return torch.tensor(float("nan"), requires_grad=True), 1
        return pinball_loss(*args, **kwargs)

    monkeypatch.setattr(train, "pinball_loss", fail_one_loss)
    out = tmp_path / "failed-probe.json"
    warning = (
        pytest.warns(UserWarning, match="lr_scheduler.step") if nan_step == 1 else nullcontext()
    )
    with warning, pytest.raises(RuntimeError, match="nonfinite_losses=1"):
        pretrain_probe(tiny_pretrain_cfg, steps=3, out=out)
    assert not out.exists()


def test_pretrain_probe_rejects_nonfinite_validation(tmp_path, tiny_pretrain_cfg):
    from demand_forecast.pretrain.corpus import corpus_manifest, write_manifest
    from demand_forecast.pretrain.probe import pretrain_probe
    from tests.helpers.corpus_fixture import write_corpus_fixture

    cfg = tiny_pretrain_cfg
    for kind in ("tsmixup", "kernel_synth"):
        write_corpus_fixture(cfg.corpus.root, kind=kind, index=1, constant=True)
    write_manifest(corpus_manifest(cfg.corpus.root), cfg.corpus.root / "manifest.json")
    out = tmp_path / "unscored-probe.json"
    with pytest.raises(RuntimeError, match="val_sql=nan"):
        pretrain_probe(cfg, steps=3, out=out)
    assert not out.exists()


def test_cli_probe_task_on_fixture_task(
    tmp_path, fixture_tasks_yaml, fixture_profiles_yaml, tiny_experiment_yaml, monkeypatch
):
    from demand_forecast.core import evaluation, training
    from demand_forecast.data import loader_fev
    from demand_forecast.inference.predictor import Predictor
    from demand_forecast.models.base import ForecastModel
    from demand_forecast.pretrain import probe

    # Observe real work: each update must finish before its per-step timer ends.
    events, contexts, prepared_batches, training_batches = [], [], [], []
    trainers = []
    sync, update, forward = probe._sync, torch.optim.AdamW.step, ForecastModel.forward
    prepare, panel_from_window, fit = (
        evaluation.prepare_validation,
        loader_fev.panel_from_window,
        training.Trainer.fit,
    )
    infer = Predictor.forecast_per_item_origins

    def observe_sync(device):
        sync(device)
        events.append("sync")

    def observe_update(self, *args, **kwargs):
        result = update(self, *args, **kwargs)
        events.append("update")
        return result

    def observe_forward(self, batch):
        events.append("forward")
        return forward(self, batch)

    def observe_prepare(*args, **kwargs):
        prepared_batches.append(kwargs["batch_size"])
        return prepare(*args, **kwargs)

    def observe_panel(*args, **kwargs):
        panel = panel_from_window(*args, **kwargs)
        contexts.append([series.history_length for series in panel.series])
        return panel

    def observe_fit(self, *args, **kwargs):
        training_batches.append(self.cfg.batch_size)
        trainers.append(weakref.ref(self))
        return fit(self, *args, **kwargs)

    def observe_infer(self, *args, **kwargs):
        assert all(ref() is None for ref in trainers), "inference retains training state"
        return infer(self, *args, **kwargs)

    monkeypatch.setattr(probe, "_sync", observe_sync)
    monkeypatch.setattr(torch.optim.AdamW, "step", observe_update)
    monkeypatch.setattr(ForecastModel, "forward", observe_forward)
    monkeypatch.setattr(evaluation, "prepare_validation", observe_prepare)
    monkeypatch.setattr(loader_fev, "panel_from_window", observe_panel)
    monkeypatch.setattr(training.Trainer, "fit", observe_fit)
    monkeypatch.setattr(Predictor, "forecast_per_item_origins", observe_infer)
    out = tmp_path / "task"
    result = CliRunner().invoke(
        app,
        [
            "pretrain",
            "probe-task",
            "--tasks",
            str(fixture_tasks_yaml),
            "--profiles",
            str(fixture_profiles_yaml),
            "--config",
            str(tiny_experiment_yaml),
            "--out",
            str(out),
            "--steps",
            "2",
            "--max-series",
            "6",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    report = json.loads((out / "fixture.json").read_text())
    for key in (
        "s_per_step",
        "eval_s",
        "infer_s",
        "init_s",
        "prep_s",
        "ckpt_s",
        "init_last_s",
        "peak_rss_bytes",
    ):
        assert report[key] > 0, key
    assert report["n_fit_ids"] == report["n_val_items"] == 6
    assert report["windows"] == 2
    assert report["resolved_max_steps"] == 20 and report["resolved_eval_every"] == 4
    assert report["infer_peak_cuda_mem_bytes"] is report["val_peak_cuda_mem_bytes"] is None
    assert contexts == [[32] * 6, [36] * 6]
    assert prepared_batches == [512] and training_batches == [128]
    assert events.count("update") == 2
    assert all(events[i + 1] == "sync" for i, event in enumerate(events) if event == "update")


def test_task_probe_includes_trainer_setup_in_init(
    tmp_path, fixture_tasks_yaml, fixture_profiles_yaml, tiny_experiment_yaml, monkeypatch
):
    from demand_forecast.bench.specs import load_task_specs
    from demand_forecast.config import Settings
    from demand_forecast.core import training
    from demand_forecast.pretrain.probe import task_probe

    build_origins = training.build_origins

    def delayed_origins(*args, **kwargs):
        time.sleep(0.2)
        return build_origins(*args, **kwargs)

    monkeypatch.setattr(training, "build_origins", delayed_origins)
    out = tmp_path / "setup-probe"
    reports = task_probe(
        load_task_specs(str(fixture_tasks_yaml), fixture_profiles_yaml),
        Settings.from_yaml(tiny_experiment_yaml),
        steps=2,
        out=out,
        max_series=6,
    )
    report = reports["fixture"]
    assert report["fit_setup_s"] >= 0.2
    assert report["init_s"] >= report["fit_setup_s"]
    assert json.loads((out / "fixture.json").read_text())["fit_setup_s"] == report["fit_setup_s"]


def test_pretrain_duration_counts_overheads(tiny_pretrain_cfg):
    from demand_forecast.pretrain.train import pretrain_duration

    cfg = tiny_pretrain_cfg.model_copy(deep=True)
    cfg.training.max_steps = 1000
    cfg.training.eval_every = 500
    cfg.training.checkpoint_every = 250
    cfg.training.batch_size = 100
    probe = dict(init_s=1, s_per_step=2, eval_s=3, checkpoint_s=4, shard_load_s=5)
    result = pretrain_duration(cfg, probe)
    assert result["n_evals"] == 2 and result["n_ckpts"] == 4
    assert result["n_shard_loads"] == 2 + 2 + 1
    assert result["steps_ts"] == 900 and result["steps_ks"] == 100
    assert result["total_s"] == pytest.approx((1 + 2000 + 6 + 16 + 25 + 300) * 1.15)
    assert result["minutes"] == math.ceil(result["total_s"] / 60)
    cfg.corpus.kernel_synth_every = 0
    assert pretrain_duration(cfg, probe)["steps_ks"] == 0


@pytest.mark.parametrize("series_embedding", [False, True])
def test_task_probe_honours_series_embedding(tmp_path, monkeypatch, series_embedding):
    from demand_forecast.core import evaluation
    from demand_forecast.pretrain.probe import task_probe
    from tests.helpers.fev_fixture import write_fev_fixture
    from tests.helpers.fev_spec import fixture_spec
    from tests.helpers.settings import tiny_settings

    spec = fixture_spec(write_fev_fixture(tmp_path / "data"))
    settings = tiny_settings(base_steps=2, d_model=8)
    settings.training.series_embedding = series_embedding
    prepare = evaluation.prepare_validation
    encoders = []

    def observe_prepare(panel, items, origin, horizon, max_context, encoder, **kwargs):
        encoders.append(encoder)
        return prepare(panel, items, origin, horizon, max_context, encoder, **kwargs)

    monkeypatch.setattr(evaluation, "prepare_validation", observe_prepare)
    task_probe([spec], settings, steps=2, out=tmp_path / "out")
    assert len(encoders) == 1
    assert ("__series__" in encoders[0].vocabularies) == series_embedding
    assert encoders[0].vocabularies["cat"] == {"A": 2, "B": 3}


@pytest.mark.parametrize("heads_only", [False, True], ids=["default", "heads_only"])
def test_task_probe_honours_trainable(tmp_path, monkeypatch, heads_only):
    from demand_forecast.core import training
    from demand_forecast.pretrain.probe import task_probe
    from tests.helpers.fev_fixture import write_fev_fixture
    from tests.helpers.fev_spec import fixture_spec
    from tests.helpers.settings import tiny_settings

    spec = fixture_spec(write_fev_fixture(tmp_path / "data"))
    settings = tiny_settings(base_steps=2, d_model=8)
    if heads_only:
        settings.training.trainable = "heads_only"
    fit = training.Trainer.fit
    counts = []

    def observe_fit(self, *args, **kwargs):
        backbone = list(self.model.backbone.parameters())
        assert backbone
        assert all(p.requires_grad == (not heads_only) for p in backbone)
        backbone_ids = {id(p) for p in backbone}
        expected = [
            p for p in self.model.parameters() if not heads_only or id(p) not in backbone_ids
        ]
        assert expected and all(p.requires_grad for p in expected)
        result = fit(self, *args, **kwargs)
        assert result.steps == 2
        optimized = [p for group in self.optimizer.param_groups for p in group["params"]]
        assert {id(p) for p in optimized} == {id(p) for p in expected}
        counts.append(
            (sum(p.numel() for p in optimized), sum(p.numel() for p in self.model.parameters()))
        )
        return result

    monkeypatch.setattr(training.Trainer, "fit", observe_fit)
    out = tmp_path / "out"
    reports = task_probe([spec], settings, steps=2, out=out)
    assert len(counts) == 1
    trainable, total = counts[0]
    assert (trainable < total) if heads_only else (trainable == total)
    report = reports[spec.name]
    assert report["params_trainable"] == trainable
    assert report["params_total"] == total
    assert json.loads((out / f"{spec.name}.json").read_text()) == report


def test_task_probe_honours_validation_scale_floor(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from demand_forecast.core import evaluation
    from demand_forecast.pretrain.probe import task_probe
    from tests.helpers.fev_fixture import write_fev_fixture
    from tests.helpers.fev_spec import fixture_spec
    from tests.helpers.settings import tiny_settings

    parquet = write_fev_fixture(tmp_path / "data")
    rows = pq.read_table(parquet).to_pylist()
    for row in rows:
        row["target"] = [1000.0 + i % 2 for i in range(40)]
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    settings = tiny_settings(base_steps=2, d_model=8)
    settings.forecast.scale_floor_frac = 0.1
    prepare = evaluation.prepare_validation
    scales, expected = [], []

    def observe_prepare(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        scales.extend(batch.scale.numpy().reshape(-1) for batch in prepared.batches)
        expected.extend(
            0.1 * np.mean(np.abs(past[-settings.forecast.max_context :]))
            for past in prepared.y_past
        )
        return prepared

    monkeypatch.setattr(evaluation, "prepare_validation", observe_prepare)
    task_probe([fixture_spec(parquet)], settings, steps=2, out=tmp_path / "out")
    assert len(expected) == 6
    assert min(expected) > 100  # The unfloored seasonal scale is exactly 1.
    np.testing.assert_allclose(np.concatenate(scales), expected)
