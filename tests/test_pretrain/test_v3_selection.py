"""Real-cell selection, EMA continuation and portable evaluation snapshots."""

import json
from contextlib import closing

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from typer.testing import CliRunner

pytest.importorskip("pyarrow")

from demand_forecast import metrics  # noqa: E402
from demand_forecast.cli import app  # noqa: E402
from demand_forecast.core.bundle import load_bundle  # noqa: E402
from demand_forecast.core.training import InterruptedRun  # noqa: E402
from demand_forecast.data.encoders import StaticEncoder  # noqa: E402
from demand_forecast.pretrain import config, validate  # noqa: E402
from demand_forecast.pretrain.config import PretrainConfig, pretrain_identity  # noqa: E402
from demand_forecast.pretrain.corpus import (  # noqa: E402
    corpus_manifest,
    shard_panel,
    write_manifest,
)
from demand_forecast.pretrain.sampler import V3ShardStream  # noqa: E402
from demand_forecast.pretrain.train import _PretrainLoop, export_pretrain, pretrain  # noqa: E402
from tests.test_pretrain.test_recipe_grids import _row, _shard  # noqa: E402


@pytest.fixture
def selection_cfg(tiny_pretrain_cfg, tmp_path):
    cfg = tiny_pretrain_cfg.model_dump()
    for kind in ("chronos_real", "gifteval_pretrain"):
        directory = tiny_pretrain_cfg.corpus.root / "training_corpus" / kind
        directory.mkdir(parents=True)
        for index in (0, 1):
            # Unbalanced frequency counts make macro vs micro distinguishable.
            path = _shard(
                tmp_path,
                [
                    _row(i, "D" if i == 0 else "h", target=(np.arange(70) ** 2 + i).tolist())
                    for i in range(4)
                ],
            )
            path.rename(directory / f"train-{index:05d}-of-00002.parquet")
    write_manifest(
        corpus_manifest(tiny_pretrain_cfg.corpus.root),
        tiny_pretrain_cfg.corpus.root / "manifest.json",
    )
    cfg.update(recipe="v3", period_policy="frequency", grid_policy="source")
    cfg["corpus"] = dict(
        root=tiny_pretrain_cfg.corpus.root,
        external_validation=None,
        val_pairs_per_kind=8,
        sources={
            kind: dict(shards=[0], holdout=[1], weight=1)
            for kind in ("tsmixup", "kernel_synth", "chronos_real", "gifteval_pretrain")
        },
        selection_shards={"chronos_real": [1], "gifteval_pretrain": [1]},
    )
    cfg["training"].update(
        max_steps=4, eval_every=2, checkpoint_every=1, ema_decay=0.5, keep_eval_snapshots=True
    )
    return PretrainConfig.model_validate(cfg)


def test_frozen_stratified_pairs_policy_identity_and_window(selection_cfg, monkeypatch):
    cfg = selection_cfg
    cfg.horizon_policy = "fev_set"
    cfg.forecast.horizon = 288
    cfg.forecast.min_context = 30
    cfg.corpus.val_pairs_per_kind = 64
    refs = [
        r for r in corpus_manifest(cfg.corpus.root) if r.kind == "chronos_real" and r.index == 1
    ]
    a = validate.selection_pairs_v3(refs, cfg)
    b = validate.selection_pairs_v3(list(reversed(refs)), cfg)
    assert [(c.key, c.items) for c in a] == [(c.key, c.items) for c in b]
    assert len(a) > 2 and sum(len(c.items) for c in a) <= 64
    assert max(len(c.items) for c in a) - min(len(c.items) for c in a) <= 2
    for cell in a:
        assert cell.horizon in config.FEV_HORIZONS
        for index, origin in cell.items:
            series = cell.group.panel.series[cell.group.row_indices.index(index)]
            assert origin >= max(30, cell.group.panel.seasonal_period + 1)
            assert origin + cell.horizon == series.length
    before = pretrain_identity(cfg, "2.0.0", {})
    monkeypatch.setitem(config.SELECTION_PAIR_POLICY, "version", 999)
    assert pretrain_identity(cfg, "2.0.0", {}) != before
    cfg.corpus.val_pairs_per_kind = 1
    with pytest.raises(ValueError, match="every available cell"):
        validate.selection_pairs_v3(refs, cfg)


class LastValue(torch.nn.Module):
    def predict(self, batch):
        value = batch.target[:, -1] * batch.scale + batch.loc
        return value[:, None, None].expand(-1, batch.labels.shape[1], 3)


def test_macro_cell_periods_and_nonfinite_cell_fails(selection_cfg):
    cfg = selection_cfg
    encoder = StaticEncoder().fit(shard_panel([]), [])
    refs = [
        r for r in corpus_manifest(cfg.corpus.root) if r.kind == "chronos_real" and r.index == 1
    ]
    cells = validate.selection_pairs_v3(refs, cfg)
    options = dict(
        max_context=16, encoder=encoder, quantiles=[0.1, 0.5, 0.9], device="cpu", batch_size=2
    )
    result = validate.evaluate_selection_v3(LastValue(), cells, **options)
    expected = []
    pooled = []
    for cell in cells:
        errors = []
        m = cell.group.panel.seasonal_period
        for index, origin in cell.items:
            series = cell.group.panel.series[cell.group.row_indices.index(index)]
            past = series.target[:origin]
            scale = metrics.seasonal_error_per_item([past], m)[0]
            errors.append(np.mean(np.abs(series.target[origin:] - past[-1])) / scale)
        expected.append(np.mean(errors))
        pooled.extend(errors)
        assert result["cells"][cell.key]["SQL"] == pytest.approx(expected[-1], rel=1e-6)
        assert result["cells"][cell.key]["period"] == m
    assert result["SQL"] == pytest.approx(np.mean(expected), rel=1e-6)
    assert result["SQL"] != pytest.approx(np.mean(pooled), rel=1e-3)
    cell = cells[0]
    for series in cell.group.panel.series:
        series.target[:] = 1
    failed = validate.evaluate_selection_v3(LastValue(), cells, **options)
    assert np.isnan(failed["SQL"])
    assert failed["failed_cells"] == [cell.key]
    assert len(failed["cells"]) == len(cells)


def test_source_mean_and_diagnostics_never_select(selection_cfg, monkeypatch):
    encoder = StaticEncoder().fit(shard_panel([]), [])
    validation = validate.ValidationSet.build(
        selection_cfg, corpus_manifest(selection_cfg.corpus.root), encoder
    )
    source = next(iter(validation.v3_sources.values()))
    validation.external = (shard_panel([]), np.array([], dtype=int))
    validation.v3_sources["external"] = source
    monkeypatch.setattr(
        validate,
        "evaluate_pairs_v3",
        lambda *a, **k: {"SQL": 10000.0, "naive_sql": 2.0, "n_excluded": 0},
    )
    values = iter([2.0, 6.0])
    monkeypatch.setattr(
        validate,
        "evaluate_selection_v3",
        lambda *a, **k: {"SQL": next(values), "n_items": 4, "cells": {}},
    )
    result = validation.evaluate(LastValue(), device="cpu")
    assert result["val_sql"] == 4
    assert result["val_sql_tsmixup"] == result["ext_sql_calendar_on"] == 10000.0
    assert result["ext_sql_calendar_off"] == 10000.0
    assert "ext_sql" not in result


@pytest.mark.parametrize("winner", ["raw", "ema"])
def test_ema_selection_records_choice_restores_live_and_ignores_diagnostics(selection_cfg, winner):
    loop = _PretrainLoop(selection_cfg)
    try:
        for value in loop.ema.shadow.values():
            if value.is_floating_point():
                value.add_(1)
        raw = {k: v.clone() for k, v in loop.model.state_dict().items()}
        for step in (1, 2):
            loop.step = step
            loop.history.append({"step": step})
            # Real selection worsens at step 2 while diagnostics improve.
            scores = [1.0, 2.0] if winner == "raw" else [2.0, 1.0]
            values = iter([score + 10 * (step - 1) for score in scores])
            loop.validation.evaluate = lambda *a, **k: {
                "val_sql": next(values),
                "ext_sql_calendar_on": 100.0 / step,
                "val_sql_tsmixup": 100.0 / step,
            }
            loop.evaluate()
        assert loop.best_step == 1 and loop.best_weights == winner
        assert loop.history[-1]["selected_weights"] == winner
        expected = raw if winner == "raw" else loop.ema.shadow
        assert all(torch.equal(v, expected[k]) for k, v in loop.best_state.items())
        assert all(torch.equal(v, raw[k]) for k, v in loop.model.state_dict().items())
    finally:
        loop.sampler.close()


def test_ema_exact_resume_snapshots_manifest_export_without_corpus(selection_cfg, tmp_path):
    full = pretrain(selection_cfg, tmp_path / "full")
    with pytest.raises(InterruptedRun):
        pretrain(selection_cfg, tmp_path / "part", stop_after=2)
    resumed = pretrain(selection_cfg, tmp_path / "part", resume=True)
    assert full.history == resumed.history
    assert (full.best_step, full.best_score) == (resumed.best_step, resumed.best_score)
    a, b = [
        torch.load(tmp_path / d / "checkpoints/latest.pt", weights_only=False)
        for d in ("full", "part")
    ]
    for key in ("model", "best_state", "ema_shadow"):
        assert all(torch.equal(v, b[key][k]) for k, v in a[key].items())
    assert a["sampler_state"] == b["sampler_state"]
    for name in ("feature_exposure", "sampling_exposure", "horizon_histogram"):
        assert a["timings"][name] == b["timings"][name]
    manifest = load_bundle(tmp_path / "part/bundle").manifest
    progress = json.loads((tmp_path / "part/checkpoints/progress.json").read_text())
    assert [r["step"] for r in manifest["eval_snapshots"]] == [2, 4]
    assert manifest["eval_snapshots"] == progress["eval_snapshots"]
    assert manifest["evaluations"] == progress["evaluations"]
    for row in manifest["evaluations"]:
        for score in row["selection_sources"].values():
            assert score["n_items"] == 4
            assert sum(c["n_items"] for c in score["cells"].values()) == 4
    # Export uses saved metadata only, even during an interrupted run.
    b["history"][b["best_step"] - 1].update(
        ext_sql_calendar_on=0.7,
        ext_sql_naive_calendar_on=1.2,
        ext_sql_calendar_off=0.8,
        ext_sql_naive_calendar_off=float("nan"),
    )
    torch.save(b, tmp_path / "part/checkpoints/latest.pt")
    selection_cfg.corpus.root.rename(tmp_path / "corpus-hidden")
    result = CliRunner().invoke(
        app,
        [
            "pretrain",
            "export",
            "--run-dir",
            str(tmp_path / "part"),
            "--step",
            "2",
            "--out",
            str(tmp_path / "export"),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    bundle = load_bundle(tmp_path / "export")
    snapshot = torch.load(tmp_path / "part/checkpoints/step-2.pt", weights_only=True)
    assert all(torch.equal(v, snapshot[k]) for k, v in bundle.model.state_dict().items())
    assert bundle.manifest["exported_step"] == 2
    assert bundle.manifest["exported_weights"] == "raw"
    for name in ("best_step", "best_score", "best_weights"):
        assert bundle.manifest[name] == b[name]
    record = b["history"][b["best_step"] - 1]
    assert bundle.manifest["external_diagnostic"] == {
        "name": selection_cfg.corpus.external_validation,
        **{k: v if np.isfinite(v) else None for k, v in record.items() if k.startswith("ext_sql")},
    }
    export_pretrain(tmp_path / "part", tmp_path / "ema-export", weights="ema")
    ema = load_bundle(tmp_path / "ema-export")
    assert all(torch.equal(v, b["ema_shadow"][k]) for k, v in ema.model.state_dict().items())
    with pytest.raises(ValueError, match="latest checkpoint"):
        export_pretrain(tmp_path / "part", tmp_path / "invalid", weights="ema", step=2)


def test_ema_only_successful_steps_and_min_context(selection_cfg, monkeypatch):
    loop = _PretrainLoop(selection_cfg)
    try:
        before = {k: v.clone() for k, v in loop.ema.shadow.items()}
        batch, batch_s, _ = loop.next_batch()
        loop.train_step(batch, batch_s)
        assert all(
            torch.equal(loop.ema.shadow[k], v * 0.5 + loop.model.state_dict()[k] * 0.5)
            for k, v in before.items()
            if v.is_floating_point()
        )
        before = {k: v.clone() for k, v in loop.ema.shadow.items()}
        monkeypatch.setattr(
            "demand_forecast.pretrain.train.pinball_loss",
            lambda *a, **k: (torch.tensor(float("nan")), 0),
        )
        loop.train_step(batch, 0.0)
        assert all(torch.equal(v, loop.ema.shadow[k]) for k, v in before.items())
    finally:
        loop.sampler.close()
    ref = next(r for r in corpus_manifest(selection_cfg.corpus.root) if r.kind == "chronos_real")
    with closing(
        V3ShardStream(
            [ref],
            "chronos_real",
            horizon=4,
            max_context=60,
            min_context=50,
            batch_size=12,
            seed=0,
            encoder=StaticEncoder().fit(shard_panel([]), []),
            prefetch=0,
            period_policy="frequency",
            grid_policy="source",
            scale_floor_frac=0.0,
        )
    ) as stream:
        assert next(stream).origin_pos.min() >= 50


def test_new_defaults_popped_and_controls_validated(tiny_pretrain_cfg):
    raw = tiny_pretrain_cfg.model_dump(mode="json")
    explicit = PretrainConfig.model_validate(
        raw | {"training": raw["training"] | {"ema_decay": None, "keep_eval_snapshots": False}}
    )
    assert explicit.model_dump(mode="json") == raw
    assert pretrain_identity(explicit, "2.0.0", {}) == pretrain_identity(
        tiny_pretrain_cfg, "2.0.0", {}
    )
    for controls in (
        {"ema_decay": 0.0},
        {"ema_decay": 1.0},
        {"ema_decay": float("nan")},
        {"ema_decay": 0.9},
        {"keep_eval_snapshots": True},
    ):
        with pytest.raises(ValidationError):
            PretrainConfig.model_validate(raw | {"training": raw["training"] | controls})
    base = PretrainConfig.model_validate(raw | {"recipe": "v3"})
    for controls in ({"ema_decay": 0.9}, {"keep_eval_snapshots": True}):
        changed = PretrainConfig.model_validate(
            raw | {"recipe": "v3", "training": raw["training"] | controls}
        )
        assert pretrain_identity(base, "2.0.0", {}) != pretrain_identity(changed, "2.0.0", {})


def test_v3_external_calendar_labels_and_adapter_masks(selection_cfg, monkeypatch):
    from demand_forecast.pretrain.sampler import V3BatchBuilder
    from tests.test_pretrain.test_validate import _external_fixture

    _external_fixture(selection_cfg.corpus.root)
    selection_cfg.corpus.external_validation = "monash_nn5_weekly"
    loop = _PretrainLoop(selection_cfg)
    seen = []
    build = V3BatchBuilder.build

    def observe(builder, items, **kwargs):
        batch = build(builder, items, **kwargs)
        if all(group.panel.series[0].series_id == "external" for group, _ in builder.builders):
            seen.append(batch)
        return batch

    monkeypatch.setattr(V3BatchBuilder, "build", observe)
    try:
        result = loop.validation.evaluate(loop.model, device="cpu")
        assert all(
            np.isfinite(result[key])
            for key in (
                "ext_sql_calendar_on",
                "ext_sql_calendar_off",
                "ext_sql_naive_calendar_on",
                "ext_sql_naive_calendar_off",
            )
        )
        assert "ext_sql" not in result and len(seen) == 2
        assert seen[0].enc_cov_mask[:, :, :16].any()
        assert not seen[1].enc_cov_mask[:, :, :16].any()
        assert not seen[1].dec_cov_mask[:, :, :16].any()
        assert torch.equal(seen[0].scale, seen[1].scale)
    finally:
        loop.sampler.close()


def test_ema_ignores_grad_scaler_skips(selection_cfg):
    class SkippedScaler:
        def __init__(self):
            self.value = 2.0

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            pass

        def get_scale(self):
            return self.value

        def update(self):
            self.value /= 2

        def is_enabled(self):
            return True

    loop = _PretrainLoop(selection_cfg)
    try:
        batch, batch_s, _ = loop.next_batch()
        loop.train_step(batch, batch_s)
        before = {key: value.clone() for key, value in loop.ema.shadow.items()}
        loop.scaler = SkippedScaler()
        batch, batch_s, _ = loop.next_batch()
        loop.train_step(batch, batch_s)
        assert loop.timings["scaler_skips"] == 1
        assert all(torch.equal(value, loop.ema.shadow[key]) for key, value in before.items())
    finally:
        loop.sampler.close()


def test_v2_checkpoint_schema_remains_legacy(tiny_pretrain_cfg, tmp_path):
    tiny_pretrain_cfg.training.max_steps = 2
    pretrain(tiny_pretrain_cfg, tmp_path / "v2")
    state = torch.load(tmp_path / "v2/checkpoints/latest.pt", weights_only=False)
    assert set(state) == {
        "identity",
        "step",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "rng",
        "sampler_state",
        "history",
        "best_state",
        "best_score",
        "best_step",
        "consecutive_nonfinite",
        "timings",
    }
