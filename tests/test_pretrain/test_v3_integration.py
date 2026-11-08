"""CPU config-to-transfer smoke and measured recipe planning, with no Hub access."""

import json
from dataclasses import asdict

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from demand_forecast.cli import app
from demand_forecast.core.bundle import load_bundle
from demand_forecast.data.frequency import CANONICAL_CALENDAR_SCHEMA
from demand_forecast.data.sampler import TARGET_LAG_SCHEMA
from demand_forecast.pretrain.config import FEV_HORIZONS, PretrainConfig
from tests.helpers.settings import tiny_settings

pytest.importorskip("pyarrow")
pytest.importorskip("fev")
pytestmark = pytest.mark.bench


@pytest.fixture
def v3_config(tmp_path, tiny_pretrain_cfg):
    from demand_forecast.pretrain.corpus import corpus_manifest, write_manifest
    from tests.helpers.corpus_fixture import write_corpus_fixture
    from tests.test_pretrain.test_recipe_grids import _row, _shard

    root = tiny_pretrain_cfg.corpus.root
    for kind in ("tsmixup", "kernel_synth"):
        for index in (0, 1):
            write_corpus_fixture(root, kind=kind, index=index, lengths=(360, 370), n_series=4)
    real = root / "training_corpus/chronos_real"
    real.mkdir()
    for index in (0, 1):
        _shard(
            tmp_path, [_row(i, target=(10 + np.arange(370) ** 1.1 + i).tolist()) for i in range(4)]
        ).rename(real / f"train-{index:05d}-of-00002.parquet")
    write_manifest(corpus_manifest(root), root / "manifest.json")
    raw = tiny_pretrain_cfg.model_dump()
    raw.update(
        recipe="v3", period_policy="frequency", grid_policy="source", horizon_policy="fev_set"
    )
    raw["forecast"].update(horizon=288)
    raw["model"]["d_model"] = 8
    raw["corpus"] = dict(
        root=root,
        external_validation=None,
        val_pairs_per_kind=16,
        sources={
            kind: dict(shards=[0], holdout=[1], weight=1)
            for kind in ("tsmixup", "kernel_synth", "chronos_real")
        },
        selection_shards={"chronos_real": [1]},
    )
    raw["training"].update(
        max_steps=6,
        batch_size=2,
        val_batch_size=4,
        eval_every=3,
        checkpoint_every=3,
        ema_decay=0.5,
        keep_eval_snapshots=True,
    )
    return PretrainConfig.model_validate(raw)


def test_v3_config_run_bundle_export_finetune_prediction(
    tmp_path, v3_config, fixture_profiles_yaml
):
    from tests.helpers.fev_fixture import write_fev_fixture
    from tests.helpers.fev_spec import fixture_spec

    runner = CliRunner()
    config = tmp_path / "pretrain.yaml"
    v3_config.to_yaml(config)
    run = tmp_path / "run"
    result = runner.invoke(app, ["pretrain", "run", "--config", str(config), "--out", str(run)])
    assert result.exit_code == 0, (result.output, result.exception)
    source = load_bundle(run / "bundle")
    manifest = source.manifest
    for key in (
        "evaluations",
        "eval_snapshots",
        "best_weights",
        "selection_sources",
        "sampling_exposure",
        "horizon_histogram",
        "feature_exposure",
    ):
        assert manifest[key], key
    assert source.model.spec.calendar_schema == CANONICAL_CALENDAR_SCHEMA
    assert source.model.spec.lag_schema == TARGET_LAG_SCHEMA
    assert manifest["input_schema"] == {
        "calendar_schema": list(CANONICAL_CALENDAR_SCHEMA),
        "lag_schema": list(TARGET_LAG_SCHEMA),
    }
    assert (
        manifest["selection_sources"]
        == manifest["evaluations"][manifest["best_step"] // 3 - 1]["selection_sources"]
    )
    assert all(
        r["raw"]["selection_sources"] and r["ema"]["selection_sources"]
        for r in manifest["evaluations"]
    )
    assert set(manifest["sampling_exposure"]) == {"tsmixup", "kernel_synth", "chronos_real"}
    progress = json.loads((run / "checkpoints/progress.json").read_text())
    assert progress["duration_estimate"]["s_per_step"] > 0
    assert progress["duration_estimate"]["evaluation_measured"]

    exported = tmp_path / "exported"
    export_args = (
        ["--step", str(manifest["best_step"])]
        if manifest["best_weights"] == "raw"
        else ["--weights", "ema"]
    )
    result = runner.invoke(
        app, ["pretrain", "export", "--run-dir", str(run), "--out", str(exported), *export_args]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    # Exported and selected bundles can contain different snapshots; the named
    # transfer map must remain identical for both, independent of selected step.
    settings = tiny_settings(base_steps=2, d_model=8)
    settings.forecast.calendar_schema = "canonical"
    settings.forecast.target_lags = "cycle"
    settings.training.max_steps_cap = 2
    task_config = tmp_path / "task.yaml"
    settings.to_yaml(task_config)
    spec = fixture_spec(write_fev_fixture(tmp_path / "data", length=80), num_windows=1)
    tasks = tmp_path / "tasks.yaml"
    task = asdict(spec)
    del task["categorical_statics"], task["target_support"]
    tasks.write_text(yaml.safe_dump({"tasks": [task]}))
    transfers = []
    for label, bundle in (("run", run / "bundle"), ("export", exported)):
        out = tmp_path / f"finetune-{label}"
        result = runner.invoke(
            app,
            [
                "fev",
                "run",
                "--tasks",
                str(tasks),
                "--config",
                str(task_config),
                "--profiles",
                str(fixture_profiles_yaml),
                "--init-from",
                str(bundle),
                "--out-dir",
                str(out),
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        report = json.loads((out / spec.name / "run.json").read_text())
        assert report["fit_result"]["steps"] == 2
        columns = report["transfer"]["copied_columns"]
        transfers.append(columns)
        for mapping in columns.values():
            assert {r["name"] for r in mapping} == set(
                CANONICAL_CALENDAR_SCHEMA + TARGET_LAG_SCHEMA
            )
            assert {r["role"] for r in mapping} == {"value", "mask"}
        with np.load(out / spec.name / "window_0.npz") as predictions:
            arrays = predictions["arrays"]
            assert arrays.shape == (6, 4, 9)
            assert np.isfinite(arrays).all()
            assert (np.diff(arrays, axis=-1) >= 0).all()
    assert transfers[0] == transfers[1]


def test_v3_probe_all_horizons_actual_evaluation_and_plan(tmp_path, v3_config, monkeypatch, caplog):
    import logging

    from demand_forecast.pretrain import corpus
    from demand_forecast.pretrain.probe import pretrain_probe
    from demand_forecast.pretrain.train import _PretrainLoop, pretrain
    from demand_forecast.pretrain.validate import ValidationSet

    calls = []
    evaluate, snapshot = _PretrainLoop.evaluate, _PretrainLoop.snapshot
    validation_evaluate = ValidationSet.evaluate
    validation_calls = []

    def observe_validation(*args, **kwargs):
        validation_calls.append(True)
        return validation_evaluate(*args, **kwargs)

    def observe_evaluate(loop):
        calls.append("evaluate")
        evaluate(loop)

    def observe_snapshot(loop, directory):
        calls.append("snapshot")
        snapshot(loop, directory)

    monkeypatch.setattr(_PretrainLoop, "evaluate", observe_evaluate)
    monkeypatch.setattr(_PretrainLoop, "snapshot", observe_snapshot)
    monkeypatch.setattr(ValidationSet, "evaluate", observe_validation)
    path = tmp_path / "probe.json"
    report = pretrain_probe(v3_config, steps=3, out=path)
    assert calls == ["evaluate", "snapshot"]
    assert len(validation_calls) == 2  # Raw and EMA, measured through the run's actual path.
    assert set(report["per_horizon"]) == {str(h) for h in FEV_HORIZONS}
    for horizon, row in report["per_horizon"].items():
        assert row["steps"] == 3 and row["eligible_windows_seen"] == 6
        assert row["batch_build_s_per_step"] > 0
        assert row["forward_backward_s_per_step"] > 0
        assert row["peak_cuda_mem_bytes"] is None
        assert row["peak_rss_bytes"] > 0
        assert (
            row["batch_size"] == 2 and row["min_context_tokens"] == row["max_context_tokens"] == 16
        )
        assert all(c["steps"] == 1 for c in row["sources"].values())
        assert report["horizon_histogram"][horizon] == 3
    assert report["snapshot_s"] > 0 and report["eval_s"] > report["snapshot_s"]
    assert report["duration_estimate"]["status"] == "measured"
    assert report == json.loads(path.read_text())
    with caplog.at_level(logging.INFO):
        pretrain(v3_config, tmp_path / "timed", probe=report)
    assert "duration estimate at step 0" in caplog.text
    assert "duration estimate at step 3" in caplog.text
    progress = json.loads((tmp_path / "timed/checkpoints/progress.json").read_text())
    assert progress["duration_estimate"]["total_s"] > 0
    assert progress["duration_estimate"]["eval_s"] > 0

    config = tmp_path / "plan.yaml"
    # Plan must resolve even after both the manifest and shards disappear.
    v3_config.corpus.root = tmp_path / "absent"
    v3_config.to_yaml(config)
    monkeypatch.setattr(corpus, "read_manifest", lambda *a: pytest.fail("plan read manifest"))
    result = CliRunner().invoke(
        app, ["pretrain", "plan", "--config", str(config), "--probe-json", str(path)]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    plan = json.loads(result.output)
    assert plan["horizons"] == list(FEV_HORIZONS)
    assert plan["calendar_schema"] == list(CANONICAL_CALENDAR_SCHEMA)
    assert plan["lag_schema"] == list(TARGET_LAG_SCHEMA)
    assert plan["duration_estimate"]["steps_per_hour"] > 0
    assert (
        plan["selection_cells_policy"]["aggregation"]
        == "equal_source_mean_of_equal_cell_sql_all_cells_finite"
    )


def test_duration_weights_sources_and_horizons(v3_config):
    from demand_forecast.pretrain.train import pretrain_duration

    v3_config.horizon_policy = "fixed"
    v3_config.forecast.horizon = 4
    v3_config.corpus.sources["tsmixup"].weight = 6
    v3_config.corpus.sources["kernel_synth"].weight = 1
    v3_config.corpus.sources["chronos_real"].weight = 3
    probe = dict(
        init_s=1,
        eval_s=3,
        checkpoint_s=4,
        source_horizon_timings={
            k: {"4": dict(steps=2, train_s=2 * s, shard_load_s=2)}
            for k, s in (("tsmixup", 1), ("kernel_synth", 10), ("chronos_real", 3))
        },
    )
    estimate = pretrain_duration(v3_config, probe)
    assert estimate["s_per_step"] == pytest.approx(2.5)
    assert estimate["steps_per_hour"] == pytest.approx(1440)
    assert estimate["total_s"] == pytest.approx((1 + 6 * 3.5 + 2 * 3 + 2 * 4 + 300) * 1.15)
    v3_config.horizon_policy = "fev_set"
    v3_config.forecast.horizon = 288
    partial = pretrain_duration(v3_config, probe)
    assert partial["status"] == "partial" and partial["imputed_cells"]
    probe["source_horizon_timings"] = {
        kind: {"5": dict(steps=2, train_s=2), "288": dict(steps=2, train_s=34)}
        for kind in v3_config.corpus.sources
    }
    probe["horizon_distribution"] = {
        kind: {"5": 0.25, "288": 0.75} for kind in v3_config.corpus.sources
    }
    estimate = pretrain_duration(v3_config, probe)
    assert estimate["s_per_step"] == pytest.approx(13)
    assert not estimate["imputed_cells"]
    v3_config.training.checkpoint_every = 4
    from demand_forecast.pretrain.planning import measured_duration

    assert measured_duration(v3_config, probe, completed_steps=6)["remaining_s"] == 0


def _legacy_fixture_fingerprint(v3_config, config_type=PretrainConfig):
    """Use the unchanged v2 sampler on exactly the integration corpus."""
    from contextlib import closing
    from dataclasses import fields
    from hashlib import sha256

    from demand_forecast.pretrain.config import pretrain_identity
    from demand_forecast.pretrain.corpus import corpus_manifest
    from demand_forecast.pretrain.sampler import CorpusSampler

    raw = v3_config.model_dump(mode="json")
    raw.update(
        recipe="v2", period_policy="legacy_one", grid_policy="legacy_weekly", horizon_policy="fixed"
    )
    raw["forecast"].update(calendar_schema="legacy", target_lags="none")
    raw["training"].update(ema_decay=None, keep_eval_snapshots=False)
    raw["corpus"].pop("selection_shards")
    cfg = config_type.model_validate(raw)
    refs = corpus_manifest(cfg.corpus.root)
    digest = sha256()
    with closing(
        CorpusSampler(
            sources={
                k: [r for r in refs if r.kind == k and r.index == 0] for k in cfg.corpus.sources
            },
            weights={k: s.weight for k, s in cfg.corpus.sources.items()},
            horizon=cfg.forecast.horizon,
            max_context=cfg.forecast.max_context,
            min_context=cfg.forecast.min_context,
            batch_size=cfg.training.batch_size,
            seed=cfg.seed,
        )
    ) as sampler:
        batches = sampler.batches()
        for _ in range(50):
            batch = next(batches)
            digest.update(sampler.last_kind.encode())
            for field in fields(batch):
                if field.name == "loss_weight":
                    continue
                value = getattr(batch, field.name)
                for key, tensor in (
                    sorted(value.items()) if isinstance(value, dict) else [(field.name, value)]
                ):
                    digest.update(key.encode())
                    digest.update(str(tuple(tensor.shape)).encode())
                    digest.update(tensor.numpy().tobytes())
    return pretrain_identity(cfg, "2.0.0", {}), digest.hexdigest()


def test_same_integration_fixture_v2_golden_identity_and_batches(v3_config):
    # Captured with HEAD 924e13e's config module and byte-unchanged v2 sampler.
    # Existing shipped identities and 50-batch golden tests remain untouched.
    assert _legacy_fixture_fingerprint(v3_config) == (
        "b0ede087c50c85fd537d8832a751e5ab2bfe3eef8a5a09edda274586bfa3bc48",
        "5a449669f9f802f83f2909f8a670aa47324744d2d648d52222880b4a6f3218da",
    )


def test_probe_marks_infeasible_source_horizon_without_losing_source_weight(
    tmp_path, v3_config, monkeypatch
):
    from demand_forecast.pretrain.probe import pretrain_probe
    from demand_forecast.pretrain.sampler import V3ShardStream

    advance = V3ShardStream.__next__

    def short_source(stream):
        if (
            stream.kind == "chronos_real"
            and stream.horizon_policy == "fixed"
            and stream.horizon == 288
        ):
            raise ValueError("chronos_real source has no usable windows")
        return advance(stream)

    monkeypatch.setattr(V3ShardStream, "__next__", short_source)
    report = pretrain_probe(v3_config, steps=1, out=tmp_path / "probe.json")
    assert report["per_horizon"]["288"]["sources"]["chronos_real"] == {
        "steps": 0,
        "feasible": False,
    }
    assert report["per_horizon"]["288"]["steps"] == 2
    assert len(report["per_horizon"]) == 16
    estimate = report["duration_estimate"]
    assert estimate["source_mix"]["chronos_real"] == pytest.approx(1 / 3)
    assert "288" not in estimate["horizon_distribution"]["chronos_real"]
    assert estimate["status"] == "measured"
