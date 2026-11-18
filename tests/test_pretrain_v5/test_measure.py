"""P5: bounded full-panel forecasts and real qualification measurements."""

import importlib
import json
import math
import weakref
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from demand_forecast.data.episode import EpisodeBuilder, collate
from demand_forecast.models.v5 import V5Model, forecast, v5_loss
from demand_forecast.synthetic.typed import PanelRecipe, generate_typed_panel
from tests.helpers.v5_loop import config, equal


def api():
    return importlib.import_module("demand_forecast.pretrain.v5.measure")


@pytest.fixture(autouse=True)
def serial_inference_on_any_host(monkeypatch):
    monkeypatch.setattr(api(), "cpu_count", lambda: 1, raising=False)


@pytest.fixture(scope="module")
def panel():
    generated = generate_typed_panel(
        PanelRecipe(
            mechanisms=("multi_target", "static_category", "planned_event"),
            n_items=20,
            length=(96, 96),
            complete_inputs=True,
            seed=81,
        )
    )
    return generated.panel


@pytest.fixture(scope="module", params=[("learned", True), ("ones", False)])
def reference_forecasts(panel, request):
    gates, pruned = request.param
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = V5Model(config().model)
    builder = EpisodeBuilder(panel)
    expected, scales = {}, {}
    for index, item in enumerate(panel.items):
        for request in builder.plan_requests(index, (80,), 8, context=32, cutoff=80):
            result = forecast(model, panel, request, builder=builder, gates=gates, pruned=pruned)
            for row, name in enumerate(result.names):
                expected[item.item_id, name] = result.quantiles[0, row]
                scales[item.item_id, name] = builder._stats(name, 80).scales[index].scale
    return model, gates, pruned, expected, scales


@pytest.mark.parametrize("batch_episodes", [1, 7, 64])
def test_forecast_panel_matches_forecast_in_scaled_frame(
    panel, reference_forecasts, batch_episodes
):
    model, gates, pruned, expected, scales = reference_forecasts
    actual = api().forecast_panel(
        model, panel, 80, 8, context=32, batch_episodes=batch_episodes, gates=gates, pruned=pruned
    )
    assert (
        actual.keys()
        == expected.keys()
        == {(item.item_id, name) for item in panel.items for name in item.targets}
    )
    for key in expected:
        assert actual[key].shape == (8, 9)
        assert np.isfinite(actual[key]).all()
        np.testing.assert_allclose(
            actual[key] / scales[key], expected[key] / scales[key], atol=1e-5, rtol=0
        )


def test_forecast_panel_late_item_fallback(panel):
    late = replace(panel.items[-1], start=100)
    panel = replace(panel, items=(*panel.items[:-1], late))
    result = api().forecast_panel(V5Model(config().model), panel, 80, 8, context=32)
    for name in late.targets:
        np.testing.assert_array_equal(result[late.item_id, name], np.zeros((8, 9)))


def test_forecast_panel_accepts_float64_model(panel):
    panel = replace(panel, items=panel.items[:2])
    model = V5Model(config().model).double()

    result = api().forecast_panel(model, panel, 80, 8, context=32)

    assert result.keys() == {(item.item_id, name) for item in panel.items for name in item.targets}
    assert all(value.dtype == np.float64 and np.isfinite(value).all() for value in result.values())


def test_forecast_panel_restores_training_mode_and_is_repeatable(panel):
    model = V5Model(config().model).train()
    first = api().forecast_panel(model, panel, 80, 8, context=32, batch_episodes=7)
    assert model.training
    second = api().forecast_panel(model, panel, 80, 8, context=32, batch_episodes=7)
    assert model.training
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    model.eval()
    api().forecast_panel(model, panel, 80, 8, context=32)
    assert not model.training


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forecast_panel_uses_model_device_without_moving_parameters(panel, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = V5Model(config().model).to(device).train()
    parameters = tuple(model.parameters())
    original_device = parameters[0].device

    def cannot_move(*args, **kwargs):
        raise AssertionError("forecast_panel must not move the caller's model")

    monkeypatch.setattr(model, "to", cannot_move)
    for requested in (None, device):
        result = api().forecast_panel(model, panel, 80, 8, context=32, device=requested)
        assert len(result) == sum(len(item.targets) for item in panel.items)
        assert all(np.isfinite(values).all() for values in result.values())
        assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
        assert all(parameter.device == original_device for parameter in model.parameters())
        assert model.training


@pytest.mark.parametrize("model_device,requested", [("cpu", "meta"), ("meta", "cpu")])
def test_forecast_panel_rejects_device_mismatch(panel, model_device, requested):
    model = V5Model(config().model).to(model_device).train()
    with pytest.raises(
        ValueError,
        match=rf"^requested device {requested} differs from model device {model_device}$",
    ):
        api().forecast_panel(model, panel, 80, 8, context=32, device=requested)
    assert model.training
    assert all(parameter.device == torch.device(model_device) for parameter in model.parameters())


def test_forecast_panel_rejects_invalid_batch_size_before_device_mismatch(panel):
    model = V5Model(config().model)

    with pytest.raises(ValueError, match="batch_episodes must be a positive integer"):
        api().forecast_panel(model, panel, 80, 8, context=32, batch_episodes=0, device="meta")


def test_forecast_panel_delegates_worker_validation(panel, monkeypatch):
    measure = api()
    real_forecast_items = measure.forecast_items
    calls = []

    def forecast_items(*args, **kwargs):
        calls.append((kwargs["workers"], kwargs["prefetch"]))
        return real_forecast_items(*args, **kwargs)

    monkeypatch.setattr(measure, "forecast_items", forecast_items)
    model = V5Model(config().model)
    with pytest.raises(ValueError, match="^workers must be a nonnegative integer$"):
        measure.forecast_panel(model, panel, 80, 8, workers=-1, prefetch=3)
    assert calls == [(-1, 3)]


def test_forecast_panel_never_materializes_more_than_batch_episodes(panel, monkeypatch):
    measure = api()
    batch_forecast = importlib.import_module("demand_forecast.models.v5.batch_forecast")
    references, sizes = [], []
    real_build = EpisodeBuilder.build

    def build(self, request):
        episode = real_build(self, request)
        references.append(weakref.ref(episode))
        return episode

    def counted(episodes):
        sizes.append(len(episodes))
        assert sum(reference() is not None for reference in references) <= 7
        return collate(episodes)

    monkeypatch.setattr(EpisodeBuilder, "build", build)
    monkeypatch.setattr(batch_forecast, "collate", counted)
    result = measure.forecast_panel(
        V5Model(config().model), panel, 80, 8, context=32, batch_episodes=7
    )
    assert len(result) == sum(len(item.targets) for item in panel.items)
    assert len(sizes) > 1 and max(sizes) == 7
    assert not any(reference() is not None for reference in references)


def test_forecast_panel_restores_mode_on_error(panel, monkeypatch):
    model = V5Model(config().model).train()

    def fail(*args, **kwargs):
        raise RuntimeError("injected forward failure")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected forward failure"):
        api().forecast_panel(model, panel, 80, 8, context=32)
    assert model.training


def test_worst_batch_maximal_shapes_and_determinism():
    c = config()
    c = replace(
        c,
        sampler=replace(
            c.sampler,
            episodes_per_batch=2,
            contexts=(32, 64),
            horizon=(8, 20),
            origins=3,
            origin_stride=1,
        ),
    )
    first, second = api().worst_batch(c), api().worst_batch(c)
    assert first.x.shape == (2, 40, 4, 16, 39)
    assert first.labels.shape == (2, 3, 16, 32)
    assert first.xh.shape == (2, 3, 16, 2, 16, 39)
    assert first.attr_stats.shape == (2, 16, 57)
    assert first.static_num.shape == (2, 16)
    assert first.cand_mask.shape == (2, 16)
    assert first.origin_patches.tolist() == [[2, 3, 4], [2, 3, 4]]
    for field in fields(first):
        equal(getattr(first, field.name), getattr(second, field.name))
    model = V5Model(c.model)
    loss, _ = v5_loss(model(first, model.gates(first, mode="ones")), first, c.model)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_worst_batch_origins_fit_context():
    c = config()
    c = replace(c, sampler=replace(c.sampler, origins=4, origin_stride=2))
    batch = api().worst_batch(c)
    assert batch.origin_patches.tolist() == [[2]]


def assert_numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            assert_numbers(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_numbers(item)
    else:
        assert value is None or isinstance(value, (int, float)) and math.isfinite(value)


@pytest.mark.parametrize("updated", [(1, 0, 0, 0, 1, 0), (1, 0, 0, 0, 0, 0)])
def test_trial_update_rate_counts_successes_over_all_tail_seconds(tmp_path, monkeypatch, updated):
    measure = api()
    elapsed = [0.0]
    durations = (100, 1, 2, 3, 4, 5)
    report = {
        "episodes": 1,
        "supervised": 80,
        "cells": 100,
        "skips": {},
        "sources": {"typed": 1},
        "paths": {},
        "paths_without_candidate": {},
    }

    def trial_train(c, directory):
        history = []
        iterator = measure.loop.loader(c, 0)
        try:
            for _ in range(c.optim.max_steps):
                index, _, _ = next(iterator)
                elapsed[0] += durations[index]
                history.append({"updates": updated[index]})
                measure.loop._json_line(directory / "telemetry.jsonl", {"event": "batch"})
        finally:
            measure.loop._release(iterator)
        return measure.loop.TrainResult(
            6, sum(updated), 6 - sum(updated), "max_steps", directory / "checkpoint.pt", history
        )

    monkeypatch.setattr(measure, "time", SimpleNamespace(perf_counter=lambda: elapsed[0]))
    monkeypatch.setattr(measure, "train", trial_train)
    monkeypatch.setattr(
        measure.loop, "loader", lambda *a, **kw: iter((i, None, report) for i in range(6))
    )
    monkeypatch.setattr(measure.loop, "_json_line", lambda *a: None)
    c = config(steps=6)
    metrics, _ = measure._trial(c, tmp_path, torch.device("cpu"))
    tail_updates = sum(updated[1:])
    assert metrics["updates"] == sum(updated)
    assert metrics["tail_batches"] == 5
    assert metrics["updates_per_second"] == pytest.approx(tail_updates / 15)
    assert metrics["updates_per_hour"] == pytest.approx(tail_updates * 240)
    assert metrics["seconds_per_update"] == (4 if tail_updates else None)
    assert metrics["supervised_cells_per_update"] == (80 if tail_updates else None)


def test_inference_panel_is_aligned_dated_and_deterministic():
    measure = api()
    first = measure._inference_panel(seed=17, items=12, length=200)
    np.random.seed(99)
    second = measure._inference_panel(seed=17, items=12, length=200)
    different = measure._inference_panel(seed=18, items=12, length=200)
    first.validate()
    assert first.aligned and first.freq == "D" and first.epoch is not None
    timestamps = first.timestamps(0, 200)
    assert len(timestamps) == 200
    assert np.all(np.diff(timestamps.values) == np.timedelta64(1, "D"))
    assert len(first.items) == len({item.item_id for item in first.items}) == 12
    assert len(first.roles) == 1 and set(first.roles.values()) == {"known"}
    assert all(len(item.static) == 1 for item in first.items)
    assert len({next(iter(item.static.values())) for item in first.items}) == 10
    levels = []
    for a, b, c in zip(first.items, second.items, different.items, strict=True):
        assert a.start == b.start == c.start == 0
        assert a.static == b.static == c.static
        assert len(a.targets) == len(a.dynamic) == 1
        target = next(iter(a.targets.values()))
        assert target.shape == (200,) and np.isfinite(target).all()
        np.testing.assert_array_equal(target, next(iter(b.targets.values())))
        assert not np.array_equal(target, next(iter(c.targets.values())))
        # A noisy weekly pattern repeats strongly but is not an exact tiled cycle.
        assert np.corrcoef(target[:-7], target[7:])[0, 1] > 0.9
        assert not np.array_equal(target[:-7], target[7:])
        levels.append(target.mean())
        flag = next(iter(a.dynamic.values()))
        assert flag.shape == (200,) and set(flag) == {0.0, 1.0}
        np.testing.assert_array_equal(flag, next(iter(b.dynamic.values())))
    assert max(levels) - min(levels) > 1


@pytest.mark.parametrize("failure", ["missing_forecast", "fallback"])
def test_qualify_rejects_missing_or_fallback_inference(tmp_path, monkeypatch, failure):
    measure = api()
    if failure == "missing_forecast":
        real_forecast = measure.forecast_panel

        def missing(*args, **kwargs):
            result = real_forecast(*args, **kwargs)
            result.pop(next(iter(result)))
            return result

        monkeypatch.setattr(measure, "forecast_panel", missing)
    else:
        real_panel = measure._inference_panel

        def constant(*args, **kwargs):
            panel = real_panel(*args, **kwargs)
            item = panel.items[0]
            item = replace(
                item, targets={name: np.ones_like(values) for name, values in item.targets.items()}
            )
            return replace(panel, items=(item, *panel.items[1:]))

        monkeypatch.setattr(measure, "_inference_panel", constant)
    c = config()
    with pytest.raises(RuntimeError, match="inference"):
        measure.qualify(
            c,
            tmp_path,
            models={"tiny": c.model},
            updates=1,
            workers=(0,),
            panel_items=2,
            panel_length=64,
        )
    assert not (tmp_path / "qualify.json").exists()


@pytest.mark.parametrize("workers", [(0,), (0, 1)], ids=["serial", "multiprocess"])
def test_qualify_measures_all_sections_and_only_writes_json(tmp_path, workers, monkeypatch):
    measure = api()
    if any(workers):
        sampler = importlib.import_module("demand_forecast.pretrain.v5.sampler")
        real_loader = sampler.DataLoader
        monkeypatch.setattr(
            sampler,
            "DataLoader",
            lambda *a, **kw: (
                real_loader(*a, timeout=10, **kw)
                if kw.get("num_workers")
                else real_loader(*a, **kw)
            ),
        )
    c = config(steps=6)
    c = replace(
        c,
        sampler=replace(
            c.sampler,
            contexts=(64,),
            origins=2,
            drop=(("lags", 1.0),),
            drop_always=("rolling",),
            summary_off=("zero_share",),
            max_cells=None,
            designate_target=True,
        ),
    )
    models = {"full": c.model, "t1": replace(c.model, d_model=24)}
    report = measure.qualify(
        c, tmp_path, models=models, updates=6, workers=workers, panel_items=12, panel_length=200
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["qualify.json"]
    assert json.loads((tmp_path / "qualify.json").read_text()) == report
    cold = report["cold_start"]
    assert {
        "corpus_index_seconds",
        "first_batch_seconds",
        "restore_seconds",
        "excluded_row_share",
        "unparsed_frequency_rows",
    } <= cold.keys()
    assert cold["corpus_index_seconds"] is None and cold["excluded_row_share"] is None
    assert cold["unparsed_frequency_rows"] is None
    assert_numbers(cold)
    for name, architecture in models.items():
        measured = report["models"][name]
        for worker in workers:
            run = measured["workers"][str(worker)]
            assert {
                "batches_per_second",
                "updates_per_second",
                "updates_per_hour",
                "supervised_cells_per_batch",
                "padded_cells_per_batch",
                "skips",
                "host_memory_bytes",
                "peak_gpu_bytes",
            } <= run.keys()
            assert run["batches"] == run["updates"] == 6 and run["tail_batches"] == 5
            assert run["batches_per_second"] > 0 and run["updates_per_hour"] > 0
            assert run["supervised_cells_per_batch"] > 0 and run["padded_cells_per_batch"] > 0
            assert_numbers(run)
        assert measured["origins"].keys() == {"1", "2"}
        for run in measured["origins"].values():
            assert {
                "supervised_cells_per_update",
                "seconds_per_update",
                "peak_gpu_bytes",
            } <= run.keys()
            assert run["supervised_cells_per_update"] > 0 and run["seconds_per_update"] > 0
            assert_numbers(run)
        assert measured["worst_batch"].keys() == {
            "forward_backward_seconds",
            "forward_backward_peak_gpu_bytes",
        }
        assert measured["worst_batch"]["forward_backward_seconds"] > 0
        assert_numbers(measured["worst_batch"])
        assert {"seconds", "bytes", "writes"} <= measured["checkpoint"].keys()
        assert measured["checkpoint"]["writes"] == 5 and measured["checkpoint"]["bytes"] > 0
        assert_numbers(measured["checkpoint"])
        assert {"seconds", "peak_gpu_bytes", "host_memory_bytes"} <= measured["inference"].keys()
        assert measured["inference"]["origin"] == 172
        assert measured["inference"]["forecasts"] == measured["inference"]["items"] == 12
        assert measured["inference"]["fallback_forecasts"] == 0
        assert_numbers(measured["inference"])
        parameters = measured["parameters"]
        assert parameters["total"] == sum(p.numel() for p in V5Model(architecture).parameters())
        assert parameters["total"] == sum(parameters["by_part"].values())
        assert_numbers(parameters)


@pytest.mark.parametrize(
    "platform,cores,expected_workers",
    [("linux", 17, 0), ("linux", 18, 16), ("linux", None, 0), ("darwin", 18, 0)],
)
def test_qualify_keys_are_unchanged(tmp_path, monkeypatch, platform, cores, expected_workers):
    measure = api()
    real_forecast_panel = measure.forecast_panel
    inference_workers = []

    def forecast_panel(*args, **kwargs):
        inference_workers.append(kwargs.pop("workers", None))
        return real_forecast_panel(*args, **kwargs)

    monkeypatch.setattr(measure, "sys", SimpleNamespace(platform=platform), raising=False)
    monkeypatch.setattr(measure, "cpu_count", lambda: cores, raising=False)
    monkeypatch.setattr(measure, "forecast_panel", forecast_panel)
    c = config()
    report = measure.qualify(
        c,
        tmp_path,
        models={"tiny": c.model},
        updates=1,
        workers=(0,),
        panel_items=2,
        panel_length=64,
    )
    assert json.loads((tmp_path / "qualify.json").read_text()) == report
    assert report.keys() == {"cold_start", "models"}
    cold = report["cold_start"]
    assert cold.keys() == {
        "corpus_index_seconds",
        "first_batch_seconds",
        "restore_seconds",
        "excluded_row_share",
        "unparsed_frequency_rows",
    }
    measured = report["models"]["tiny"]
    assert measured.keys() == {
        "workers",
        "origins",
        "worst_batch",
        "checkpoint",
        "inference",
        "parameters",
    }
    assert measured["inference"].keys() == {
        "seconds",
        "peak_gpu_bytes",
        "host_memory_bytes",
        "origin",
        "items",
        "forecasts",
        "fallback_forecasts",
    }
    assert inference_workers == [expected_workers]


def test_qualify_uses_expensive_settings_and_real_optimizer_ema_checkpoint(tmp_path, monkeypatch):
    measure = api()
    real_train, real_save = measure.train, measure.save_checkpoint
    settings, checkpoints = [], []

    def observed_train(c, *args, **kwargs):
        settings.append(c)
        return real_train(c, *args, **kwargs)

    def observed_save(directory, payload, **kwargs):
        checkpoints.append(payload)
        return real_save(directory, payload, **kwargs)

    monkeypatch.setattr(measure, "train", observed_train)
    monkeypatch.setattr(measure, "save_checkpoint", observed_save)
    c = config()
    c = replace(
        c,
        sampler=replace(
            c.sampler,
            drop=(("features", 1.0),),
            drop_always=("rolling",),
            summary_off=("zero_share",),
            designate_target=True,
        ),
    )
    measure.qualify(
        c,
        tmp_path,
        models={"tiny": c.model},
        updates=2,
        workers=(0,),
        panel_items=2,
        panel_length=64,
    )
    assert settings
    assert all(
        c.gates == "ones"
        and c.sampler.drop == ()
        and c.sampler.drop_always == ()
        and c.sampler.summary_off == ()
        for c in settings
    )
    assert len(checkpoints) == 5
    assert all(p["optimizer"]["state"] and p["ema"] is not None for p in checkpoints)
    assert not checkpoints[-1]["config"]["sampler"]["drop"]


@pytest.mark.parametrize(
    "labels,expected",
    [(["D", "D"], {}), (["D", "nonsense"], {"nonsense": 1})],
    ids=["all_parsed", "unparsed"],
)
def test_qualify_corpus_reports_exclusions_and_unparsed_rows(
    tmp_path, monkeypatch, labels, expected
):
    pytest.importorskip("pyarrow")
    from demand_forecast.pretrain.v5.config import SourceConfig
    from demand_forecast.pretrain.v5.corpus import CorpusIndex
    from tests.helpers.corpus_v4_fixture import write_rows

    measure = api()
    root = tmp_path / "corpus"
    # Two admitted rows, one excluded row: the row share is 1/3, not 1/2 groups.
    write_rows(
        root / "univariate/a/train-000.parquet",
        [[np.arange(64)], [np.arange(64)]],
        freqs=labels,
        row_group_size=2,
    )
    write_rows(root / "univariate/b/train-000.parquet", [[np.arange(512)]], freqs=["nonsense"])
    build = CorpusIndex.build

    def limited(cls, location, **kwargs):
        return build(location, max_group_bytes=4000)

    monkeypatch.setattr(CorpusIndex, "build", classmethod(limited))
    c = config()
    c = replace(c, sources=(SourceConfig("corpus", 1, root=str(root)),))
    report = measure.qualify(
        c,
        tmp_path / "out",
        models={"tiny": c.model},
        updates=2,
        workers=(0,),
        panel_items=2,
        panel_length=64,
    )
    assert report["cold_start"]["corpus_index_seconds"] >= 0
    assert report["cold_start"]["excluded_row_share"] == pytest.approx(1 / 3)
    assert report["cold_start"]["unparsed_frequency_rows"] == expected


def test_checkpoint_measure_serializes_state_on_measured_device(tmp_path, monkeypatch):
    measure = api()
    c = config(ema=0.9, steps=1)
    checkpoint = measure.train(c, tmp_path / "state").checkpoint
    real_save = measure.save_checkpoint

    def on_device(directory, payload, **kwargs):
        # Meta exercises device placement without requiring a GPU in CPU tests.
        assert all(value.device.type == "meta" for value in payload["model"].values())
        assert all(value.device.type == "meta" for value in payload["ema"].values())
        for state in payload["optimizer"]["state"].values():
            assert all(
                value.device.type == "meta"
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
        return real_save(directory, payload, **kwargs)

    monkeypatch.setattr(measure, "save_checkpoint", on_device)
    result = measure._checkpoint_measure(checkpoint, tmp_path / "write", torch.device("meta"))
    assert result["writes"] == 5 and result["bytes"] > 0


def test_inference_memory_excludes_panel_generation(tmp_path, monkeypatch):
    measure = api()
    real_peak, real_generate = measure._HostPeak, measure._inference_panel
    active, generating = [], [False]

    class Peak(real_peak):
        def __enter__(self):
            result = super().__enter__()
            active.append(self)
            return result

        def __exit__(self, *args):
            active.remove(self)
            super().__exit__(*args)

    def generate(*args, **kwargs):
        generating[0] = True
        # Synchronous sampling makes the transient generation peak deterministic.
        for observer in active:
            observer.sample()
        result = real_generate(*args, **kwargs)
        generating[0] = False
        return result

    monkeypatch.setattr(measure, "_HostPeak", Peak)
    monkeypatch.setattr(measure, "_host_rss", lambda: 999_999 if generating[0] else 100)
    monkeypatch.setattr(measure, "_inference_panel", generate)
    c = config()
    result = measure.qualify(
        c,
        tmp_path,
        models={"tiny": c.model},
        updates=1,
        workers=(0,),
        panel_items=2,
        panel_length=64,
    )
    assert result["models"]["tiny"]["inference"]["host_memory_bytes"] == 100


def test_qualify_releases_cold_loader_before_measuring_training(tmp_path, monkeypatch):
    measure = api()
    real_loader, real_train = measure.loader, measure.train
    references = []

    def tracked_loader(*args, **kwargs):
        iterator = real_loader(*args, **kwargs)
        references.append(weakref.ref(iterator))
        return iterator

    def train_after_release(*args, **kwargs):
        assert all(reference() is None for reference in references)
        return real_train(*args, **kwargs)

    monkeypatch.setattr(measure, "loader", tracked_loader)
    monkeypatch.setattr(measure, "train", train_after_release)
    c = config()
    measure.qualify(
        c,
        tmp_path,
        models={"tiny": c.model},
        updates=1,
        workers=(0,),
        panel_items=2,
        panel_length=64,
    )
