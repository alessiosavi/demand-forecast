"""P5: bounded full-panel forecasts and real qualification measurements."""

import importlib
import json
import math
import weakref
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from demand_forecast.data.episode import EpisodeBuilder, collate
from demand_forecast.models.v5 import V5Model, forecast, v5_loss
from demand_forecast.synthetic.typed import PanelRecipe, generate_typed_panel
from tests.helpers.v5_loop import config, equal


def api():
    return importlib.import_module("demand_forecast.pretrain.v5.measure")


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


def test_forecast_panel_never_materializes_more_than_batch_episodes(panel, monkeypatch):
    measure = api()
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
    monkeypatch.setattr(measure, "collate", counted)
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
    } <= cold.keys()
    assert cold["corpus_index_seconds"] is None and cold["excluded_row_share"] is None
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
        assert {"forward_seconds", "backward_seconds", "peak_gpu_bytes"} <= measured[
            "worst_batch"
        ].keys()
        assert_numbers(measured["worst_batch"])
        assert {"seconds", "bytes", "writes"} <= measured["checkpoint"].keys()
        assert measured["checkpoint"]["writes"] == 5 and measured["checkpoint"]["bytes"] > 0
        assert_numbers(measured["checkpoint"])
        assert {"seconds", "peak_gpu_bytes", "host_memory_bytes"} <= measured["inference"].keys()
        assert measured["inference"]["forecasts"] == 12
        assert_numbers(measured["inference"])
        parameters = measured["parameters"]
        assert parameters["total"] == sum(p.numel() for p in V5Model(architecture).parameters())
        assert parameters["total"] == sum(parameters["by_part"].values())
        assert_numbers(parameters)


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


def test_qualify_corpus_exclusion_is_rows_not_groups(tmp_path, monkeypatch):
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
        row_group_size=2,
    )
    write_rows(root / "univariate/b/train-000.parquet", [[np.arange(512)]])
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
    real_peak, real_generate = measure._HostPeak, measure.generate_typed_panel
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
    monkeypatch.setattr(measure, "generate_typed_panel", generate)
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
