"""B2c: forked episode construction with forecasts in the calling process."""

import multiprocessing
import os
import sys
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, get_worker_info

from demand_forecast.data.episode import EpisodeBuilder
from demand_forecast.data.episode.panel_stats import PanelStats
from demand_forecast.models import v5
from demand_forecast.models.v5 import batch_forecast
from demand_forecast.pretrain.v5 import measure, sampler
from tests.helpers.episodes import toy_panel
from tests.helpers.v5_loop import config

CONTEXT = 32
HORIZON = 3
linux_workers = pytest.mark.skipif(sys.platform != "linux", reason="requires Linux fork workers")


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261119)
        return v5.V5Model(config().model)


@pytest.fixture
def panel():
    panel = toy_panel(n_items=4, length=56, seed=37, n_targets=2)
    return replace(
        panel,
        items=tuple(
            replace(item, static={"brand": "shared"}, dynamic={"promo": item.dynamic["promo"]})
            for item in panel.items
        ),
        roles={"promo": "known"},
    )


@pytest.fixture
def worker_timeout(monkeypatch):
    def loader(*args, **kwargs):
        return DataLoader(*args, timeout=30, **kwargs)

    monkeypatch.setattr(batch_forecast, "DataLoader", loader, raising=False)


class InProcessLoader(DataLoader):
    # Record the context without resolving it, including on platforms without fork.
    multiprocessing_context = None

    def __iter__(self):
        return (self.dataset[i] for i in range(len(self.dataset)))


def _assert_evidence_equal(actual, expected):
    assert actual.requests == expected.requests
    assert actual.candidates == expected.candidates
    assert actual.rules == expected.rules
    for field in ("status", "rule", "gate", "score"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


@pytest.fixture
def in_process_loader(monkeypatch):
    monkeypatch.setattr(batch_forecast, "sys", SimpleNamespace(platform="linux"), raising=False)
    monkeypatch.setattr(sampler, "_ensure_worker_file_limit", lambda workers: None)
    monkeypatch.setattr(batch_forecast, "DataLoader", InProcessLoader, raising=False)


def test_one_chunk_uses_the_calling_process(model, panel, monkeypatch, in_process_loader):
    def fail_loader(*args, **kwargs):
        pytest.fail("a single chunk should not create a DataLoader")

    monkeypatch.setattr(batch_forecast, "DataLoader", fail_loader)
    jobs = [(0, 40)]
    options = {"context": CONTEXT, "batch_episodes": 2}
    serial = v5.forecast_items(model, panel, jobs, HORIZON, workers=0, **options)
    actual = v5.forecast_items(model, panel, jobs, HORIZON, workers=2, **options)

    assert list(actual.values) == list(serial.values)
    assert actual.fallback == serial.fallback
    assert actual.stats_builds == serial.stats_builds
    _assert_evidence_equal(actual.evidence, serial.evidence)
    for key, value in actual.values.items():
        np.testing.assert_array_equal(value, serial.values[key])


@pytest.mark.parametrize("chunks,workers,expected_workers", [(3, 8, 3), (5, 2, 2)])
def test_loader_worker_count_is_capped_by_chunks(
    model, monkeypatch, in_process_loader, chunks, workers, expected_workers
):
    calls = []

    def loader(dataset, **kwargs):
        calls.append((len(dataset), kwargs["num_workers"]))
        return InProcessLoader(dataset, **kwargs)

    monkeypatch.setattr(batch_forecast, "DataLoader", loader)
    panel = toy_panel(n_items=chunks, length=56, seed=37)
    jobs = [(index, 40) for index in range(chunks)]
    v5.forecast_items(
        model, panel, jobs, HORIZON, context=CONTEXT, batch_episodes=1, workers=workers
    )

    assert calls == [(chunks, expected_workers)]


@pytest.mark.parametrize(
    "platform,requested,limit_error,expected",
    [
        ("linux", 0, None, (0, None)),
        ("linux", 2, "injected file limit error", (0, "injected file limit error")),
        ("darwin", 2, None, (0, "forecast workers are supported only on Linux")),
        ("linux", 2, None, (2, None)),
    ],
    ids=["zero", "file-limit", "off-linux", "supported"],
)
def test_usable_workers(monkeypatch, platform, requested, limit_error, expected):
    checked = []

    def ensure_file_limit(workers):
        checked.append(workers)
        if limit_error is not None:
            raise RuntimeError(limit_error)

    monkeypatch.setattr(batch_forecast, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(sampler, "_ensure_worker_file_limit", ensure_file_limit)

    assert v5.usable_workers(requested) == expected
    assert checked == ([requested] if requested and platform == "linux" else [])


@pytest.mark.parametrize("requested", [-1, True, 1.5, "2"])
def test_usable_workers_refuses_what_forecast_items_refuses(requested):
    with pytest.raises(ValueError, match="nonnegative integer"):
        v5.usable_workers(requested)


@linux_workers
def test_workers_give_identical_forecasts(model, panel, monkeypatch, worker_timeout):
    jobs = [(2, 40), (0, 40), (3, 40), (1, 36)]
    options = {"context": CONTEXT, "batch_episodes": 1}
    builder = EpisodeBuilder(panel)
    expected, scales = {}, {}
    for index, origin in jobs:
        for request in builder.plan_requests(index, (origin,), HORIZON, context=CONTEXT):
            result = v5.forecast(model, panel, request, builder=builder)
            for row, target in enumerate(result.names):
                expected[index, target] = result.quantiles[0, row]
                scales[index, target] = builder.stats.get(target, origin).scales[index].scale
    serial = v5.forecast_items(model, panel, jobs, HORIZON, workers=0, **options)

    parent = os.getpid()
    real_build, real_collate = EpisodeBuilder.build, batch_forecast.collate
    real_gates, real_forward = model.gates, model.forward

    def build(self, request):
        assert os.getpid() != parent and get_worker_info() is not None
        return real_build(self, request)

    def collate(episodes):
        assert os.getpid() != parent and get_worker_info() is not None
        return real_collate(episodes)

    def gates(*args, **kwargs):
        assert os.getpid() == parent
        return real_gates(*args, **kwargs)

    def forward(*args, **kwargs):
        assert os.getpid() == parent
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(EpisodeBuilder, "build", build)
    monkeypatch.setattr(batch_forecast, "collate", collate)
    monkeypatch.setattr(model, "gates", gates)
    monkeypatch.setattr(model, "forward", forward)
    assert serial.values.keys() == expected.keys()
    for workers in (1, 2):
        actual = v5.forecast_items(model, panel, jobs, HORIZON, workers=workers, **options)
        assert list(actual.values) == list(serial.values)
        assert actual.fallback == serial.fallback
        _assert_evidence_equal(actual.evidence, serial.evidence)
        for key, value in actual.values.items():
            assert np.array_equal(value, serial.values[key]), key
            np.testing.assert_allclose((value - expected[key]) / scales[key], 0, atol=1e-5, rtol=0)


@linux_workers
@pytest.mark.parametrize(
    "origins,has_statics,batch_episodes",
    [((40, 40, 40, 40), True, 2), ((40, 36, 40, 44), True, 1), ((40, 40, 40, 40), False, 2)],
    ids=["one", "three", "joint-no-statics"],
)
def test_no_statistics_are_constructed_in_a_worker(
    model, panel, monkeypatch, worker_timeout, origins, has_statics, batch_episodes
):
    if not has_statics:
        panel = replace(panel, items=tuple(replace(item, static={}) for item in panel.items))
    parent = os.getpid()
    worker_builds = multiprocessing.get_context("fork").Value("i", 0)
    parent_builds = Counter()
    real_init = PanelStats.__init__

    def init(self, panel, target, cutoff):
        if os.getpid() == parent:
            parent_builds[target, cutoff] += 1
        else:
            with worker_builds.get_lock():
                worker_builds.value += 1
        real_init(self, panel, target, cutoff)

    monkeypatch.setattr(PanelStats, "__init__", init)
    episodes_in_workers = multiprocessing.get_context("fork").Value("i", 0)
    real_build = EpisodeBuilder.build

    def build(self, request):
        if os.getpid() != parent:
            with episodes_in_workers.get_lock():
                episodes_in_workers.value += 1
        return real_build(self, request)

    monkeypatch.setattr(EpisodeBuilder, "build", build)
    result = v5.forecast_items(
        model,
        panel,
        list(enumerate(origins)),
        HORIZON,
        context=CONTEXT,
        batch_episodes=batch_episodes,
        layouts=None if has_statics else ["joint"] * len(origins),
        workers=2,
    )

    # A serial run would satisfy the statistics assertions trivially: workers must have built episodes.
    assert episodes_in_workers.value > 0
    assert worker_builds.value == 0
    targets = ("y", "y1") if has_statics else ("y",)
    expected = {(target, origin): 1 for target in targets for origin in set(origins)}
    assert parent_builds == expected
    assert result.stats_builds == len(expected)


def test_joint_statistics_match_serial_without_statics(model, panel, in_process_loader):
    panel = replace(panel, items=tuple(replace(item, static={}) for item in panel.items))
    jobs = [(index, 40) for index in range(len(panel.items))]
    options = {"context": CONTEXT, "batch_episodes": 2, "layouts": ["joint"] * len(jobs)}

    serial = v5.forecast_items(model, panel, jobs, HORIZON, workers=0, **options)
    actual = v5.forecast_items(model, panel, jobs, HORIZON, workers=2, **options)

    assert actual.stats_builds == serial.stats_builds == 1
    _assert_evidence_equal(actual.evidence, serial.evidence)


def test_workers_are_refused_off_linux(model, panel, monkeypatch):
    monkeypatch.setattr(batch_forecast, "sys", SimpleNamespace(platform="darwin"), raising=False)

    with pytest.raises(ValueError, match="Linux"):
        v5.forecast_items(None, None, [(0, 40)], HORIZON, context=CONTEXT, workers=1)
    with pytest.raises(ValueError, match="Linux"):
        measure.forecast_panel(model, panel, 40, HORIZON, context=CONTEXT, workers=1)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("prefetch", [0, -5, 1.5, True, None])
def test_prefetch_must_be_a_positive_integer(model, panel, monkeypatch, workers, prefetch):
    monkeypatch.setattr(batch_forecast, "sys", SimpleNamespace(platform="linux"), raising=False)

    with pytest.raises(ValueError, match="prefetch must be a positive integer"):
        v5.forecast_items(
            model, panel, [], HORIZON, context=CONTEXT, workers=workers, prefetch=prefetch
        )


def test_loader_is_created_with_the_prefetch(model, panel, monkeypatch, in_process_loader):
    limits, loaders = [], []
    parent_builds = Counter()
    real_init = PanelStats.__init__

    def init(self, panel, target, cutoff):
        assert limits == [2]
        parent_builds[target, cutoff] += 1
        real_init(self, panel, target, cutoff)

    class RecordingLoader(InProcessLoader):
        def __init__(self, *args, **kwargs):
            assert parent_builds == {("y", 40): 1, ("y1", 40): 1}
            super().__init__(*args, **kwargs)
            loaders.append(self)

    monkeypatch.setattr(PanelStats, "__init__", init)
    monkeypatch.setattr(sampler, "_ensure_worker_file_limit", limits.append)
    monkeypatch.setattr(batch_forecast, "DataLoader", RecordingLoader)

    result = measure.forecast_panel(
        model, panel, 40, HORIZON, context=CONTEXT, batch_episodes=3, workers=2, prefetch=5
    )

    (loader,) = loaders
    assert loader.num_workers == 2
    assert loader.multiprocessing_context == "fork"
    assert loader.batch_size is None
    assert loader.prefetch_factor == 5
    assert len(result) == 8


@pytest.mark.parametrize("failure", [None, "loader", "model"], ids=["release", "loader", "model"])
def test_loader_release_failure_preserves_the_forecast_error(
    model, panel, monkeypatch, in_process_loader, failure
):
    released = []

    def fail_release(iterator):
        released.append(iterator)
        raise RuntimeError("injected release failure")

    def fail_forecast(*args, **kwargs):
        raise RuntimeError(f"injected {failure} failure")

    monkeypatch.setattr(measure.loop, "_release", fail_release)
    if failure == "loader":
        monkeypatch.setattr(EpisodeBuilder, "build", fail_forecast)
    elif failure == "model":
        monkeypatch.setattr(model, "forward", fail_forecast)

    model.train()
    with pytest.raises(RuntimeError, match=f"injected {failure or 'release'} failure"):
        v5.forecast_items(
            model, panel, [(0, 40)], HORIZON, context=CONTEXT, batch_episodes=1, workers=2
        )
    assert len(released) == 1
    assert model.training


@linux_workers
def test_worker_failure_surfaces_and_restores_the_model(model, panel, monkeypatch, worker_timeout):
    def fail(self, request):
        assert get_worker_info() is not None
        raise RuntimeError("injected episode failure")

    model.train()
    monkeypatch.setattr(EpisodeBuilder, "build", fail)
    with pytest.raises(RuntimeError, match="injected episode failure"):
        v5.forecast_items(
            model, panel, [(0, 40)], HORIZON, context=CONTEXT, batch_episodes=1, workers=1
        )
    assert model.training
