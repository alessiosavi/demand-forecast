"""B2b: bounded forecasts with one cutoff per item."""

from copy import deepcopy
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from demand_forecast.data.episode import (
    EpisodeBuilder,
    EpisodeRequest,
    PanelStatsProvider,
    ScreeningConfig,
)
from demand_forecast.data.episode.schema import (
    MAX_FORECAST_ROWS,
    N_SUMMARY,
    ROLE_KNOWN,
    ROLE_PEER,
    SUMMARY_FEATURES,
)
from demand_forecast.models import v5
from demand_forecast.models.v5 import predict
from tests.helpers.episodes import perturb_after, toy_panel
from tests.helpers.v5_loop import config, equal

CONTEXT = 32


def _panel(*, n_items=4, length=64, n_targets=1):
    panel = toy_panel(n_items=n_items, length=length, seed=37, n_targets=n_targets)
    return replace(
        panel,
        items=tuple(
            replace(item, static={"brand": "shared"}, dynamic={"promo": item.dynamic["promo"]})
            for item in panel.items
        ),
        roles={"promo": "known"},
    )


def _panel_with_known_category():
    panel = toy_panel(n_items=3, length=64, seed=9, n_targets=2)
    return replace(panel, items=tuple(replace(item, static={}) for item in panel.items))


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261118)
        return v5.V5Model(replace(config().model, n_experts=2))


def _reference(model, panel, builder, requests):
    values, scales = {}, {}
    for request in requests:
        result = v5.forecast(model, panel, request, builder=builder)
        for row, name in enumerate(result.names):
            key = request.item_index, name
            values[key] = result.quantiles[0, row]
            scales[key] = builder.stats.get(name, request.origins[0]).scales[key[0]].scale
    return values, scales


def _planned_reference(model, panel, jobs, horizon):
    builder = EpisodeBuilder(panel)
    requests = (
        request
        for index, origin in jobs
        for request in builder.plan_requests(
            index, (origin,), horizon, context=CONTEXT, cutoff=origin
        )
    )
    return _reference(model, panel, builder, requests)


def _assert_forecasts(actual, expected, scales, tolerance=1e-5):
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key].shape == expected[key].shape
        assert np.isfinite(actual[key]).all()
        np.testing.assert_allclose(
            (actual[key] - expected[key]) / scales[key], 0, atol=tolerance, rtol=0
        )


def _record_batches(model, monkeypatch):
    batches = []
    real_gates = model.gates

    def gates(batch, *args, **kwargs):
        selected = real_gates(batch, *args, **kwargs)
        batches.append((batch, selected))
        return selected

    monkeypatch.setattr(model, "gates", gates)
    return batches


def test_equals_single_episode_forecast(model):
    panel = _panel()
    jobs = [(i, 40) for i in range(len(panel.items))]
    builder = EpisodeBuilder(panel)
    episode = builder.build(EpisodeRequest(0, (40,), 4, CONTEXT, cutoff=40))
    assert episode.cand_names == ["brand"]
    assert (episode.row_role == ROLE_PEER).any()
    assert episode.row_names[episode.known_rows[0]] == "promo"
    assert episode.row_role[episode.known_rows[0]] == ROLE_KNOWN
    assert panel.epoch is not None
    expected, scales = _planned_reference(model, panel, jobs, 4)

    actual = v5.forecast_items(model, panel, jobs, 4, context=CONTEXT)

    _assert_forecasts(actual.values, expected, scales)
    assert model.training


def test_float64_forecasts_match_float32_and_single_episode(model, monkeypatch):
    panel = _panel(n_items=2, n_targets=2)
    jobs = [(0, 40), (1, 44)]
    double_model = deepcopy(model).double()
    single = v5.forecast_items(model, panel, jobs, 3, context=CONTEXT)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        double = v5.forecast_items(double_model, panel, jobs, 3, context=CONTEXT)

    # The single-episode API takes its input dtype from the episode arrays.
    # Feed it double arrays while leaving its forecast and inverse transform intact.
    real_collate = predict.collate

    def collate_double(episodes):
        for episode in episodes:
            for field in fields(episode):
                value = getattr(episode, field.name)
                if isinstance(value, np.ndarray) and value.dtype == np.float32:
                    setattr(episode, field.name, value.astype(np.float64))
        return real_collate(episodes)

    monkeypatch.setattr(predict, "collate", collate_double)
    expected, scales = _planned_reference(double_model, panel, jobs, 3)
    assert all(value.dtype == np.float64 for value in double.values.values())
    _assert_forecasts(double.values, single.values, scales, tolerance=1e-6)
    _assert_forecasts(double.values, expected, scales, tolerance=1e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
def test_mixed_or_nonfloating_parameter_dtypes_are_refused(model, dtype):
    model.query = torch.nn.Parameter(model.query.detach().to(dtype), requires_grad=False)

    with pytest.raises(ValueError, match="floating-point dtype") as error:
        v5.forecast_items(model, _panel(), [(0, 40)], 3, context=CONTEXT)

    assert str(torch.float32) in str(error.value)
    assert str(dtype) in str(error.value)


def test_matches_the_old_forecast_panel(model):
    panel = _panel(n_items=2, length=48)
    jobs = [(0, 40), (1, 40)]
    # measure.forecast_panel at 508a326, CPU, batch_episodes=64, rounded to 8 decimals.
    expected = {
        (0, "y"): np.array(
            [
                [
                    10.07089271,
                    10.26492358,
                    10.39573302,
                    10.52353976,
                    10.62071368,
                    10.71751861,
                    10.91330736,
                    11.07356927,
                    11.22793801,
                ]
            ]
        ),
        (1, "y"): np.array(
            [
                [
                    11.22908448,
                    11.42113429,
                    11.54908213,
                    11.67332637,
                    11.76840220,
                    11.86392812,
                    12.05653628,
                    12.21524327,
                    12.36734965,
                ]
            ]
        ),
    }
    oracle, scales = _planned_reference(model, panel, jobs, 1)
    actual = v5.forecast_items(model, panel, jobs, 1, context=CONTEXT, batch_episodes=64)

    _assert_forecasts(actual.values, expected, scales, tolerance=1e-6)
    _assert_forecasts(actual.values, oracle, scales)


@pytest.mark.parametrize("batch_episodes", [1, 3, 64])
def test_batch_sizes_agree_within_tolerance(model, batch_episodes):
    panel = _panel(n_targets=2)
    items = panel.items
    panel = replace(panel, items=(*items[:-1], replace(items[-1], static={})))
    jobs = [(i, 41) for i in range(len(items))]
    expected, scales = _planned_reference(model, panel, jobs, 4)

    actual = v5.forecast_items(
        model, panel, jobs, 4, context=CONTEXT, batch_episodes=batch_episodes
    )

    _assert_forecasts(actual.values, expected, scales)


def test_per_item_origins_use_their_own_cutoff(model, monkeypatch):
    panel = _panel(n_targets=2)
    jobs = [(2, 46), (0, 30), (1, 39), (3, 30)]
    origins = dict(jobs)
    horizon = 3
    items = []
    for index, item in enumerate(panel.items):
        history = origins[index] - item.start
        items.append(
            replace(
                item,
                targets={
                    name: np.pad(values[:history], (0, horizon), constant_values=np.nan)
                    for name, values in item.targets.items()
                },
                dynamic={
                    name: values[: history + horizon] for name, values in item.dynamic.items()
                },
            )
        )
    panel = replace(panel, items=tuple(items))
    expected, scales = _planned_reference(model, panel, jobs, horizon)
    events = []
    real_build, real_release = EpisodeBuilder.build, PanelStatsProvider.release

    def build(self, request):
        events.append(("build", request.cutoff))
        assert request.origins == (origins[request.item_index],)
        assert request.cutoff == origins[request.item_index]
        return real_build(self, request)

    def release(self, cutoff):
        events.append(("release", cutoff))
        real_release(self, cutoff)

    monkeypatch.setattr(EpisodeBuilder, "build", build)
    monkeypatch.setattr(PanelStatsProvider, "release", release)
    actual = v5.forecast_items(model, panel, jobs, horizon, context=CONTEXT, batch_episodes=3)

    _assert_forecasts(actual.values, expected, scales)
    distinct = sorted(set(origins.values()))
    assert actual.stats_builds == len(distinct) * len(panel.items[0].targets) == 6
    expected_events = []
    for origin in distinct:
        expected_events.extend([("build", origin)] * (2 * list(origins.values()).count(origin)))
        expected_events.append(("release", origin))
    assert events == expected_events


def test_later_observations_of_another_item_do_not_change_a_forecast(model, monkeypatch):
    panel = _panel()
    jobs = [(0, 48), (1, 40)]
    with torch.no_grad():
        model.gate_net.net[-1].weight.zero_()
        model.gate_net.net[-1].bias.fill_(20)
    expected, scales = _planned_reference(model, panel, jobs, 4)
    batches = _record_batches(model, monkeypatch)

    def forecast_b(source):
        batches.clear()
        result = v5.forecast_items(model, source, jobs, 4, context=CONTEXT, batch_episodes=1)
        batch, gates = batches[0]
        peer = (batch.row_role[0] == ROLE_PEER).nonzero().item()
        assert gates.row[0, peer].item() == 1
        return batch, batch.x[0, peer], result.values[1, "y"]

    batch, peer, values = forecast_b(panel)
    later = perturb_after(panel, 40, keep_known=True, only_items=(0,))
    later_batch, _, later_values = forecast_b(later)
    for field in fields(batch):
        equal(getattr(batch, field.name), getattr(later_batch, field.name))
    np.testing.assert_array_equal(later_values, values)
    key = (1, "y")
    _assert_forecasts({key: later_values}, {key: expected[key]}, scales)

    earlier = perturb_after(panel, 39, keep_known=True, only_items=(0,))
    _, earlier_peer, _ = forecast_b(earlier)
    assert not torch.equal(peer, earlier_peer)


def test_layouts_are_honoured(model, monkeypatch):
    panel = _panel(n_targets=2)
    bare = replace(panel, items=tuple(replace(item, static={}) for item in panel.items))
    jobs = [(0, 40)]
    assert v5.planned_layouts(panel, jobs, 3, context=CONTEXT) == ["per_target"]
    assert v5.planned_layouts(bare, jobs, 3, context=CONTEXT) == ["joint"]
    typed_names, _ = _planned_reference(model, panel, jobs, 3)
    designated = [EpisodeRequest(0, (40,), 3, CONTEXT, target, cutoff=40) for target in ("y", "y1")]
    expected, scales = _reference(model, bare, EpisodeBuilder(bare), designated)
    batches = _record_batches(model, monkeypatch)

    actual = v5.forecast_items(model, bare, jobs, 3, context=CONTEXT, layouts=["per_target"])

    assert batches[0][0].context_target == ["y", "y1"]
    assert actual.values.keys() == typed_names.keys()
    _assert_forecasts(actual.values, expected, scales)

    # Layouts describe input jobs, whose origins may arrive out of order.
    mixed = v5.forecast_items(
        model, bare, [(0, 40), (1, 36)], 3, context=CONTEXT, layouts=["per_target", "joint"]
    )
    assert batches[1][0].context_target == [None]
    assert batches[2][0].context_target == ["y", "y1"]
    mixed_expected, mixed_scales = _reference(
        model, bare, EpisodeBuilder(bare), [EpisodeRequest(1, (36,), 3, CONTEXT), *designated]
    )
    _assert_forecasts(mixed.values, mixed_expected, mixed_scales)


@pytest.mark.parametrize(
    "drop,layout,context_targets",
    [(frozenset(), "per_target", ["y", "y1"]), (frozenset({"features"}), "joint", [None])],
)
def test_planned_layouts_match_forecasts_with_feature_drop(
    model, monkeypatch, drop, layout, context_targets
):
    panel = _panel_with_known_category()
    jobs = [(0, 40)]
    batches = _record_batches(model, monkeypatch)

    v5.forecast_items(model, panel, jobs, 3, context=CONTEXT, drop=drop)

    assert [batch.context_target for batch, _ in batches] == [context_targets]
    assert v5.planned_layouts(panel, jobs, 3, context=CONTEXT, drop=drop) == [layout]


@pytest.mark.parametrize(
    "jobs,layouts",
    [
        ([(0, 40)], ["joint"]),
        ([(2, 44), (0, 40), (1, 36)], ["joint", "joint", "per_target"]),
    ],
    ids=["single-job", "unsorted-jobs"],
)
def test_incompatible_joint_layout_is_refused_before_forecasting(model, monkeypatch, jobs, layouts):
    panel = _panel_with_known_category()
    batches = _record_batches(model, monkeypatch)
    mode_changes, built_at = [], []
    real_train, real_get = model.train, PanelStatsProvider.get

    def train(mode=True):
        mode_changes.append(mode)
        return real_train(mode)

    def get(self, target, cutoff):
        before = self.builds
        stats = real_get(self, target, cutoff)
        if self.builds != before:
            built_at.append(cutoff)
        return stats

    monkeypatch.setattr(model, "train", train)
    monkeypatch.setattr(PanelStatsProvider, "get", get)

    with pytest.raises(ValueError, match=r"item index 0.*origin 40.*must be forecast per target"):
        v5.forecast_items(model, panel, jobs, 3, context=CONTEXT, layouts=layouts)

    assert not mode_changes
    assert model.training
    assert not batches
    assert 44 not in built_at


def test_more_than_sixteen_targets_always_split(model):
    panel = _panel(n_items=1, n_targets=MAX_FORECAST_ROWS + 1)
    jobs = [(0, 40)]
    assert v5.planned_layouts(panel, jobs, 1, context=CONTEXT) == ["per_target"]
    expected, scales = _planned_reference(model, panel, jobs, 1)

    actual = v5.forecast_items(model, panel, jobs, 1, context=CONTEXT, layouts=["joint"])

    assert len(actual.values) == MAX_FORECAST_ROWS + 1
    _assert_forecasts(actual.values, expected, scales)


def test_drop_and_summary_off_change_the_named_inputs_only(model, monkeypatch):
    panel = _panel()
    batches = _record_batches(model, monkeypatch)
    v5.forecast_items(model, panel, [(0, 40)], 4, context=CONTEXT)
    v5.forecast_items(
        model,
        panel,
        [(0, 40)],
        4,
        context=CONTEXT,
        drop=frozenset({"calendar"}),
        summary_off=("log_length",),
    )
    (full, _), (ablated, _) = batches
    calendar = {"cal_phases", "cal_ids", "calh_phases", "calh_ids"}
    summary = {"row_summary", "row_summary_flags", "cand_summary", "cand_flags", "attr_stats"}
    j = SUMMARY_FEATURES.index("log_length")
    for field in fields(full):
        name = field.name
        original, actual = getattr(full, name), getattr(ablated, name)
        if name in calendar:
            expected = torch.full_like(original, -1 if name.endswith("ids") else 0)
        elif name in summary:
            expected = original.clone()
            expected[..., j] = 0
            if name == "attr_stats":
                expected[..., N_SUMMARY + j] = 0
        else:
            equal(original, actual)
            continue
        assert not torch.equal(original, expected), name
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_screening_no_evidence_reaches_the_model(model, monkeypatch):
    panel = _panel()
    panel = replace(
        panel,
        items=tuple(
            replace(item, targets={"y": item.dynamic["promo"].copy()}) for item in panel.items
        ),
    )
    batches = _record_batches(model, monkeypatch)
    for screening in (ScreeningConfig(), ScreeningConfig(no_evidence=2)):
        v5.forecast_items(model, panel, [(0, 40)], 4, context=CONTEXT, screening=screening)
    (full, _), (screened, _) = batches
    known = full.row_names[0].index("promo")
    assert full.row_score[0, known] > 0
    assert screened.row_names[0].index("promo") == known
    assert screened.row_score[0, known] == 0


def test_gates_ones_and_uniform_router_reach_the_model(model, monkeypatch):
    panel = _panel()
    with torch.no_grad():
        model.gate_net.net[-1].bias.fill_(-20)
        model.family_logits.bias.fill_(-20)
        model.global_mask.logits.fill_(-20)
    forwards = []
    real_gates, real_forward = model.gates, model.forward

    def gates(*args, **kwargs):
        assert not torch.is_autocast_enabled("cpu")
        return real_gates(*args, **kwargs)

    def forward(*args, **kwargs):
        assert not torch.is_autocast_enabled("cpu")
        output = real_forward(*args, **kwargs)
        forwards.append(output)
        return output

    monkeypatch.setattr(model, "gates", gates)
    monkeypatch.setattr(model, "forward", forward)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        v5.forecast_items(
            model,
            panel,
            [(0, 40)],
            4,
            context=CONTEXT,
            gates="ones",
            router="uniform",
            pruned=False,
        )
        assert torch.is_autocast_enabled("cpu")
    (output,) = forwards
    assert output.quantiles.dtype == torch.float32
    for weights in (
        output.gates.row,
        output.gates.cand,
        output.gates.family,
        output.gates.summary_mask,
    ):
        torch.testing.assert_close(weights, torch.ones_like(weights), atol=0, rtol=0)
    torch.testing.assert_close(output.router, torch.full_like(output.router, 0.5), atol=0, rtol=0)


def test_fallback_rows_are_reported(model):
    panel = _panel()
    items = list(panel.items)
    items[0] = replace(items[0], targets={"y": np.full_like(items[0].targets["y"], 12.0)})
    items[1] = replace(items[1], targets={"y": np.full_like(items[1].targets["y"], np.nan)})
    panel = replace(panel, items=tuple(items))
    jobs = [(0, 40), (1, 40), (2, items[2].start + 1), (3, 40)]
    expected, scales = _planned_reference(model, panel, jobs, 3)

    actual = v5.forecast_items(model, panel, jobs, 3, context=CONTEXT)

    _assert_forecasts(actual.values, expected, scales)
    assert actual.fallback == frozenset({(0, "y"), (1, "y"), (2, "y")})
    np.testing.assert_array_equal(actual.values[0, "y"], np.full((3, 9), 12.0))
    np.testing.assert_array_equal(actual.values[1, "y"], np.zeros((3, 9)))


def test_rolled_router_is_refused(model):
    with pytest.raises(ValueError, match="rolled"):
        v5.forecast_items(model, _panel(), [(0, 40)], 4, context=CONTEXT, router="rolled")


@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("nonfinite", [False, True], ids=["forward-error", "nonfinite"])
def test_training_mode_is_restored_after_an_error(model, monkeypatch, training, nonfinite):
    panel = _panel()
    model.train(training)
    real_forward = model.forward

    def fail(*args, **kwargs):
        assert not model.training
        if nonfinite:
            output = real_forward(*args, **kwargs)
            return replace(output, quantiles=torch.full_like(output.quantiles, float("nan")))
        raise RuntimeError("injected forward failure")

    monkeypatch.setattr(model, "forward", fail)
    error = pytest.raises(
        ValueError if nonfinite else RuntimeError,
        match="item_0.*'y'" if nonfinite else "injected forward failure",
    )
    with error:
        v5.forecast_items(model, panel, [(0, 40)], 4, context=CONTEXT)
    assert model.training is training
