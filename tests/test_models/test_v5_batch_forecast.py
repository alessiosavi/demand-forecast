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
    collate,
)
from demand_forecast.data.episode.schema import (
    MAX_FEATURE_ROWS,
    MAX_FORECAST_ROWS,
    N_SUMMARY,
    ROLE_KNOWN,
    ROLE_PEER,
    STEP_FAMILIES,
    SUMMARY_FEATURES,
)
from demand_forecast.models import v5
from demand_forecast.models.v5 import predict
from tests.helpers.episodes import lag_panel, perturb_after, toy_panel
from tests.helpers.v5_loop import config, equal

CONTEXT = 32


@pytest.mark.parametrize("layout", [None, "per_target", "joint", "plan"])
def test_derive_reaches_every_request(model, monkeypatch, layout):
    panel = _panel_with_known_category()
    if layout == "joint":
        panel = replace(
            panel, items=tuple(replace(item, dynamic={}) for item in panel.items), roles={}
        )
    derive = frozenset({"increment_spread", "lag_rows"})
    planned, built = [], []
    validate = EpisodeBuilder._validate
    build = EpisodeBuilder.build

    def observe_probe(builder, request):
        planned.append(request.derive)
        return validate(builder, request)

    def observe_build(builder, request):
        built.append(request.derive)
        return build(builder, request)

    monkeypatch.setattr(EpisodeBuilder, "_validate", observe_probe)
    monkeypatch.setattr(EpisodeBuilder, "build", observe_build)
    kwargs = dict(context=CONTEXT, derive=derive)
    if layout == "plan":
        assert v5.planned_layouts(panel, [(0, 40)], 3, **kwargs) == ["per_target"]
        assert planned == [derive] and not built
    else:
        v5.forecast_items(
            model, panel, [(0, 40)], 3, layouts=None if layout is None else [layout], **kwargs
        )
        count = 1 if layout == "joint" else 2
        assert built == [derive] * count
        assert planned == [derive] * (count + (layout != "per_target"))


def test_increment_spread_hidden_targets_leave_forecasts_and_gates_unchanged(model):
    panel = toy_panel()
    origin = 336
    item = panel.items[0]
    event = np.resize(np.repeat(np.array(["a", "b", "c"], dtype=object), 4), len(item.targets["y"]))
    item = replace(item, dynamic={**item.dynamic, "event": event})
    panel = replace(panel, items=(item, *panel.items[1:]))
    hidden = perturb_after(panel, origin, keep_known=True)
    request = EpisodeRequest(0, (origin,), 16, CONTEXT, derive=frozenset({"increment_spread"}))
    before = EpisodeBuilder(panel).build(request)
    after = EpisodeBuilder(hidden).build(request)
    for field in fields(before):
        if field.name == "labels":
            continue
        expected, actual = getattr(before, field.name), getattr(after, field.name)
        if isinstance(expected, np.ndarray):
            assert expected.dtype == actual.dtype
            np.testing.assert_array_equal(actual, expected, err_msg=field.name)
        else:
            assert actual == expected
    full = v5.forecast(model, panel, request)
    changed = v5.forecast(model, hidden, request)
    np.testing.assert_array_equal(changed.quantiles, full.quantiles)
    assert changed.names == full.names and changed.gates == full.gates


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
    if not torch.amp.autocast_mode.is_autocast_available("cpu"):
        pytest.skip("CPU autocast is unavailable")
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


def test_explicit_layouts_hold_statistics_of_one_origin_at_a_time(model, monkeypatch):
    panel = _panel(n_items=6, n_targets=2)
    panel = replace(panel, items=tuple(replace(item, static={}) for item in panel.items))
    jobs = [(index, 40 + index) for index in range(6)]
    expected = v5.forecast_items(model, panel, jobs, 3, context=CONTEXT)
    cached_cutoffs = []
    real_get = PanelStatsProvider.get

    def get(self, target, cutoff):
        stats = real_get(self, target, cutoff)
        cached_cutoffs.append({cached_cutoff for _, cached_cutoff in self._cache})
        return stats

    monkeypatch.setattr(PanelStatsProvider, "get", get)
    actual = v5.forecast_items(model, panel, jobs, 3, context=CONTEXT, layouts=["joint"] * 6)

    assert max(map(len, cached_cutoffs)) == 1
    assert actual.stats_builds == expected.stats_builds
    assert actual.fallback == expected.fallback
    assert actual.values.keys() == expected.values.keys()
    for key in expected.values:
        np.testing.assert_array_equal(actual.values[key], expected.values[key])


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


@pytest.mark.parametrize(
    "jobs, layouts, message",
    [
        ([(0, 40), (0, 41)], None, "one origin per item"),
        ([(4, 40)], None, "item index outside the panel"),
        ([(-1, 40)], None, "item index outside the panel"),
        ([(0, 40)], ["joint", "joint"], "layouts must give"),
        ([(0, 40)], ["both"], "layouts must give"),
    ],
)
def test_invalid_jobs_and_layouts_are_refused_before_the_model_is_touched(
    model, monkeypatch, jobs, layouts, message
):
    monkeypatch.setattr(model, "eval", lambda: pytest.fail("the model was touched"))
    with pytest.raises(ValueError, match=message):
        v5.forecast_items(model, _panel(), jobs, 3, context=CONTEXT, layouts=layouts)


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
    "jobs,layouts,earlier_origins",
    [
        ([(0, 40)], ["joint"], ()),
        ([(2, 44), (0, 40), (1, 36)], ["joint", "joint", "per_target"], (36,)),
        ([(1, 36), (0, 40)], ["per_target", "joint"], (36,)),
    ],
    ids=["single-job", "unsorted-jobs", "later-origin"],
)
def test_incompatible_joint_layout_is_refused_before_forecasting_its_origin(
    model, monkeypatch, jobs, layouts, earlier_origins
):
    panel = _panel_with_known_category()
    batches = _record_batches(model, monkeypatch)
    built_at = []
    real_get = PanelStatsProvider.get

    def get(self, target, cutoff):
        before = self.builds
        stats = real_get(self, target, cutoff)
        if self.builds != before:
            built_at.append(cutoff)
        return stats

    monkeypatch.setattr(PanelStatsProvider, "get", get)

    with pytest.raises(
        ValueError, match=r"^item index 0 at origin 40 must be forecast per target$"
    ):
        v5.forecast_items(model, panel, jobs, 3, context=CONTEXT, layouts=layouts)

    assert model.training
    assert [batch.context_target for batch, _ in batches] == [["y", "y1"] for _ in earlier_origins]
    assert set(built_at) == {*earlier_origins, 40}


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


def test_evidence_of_a_hand_built_panel(model, monkeypatch):
    panel = _panel(n_items=3, length=48)
    known = [f"known_{i:02d}" for i in range(MAX_FEATURE_ROWS - 2)] + ["promo"]
    items = []
    for i, item in enumerate(panel.items):
        target = item.targets["y"]
        rare = np.full_like(target, np.nan)
        rare[:2] = (0, 1)
        items.append(
            replace(
                item,
                dynamic={
                    **{name: target.copy() for name in known},
                    "constant": np.zeros_like(target),
                    "past": target.copy(),
                    "rare": rare,
                },
                static={"promo": "shared"} if i < 2 else {},
            )
        )
    panel = replace(
        panel,
        items=tuple(items),
        roles=dict.fromkeys(known, "known") | dict.fromkeys(("constant", "past", "rare"), "past"),
    )
    builder = EpisodeBuilder(panel)
    jobs = [(0, 40), (2, 40)]
    requests = [EpisodeRequest(index, (origin,), 3, CONTEXT) for index, origin in jobs]
    episodes = [builder.build(request) for request in requests]
    assert {c.name: c.removed for c in episodes[0].trace if c.source == "past"} == {
        "constant": "constant",
        "past": None,
        "rare": "budget",
    }
    with torch.no_grad():
        model.gate_net.net[-1].weight.zero_()
        model.gate_net.net[-1].bias.fill_(-20)
        model.family_logits.weight.zero_()
        model.family_logits.bias.fill_(20)
        model.family_logits.bias[list(STEP_FAMILIES).index("lags")] = -20
    expected, scales = _reference(model, panel, builder, requests)
    reference = v5.forecast(model, panel, requests[0], builder=builder)

    forwarded = []
    forward = model.forward

    def record_forward(batch, *args, **kwargs):
        forwarded.append(batch.row_names)
        return forward(batch, *args, **kwargs)

    monkeypatch.setattr(model, "forward", record_forward)
    for pruned in (True, False):
        result = v5.forecast_items(model, panel, jobs, 3, context=CONTEXT, pruned=pruned)
        _assert_forecasts(result.values, expected, scales)
        evidence = result.evidence
        assert evidence.requests == [(0, None), (2, None)]
        assert evidence.candidates == [
            "dynamic:constant",
            *[f"dynamic:{name}" for name in known[:-1]],
            "dynamic:past",
            "dynamic:promo",
            "dynamic:rare",
            "family:lags",
            "family:rolling",
            "family:zeros",
            "static:promo",
        ]
        assert evidence.rules == ("constant",)
        assert evidence.status.dtype == evidence.rule.dtype == np.int8
        assert evidence.gate.dtype == evidence.score.dtype == np.float32
        zero = 5 if pruned else 4
        statuses = [2, *([1] * len(known[:-1])), zero, 1, 3, 4, 1, 1]
        np.testing.assert_array_equal(evidence.status, [statuses + [zero], statuses + [0]])
        np.testing.assert_array_equal(evidence.rule, [[0] + [-1] * (len(statuses))] * 2)
        assert ("past" in forwarded[-1][0]) is not pruned
        for row, episode in enumerate(episodes):
            for candidate in episode.trace:
                source = "dynamic" if candidate.source in ("known", "past") else candidate.source
                column = evidence.candidates.index(f"{source}:{candidate.name}")
                assert evidence.score[row, column] == pytest.approx(candidate.score)
        column = {name: i for i, name in enumerate(evidence.candidates)}
        for name in ("dynamic:constant", "dynamic:rare"):
            assert np.isnan(evidence.gate[:, column[name]]).all()
        assert np.isnan(evidence.gate[1, column["static:promo"]])
        assert np.isnan(evidence.score[1, column["static:promo"]])
        for name, report_key in (
            ("dynamic:past", "row:past"),
            ("dynamic:promo", "row:promo"),
            ("static:promo", "candidate:promo"),
            ("family:lags", "family:lags"),
        ):
            assert evidence.gate[0, column[name]] == pytest.approx(reference.gates[report_key])


def test_evidence_rows_follow_forecast_order_for_per_target_layouts(model):
    panel = _panel(n_items=3, n_targets=2)
    panel = replace(panel, items=tuple(replace(item, static={}) for item in panel.items))
    jobs = [(2, 44), (1, 36), (0, 36)]
    order = [(1, None), (0, "y"), (0, "y1"), (2, "y"), (2, "y1")]
    result = v5.forecast_items(
        model,
        panel,
        jobs,
        3,
        context=CONTEXT,
        layouts=["per_target", "joint", "per_target"],
        gates="evidence",
        batch_episodes=1,
    )
    assert result.evidence.requests == order
    assert list(result.values) == [(i, target) for i in (1, 0, 2) for target in ("y", "y1")]
    assert result.evidence.candidates == [
        "dynamic:promo",
        "family:lags",
        "family:rolling",
        "family:zeros",
        "sibling:y",
        "sibling:y1",
    ]
    np.testing.assert_array_equal(
        result.evidence.status[:, -2:], [[0, 0], [0, 1], [1, 0], [0, 1], [1, 0]]
    )
    builder = EpisodeBuilder(panel)
    for index, target in order:
        origin = dict(jobs)[index]
        request = EpisodeRequest(index, (origin,), 3, CONTEXT, target)
        expected = v5.forecast(model, panel, request, builder=builder, gates="evidence")
        for row, name in enumerate(expected.names):
            scale = builder.stats.get(name, origin).scales[index].scale
            np.testing.assert_allclose(
                (result.values[index, name] - expected.quantiles[0, row]) / scale,
                0,
                atol=1e-5,
                rtol=0,
            )


def test_evidence_keeps_unrepresented_candidate(model):
    panel = _panel(n_items=1, length=48)
    (item,) = panel.items
    event = np.array([None] * 40 + ["future"] * 8, dtype=object)
    panel = replace(
        panel,
        items=(replace(item, dynamic={"event": event, "event=future": item.targets["y"]}),),
        roles={"event": "known", "event=future": "known"},
    )
    drop = frozenset({"value_stats", "value_effects"})
    request = EpisodeRequest(0, (40,), 3, CONTEXT, drop=drop)
    episode = EpisodeBuilder(panel).build(request)
    candidate = next(c for c in episode.trace if c.name == "event")
    assert candidate.removed is None
    assert "event" not in episode.row_names and "event" not in episode.cand_names
    assert "event=future" in episode.row_names

    result = v5.forecast_items(
        model, panel, [(0, 40)], 3, context=CONTEXT, drop=drop, gates="evidence"
    )
    missing, represented = (
        result.evidence.candidates.index(name) for name in ("dynamic:event", "dynamic:event=future")
    )
    np.testing.assert_array_equal(result.evidence.status[0, [missing, represented]], [1, 1])
    np.testing.assert_array_equal(result.evidence.gate[0, [missing, represented]], [np.nan, 1])
    assert result.evidence.score[0, missing] == pytest.approx(candidate.score)


def _panel_with_colliding_row_names():
    panel = _panel(n_items=1, length=48)
    (item,) = panel.items
    target = np.arange(48, dtype=np.float64)
    category = np.resize(np.array(["b", "c", "d", "e"], dtype=object), len(target))
    item = replace(item, targets={"y": target}, dynamic={"a": category, "a=b": target.copy()})
    return replace(panel, items=(item,), roles={"a": "past", "a=b": "past"})


def test_evidence_reports_maximum_learned_gate_for_expanded_category(model):
    panel = _panel_with_colliding_row_names()
    drop = frozenset({"value_stats"})
    episode = EpisodeBuilder(panel).build(EpisodeRequest(0, (40,), 3, CONTEXT, drop=drop))
    batch = collate([episode])
    with torch.no_grad():
        # The last layer starts with zero weights, which gives every past row the same gate.
        model.gate_net.net[-1].weight.fill_(0.25)
        model.gate_net.net[-1].bias.zero_()
        row_gates = model.eval().gates(batch).row[0]

    rows = [i for i, key in enumerate(episode.row_keys) if key == ("past", "a")]
    expanded_gates = row_gates[rows]
    assert len(expanded_gates) == 5
    assert expanded_gates.min() < expanded_gates.max()
    evidence = v5.forecast_items(
        model, panel, [(0, 40)], 3, context=CONTEXT, drop=drop, gates="learned"
    ).evidence
    column = evidence.candidates.index("dynamic:a")
    assert evidence.gate[0, column] == expanded_gates.max().item()
    assert evidence.status[0, column] == 1


@pytest.mark.parametrize("pruned, zero_status", [(False, 4), (True, 5)])
def test_evidence_keeps_colliding_row_names_with_their_candidates(model, pruned, zero_status):
    panel = _panel_with_colliding_row_names()
    screening = ScreeningConfig(no_evidence=0.5)
    drop = frozenset({"value_stats"})
    episode = EpisodeBuilder(panel, screening).build(
        EpisodeRequest(0, (40,), 3, CONTEXT, drop=drop)
    )
    assert episode.row_names.count("a=b") == 2
    literal_rows = [i for i, key in enumerate(episode.row_keys) if key and key[1] == "a=b"]
    category_rows = [i for i, key in enumerate(episode.row_keys) if key and key[1] == "a"]
    assert [episode.row_names[i] for i in literal_rows] == ["a=b"]
    assert "a=b" in [episode.row_names[i] for i in category_rows]

    evidence = v5.forecast_items(
        model,
        panel,
        [(0, 40)],
        3,
        context=CONTEXT,
        screening=screening,
        drop=drop,
        gates="evidence",
        pruned=pruned,
    ).evidence
    category = evidence.candidates.index("dynamic:a")
    literal = evidence.candidates.index("dynamic:a=b")
    assert evidence.score[0, category] == 0 < evidence.score[0, literal]
    np.testing.assert_array_equal(evidence.gate[0, [category, literal]], [0, 1])
    np.testing.assert_array_equal(evidence.status[0, [category, literal]], [zero_status, 1])


def test_evidence_rejects_episode_without_row_keys(model, monkeypatch):
    build = EpisodeBuilder.build

    def without_keys(builder, request):
        episode = build(builder, request)
        episode.row_keys = ()
        return episode

    monkeypatch.setattr(EpisodeBuilder, "build", without_keys)
    with pytest.raises(ValueError, match=r"Episode\.row_keys has length 0 for 2 rows"):
        v5.forecast_items(model, _panel(n_items=1), [(0, 40)], 3, context=CONTEXT)


def test_gates_ones_and_uniform_router_reach_the_model(model, monkeypatch):
    if not torch.amp.autocast_mode.is_autocast_available("cpu"):
        pytest.skip("CPU autocast is unavailable")
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


def test_base_evidence_ignores_its_derived_row(model):
    panel = lag_panel(start=19)
    origins = (531, 547)
    derive = frozenset({"lag_rows"})
    with torch.no_grad():
        model.gate_net.net[-1].weight.zero_()
        model.gate_net.net[-1].bias.fill_(-20)
    evidence = v5.forecast_items(
        model, panel, [(0, origins[0])], 12, context=CONTEXT, derive=derive, pruned=True
    ).evidence
    base = evidence.candidates.index("dynamic:driver")
    lag = evidence.candidates.index("derived:driver~lag")
    np.testing.assert_array_equal(evidence.status[0, [base, lag]], [5, 1])
    assert evidence.gate[0, base] == 0
    assert evidence.gate[0, lag] == pytest.approx(model.gate_net.known_gate_floor)
    assert evidence.score[0, lag] > evidence.score[0, base]

    request = EpisodeRequest(0, origins, 12, CONTEXT, derive=derive)
    original = v5.forecast(model, panel, request)
    modified = v5.forecast(model, perturb_after(panel, origins[0]), request)
    assert original.gates == modified.gates
    np.testing.assert_array_equal(original.quantiles[0], modified.quantiles[0])
