"""In-memory v5 benchmark adapter fixtures and contracts."""

import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.v5_panels import time_axis, window_panels  # noqa: E402
from demand_forecast.bench.yardsticks.runner import Options, SeriesInput, WindowInput  # noqa: E402
from demand_forecast.data.episode import EpisodeBatch, EpisodeBuilder, EpisodeRequest  # noqa: E402
from demand_forecast.data.episode.scale import fit_scale  # noqa: E402
from demand_forecast.data.episode.schema import (  # noqa: E402
    CH_STEP,
    CH_STEP_FLAG,
    CH_VALUE_STATS,
    ROLE_PEER,
    ROLE_SIBLING,
    STEP_FAMILIES,
    SUMMARY_FEATURES,
)
from demand_forecast.models.v5 import V5Config, V5Model, forecast, forecast_items  # noqa: E402
from demand_forecast.models.v5.export import write_export  # noqa: E402
from demand_forecast.pretrain.v5 import (  # noqa: E402
    OptimConfig,
    SamplerConfig,
    V5TrainConfig,
    identity,
    save_checkpoint,
)
from tests.helpers.v5_sampler import paired_static_panel  # noqa: E402

HORIZON = 2
TOL = {"atol": 1e-5, "rtol": 1e-5}
TASK = SimpleNamespace(
    task_name="example", known_dynamic_columns=["plan"], past_dynamic_columns=["event", "past"]
)


def _export(
    directory,
    *,
    drop=(),
    summary=(),
    threshold=0.0,
    fair_spread=False,
    sampler_options=None,
    naive_mix=False,
    mix_evidence=False,
):
    config = V5TrainConfig(
        V5Config(
            d_model=16,
            nhead=2,
            enc_layers=1,
            dec_layers=1,
            dim_feedforward=16,
            expert_enc_layers=1,
            expert_dec_layers=1,
            d_calendar=4,
            d_step=4,
            d_out=4,
            **({"naive_mix": True} if naive_mix else {}),
            **({"mix_evidence": True} if mix_evidence else {}),
        ),
        (),
        SamplerConfig(
            contexts=(32, 48),
            drop_always=drop,
            summary_off=summary,
            screen_no_evidence=threshold,
            max_cells=None if drop or summary else 4_000_000,
            designate_target=bool(drop or summary),
            **(sampler_options or {}),
        ),
        OptimConfig(max_steps=1, warmup_steps=0, ema_decay=None),
        device="cpu",
    )
    if fair_spread:
        config = replace(config, sampler=replace(config.sampler, screen_fair_spread=True))
    config.to_yaml(directory / "training.yaml")
    config = V5TrainConfig.from_yaml(directory / "training.yaml")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(9)
        weights = V5Model(config.model).state_dict()
    checkpoint, _ = save_checkpoint(
        directory,
        {
            "model": weights,
            "export": "model",
            "config": asdict(config),
            "identity": identity(config, code="fixture", corpus=None, init=None, env=None),
            "counters": {"batch": 0, "updates": 0, "saves": 1},
        },
        counter=1,
    )
    dest = directory / "export"
    write_export(checkpoint, dest)
    return dest


@pytest.fixture(scope="module")
def v5_export(tmp_path_factory):
    return _export(tmp_path_factory.mktemp("adapter-export"))


@pytest.fixture(scope="module")
def make_adapter(v5_export):
    from demand_forecast.bench.yardsticks.v5_adapter import V5Adapter

    def make(*, checkpoint=v5_export, task=TASK, **kwargs):
        return V5Adapter(
            Options("v5", str(checkpoint), "native", processing="raw", device="cpu", **kwargs),
            task,
        )

    return make


def _window(n=1, *, targets=1, features=False, length=48, ragged=False):
    names = [f"z{j:02}" for j in reversed(range(targets))]
    items = []
    for i in range(n):
        size = length + (i % 2 if ragged else 0)
        t = np.arange(size, dtype=np.float64)
        dates = pd.date_range("2024-01-01", periods=size + HORIZON, freq="D").to_numpy()
        items.append(
            SeriesInput(
                i,
                tuple(range(targets)),
                np.stack([20 * (j + 1) + (i + 1) * np.sin(t / (2 + j)) for j in range(targets)]),
                {
                    "plan": t / 7,
                    "event": np.take(["closed", "work", "sale"], t.astype(int) % 3),
                    "past": np.where(t % 4 == 0, 0, t / 11),
                }
                if features
                else {},
                {"plan": np.array([np.nan, 10])} if features else {},
                time_axis(dates[:size], "D", future=dates[size:], horizon=HORIZON),
                {"group": "a" if i % 2 else "b", "size": i // 2 + 0.25} if features else {},
            )
        )
    return WindowInput([f"sku-{i}" for i in range(n)], names, items, "D")


def _panels(window):
    return window_panels(
        window,
        name=TASK.task_name,
        horizon=HORIZON,
        known=TASK.known_dynamic_columns,
        past=TASK.past_dynamic_columns,
    )


def test_predict_shapes_order_and_units(make_adapter):
    adapter = make_adapter()
    window = _window(4, targets=2, features=True, ragged=True)
    # A part's item positions refer to the input list, not to the runner's ids.
    window.items = [window.items[i] for i in (2, 0, 3, 1)]
    window.items[1] = replace(window.items[1], axis=None)
    levels = list(adapter.model.config.quantiles)
    (actual,) = adapter.predict([window], HORIZON, levels)
    assert len(actual) == 4
    expected = [None] * len(window.items)
    requests = set()
    for part in _panels(window).parts:
        jobs = list(enumerate(part.origins))
        direct = forecast_items(adapter.model, part.panel, jobs, HORIZON, context=48)
        requests.update((part.positions[i], target) for i, target in direct.evidence.requests)
        for i, position in enumerate(part.positions):
            expected[position] = np.stack([direct.values[i, target] for target in window.targets])
    assert all(a.shape == (2, HORIZON, len(levels)) for a in actual)
    np.testing.assert_allclose(actual, expected, **TOL)
    dated = _panels(window).parts[0]
    builder = EpisodeBuilder(dated.panel)
    single = forecast(
        adapter.model,
        dated.panel,
        EpisodeRequest(0, (dated.origins[0],), HORIZON, 48, window.targets[0]),
        builder=builder,
    )
    assert single.names == [window.targets[0]]
    np.testing.assert_allclose(actual[0][0], single.quantiles[0, 0], **TOL)
    (evidence,) = adapter.evidence()
    assert set(evidence.requests) == requests
    assert len(evidence.requests) == len(requests)
    assert evidence.candidates == sorted(set(evidence.candidates))


@pytest.mark.parametrize("workers,reason", [(0, "injected"), (16, None)])
@pytest.mark.parametrize("with_evidence", [False, True])
def test_predict_uses_preflight_worker_count(
    make_adapter, monkeypatch, workers, reason, with_evidence, tmp_path
):
    from demand_forecast.bench.yardsticks import v5_adapter

    requested_counts, forecast_counts = [], []

    def preflight(requested, **kwargs):
        assert kwargs == ({"naive_anchor": True, "naive_evidence": True} if with_evidence else {})
        requested_counts.append(requested)
        return workers, reason

    def record_forecast(*args, workers, **kwargs):
        forecast_counts.append(workers)
        # Run inference serially so this test does not need Linux workers.
        return forecast_items(*args, workers=0, **kwargs)

    monkeypatch.setattr(v5_adapter, "_worker_count", lambda device: 16)
    monkeypatch.setattr(v5_adapter, "usable_workers", preflight)
    monkeypatch.setattr(v5_adapter, "forecast_items", record_forecast)
    options = {}
    if with_evidence:
        options["checkpoint"] = _export(
            tmp_path,
            sampler_options={"derive": ("naive_anchor", "naive_evidence")},
            naive_mix=True,
            mix_evidence=True,
        )
    adapter = make_adapter(**options)
    [[actual]] = adapter.predict([_window()], HORIZON, [0.5])

    assert actual.shape == (1, HORIZON, 1)
    assert forecast_counts == [workers]
    assert adapter.metadata()["v5_worker_reason"] == reason
    assert adapter.metadata()["v5_workers"] == workers
    assert requested_counts == [16]


def test_anchored_export_and_forecast_agree_on_worker_fallback(tmp_path, make_adapter, monkeypatch):
    from demand_forecast.bench.yardsticks import v5_adapter

    checkpoint = _export(tmp_path, sampler_options={"derive": ("naive_anchor",)}, naive_mix=True)
    plain_need = 256 + 4 * len(fields(EpisodeBatch))
    monkeypatch.setattr(v5_adapter.resource, "getrlimit", lambda key: (plain_need, plain_need))
    monkeypatch.setattr(v5_adapter.sys, "platform", "linux")
    monkeypatch.setattr(v5_adapter, "_worker_count", lambda device: 1)

    adapter = make_adapter(checkpoint=checkpoint)
    [[actual]] = adapter.predict([_window(targets=2)], HORIZON, [0.5])

    assert actual.shape == (2, HORIZON, 1)
    assert np.isfinite(actual).all()
    metadata = adapter.metadata()
    assert metadata["v5_workers"] == 0
    assert "RLIMIT_NOFILE" in metadata["v5_worker_reason"]


def test_levels_are_indexed_or_interpolated(make_adapter):
    adapter = make_adapter()
    window = _window()
    (part,) = _panels(window).parts
    (origin,) = part.origins
    native = forecast(adapter.model, part.panel, EpisodeRequest(0, (origin,), HORIZON, 48))
    levels = adapter.model.config.quantiles
    lo, hi = levels[0], levels[1]
    mid = lo + 0.25 * (hi - lo)
    values = native.quantiles[0, 0]
    [[actual]] = adapter.predict([window], HORIZON, [0.99, mid, hi, 0.01, 0.5])
    expected = np.stack(
        [
            values[:, -1],
            0.75 * values[:, 0] + 0.25 * values[:, 1],
            values[:, 1],
            values[:, 0],
            values[:, levels.index(0.5)],
        ],
        axis=-1,
    )
    np.testing.assert_allclose(actual[0], expected, **TOL)


@pytest.mark.parametrize(
    "case",
    [
        "one-step",
        "all-missing",
        "constant",
        "single-item",
        "seventeen-targets",
        "off-grid",
        "future-off-grid",
        "no-frequency",
        "missing-static",
        "null-history",
        "known-future-gap",
    ],
)
def test_every_degenerate_item_gets_a_finite_forecast(make_adapter, case):
    window = _window(
        1 if case == "single-item" else 3,
        targets=17 if case == "seventeen-targets" else 1,
        length=1 if case == "one-step" else 48,
        features=case in {"missing-static", "null-history", "known-future-gap"},
    )
    item = window.items[0]
    if case == "all-missing":
        item.target[:] = np.nan
    if case == "constant":
        item.target[:] = 7.25
    if case == "off-grid":
        history = pd.to_datetime(["2024-01-01", "2024-01-02 00:30"], format="mixed")
        item.axis = time_axis(history, "D")
    if case == "future-off-grid":
        dates = pd.date_range("2024-01-01", periods=51, freq="D")
        item.axis = time_axis(dates[:48], "D", future=dates[-2:], horizon=HORIZON)
    if case == "no-frequency":
        window.freq = None
    if case == "missing-static":
        item.static["size"] = None
    if case == "null-history":
        window.declared = {"plan": "number"}
        item.past["plan"][:] = np.nan

    adapter = make_adapter()
    (actual,) = adapter.predict([window], HORIZON, [0.1, 0.5, 0.9])
    assert np.asarray(actual).shape == (len(window.items), len(window.targets), HORIZON, 3)
    assert np.isfinite(actual).all()
    if case in {"one-step", "all-missing", "constant"}:
        assert adapter.metadata()["v5_fallback_rows"] == (3 if case == "one-step" else 1)
        expected = {"one-step": 20, "all-missing": 0, "constant": 7.25}[case]
        np.testing.assert_allclose(actual[0], expected, **TOL)
    if case in {"off-grid", "future-off-grid", "no-frequency"}:
        assert adapter.metadata()["v5_items_alone"] == (3 if case == "no-frequency" else 1)


def _capture_episodes(monkeypatch):
    episodes = {}
    build = EpisodeBuilder.build

    def record(builder, request):
        episode = build(builder, request)
        item_id = builder.panel.items[request.item_index].item_id
        episodes[item_id, request.target] = episode
        return episode

    monkeypatch.setattr(EpisodeBuilder, "build", record)
    return episodes


def test_fair_spread_reaches_every_screening_site(make_adapter, tmp_path, monkeypatch):
    legacy = make_adapter()
    calibrated = make_adapter(checkpoint=_export(tmp_path, fair_spread=True))
    panel = paired_static_panel()
    window = _window(len(panel.items))
    for series, item in zip(window.items, panel.items, strict=True):
        series.target = item.targets["y0"][None, :48]
        series.static = item.static
    episodes = _capture_episodes(monkeypatch)

    def score(adapter):
        adapter.predict([window], HORIZON, [0.5])
        episode = episodes["sku-0", None]
        return episode.cand_score[episode.cand_names.index("pair")]

    assert score(calibrated) < score(legacy) == pytest.approx(1.0)


def test_scale_floor_reaches_every_adapter_request(make_adapter, tmp_path, monkeypatch):
    adapter = make_adapter(checkpoint=_export(tmp_path, sampler_options={"scale_floor": 0.01}))
    window = _window(3, targets=2, features=True)
    window.items[0].axis = None
    for series in window.items:
        prefix = 1000 + 1e-9 * np.random.default_rng(series.item_index).normal(size=48)
        series.target = np.stack((prefix, 2 * prefix))
    seen = []
    validate = EpisodeBuilder._validate

    def observe(builder, request):
        seen.append(builder.config.scale_floor)
        return validate(builder, request)

    monkeypatch.setattr(EpisodeBuilder, "_validate", observe)
    episodes = _capture_episodes(monkeypatch)
    adapter.predict([window], HORIZON, [0.5])
    assert len(episodes) == 6
    assert len(seen) > len(episodes) and set(seen) == {0.01}
    for episode in episodes.values():
        np.testing.assert_allclose(episode.scale, 0.01 * np.abs(episode.loc), rtol=1e-12)


def test_off_grid_items_are_forecast_alone_and_the_rest_keep_their_peers(make_adapter, monkeypatch):
    adapter = make_adapter()
    window = _window(10, features=True)
    window.items[4].axis = None
    episodes = _capture_episodes(monkeypatch)
    adapter.predict([window], HORIZON, [0.5])

    assert adapter.metadata()["v5_items_alone"] == 1
    assert adapter.metadata()["v5_items_without_time_axis"] == 1
    assert ROLE_PEER not in episodes["sku-4", None].row_role
    assert all(ROLE_PEER in episodes[f"sku-{i}", None].row_role for i in range(10) if i != 4)
    (evidence,) = adapter.evidence()
    i = evidence.requests.index((4, None))
    assert evidence.status[i, evidence.candidates.index("static:group")] == 0


def test_max_context_limits_the_encoder_not_the_history(make_adapter, monkeypatch):
    window = _window(length=80)
    values = window.items[0].target[0]
    values[:48] += 100
    expected = fit_scale(values, len(values), season=7)
    assert abs(expected.loc - fit_scale(values[-32:], 32, season=7).loc) > 10
    episodes = _capture_episodes(monkeypatch)
    for cap, context in ((37, 32), (5000, 2048)):
        adapter = make_adapter(max_context=cap)
        adapter.predict([window], HORIZON, [0.5])
        episode = episodes["sku-0", None]
        assert episode.x.shape[1:3] == (context // 16, 16)
        assert episode.loc[0] == pytest.approx(expected.loc, abs=1e-5, rel=1e-5)
        assert episode.scale[0] == pytest.approx(expected.scale, abs=1e-5, rel=1e-5)
    with pytest.raises(ValueError, match="max.context"):
        Options("v5", ".", "native", processing="raw", max_context=8).resolved()


REMOVALS = {
    "full": (),
    "no_features": ("plan", "past", "event"),
    "no_known": ("plan",),
    "no_past": ("past", "event"),
    "no_dynamic_categories": ("event",),
    "no_statics": ("group", "size"),
    "no_static_categories": ("group",),
    "no_static_numbers": ("size",),
    "no_peers": (),
    "no_panel_row": (),
    "no_typed_context": (),
    "no_value_stats": (),
    "no_derived": (),
    "no_calendar": (),
    "no_rolling": (),
    "no_lags": (),
    "no_zeros": (),
    "no_step_families": (),
    "no_summary": (),
    "per_target": (),
    "no_siblings": (),
    "gates_one": (),
    "screening_alone": (),
    "router_uniform": (),
}


@pytest.mark.parametrize("name", REMOVALS)
def test_each_variant_changes_only_its_inputs(make_adapter, monkeypatch, name):
    from demand_forecast.bench.yardsticks import v5_adapter as api

    # Two targets really split in full. Even removing all typed inputs preserves this layout.
    window = _window(8, targets=2, features=True)
    # A five-target item without typed context normally forecasts jointly.
    plain = _window(targets=5)
    episodes = _capture_episodes(monkeypatch)
    calls = []
    real = api.forecast_items

    def record(model, panel, jobs, horizon, **kwargs):
        calls.append((panel, kwargs))
        return real(model, panel, jobs, horizon, **kwargs)

    monkeypatch.setattr(api, "forecast_items", record)
    baseline_name = "per_target" if name == "no_siblings" else "full"
    adapter = make_adapter(variant=baseline_name)
    adapter.predict([window, plain], HORIZON, [0.5])
    full_episodes, full_calls = episodes.copy(), calls.copy()
    episodes.clear()
    calls.clear()
    adapter = make_adapter(variant=name)
    adapter.predict([window, plain], HORIZON, [0.5])

    panel, kwargs = calls[0]
    for full_item, item in zip(full_calls[0][0].items, panel.items, strict=True):
        assert set(item.dynamic) == set(full_item.dynamic) - set(REMOVALS[name])
        assert set(item.static) == set(full_item.static) - set(REMOVALS[name])
    assert set(panel.roles) == set(full_calls[0][0].roles) - set(REMOVALS[name])
    assert kwargs["layouts"] == ["per_target"] * 8
    assert calls[1][1]["layouts"] == (
        ["per_target"] if name in {"per_target", "no_siblings"} else ["joint"]
    )
    assert kwargs["gates"] == {"gates_one": "ones", "screening_alone": "evidence"}.get(
        name, "learned"
    )
    assert kwargs["router"] == ("uniform" if name == "router_uniform" else "learned")
    assert kwargs["pruned"] is (name != "gates_one")
    assert kwargs["drop"] == frozenset(
        {
            "no_peers": ("peers",),
            "no_panel_row": ("panel_row",),
            "no_typed_context": (
                "attributes",
                "static_numbers",
                "peers",
                "panel_row",
                "value_stats",
                "value_effects",
            ),
            "no_value_stats": ("value_stats",),
            "no_calendar": ("calendar",),
            "no_rolling": ("rolling",),
            "no_lags": ("lags",),
            "no_zeros": ("zeros",),
            "no_step_families": ("rolling", "lags", "zeros"),
            "no_siblings": ("siblings",),
        }.get(name, ())
    )
    assert kwargs["summary_off"] == (SUMMARY_FEATURES if name == "no_summary" else ())

    # Use sku-1 for the typed case to avoid the plain window's sku-0 key.
    actual = episodes["sku-1", "z00"]
    full = full_episodes["sku-1", "z00"]
    missing = set(full.row_names) - set(actual.row_names)
    added = set(actual.row_names) - set(full.row_names)
    if name in {"no_value_stats", "no_typed_context"}:
        assert "event" in missing
        assert added == {"event=closed", "event=work", "event=sale"} | (
            {"event~effect"} if name == "no_value_stats" else set()
        )
    else:
        assert not added
    if name in {"no_peers", "no_typed_context"}:
        assert ROLE_PEER in full.row_role and ROLE_PEER not in actual.row_role
    if name in {"no_panel_row", "no_typed_context"}:
        assert "__panel__" in missing
    if name == "no_typed_context":
        assert len(full.attr_names) and not actual.attr_names
        assert full.static_num.size and not actual.static_num.size
    if name == "no_calendar":
        assert (full.cal_ids >= 0).any() and (actual.cal_ids == -1).all()
        assert (actual.calh_ids == -1).all()
    for family, span in STEP_FAMILIES.items():
        if name in {f"no_{family}", "no_step_families"}:
            for start in (CH_STEP, CH_STEP_FLAG):
                columns = slice(start + span.start, start + span.stop)
                assert full.x[..., columns].any()
                assert not actual.x[..., columns].any()
    if name == "no_summary":
        assert full.row_summary_flags.any() and full.cand_flags.any()
        for field in ("row_summary", "row_summary_flags", "cand_summary", "cand_flags"):
            assert not getattr(actual, field).any()
        assert not actual.attr_stats[..., :-1].any()
        np.testing.assert_allclose(actual.attr_stats[..., -1], full.attr_stats[..., -1], **TOL)
    if name == "no_siblings":
        for target in plain.targets:
            before, after = full_episodes["sku-0", target], episodes["sku-0", target]
            keep_rows = np.flatnonzero(before.row_role != ROLE_SIBLING)
            assert len(before.row_names) - len(keep_rows) == 4
            assert [before.row_names[i] for i in keep_rows] == after.row_names == [target]
            np.testing.assert_allclose(before.x[keep_rows], after.x, **TOL)
            np.testing.assert_allclose(before.row_summary[keep_rows], after.row_summary, **TOL)
            np.testing.assert_allclose(before.cal_phases, after.cal_phases, **TOL)
            np.testing.assert_array_equal(before.cal_ids, after.cal_ids)


def test_training_drops_are_applied_at_scoring(make_adapter, monkeypatch, tmp_path):
    exported = _export(tmp_path, drop=("peers",), summary=("log_length",))
    adapter = make_adapter(checkpoint=exported, variant="no_panel_row")
    episodes = _capture_episodes(monkeypatch)
    adapter.predict([_window(8, features=True)], HORIZON, [0.5])
    episode = episodes["sku-0", None]
    assert "__panel__" not in episode.row_names and ROLE_PEER not in episode.row_role
    adapter = make_adapter(checkpoint=exported)
    adapter.predict([_window(8, features=True)], HORIZON, [0.5])
    episode = episodes["sku-0", None]
    assert "__panel__" in episode.row_names and ROLE_PEER not in episode.row_role
    index = SUMMARY_FEATURES.index("log_length")
    assert not episode.row_summary_flags[..., index].any()


def test_a_legacy_export_replays_and_scores_unchanged(make_adapter):
    exported = Path(__file__).parents[1] / "fixtures" / "v5_legacy_export"
    expected = np.asarray(json.loads((exported / "forecasts.json").read_text()))
    window = _window(1, features=True)
    for variant in ("full", "no_derived"):
        adapter = make_adapter(checkpoint=exported, variant=variant)
        (actual,) = adapter.predict([window], HORIZON, [0.5])
        assert np.asarray(actual).dtype == expected.dtype
        assert np.asarray(actual).shape == expected.shape
        np.testing.assert_allclose(actual, expected, **TOL)


def test_naive_rows_keep_adapter_layouts(make_adapter, monkeypatch, tmp_path):
    exported = _export(tmp_path, sampler_options={"derive": ("naive_rows",)})
    episodes = _capture_episodes(monkeypatch)
    for count in (1, 5, 6, 17):
        window = _window(targets=count)
        outputs = {}
        for variant in ("full", "no_derived"):
            adapter = make_adapter(checkpoint=exported, variant=variant)
            episodes.clear()
            ((output,),) = adapter.predict([window], HORIZON, [0.5])
            assert output.shape == (count, HORIZON, 1)
            assert np.isfinite(output).all()
            outputs[variant] = output
            layout = set(window.targets) if count == 17 else {None}
            assert {target for _, target in episodes} == layout
            for (_, target), episode in episodes.items():
                assert episode.keep.sum() == (1 if count == 17 else count)
                expected = set()
                if variant == "full" and count in (1, 17):
                    kept = window.targets[0] if count == 1 else target
                    expected = {f"{kept}~{name}" for name in ("last", "season", "drift")}
                assert {c.name for c in episode.trace if c.source == "derived"} == expected
        if count in (5, 6):
            np.testing.assert_allclose(outputs["full"], outputs["no_derived"], **TOL)


def test_derive_reaches_every_request(make_adapter, monkeypatch, tmp_path):
    from demand_forecast.bench.yardsticks import v5_adapter as api

    derive = ("increment_spread", "onset_rows")
    exported = _export(tmp_path, sampler_options={"derive": derive})
    planned = []
    plan = api.planned_layouts

    def observe(*args, **kwargs):
        planned.append(kwargs["derive"])
        return plan(*args, **kwargs)

    monkeypatch.setattr(api, "planned_layouts", observe)
    episodes = _capture_episodes(monkeypatch)
    built = []
    for variant in ("full", "no_derived"):
        adapter = make_adapter(checkpoint=exported, variant=variant)
        adapter.predict([_window(1, features=True)], HORIZON, [0.5])
        built.append(episodes["sku-0", None])
    assert planned == [frozenset(derive)] * 2
    full, off = built
    # The trained presentations reach scoring: the derived row is built, and `no_derived` removes it.
    assert "event~onset" in full.row_names
    assert "event~onset" not in off.row_names
    assert [name for name in full.row_names if name != "event~onset"] == list(off.row_names)
    legacy = [full.row_names.index(name) for name in off.row_names]
    index = off.row_names.index("event")
    spread = CH_VALUE_STATS + 1
    kept = full.x[legacy]
    assert not kept[index, ..., spread].any()  # The window's labels alternate.
    assert off.x[index, ..., spread].any()
    np.testing.assert_array_equal(kept[..., :spread], off.x[..., :spread])
    np.testing.assert_array_equal(kept[..., spread + 1 :], off.x[..., spread + 1 :])


def test_non_default_level_survives_yaml_checkpoint_export_and_adapter(make_adapter, tmp_path):
    exported = _export(tmp_path, threshold=0.95)
    adapter = make_adapter(checkpoint=exported)
    assert adapter.screening.no_evidence == pytest.approx(0.95, abs=1e-9)
    adapter.predict([_window(8, features=True)], HORIZON, [0.5])
    (evidence,) = adapter.evidence()
    group = evidence.candidates.index("static:group")
    np.testing.assert_allclose(evidence.score[:, group], 0, **TOL)


@pytest.mark.parametrize("failed_call", [1, 2])
def test_parity_failure_stops_before_any_forecast(make_adapter, monkeypatch, failed_call):
    from demand_forecast.bench.yardsticks import v5_adapter as api

    calls = []
    parity = api.require_parity

    def check(path, device="cpu"):
        calls.append(str(device))
        if len(calls) == failed_call:
            raise ValueError("v5 parity failed in fixture")
        parity(path, device)

    monkeypatch.setattr(api, "require_parity", check)
    with pytest.raises(ValueError, match="v5 parity failed"):
        make_adapter().predict([_window()], HORIZON, [0.5])
    assert calls == ["cpu"] * failed_call


def test_variant_names_module_matches_the_table():
    from demand_forecast.bench.yardsticks.v5_adapter import VARIANTS
    from demand_forecast.bench.yardsticks.v5_variants import VARIANT_NAMES

    assert VARIANT_NAMES == tuple(REMOVALS) == tuple(VARIANTS)


@pytest.mark.parametrize(("platform", "factor"), [("darwin", 1), ("linux", 1024)])
def test_peak_rss_is_reported_in_bytes(monkeypatch, make_adapter, platform, factor):
    from demand_forecast.bench.yardsticks import v5_adapter as api

    adapter = make_adapter()
    monkeypatch.setattr(api.sys, "platform", platform)
    monkeypatch.setattr(
        api.resource,
        "getrusage",
        lambda who: SimpleNamespace(
            ru_maxrss={api.resource.RUSAGE_SELF: 7, api.resource.RUSAGE_CHILDREN: 13}[who]
        ),
    )
    assert adapter.metadata()["peak_rss_bytes"] == 7 * factor
    assert adapter.metadata()["peak_rss_children_bytes"] == 13 * factor


@pytest.mark.parametrize(
    ("platform", "device", "cores", "expected"),
    [
        ("linux", "cuda", 17, 0),
        ("linux", "cuda", 18, 16),
        ("linux", "cpu", 32, 0),
        ("darwin", "cuda", 32, 0),
    ],
)
def test_workers_use_process_affinity(platform, device, cores, expected, monkeypatch):
    from demand_forecast.bench.yardsticks import v5_adapter as api

    monkeypatch.setattr(api.sys, "platform", platform)
    monkeypatch.setattr(api.os, "sched_getaffinity", lambda pid: range(cores), raising=False)
    assert api._worker_count(device) == expected


def test_naive_anchor_scores_each_target_and_no_derived_runs_head_only(tmp_path, make_adapter):
    checkpoint = _export(tmp_path, sampler_options={"derive": ("naive_anchor",)}, naive_mix=True)
    full = make_adapter(checkpoint=checkpoint)
    per_target = make_adapter(checkpoint=checkpoint, variant="per_target")
    no_derived = make_adapter(checkpoint=checkpoint, variant="no_derived")
    window = _window(targets=2)
    levels = list(full.model.config.quantiles)
    for adapter in (full, per_target, no_derived):
        with torch.no_grad():
            adapter.model.step_projection.weight.zero_()
            adapter.model.step_projection.bias.zero_()
            adapter.model.head.projection.bias.zero_()
            adapter.model.mix_projection.bias.fill_(-50)
            adapter.model.mix_projection.bias[3] = 50
    (anchored,) = full.predict([window], HORIZON, levels)
    (split,) = per_target.predict([window], HORIZON, levels)
    (head,) = no_derived.predict([window], HORIZON, levels)
    np.testing.assert_allclose(anchored, split, **TOL)
    (part,) = _panels(window).parts
    y = window.items[0].target
    season = part.panel.season
    np.testing.assert_allclose(
        anchored[0][..., levels.index(0.5)], y[:, -season:][:, :HORIZON], **TOL
    )
    np.testing.assert_allclose(
        head[0][..., levels.index(0.5)],
        np.median(y, axis=1)[:, None] + np.zeros((2, HORIZON)),
        **TOL,
    )
    assert full.evidence()[0].requests == per_target.evidence()[0].requests
    assert full.evidence()[0].requests == no_derived.evidence()[0].requests
