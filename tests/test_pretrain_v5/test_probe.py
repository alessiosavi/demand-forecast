"""The absolute probe has fixed inputs and working positive and negative controls."""

import hashlib
import importlib
import json
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from demand_forecast.data.episode.schema import QUANTILES
from demand_forecast.models.v5 import V5Model, forecast_items
from tests.helpers.v5_loop import config

FAMILIES = (
    "seasonal_h",
    "seasonal_h2",
    "seasonal_d",
    "trend",
    "seasonal_trend",
    "noise",
    "intermittent",
    "level_shift",
)
ORIGIN, HORIZON = 2208, 96


@pytest.fixture
def probe():
    return importlib.import_module("demand_forecast.pretrain.v5.probe")


def control_report(probe, forecaster):
    return {
        "levels": list(QUANTILES),
        "families": {
            name: probe.score_panel(panel, forecaster(panel, QUANTILES), QUANTILES)
            for name, panel in probe.probe_panels().items()
        },
    }


def test_probe_panels_are_fixed(probe):
    panels = probe.probe_panels()
    assert tuple(panels) == FAMILIES
    digest = hashlib.sha256()
    for index, (name, panel) in enumerate(panels.items()):
        panel.validate()
        assert panel.name == name and len(panel.items) == 24
        assert panel.freq == ("h" if index < 2 else "D")
        for i, item in enumerate(panel.items):
            y = item.targets["y"]
            assert item.start == 0 and y.shape == (2400,)
            digest.update(np.array([index, i, item.start, len(y)], dtype="<i8").tobytes())
            digest.update(np.rint(y * 1e6).astype("<i8").tobytes())
    assert digest.hexdigest() == "1b245dd7b1334a7d9e917fd496df634194935a478454e3ed51a5894cf905b7b4"


def test_comparators_equal_a_direct_computation(probe):
    for name, panel in probe.probe_panels().items():
        history = np.stack([item.targets["y"][:ORIGIN] for item in panel.items])
        point = probe._reference_path(panel, history)
        figures = probe.score_panel(panel, probe.flat(panel, QUANTILES), QUANTILES)
        errors = {"reference_mase": [], "naive_mase": [], "level_mase": []}
        season = panel.season
        for i, item in enumerate(panel.items):
            y = item.targets["y"]
            h, truth = y[:ORIGIN], y[ORIGIN : ORIGIN + HORIZON]
            denom = max(1e-8, np.abs(h[season:] - h[:-season]).mean())
            naive = np.resize(h[-season:], HORIZON)
            level = np.median(h)
            if name in ("seasonal_h", "seasonal_d"):
                ref = np.resize(h[-4 * season :].reshape(4, season).mean(axis=0), HORIZON)
            elif name == "seasonal_h2":
                ref = np.resize(h[-168:], HORIZON)
            elif name in ("trend", "seasonal_trend"):
                width = season * ((64 + season - 1) // season)
                slope = (h[-width:] - h[-width - season : -season]).mean() / season
                ref = naive + slope * (np.arange(HORIZON) // season + 1) * season
            elif name == "level_shift":
                ref = h[-28:].mean()
            else:
                ref = level
            np.testing.assert_allclose(
                point[i], np.broadcast_to(ref, (HORIZON,)), atol=1e-9, rtol=0
            )
            for label, prediction in zip(errors, (ref, naive, level), strict=True):
                errors[label].append(np.abs(truth - prediction).mean() / denom)
        for label, values in errors.items():
            assert figures[label] == pytest.approx(np.mean(values))


@pytest.mark.parametrize("name", ["seasonal_d", "trend"])
def test_reference_uses_eight_scaled_backtests_per_step(probe, name):
    panel = probe.probe_panels()[name]
    season = panel.season
    steps = np.array([0, 47, 95])

    def point_and_scale(y, origin):
        past = y[:origin]
        scale = max(1e-8, np.abs(past[season:] - past[:-season]).mean())
        phase = origin + steps % season
        if name == "seasonal_d":
            point = np.mean([y[phase - k * season] for k in range(1, 5)], axis=0)
        else:
            width = season * ((64 + season - 1) // season)
            pairs = np.arange(origin - width, origin)
            slope = np.mean((y[pairs] - y[pairs - season]) / season)
            last = phase - season
            point = y[last] + slope * (origin + steps - last)
        return point, scale

    errors, points, scales = [], [], []
    for item in panel.items:
        y = item.targets["y"]
        for k in range(1, 9):
            origin = ORIGIN - HORIZON * k
            point, scale = point_and_scale(y, origin)
            errors.append((y[origin + steps] - point) / scale)
        point, scale = point_and_scale(y, ORIGIN)
        points.append(point)
        scales.append(scale)

    assert np.shape(errors) == (192, 3)
    offsets = np.quantile(errors, QUANTILES, axis=0).T
    point_paths = np.asarray(points)[..., None]
    expected = point_paths + np.asarray(scales)[:, None, None] * offsets
    actual = probe.reference(panel, QUANTILES)[:, steps, :]
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=0)
    if name == "trend":
        actual_offsets = actual - point_paths
        assert not np.allclose(actual_offsets[:, 0], actual_offsets[:, -1], rtol=1e-9, atol=1e-9)


def test_flat_control_fails_every_shape_family(probe):
    report = control_report(probe, probe.flat)
    verdict = probe.gate(report)
    assert not verdict["passed"]
    for name in FAMILIES[:5]:
        figures = report["families"][name]
        assert figures["correlation"] == pytest.approx(0.0)
        criterion = "mase" if figures["mase"] > figures["naive_mase"] else "correlation"
        assert verdict["families"][name] == {"passed": False, "criterion": criterion}


def test_reference_control_passes_the_gate(probe):
    report = control_report(probe, probe.reference)
    verdict = probe.gate(report)
    assert verdict["passed"], (verdict, report)
    assert all(verdict["families"][name]["passed"] for name in FAMILIES)


def test_ties_do_not_fail_a_correct_intermittent_forecast(probe):
    report = control_report(probe, probe.reference)
    figures = report["families"]["intermittent"]
    assert figures["coverage"][4]["lt"] == pytest.approx(0)
    assert figures["coverage"][4]["le"] > 0.65
    assert figures["min_mean_gap"] == pytest.approx(0)
    assert probe.gate(report)["families"]["intermittent"]["passed"]


@pytest.mark.parametrize("name", ["noise", "seasonal_d"])
def test_gate_rejects_narrow_reference_coverage(probe, name):
    report = control_report(probe, probe.reference)
    baseline = report["families"][name]
    panel = probe.probe_panels()[name]
    knots = probe.reference(panel, QUANTILES)
    median = knots[..., 4:5]
    for factor, criterion in [(0.75, None), (0.5, "coverage")]:
        figures = probe.score_panel(panel, median + factor * (knots - median), QUANTILES)
        assert figures["mase"] == pytest.approx(baseline["mase"])
        assert figures["correlation"] == pytest.approx(baseline["correlation"])
        if name == "noise":
            assert figures["mase"] <= 1.05 * figures["level_mase"]
        else:
            assert figures["mase"] <= figures["naive_mase"]
            assert figures["correlation"] >= 0.5
        assert figures["min_mean_gap"] > 0
        assert np.isfinite(figures["spread_ratio"])
        report["families"][name] = figures
        assert probe.gate(report)["families"][name] == {
            "passed": criterion is None,
            "criterion": criterion,
        }


@pytest.mark.parametrize(
    ("name", "criterion"), [("seasonal_d", "quantile_gap"), ("intermittent", None)]
)
def test_gate_rejects_collapsed_inner_knots_only_on_continuous_families(probe, name, criterion):
    report = control_report(probe, probe.reference)
    assert probe.gate(report)["passed"]
    panel = probe.probe_panels()[name]
    knots = probe.reference(panel, QUANTILES)
    assert np.mean(knots[..., 6] - knots[..., 5]) > 0
    knots[..., 6] = knots[..., 5]
    figures = probe.score_panel(panel, knots, QUANTILES)
    assert figures["min_mean_gap"] == pytest.approx(0)
    for level, knot in zip(QUANTILES, figures["coverage"], strict=True):
        assert knot["lt"] - 0.15 <= level <= knot["le"] + 0.15
    report["families"][name] = figures
    assert probe.gate(report)["families"][name] == {
        "passed": criterion is None,
        "criterion": criterion,
    }


@pytest.fixture
def passing_report():
    return {
        "levels": [0.1, 0.5, 0.9],
        "families": {
            name: {
                "mase": 0.75,
                "naive_mase": 1.0,
                "level_mase": 1.0,
                "reference_mase": 1.0,
                "correlation": 0.75,
                "spread_ratio": 1.0,
                "min_mean_gap": 1.0,
                "coverage": [
                    {"lt": 0.1, "le": 0.1},
                    {"lt": 0.45, "le": 0.55},
                    {"lt": 0.9, "le": 0.9},
                ],
            }
            for name in FAMILIES
        },
    }


@pytest.mark.parametrize(
    ("lt", "le", "shift"), [(0.65, 0.75, 1), (0.25, 0.35, -1)], ids=["below", "above"]
)
def test_gate_coverage_tolerance_is_inclusive(probe, passing_report, lt, le, shift):
    figures = passing_report["families"]["seasonal_d"]
    knot = figures["coverage"][1] = {"lt": lt, "le": le}
    assert probe.gate(passing_report)["passed"]
    for endpoint in ("lt", "le"):
        knot[endpoint] += shift * 1e-9
    assert probe.gate(passing_report)["families"]["seasonal_d"] == {
        "passed": False,
        "criterion": "coverage",
    }


@pytest.mark.parametrize("name", ["noise", "intermittent", "level_shift"])
@pytest.mark.parametrize(("naive", "level"), [(1.0, 2.0), (2.0, 1.0)])
def test_gate_allows_only_five_percent_above_mase_baseline(
    probe, passing_report, name, naive, level
):
    limit = level if name == "noise" else 1.0
    figures = passing_report["families"][name]
    figures.update(naive_mase=naive, level_mase=level, mase=1.05 * limit)
    assert probe.gate(passing_report)["passed"]
    figures["mase"] *= 1 + 1e-9
    assert probe.gate(passing_report)["families"][name] == {"passed": False, "criterion": "mase"}


@pytest.mark.parametrize("name", FAMILIES[:5])
def test_gate_shape_mase_and_correlation_boundaries(probe, passing_report, name):
    figures = passing_report["families"][name]
    figures.update(naive_mase=2.0, level_mase=1.0, mase=2.0, correlation=0.5)
    assert probe.gate(passing_report)["passed"]
    figures["mase"] += 1e-9
    assert probe.gate(passing_report)["families"][name] == {"passed": False, "criterion": "mase"}
    figures.update(mase=2.0, correlation=0.5 - 1e-9)
    assert probe.gate(passing_report)["families"][name] == {
        "passed": False,
        "criterion": "correlation",
    }


@pytest.mark.parametrize("value", [None, np.nan], ids=["none", "nan"])
@pytest.mark.parametrize(
    ("field", "criterion"),
    [
        ("mase", "mase"),
        ("naive_mase", "naive_mase"),
        ("level_mase", "level_mase"),
        ("reference_mase", "reference_mase"),
        ("correlation", "correlation"),
        ("spread_ratio", "spread_ratio"),
        ("min_mean_gap", "quantile_gap"),
        ("lt", "coverage"),
        ("le", "coverage"),
        ("level", "coverage"),
    ],
)
def test_gate_records_undefined_figures(probe, passing_report, field, criterion, value):
    name = "seasonal_d"
    figures = passing_report["families"][name]
    if field == "level":
        passing_report["levels"][1] = value
    elif field in ("lt", "le"):
        figures["coverage"][1][field] = value
    else:
        figures[field] = value
    verdict = probe.gate(passing_report)
    assert not verdict["passed"]
    assert verdict["families"][name] == {"passed": False, "criterion": criterion}


def test_gate_records_missing_family(probe, passing_report):
    del passing_report["families"]["seasonal_d"]
    verdict = probe.gate(passing_report)
    assert not verdict["passed"]
    assert verdict["families"]["seasonal_d"] == {"passed": False, "criterion": "mase"}


@pytest.mark.parametrize("length", [2, 4])
def test_gate_records_wrong_coverage_length(probe, passing_report, length):
    coverage = passing_report["families"]["seasonal_d"]["coverage"]
    coverage[:] = (coverage + [coverage[-1]])[:length]
    verdict = probe.gate(passing_report)
    assert not verdict["passed"]
    assert verdict["families"]["seasonal_d"] == {"passed": False, "criterion": "coverage"}


def test_score_panel_without_a_median_has_undefined_point_figures(probe, passing_report):
    panel = probe.probe_panels()["noise"]
    levels = (0.1, 0.9)
    figures = probe.score_panel(panel, probe.reference(panel, levels), levels)
    assert all(figures[key] is None for key in ("mase", "correlation", "spread_ratio"))
    assert len(figures["coverage"]) == len(levels)
    assert figures["min_mean_gap"] > 0
    passing_report["levels"] = list(levels)
    for family in passing_report["families"].values():
        del family["coverage"][1]
    passing_report["families"]["noise"] = figures
    verdict = probe.gate(passing_report)
    assert not verdict["passed"]
    assert verdict["families"]["noise"] == {"passed": False, "criterion": "mase"}


def test_probe_runs_on_a_tiny_model_through_the_scoring_path(probe, monkeypatch):
    levels = (0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = V5Model(replace(config().model, quantiles=levels))
    settings = probe.ProbeSettings(derive=frozenset({"naive_rows"}))
    calls = []
    forecast_items = probe.forecast_items

    def scoring(model, panel, jobs, horizon, **options):
        assert list(jobs) == [(i, ORIGIN) for i in range(24)]
        assert horizon == HORIZON and options["context"] == 512
        assert options["batch_episodes"] == 16
        assert options["derive"] == settings.derive
        calls.append((panel.name, options.get("batch_hook")))
        return forecast_items(model, panel, jobs, horizon, **options)

    monkeypatch.setattr(probe, "forecast_items", scoring)

    def in_eval_mode(module, _):
        assert not module.training

    with model.register_forward_pre_hook(in_eval_mode):
        report = probe.run_probe(model, settings)
        repeat = probe.run_probe(model, settings)
    assert model.training
    assert report == repeat  # Same weights, runtime and CPU.
    assert (
        calls == [(name, hook) for name in FAMILIES for hook in (None, probe.zero_naive_rows)] * 2
    )
    assert report["levels"] == pytest.approx(levels)
    for family in report["families"].values():
        for figures in (family, family["zeroed_naive_rows"]):
            assert len(figures["coverage"]) == len(levels)
            for key in ("mase", "correlation", "spread_ratio", "min_mean_gap"):
                assert np.isfinite(figures[key]), key
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("derive", [frozenset({"naive_rows"}), frozenset()], ids=["on", "off"])
def test_zeroed_naive_rows_keep_the_layout(probe, derive):
    panel = probe.probe_panels()["seasonal_h2"]
    base = panel.items[0]
    item = replace(
        base,
        dynamic={"promo~last": base.targets["y"].copy()},
    )
    panel = replace(panel, items=(item,), roles={"promo~last": "known"})
    batches = []

    def ablate(batch):
        after = probe.zero_naive_rows(batch)
        batches.append((batch, after))
        return after

    forecast_items(
        V5Model(config().model),
        panel,
        [(0, ORIGIN)],
        HORIZON,
        context=512,
        derive=derive,
        batch_hook=ablate,
    )
    [(before, after)] = batches
    for field in fields(before):
        if field.name != "xh":
            left, right = getattr(before, field.name), getattr(after, field.name)
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, equal_nan=True)
            else:
                assert left == right
    names = [before.row_names[0][i] for i in before.known_rows[0]]
    naive_names = {"y~last", "y~season", "y~season2", "y~drift"} if derive else set()
    assert set(names) == {"promo~last"} | naive_names
    for slot, name in enumerate(names):
        original = before.xh[0, :, slot]
        assert torch.count_nonzero(original)
        expected = torch.zeros_like(original) if name in naive_names else original
        torch.testing.assert_close(after.xh[0, :, slot], expected)


def test_zero_naive_rows_refuses_a_batch_without_row_keys(probe):
    from demand_forecast.data.episode.collate import EpisodeBatch

    panel = probe.probe_panels()["seasonal_d"]
    panel = replace(panel, items=panel.items[:1])
    seen = []
    forecast_items(
        V5Model(config().model),
        panel,
        [(0, ORIGIN)],
        HORIZON,
        context=512,
        derive=frozenset({"naive_rows"}),
        batch_hook=lambda batch: seen.append(batch) or batch,
    )
    [keyed] = seen
    plain = EpisodeBatch(**{f.name: getattr(keyed, f.name) for f in fields(EpisodeBatch)})
    with pytest.raises(TypeError, match="row keys"):
        probe.zero_naive_rows(plain)


def test_probe_reports_mix_anchor_only_and_anchor_off(probe, monkeypatch):
    panel = probe.probe_panels()["seasonal_d"]
    panel = replace(panel, items=panel.items[:2])
    monkeypatch.setattr(probe, "probe_panels", lambda: {panel.name: panel})
    settings = probe.ProbeSettings(derive=frozenset({"naive_rows", "naive_anchor"}))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = V5Model(replace(config().model, naive_mix=True)).eval()
        with torch.no_grad():
            model.mix_projection.weight.normal_(std=0.75)
    report = probe.run_probe(model, settings)["families"][panel.name]
    levels = model.config.quantiles
    options = dict(context=512, derive=settings.derive)
    jobs = [(i, ORIGIN) for i in range(len(panel.items))]
    from demand_forecast.data.episode import EpisodeBatch

    seen = []

    def inspect(batch, output):
        assert batch.naive_seen[..., 0].all()
        seen.append((batch, output))
        cleared = probe.anchor_off(batch)
        assert cleared.xh is batch.xh and cleared.naive_paths is batch.naive_paths
        assert not cleared.naive_seen.any()
        off = model(cleared, output.gates).quantiles
        plain = EpisodeBatch(**{f.name: getattr(batch, f.name) for f in fields(EpisodeBatch)})
        torch.testing.assert_close(off, model(plain, output.gates).quantiles, atol=0, rtol=0)

    actual = forecast_items(model, panel, jobs, HORIZON, output_hook=inspect, **options)
    [(batch, output)] = seen
    assert report["mix_mean"] == pytest.approx(
        dict(
            zip(
                ("none", "last", "level", "season", "season2", "drift"),
                output.mix[:, 0, 0, :HORIZON].mean(dim=(0, 1)).tolist(),
                strict=True,
            )
        )
    )
    assert report["mix_no_seen_share"] == pytest.approx(0)
    knots = np.stack([actual.values[i, "y"] for i, _ in jobs])
    anchor = batch.loc[:, 0, None] + batch.scale[:, 0, None] * output.anchor[:, 0, 0, :HORIZON]
    knots[..., levels.index(0.5)] = anchor.numpy()
    assert report["anchor_only"] == probe.score_panel(panel, knots, levels)
    off = forecast_items(model, panel, jobs, HORIZON, batch_hook=probe.anchor_off, **options)
    off_knots = np.stack([off.values[i, "y"] for i, _ in jobs])
    assert report["anchor_off"] == probe.score_panel(panel, off_knots, levels)
    assert report["anchor_off"]["mase"] != pytest.approx(report["mase"])
    assert report["anchor_only"]["mase"] != pytest.approx(report["mase"])
    assert report["zeroed_naive_rows"]["mase"] != pytest.approx(report["mase"])


def test_probe_mix_excludes_a_missing_horizon_label(probe):
    panel = probe.probe_panels()["seasonal_d"]
    item = panel.items[0]
    target = item.targets["y"].copy()
    missing_step = 3
    target[ORIGIN + missing_step] = np.nan
    panel = replace(panel, items=(replace(item, targets={"y": target}),))
    model = V5Model(replace(config().model, naive_mix=True)).eval()
    settings = probe.ProbeSettings(derive=frozenset({"naive_anchor"}))

    def last_only(batch):
        assert not batch.label_mask[0, 0, 0, missing_step]
        seen = torch.zeros_like(batch.naive_seen)
        seen[..., :HORIZON, 1] = True
        seen[..., missing_step, 1] = False
        return replace(batch, naive_seen=seen)

    report = probe._score_model(model, panel, settings, batch_hook=last_only, observe_mix=True)

    assert report["mix_mean"] == pytest.approx(
        dict(none=0, last=1, level=0, season=0, season2=0, drift=0)
    )
    assert report["mix_no_seen_share"] == pytest.approx(0)


def test_probe_evidence_reports_the_ablations_and_gates_full_figures(probe, monkeypatch):
    panel = probe.probe_panels()["seasonal_d"]
    panel = replace(panel, items=panel.items[:2])
    monkeypatch.setattr(probe, "probe_panels", lambda: {panel.name: panel})
    settings = probe.ProbeSettings(derive=frozenset({"naive_anchor", "naive_evidence"}))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = V5Model(replace(config().model, naive_mix=True, mix_evidence=True)).eval()
    original = {key: value.cpu().numpy().tobytes() for key, value in model.state_dict().items()}
    report = probe.run_probe(model, settings)
    parameters = report["evidence_parameters"]
    assert set(parameters) == {"beta", "gamma", "head_gain", "evidence_bias"}
    for key, expected in (("beta", 4), ("gamma", 1), ("head_gain", 1e-3)):
        assert type(parameters[key]) is float
        assert parameters[key] == pytest.approx(expected)
    assert all(type(bias) is float for bias in parameters["evidence_bias"])
    assert parameters["evidence_bias"] == pytest.approx([0] * 6)
    assert {
        key: value.cpu().numpy().tobytes() for key, value in model.state_dict().items()
    } == original
    figures = report["families"][panel.name]
    assert figures["evidence_share"] == 1
    assert sum(figures["mix_mean"].values()) == pytest.approx(1)
    hooks = {
        "fan_off": probe.fan_off,
        "evidence_off": probe.evidence_off,
        "anchor_off": probe.anchor_off,
    }
    for name, hook in hooks.items():
        expected = probe._score_model(model, panel, settings, batch_hook=hook)
        assert figures[name] == expected
        assert figures["mase"] != pytest.approx(expected["mase"])
    assert figures["anchor_only"]["mase"] == pytest.approx(figures["fan_off"]["mase"], abs=0.001)
    full_gate = probe.gate(report)
    assert full_gate == probe.gate({**report, "evidence_parameters": {}})
    gate = full_gate["families"][panel.name]
    assert gate["criterion"] == probe._failure(panel.name, figures, report["levels"])
    assert (
        probe.gate({**report, "families": {panel.name: figures["fan_off"]}})["families"][
            panel.name
        ]["criterion"]
        == "coverage"
    )


def test_probe_head_off_scores_with_zero_gain_and_restores_calibration(probe, monkeypatch):
    panel = probe.probe_panels()["seasonal_d"]
    panel = replace(panel, items=panel.items[:2])
    monkeypatch.setattr(probe, "probe_panels", lambda: {panel.name: panel})
    settings = probe.ProbeSettings(derive=frozenset({"naive_anchor", "naive_evidence"}))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = V5Model(replace(config().model, naive_mix=True, mix_evidence=True)).eval()
    with torch.no_grad():
        model.head_gain.fill_(-0.5)
        model.evidence_bias.copy_(torch.arange(6) / 2)
    zero_gain = V5Model(model.config).eval()
    zero_gain.load_state_dict(model.state_dict())
    with torch.no_grad():
        zero_gain.head_gain.zero_()
    expected = probe._score_model(zero_gain, panel, settings)

    report = probe.run_probe(model, settings)
    assert report["evidence_parameters"]["head_gain"] == pytest.approx(0.5)
    assert report["evidence_parameters"]["evidence_bias"] == pytest.approx(
        [i / 2 for i in range(6)]
    )
    figures = report["families"][panel.name]
    head_off = figures["head_off"]
    assert set(head_off) == set(expected)
    assert head_off["mase"] != pytest.approx(figures["mase"])
    for key, value in expected.items():
        if key == "coverage":
            for actual_knot, expected_knot in zip(head_off[key], value, strict=True):
                assert actual_knot == pytest.approx(expected_knot)
        else:
            assert head_off[key] == pytest.approx(value)
    assert model.head_gain.item() == pytest.approx(-0.5)


def test_probe_restores_the_gain_if_head_off_scoring_raises(probe, monkeypatch):
    panel = probe.probe_panels()["seasonal_d"]
    monkeypatch.setattr(probe, "probe_panels", lambda: {panel.name: panel})
    model = V5Model(replace(config().model, naive_mix=True, mix_evidence=True)).eval()
    with torch.no_grad():
        model.head_gain.fill_(-0.5)

    def score(model, *args, **kwargs):
        if abs(model.head_gain.item()) < 1e-8:
            raise RuntimeError("head-off score failed")
        return {}

    monkeypatch.setattr(probe, "_score_model", score)
    with pytest.raises(RuntimeError, match="head-off score failed"):
        probe.run_probe(model, probe.ProbeSettings())
    assert model.head_gain.item() == pytest.approx(-0.5)
