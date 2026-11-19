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
