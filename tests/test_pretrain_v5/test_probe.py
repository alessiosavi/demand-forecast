"""The absolute probe has fixed inputs and working positive and negative controls."""

import hashlib
import importlib
import json
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from demand_forecast.data.episode import EpisodeBuilder, EpisodeRequest, collate
from demand_forecast.data.episode.schema import QUANTILES
from demand_forecast.models.v5 import V5Model
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
        assert panel.name == name and len(panel.items) == 6
        assert panel.freq == ("h" if index < 2 else "D")
        for i, item in enumerate(panel.items):
            y = item.targets["y"]
            assert item.start == 0 and y.shape == (2400,)
            digest.update(np.array([index, i, item.start, len(y)], dtype="<i8").tobytes())
            digest.update(np.rint(y * 1e6).astype("<i8").tobytes())
    assert digest.hexdigest() == "d055057834d176d36f19de6b65f76b8ebf034be1a96a919d3cfdb2e049ca9f1a"


def test_comparators_equal_a_direct_computation(probe):
    for name, panel in probe.probe_panels().items():
        medians = probe.reference(panel, QUANTILES)[..., 4]
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
                width = season * ((8 + season - 1) // season)
                slope = (h[-width:] - h[-width - season : -season]).mean() / season
                ref = naive + slope * (np.arange(HORIZON) // season + 1) * season
            elif name == "level_shift":
                ref = h[-28:].mean()
            else:
                ref = level
            np.testing.assert_allclose(medians[i], np.broadcast_to(ref, (HORIZON,)), atol=1e-9)
            for label, prediction in zip(errors, (ref, naive, level), strict=True):
                errors[label].append(np.abs(truth - prediction).mean() / denom)
        for label, values in errors.items():
            assert figures[label] == pytest.approx(np.mean(values))


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


def test_undefined_figures_fail_before_later_criteria(probe):
    report = control_report(probe, probe.reference)
    for name, value in (("mase", None), ("correlation", float("nan"))):
        figures = report["families"]["seasonal_h"]
        figures[name] = value
        figures["coverage"] = []
        assert probe.gate(report)["families"]["seasonal_h"] == {
            "passed": False,
            "criterion": name,
        }
        figures[name] = 0.75


def test_probe_runs_on_a_tiny_model_through_the_scoring_path(probe, monkeypatch):
    levels = (0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = V5Model(replace(config().model, quantiles=levels))
    settings = probe.ProbeSettings(derive=frozenset({"naive_rows"}))
    calls = []
    forecast_items = probe.forecast_items

    def scoring(model, panel, jobs, horizon, **options):
        assert list(jobs) == [(i, ORIGIN) for i in range(6)]
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
    assert calls == [
        (name, hook) for name in FAMILIES for hook in (None, probe.zero_naive_rows)
    ] * 2
    assert report["levels"] == pytest.approx(levels)
    for family in report["families"].values():
        for figures in (family, family["zeroed_naive_rows"]):
            assert len(figures["coverage"]) == len(levels)
            for key in ("mase", "correlation", "spread_ratio", "min_mean_gap"):
                assert np.isfinite(figures[key]), key
    json.dumps(report, allow_nan=False)


def test_zeroed_naive_rows_keep_the_layout(probe):
    panel = probe.probe_panels()["seasonal_h2"]
    base = panel.items[0]
    item = replace(
        base,
        dynamic={"promo": base.targets["y"].copy()},
    )
    panel = replace(panel, items=(item,), roles={"promo": "known"})
    before = collate(
        [
            EpisodeBuilder(panel).build(
                EpisodeRequest(0, (ORIGIN,), HORIZON, 512, derive=frozenset({"naive_rows"}))
            )
        ]
    )
    after = probe.zero_naive_rows(before)
    for field in fields(before):
        if field.name != "xh":
            left, right = getattr(before, field.name), getattr(after, field.name)
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, equal_nan=True)
            else:
                assert left == right
    names = [before.row_names[0][i] for i in before.known_rows[0]]
    assert set(names) == {"promo", "y~last", "y~season", "y~season2", "y~drift"}
    for slot, name in enumerate(names):
        original = before.xh[0, :, slot]
        assert torch.count_nonzero(original)
        expected = original if name == "promo" else torch.zeros_like(original)
        torch.testing.assert_close(after.xh[0, :, slot], expected)
