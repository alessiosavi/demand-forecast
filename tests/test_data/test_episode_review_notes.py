"""Regression checks for the merged episode pipeline's review notes."""

import ast
import inspect
import subprocess
import sys
import textwrap
from dataclasses import replace
from time import perf_counter

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.episode import EpisodeBuilder, Panel
from demand_forecast.data.episode import panel_stats as ps
from demand_forecast.data.episode.kinds import is_missing, label_of
from demand_forecast.data.episode.panel_stats import PanelStats
from demand_forecast.data.episode.schema import VALUE_STATS
from demand_forecast.data.episode.screening import screen
from demand_forecast.data.episode.value_stats import category_step_stats
from tests.helpers.episodes import toy_panel


@pytest.mark.parametrize("cutoff", [47, 49])
def test_screen_rejects_mismatched_cutoff(cutoff):
    panel = toy_panel(length=80)
    stats = PanelStats(panel, "y", 48)
    with pytest.raises(ValueError, match="cutoff"):
        screen(panel, stats, 0, ("y",), cutoff)
    assert screen(panel, stats, 0, ("y",), 48)


@pytest.mark.parametrize("cell_limit", [650, 10])
def test_summary_chunks_obey_cell_and_row_caps(monkeypatch, cell_limit):
    panel = toy_panel(n_items=8)
    expected = PanelStats(panel, "y", 300)
    calls = []
    original = ps.summary_vectors

    def checked(scaled, raw, **kwargs):
        calls.append(raw.shape)
        # Keep the original one-chunk padding width so reductions are identical.
        assert raw.shape[1] == 300
        assert raw.shape[0] <= PanelStats.SUMMARY_CHUNK
        assert raw.size <= cell_limit or raw.shape[0] == 1
        return original(scaled, raw, **kwargs)

    monkeypatch.setattr(PanelStats, "SUMMARY_CHUNK_CELLS", cell_limit, raising=False)
    monkeypatch.setattr(ps, "summary_vectors", checked)
    actual = PanelStats(panel, "y", 300)
    assert len(calls) > 3
    np.testing.assert_array_equal(actual.summaries, expected.summaries)
    np.testing.assert_array_equal(actual.summary_flags, expected.summary_flags)


def test_custom_business_hour_is_explicit_in_hourly_classes():
    # Inheritance must not be the reason this offset is recognised as hourly.
    tree = ast.parse(textwrap.dedent(inspect.getsource(Panel.freq_class.fget)))
    hourly_classes = [
        node.elts[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Tuple)
        and len(node.elts) == 2
        and isinstance(node.elts[1], ast.Constant)
        and node.elts[1].value == "hourly"
    ]
    assert len(hourly_classes) == 1
    classes = eval(compile(ast.Expression(hourly_classes[0]), "<freq classes>", "eval"), {"pd": pd})
    assert pd.offsets.CustomBusinessHour in classes


def _category_reference(labels, scaled_target, *, left, right, frozen_at=None):
    """Pre-fix implementation, preserving the exact reduction order."""
    labels = np.asarray(labels, dtype=object)
    target = np.asarray(scaled_target, dtype=np.float64)
    if labels.ndim != 1 or target.ndim != 1:
        raise ValueError("labels and scaled_target must be one-dimensional")
    if right < left:
        raise ValueError("right must be at least left")
    values = np.zeros((right - left, VALUE_STATS), dtype=np.float32)
    flags = np.zeros(right - left, dtype=bool)
    steps = np.arange(max(0, left), min(right, len(labels)))
    if not len(steps):
        return values, flags
    missing = np.frompyfunc(is_missing, 1, 1)(labels).astype(bool)
    names = np.frompyfunc(label_of, 1, 1)(labels)
    names[missing] = None
    codes, unique = pd.factorize(names, sort=False)
    bounds = steps if frozen_at is None else np.minimum(steps, frozen_at)
    for code in range(len(unique)):
        selected = codes[steps] == code
        if not selected.any():
            continue
        rows = steps[selected] - left
        ends = bounds[selected]
        occurrences = np.flatnonzero(codes == code)
        means = np.zeros((len(rows), 4), dtype=np.float64)
        spread = np.zeros(len(rows), dtype=np.float64)
        support = np.zeros(len(rows), dtype=np.int64)
        for offset in range(4):
            completed = occurrences + offset
            completed = completed[(completed < len(target)) & (completed < ends.max())]
            responses = target[completed]
            finite = np.isfinite(responses)
            completed, responses = completed[finite], responses[finite]
            n = np.searchsorted(completed, ends, side="left")
            anchor = responses[0] if len(responses) else 0.0
            with np.errstate(over="ignore", invalid="ignore"):
                shifted = responses - anchor
                total = np.r_[0.0, shifted.cumsum()][n]
                mean = total / np.maximum(n, 1)
                means[:, offset] = np.where(n > 0, anchor + mean, 0.0)
                if offset == 0:
                    support = n
                    squares = np.r_[0.0, (shifted**2).cumsum()][n]
                    spread = np.sqrt(np.maximum(0.0, squares / np.maximum(n, 1) - mean**2))
        valid = support > 0
        shrink = support / (support + 4)
        with np.errstate(over="ignore", invalid="ignore"):
            values[rows, 0] = means[:, 0] * shrink
            values[rows, 1] = spread
            values[rows, 2:5] = means[:, 1:] * shrink[:, None]
        values[rows, 5] = np.log1p(support)
        values[rows[~valid]] = 0
        flags[rows] = valid
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return values, flags


@pytest.mark.parametrize("n_labels", [2, 20, 500])
@pytest.mark.parametrize("frozen_at", [None, 1300, -1])
def test_category_grouping_is_bit_identical(n_labels, frozen_at):
    rng = np.random.default_rng(19)
    labels = np.array([f"label{i}" for i in range(n_labels)] * 4, dtype=object)
    rng.shuffle(labels)
    labels[::17] = None
    target = 1e8 + rng.normal(size=len(labels) - 3)
    target[::7] = np.nan
    kwargs = {"left": -2, "right": len(labels) + 3, "frozen_at": frozen_at}
    expected = _category_reference(labels, target, **kwargs)
    actual = category_step_stats(labels, target, **kwargs)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_category_grouping_does_not_rescan_history_per_label(monkeypatch):
    labels = np.array([f"label{i}" for i in range(1000)] * 2, dtype=object)
    scanned = []
    original = np.flatnonzero

    def counted(values):
        scanned.append(values.size)
        return original(values)

    monkeypatch.setattr(np, "flatnonzero", counted)
    start = perf_counter()
    values, flags = category_step_stats(labels, np.arange(2000.0), left=0, right=2000)
    elapsed = perf_counter() - start
    assert flags.sum() == 1000 and values.shape == (2000, 6)
    assert elapsed < 0.75, elapsed
    assert sum(scanned) <= 4 * len(labels), sum(scanned)


@pytest.mark.parametrize(
    "keep_columns,targets",
    [
        (None, ["y", "y1", "y2"]),
        (frozenset({"temp"}), [None]),
        (frozenset({"event"}), ["y", "y1", "y2"]),
    ],
)
def test_plan_requests_propagates_columns_and_uses_them_to_split(keep_columns, targets):
    panel = toy_panel(n_targets=3)
    panel = replace(panel, items=tuple(replace(item, static={}) for item in panel.items))
    builder = EpisodeBuilder(panel)
    kwargs = {"context": 64, "cutoff": 320, "drop": frozenset({"calendar"})}
    requests = builder.plan_requests(0, (336,), 5, keep_columns=keep_columns, **kwargs)
    assert [r.target for r in requests] == targets
    assert all(r.keep_columns == keep_columns for r in requests)
    assert all(r.cutoff == 320 and r.drop == kwargs["drop"] for r in requests)
    for request in requests:
        builder.build(request)
    if keep_columns is None:
        assert requests == builder.plan_requests(0, (336,), 5, **kwargs)


def test_screening_imports_before_any_other_episode_module():
    script = """
import ast
import importlib
import inspect
import sys
assert not any(name.startswith('demand_forecast.data.episode') for name in sys.modules)
screening = importlib.import_module('demand_forecast.data.episode.screening')
from demand_forecast.data.episode.schema import Panel
assert screening.Panel is Panel
imports = [node for node in ast.walk(ast.parse(inspect.getsource(screening)))
           if isinstance(node, ast.ImportFrom) and node.level == 1
           and any(alias.name == 'Panel' for alias in node.names)]
assert len(imports) == 1 and imports[0].module == 'schema', 'Panel must come directly from schema'
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
