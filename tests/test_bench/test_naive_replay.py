import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

fev = pytest.importorskip("fev")
import datasets  # noqa: E402

from demand_forecast.bench.baselines import (  # noqa: E402
    BASELINE_MODELS,
    REPLAY_MODELS,
    SeasonalNaiveBaseline,
    make_baseline,
)
from demand_forecast.bench.run import run_baseline  # noqa: E402

pytestmark = pytest.mark.bench
RUNGS = ("snaive", "snaive_v5lag", "snaive_v5point", "snaive_v5pairs", "snaive_v5patch")
LEVELS = [0.1, 0.5, 0.9]
Z90 = 1.2815515655446004


def test_baseline_job_models_match_baseline_registry():
    path = Path(__file__).resolve().parents[2] / "scripts/gcp/jobs/make_baseline_job.py"
    spec = importlib.util.spec_from_file_location("make_baseline_job", path)
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)

    assert set(job.MODELS) == set(BASELINE_MODELS) | set(REPLAY_MODELS)


def synthetic_task(tmp_path, histories, freq, horizon, *, seasonality=1, levels=LEVELS, windows=1):
    targets = sorted(histories[0])
    rows = []
    for index, history in enumerate(histories):
        length = len(history[targets[0]]) + horizon * windows
        rows.append(
            {
                "id": str(index),
                "timestamp": pd.date_range("1900-01-01", periods=length, freq=freq).to_numpy(),
                **{
                    name: np.r_[values, np.arange(1, horizon * windows + 1, dtype=values.dtype)]
                    for name, values in history.items()
                },
            }
        )
    tmp_path.mkdir(exist_ok=True)
    parquet = tmp_path / "data.parquet"
    datasets.Dataset.from_list(rows).to_parquet(parquet)
    return fev.Task(
        dataset_path=str(parquet),
        target=targets if len(targets) > 1 else targets[0],
        horizon=horizon,
        num_windows=windows,
        seasonality=seasonality,
        quantile_levels=levels,
        eval_metric="SQL" if levels else "MASE",
    )


@pytest.mark.parametrize(
    "case", ["plain", "gappy", "short", "lag1", "multitarget", "integer", "long", "point_only"]
)
def test_snaive_matches_statsforecast_arrays(tmp_path, case):
    size = {"short": 3, "long": 1060}.get(case, 46)
    rng = np.random.default_rng(21)
    history = rng.uniform(-50, 50, size).astype(np.float32)
    if case in {"gappy", "long"}:
        history[:4] = np.nan
        history[-9:-4] = np.nan
        if case == "long":
            history[52:75] = np.nan  # The cap starts within this gap.
    elif case == "integer":
        history = 2**24 + rng.integers(-100, 100, size, dtype=np.int64)
    histories = [{"first": history}]
    if case == "multitarget":
        histories = [{"first": history + index, "second": -history - index} for index in range(12)]

    levels = [] if case == "point_only" else [0.01, 0.123456789123, 0.5, 0.76, 0.9, 0.99]
    task = synthetic_task(
        tmp_path, histories, "W-SUN" if case == "lag1" else "D", 17, levels=levels
    )
    expected = SeasonalNaiveBaseline().fit_predict(task)[0]
    actual = make_baseline("snaive").fit_predict(task)[0]
    assert list(actual) == list(expected)
    for target in task.target_columns:
        assert actual[target].features == expected[target].features
        assert actual[target].column_names == ["predictions", *map(str, levels)]
        for column in actual[target].column_names:
            np.testing.assert_allclose(
                actual[target][column], expected[target][column], rtol=0, atol=1e-6
            )
    if case == "short":
        np.testing.assert_allclose(
            actual["first"]["predictions"][0], np.resize(np.r_[history, np.zeros(4)], 17)
        )


def test_adjacent_rungs_agree_when_each_change_is_inert(tmp_path):
    task = synthetic_task(tmp_path, [{"target": np.arange(50, dtype=float) ** 2}], "D", 7)
    previous = make_baseline(RUNGS[0]).fit_predict(task)[0]["target"]
    for name in RUNGS[1:]:
        current = make_baseline(name).fit_predict(task)[0]["target"]
        for column in previous.column_names:
            np.testing.assert_allclose(current[column], previous[column], atol=1e-6)
        previous = current


def test_snaive_runs_without_pretrain_period(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "demand_forecast.pretrain.periods", None)
    task = synthetic_task(tmp_path, [{"target": np.arange(40, dtype=float)}], "D", 7)

    actual = make_baseline("snaive").fit_predict(task)[0]["target"]
    np.testing.assert_allclose(actual["predictions"][0], np.arange(33, 40))


@pytest.mark.parametrize("name", RUNGS)
@pytest.mark.parametrize("freq,seasonality,v5_lag", [("QS", 1, 4), ("D", 14, 7)])
def test_replay_rungs_use_the_expected_lag(tmp_path, name, freq, seasonality, v5_lag):
    history = np.arange(40, dtype=float)
    horizon = 17
    task = synthetic_task(tmp_path, [{"target": history}], freq, horizon, seasonality=seasonality)
    lag = seasonality if name == "snaive" else v5_lag
    expected = np.resize(history[-lag:], horizon)
    growth = np.sqrt(1 + np.arange(horizon) // lag)
    if name == "snaive_v5patch":
        growth[:16] = growth[:16].mean()
    spread = Z90 * lag * growth

    actual = make_baseline(name).fit_predict(task)[0]["target"]
    np.testing.assert_allclose(actual["predictions"][0], expected)
    np.testing.assert_allclose(actual["0.1"][0], expected - spread)
    np.testing.assert_allclose(actual["0.9"][0], expected + spread)


def replay(history, lag, horizon, **switches):
    from demand_forecast.bench.naive_replay import seasonal_naive_replay

    return seasonal_naive_replay(history, lag, horizon, LEVELS, **switches)


def test_v5point_tries_three_earlier_cycles_and_counts_a_miss_per_step():
    history = np.arange(35, dtype=float)
    history[[28, 29, 22, 30, 23, 16, 31, 24, 17, 10]] = np.nan
    old_point, old_quantiles, _, _ = replay(history, 7, 14)
    point, quantiles, misses, fallbacks = replay(history, 7, 14, v5_point=True)
    np.testing.assert_allclose(point, np.tile([21, 15, 9, 27, 32, 33, 34], 2))
    np.testing.assert_allclose(quantiles - point[:, None], old_quantiles - old_point[:, None])
    assert (misses, fallbacks) == (2, 0)


@pytest.mark.parametrize("lag,end", [(7, 35), (7, 4), (1, 32), (1, 33)])
def test_v5point_matches_naive_history_season_and_last(lag, end):
    from demand_forecast.data.episode.derived import NaiveHistory

    history = np.arange(end + 14, dtype=float) + 5
    if lag == 1:
        history[1:end] = np.nan
    else:
        history[[0, 2, 3]] = np.nan
        if end == 35:
            history[[28, 29, 22, 30, 23, 16, 31, 24, 17, 10]] = np.nan
    paths = NaiveHistory(history, season=lag, long_lag=None, end=end).series(
        left=end, right=end + 14, origin=end
    )
    expected, seen = paths["last" if lag == 1 else "season"]
    previous, _, _, _ = replay(history[:end], lag, 14)
    actual, _, misses, _ = replay(history[:end], lag, 14, v5_point=True)
    np.testing.assert_allclose(actual, np.where(seen, expected, previous))
    assert misses == np.count_nonzero(~seen)


@pytest.mark.parametrize("lag", [7, 24])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_v5pairs_uses_finite_rms_at_the_minimum_and_falls_back_below(lag, offset):
    count = max(8, lag) + offset
    history = np.r_[np.nan, np.arange(lag + count, dtype=float)]
    previous_point, previous_quantiles, _, _ = replay(history, lag, lag, v5_point=True)
    point, quantiles, misses, fallbacks = replay(history, lag, lag, v5_point=True, v5_pairs=True)
    np.testing.assert_allclose(point, previous_point)
    np.testing.assert_allclose(
        previous_quantiles[:, 2] - point,
        Z90 * np.sqrt((count * lag**2 + (lag - 1) ** 2) / (count + 1)),
    )
    expected_width = previous_quantiles[:, 2] - point if offset < 0 else Z90 * lag
    np.testing.assert_allclose(quantiles[:, 2] - point, expected_width)
    assert misses == 0
    assert fallbacks == int(offset < 0)


def test_v5pairs_requires_both_ends_inside_the_last_1000_cells():
    history = np.r_[np.arange(21, dtype=float), np.full(986, np.nan)]
    _, previous, _, _ = replay(history, 7, 7, v5_point=True)
    _, actual, _, fallbacks = replay(history, 7, 7, v5_point=True, v5_pairs=True)
    assert fallbacks == 1  # Seven pairs are inside the cap and seven more cross it.
    np.testing.assert_allclose(actual, previous)


@pytest.mark.parametrize(
    "lag,horizon,patch_growth",
    [
        (7, 28, [(7 + 7 * math.sqrt(2) + 2 * math.sqrt(3)) / 16, (5 * math.sqrt(3) + 14) / 12]),
        (
            24,
            168,
            [
                1,
                (1 + math.sqrt(2)) / 2,
                math.sqrt(2),
                math.sqrt(3),
                (math.sqrt(3) + 2) / 2,
                2,
                math.sqrt(5),
                (math.sqrt(5) + math.sqrt(6)) / 2,
                math.sqrt(6),
                math.sqrt(7),
                math.sqrt(7),
            ],
        ),
        (1, 13, [sum(map(math.sqrt, range(1, 14))) / 13]),
    ],
)
def test_v5patch_averages_growth_on_full_and_partial_patches(lag, horizon, patch_growth):
    history = np.arange(96, dtype=float)
    point, by_step, _, _ = replay(history, lag, horizon, v5_point=True, v5_pairs=True)
    patched, by_patch, misses, fallbacks = replay(
        history, lag, horizon, v5_point=True, v5_pairs=True, v5_patch=True
    )
    np.testing.assert_allclose(patched, point)
    np.testing.assert_allclose(
        by_step[:, 2] - point, Z90 * lag * np.sqrt(1 + np.arange(horizon) // lag)
    )
    width = Z90 * lag * np.repeat(patch_growth, 16)[:horizon]
    np.testing.assert_allclose(by_patch, np.column_stack([point - width, point, point + width]))
    assert (misses, fallbacks) == (0, 0)


def test_replay_counts_are_saved_per_task_and_model_on_resume(tmp_path):
    gappy = np.r_[3.0, np.full(39, np.nan)]
    complete = np.arange(40, dtype=float)
    task = synthetic_task(
        tmp_path,
        [{"first": gappy, "second": complete}, {"first": complete, "second": gappy}],
        "W-SUN",
        4,
        windows=2,
    )
    complete_task = synthetic_task(tmp_path / "complete", [{"target": complete}], "W-SUN", 4)
    specs = [
        SimpleNamespace(name="complete", to_task=lambda: complete_task),
        SimpleNamespace(name="gappy", to_task=lambda: task),
    ]
    out = tmp_path / "out"
    for name in RUNGS:
        result = run_baseline(specs, out, model=name)
    saved = result.set_index(["task", "model_name"])
    expected = [(0, 0), (0, 0), (8, 0), (8, 4), (8, 4)]
    for name, counts in zip(RUNGS, expected, strict=True):
        columns = ["point_lookup_misses", "sigma_fallbacks"]
        assert tuple(saved.loc[("complete", name), columns]) == (0, 0)
        assert tuple(saved.loc[("gappy", name), columns]) == counts
    pd.testing.assert_frame_equal(run_baseline(specs, out, model=RUNGS[-1]), result)
