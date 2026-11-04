import numpy as np
import pandas as pd
import pytest

from demand_forecast.bench.run import predictions_for_fev
from demand_forecast.bench.specs import FevTaskSpec
from scripts.world_signals.fev_io import write_fev_parquet
from scripts.world_signals.report import (
    block_bootstrap,
    price_point_path,
    return_window_metrics,
    verdict,
)


def test_price_path_compounds_sessions_only():
    result = price_point_path(np.log([[2, 100, 3, 100]]), np.array([10]), [[1, 0, 1, 0]])
    np.testing.assert_allclose(result, [[20, 20, 60, 60]])


def test_axis_and_multi_asset_shapes():
    result = price_point_path(np.log([[2, 3], [4, 5]]), np.array([10, 100]), np.ones((2, 2)))
    np.testing.assert_allclose(result, [[20, 60], [400, 2000]])
    with pytest.raises(ValueError, match="shape"):
        price_point_path(np.zeros((2, 2, 3)), np.ones(2), np.ones((2, 2)))


@pytest.mark.bench
def test_sql_recompute_matches_fev(tmp_path, monkeypatch):
    import datasets

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    rng = np.random.default_rng(42)
    frames = []
    for i in range(3):
        values = rng.normal(0, 0.02, 60)
        values[::5] = np.nan
        if i == 2:
            values[:] = 0  # Undefined scale is excluded by fev.
        frames.append(
            pd.DataFrame(
                {
                    "id": str(i),
                    "timestamp": pd.date_range("2022-01-01", periods=60),
                    "target": values,
                }
            )
        )
    path = tmp_path / "panel.parquet"
    write_fev_parquet(pd.concat(frames), {}, path)
    task = FevTaskSpec(
        name="synthetic",
        dataset_path=path,
        target_support="real",
        horizon=4,
        num_windows=2,
        window_step_size=5,
        initial_cutoff="2022-02-15",
        seasonality=7,
        quantile_levels=[0.1, 0.5, 0.9],
        eval_metric="SQL",
        extra_metrics=["MASE", "WQL", "WAPE"],
    ).to_task()
    predictions, metrics = [], []
    for window in task.iter_windows(num_proc=1):
        past, _ = window.get_input_data()
        truth = list(window.get_ground_truth().with_format(None))
        ids = [row["id"] for row in truth]
        histories = {row["id"]: row["target"] for row in past.with_format(None)}
        arrays = rng.normal(0, 0.01, (3, 4, 3))
        predictions.append(predictions_for_fev(ids, arrays, ids, task.quantile_levels))
        metrics.append(
            return_window_metrics(
                np.array([row["target"] for row in truth]),
                arrays,
                [histories[key] for key in ids],
                7,
                task.quantile_levels,
            )
        )
    summary = task.evaluation_summary(predictions, model_name="fake")
    for name in ("SQL", "MASE", "WQL", "WAPE"):
        assert np.mean([m[name] for m in metrics]) == pytest.approx(summary[name], rel=1e-10)


def folds(sql=(-1, -1, -1, 1), price=(-1, -1, -1, 1), best=(100, 100, 100, 100), noise=(0.1,) * 4):
    return pd.DataFrame(
        {
            "fold": [2022, 2023, 2024, 2025],
            "delta_SQL": sql,
            "delta_macroWAPE": price,
            "best_step_on": best,
            "noise_SQL": noise,
            "noise_macroWAPE": [0.1] * 4,
        }
    )


def test_verdict_requires_same_folds():
    assert verdict(folds(price=(1, -1, -1, -1)), (-2, -0.1))["verdict"] == "inconclusive"
    with pytest.raises(ValueError, match="four|folds"):
        verdict(folds().iloc[:3], (-2, -0.1))


def test_best_step_zero_is_no_learning():
    result = verdict(folds(best=(0, 0, 0, 0)), (-2, -0.1))
    assert result["verdict"] == "inconclusive"
    assert all(row["label"] == "no-learning" for row in result["folds"])


@pytest.mark.parametrize(
    "table,ci,label",
    [
        (folds(), (-2, -0.1), "solid"),
        (folds(), (-2, 0), "suggestive"),
        (folds(noise=(1,) * 4), (-2, -0.1), "suggestive"),
        (folds(sql=(1, 1, 1, -1)), (0.1, 2), "negative"),
        (folds(sql=(1, 1, -1, -1)), (-1, 1), "inconclusive"),
    ],
)
def test_verdict_labels(table, ci, label):
    assert verdict(table, ci)["verdict"] == label


def test_block_bootstrap_keeps_windows_together():
    on = np.array([[[1, 3], [5, np.nan]], [[10, 30], [50, 70]]], dtype=float)
    off = np.zeros_like(on)
    off[np.isnan(on)] = np.nan
    result = block_bootstrap(on, off, n=2000, seed=0)
    draws = np.random.default_rng(0).integers(0, 2, size=(2000, 2))
    expected = np.array([np.nanmean(on[index]) for index in draws])
    assert result["delta"] == pytest.approx(np.nanmean(on))
    np.testing.assert_allclose(
        [result["ci_low"], result["ci_high"]], np.percentile(expected, [2.5, 97.5])
    )
    assert result["blocks"] == 2
    with pytest.raises(ValueError, match="paired"):
        block_bootstrap(on, np.zeros_like(on))


def test_saved_predictions_align_by_id_and_reject_corruption(tmp_path):
    from scripts.world_signals.report import load_arrays

    path = tmp_path / "window_0.npz"
    arrays = np.array([[[10, 20, 30]], [[1, 2, 3]]], dtype=float)
    np.savez(path, ids=["GOLD", "BTC"], arrays=arrays, identity="identity")
    aligned = load_arrays(path, "identity", ["BTC", "GOLD"], 1, [0.1, 0.5, 0.9])
    np.testing.assert_array_equal(aligned, arrays[::-1])
    with pytest.raises(ValueError, match="identity"):
        load_arrays(path, "wrong", ["BTC", "GOLD"], 1, [0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="horizon|shape"):
        load_arrays(path, "identity", ["BTC", "GOLD"], 2, [0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="missing"):
        load_arrays(path, "identity", ["BTC", "SILVER"], 1, [0.1, 0.5, 0.9])


def test_price_bootstrap_recomputes_asset_ratios():
    from scripts.world_signals.report import price_bootstrap

    on = np.array([[1, 20], [3, 4], [5, 100]], dtype=float)
    off = on / 2
    denominator = np.array([[10, 100], [100, 100], [10, 1000]], dtype=float)
    result = price_bootstrap(on, off, denominator)
    samples = np.random.default_rng(0).integers(0, 3, size=(2000, 3))
    expected = [
        np.mean((on[s].sum(axis=0) - off[s].sum(axis=0)) / denominator[s].sum(axis=0))
        for s in samples
    ]
    np.testing.assert_allclose(
        [result["ci_low"], result["ci_high"]], np.percentile(expected, [2.5, 97.5])
    )
    assert result["delta"] == pytest.approx(
        np.mean((on.sum(axis=0) - off.sum(axis=0)) / denominator.sum(axis=0))
    )


def test_fold_price_metrics_pool_per_asset_before_macro_average():
    from types import SimpleNamespace

    from scripts.world_signals.report import WindowData, score_task

    contexts = []
    for i, actual in enumerate(([[11, np.nan], [110, 120]], [[20, 30], [90, 100]])):
        contexts.append(
            WindowData(
                i,
                pd.Timestamp("2022-01-03"),
                ["GOLD", "BTC"],
                np.tile(pd.date_range("2022-01-04", periods=2).to_numpy(), (2, 1)),
                np.ones((2, 2)),
                [[1, 2, 3], [2, 4, 6]],
                np.array([10, 100]),
                np.array([[True, i == 1], [True, True]]),
                np.array(actual),
                np.array([2, 10]),
            )
        )
    spec = SimpleNamespace(
        name="probe_2022_on",
        quantile_levels=[0.1, 0.5, 0.9],
        horizon=2,
        seasonality=1,
        initial_cutoff="2022-01-03",
    )
    result = score_task("random_walk", spec, contexts, [np.zeros((2, 2, 3))] * 2, [], [])
    assert result["assets"][0]["price_WAPE"] == pytest.approx(31 / 61)
    assert result["assets"][1]["price_WAPE"] == pytest.approx(40 / 420)
    assert result["metrics"]["macroWAPE"] == pytest.approx((31 / 61 + 40 / 420) / 2)
    assert result["metrics"]["price_WAPE"] == pytest.approx(71 / 481)
    assert result["assets"][0]["price_MASE"] == pytest.approx(31 / 3 / 2)
