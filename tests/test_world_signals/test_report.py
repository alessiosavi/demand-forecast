import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("datasets")
pytest.importorskip("fev")
pytest.importorskip("statsforecast")
pytestmark = pytest.mark.bench

from demand_forecast.bench.run import predictions_for_fev  # noqa: E402
from demand_forecast.bench.specs import FevTaskSpec  # noqa: E402
from scripts.world_signals.fev_io import write_fev_parquet  # noqa: E402
from scripts.world_signals.report import (  # noqa: E402
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


@pytest.fixture(scope="module")
def report_source(tmp_path_factory):
    """Complete synthetic artifacts with real fev summaries; never fit a model."""
    import yaml

    from demand_forecast.bench.run import settings_for_task
    from demand_forecast.bench.specs import dataset_fingerprint, load_task_specs
    from demand_forecast.config import Settings
    from scripts.world_signals import build_dataset as builder
    from scripts.world_signals import report as reporting
    from scripts.world_signals.fetch_calendars import holiday_flags
    from scripts.world_signals.fetch_prices import PANEL_SYMBOLS

    root = tmp_path_factory.mktemp("report-source")
    grid = pd.date_range("2021-12-01", "2025-12-31")
    prices = pd.concat(
        [
            pd.DataFrame(
                {
                    "id": asset,
                    "date": grid,
                    "close": 100
                    + i
                    + np.sin(np.arange(len(grid)) / 7)
                    + np.arange(len(grid)) / 100,
                }
            )
            for i, asset in enumerate(sorted(PANEL_SYMBOLS))
        ],
        ignore_index=True,
    )
    raw = builder.BuildInputs(
        prices,
        pd.DataFrame(columns=["id", "date", "news_count_1d", "news_count_3d"]),
        pd.DataFrame({"date": grid, **{key: 1.0 for key in builder.GLOBAL_COLUMNS}}),
        holiday_flags(grid).assign(fomc_day=0),
        {"keep": False},
    )
    bundle = root / "bundle"
    builder.write_bundle(raw, bundle, start=grid[0], end=grid[-1])
    sidecar = pd.read_parquet(bundle / "prices.parquet")
    for variant in ("on", "off"):
        for spec in load_task_specs(bundle / f"probe_{variant}.yaml", bundle / "profiles.yaml"):
            task = spec.to_task()
            contexts = reporting.window_data(task, sidecar)
            arrays = np.zeros((10, 13, 9))
            predictions = [
                predictions_for_fev(data.ids, arrays, data.ids, spec.quantile_levels)
                for data in contexts
            ]
            summary = task.evaluation_summary(predictions, model_name="synthetic-audit")
            for run in reporting.RUN_NAMES:
                if run.endswith("-on") != (variant == "on"):
                    continue
                directory = root / "runs" / run
                target = directory / spec.name
                target.mkdir(parents=True)
                identity = f"synthetic-{run}-{spec.name}"
                for data in contexts:
                    np.savez(
                        target / f"window_{data.index}.npz",
                        ids=data.ids,
                        arrays=arrays,
                        identity=identity,
                    )
                config = (
                    Path("configs/fev-v2-pretrained-finetune.yaml")
                    if run.startswith("l3-")
                    else Path("configs/world_signals/probe-scratch.yaml")
                )
                base = Settings.model_validate(yaml.safe_load(config.read_text()))
                settings = settings_for_task(spec, base, task.freq).model_dump(mode="json")
                seed = int(run.endswith("-s1"))
                settings["seed"] = seed
                report = {
                    "synthetic": True,
                    "run_identity": identity,
                    "seed": seed,
                    "settings": settings,
                    "task_definition": spec.task_definition(),
                    "dataset_fingerprint": dataset_fingerprint(task),
                    "fit_result": {
                        "best_step": 100,
                        "best_score": 0.5,
                        "steps": 100,
                        "stop_reason": "early_stopping",
                    },
                    "resolved_budget": {"max_steps": 500},
                    "n_val_items": 10,
                }
                if run.startswith("l3-"):
                    settings["training"]["init_from"] = "/synthetic/pretrain"
                    report["transfer"] = {
                        "pretrain_identity": "83d6acb1f7f2c1b57ba33c0ca0a870775369d52bf8fb739a9e3f7e28858826c2"
                    }
                (target / "run.json").write_text(json.dumps(report))
                (target / "history.json").write_text(
                    json.dumps(
                        {
                            "history": [{"step": 0, "SQL": 1.0}],
                            "best_step": 100,
                            "best_score": 0.5,
                            "max_steps": 500,
                        }
                    )
                )
                summary_path = directory / "summaries.csv"
                pd.DataFrame([{"task": spec.name, **summary, "run_identity": identity}]).to_csv(
                    summary_path,
                    mode="a",
                    header=not summary_path.exists(),
                    index=False,
                )
    return root


@pytest.fixture
def report_case(report_source, tmp_path, monkeypatch):
    from scripts.world_signals import report as reporting

    class SyntheticZeroBaseline:
        def fit_predict(self, task):
            return [
                [
                    {
                        "predictions": [0.0] * task.horizon,
                        **{str(q): [0.0] * task.horizon for q in task.quantile_levels},
                    }
                    for _ in window.get_ground_truth()
                ]
                for window in task.iter_windows(num_proc=1)
            ]

    monkeypatch.setattr(reporting, "SeasonalNaiveBaseline", SyntheticZeroBaseline)
    shutil.copytree(report_source / "runs", tmp_path / "runs")
    return report_source / "bundle", tmp_path / "runs", tmp_path / "report"


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("violation", ["copied_seed0", "missing_transfer", "scratch_transfer"])
def test_report_rejects_wrong_arm_artifacts(report_case, monkeypatch, violation, partial):
    from scripts.world_signals import report as reporting

    bundle, runs, out = report_case

    def forbidden(*args, **kwargs):
        raise AssertionError("arm validation must happen before scoring")

    monkeypatch.setattr(reporting, "score_task", forbidden)
    if violation == "copied_seed0":
        shutil.rmtree(runs / "l3-off-s1")
        shutil.copytree(runs / "l3-off", runs / "l3-off-s1")
        match = "seed"
    else:
        run = "l3-on" if violation == "missing_transfer" else "scratch-on"
        path = runs / run / "probe_2022_on/run.json"
        data = json.loads(path.read_text())
        if violation == "missing_transfer":
            data.pop("transfer")
        else:
            data["transfer"] = {"pretrain_identity": "unexpected"}
        path.write_text(json.dumps(data))
        match = "transfer|pretrain_identity"
    with pytest.raises(ValueError, match=match):
        reporting.build_report(bundle, runs, out, partial=partial)
    assert not (out / "verdict.json").exists()


@pytest.mark.parametrize(
    "keys,value,match",
    [
        (("settings", "seed"), 1, "seed"),
        (("settings", "model", "d_model"), 64, "d_model"),
        (("settings", "model", "nhead"), 4, "nhead"),
        (("settings", "model", "num_encoder_layers"), 2, "num_encoder_layers"),
        (("settings", "model", "num_decoder_layers"), 2, "num_decoder_layers"),
        (("settings", "model", "dim_feedforward"), 256, "dim_feedforward"),
        (("resolved_budget", "max_steps"), 150, "max_steps"),
        (("transfer", "pretrain_identity"), "wrong", "pretrain_identity"),
        (("settings", "training", "learning_rate"), 0.01, "settings differ"),
        (("settings", "model", "dropout"), 0.9, "settings differ"),
    ],
)
def test_report_rejects_changed_controls(report_case, keys, value, match):
    from scripts.world_signals import report as reporting

    bundle, runs, out = report_case
    path = runs / "l3-on/probe_2022_on/run.json"
    data = json.loads(path.read_text())
    cursor = data
    for key in keys[:-1]:
        cursor = cursor[key]
    cursor[keys[-1]] = value
    path.write_text(json.dumps(data))
    out.mkdir()
    (out / "verdict.json").write_text('{"verdict": "stale"}')
    with pytest.raises(SystemExit) as exc:
        reporting.main(
            ["--bundle", str(bundle), "--runs", str(runs), "--out", str(out), "--partial"]
        )
    assert exc.value.code == 1
    assert not (out / "verdict.json").exists()
    with pytest.raises(ValueError, match=match):
        reporting.build_report(bundle, runs, out)


def test_arm_settings_ignore_data_roles_and_smoke_uses_150_steps(report_case):
    from scripts.world_signals import report as reporting

    _, runs, _ = report_case
    path = runs / "scratch-on/probe_2022_on/run.json"
    data = json.loads(path.read_text())
    data["settings"]["data"] = {"known_dynamic": ["different_role"]}
    path.write_text(json.dumps(data))
    reporting._run_metadata([(name, runs / name) for name in reporting.RUN_NAMES])
    with pytest.raises(ValueError, match="150"):
        reporting._validate_arm("smoke", data, path.parent)
    data["resolved_budget"]["max_steps"] = 150
    assert reporting._validate_arm("smoke", data, path.parent)[0] == "smoke"


def test_partial_report_validates_run_even_without_summary(report_case):
    from scripts.world_signals import report as reporting

    bundle, runs, out = report_case
    path = runs / "l3-off-s1/probe_2022_off/run.json"
    data = json.loads(path.read_text())
    data["seed"] = data["settings"]["seed"] = 0
    path.write_text(json.dumps(data))
    (runs / "l3-off-s1/summaries.csv").unlink()
    with pytest.raises(ValueError, match="seed 1"):
        reporting.build_report(bundle, runs, out, partial=True)


@pytest.mark.parametrize("partial", [False, True])
def test_report_validates_present_smoke_budget(report_case, partial):
    from scripts.world_signals import report as reporting

    bundle, runs, out = report_case
    shutil.copytree(runs / "scratch-on", runs / "smoke")
    with pytest.raises(ValueError, match="smoke requires resolved_budget.max_steps == 150"):
        reporting.build_report(bundle, runs, out, partial=partial)


@pytest.mark.parametrize("step0", [None, np.nan, np.inf, 1.0002])
def test_step0_checks_reject_missing_nonfinite_or_unequal_sql(step0):
    from scripts.world_signals import report as reporting

    diagnostics = [
        {"run": run, "fold": fold, "history_present": True, "step0_SQL": 1.0}
        for run in ("l3-on", "l3-off")
        for fold in reporting.FOLDS
    ]
    diagnostics[-1]["step0_SQL"] = step0
    with pytest.raises(ValueError, match="2025.*step-0"):
        reporting._step0_checks(diagnostics, partial=False)
    assert len(reporting._step0_checks(diagnostics, partial=True)) == 1
    diagnostics[-1]["step0_SQL"] = 1.00005
    assert reporting._step0_checks(diagnostics, partial=False) == []
    diagnostics[-1]["history_present"] = False
    with pytest.raises(ValueError, match="2025.*history.json"):
        reporting._step0_checks(diagnostics, partial=False)
    assert reporting._step0_checks(diagnostics, partial=True) == []


def test_full_report_requires_histories(report_case):
    from scripts.world_signals import report as reporting

    bundle, runs, out = report_case
    for path in runs.glob("*/probe_*/history.json"):
        path.unlink()
    with pytest.raises(ValueError, match="step-0.*history|history.*step-0"):
        reporting.build_report(bundle, runs, out)
    assert not (out / "verdict.json").exists()


def test_full_report_accepts_equal_step0_histories(report_case):
    from scripts.world_signals import report as reporting

    bundle, runs, out = report_case
    result = reporting.build_report(bundle, runs, out)
    assert result["verdict"]["audit_passed"]
    assert result["synthetic"]
    diagnostics = pd.read_csv(out / "tables/training_diagnostics.csv")
    pairs = diagnostics.loc[diagnostics.run.isin(["l3-on", "l3-off"])]
    assert len(pairs) == 8 and pairs.step0_pair_equal.all()


def test_partial_report_allows_unknown_history(report_case):
    from scripts.world_signals import report as reporting

    bundle, runs, out = report_case
    for path in runs.glob("*/probe_*/history.json"):
        path.unlink()
    result = reporting.build_report(bundle, runs / "l3-on", out, partial=True)
    assert result["verdict"] is None
    assert not (out / "verdict.json").exists()
