import pytest

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

import numpy as np  # noqa: E402

from demand_forecast.metrics import mase, seasonal_error_per_item, summarize, wape  # noqa: E402
from tests.helpers.fev_fixture import fixture_task, write_fev_fixture  # noqa: E402


def arrays_from_window(window, predictions):
    """Read a univariate fev window in its positional item and quantile order."""
    (target,) = window.target_columns
    past, _ = window.get_input_data()
    truth = window.get_ground_truth()
    assert list(past[window.id_column]) == list(truth[window.id_column])
    forecast = predictions[target]
    quantiles = sorted(float(name) for name in forecast.column_names if name != "predictions")
    y_true = np.asarray(truth[target], dtype=np.float64)
    y_pred = np.asarray(forecast["predictions"], dtype=np.float64)
    q_pred = np.stack([np.asarray(forecast[str(q)], dtype=np.float64) for q in quantiles], axis=-1)
    y_past = [np.asarray(values, dtype=np.float64) for values in past[target]]
    return y_true, q_pred, y_pred, y_past


def fev_metrics(window, predictions, task):
    try:
        from fev.metrics import get_metric
    except ImportError as exc:
        raise ImportError("install demand-forecast[bench]") from exc

    return window.compute_metrics(
        predictions,
        metrics=[get_metric(metric) for metric in [task.eval_metric] + task.extra_metrics],
        seasonality=task.seasonality,
        quantile_levels=task.quantile_levels,
    )


def parity_case(tmp_path, case):
    """Edit the local parquet before windowing; forecast with the vendored baseline."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from demand_forecast.bench.baselines import SeasonalNaiveBaseline

    parquet = write_fev_fixture(tmp_path)
    table = pq.read_table(parquet)
    rows = table.to_pylist()
    horizon = 4
    for index, row in enumerate(rows):
        values = np.array(row["target"], dtype=np.float64)
        if case == "missing_labels":
            values[-horizon + 1] = np.nan
            values[-1] = np.nan
        elif (case == "constant_items" and index < 2) or case == "all_constant_items":
            values[:] = 5.0 + index
        elif case == "tiny_magnitudes":
            values *= 1e-6
        elif case == "unequal_label_counts":
            count = [4, 1, 3, 2, 0, 1][index]
            if count < horizon:
                values[len(values) - horizon + count :] = np.nan
        elif case == "no_labels_at_all":
            values[-horizon:] = np.nan
        row["target"] = values.tolist()
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), parquet)
    task = fixture_task(parquet, horizon=horizon, num_windows=1)
    predictions = task.clean_and_validate_predictions(SeasonalNaiveBaseline().fit_predict(task)[0])
    if case != "baseline":
        # Different errors by item and step expose incorrect aggregation. Keep
        # tiny-case errors small so epsilon changes the answer by many orders.
        magnitude = 1e-6 if case == "tiny_magnitudes" else 1.0

        def perturb(row, index):
            offset = magnitude * (3 * (index + 1) + np.arange(horizon))
            return {name: (np.asarray(values) + offset).tolist() for name, values in row.items()}

        predictions["target"] = predictions["target"].map(perturb, with_indices=True)
    return next(task.iter_windows()), predictions, task


@pytest.mark.parametrize(
    "case",
    [
        "baseline",
        "missing_labels",
        "constant_items",
        "all_constant_items",
        "tiny_magnitudes",
        "unequal_label_counts",
        "no_labels_at_all",
    ],
)
def test_metrics_match_fev(tmp_path, case):
    window, predictions, task = parity_case(tmp_path, case)
    y_true, q_pred, y_pred, y_past = arrays_from_window(window, predictions)
    np.testing.assert_array_equal(y_pred, q_pred[..., task.quantile_levels.index(0.5)])
    assert y_true.shape == y_pred.shape == (6, 4)
    assert q_pred.shape == (6, 4, 9)
    assert all(len(values) == 36 for values in y_past)
    scales = seasonal_error_per_item(y_past, 1)
    if case == "missing_labels":
        assert np.isnan(y_true).sum() == 12
    elif case == "constant_items":
        assert (scales == 0).sum() == 2
    elif case == "all_constant_items":
        assert (scales == 0).all()
    elif case == "tiny_magnitudes":
        assert 0 < np.mean(np.abs(y_true)) < 1e-4
    elif case == "unequal_label_counts":
        np.testing.assert_array_equal((~np.isnan(y_true)).sum(axis=1), [4, 1, 3, 2, 0, 1])
    elif case == "no_labels_at_all":
        assert np.isnan(y_true).all()

    ours = summarize(y_true, q_pred, y_past, m=1, quantiles=task.quantile_levels)
    theirs = fev_metrics(window, predictions, task)
    for key in ("SQL", "MASE", "WAPE", "WQL"):
        assert ours[key] == pytest.approx(theirs[key], rel=1e-9, abs=0, nan_ok=True), key
    assert mase(y_true, y_pred, y_past, 1) == pytest.approx(theirs["MASE"], rel=1e-9, nan_ok=True)
    assert wape(y_true, y_pred) == pytest.approx(theirs["WAPE"], rel=1e-9, nan_ok=True)
    if case == "all_constant_items":
        assert np.isnan(ours["SQL"]) and np.isnan(ours["MASE"])
        assert np.isfinite(ours["WAPE"]) and np.isfinite(ours["WQL"])
    elif case == "no_labels_at_all":
        assert all(np.isnan(value) for value in ours.values())


@pytest.mark.parametrize(
    "case",
    [
        "ragged_missing_history",
        "all_undefined_scales",
        "infinite_history",
        "nonfinite_quantiles",
        "empty_quantile",
    ],
)
def test_fev_edge_arithmetic_and_seasonality(tmp_path, case):
    window, predictions, task = parity_case(tmp_path, "baseline")
    if case == "ragged_missing_history":
        task.seasonality = 2

        def shorten(row, index):
            for name in ("timestamp", "target", "promo", "oil"):
                row[name] = row[name][index:]
            row["target"][2] = np.nan
            row["target"][4] = np.nan
            return row

        window.full_dataset = window.full_dataset.map(shorten, with_indices=True)
    elif case == "all_undefined_scales":
        task.seasonality = 40  # All histories are shorter than the lag.
    elif case == "infinite_history":

        def nonfinite_history(row, index):
            if index == 0:
                row["target"][:3] = [np.inf, np.inf, 1.0]
            elif index == 1:
                row["target"][: -window.horizon] = [np.nan] * (len(row["target"]) - window.horizon)
            return row

        window.full_dataset = window.full_dataset.map(nonfinite_history, with_indices=True)
    else:
        # compute_metrics handles these even though the forecasting contract
        # normally requires finite predictions. This distinguishes per-knot
        # averaging from a single nanmean over all cells and knots.
        def nonfinite_predictions(row, index):
            if case == "empty_quantile":
                row["0.1"] = [np.nan] * window.horizon
            elif index == 0:
                row["0.1"][0] = np.inf
                row["0.1"][1] = np.nan
                row["0.5"][0] = row["predictions"][0] = np.inf
            return row

        predictions["target"] = predictions["target"].map(nonfinite_predictions, with_indices=True)

    y_true, q_pred, y_pred, y_past = arrays_from_window(window, predictions)
    ours = summarize(y_true, q_pred, y_past, task.seasonality, task.quantile_levels)
    ours["MASE"] = mase(y_true, y_pred, y_past, task.seasonality)
    theirs = fev_metrics(window, predictions, task)
    for key in ("SQL", "MASE", "WAPE", "WQL"):
        assert ours[key] == pytest.approx(theirs[key], rel=1e-9, abs=0, nan_ok=True), key
    if case == "all_undefined_scales":
        assert np.isnan(ours["SQL"]) and np.isnan(ours["MASE"])
    elif case == "infinite_history":
        scales = seasonal_error_per_item(y_past, task.seasonality)
        assert np.isposinf(scales[0]) and np.isnan(scales[1])
    elif case == "empty_quantile":
        assert np.isnan(ours["SQL"]) and np.isnan(ours["WQL"])
