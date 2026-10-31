import pytest

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from tests.helpers.fev_fixture import fixture_task, write_fev_fixture  # noqa: E402


def test_ragged_fixture_round_trips_through_fev(tmp_path):
    import pyarrow.parquet as pq

    lengths = (40, 38, 40, 28, 38, 40)
    parquet = write_fev_fixture(tmp_path, lengths=lengths)
    for i, row in enumerate(pq.read_table(parquet).to_pylist()):
        assert all(len(row[name]) == lengths[i] for name in ("timestamp", "target", "promo", "oil"))
    task = fixture_task(parquet, num_windows=1)
    past, future = task.get_window(0).get_input_data()
    assert [len(row["timestamp"]) for row in past] == [length - 4 for length in lengths]
    assert [len(row["timestamp"]) for row in future] == [4] * 6
    assert len({row["timestamp"][0] for row in future}) == 3


@pytest.mark.parametrize("lengths", [(40, 38), (40, 38, 40, 41, 38, 40)])
def test_fixture_rejects_invalid_lengths(tmp_path, lengths):
    with pytest.raises(ValueError):
        write_fev_fixture(tmp_path, lengths=lengths)


def test_fixture_round_trips_through_fev(tmp_path):
    parquet = write_fev_fixture(tmp_path)
    task = fixture_task(parquet)
    windows = list(task.iter_windows())
    assert len(windows) == 2
    past, future = windows[0].get_input_data()
    assert {"id", "timestamp", "target", "promo", "oil", "cat"} <= set(past.column_names)
    assert {"id", "timestamp", "promo"} <= set(future.column_names)
    assert "target" not in future.column_names
    assert "oil" not in future.column_names
    assert len(future[0]["timestamp"]) == 4
    assert task.freq == "W-SUN"
    assert task.dataset_path == str(parquet)
    assert task.dataset_config is None
    assert task.quantile_levels == [i / 10 for i in range(1, 10)]
    assert task.eval_metric == "SQL"
    assert task.extra_metrics == [
        "MASE",
        {"name": "WAPE", "epsilon": 1.0},
        {"name": "WQL", "epsilon": 1.0},
    ]
    assert task.seasonality == 1


def test_fixture_schema_values_and_reproducibility(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = write_fev_fixture(tmp_path / "first")
    table = pq.read_table(parquet)
    assert parquet == tmp_path / "first" / "data.parquet"
    assert table.num_rows == 6
    assert table.schema.field("id").type == pa.string()
    assert table.schema.field("timestamp").type == pa.list_(pa.timestamp("ns"))
    assert table.schema.field("target").type == pa.list_(pa.float32())
    assert table.schema.field("cat").type == pa.string()
    assert len(set(table.column("id").to_pylist())) == 6
    for row in table.to_pylist():
        assert all(len(row[name]) == 40 for name in ("timestamp", "target", "promo", "oil"))
        assert all(value >= 0 for value in row["target"])
        assert row["target"].count(0) == 6
        assert set(row["promo"]) <= {0, 1}
        assert np.std(row["oil"]) > 0
        assert row["timestamp"][0].isoformat() == "2022-01-02T00:00:00"
    assert table.equals(pq.read_table(write_fev_fixture(tmp_path / "same", seed=0)))
    assert not table.equals(pq.read_table(write_fev_fixture(tmp_path / "different", seed=1)))


def test_promo_is_informative(tmp_path):
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(write_fev_fixture(tmp_path, n_series=2, length=2000, seed=7))
    targets = np.array(table.column("target").to_pylist())
    promo = np.array(table.column("promo").to_pylist(), dtype=bool)
    positive = targets > 0
    uplift = targets[promo & positive].mean() / targets[~promo & positive].mean()
    assert uplift == pytest.approx(1.5, rel=0.05)


@pytest.mark.parametrize(
    "known,past,statics", [((), (), ()), (("discount",), ("price",), ("store",))]
)
def test_fixture_supports_custom_roles_and_windows(tmp_path, known, past, statics):
    parquet = write_fev_fixture(
        tmp_path,
        n_series=3,
        length=24,
        freq="D",
        start="2023-01-01",
        known=known,
        past=past,
        statics=statics,
        seed=42,
    )
    task = fixture_task(parquet, horizon=3, num_windows=3, known=known, past=past, statics=statics)
    windows = list(task.iter_windows())
    assert len(windows) == 3
    assert task.freq == "D"
    for window in windows:
        history, future = window.get_input_data()
        assert len(history) == len(future) == 3
        assert set(history.column_names) == {"id", "timestamp", "target", *known, *past, *statics}
        assert set(future.column_names) == {"id", "timestamp", *known, *statics}
        assert all(len(row["timestamp"]) == 3 for row in future)


@pytest.mark.parametrize("dependency", ["fev", "datasets"])
def test_helpers_explain_missing_bench_dependency(tmp_path, monkeypatch, dependency):
    import sys

    monkeypatch.setitem(sys.modules, dependency, None)
    helper = fixture_task if dependency == "fev" else write_fev_fixture
    with pytest.raises(ImportError, match=r"^install demand-forecast\[bench\]$"):
        helper(tmp_path)
