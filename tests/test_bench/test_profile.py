import importlib.util
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from tests.helpers.fev_fixture import write_fev_fixture  # noqa: E402
from tests.helpers.fev_spec import fixture_spec  # noqa: E402

# Load the standalone script by path; scripts is not part of the installed package.
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "profile_fev.py"
module_spec = importlib.util.spec_from_file_location("profile_fev", SCRIPT)
profiler = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(profiler)
profile_task = profiler.profile_task


def test_profile_task_on_fixture(tmp_path):
    row = profile_task(fixture_spec(write_fev_fixture(tmp_path)))
    assert row["n_series"] == 6
    assert row["anchor_weekday"] == 6
    assert row["length_min"] == 40
    assert row["eligible_items_first_window"] == 6
    assert row["length_median"] == row["length_max"] == 40
    assert row["nan_share"] == row["negative_share"] == row["static_nan_share"] == 0.0
    assert row["zero_share"] == pytest.approx(0.15)
    assert row["zero_scale_items"] == 0
    assert set(row) == set(profiler.PROFILE_COLUMNS) - {"task"}


def test_profile_counts_distinct_first_window_origins(tmp_path):
    aligned = profile_task(fixture_spec(write_fev_fixture(tmp_path / "a")))
    ragged = profile_task(
        fixture_spec(
            write_fev_fixture(tmp_path / "b", lengths=(40, 38, 40, 28, 38, 40)),
            num_windows=1,
        )
    )
    assert aligned["distinct_origins_first_window"] == 1
    assert ragged["distinct_origins_first_window"] == 3
    assert profiler.PROFILE_COLUMNS[-1] == "distinct_origins_first_window"


def test_profile_counts_full_histories_but_eligibility_uses_first_past(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = write_fev_fixture(tmp_path, length=12)
    rows = pq.read_table(parquet).to_pylist()
    targets = [
        np.arange(-1, 11, dtype=float),
        np.zeros(12),
        np.full(12, np.nan),
        np.array([3.0] * 8 + [4.0, 5.0, 6.0, 7.0]),
        np.arange(6, dtype=float),
        np.arange(3, dtype=float),
    ]
    for index, (row, values) in enumerate(zip(rows, targets, strict=True)):
        row["target"] = values.tolist()
        for column in ("timestamp", "promo", "oil"):
            row[column] = row[column][: len(values)]
        row["cat"] = None if index < 2 else "A"
        row["size"] = None if index < 3 else 2.0
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    result = profile_task(
        fixture_spec(parquet, horizon=2, min_context_length=4, static_columns=["cat", "size"])
    )
    assert result["n_series"] == 6
    assert (result["length_min"], result["length_median"], result["length_max"]) == (3, 12, 12)
    assert result["nan_share"] == pytest.approx(12 / 57)
    assert result["negative_share"] == pytest.approx(1 / 57)
    assert result["zero_share"] == pytest.approx(15 / 57)
    assert result["static_nan_share"] == pytest.approx(5 / 12)
    assert result["zero_scale_items"] == 2
    assert result["eligible_items_first_window"] == 1


@pytest.mark.parametrize("seasonality,zero_scale,eligible", [(1, 0, 1), (2, 1, 0), (20, 1, 0)])
def test_profile_uses_task_seasonality_and_observed_pairs(
    tmp_path, seasonality, zero_scale, eligible
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = write_fev_fixture(tmp_path, n_series=1, length=12, statics=())
    rows = pq.read_table(parquet).to_pylist()
    rows[0]["target"] = [np.nan, 2.0] + [1.0, 2.0] * 5
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    result = profile_task(
        fixture_spec(parquet, seasonality=seasonality, static_columns=[], categorical_statics=[])
    )
    assert result["zero_scale_items"] == zero_scale
    assert result["eligible_items_first_window"] == eligible
    assert result["static_nan_share"] == 0.0


def test_profile_uses_declared_columns_and_first_timestamp(tmp_path):
    import pyarrow.parquet as pq

    parquet = write_fev_fixture(tmp_path, freq="W-WED", start="2022-01-05")
    table = pq.read_table(parquet)
    names = {"id": "item", "timestamp": "date", "target": "sales"}
    pq.write_table(
        table.rename_columns([names.get(col, col) for col in table.column_names]), parquet
    )
    result = profile_task(
        fixture_spec(parquet, id_column="item", timestamp_column="date", target="sales")
    )
    assert result["anchor_weekday"] == 2
    assert result["eligible_items_first_window"] == 6
    assert result["distinct_origins_first_window"] == 1


def test_profile_empty_first_window_has_zero_eligible_items(tmp_path):
    result = profile_task(fixture_spec(write_fev_fixture(tmp_path), min_context_length=100))
    assert result["n_series"] == 6
    assert result["eligible_items_first_window"] == 0
    assert result["distinct_origins_first_window"] == 0


def test_profile_resume_preserves_completed_tasks_after_interruption(tmp_path, monkeypatch):
    specs = [
        fixture_spec(write_fev_fixture(tmp_path / name), name=name) for name in ("first", "second")
    ]
    monkeypatch.setattr(profiler, "load_task_specs", lambda *args: specs)

    def interrupted(spec):
        if spec.name == "second":
            raise RuntimeError("interrupted")
        return profile_task(spec)

    monkeypatch.setattr(profiler, "profile_task", interrupted)
    output = tmp_path / "results" / "profile.csv"
    command = ["--tasks", "weekly", "--out", str(output), "--resume"]
    with pytest.raises(RuntimeError, match="interrupted"):
        profiler.main(command)
    saved = pd.read_csv(output)
    assert saved["task"].tolist() == ["first"]

    def remaining(spec):
        assert spec.name == "second", "a completed task must not be loaded again"
        return profile_task(spec)

    monkeypatch.setattr(profiler, "profile_task", remaining)
    profiler.main(command)
    resumed = pd.read_csv(output)
    assert resumed["task"].tolist() == ["first", "second"]
    pd.testing.assert_frame_equal(resumed.iloc[:1], saved)
    before = output.read_bytes()
    profiler.main(command)
    assert output.read_bytes() == before


def _profile_cli_command(tmp_path):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    definition = asdict(spec)
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(
        yaml.safe_dump(
            {
                spec.name: {
                    "target_support": definition.pop("target_support"),
                    "categorical_statics": definition.pop("categorical_statics"),
                }
            }
        )
    )
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text(yaml.safe_dump({"tasks": [definition]}))
    output = tmp_path / "profile.csv"
    command = [
        sys.executable,
        str(SCRIPT),
        "--tasks",
        str(tasks),
        "--profiles",
        str(profiles),
        "--out",
        str(output),
        "--resume",
    ]
    return output, command


def test_profile_cli_with_local_yaml(tmp_path):
    output, command = _profile_cli_command(tmp_path)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    frame = pd.read_csv(output)
    assert frame.columns.tolist() == profiler.PROFILE_COLUMNS
    assert frame["task"].tolist() == ["fixture"]
    assert frame["eligible_items_first_window"].tolist() == [6]


def test_resume_rejects_twelve_column_profile_and_leaves_it_unchanged(tmp_path):
    output, command = _profile_cli_command(tmp_path)
    columns = profiler.PROFILE_COLUMNS[:12]
    pd.DataFrame([{"task": "fixture", **dict.fromkeys(columns[1:], 0)}], columns=columns).to_csv(
        output, index=False
    )
    before = output.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "invalid profile CSV: expected profile columns and unique task names" in result.stderr
    assert output.read_bytes() == before
