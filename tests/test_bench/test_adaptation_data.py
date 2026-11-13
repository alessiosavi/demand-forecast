"""Offline cut-off, native-row, adapter and scoring contracts for adaptation."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
fev = pytest.importorskip("fev")
datasets = pytest.importorskip("datasets")
pytestmark = pytest.mark.bench

from demand_forecast.bench.adaptation.data import build_manifest, build_task_manifest  # noqa: E402
from demand_forecast.bench.yardsticks.adapters import SimonsAdapter  # noqa: E402
from demand_forecast.bench.yardsticks.runner import Options, prepare_windows  # noqa: E402


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    import datasets.config
    import huggingface_hub.constants

    monkeypatch.setattr(datasets.config, "HF_DATASETS_OFFLINE", True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def task_rows(*, lengths=(24, 22), starts=(0, 0), channels=1, known=True):
    rows = []
    targets = [f"y{i}" for i in range(channels)]
    for i, (length, start) in enumerate(zip(lengths, starts, strict=True)):
        rows.append(
            {
                "id": f"item-{i}",
                "timestamp": pd.date_range("2020-01-01", periods=length + start, freq="D")[start:]
                .to_numpy()
                .copy(),
                **{
                    c: (np.arange(length, dtype=float) + j * 100 + i) for j, c in enumerate(targets)
                },
                **(
                    {
                        "promo": np.arange(length) % 2,
                        "event": ["a", "b", None] * (length // 3) + ["a"] * (length % 3),
                    }
                    if known
                    else {}
                ),
                "past": np.full(length, 9999.0),
                "static": "SECRET",
            }
        )
    return rows, targets


class SyntheticTask:
    """fev-shaped seam avoids loading/filtering away short items."""

    def __init__(self, rows, targets, *, cutoff=-4, known=True, max_context=None):
        self.rows = rows
        self.target_columns = targets
        self.horizon = 2
        self.initial_cutoff = cutoff
        self.cutoffs = [cutoff]
        self.min_context_length = 1
        self.max_context_length = max_context
        self.id_column = "id"
        self.timestamp_column = "timestamp"
        self.known_dynamic_columns = ["promo", "event"] if known else []
        self.past_dynamic_columns = ["past"]
        self.static_columns = ["static"]
        self.task_name = "family/task"
        self.freq = "D"
        self.eval_metric = "SQL"
        self.extra_metrics = ["MASE"]
        self.seasonality = 2
        self.quantile_levels = [0.1, 0.5, 0.9]

    def load_full_dataset(self):
        return datasets.Dataset.from_list(self.rows)

    def to_dict(self):
        return {
            "task_name": self.task_name,
            "initial_cutoff": self.initial_cutoff,
            "horizon": self.horizon,
        }


def build(tmp_path, rows=None, targets=None, *, name="out", **kwargs):
    if rows is None:
        rows, targets = task_rows()
    task = SyntheticTask(rows, targets, **kwargs)
    yaml = tmp_path / "tasks.yaml"
    yaml.write_text("tasks: []\n")
    manifest = build_manifest(task, task_yaml=yaml, output=tmp_path / name)
    return manifest, tmp_path / name


def files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def inner(root, i):
    return json.loads((root / f"inner-{i}.json").read_text())


def test_adaptation_native_layout_roles_and_ragged_cutoffs(tmp_path):
    manifest, root = build(tmp_path)
    assert manifest == json.loads((root / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["phase"] == "development"
    assert manifest["task"]["name"] == "family/task"
    assert len(manifest["fingerprint"]) == 64
    assert manifest["cutoffs"]["shared_training_boundary"] == "2020-01-15T00:00:00"
    assert [i["outer_origin"]["index"] for i in manifest["items"]] == [20, 18]
    assert [i["training_end_index"] for i in manifest["items"]] == [14, 14]
    assert manifest["covariates"]["past_only_dropped"] == ["past"]
    assert manifest["covariates"]["static_dropped"] == ["static"]
    train = pq.read_table(root / "univariate/train-00000.parquet").to_pylist()
    assert len(train) == 2
    assert np.asarray(train[0]["target"]).shape == (1, 14)
    assert set(train[0]) == {"item_id", "source", "freq", "target"}
    cov = pq.read_table(root / "covariates/train-00000.parquet").to_pylist()
    assert len(cov) == 2
    assert np.asarray(cov[0]["target"]).shape == (14,)
    assert np.asarray(cov[0]["covariates"]).shape == (2, 14)
    assert manifest["counts"]["training_target_values"] == 28
    assert manifest["covariates"]["dropped_value_counts"] == {"past_only": 38, "static": 2}
    for i in range(2):
        record = inner(root, i)["items"][0]
        length = 16 + i * 2
        assert np.asarray(record["model_input"][0]["target"]).shape == (1, length)
        assert len(record["model_input"][0]["known_covariates"]["promo"]) == length + 2
        assert np.asarray(record["score_target"]).shape == (1, 2)
        assert np.asarray(record["score_past"]).shape == (1, length)
    assert b"SECRET" not in b"".join(files(root).values())
    assert b"9999" not in b"".join(files(root).values())


def test_adaptation_groups_match_adapter_blocks(tmp_path):
    rows, targets = task_rows(channels=35)
    manifest, root = build(tmp_path, rows, targets)
    native = pq.read_table(root / "univariate/train-00000.parquet").to_pylist()
    assert [np.asarray(r["target"]).shape[0] for r in native] == [32, 3, 32, 3]
    assert manifest["counts"]["channels"] == 70
    assert [len(g["target_indices"]) for g in inner(root, 0)["items"][0]["model_input"]] == [32, 3]
    assert len(pq.read_table(root / "covariates/train-00000.parquet")) == 70


@pytest.mark.parametrize("cutoff", [18, -6, "2020-01-18"])
def test_adaptation_integer_and_inclusive_timestamp_origins(tmp_path, cutoff):
    rows, targets = task_rows(lengths=(24,), starts=(0,))
    manifest, _ = build(tmp_path, rows, targets, cutoff=cutoff)
    item = manifest["items"][0]
    assert item["outer_origin"] == {"index": 18, "timestamp": "2020-01-19T00:00:00"}
    assert [o["index"] for o in item["inner_origins"]] == [14, 16]
    assert item["training_end_index"] == 14


def test_adaptation_short_and_empty_items_keep_fallback_coverage(tmp_path):
    rows, targets = task_rows(lengths=(24, 8, 5, 2), starts=(0, 0, 0, 0), known=False)
    manifest, root = build(tmp_path, rows, targets, known=False)
    assert manifest["counts"]["items"] == 4
    assert manifest["counts"]["eligible_items"] == 1
    assert len(manifest["excluded_items"]) == 3
    assert {i["reason"] for i in manifest["excluded_items"]} == {
        "insufficient_inner_history",
        "insufficient_outer_history",
    }
    assert not (root / "covariates").exists()
    all_short, _ = build(tmp_path, rows[1:], targets, name="short", known=False)
    assert all_short["counts"]["eligible_items"] == 0
    assert all_short["status"] == "frozen_fallback"


@pytest.mark.parametrize("where", ["training_boundary", "outer_origin"])
def test_adaptation_leakage_perturbation_invariance(tmp_path, where):
    rows, targets = task_rows(lengths=(24,), starts=(0,))
    _, a = build(tmp_path, rows, targets, name="a")
    changed = deepcopy(rows)
    start = 16 if where == "training_boundary" else 20
    for column in [*targets, "promo", "event", "past"]:
        changed[0][column] = list(changed[0][column])
        changed[0][column][start:] = (
            [123456] * (24 - start) if column != "event" else ["UNSEEN"] * (24 - start)
        )
    _, b = build(tmp_path, changed, targets, name="b")
    if where == "outer_origin":
        assert files(a) == files(b)
    else:
        assert (a / "univariate/train-00000.parquet").read_bytes() == (
            b / "univariate/train-00000.parquet"
        ).read_bytes()
        assert (a / "covariates/train-00000.parquet").read_bytes() == (
            b / "covariates/train-00000.parquet"
        ).read_bytes()
        assert (
            json.loads((a / "manifest.json").read_text())["fingerprint"]
            != json.loads((b / "manifest.json").read_text())["fingerprint"]
        )


@pytest.mark.parametrize("origin", [0, 1])
def test_adaptation_inner_target_changes_only_causally_dependent_material(tmp_path, origin):
    rows, targets = task_rows(lengths=(24,), starts=(0,))
    _, a = build(tmp_path, rows, targets, name="a")
    rows[0]["y0"][16 + origin * 2] += 0.25
    _, b = build(tmp_path, rows, targets, name="b")
    changed = {name for name, value in files(a).items() if files(b)[name] != value}
    expected = {"manifest.json", f"inner-{origin}.json"}
    if origin == 0:
        expected.add("inner-1.json")  # earlier labels become later history
    assert changed == expected


def test_adaptation_deterministic_and_sensitive_to_permitted_input(tmp_path):
    rows, targets = task_rows()
    first, a = build(tmp_path, rows, targets, name="a")
    second, b = build(tmp_path, rows, targets, name="b")
    assert files(a) == files(b)
    assert first == second
    rows[0]["y0"][0] += 0.5
    third, _ = build(tmp_path, rows, targets, name="c")
    assert first["fingerprint"] != third["fingerprint"]
    with pytest.raises(FileExistsError):
        build(tmp_path, rows, targets, name="a")


@pytest.mark.parametrize(
    "fault",
    ["duplicate", "descending", "invalid_date", "ragged_target", "ragged_known", "infinity"],
)
def test_adaptation_rejects_invalid_requests_without_partial_output(tmp_path, fault):
    rows, targets = task_rows(lengths=(24,), starts=(0,))
    cutoff = -4
    if fault == "duplicate":
        rows[0]["timestamp"][1] = rows[0]["timestamp"][0]
    elif fault == "descending":
        rows[0]["timestamp"] = rows[0]["timestamp"][::-1]
    elif fault == "invalid_date":
        cutoff = "not-a-date"
    elif fault == "ragged_target":
        rows[0]["y0"] = rows[0]["y0"][:-1]
    elif fault == "ragged_known":
        rows[0]["promo"] = rows[0]["promo"][:-1]
    else:
        rows[0]["y0"][0] = np.inf
    with pytest.raises(ValueError):
        build(tmp_path, rows, targets, cutoff=cutoff)
    assert not (tmp_path / "out").exists()


def cutoff_task(tmp_path, rows, cutoff, step, *, timestamp_type=None):
    """Load synthetic Parquet through the real fev validation/window path."""
    parquet = tmp_path / "source.parquet"
    dataset = datasets.Dataset.from_list(rows)
    if timestamp_type:
        dataset = dataset.cast_column(
            "timestamp", datasets.Sequence(datasets.Value(timestamp_type))
        )
    dataset.to_parquet(parquet)
    return fev.Task(
        dataset_path=str(parquet),
        task_name="cutoff/parity",
        target="target",
        horizon=2,
        initial_cutoff=cutoff,
        num_windows=2,
        window_step_size=step,
        min_context_length=2,
        max_context_length=5,
    )


@pytest.mark.parametrize(
    "freq,date",
    [
        ("W-SUN", "2020-05-08"),  # Friday between weekly observations, as in the pilot.
        ("ME", "2021-07-15"),  # An anchored monthly offset must not shift window zero.
        ("D", "2020-01-19"),
        ("6h", "2020-01-05T13:00:00"),
    ],
)
@pytest.mark.parametrize(
    "cutoff_form,step",
    [
        ("positive", None),
        ("positive", 2),
        ("negative", None),
        ("negative", 2),
        ("date", None),
        ("date", 2),
        ("date", "2D"),
        ("date", "2ME"),
    ],
)
def test_adaptation_first_origin_matches_real_fev(tmp_path, freq, date, cutoff_form, step):
    times = pd.date_range("2020-01-01", periods=36, freq=freq)
    rows = [
        {"id": key, "timestamp": ts, "target": np.arange(len(ts), dtype=float)}
        for key, ts in [
            ("full", times),
            ("ragged", times[3:32]),
            ("ended", times[:8]),
            ("late", times[28:]),
        ]
    ]
    cutoff = {"positive": 19, "negative": -6, "date": date}[cutoff_form]
    task = cutoff_task(tmp_path, rows, cutoff, step)
    full = {r["id"]: pd.DatetimeIndex(r["timestamp"]) for r in task.load_full_dataset()}
    window = next(iter(task.iter_windows()))
    past, future = window.get_input_data()
    # Derive T independently from fev's first forecast timestamp, even when
    # max_context_length makes the returned history shorter than T.
    expected = {
        row["id"]: int(full[row["id"]].get_loc(pd.Timestamp(row["timestamp"][0]))) for row in future
    }
    assert {row["id"]: len(row["timestamp"]) for row in past} == {
        key: min(origin, 5) for key, origin in expected.items()
    }
    yaml = tmp_path / "tasks.yaml"
    yaml.write_text("tasks: []\n")
    manifest = build_manifest(task, task_yaml=yaml, output=tmp_path / "out")
    for item in manifest["items"]:
        key, origin = item["item_id"], item["outer_origin"]
        if key in expected:
            assert origin == {
                "index": expected[key],
                "timestamp": full[key][expected[key]].isoformat(),
            }
            assert [o["index"] for o in item["inner_origins"]] == [
                expected[key] - 4,
                expected[key] - 2,
            ]
    excluded_outer = {
        row["item_id"]
        for row in manifest["excluded_items"]
        if row["reason"] == "insufficient_outer_history"
    }
    assert excluded_outer == set(full) - set(expected)
    if cutoff_form == "date":
        assert excluded_outer == {"ended", "late"}
        origins = {row["item_id"]: row["outer_origin"]["index"] for row in manifest["items"]}
        assert origins["ended"] == 8
        assert origins["late"] == 0


@pytest.mark.parametrize(
    "cutoff,expected", [(100, 24), (-100, 0), ("2100-01-01", 24), ("1900-01-01", 0)]
)
def test_adaptation_out_of_range_cutoff_matches_real_fev(tmp_path, cutoff, expected):
    rows = [
        {"id": "item", "timestamp": pd.date_range("2020-01-01", periods=24), "target": [1.0] * 24}
    ]
    task = cutoff_task(tmp_path, rows, cutoff, 2)
    with pytest.raises(ValueError, match="All time series.*too short"):
        next(iter(task.iter_windows())).get_input_data()
    yaml = tmp_path / "tasks.yaml"
    yaml.write_text("tasks: []\n")
    manifest = build_manifest(task, task_yaml=yaml, output=tmp_path / "out")
    assert manifest["items"][0]["outer_origin"]["index"] == expected
    assert manifest["status"] == "frozen_fallback"
    assert manifest["excluded_items"] == [
        {"item_id": "item", "reason": "insufficient_outer_history", "fallback": "frozen_model"}
    ]


@pytest.mark.parametrize("irregular_item", [0, 1])
def test_adaptation_irregular_timestamps_rejected_by_real_fev(tmp_path, irregular_item):
    times = pd.date_range("2020-01-01", periods=24).delete(10)
    rows = [
        {"id": "a", "timestamp": pd.date_range("2020-01-01", periods=24), "target": [1.0] * 24},
        {"id": "b", "timestamp": pd.date_range("2020-01-01", periods=24), "target": [1.0] * 24},
    ]
    rows[irregular_item].update(timestamp=times, target=[1.0] * len(times))
    task = cutoff_task(tmp_path, rows, "2020-01-15T12:00:00", "2D")
    with pytest.raises(AssertionError, match="infer.*frequency"):
        next(iter(task.iter_windows())).get_input_data()
    yaml = tmp_path / "tasks.yaml"
    yaml.write_text("tasks: []\n")
    with pytest.raises(AssertionError, match="infer.*frequency"):
        build_manifest(task, task_yaml=yaml, output=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "timestamp_type,cutoff",
    [
        ("timestamp[us]", "2020-01-19T00:00:00+01:00"),  # Arrow rejects zone mismatch.
        ("timestamp[us]", "2020-01-19T00:00:00.000000001"),  # Not representable in source units.
    ],
)
def test_adaptation_unresolvable_date_only_when_real_fev_rejects(tmp_path, timestamp_type, cutoff):
    rows = [
        {"id": "item", "timestamp": pd.date_range("2020-01-01", periods=24), "target": [1.0] * 24}
    ]
    task = cutoff_task(tmp_path, rows, cutoff, "2D", timestamp_type=timestamp_type)
    with pytest.raises(pa.ArrowInvalid):
        next(iter(task.iter_windows())).get_input_data()
    yaml = tmp_path / "tasks.yaml"
    yaml.write_text("tasks: []\n")
    with pytest.raises(ValueError, match="unresolvable date cut-off"):
        build_manifest(task, task_yaml=yaml, output=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_adaptation_adapter_and_fev_metric_parity(tmp_path):
    rows, targets = task_rows(lengths=(40, 38), starts=(0, 2), channels=3)
    parquet = tmp_path / "source.parquet"
    datasets.Dataset.from_list(rows).to_parquet(parquet)
    task = fev.Task(
        dataset_path=str(parquet),
        task_name="family/task",
        target=targets,
        horizon=2,
        initial_cutoff=-4,
        seasonality=2,
        eval_metric="SQL",
        quantile_levels=[0.1, 0.5, 0.9],
        known_dynamic_columns=["promo", "event"],
        past_dynamic_columns=["past"],
        static_columns=["static"],
        max_context_length=9,
    )
    yaml = tmp_path / "tasks.yaml"
    yaml.write_text("tasks: []\n")
    build_manifest(task, task_yaml=yaml, output=tmp_path / "out", max_context=5)
    for i, cutoff in enumerate([-8, -6]):
        shifted = replace(task, initial_cutoff=cutoff)
        options = Options(
            "simons", "unused", "recording", processing="raw", max_context=5
        ).resolved()
        calls = []
        adapter = SimonsAdapter.__new__(SimonsAdapter)
        adapter.options, adapter.model = options, None

        def record(model, series, horizon, **kwargs):
            calls.append((series, kwargs))
            return [np.zeros((len(a), horizon, 9)) for a in series]

        adapter.forecast_series = record
        adapter.predict(prepare_windows(shifted, options), 2, shifted.quantile_levels)
        data = inner(tmp_path / "out", i)
        for record in data["items"]:
            origin = record["origin"]["index"]
            assert all(len(channel) == 9 for channel in record["score_past"])
            assert record["score_history_start_index"] == origin - 9
            assert record["history_start_index"] == origin - 5
        groups = [g for it in data["items"] for g in it["model_input"]]
        for group, recorded, cov in zip(
            groups, calls[0][0], calls[0][1]["future_covariates"], strict=True
        ):
            np.testing.assert_array_equal(group["target"], recorded)
            for c, values in cov.items():
                assert group["known_covariates"][c] == values.tolist()
        window = shifted.get_window(0)
        predictions = {
            t: [
                {"predictions": [0.0, 0.0], **{str(q): [q, q] for q in task.quantile_levels}}
                for _ in rows
            ]
            for t in targets
        }
        cleaned = shifted.clean_and_validate_predictions(predictions)
        from fev.metrics import get_metric

        expected = window.compute_metrics(
            cleaned, [get_metric(shifted.eval_metric)], task.seasonality, task.quantile_levels
        )["SQL"]

        truth = np.stack([np.asarray(it["score_target"]).T for it in data["items"]])
        past = np.concatenate([np.asarray(it["score_past"]).T for it in data["items"]])
        q_pred = np.broadcast_to(task.quantile_levels, (*truth.shape, 3))
        score = get_metric(data["scoring"]["eval_metric"]).compute_scores(
            y_true=truth,
            y_pred=np.zeros_like(truth),
            q_pred=q_pred,
            y_past=past,
            y_past_lengths=np.array([len(it["score_past"][0]) for it in data["items"]]),
            seasonality=data["scoring"]["seasonality"],
            quantile_levels=data["scoring"]["quantile_levels"],
        )
        assert float(np.mean(score["SQL"])) == pytest.approx(expected)


def test_adaptation_public_seam_accepts_numpy_boolean_covariates(tmp_path):
    class ScalarRows(list):
        """Dataset seam retaining NumPy scalars even with Python formatting."""

        def select_columns(self, columns):
            return ScalarRows({c: row[c] for c in columns} for row in self)

        def with_format(self, formatter):
            assert formatter is None
            return self

    rows, targets = task_rows(lengths=(24,), starts=(0,))
    rows[0]["promo"] = [np.bool_(i % 2) for i in range(24)]
    integer = 2**53 + 2**29 + 1
    rows[0]["integer"] = [np.int64(integer)] * 24
    task = SyntheticTask(rows, targets)
    task.known_dynamic_columns.append("integer")
    task.load_full_dataset = lambda: ScalarRows(rows)
    yaml = tmp_path / "tasks.yaml"
    yaml.write_text("tasks: []\n")
    manifest = build_manifest(task, task_yaml=yaml, output=tmp_path / "out")
    assert manifest["covariates"]["training_encodings"][0]["columns"]["promo"] == {
        "kind": "numeric"
    }
    cov = pq.read_table(tmp_path / "out/covariates/train-00000.parquet").to_pylist()
    assert cov[0]["covariates"][0] == [float(i % 2) for i in range(16)]
    assert cov[0]["covariates"][2] == [float(np.float32(np.float64(integer)))] * 16


@pytest.mark.parametrize("formatter", ["numpy", None])
@pytest.mark.parametrize("target_dtype", ["float64", "int64"])
def test_adaptation_real_fev_matches_scorer_nulls_booleans_and_precision(
    tmp_path, formatter, target_dtype
):
    rows, targets = task_rows(lengths=(40,), starts=(0,))
    row = rows[0]
    row["promo"] = (np.arange(40) % 2).astype(bool)
    # Establish the unchanged scorer's treatment of actual/literal nulls.
    row["event"] = ["a", None, "None", "nan", "b"] * 8
    row["nullable"] = [1.0000000001] * 40
    row["y0"] = [1.0000000001 + i for i in range(40)]
    if target_dtype == "int64":
        # Direct int64->float32 differs from the scorer's float64 intermediate.
        row["y0"] = [2**53 + 2**29 + 1 + i for i in range(40)]
    for index in (0, 28, 32, 34):
        row["nullable"][index] = None
        if target_dtype == "float64":
            row["y0"][index] = None
    row["static"] = None
    parquet = tmp_path / "source.parquet"
    datasets.Dataset.from_list(rows).to_parquet(parquet)
    task = fev.Task(
        dataset_path=str(parquet),
        target=targets,
        horizon=2,
        initial_cutoff=-4,
        known_dynamic_columns=["promo", "event", "nullable"],
        past_dynamic_columns=["past"],
        static_columns=["static"],
        max_context_length=9,
        eval_metric="SQL",
        quantile_levels=[0.1, 0.5, 0.9],
    )
    full = task.load_full_dataset()
    assert full.format["type"] == "numpy"
    assert full.features["y0"].feature.dtype == target_dtype
    task._full_dataset = full.with_format(formatter)
    yaml = tmp_path / "tasks.yaml"
    yaml.write_text("tasks: []\n")
    manifest = build_manifest(task, task_yaml=yaml, output=tmp_path / "numpy", max_context=5)
    encodings = manifest["covariates"]["training_encodings"][0]["columns"]
    assert encodings["event"]["categories"] == ["None", "a", "b", "nan"]
    assert encodings["promo"] == encodings["nullable"] == {"kind": "numeric"}
    train = pq.read_table(tmp_path / "numpy/univariate/train-00000.parquet").to_pylist()
    cov = pq.read_table(tmp_path / "numpy/covariates/train-00000.parquet").to_pylist()
    native_covariates = dict(zip(task.known_dynamic_columns, cov[0]["covariates"], strict=True))
    if target_dtype == "float64":
        assert np.isnan(train[0]["target"][0][0])
    assert np.isnan(native_covariates["event"][1])
    assert native_covariates["event"][2] == encodings["event"]["categories"].index("None")
    assert np.isnan(native_covariates["nullable"][0])
    assert native_covariates["nullable"][1] == float(np.float32(np.float64(row["nullable"][1])))
    assert native_covariates["promo"] == [float(v) for v in row["promo"][:32]]
    np.testing.assert_equal(
        train[0]["target"][0], np.asarray(row["y0"][:32], dtype=np.float64).astype(np.float32)
    )
    assert manifest["counts"]["training_observed_target_values"] == (
        30 if target_dtype == "float64" else 32
    )
    options = Options("simons", "unused", "recording", processing="raw", max_context=5).resolved()
    for i, cutoff in enumerate((-8, -6)):
        record = inner(tmp_path / "numpy", i)["items"][0]
        start, origin = record["history_start_index"], record["origin"]["index"]
        assert record["score_past"][0] == [
            None if v is None else float(v) for v in row["y0"][origin - 9 : origin]
        ]
        assert record["score_target"][0] == [
            None if v is None else float(v) for v in row["y0"][origin : origin + 2]
        ]
        known = record["model_input"][0]["known_covariates"]
        assert known["event"] == row["event"][start : origin + 2]
        assert known["promo"] == list(row["promo"][start : origin + 2])
        expected_numeric = np.asarray(
            row["nullable"][start : origin + 2],
            dtype=np.float32 if formatter == "numpy" else np.float64,
        )
        np.testing.assert_equal(np.asarray(known["nullable"], dtype=np.float64), expected_numeric)
        calls = []
        adapter = SimonsAdapter.__new__(SimonsAdapter)
        adapter.options, adapter.model = options, None

        def record_call(model, series, horizon, **kwargs):
            calls.append((series, kwargs))
            return [np.zeros((len(a), horizon, 9)) for a in series]

        adapter.forecast_series = record_call
        shifted = replace(task, initial_cutoff=cutoff)
        shifted._full_dataset = task._full_dataset
        adapter.predict(prepare_windows(shifted, options), 2, shifted.quantile_levels)
        np.testing.assert_equal(
            np.asarray(record["model_input"][0]["target"], dtype=np.float32), calls[0][0][0]
        )
        for column, values in calls[0][1]["future_covariates"][0].items():
            if column == "nullable":
                np.testing.assert_equal(
                    np.asarray(values, dtype=np.float64),
                    np.asarray(known[column], dtype=np.float64),
                )
                if formatter == "numpy":
                    assert all(v is not None for v in values)
            else:
                assert values.tolist() == known[column]
                if column == "promo":
                    assert all(isinstance(v, bool) for v in values)
                else:
                    assert None in values  # Text null remains missing, not a "None" category.
    # Dropped nullable/static cells cannot affect the permitted representations.
    rows[0]["static"] = "OTHER SECRET"
    rows[0]["past"] = [None] * 40
    task._full_dataset = datasets.Dataset.from_list(rows).with_format(formatter)
    build_manifest(task, task_yaml=yaml, output=tmp_path / "dropped", max_context=5)
    assert files(tmp_path / "numpy") == files(tmp_path / "dropped")
    assert task._full_dataset.format["type"] == formatter
    assert full.format["type"] == "numpy"  # Builder and runner must not mutate shared data.


@pytest.mark.parametrize("invalid_timestamp", [False, True])
def test_adaptation_formatter_preserves_timestamp_zones_and_rejects_nulls(
    tmp_path, invalid_timestamp
):
    rows, targets = task_rows(lengths=(24,), starts=(0,), known=False)
    times = list(pd.date_range("2020-01-01", periods=24, freq="D", tz="Europe/Rome"))
    if invalid_timestamp:
        times[2] = None
    rows[0]["timestamp"] = times
    task = SyntheticTask(rows, targets, cutoff="2020-01-18T00:00:00+01:00", known=False)
    yaml = tmp_path / "tasks.yaml"
    yaml.write_text("tasks: []\n")
    for formatter, name in ((None, "python"), ("numpy", "numpy")):
        task.load_full_dataset = lambda formatter=formatter: datasets.Dataset.from_list(
            rows
        ).with_format(formatter)
        if invalid_timestamp:
            with pytest.raises(
                ValueError, match="timestamps must be strictly monotonic and non-null"
            ):
                build_manifest(task, task_yaml=yaml, output=tmp_path / name)
            assert not (tmp_path / name).exists()
        else:
            manifest = build_manifest(task, task_yaml=yaml, output=tmp_path / name)
            assert manifest["items"][0]["outer_origin"] == {
                "index": 18,
                "timestamp": "2020-01-19T00:00:00+01:00",
            }
    if not invalid_timestamp:
        assert files(tmp_path / "numpy") == files(tmp_path / "python")


def test_adaptation_yaml_selection_and_cli(tmp_path):
    import subprocess
    import sys

    rows, targets = task_rows(lengths=(24,), starts=(0,), known=False)
    parquet = tmp_path / "source.parquet"
    datasets.Dataset.from_list(rows).to_parquet(parquet)
    yaml = tmp_path / "tasks.yaml"
    yaml.write_text(
        "tasks:\n- name: family/task\n  dataset_path: source.parquet\n  target: y0\n  horizon: 2\n  initial_cutoff: -4\n  eval_metric: SQL\n  quantile_levels: [0.1, 0.5, 0.9]\n"
    )
    manifest = build_task_manifest(yaml, "family/task", tmp_path / "library")
    assert manifest["task"]["name"] == "family/task"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "demand_forecast.bench.adaptation",
            "--tasks",
            str(yaml),
            "--task",
            "family/task",
            "--output",
            str(tmp_path / "cli"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert files(tmp_path / "library") == files(tmp_path / "cli")
    assert manifest["fingerprint"] in result.stdout
    with pytest.raises(ValueError, match="names"):
        build_task_manifest(yaml, "unknown", tmp_path / "missing")


def test_adaptation_many_short_items_and_shared_calendar_exclusion(tmp_path):
    rows, targets = task_rows(lengths=(24,) * 60 + (6,) * 40, starts=(0,) * 100, known=False)
    manifest, _ = build(tmp_path, rows, targets, known=False)
    assert manifest["counts"]["items"] == 100
    assert manifest["counts"]["eligible_items"] == 60
    assert len(manifest["excluded_items"]) == 40
    rows, targets = task_rows(lengths=(24, 24), starts=(0, 20), known=False)
    manifest, _ = build(tmp_path, rows, targets, known=False, name="staggered")
    assert manifest["counts"]["eligible_items"] == 1
    assert manifest["excluded_items"] == [
        {
            "item_id": "item-1",
            "reason": "insufficient_shared_training_history",
            "fallback": "frozen_model",
        }
    ]


def test_adaptation_monthly_timestamp_cutoff_uses_observed_steps(tmp_path):
    rows, targets = task_rows(lengths=(24,), starts=(0,), known=False)
    rows[0]["timestamp"] = pd.date_range("2020-01-31", periods=24, freq="ME").to_numpy()
    task = SyntheticTask(rows, targets, cutoff="2021-06-30", known=False)
    task.freq = "ME"
    yaml = tmp_path / "tasks.yaml"
    yaml.write_text("tasks: []\n")
    manifest = build_manifest(task, task_yaml=yaml, output=tmp_path / "monthly")
    assert manifest["items"][0]["outer_origin"]["timestamp"] == "2021-07-31T00:00:00"
    assert manifest["items"][0]["training_boundary"]["timestamp"] == "2021-03-31T00:00:00"
    assert inner(tmp_path / "monthly", 1)["items"][0]["horizon_timestamps"] == [
        "2021-05-31T00:00:00",
        "2021-06-30T00:00:00",
    ]


def test_adaptation_dropped_covariates_have_no_effect_on_identity(tmp_path):
    rows, targets = task_rows()
    _, a = build(tmp_path, rows, targets, name="a")
    for row in rows:
        row["past"] = np.full(len(row["past"]), -1e10)
        row["static"] = "OTHER SECRET"
    _, b = build(tmp_path, rows, targets, name="b")
    assert files(a) == files(b)


def test_adaptation_missing_values_encodings_and_file_hashes(tmp_path):
    import hashlib

    rows, targets = task_rows(lengths=(24,), starts=(0,))
    rows[0]["y0"][0] = np.nan
    rows[0]["event"][16:] = ["inner-only"] * 4 + ["outer-secret"] * 4
    manifest, root = build(tmp_path, rows, targets)
    table = pq.read_table(root / "univariate/train-00000.parquet").to_pylist()
    assert np.isnan(table[0]["target"][0][0])
    assert inner(root, 0)["items"][0]["score_past"][0][0] is None
    encodings = manifest["covariates"]["training_encodings"]
    assert encodings[0]["columns"]["event"]["categories"] == ["a", "b"]
    assert manifest["counts"]["training_observed_target_values"] == 15
    for name, meta in manifest["files"].items():
        contents = (root / name).read_bytes()
        assert meta == {"sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents)}
    fingerprint = manifest.pop("fingerprint")
    encoded = (
        json.dumps(
            manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == fingerprint
    assert b"outer-secret" not in b"".join(files(root).values())


@pytest.mark.parametrize(
    "kwargs", [{"max_context": 0}, {"max_context": True}, {"phase": "unknown"}]
)
def test_adaptation_rejects_invalid_export_settings(tmp_path, kwargs):
    rows, targets = task_rows()
    task = SyntheticTask(rows, targets)
    yaml = tmp_path / "tasks.yaml"
    yaml.write_text("tasks: []\n")
    with pytest.raises(ValueError):
        build_manifest(task, task_yaml=yaml, output=tmp_path / "out", **kwargs)
    assert not (tmp_path / "out").exists()


def test_adaptation_fingerprint_identifies_precise_training_input_before_float32_cast(tmp_path):
    rows, targets = task_rows(lengths=(24,), starts=(0,))
    first, a = build(tmp_path, rows, targets, name="a", max_context=4)
    rows[0]["y0"][0] = 1e-50  # changes float64 input, rounds back to float32 zero
    second, b = build(tmp_path, rows, targets, name="b", max_context=4)
    assert (a / "univariate/train-00000.parquet").read_bytes() == (
        b / "univariate/train-00000.parquet"
    ).read_bytes()
    assert first["fingerprint"] != second["fingerprint"]


@pytest.mark.parametrize("column", ["y0", "promo"])
def test_adaptation_numpy_formatter_preserves_precise_training_fingerprint(tmp_path, column):
    rows, targets = task_rows(lengths=(24,), starts=(0,))
    rows[0][column] = [1.0000000001] * 24
    task = SyntheticTask(rows, targets, max_context=4)
    task.load_full_dataset = lambda: datasets.Dataset.from_list(rows).with_format("numpy")
    yaml = tmp_path / "tasks.yaml"
    yaml.write_text("tasks: []\n")
    first = build_manifest(task, task_yaml=yaml, output=tmp_path / "first")
    rows[0][column][0] += 1e-10  # Raw input changes; float32 payloads stay identical.
    second = build_manifest(task, task_yaml=yaml, output=tmp_path / "second")
    assert first["items"][0]["training_input_sha256"] != second["items"][0]["training_input_sha256"]
    assert first["fingerprint"] != second["fingerprint"]
    for name in first["files"]:
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
