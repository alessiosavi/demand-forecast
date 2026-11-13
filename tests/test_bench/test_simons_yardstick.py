"""Offline Simons contracts: processing, input roles, quantiles and resume isolation."""

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.yardsticks import (  # noqa: E402
    Options,
    run_yardstick,
    runner,  # noqa: E402
)
from tests.helpers.fev_fixture import fixture_task, write_fev_fixture  # noqa: E402

PIN = "8e338e915d26c8960713889ef12ab1cf5c08d591"
RULES = ("views", "sign_clip", "integer_snap", "fit_row", "xreg", "xreg_guard")


@dataclass(frozen=True)
class ExportConfig:
    views: bool = True
    sign_clip: bool = True
    integer_snap: bool = True
    fit_row: bool = True
    xreg: bool = True
    xreg_guard: bool = True
    covariates: bool = True
    binary_rows: bool = True
    read_after: int = 5
    context_length: int = 64


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture
def exported(tmp_path):
    path = tmp_path / "export"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(asdict(ExportConfig())))
    # Only the fake loader consumes this file; never load or download real weights.
    (path / "model.safetensors").write_bytes(b"synthetic checkpoint A")
    return path


@pytest.fixture
def fake_simons(monkeypatch):
    loaded, calls = [], []

    class FakeSimons:
        @classmethod
        def from_pretrained(cls, name_or_path, device="cpu", revision=None):
            model = cls()
            model.config = ExportConfig(
                **json.loads((Path(name_or_path) / "config.json").read_text())
            )
            loaded.append((name_or_path, device, model, model.config))
            return model

    def forecast_series(
        model,
        series,
        horizon,
        batch_size=1024,
        future_covariates=None,
        past_covariates=None,
        shared_categories=True,
        **options,
    ):
        calls.append(
            (
                model,
                series,
                horizon,
                dict(
                    batch_size=batch_size,
                    future_covariates=future_covariates,
                    past_covariates=past_covariates,
                    shared_categories=shared_categories,
                    **options,
                ),
            )
        )
        # Deliberately unordered quantiles test the upstream sort before interpolation.
        return [
            s[:, -1:, None]
            + np.arange(9, 0, -1)[None, None, :]
            + np.zeros((s.shape[0], horizon, 9))
            for s in series
        ]

    monkeypatch.setitem(
        sys.modules, "simons", SimpleNamespace(Simons=FakeSimons, forecast_series=forecast_series)
    )
    return SimpleNamespace(loaded=loaded, calls=calls)


def simons_options(exported, **kwargs):
    return Options(
        "simons", str(exported), "simons-test", processing="raw", device="cpu", **kwargs
    ).resolved()


@pytest.fixture
def task(tmp_path):
    import fev
    from datasets import Dataset, Sequence, Value

    path = write_fev_fixture(
        tmp_path / "data", n_series=2, known=("promo", "event"), text_dynamic=("event",)
    )
    data = Dataset.from_parquet(str(path))
    features = data.features.copy()
    features["other"] = Sequence(Value("float32"))
    data.map(lambda r: {"other": [2 * v for v in r["target"]]}, features=features).to_parquet(path)
    base = fixture_task(path, known=("promo", "event"))
    return fev.Task(**{**base.to_dict(), "target": ["target", "other"]})


@pytest.mark.parametrize("processing", ["raw", "published"])
def test_exported_rules_raw_overrides_and_published_untouched(
    exported, fake_simons, task, processing
):
    from demand_forecast.bench.yardsticks.adapters import SimonsAdapter

    options = replace(simons_options(exported), processing=processing)
    adapter = SimonsAdapter(options, task)
    adapter.predict(runner.prepare_windows(task, options), 4, [0.05, 0.15, 0.5, 0.95])
    path, device, model, original = fake_simons.loaded[0]
    assert path == str(exported) and device == "cpu"
    assert all(getattr(model.config, rule) is (processing == "published") for rule in RULES)
    assert model.config.covariates and model.config.binary_rows and model.config.read_after == 5
    assert model.config.context_length == 64
    if processing == "published":
        assert model.config is original
    assert all(call[3]["shared_categories"] is False for call in fake_simons.calls)


def test_past_only_dropped_counted_shape_and_median(exported, fake_simons, task, tmp_path):
    options = simons_options(exported, max_context=5)
    out = tmp_path / "out"
    result = run_yardstick([task], options, out)
    assert len(fake_simons.calls) == 2
    for _, series, horizon, kwargs in fake_simons.calls:
        assert all(s.shape == (2, 5) for s in series)
        assert kwargs["past_covariates"] is None
        assert len(kwargs["future_covariates"]) == 2
        for covs in kwargs["future_covariates"]:
            assert set(covs) == {"promo", "event"}
            assert all(a.shape == (5 + horizon,) for a in covs.values())
    assert result.dropped_past_only_covariates.iloc[0] == 4  # 2 items x 2 windows x oil
    record = json.loads((out / task.task_name / "run.json").read_text())
    assert record["input_metadata"]["dropped_past_only_covariates"] == 4
    with np.load(out / task.task_name / "window_0.npz") as saved:
        assert saved["arrays"].shape == (2, 2, 4, 9)
        levels = saved["quantiles"].tolist()
        predictions = runner.fev_predictions(saved["arrays"], levels, task.target_columns)
        past, _ = task.get_window(0).get_input_data()
        np.testing.assert_allclose(
            predictions["target"][0]["predictions"], past[0]["target"][-1] + 5
        )
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    run_yardstick([task], options, out, resume=True)
    assert len(fake_simons.calls) == 2
    assert before == {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_large_target_groups_keep_item_order_and_quantile_interpolation(
    exported, fake_simons, task
):
    from demand_forecast.bench.yardsticks.adapters import SimonsAdapter

    options = simons_options(exported, batch_size=2)
    items = [
        runner.SeriesInput(
            i, tuple(range(35)), np.repeat((np.arange(35) + 100 * i)[:, None], 6, axis=1), {}, {}
        )
        for i in range(2)
    ]
    window = runner.WindowInput(["a", "b"], [f"t{i}" for i in range(35)], items)
    outputs = SimonsAdapter(options, task).predict([window], 4, [0.05, 0.15, 0.5, 0.95])
    series = [s for call in fake_simons.calls for s in call[1]]
    assert [s.shape[0] for s in series] == [32, 3, 32, 3]
    assert all(call[3]["batch_size"] == 2 for call in fake_simons.calls)
    arrays = runner.assemble(window, outputs[0], 4, [0.05, 0.15, 0.5, 0.95])
    assert arrays.shape == (2, 35, 4, 4)
    np.testing.assert_allclose(arrays[0, 31, 0], [32, 32.5, 36, 40])
    np.testing.assert_allclose(arrays[1, 34, 0], [135, 135.5, 139, 143])


def test_covariate_ablation_removes_all_inputs(exported, fake_simons, task, tmp_path):
    options = simons_options(exported, covariates="none", max_context=3)
    frame = run_yardstick([task], options, tmp_path / "none")
    assert frame.dropped_past_only_covariates.iloc[0] == 0
    for _, series, _, kwargs in fake_simons.calls:
        assert all(s.shape == (2, 3) for s in series)
        assert kwargs["future_covariates"] == [None, None]
        assert kwargs["past_covariates"] is None


@pytest.mark.parametrize("change", ["processing", "weights", "config", "checkpoint"])
def test_resume_rejects_mode_or_checkpoint_change(exported, fake_simons, task, tmp_path, change):
    import shutil

    options, out = simons_options(exported), tmp_path / "out"
    run_yardstick([task], options, out)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    if change == "processing":
        options = replace(options, processing="published")
    elif change == "checkpoint":
        swapped = tmp_path / "swapped"
        shutil.copytree(exported, swapped)
        (swapped / "model.safetensors").write_bytes(b"checkpoint B")
        options = replace(options, weights=str(swapped))
    else:
        path = exported / ("config.json" if change == "config" else "model.safetensors")
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="identity mismatch"):
        run_yardstick([task], options, out, resume=True)
    assert len(fake_simons.calls) == 2
    assert before == {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_identity_records_pin_and_hashes_only_consumed_files(exported, fake_simons):
    options = runner.resolve_weights(simons_options(exported))
    identity = runner.runtime_identity(options)
    assert identity["simons_source"] == PIN
    assert identity["options"]["processing"] == "raw"
    assert identity["options"]["weights_revision"].startswith("local-sha256:")
    (exported / "training.log").write_text("not consumed by from_pretrained")
    assert (
        runner.resolve_weights(simons_options(exported)).weights_revision
        == options.weights_revision
    )


@pytest.mark.parametrize(
    "model,processing",
    [("simons", None), ("simons", "invalid"), ("chronos2", "raw"), ("timestral", "published")],
)
def test_processing_validation(model, processing):
    with pytest.raises(ValueError, match="processing"):
        Options(model, "unused", "test", processing=processing).resolved()


@pytest.mark.parametrize("missing", ["directory", "config.json", "model.safetensors"])
def test_simons_export_must_be_complete_and_local(exported, missing):
    options = simons_options(exported)
    if missing == "directory":
        options = replace(options, weights="nonexistent-hub/repo")
    else:
        (exported / missing).unlink()
    with pytest.raises(ValueError, match="(local|config.json|model.safetensors)"):
        runner.resolve_weights(options)


def test_resume_rejects_edited_upstream_at_same_package_version(
    exported, fake_simons, task, tmp_path, monkeypatch
):
    from importlib.machinery import ModuleSpec

    source = tmp_path / "installed-simons"
    source.mkdir()
    (source / "__init__.py").write_text("# synthetic source identity\n")
    forecast = source / "forecast.py"
    forecast.write_text("# implementation A\n")
    monkeypatch.setattr(
        sys.modules["simons"],
        "__spec__",
        ModuleSpec("simons", loader=None, origin=str(source / "__init__.py")),
        raising=False,
    )
    original_version = runner.version
    monkeypatch.setattr(
        runner, "version", lambda name: "0.1.1" if name == "simons" else original_version(name)
    )
    options, out = simons_options(exported), tmp_path / "out"
    run_yardstick([task], options, out)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    forecast.write_text("# implementation B, unchanged version\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        run_yardstick([task], options, out, resume=True)
    assert len(fake_simons.calls) == 2
    assert before == {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_simons_records_installed_vcs_commit(exported, fake_simons, monkeypatch):
    direct_url = json.dumps({"vcs_info": {"commit_id": "b" * 40}})
    monkeypatch.setattr(
        runner,
        "distribution",
        lambda _: SimpleNamespace(read_text=lambda _: direct_url),
        raising=False,
    )
    identity = runner.runtime_identity(runner.resolve_weights(simons_options(exported)))
    assert identity["simons_implementation"]["vcs_commit"] == "b" * 40


def test_optional_import_explains_pinned_install(exported, task, monkeypatch):
    from demand_forecast.bench.yardsticks.adapters import SimonsAdapter

    monkeypatch.setitem(sys.modules, "simons", None)
    with pytest.raises(ImportError, match=PIN):
        SimonsAdapter(simons_options(exported), task)


@pytest.mark.parametrize("model", ["chronos2", "timestral"])
def test_stage1_legacy_manifest_resumes_without_any_changes(
    exported, task, tmp_path, monkeypatch, model
):
    from tests.test_bench.test_yardsticks import FakeAdapter

    options = Options(model, str(exported), "stage1", device="cpu")
    current_identity = runner.runtime_identity

    def legacy_identity(resolved):
        # Frozen stage-1 schema, independent of new dataclass fields.
        fields = (
            "model",
            "weights",
            "model_name",
            "revision",
            "max_context",
            "covariates",
            "cross_series",
            "batch_size",
            "device",
            "weights_revision",
        )
        package_names = (
            "fev",
            "datasets",
            "numpy",
            "pandas",
            "pyarrow",
            "torch",
            "chronos-forecasting" if model == "chronos2" else "timestral",
        )
        packages = {}
        for name in package_names:
            try:
                packages[name] = runner.version(name)
            except runner.PackageNotFoundError:
                packages[name] = "not-installed"
        return {
            "options": {key: getattr(resolved, key) for key in fields},
            "packages": packages,
            "protocol_version": 1,
            "published_adapter_source": "9cf943c388e5cfaf352175ea3c69e408b197d997",
        }

    monkeypatch.setattr(runner, "runtime_identity", legacy_identity)
    out = tmp_path / "stage1"
    frame = run_yardstick([task], options, out, adapter_factory=lambda _: FakeAdapter())
    monkeypatch.setattr(runner, "runtime_identity", current_identity)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    after = run_yardstick(
        [task],
        options,
        out,
        resume=True,
        adapter_factory=lambda _: pytest.fail("stage-1 rows did not resume"),
    )
    pd.testing.assert_frame_equal(frame, after)
    assert before == {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_no_target_after_cutoff_or_covariate_beyond_horizon_reaches_simons(
    exported, fake_simons, tmp_path
):
    import fev
    from datasets import Dataset

    from demand_forecast.bench.yardsticks.adapters import SimonsAdapter

    path = write_fev_fixture(tmp_path / "first", n_series=2)
    base = fixture_task(path, num_windows=1)
    task = fev.Task(**{**base.to_dict(), "initial_cutoff": -12})
    options = simons_options(exported, max_context=5)
    first = SimonsAdapter(options, task).predict(runner.prepare_windows(task, options), 4, [0.5])
    original_calls = list(fake_simons.calls)
    dataset = Dataset.from_parquet(str(path))
    mutated = dataset.map(
        lambda row: {
            "target": row["target"][:28] + [99999.0] * 12,
            "promo": row["promo"][:32] + [111] * 8,
            "oil": [99999.0] * 40,
        }
    )
    changed_path = tmp_path / "changed.parquet"
    mutated.to_parquet(changed_path)
    changed = fev.Task(**{**task.to_dict(), "dataset_path": str(changed_path)})
    fake_simons.calls.clear()
    second = SimonsAdapter(options, changed).predict(
        runner.prepare_windows(changed, options), 4, [0.5]
    )
    for a, b in zip(first[0], second[0], strict=True):
        np.testing.assert_array_equal(a, b)
    for original, actual in zip(original_calls, fake_simons.calls, strict=True):
        for i, series in enumerate(actual[1]):
            np.testing.assert_array_equal(series, original[1][i])
            expected = np.asarray(dataset[i]["target"])[None, 23:28]
            np.testing.assert_allclose(series, expected)
            covs = actual[3]["future_covariates"][i]
            assert set(covs) == {"promo"}
            np.testing.assert_array_equal(covs["promo"], dataset[i]["promo"][23:32])
