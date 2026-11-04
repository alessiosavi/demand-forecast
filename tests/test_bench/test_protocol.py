"""Per-window protocol checks on the two-window offline fev fixture."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.run import run_fev  # noqa: E402
from demand_forecast.core import run  # noqa: E402
from demand_forecast.core.training import InterruptedRun, Trainer  # noqa: E402
from demand_forecast.inference.predictor import Predictor  # noqa: E402
from tests.helpers.fev_fixture import write_fev_fixture  # noqa: E402
from tests.helpers.fev_spec import fixture_spec  # noqa: E402
from tests.helpers.settings import tiny_settings  # noqa: E402


def _states(directory):
    return {
        p.relative_to(directory): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in directory.rglob("*")
        if p.is_file()
    }


def test_per_window_owns_bundles_identities_timings_and_skips_completed_windows(
    tmp_path, monkeypatch
):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=4, d_model=8)
    settings.training.refit = "full"
    settings.forecast.scale_floor_frac = 0.9
    out = tmp_path / "out"
    summary = run_fev([spec], settings, out, policy="per_window")
    task_dir = out / spec.name
    identities, timings = [], []
    floored = 0
    for index in range(2):
        directory = task_dir / f"window_{index}"
        report = json.loads((directory / "run.json").read_text())
        manifest = json.loads((directory / "bundle/manifest.json").read_text())
        identities.append(report["run_identity"])
        assert report["policy"] == "per_window" and report["refit"]["performed"]
        assert report["run_identity"] == manifest["run_identity"]
        timings.append(
            sum(report["timings"]["cumulative"][k] for k in ("train_s", "eval_s", "checkpoint_s"))
        )
        with np.load(task_dir / f"window_{index}.npz", allow_pickle=False) as saved:
            assert saved["identity"].item() == identities[-1]
            assert saved["floored_target_samples"].dtype == np.int64
            floored += saved["floored_target_samples"].item()
    assert identities[0] != identities[1]
    assert (
        summary.run_identity.iloc[0]
        == hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
    )
    assert summary.policy.iloc[0] == "per_window"
    assert summary.training_time_s.iloc[0] == sum(timings)
    assert summary.floored_target_samples.iloc[0] == floored > 0
    before = _states(out)

    def no_training(*args, **kwargs):
        pytest.fail("completed windows must not build or train a model")

    monkeypatch.setattr(run, "build_model", no_training)
    monkeypatch.setattr(Trainer, "fit", no_training)
    monkeypatch.setattr(Predictor, "forecast_with_diagnostics", no_training)
    resumed = run_fev([spec], settings, out, policy="per_window", resume=True)
    pd.testing.assert_frame_equal(summary, resumed)
    assert before == _states(out)


def test_per_window_interrupted_second_window_resumes_exactly(tmp_path, monkeypatch):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=4, d_model=8)
    whole, parts = tmp_path / "whole", tmp_path / "parts"
    run_fev([spec], settings, whole, policy="per_window")
    train = run.train_panel

    def interrupt_second(panel, origin, settings, directory, **kwargs):
        if directory.name == "window_1":
            kwargs["stop_after"] = 2
        return train(panel, origin, settings, directory, **kwargs)

    monkeypatch.setattr(run, "train_panel", interrupt_second)
    with pytest.raises(InterruptedRun, match="step 2"):
        run_fev([spec], settings, parts, policy="per_window")
    task_dir = parts / spec.name
    assert (task_dir / "window_0.npz").exists()
    assert not (task_dir / "window_1.npz").exists()
    assert not (parts / "summaries.csv").exists()
    first_before = _states(task_dir / "window_0")
    first_predictions = (task_dir / "window_0.npz").read_bytes()
    monkeypatch.setattr(run, "train_panel", train)
    run_fev([spec], settings, parts, policy="per_window", resume=True)
    assert first_before == _states(task_dir / "window_0")
    assert first_predictions == (task_dir / "window_0.npz").read_bytes()
    report = json.loads((task_dir / "window_1/run.json").read_text())
    assert report["resumed_from_step"] == 2
    for index in range(2):
        with (
            np.load(whole / spec.name / f"window_{index}.npz") as expected,
            np.load(task_dir / f"window_{index}.npz") as actual,
        ):
            np.testing.assert_array_equal(actual["arrays"], expected["arrays"])
            np.testing.assert_array_equal(actual["identity"], expected["identity"])


@pytest.mark.parametrize(
    "artifact", ["run.json", "bundle/manifest.json", "checkpoints/latest.pt", "prediction"]
)
def test_per_window_resume_rejects_mismatched_window_artifact(tmp_path, artifact):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=2, d_model=8)
    out = tmp_path / "out"
    run_fev([spec], settings, out, policy="per_window")
    task_dir = out / spec.name
    path = task_dir / "window_1" / artifact
    if artifact == "prediction":
        path = task_dir / "window_1.npz"
        with np.load(path) as saved:
            payload = dict(saved)
        np.savez(path, **{**payload, "identity": "wrong"})
    elif path.suffix == ".pt":
        payload = torch.load(path, weights_only=False)
        payload["run_identity"] = "wrong"
        torch.save(payload, path)
    else:
        payload = json.loads(path.read_text())
        payload["run_identity"] = "wrong"
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="run identity mismatch"):
        run_fev([spec], settings, out, policy="per_window", resume=True)


def test_first_window_legacy_summary_without_policy_still_loads(tmp_path):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=2, d_model=8)
    out = tmp_path / "out"
    original = run_fev([spec], settings, out)
    original.drop(columns=["policy"]).to_csv(out / "summaries.csv", index=False)
    resumed = run_fev([spec], settings, out, resume=True)
    assert resumed.policy.iloc[0] == "first_window"
    assert resumed.run_identity.iloc[0] == original.run_identity.iloc[0]
    assert (out / spec.name / "bundle/manifest.json").exists()
    assert not (out / spec.name / "window_0").exists()
