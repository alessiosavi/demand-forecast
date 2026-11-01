"""Training outcomes, terminal failures and exact continuation on tiny shards."""

import json

import pytest
import torch

pyarrow = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.bench


def test_pretrain_runs_saves_bundle_and_history(tmp_path, tiny_pretrain_cfg):
    from demand_forecast.core.bundle import load_bundle
    from demand_forecast.pretrain.train import pretrain

    out = tmp_path / "out"
    res = pretrain(tiny_pretrain_cfg, out)
    assert res.steps == 12 and res.stop_reason == "max_steps" and len(res.history) == 12
    assert "val_sql" in res.history[5] and "val_sql" not in res.history[4]
    assert "val_sql" in res.history[11]
    assert [h["step"] for h in res.history] == list(range(1, 13))
    assert [h["kind"] for h in res.history].count("kernel_synth") == 1
    bundle = load_bundle(out / "bundle")
    assert bundle.manifest["pretrain_identity"] == res.identity
    assert bundle.model.spec.enc_channels == 10 and bundle.model.spec.dec_channels == 8
    assert bundle.model.spec.vocab_sizes == {"__series__": 2}
    assert bundle.dataset_cfg is None and bundle.manifest["external_diagnostic"] is None
    assert json.loads((out / "pretrain.json").read_text())["steps"] == 12
    assert len((out / "history.csv").read_text().splitlines()) == 13
    checkpoint = torch.load(out / "checkpoints/latest.pt", weights_only=False)
    assert checkpoint["step"] == checkpoint["sampler_state"]["step"] == 12
    assert all(value.device.type == "cpu" for value in checkpoint["best_state"].values())
    assert all(
        torch.equal(value, checkpoint["best_state"][key])
        for key, value in bundle.model.state_dict().items()
    )


def test_interrupt_checkpoints_at_stop_step_and_resume_matches(tmp_path, tiny_pretrain_cfg):
    from demand_forecast.core.bundle import load_bundle
    from demand_forecast.core.training import InterruptedRun
    from demand_forecast.pretrain.train import pretrain

    full = pretrain(tiny_pretrain_cfg, tmp_path / "a")
    with pytest.raises(InterruptedRun):
        pretrain(tiny_pretrain_cfg, tmp_path / "b", stop_after=5)
    ck = torch.load(tmp_path / "b/checkpoints/latest.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 5 and ck["sampler_state"]["step"] == 5
    assert not (tmp_path / "b/bundle").exists()
    # A second interruption counts from this call, not from the original run.
    with pytest.raises(InterruptedRun):
        pretrain(tiny_pretrain_cfg, tmp_path / "b", resume=True, stop_after=2)
    assert torch.load(tmp_path / "b/checkpoints/latest.pt", weights_only=False)["step"] == 7
    changed = tiny_pretrain_cfg.model_copy(deep=True)
    changed.seed += 1
    with pytest.raises(ValueError, match=f"identity.*{full.identity}"):
        pretrain(changed, tmp_path / "b", resume=True)
    part = pretrain(tiny_pretrain_cfg, tmp_path / "b", resume=True)
    assert part.history == full.history
    assert (part.best_step, part.best_score) == (full.best_step, full.best_score)
    a = load_bundle(tmp_path / "a/bundle").model.state_dict()
    b = load_bundle(tmp_path / "b/bundle").model.state_dict()
    assert all(torch.equal(value, b[key]) for key, value in a.items())


def test_nonfinite_stops_with_checkpoint(tmp_path, nonfinite_cfg, monkeypatch):
    from demand_forecast.core.training import NONFINITE_STOP_AFTER, NonFiniteTraining
    from demand_forecast.pretrain.train import pretrain

    monkeypatch.setattr(
        "demand_forecast.pretrain.train.pinball_loss",
        lambda *a, **k: (torch.tensor(float("nan"), requires_grad=True), 1),
    )
    with pytest.warns(UserWarning, match="lr_scheduler.step"), pytest.raises(NonFiniteTraining):
        pretrain(nonfinite_cfg, tmp_path / "c")
    progress = json.loads((tmp_path / "c/checkpoints/progress.json").read_text())
    assert progress["stop_reason"] == "nonfinite_losses"
    assert progress["step"] == progress["timings"]["nonfinite_losses"] == NONFINITE_STOP_AFTER
    assert not (tmp_path / "c/bundle").exists()
    with pytest.raises(NonFiniteTraining):
        pretrain(nonfinite_cfg, tmp_path / "c", resume=True)
    assert torch.load(tmp_path / "c/checkpoints/latest.pt", weights_only=False)["step"] == 20
