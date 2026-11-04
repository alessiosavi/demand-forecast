import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.world_signals.run_tools import finalize


def setup_run(tmp_path):
    task = tmp_path / "probe_2022_on"
    (task / "checkpoints").mkdir(parents=True)
    state = {
        "history": [{"step": 0, "SQL": 1.0}],
        "best_step": 0,
        "best_score": 1.0,
        "max_steps": 500,
        "run_identity": "test",
    }
    torch.save(state, task / "checkpoints/latest.pt")
    (task / "checkpoints/progress.json").write_text("{}")
    report = {
        "run_identity": "test",
        "task_definition": {"num_windows": 2},
        "fit_result": {"best_step": 0, "best_score": 1.0},
        "resolved_budget": {"max_steps": 500},
    }
    (task / "run.json").write_text(json.dumps(report))
    return task, state


def test_finalize_refuses_incomplete_and_extracts_history(tmp_path):
    task, state = setup_run(tmp_path)
    rows = finalize(tmp_path)
    assert rows[0]["status"] == "incomplete"
    assert (task / "checkpoints/latest.pt").exists()
    history = json.loads((task / "history.json").read_text())
    assert history["history"] == state["history"]
    pd.DataFrame([{"task": task.name, "model_name": "fake", "run_identity": "test"}]).to_csv(
        tmp_path / "summaries.csv", index=False
    )
    (task / "bundle").mkdir()
    (task / "bundle/manifest.json").write_text('{"run_identity": "test"}')
    for i in range(2):
        np.savez(
            task / f"window_{i}.npz", ids=["BTC"], arrays=np.zeros((1, 13, 9)), identity="test"
        )
    assert finalize(tmp_path, dry_run=True)[0]["status"] == "would-prune"
    assert (task / "checkpoints/latest.pt").exists()
    assert finalize(tmp_path)[0]["status"] == "complete"
    assert not (task / "checkpoints/latest.pt").exists()
    assert (task / "checkpoints/progress.json").exists()
    assert finalize(tmp_path)[0]["status"] == "complete"


def test_finalize_dry_run_never_writes_and_rejects_identity(tmp_path):
    task, state = setup_run(tmp_path)
    assert finalize(tmp_path, dry_run=True)[0]["status"] == "incomplete"
    assert not (task / "history.json").exists()
    state["run_identity"] = "wrong"
    torch.save(state, task / "checkpoints/latest.pt")
    assert finalize(tmp_path)[0]["status"] == "inconsistent"
    assert (task / "checkpoints/latest.pt").exists()
    assert not (task / "history.json").exists()


def test_finalize_preserves_checkpoints_when_a_window_is_missing(tmp_path):
    from scripts.world_signals.run_tools import main

    task, _ = setup_run(tmp_path)
    pd.DataFrame([{"task": task.name, "model_name": "fake", "run_identity": "test"}]).to_csv(
        tmp_path / "summaries.csv", index=False
    )
    (task / "bundle").mkdir()
    (task / "bundle/manifest.json").write_text('{"run_identity": "test"}')
    np.savez(task / "window_0.npz", ids=["BTC"], arrays=np.zeros((1, 13, 9)), identity="test")
    with pytest.raises(SystemExit) as exc:
        main(["finalize", "--run-dir", str(tmp_path)])
    assert exc.value.code == 1
    assert (task / "history.json").exists()
    assert (task / "checkpoints/latest.pt").exists()
    assert "window set" in finalize(tmp_path)[0]["detail"]
