"""Tiny real-model fixtures and the SIGTERM child for P4's loop tests."""

import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from demand_forecast.models.v5 import V5Config
from demand_forecast.pretrain.v5.config import (
    OptimConfig,
    SamplerConfig,
    SourceConfig,
    V5TrainConfig,
)


def config(*, ema=None, workers=0, steps=12, **kwargs):
    return V5TrainConfig(
        model=V5Config(
            d_model=16,
            nhead=2,
            enc_layers=1,
            dec_layers=1,
            dim_feedforward=32,
            n_experts=1,
            top_k=1,
            expert_enc_layers=0,
            expert_dec_layers=0,
            d_calendar=4,
            d_step=4,
            d_out=4,
            dropout=0.1,
        ),
        sources=(SourceConfig("typed", 1, recipes=((),), max_items=1, length=(64, 64)),),
        sampler=SamplerConfig(
            episodes_per_batch=1,
            episodes_per_panel=1,
            origins=1,
            contexts=(32,),
            horizon=(8, 8),
            drop=(),
            max_cells=None,
        ),
        optim=OptimConfig(
            max_steps=steps,
            learning_rate=0.005,
            warmup_steps=1,
            amp=False,
            ema_decay=ema,
            log_every=3,
        ),
        workers=workers,
        seed=17,
        device="cpu",
        **kwargs,
    )


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


def training_equal(left, right):
    for name in ("model", "ema", "optimizer", "scheduler", "scaler", "history", "pending"):
        equal(left[name], right[name])
    # Physical saves depend on interruptions and wall clock, unlike training counters.
    equal(
        {k: v for k, v in left["counters"].items() if k != "saves"},
        {k: v for k, v in right["counters"].items() if k != "saves"},
    )


def terminate_child(directory):
    from demand_forecast.pretrain.v5 import loop

    torch.set_num_threads(1)
    original = loop.v5_loss

    def waiting_loss(*args):
        (directory / "ready").write_text("ready")
        deadline = time.monotonic() + 15
        while not (directory / "go").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("parent did not release update")
            time.sleep(0.01)
        return original(*args)

    loop.v5_loss = waiting_loss
    result = loop.train(replace(config(), device="cpu"), directory / "state")
    (directory / "result.json").write_text(
        json.dumps({"reason": result.stop_reason, "batch": result.batch})
    )


if __name__ == "__main__":
    import sys

    terminate_child(Path(sys.argv[1]))
