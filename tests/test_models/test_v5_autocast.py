"""Mixed precision on an accelerator: dtype errors that a processor's autocast does not show.

Accelerator autocast (CUDA, MPS) changes the dtype of single operations, softmax among them; the same
code under a processor's autocast keeps one dtype and passes. These cases run one synthetic batch
through gates, forward, loss and backward on whatever accelerator the machine has, and are skipped on
hosted CI runners, which expose none that is usable.
"""

import os
from dataclasses import fields, replace

import pytest
import torch

from demand_forecast.data.episode import EpisodeBuilder, EpisodeRequest, collate
from demand_forecast.models.v5 import V5Model, v5_loss
from demand_forecast.pretrain.v5.config import (
    T1_MODEL,
    OptimConfig,
    SamplerConfig,
    SourceConfig,
    V5TrainConfig,
)
from demand_forecast.pretrain.v5.sampler import make_batch
from tests.helpers.episodes import toy_panel
from tests.helpers.v5_loop import config

CONFIG = V5TrainConfig(
    model=T1_MODEL,
    sources=(SourceConfig("typed", 1.0, max_items=12, length=(128, 400)),),
    sampler=SamplerConfig(
        episodes_per_batch=8,
        episodes_per_panel=4,
        origins=2,
        contexts=(64, 128),
        max_cells=None,
        designate_target=True,
    ),
    optim=OptimConfig(max_steps=1),
)


def accelerator():
    if os.environ.get("CI"):
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("router", ["learned", "uniform", "rolled"])
@pytest.mark.parametrize("gates", ["learned", "ones"])
def test_training_step_under_accelerator_autocast(dtype, router, gates):
    device = accelerator()
    if device is None:
        pytest.skip("no usable accelerator on this machine")
    batch, _ = make_batch(CONFIG, 0)
    if device.type == "mps":  # MPS has no float64
        batch = replace(
            batch,
            **{
                f.name: getattr(batch, f.name).float()
                for f in fields(batch)
                if isinstance(getattr(batch, f.name), torch.Tensor)
                and getattr(batch, f.name).dtype == torch.float64
            },
        )
    batch = batch.to(device)
    torch.manual_seed(0)
    model = V5Model(CONFIG.model).to(device).train()
    with torch.autocast(device.type, dtype=dtype):
        output = model(batch, model.gates(batch, mode=gates), router=router)
        loss, _ = v5_loss(output, batch, CONFIG.model)
    loss.backward()
    assert torch.isfinite(loss.detach().float())


def test_evidence_mode_under_autocast():
    panel = toy_panel(n_items=2, length=48)
    batch = collate([EpisodeBuilder(panel).build(EpisodeRequest(0, (40,), 3, 32))])
    model = V5Model(config().model).eval()

    with torch.no_grad():
        expected = model.gates(batch, mode="evidence")
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model.gates(batch, mode="evidence")
            output = model(batch, actual)

    for field in fields(expected):
        torch.testing.assert_close(
            getattr(actual, field.name), getattr(expected, field.name), atol=0, rtol=0
        )
    assert torch.isfinite(output.quantiles).all()


def test_dense_mode_under_autocast():
    panel = toy_panel(n_items=2, length=48)
    batch = collate([EpisodeBuilder(panel).build(EpisodeRequest(0, (40,), 3, 32))])
    cfg = replace(config().model, expert_enc_layers=1, expert_dec_layers=1, expert_mode="dense")
    model = V5Model(cfg).eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch, router="learned")
        loss, _ = v5_loss(output, batch, cfg)
    loss.backward()
    assert output.quantiles.dtype == torch.bfloat16
    assert output.router is None
    assert output.aux["balance"].count_nonzero().item() == 0
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
