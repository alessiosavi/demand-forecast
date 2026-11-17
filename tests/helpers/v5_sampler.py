"""Small deterministic records for batch-stream contract tests."""

from dataclasses import fields, replace

import numpy as np
import torch

from demand_forecast.data.episode import Item, Panel
from demand_forecast.models.v5.config import V5Config
from demand_forecast.pretrain.v5.config import (
    OptimConfig,
    SamplerConfig,
    SourceConfig,
    V5TrainConfig,
)
from demand_forecast.pretrain.v5.sources import SourceMix


def config(**sampler):
    return V5TrainConfig(
        model=V5Config(),
        sources=(SourceConfig("typed", 1, recipes=((),), max_items=4, length=(64, 96)),),
        sampler=replace(
            SamplerConfig(
                episodes_per_batch=4,
                episodes_per_panel=4,
                contexts=(64,),
                horizon=(8, 16),
                origins=2,
                max_cells=None,
            ),
            **sampler,
        ),
        optim=OptimConfig(max_steps=1),
        seed=12,
    )


def panel(length=96, count=4, targets=1):
    return Panel(
        "fixture",
        tuple(
            Item(
                str(i),
                0,
                {f"y{j}": np.arange(length, dtype=np.float64) + i + j for j in range(targets)},
                {},
                {},
            )
            for i in range(count)
        ),
        {},
        None,
        None,
        True,
    )


class FixedSource:
    def __init__(self, value, record=None):
        self.value = value
        self.record = record or {"source": "typed"}

    def panel(self, rng):
        return self.value, self.record


def fixed_mix(c, source):
    """Inject a deterministic panel source through the production mixture API."""
    mixture = SourceMix(c.sources)
    source.config = c.sources[0]
    mixture.sources = (source,)
    return mixture


def equal_batches(left, right):
    for field in fields(left):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), field.name
        else:
            assert a == b, field.name
