"""The shared skeleton learns from normalised labels with a tiny CPU backbone."""

import numpy as np
import torch

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.sampler import WindowSampler, build_origins, make_batch
from demand_forecast.models.base import ForecastModel, ModelSpec
from demand_forecast.models.losses import pinball_loss
from tests.helpers.panels import synthetic_panel
from tests.helpers.toy_backbone import ToyBackbone


def test_toy_model_reduces_pinball_loss():
    panel = synthetic_panel(n_series=20, length=60, freq="W-SUN", seed=1, signal="seasonal_promo")
    enc = StaticEncoder().fit(panel, [s.series_id for s in panel.series])
    origins = build_origins(panel, 4, 8)
    sampler = WindowSampler(panel, origins, 4, 32, enc, seed=0)
    eval_batches = [
        make_batch(panel, [(i, 56) for i in range(0, 10)], 4, 32, enc),
        make_batch(panel, [(i, 56) for i in range(10, 20)], 4, 32, enc),
    ]
    torch.manual_seed(0)
    model = ForecastModel(
        ModelSpec.from_panel(panel, enc, ForecastConfig(horizon=4), 16, 4), ToyBackbone(16)
    )
    opt = torch.optim.Adam(model.parameters(), 2e-2)
    q = torch.tensor(model.spec.quantiles)

    def evaluation_loss():
        model.eval()
        with torch.no_grad():
            return np.mean(
                [
                    pinball_loss(
                        model(batch), batch.labels, batch.label_mask, batch.loss_eligible, q
                    )[0].item()
                    for batch in eval_batches
                ]
            )

    before = evaluation_loss()
    model.train()
    for epoch in range(80):
        sampler.set_epoch(epoch)
        for batch in sampler.batches(10):
            loss, _ = pinball_loss(
                model(batch), batch.labels, batch.label_mask, batch.loss_eligible, q
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    after = evaluation_loss()
    print(f"before={before:.9f}, after={after:.9f}, ratio={after / before:.9f}")
    assert after < 0.75 * before and all(p.grad is not None for p in model.head.parameters())


def test_fallback_flags_are_disjoint_and_cover_ineligible_contexts():
    panel = synthetic_panel(n_series=3, length=20, freq="W-SUN", seed=1, signal="seasonal_promo")
    panel.series[0].target[:] = np.nan
    panel.series[1].target[:] = 6
    enc = StaticEncoder().fit(panel, [s.series_id for s in panel.series])
    batch = make_batch(panel, [(i, 16) for i in range(3)], 4, 32, enc)
    model = ForecastModel(
        ModelSpec.from_panel(panel, enc, ForecastConfig(horizon=4), 16, 4), ToyBackbone(16)
    )
    flags = model.fallback_flags(batch)
    assert flags["no_context"].tolist() == [True, False, False]
    assert flags["undefined_scale"].tolist() == [False, True, False]
    prediction = model.predict(batch)
    assert torch.all(prediction[0] == 0) and torch.all(prediction[1] == 6)
