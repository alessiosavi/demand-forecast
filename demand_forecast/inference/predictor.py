"""Panel forecasts in original units using a portable inference bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from demand_forecast.core.bundle import Bundle, load_bundle
from demand_forecast.data.frequency import grid_range, steps_between, to_offset, validate_on_grid
from demand_forecast.data.panel import Panel, Roles
from demand_forecast.data.sampler import align_to_origin, make_batch


@dataclass
class Predictor:
    """Serve every input series, including unseen ids and fallback-only histories."""

    bundle: Bundle

    @classmethod
    def from_bundle(cls, directory: Path | str, device: torch.device | str = "cpu") -> Predictor:
        return cls(load_bundle(directory, device=device))

    def forecast(self, panel: Panel, origin: pd.Timestamp, batch_size: int = 512) -> pd.DataFrame:
        """Return one row per series and forecast step, with median point forecasts."""
        origin = pd.Timestamp(origin)
        ids, values = self._forecast_arrays(panel, origin, batch_size)
        cfg = self.bundle.forecast_cfg
        data = {
            "series_id": np.repeat(ids, cfg.horizon),
            "origin": origin,
            "timestamp": np.tile(grid_range(origin, cfg.horizon, panel.freq), len(ids)),
        }
        flat = values.reshape(len(ids) * cfg.horizon, len(cfg.quantiles))
        data.update({f"q{q}": flat[:, i] for i, q in enumerate(cfg.quantiles)})
        data["point"] = flat[:, cfg.median_index]
        return pd.DataFrame(data)

    def forecast_arrays(self, panel: Panel, origin: pd.Timestamp) -> tuple[list[str], np.ndarray]:
        """Return ids in input order and original-unit quantiles of shape (N, H, Q)."""
        return self._forecast_arrays(panel, pd.Timestamp(origin), 512)

    @torch.no_grad()
    def _forecast_arrays(
        self, panel: Panel, origin: pd.Timestamp, batch_size: int
    ) -> tuple[list[str], np.ndarray]:
        metadata = self.bundle.panel_metadata
        if origin < pd.Timestamp(metadata["earliest_origin"]):
            raise ValueError("origin before earliest permissible origin")
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        roles = Roles(
            **{
                key: value if key == "target" else tuple(value)
                for key, value in metadata["roles"].items()
            }
        )
        if to_offset(panel.freq) != to_offset(metadata["freq"]):
            raise ValueError("panel frequency does not match bundle")
        if panel.roles != roles:
            raise ValueError("panel roles do not match bundle")
        if panel.target_support != metadata["target_support"]:
            raise ValueError("panel target_support does not match bundle")
        if panel.seasonal_period != metadata["seasonal_period"]:
            raise ValueError("panel seasonal_period does not match bundle")
        validate_on_grid(pd.DatetimeIndex([origin]), panel.freq, "forecast origin")
        cfg = self.bundle.forecast_cfg
        ids = [series.series_id for series in panel.series]
        items = [
            (i, max(steps_between(series.start, origin, panel.freq), 0))
            for i, series in enumerate(panel.series)
        ]
        batch_panel = align_to_origin(panel, items)
        predictions = np.empty((len(ids), cfg.horizon, len(cfg.quantiles)), dtype=np.float32)
        model = self.bundle.model
        model.eval()
        for start in range(0, len(items), batch_size):
            batch = make_batch(
                batch_panel,
                items[start : start + batch_size],
                cfg.horizon,
                cfg.max_context,
                self.bundle.encoder,
            ).to(model.fallback_level.device)
            # predict performs the single denormalisation and the model's fallbacks.
            predictions[start : start + batch.size] = model.predict(batch).cpu().numpy()
        return ids, predictions
