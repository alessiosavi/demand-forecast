"""Panel forecasts in original units using a portable inference bundle."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from demand_forecast.core.bundle import Bundle, load_bundle
from demand_forecast.data.frequency import grid_range, steps_between, to_offset, validate_on_grid
from demand_forecast.data.panel import Panel, Roles
from demand_forecast.data.sampler import BatchBuilder, align_to_origin


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
        ids, values, _ = self.forecast_with_diagnostics(panel, origin, batch_size)
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
        ids, arrays, _ = self.forecast_with_diagnostics(panel, origin)
        return ids, arrays

    def forecast_per_item_origins(
        self, panel: Panel, origins: Mapping[str, pd.Timestamp], batch_size: int = 512
    ) -> tuple[list[str], np.ndarray, dict]:
        """Group items by origin, returning forecasts in the original panel order.

        Early-ending items are forecast for calendar periods that other items'
        available past (and selection labels) already cover. No item's own future
        is used: fev truncates each item at its own cutoff and the loader reads only
        known covariates from future rows. This equals the information set fev gives
        every model and is not an out-of-time test for those items. The bundle's
        earliest permissible origin is a global floor, not a per-item guarantee.
        """
        started = time.perf_counter()
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        groups = {}
        ids = [series.series_id for series in panel.series]
        for index, key in enumerate(ids):
            if key not in origins:
                raise ValueError(f"missing origin for item {key}")
            groups.setdefault(pd.Timestamp(origins[key]), []).append(index)
        positions = {key: index for index, key in enumerate(ids)}
        cfg = self.bundle.forecast_cfg
        predictions = np.empty((len(ids), cfg.horizon, len(cfg.quantiles)), dtype=np.float32)
        diagnostics = {
            "fallback_no_context": 0,
            "fallback_undefined_scale": 0,
            "n_items": 0,
            "device": str(self.bundle.model.fallback_level.device),
            "n_origins": len(groups),
        }
        for origin, indices in groups.items():
            group = Panel(
                [panel.series[index] for index in indices],
                panel.roles,
                panel.freq,
                panel.seasonal_period,
                panel.target_support,
            )
            group_ids, arrays, diag = self.forecast_with_diagnostics(group, origin, batch_size)
            predictions[[positions[key] for key in group_ids]] = arrays
            for key in ("fallback_no_context", "fallback_undefined_scale", "n_items"):
                diagnostics[key] += diag[key]
            diagnostics["device"] = diag["device"]
        diagnostics["inference_time_s"] = time.perf_counter() - started
        return ids, predictions, diagnostics

    @torch.no_grad()
    def forecast_with_diagnostics(
        self, panel: Panel, origin: pd.Timestamp, batch_size: int = 512
    ) -> tuple[list[str], np.ndarray, dict]:
        """Forecast every item, measuring fallback counts and inference wall time."""
        started = time.perf_counter()
        origin = pd.Timestamp(origin)
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
        builder = BatchBuilder(batch_panel, cfg.horizon, cfg.max_context, self.bundle.encoder)
        predictions = np.empty((len(ids), cfg.horizon, len(cfg.quantiles)), dtype=np.float32)
        model = self.bundle.model
        model.eval()
        diagnostics = {
            "fallback_no_context": 0,
            "fallback_undefined_scale": 0,
            "n_items": len(ids),
            "device": str(model.fallback_level.device),
        }
        for start in range(0, len(items), batch_size):
            batch = builder.build(items[start : start + batch_size]).to(model.fallback_level.device)
            for name, flag in model.fallback_flags(batch).items():
                diagnostics[f"fallback_{name}"] += int(flag.sum().item())
            # predict performs the single denormalisation and the model's fallbacks.
            predictions[start : start + batch.size] = model.predict(batch).cpu().numpy()
        # Each .cpu().numpy() above waits for its CUDA output, so this includes
        # device work and host transfer rather than only asynchronous launches.
        diagnostics["inference_time_s"] = time.perf_counter() - started
        return ids, predictions, diagnostics
