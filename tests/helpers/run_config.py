"""Tiny CPU configurations for the public training/forecasting entry points."""

from pathlib import Path

import pandas as pd
import yaml

from demand_forecast.config import Settings


def write_cfg(
    tmp_path: Path,
    long: pd.DataFrame,
    static: pd.DataFrame,
    horizon: int,
    base_steps: int,
    d_model: int,
) -> Path:
    """Write weekly long/static CSVs and a small, reproducible YAML config."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    long.to_csv(tmp_path / "long.csv", index=False)
    static.to_csv(tmp_path / "static.csv", index=False)
    config = {
        "data": {
            "source": "long",
            "path": str(tmp_path / "long.csv"),
            "static_path": str(tmp_path / "static.csv"),
            "id_columns": ["id"],
            "target": "target",
            "target_support": "nonnegative",
            "known_dynamic": ["promo"],
            "past_dynamic": ["oil"],
            "static_categorical": ["cat"],
            "freq": "W-SUN",
        },
        "forecast": {"horizon": horizon, "max_context": 64},
        "model": {
            "d_model": d_model,
            "nhead": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 32,
            "hidden_continuous_size": 16,
            "cat_emb_dim": 4,
        },
        "training": {"base_steps": base_steps, "batch_size": 8, "warmup_steps": 2, "amp": False},
        "seed": 3,
        "device": "cpu",
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def settings_with(**training) -> Settings:
    """Build a settings object without requiring files to exist."""
    return Settings(
        data={"id_columns": ["id"], "target": "y", "target_support": "nonnegative"},
        forecast={"horizon": 4},
        training=training,
    )
