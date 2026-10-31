"""Small CPU-only configurations for shared trainer integration tests."""

from demand_forecast.config import (
    DatasetConfig,
    ForecastConfig,
    ModelConfig,
    Settings,
    TrainingConfig,
)


def tiny_settings(base_steps: int = 20, d_model: int = 16) -> Settings:
    """Use placeholder data roles that the fev task adapter replaces."""
    return Settings(
        data=DatasetConfig(id_columns=["id"], target="target", target_support="nonnegative"),
        forecast=ForecastConfig(horizon=4, max_context=64),
        model=ModelConfig(
            d_model=d_model,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            cat_emb_dim=4,
        ),
        training=TrainingConfig(
            base_steps=base_steps,
            eval_every=5,
            batch_size=8,
            warmup_steps=2,
            amp=False,
            checkpoint_every=5,
        ),
        seed=0,
        device="cpu",
    )
