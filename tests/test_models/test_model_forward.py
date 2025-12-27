"""Tests for model forward passes and basic functionality.

These tests verify that all model architectures:
1. Compile correctly
2. Produce outputs with correct shapes
3. Support gradient computation
4. Handle various input configurations
"""

import pytest
import torch
import torch.nn as nn

from demand_forecast.models import (
    AdvancedDemandForecastModel,
    AdvancedDemandForecastModelV2,
    LightweightDemandModel,
    LightweightMixerModel,
)
from demand_forecast.models.wrapper import ModelWrapper, create_model

# Test configuration
BATCH_SIZE = 4
PAST_LEN = 16
FUTURE_LEN = 8
N_OUT = 8
SKU_VOCAB_SIZE = 100
CAT_FEATURES_DIM = {"category": 10, "color": 5}


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing.

    Note: cats format is (batch, vocab_size) with boolean/int indices,
    matching the real dataset format where categorical features are one-hot encoded.
    """
    # Create categorical features as 2D tensors (batch, vocab_size)
    # Each row contains indices to look up in the embedding
    cats = {}
    for name, vocab in CAT_FEATURES_DIM.items():
        # Create random indices for each sample in batch
        # Shape: (batch, vocab_size) - this will be max-pooled in the model
        cat_indices = torch.zeros(BATCH_SIZE, vocab, dtype=torch.long)
        for i in range(BATCH_SIZE):
            # Set a few random positions to create multi-hot encoding
            num_active = torch.randint(1, min(3, vocab) + 1, (1,)).item()
            active_indices = torch.randperm(vocab)[:num_active]
            cat_indices[i, active_indices] = active_indices
        cats[name] = cat_indices

    return {
        "qty": torch.randn(BATCH_SIZE, PAST_LEN, 1),
        "past_time": torch.randn(BATCH_SIZE, PAST_LEN, 4),
        "future_time": torch.randn(BATCH_SIZE, FUTURE_LEN, 4),
        "sku": torch.randint(0, SKU_VOCAB_SIZE, (BATCH_SIZE,)),
        "cats": cats,
        "y": torch.randn(BATCH_SIZE, N_OUT),
    }


@pytest.fixture
def base_model_kwargs():
    """Base kwargs for model creation."""
    return {
        "sku_vocab_size": SKU_VOCAB_SIZE,
        "cat_features_dim": CAT_FEATURES_DIM,
        "sku_emb_dim": 16,
        "cat_emb_dims": 16,
        "past_time_features_dim": 5,  # qty (1) + time (4)
        "future_time_features_dim": 4,
        "d_model": 32,
        "nhead": 4,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.1,
        "n_out": N_OUT,
    }


class TestAdvancedDemandForecastModel:
    """Tests for the standard transformer model."""

    def test_forward_basic(self, sample_batch, base_model_kwargs):
        """Test basic forward pass."""
        model = AdvancedDemandForecastModel(**base_model_kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        assert output.shape == (BATCH_SIZE, N_OUT)
        assert not torch.isnan(output).any()

    def test_forward_with_rope(self, sample_batch, base_model_kwargs):
        """Test forward pass with RoPE enabled."""
        kwargs = {**base_model_kwargs, "use_rope": True}
        model = AdvancedDemandForecastModel(**kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        assert output.shape == (BATCH_SIZE, N_OUT)

    def test_forward_with_pre_layernorm(self, sample_batch, base_model_kwargs):
        """Test forward pass with Pre-LayerNorm."""
        kwargs = {**base_model_kwargs, "use_pre_layernorm": True}
        model = AdvancedDemandForecastModel(**kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        assert output.shape == (BATCH_SIZE, N_OUT)

    def test_forward_with_film(self, sample_batch, base_model_kwargs):
        """Test forward pass with FiLM conditioning."""
        kwargs = {**base_model_kwargs, "use_film_conditioning": True}
        model = AdvancedDemandForecastModel(**kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        assert output.shape == (BATCH_SIZE, N_OUT)

    def test_forward_with_improved_head(self, sample_batch, base_model_kwargs):
        """Test forward pass with improved output head."""
        kwargs = {**base_model_kwargs, "use_improved_head": True}
        model = AdvancedDemandForecastModel(**kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        assert output.shape == (BATCH_SIZE, N_OUT)

    def test_forward_all_improvements(self, sample_batch, base_model_kwargs):
        """Test forward pass with all improvements enabled."""
        kwargs = {
            **base_model_kwargs,
            "use_rope": True,
            "use_pre_layernorm": True,
            "use_film_conditioning": True,
            "use_improved_head": True,
            "stochastic_depth_rate": 0.1,
        }
        model = AdvancedDemandForecastModel(**kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        assert output.shape == (BATCH_SIZE, N_OUT)

    def test_gradient_flow(self, sample_batch, base_model_kwargs):
        """Test that gradients flow through the model."""
        model = AdvancedDemandForecastModel(**base_model_kwargs)
        model.train()

        output = model(
            sample_batch["qty"],
            sample_batch["past_time"],
            sample_batch["future_time"],
            sample_batch["sku"],
            sample_batch["cats"],
        )

        loss = nn.MSELoss()(output, sample_batch["y"])
        loss.backward()

        # Check that gradients exist for key parameters
        assert model.sku_embedding.weight.grad is not None
        assert not torch.isnan(model.sku_embedding.weight.grad).any()


class TestAdvancedDemandForecastModelV2:
    """Tests for the research-grade model."""

    @pytest.fixture
    def v2_model_kwargs(self, base_model_kwargs):
        """Kwargs for V2 model."""
        return {
            "sku_vocab_size": SKU_VOCAB_SIZE,
            "cat_features_dim": CAT_FEATURES_DIM,
            "d_model": 32,
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "n_out": N_OUT,
            "sku_emb_dim": 16,
            "cat_emb_dims": 16,
            "past_time_features_dim": 5,
            "future_time_features_dim": 4,
            "patch_size": 4,
            "num_quantiles": 3,
            "use_decomposition": True,
            "use_patch_embedding": True,
        }

    def test_forward_basic(self, sample_batch, v2_model_kwargs):
        """Test basic forward pass returns dict output."""
        model = AdvancedDemandForecastModelV2(**v2_model_kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        assert isinstance(output, dict)
        assert "prediction" in output
        assert "quantiles" in output
        assert output["prediction"].shape == (BATCH_SIZE, N_OUT)
        assert output["quantiles"].shape == (BATCH_SIZE, N_OUT, 3)

    def test_forward_with_decomposition(self, sample_batch, v2_model_kwargs):
        """Test that decomposition outputs are present."""
        model = AdvancedDemandForecastModelV2(**v2_model_kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        assert "trend" in output
        assert "seasonality" in output
        assert output["trend"].shape == (BATCH_SIZE, N_OUT)
        assert output["seasonality"].shape == (BATCH_SIZE, N_OUT)

    def test_forward_without_decomposition(self, sample_batch, v2_model_kwargs):
        """Test forward pass without decomposition."""
        kwargs = {**v2_model_kwargs, "use_decomposition": False}
        model = AdvancedDemandForecastModelV2(**kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        assert "prediction" in output
        assert "trend" not in output

    def test_forward_with_attention_weights(self, sample_batch, v2_model_kwargs):
        """Test that attention weights can be returned."""
        model = AdvancedDemandForecastModelV2(**v2_model_kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
                return_attention=True,
            )

        assert "encoder_attention" in output
        assert "decoder_attention" in output

    def test_feature_weights(self, sample_batch, v2_model_kwargs):
        """Test that feature weights are returned."""
        model = AdvancedDemandForecastModelV2(**v2_model_kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        assert "feature_weights" in output
        # 1 (SKU) + 2 (categories) = 3 features
        assert output["feature_weights"].shape[1] == 3

    def test_get_prediction_intervals(self, sample_batch, v2_model_kwargs):
        """Test prediction interval extraction."""
        model = AdvancedDemandForecastModelV2(**v2_model_kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        lower, median, upper = model.get_prediction_intervals(output)
        assert lower.shape == (BATCH_SIZE, N_OUT)
        assert median.shape == (BATCH_SIZE, N_OUT)
        assert upper.shape == (BATCH_SIZE, N_OUT)

    def test_gradient_flow(self, sample_batch, v2_model_kwargs):
        """Test gradient flow for V2 model."""
        model = AdvancedDemandForecastModelV2(**v2_model_kwargs)
        model.train()

        output = model(
            sample_batch["qty"],
            sample_batch["past_time"],
            sample_batch["future_time"],
            sample_batch["sku"],
            sample_batch["cats"],
        )

        loss = nn.MSELoss()(output["prediction"], sample_batch["y"])
        loss.backward()

        assert model.sku_embedding.weight.grad is not None


class TestLightweightDemandModel:
    """Tests for the lightweight TCN model."""

    @pytest.fixture
    def lightweight_kwargs(self):
        """Kwargs for lightweight model."""
        return {
            "sku_vocab_size": SKU_VOCAB_SIZE,
            "cat_features_dim": CAT_FEATURES_DIM,
            "sku_emb_dim": 8,
            "cat_emb_dims": 8,
            "tcn_channels": [16, 32],
            "kernel_size": 3,
            "dropout": 0.1,
            "n_out": N_OUT,
            "past_time_features_dim": 5,
            "future_time_features_dim": 4,
        }

    def test_forward_basic(self, sample_batch, lightweight_kwargs):
        """Test basic forward pass."""
        model = LightweightDemandModel(**lightweight_kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        assert output.shape == (BATCH_SIZE, N_OUT)
        assert not torch.isnan(output).any()

    def test_forward_without_film(self, sample_batch, lightweight_kwargs):
        """Test forward without FiLM conditioning."""
        kwargs = {**lightweight_kwargs, "use_film": False}
        model = LightweightDemandModel(**kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        assert output.shape == (BATCH_SIZE, N_OUT)

    def test_parameter_count(self, lightweight_kwargs):
        """Test that model has fewer than 3M parameters."""
        model = LightweightDemandModel(**lightweight_kwargs)
        param_count = model.count_parameters()

        assert param_count < 3_000_000
        print(f"Lightweight model has {param_count:,} parameters")

    def test_model_size(self, lightweight_kwargs):
        """Test model size estimation."""
        model = LightweightDemandModel(**lightweight_kwargs)

        fp32_size = model.get_model_size_mb(quantized=False)
        int8_size = model.get_model_size_mb(quantized=True)

        assert fp32_size < 50  # MB
        assert int8_size < 15  # MB

    def test_gradient_flow(self, sample_batch, lightweight_kwargs):
        """Test gradient flow."""
        model = LightweightDemandModel(**lightweight_kwargs)
        model.train()

        output = model(
            sample_batch["qty"],
            sample_batch["past_time"],
            sample_batch["future_time"],
            sample_batch["sku"],
            sample_batch["cats"],
        )

        loss = nn.MSELoss()(output, sample_batch["y"])
        loss.backward()

        assert model.sku_embedding.weight.grad is not None


class TestLightweightMixerModel:
    """Tests for the MLP-Mixer model."""

    @pytest.fixture
    def mixer_kwargs(self):
        """Kwargs for mixer model."""
        return {
            "sku_vocab_size": SKU_VOCAB_SIZE,
            "cat_features_dim": CAT_FEATURES_DIM,
            "sku_emb_dim": 8,
            "cat_emb_dims": 8,
            "hidden_dim": 32,
            "num_layers": 2,
            "dropout": 0.1,
            "n_out": N_OUT,
            "past_time_features_dim": 5,
            "future_time_features_dim": 4,
            "max_seq_len": PAST_LEN,
        }

    def test_forward_basic(self, sample_batch, mixer_kwargs):
        """Test basic forward pass."""
        model = LightweightMixerModel(**mixer_kwargs)
        model.eval()

        with torch.no_grad():
            output = model(
                sample_batch["qty"],
                sample_batch["past_time"],
                sample_batch["future_time"],
                sample_batch["sku"],
                sample_batch["cats"],
            )

        assert output.shape == (BATCH_SIZE, N_OUT)
        assert not torch.isnan(output).any()


class TestModelWrapper:
    """Tests for the model wrapper."""

    def test_wrapper_standard_model(self, sample_batch, base_model_kwargs):
        """Test wrapper with standard model."""
        wrapper = ModelWrapper(n=2, model_type="standard", **base_model_kwargs)

        output = wrapper(
            "0",
            sample_batch["qty"],
            sample_batch["past_time"],
            sample_batch["future_time"],
            sample_batch["sku"],
            sample_batch["cats"],
        )

        assert output.shape == (BATCH_SIZE, N_OUT)

    def test_wrapper_advanced_model(self, sample_batch):
        """Test wrapper with advanced model."""
        kwargs = {
            "sku_vocab_size": SKU_VOCAB_SIZE,
            "cat_features_dim": CAT_FEATURES_DIM,
            "d_model": 32,
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "n_out": N_OUT,
        }
        wrapper = ModelWrapper(n=1, model_type="advanced", **kwargs)

        output = wrapper(
            "0",
            sample_batch["qty"],
            sample_batch["past_time"],
            sample_batch["future_time"],
            sample_batch["sku"],
            sample_batch["cats"],
        )

        assert isinstance(output, dict)
        assert "prediction" in output

    def test_wrapper_lightweight_model(self, sample_batch):
        """Test wrapper with lightweight model."""
        kwargs = {
            "sku_vocab_size": SKU_VOCAB_SIZE,
            "cat_features_dim": CAT_FEATURES_DIM,
            "n_out": N_OUT,
            "tcn_channels": [16, 32],
        }
        wrapper = ModelWrapper(n=1, model_type="lightweight", **kwargs)

        output = wrapper(
            "0",
            sample_batch["qty"],
            sample_batch["past_time"],
            sample_batch["future_time"],
            sample_batch["sku"],
            sample_batch["cats"],
        )

        assert output.shape == (BATCH_SIZE, N_OUT)

    def test_wrapper_model_info(self, base_model_kwargs):
        """Test model info retrieval."""
        wrapper = ModelWrapper(n=3, model_type="standard", **base_model_kwargs)
        info = wrapper.get_model_info()

        assert info["model_type"] == "standard"
        assert info["num_clusters"] == 3
        assert "total_parameters" in info


class TestCreateModel:
    """Tests for the create_model factory function."""

    def test_create_standard(self, base_model_kwargs):
        """Test creating standard model."""
        model = create_model("standard", **base_model_kwargs)
        assert isinstance(model, AdvancedDemandForecastModel)

    def test_create_advanced(self):
        """Test creating advanced model."""
        kwargs = {
            "sku_vocab_size": SKU_VOCAB_SIZE,
            "cat_features_dim": CAT_FEATURES_DIM,
            "d_model": 32,
            "nhead": 4,
            "n_out": N_OUT,
        }
        model = create_model("advanced", **kwargs)
        assert isinstance(model, AdvancedDemandForecastModelV2)

    def test_create_lightweight(self):
        """Test creating lightweight model."""
        kwargs = {
            "sku_vocab_size": SKU_VOCAB_SIZE,
            "cat_features_dim": CAT_FEATURES_DIM,
            "n_out": N_OUT,
        }
        model = create_model("lightweight", **kwargs)
        assert isinstance(model, LightweightDemandModel)

    def test_create_invalid_type(self, base_model_kwargs):
        """Test that invalid type raises error."""
        with pytest.raises(ValueError):
            create_model("invalid_type", **base_model_kwargs)
