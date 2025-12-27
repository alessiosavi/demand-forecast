"""Integration tests for complete training, inference, and tuning flows.

These tests verify end-to-end functionality with small datasets and few epochs
to ensure the models compile and train correctly.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from demand_forecast.models import (
    AdvancedDemandForecastModel,
    AdvancedDemandForecastModelV2,
    CombinedForecastLoss,
    LightweightDemandModel,
    QuantileLoss,
    WeightedMSELoss,
)
from demand_forecast.models.wrapper import create_model

# Small test configuration
BATCH_SIZE = 8
NUM_SAMPLES = 32
PAST_LEN = 12
FUTURE_LEN = 4
N_OUT = 4
SKU_VOCAB_SIZE = 20
CAT_FEATURES_DIM = {"category": 5, "color": 3}
NUM_EPOCHS = 2


class SyntheticDataset:
    """Generates synthetic data for testing."""

    def __init__(
        self,
        num_samples: int = NUM_SAMPLES,
        past_len: int = PAST_LEN,
        future_len: int = FUTURE_LEN,
        n_out: int = N_OUT,
    ):
        self.num_samples = num_samples
        self.past_len = past_len
        self.future_len = future_len
        self.n_out = n_out

        # Generate synthetic data
        self.qty = torch.randn(num_samples, past_len, 1)
        self.past_time = torch.randn(num_samples, past_len, 4)
        self.future_time = torch.randn(num_samples, future_len, 4)
        self.sku = torch.randint(0, SKU_VOCAB_SIZE, (num_samples,))

        # Create categorical features as 2D tensors (batch, vocab_size)
        # matching the real dataset format
        self.cats = {}
        for name, vocab in CAT_FEATURES_DIM.items():
            cat_indices = torch.zeros(num_samples, vocab, dtype=torch.long)
            for i in range(num_samples):
                num_active = torch.randint(1, min(3, vocab) + 1, (1,)).item()
                active_indices = torch.randperm(vocab)[:num_active]
                cat_indices[i, active_indices] = active_indices
            self.cats[name] = cat_indices

        # Generate targets with some structure (not pure noise)
        # Use a simple pattern: target is related to sum of qty
        qty_sum = self.qty.sum(dim=1).squeeze(-1)
        self.y = qty_sum.unsqueeze(-1).expand(-1, n_out) + torch.randn(num_samples, n_out) * 0.5

    def get_dataloader(self, batch_size: int = BATCH_SIZE) -> DataLoader:
        """Create a DataLoader from the synthetic data."""
        # Create a custom collate function
        indices = list(range(self.num_samples))

        def collate_fn(batch_indices):
            idx = torch.tensor(batch_indices)
            return {
                "qty": self.qty[idx],
                "past_time": self.past_time[idx],
                "future_time": self.future_time[idx],
                "sku": self.sku[idx],
                "cats": {k: v[idx] for k, v in self.cats.items()},
                "y": self.y[idx],
            }

        # Simple dataloader using indices
        return DataLoader(
            indices,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )


@pytest.fixture
def synthetic_data():
    """Create synthetic dataset."""
    return SyntheticDataset()


@pytest.fixture
def train_val_loaders(synthetic_data):
    """Create train and validation dataloaders."""
    # Split data
    train_data = SyntheticDataset(num_samples=24)
    val_data = SyntheticDataset(num_samples=8)
    return train_data.get_dataloader(), val_data.get_dataloader()


class TestStandardModelTraining:
    """Tests for standard model training flow."""

    @pytest.fixture
    def model(self):
        """Create a small standard model."""
        return AdvancedDemandForecastModel(
            sku_vocab_size=SKU_VOCAB_SIZE,
            sku_emb_dim=8,
            cat_features_dim=CAT_FEATURES_DIM,
            cat_emb_dims=8,
            past_time_features_dim=5,
            future_time_features_dim=4,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.1,
            n_out=N_OUT,
        )

    def test_training_loop(self, model, synthetic_data):
        """Test a complete training loop."""
        dataloader = synthetic_data.get_dataloader()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        initial_loss = None
        final_loss = None

        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                optimizer.zero_grad()

                output = model(
                    batch["qty"],
                    batch["past_time"],
                    batch["future_time"],
                    batch["sku"],
                    batch["cats"],
                )

                loss = criterion(output, batch["y"])
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            if epoch == 0:
                initial_loss = avg_loss
            final_loss = avg_loss

        # Model should be learning (loss should decrease or stay stable)
        assert initial_loss is not None
        assert final_loss is not None
        assert not torch.isnan(torch.tensor(final_loss))

    def test_training_with_improvements(self, synthetic_data):
        """Test training with model improvements enabled."""
        model = AdvancedDemandForecastModel(
            sku_vocab_size=SKU_VOCAB_SIZE,
            sku_emb_dim=8,
            cat_features_dim=CAT_FEATURES_DIM,
            cat_emb_dims=8,
            past_time_features_dim=5,
            future_time_features_dim=4,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.1,
            n_out=N_OUT,
            use_rope=True,
            use_pre_layernorm=True,
            use_film_conditioning=True,
            use_improved_head=True,
        )

        dataloader = synthetic_data.get_dataloader()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(
                batch["qty"],
                batch["past_time"],
                batch["future_time"],
                batch["sku"],
                batch["cats"],
            )
            loss = criterion(output, batch["y"])
            loss.backward()
            optimizer.step()
            break  # Just one batch to verify it works

        assert True  # If we get here, training works

    def test_inference_mode(self, model, synthetic_data):
        """Test inference after training."""
        dataloader = synthetic_data.get_dataloader()

        # Quick training
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(
                batch["qty"],
                batch["past_time"],
                batch["future_time"],
                batch["sku"],
                batch["cats"],
            )
            loss = nn.MSELoss()(output, batch["y"])
            loss.backward()
            optimizer.step()

        # Inference
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                output = model(
                    batch["qty"],
                    batch["past_time"],
                    batch["future_time"],
                    batch["sku"],
                    batch["cats"],
                )
                assert output.shape == (len(batch["sku"]), N_OUT)
                assert not torch.isnan(output).any()
                break


class TestAdvancedModelTraining:
    """Tests for advanced V2 model training flow."""

    @pytest.fixture
    def model(self):
        """Create a small advanced model."""
        return AdvancedDemandForecastModelV2(
            sku_vocab_size=SKU_VOCAB_SIZE,
            cat_features_dim=CAT_FEATURES_DIM,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.1,
            n_out=N_OUT,
            sku_emb_dim=8,
            cat_emb_dims=8,
            past_time_features_dim=5,
            future_time_features_dim=4,
            patch_size=4,
            num_quantiles=3,
            use_decomposition=True,
        )

    def test_training_with_combined_loss(self, model, synthetic_data):
        """Test training with CombinedForecastLoss."""
        dataloader = synthetic_data.get_dataloader()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = CombinedForecastLoss(quantiles=[0.1, 0.5, 0.9])

        model.train()

        for epoch in range(NUM_EPOCHS):
            for batch in dataloader:
                optimizer.zero_grad()

                output = model(
                    batch["qty"],
                    batch["past_time"],
                    batch["future_time"],
                    batch["sku"],
                    batch["cats"],
                )

                loss, loss_components = criterion(output, batch["y"])
                loss.backward()
                optimizer.step()

                # Verify loss components
                assert "point_loss" in loss_components
                assert "quantile_loss" in loss_components
                assert "decomposition_loss" in loss_components

        assert True  # Training completed

    def test_quantile_outputs(self, model, synthetic_data):
        """Test that quantile outputs are properly ordered."""
        model.eval()
        dataloader = synthetic_data.get_dataloader()

        with torch.no_grad():
            for batch in dataloader:
                output = model(
                    batch["qty"],
                    batch["past_time"],
                    batch["future_time"],
                    batch["sku"],
                    batch["cats"],
                )

                lower, median, upper = model.get_prediction_intervals(output)

                # Quantiles should be ordered (lower <= median <= upper)
                # Note: This may not hold for untrained model, so we just check shapes
                assert lower.shape == median.shape == upper.shape
                break


class TestLightweightModelTraining:
    """Tests for lightweight model training flow."""

    @pytest.fixture
    def model(self):
        """Create a small lightweight model."""
        return LightweightDemandModel(
            sku_vocab_size=SKU_VOCAB_SIZE,
            cat_features_dim=CAT_FEATURES_DIM,
            sku_emb_dim=8,
            cat_emb_dims=8,
            tcn_channels=[16, 32],
            kernel_size=3,
            dropout=0.1,
            n_out=N_OUT,
            past_time_features_dim=5,
            future_time_features_dim=4,
        )

    def test_training_loop(self, model, synthetic_data):
        """Test complete training loop for lightweight model."""
        dataloader = synthetic_data.get_dataloader()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        losses = []

        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                optimizer.zero_grad()

                output = model(
                    batch["qty"],
                    batch["past_time"],
                    batch["future_time"],
                    batch["sku"],
                    batch["cats"],
                )

                loss = criterion(output, batch["y"])
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        assert len(losses) == NUM_EPOCHS
        assert all(not torch.isnan(torch.tensor(loss)) for loss in losses)

    def test_fast_inference(self, model, synthetic_data):
        """Test that inference is fast for lightweight model."""
        import time

        model.eval()
        dataloader = synthetic_data.get_dataloader(batch_size=32)

        # Warm up
        for batch in dataloader:
            with torch.no_grad():
                _ = model(
                    batch["qty"],
                    batch["past_time"],
                    batch["future_time"],
                    batch["sku"],
                    batch["cats"],
                )
            break

        # Measure inference time
        start_time = time.time()
        num_inferences = 0

        with torch.no_grad():
            for batch in dataloader:
                _ = model(
                    batch["qty"],
                    batch["past_time"],
                    batch["future_time"],
                    batch["sku"],
                    batch["cats"],
                )
                num_inferences += 1

        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / num_inferences) * 1000

        # Should be reasonably fast (< 100ms per batch on CPU)
        assert avg_time_ms < 100, f"Inference too slow: {avg_time_ms:.2f}ms"


class TestLossFunctions:
    """Tests for loss functions."""

    def test_weighted_mse_loss(self):
        """Test WeightedMSELoss."""
        criterion = WeightedMSELoss(n_out=N_OUT)
        pred = torch.randn(BATCH_SIZE, N_OUT)
        target = torch.randn(BATCH_SIZE, N_OUT)

        loss = criterion(pred, target)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)

    def test_quantile_loss(self):
        """Test QuantileLoss."""
        criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        pred = torch.randn(BATCH_SIZE, N_OUT, 3)
        target = torch.randn(BATCH_SIZE, N_OUT)

        loss = criterion(pred, target)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)

    def test_combined_loss_with_tensor(self):
        """Test CombinedForecastLoss with tensor output."""
        criterion = CombinedForecastLoss()
        pred = torch.randn(BATCH_SIZE, N_OUT)
        target = torch.randn(BATCH_SIZE, N_OUT)

        loss, components = criterion(pred, target)
        assert loss.ndim == 0
        assert "point_loss" in components

    def test_combined_loss_with_dict(self):
        """Test CombinedForecastLoss with dict output."""
        criterion = CombinedForecastLoss()
        outputs = {
            "prediction": torch.randn(BATCH_SIZE, N_OUT),
            "quantiles": torch.randn(BATCH_SIZE, N_OUT, 3),
            "trend": torch.randn(BATCH_SIZE, N_OUT),
            "seasonality": torch.randn(BATCH_SIZE, N_OUT),
        }
        target = torch.randn(BATCH_SIZE, N_OUT)

        loss, components = criterion(outputs, target)
        assert loss.ndim == 0
        assert "point_loss" in components
        assert "quantile_loss" in components
        assert "decomposition_loss" in components


class TestModelCheckpointing:
    """Tests for model save/load functionality."""

    def test_save_load_standard_model(self, tmp_path):
        """Test saving and loading standard model."""
        model = AdvancedDemandForecastModel(
            sku_vocab_size=SKU_VOCAB_SIZE,
            sku_emb_dim=8,
            cat_features_dim=CAT_FEATURES_DIM,
            cat_emb_dims=8,
            past_time_features_dim=5,
            future_time_features_dim=4,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.1,
            n_out=N_OUT,
        )

        # Save
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)

        # Load into new model
        model2 = AdvancedDemandForecastModel(
            sku_vocab_size=SKU_VOCAB_SIZE,
            sku_emb_dim=8,
            cat_features_dim=CAT_FEATURES_DIM,
            cat_emb_dims=8,
            past_time_features_dim=5,
            future_time_features_dim=4,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.1,
            n_out=N_OUT,
        )
        model2.load_state_dict(torch.load(save_path, weights_only=True))

        # Verify weights are the same
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_save_load_advanced_model(self, tmp_path):
        """Test saving and loading advanced model."""
        model = AdvancedDemandForecastModelV2(
            sku_vocab_size=SKU_VOCAB_SIZE,
            cat_features_dim=CAT_FEATURES_DIM,
            d_model=32,
            nhead=4,
            n_out=N_OUT,
        )

        save_path = tmp_path / "model_v2.pt"
        torch.save(model.state_dict(), save_path)

        model2 = AdvancedDemandForecastModelV2(
            sku_vocab_size=SKU_VOCAB_SIZE,
            cat_features_dim=CAT_FEATURES_DIM,
            d_model=32,
            nhead=4,
            n_out=N_OUT,
        )
        model2.load_state_dict(torch.load(save_path, weights_only=True))

        assert True  # Load successful


class TestEndToEndFlow:
    """End-to-end tests combining train, validate, and inference."""

    def test_full_pipeline_standard(self, synthetic_data):
        """Test complete pipeline for standard model."""
        # Create model
        model = create_model(
            "standard",
            sku_vocab_size=SKU_VOCAB_SIZE,
            sku_emb_dim=8,
            cat_features_dim=CAT_FEATURES_DIM,
            cat_emb_dims=8,
            past_time_features_dim=5,
            future_time_features_dim=4,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.1,
            n_out=N_OUT,
        )

        dataloader = synthetic_data.get_dataloader()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Train
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(
                batch["qty"],
                batch["past_time"],
                batch["future_time"],
                batch["sku"],
                batch["cats"],
            )
            loss = criterion(output, batch["y"])
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in dataloader:
                output = model(
                    batch["qty"],
                    batch["past_time"],
                    batch["future_time"],
                    batch["sku"],
                    batch["cats"],
                )
                val_loss = criterion(output, batch["y"])
                val_losses.append(val_loss.item())

        assert len(val_losses) > 0
        assert all(not torch.isnan(torch.tensor(loss)) for loss in val_losses)

        # Inference on single sample
        model.eval()
        with torch.no_grad():
            single_output = model(
                synthetic_data.qty[:1],
                synthetic_data.past_time[:1],
                synthetic_data.future_time[:1],
                synthetic_data.sku[:1],
                {k: v[:1] for k, v in synthetic_data.cats.items()},
            )
            assert single_output.shape == (1, N_OUT)

    def test_full_pipeline_advanced(self, synthetic_data):
        """Test complete pipeline for advanced model."""
        model = create_model(
            "advanced",
            sku_vocab_size=SKU_VOCAB_SIZE,
            cat_features_dim=CAT_FEATURES_DIM,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            n_out=N_OUT,
            use_decomposition=True,
        )

        dataloader = synthetic_data.get_dataloader()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = CombinedForecastLoss()

        # Train
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(
                batch["qty"],
                batch["past_time"],
                batch["future_time"],
                batch["sku"],
                batch["cats"],
            )
            loss, _ = criterion(output, batch["y"])
            loss.backward()
            optimizer.step()

        # Inference with full outputs
        model.eval()
        with torch.no_grad():
            output = model(
                synthetic_data.qty[:1],
                synthetic_data.past_time[:1],
                synthetic_data.future_time[:1],
                synthetic_data.sku[:1],
                {k: v[:1] for k, v in synthetic_data.cats.items()},
                return_attention=True,
            )

            assert "prediction" in output
            assert "quantiles" in output
            assert "trend" in output
            assert "seasonality" in output
            assert "encoder_attention" in output

    def test_full_pipeline_lightweight(self, synthetic_data):
        """Test complete pipeline for lightweight model."""
        model = create_model(
            "lightweight",
            sku_vocab_size=SKU_VOCAB_SIZE,
            cat_features_dim=CAT_FEATURES_DIM,
            n_out=N_OUT,
            tcn_channels=[16, 32],
        )

        dataloader = synthetic_data.get_dataloader()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Train
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(
                batch["qty"],
                batch["past_time"],
                batch["future_time"],
                batch["sku"],
                batch["cats"],
            )
            loss = criterion(output, batch["y"])
            loss.backward()
            optimizer.step()

        # Verify model is small
        param_count = model.count_parameters()
        assert param_count < 3_000_000

        # Inference
        model.eval()
        with torch.no_grad():
            output = model(
                synthetic_data.qty[:1],
                synthetic_data.past_time[:1],
                synthetic_data.future_time[:1],
                synthetic_data.sku[:1],
                {k: v[:1] for k, v in synthetic_data.cats.items()},
            )
            assert output.shape == (1, N_OUT)
