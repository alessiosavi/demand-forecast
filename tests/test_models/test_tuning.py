"""Tests for hyperparameter tuning functionality.

These tests verify that the tuning module works correctly with small
configurations to ensure quick test execution.
"""

import pytest
import torch
from torch.utils.data import DataLoader

# Check if optuna is available
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from demand_forecast.models import (
    AdvancedDemandForecastModel,
)

# Skip all tests if optuna is not installed
pytestmark = pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna is not installed")


# Test configuration
BATCH_SIZE = 4
NUM_SAMPLES = 16
PAST_LEN = 8
FUTURE_LEN = 4
N_OUT = 4
SKU_VOCAB_SIZE = 10
CAT_FEATURES_DIM = {"category": 5}


class SyntheticTuningDataset:
    """Small synthetic dataset for tuning tests."""

    def __init__(self, num_samples: int = NUM_SAMPLES):
        self.num_samples = num_samples
        self.qty = torch.randn(num_samples, PAST_LEN, 1)
        self.past_time = torch.randn(num_samples, PAST_LEN, 4)
        self.future_time = torch.randn(num_samples, FUTURE_LEN, 4)
        self.sku = torch.randint(0, SKU_VOCAB_SIZE, (num_samples,))

        # Create categorical features as 2D tensors (batch, vocab_size)
        self.cats = {}
        for name, vocab in CAT_FEATURES_DIM.items():
            cat_indices = torch.zeros(num_samples, vocab, dtype=torch.long)
            for i in range(num_samples):
                num_active = torch.randint(1, min(3, vocab) + 1, (1,)).item()
                active_indices = torch.randperm(vocab)[:num_active]
                cat_indices[i, active_indices] = active_indices
            self.cats[name] = cat_indices

        qty_sum = self.qty.sum(dim=1).squeeze(-1)
        self.y = qty_sum.unsqueeze(-1).expand(-1, N_OUT) + torch.randn(num_samples, N_OUT) * 0.5

    def get_dataloader(self, batch_size: int = BATCH_SIZE) -> DataLoader:
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

        return DataLoader(
            indices,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )


@pytest.fixture
def train_loader():
    """Create training dataloader."""
    return SyntheticTuningDataset(num_samples=16).get_dataloader()


@pytest.fixture
def val_loader():
    """Create validation dataloader."""
    return SyntheticTuningDataset(num_samples=8).get_dataloader()


@pytest.fixture
def base_model_kwargs():
    """Base kwargs for model creation."""
    return {
        "sku_vocab_size": SKU_VOCAB_SIZE,
        "cat_features_dim": CAT_FEATURES_DIM,
        "sku_emb_dim": 8,
        "cat_emb_dims": 8,
        "past_time_features_dim": 5,
        "future_time_features_dim": 4,
        "d_model": 32,
        "nhead": 4,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.1,
        "n_out": N_OUT,
    }


class TestTuningConfig:
    """Tests for TuningConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        from demand_forecast.core.tuning import TuningConfig

        config = TuningConfig()
        assert config.n_trials == 50
        assert config.direction == "minimize"
        assert config.metric == "mse"
        assert config.sampler == "tpe"
        assert config.pruner == "median"

    def test_custom_config(self):
        """Test custom configuration."""
        from demand_forecast.core.tuning import TuningConfig

        config = TuningConfig(
            n_trials=10,
            timeout=60,
            direction="maximize",
            metric="mae",
        )
        assert config.n_trials == 10
        assert config.timeout == 60
        assert config.direction == "maximize"
        assert config.metric == "mae"


class TestSearchSpace:
    """Tests for SearchSpace dataclass."""

    def test_default_search_space(self):
        """Test default search space."""
        from demand_forecast.core.tuning import SearchSpace

        space = SearchSpace()
        assert space.d_model == (64, 512)
        assert space.learning_rate == (1e-6, 1e-3)
        assert 4 in space.nhead
        assert 8 in space.nhead

    def test_custom_search_space(self):
        """Test custom search space."""
        from demand_forecast.core.tuning import SearchSpace

        space = SearchSpace(
            d_model=(32, 128),
            nhead=[2, 4],
            learning_rate=(1e-5, 1e-4),
        )
        assert space.d_model == (32, 128)
        assert space.nhead == [2, 4]


class TestSamplerAndPruner:
    """Tests for sampler and pruner creation."""

    def test_create_tpe_sampler(self):
        """Test TPE sampler creation."""
        from demand_forecast.core.tuning import create_sampler

        sampler = create_sampler("tpe", seed=42)
        assert isinstance(sampler, optuna.samplers.TPESampler)

    def test_create_random_sampler(self):
        """Test random sampler creation."""
        from demand_forecast.core.tuning import create_sampler

        sampler = create_sampler("random", seed=42)
        assert isinstance(sampler, optuna.samplers.RandomSampler)

    def test_create_median_pruner(self):
        """Test median pruner creation."""
        from demand_forecast.core.tuning import create_pruner

        pruner = create_pruner("median")
        assert isinstance(pruner, optuna.pruners.MedianPruner)

    def test_create_hyperband_pruner(self):
        """Test hyperband pruner creation."""
        from demand_forecast.core.tuning import create_pruner

        pruner = create_pruner("hyperband")
        assert isinstance(pruner, optuna.pruners.HyperbandPruner)

    def test_create_nop_pruner(self):
        """Test nop pruner creation."""
        from demand_forecast.core.tuning import create_pruner

        pruner = create_pruner("none")
        assert isinstance(pruner, optuna.pruners.NopPruner)

    def test_invalid_sampler(self):
        """Test invalid sampler raises error."""
        from demand_forecast.core.tuning import create_sampler

        with pytest.raises(ValueError):
            create_sampler("invalid", seed=42)

    def test_invalid_pruner(self):
        """Test invalid pruner raises error."""
        from demand_forecast.core.tuning import create_pruner

        with pytest.raises(ValueError):
            create_pruner("invalid")


class TestSuggestHyperparameters:
    """Tests for hyperparameter suggestion."""

    def test_suggest_for_standard_model(self):
        """Test parameter suggestion for standard model."""
        from demand_forecast.core.tuning import SearchSpace, suggest_hyperparameters

        study = optuna.create_study()
        trial = study.ask()
        space = SearchSpace()

        params = suggest_hyperparameters(trial, space, model_type="standard")

        assert "d_model" in params or "d_model_base" in trial.params
        assert "nhead" in params
        assert "dropout" in params
        assert "learning_rate" in params

    def test_suggest_for_advanced_model(self):
        """Test parameter suggestion for advanced model."""
        from demand_forecast.core.tuning import SearchSpace, suggest_hyperparameters

        study = optuna.create_study()
        trial = study.ask()
        space = SearchSpace()

        params = suggest_hyperparameters(trial, space, model_type="advanced")

        assert "patch_size" in params
        assert "use_decomposition" in params

    def test_suggest_for_lightweight_model(self):
        """Test parameter suggestion for lightweight model."""
        from demand_forecast.core.tuning import SearchSpace, suggest_hyperparameters

        study = optuna.create_study()
        trial = study.ask()
        space = SearchSpace()

        params = suggest_hyperparameters(trial, space, model_type="lightweight")

        assert "tcn_channels" in params
        assert "kernel_size" in params


class TestHyperparameterTuner:
    """Tests for the HyperparameterTuner class."""

    def test_tuner_initialization(self, train_loader, val_loader, base_model_kwargs):
        """Test tuner initialization."""
        from demand_forecast.core.tuning import HyperparameterTuner, TuningConfig

        tuner = HyperparameterTuner(
            model_type="standard",
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            base_model_kwargs=base_model_kwargs,
            config=TuningConfig(n_trials=2),
            num_epochs_per_trial=1,
        )

        assert tuner.model_type == "standard"
        assert tuner.num_epochs_per_trial == 1

    def test_tune_standard_model(self, train_loader, val_loader, base_model_kwargs):
        """Test tuning a standard model with minimal trials."""
        from demand_forecast.core.tuning import (
            HyperparameterTuner,
            SearchSpace,
            TuningConfig,
        )

        # Use very limited search space for speed
        search_space = SearchSpace(
            d_model=None,  # Disable
            nhead=None,
            num_encoder_layers=None,
            num_decoder_layers=None,
            dim_feedforward=None,
            learning_rate=(1e-4, 1e-3),  # Narrow range
            weight_decay=None,
            batch_size=None,
            use_rope=None,
            use_pre_layernorm=None,
            use_film_conditioning=None,
            use_improved_head=None,
            stochastic_depth_rate=None,
        )

        tuner = HyperparameterTuner(
            model_type="standard",
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            base_model_kwargs=base_model_kwargs,
            config=TuningConfig(n_trials=2, sampler="random"),
            search_space=search_space,
            num_epochs_per_trial=1,
            early_stop_patience=1,
        )

        best_params = tuner.tune(n_trials=2, show_progress_bar=False)

        assert tuner.study is not None
        assert tuner.study.best_trial is not None
        assert "learning_rate" in best_params

    def test_tune_lightweight_model(self, train_loader, val_loader):
        """Test tuning a lightweight model."""
        from demand_forecast.core.tuning import (
            HyperparameterTuner,
            SearchSpace,
            TuningConfig,
        )

        base_kwargs = {
            "sku_vocab_size": SKU_VOCAB_SIZE,
            "cat_features_dim": CAT_FEATURES_DIM,
            "n_out": N_OUT,
            "tcn_channels": [16, 32],
        }

        search_space = SearchSpace(
            d_model=None,
            nhead=None,
            num_encoder_layers=None,
            num_decoder_layers=None,
            dim_feedforward=None,
            sku_emb_dim=None,
            cat_emb_dims=None,
            learning_rate=(1e-4, 1e-3),
            weight_decay=None,
            batch_size=None,
            tcn_channels_depth=None,
            tcn_channels_width=None,
            tcn_kernel_size=None,
        )

        tuner = HyperparameterTuner(
            model_type="lightweight",
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            base_model_kwargs=base_kwargs,
            config=TuningConfig(n_trials=2, sampler="random"),
            search_space=search_space,
            num_epochs_per_trial=1,
        )

        _ = tuner.tune(n_trials=2, show_progress_bar=False)
        assert tuner.study is not None

    def test_get_best_model(self, train_loader, val_loader, base_model_kwargs):
        """Test retrieving best model after tuning."""
        from demand_forecast.core.tuning import (
            HyperparameterTuner,
            SearchSpace,
            TuningConfig,
        )

        search_space = SearchSpace(
            d_model=None,
            nhead=None,
            num_encoder_layers=None,
            num_decoder_layers=None,
            dim_feedforward=None,
            learning_rate=(1e-4, 1e-3),
            weight_decay=None,
            batch_size=None,
            use_rope=None,
            use_pre_layernorm=None,
            use_film_conditioning=None,
            use_improved_head=None,
            stochastic_depth_rate=None,
        )

        tuner = HyperparameterTuner(
            model_type="standard",
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            base_model_kwargs=base_model_kwargs,
            config=TuningConfig(n_trials=2, sampler="random"),
            search_space=search_space,
            num_epochs_per_trial=1,
        )

        tuner.tune(n_trials=2, show_progress_bar=False)
        best_model = tuner.get_best_model()

        assert isinstance(best_model, AdvancedDemandForecastModel)

    def test_get_best_model_before_tuning_raises(self, train_loader, val_loader, base_model_kwargs):
        """Test that getting best model before tuning raises error."""
        from demand_forecast.core.tuning import HyperparameterTuner

        tuner = HyperparameterTuner(
            model_type="standard",
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            base_model_kwargs=base_model_kwargs,
        )

        with pytest.raises(RuntimeError):
            tuner.get_best_model()

    def test_study_dataframe(self, train_loader, val_loader, base_model_kwargs):
        """Test getting study results as dataframe."""
        from demand_forecast.core.tuning import (
            HyperparameterTuner,
            SearchSpace,
            TuningConfig,
        )

        search_space = SearchSpace(
            d_model=None,
            nhead=None,
            num_encoder_layers=None,
            num_decoder_layers=None,
            dim_feedforward=None,
            learning_rate=(1e-4, 1e-3),
            weight_decay=None,
            batch_size=None,
        )

        tuner = HyperparameterTuner(
            model_type="standard",
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            base_model_kwargs=base_model_kwargs,
            config=TuningConfig(n_trials=2, sampler="random"),
            search_space=search_space,
            num_epochs_per_trial=1,
        )

        tuner.tune(n_trials=2, show_progress_bar=False)
        df = tuner.get_study_dataframe()

        assert len(df) == 2
        assert "value" in df.columns


class TestQuickTune:
    """Tests for the quick_tune convenience function."""

    def test_quick_tune(self, train_loader, val_loader, base_model_kwargs):
        """Test quick_tune function."""
        from demand_forecast.core.tuning import quick_tune

        best_params, best_model = quick_tune(
            model_type="standard",
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            base_model_kwargs=base_model_kwargs,
            n_trials=2,
        )

        assert isinstance(best_params, dict)
        assert isinstance(best_model, AdvancedDemandForecastModel)


class TestTuningIntegration:
    """Integration tests for the complete tuning workflow."""

    def test_train_tune_inference_flow(self, train_loader, val_loader, base_model_kwargs):
        """Test complete train-tune-inference workflow."""
        import torch.nn as nn

        from demand_forecast.core.tuning import (
            HyperparameterTuner,
            SearchSpace,
            TuningConfig,
        )

        # 1. Tune hyperparameters
        search_space = SearchSpace(
            d_model=None,
            nhead=None,
            num_encoder_layers=None,
            num_decoder_layers=None,
            dim_feedforward=None,
            dropout=(0.1, 0.3),
            learning_rate=(1e-4, 1e-3),
            weight_decay=None,
            batch_size=None,
        )

        tuner = HyperparameterTuner(
            model_type="standard",
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            base_model_kwargs=base_model_kwargs,
            config=TuningConfig(n_trials=2, sampler="random"),
            search_space=search_space,
            num_epochs_per_trial=1,
        )

        best_params = tuner.tune(n_trials=2, show_progress_bar=False)
        best_model = tuner.get_best_model()

        # 2. Additional training with best model
        optimizer = torch.optim.AdamW(
            best_model.parameters(),
            lr=best_params.get("learning_rate", 1e-4),
        )
        criterion = nn.MSELoss()

        best_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            output = best_model(
                batch["qty"],
                batch["past_time"],
                batch["future_time"],
                batch["sku"],
                batch["cats"],
            )
            loss = criterion(output, batch["y"])
            loss.backward()
            optimizer.step()

        # 3. Inference
        best_model.eval()
        with torch.no_grad():
            for batch in val_loader:
                output = best_model(
                    batch["qty"],
                    batch["past_time"],
                    batch["future_time"],
                    batch["sku"],
                    batch["cats"],
                )
                assert output.shape[1] == N_OUT
                assert not torch.isnan(output).any()
                break

        assert True  # Complete flow succeeded


class TestTuningWithDifferentMetrics:
    """Tests for tuning with different optimization metrics."""

    def test_tune_with_mae_metric(self, train_loader, val_loader, base_model_kwargs):
        """Test tuning optimizing MAE."""
        from demand_forecast.core.tuning import (
            HyperparameterTuner,
            SearchSpace,
            TuningConfig,
        )

        search_space = SearchSpace(
            d_model=None,
            nhead=None,
            learning_rate=(1e-4, 1e-3),
        )

        tuner = HyperparameterTuner(
            model_type="standard",
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            base_model_kwargs=base_model_kwargs,
            config=TuningConfig(n_trials=2, metric="mae", sampler="random"),
            search_space=search_space,
            num_epochs_per_trial=1,
        )

        tuner.tune(n_trials=2, show_progress_bar=False)
        assert tuner.study is not None

    def test_tune_with_loss_metric(self, train_loader, val_loader, base_model_kwargs):
        """Test tuning optimizing loss directly."""
        from demand_forecast.core.tuning import (
            HyperparameterTuner,
            SearchSpace,
            TuningConfig,
        )

        search_space = SearchSpace(
            d_model=None,
            nhead=None,
            learning_rate=(1e-4, 1e-3),
        )

        tuner = HyperparameterTuner(
            model_type="standard",
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            base_model_kwargs=base_model_kwargs,
            config=TuningConfig(n_trials=2, metric="loss", sampler="random"),
            search_space=search_space,
            num_epochs_per_trial=1,
        )

        tuner.tune(n_trials=2, show_progress_bar=False)
        assert tuner.study is not None
