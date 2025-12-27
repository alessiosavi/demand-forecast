"""Tests for model evaluator."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from demand_forecast.core.evaluator import Evaluator, ValidationResult


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, output_value: float = 1.0):
        super().__init__()
        self.output_value = output_value
        self.linear = nn.Linear(1, 1)  # Dummy layer to make it a valid module

    def forward(self, model_idx, qty, past_time, future_time, sku, cats):
        batch_size = qty.shape[0]
        n_out = future_time.shape[1]
        return torch.full((batch_size, n_out, 1), self.output_value)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(
            predictions=[np.array([1.0, 2.0])],
            actuals=[np.array([1.1, 2.1])],
            flatten_predictions=[1.0, 2.0],
            flatten_actuals=[1.1, 2.1],
            skus=[1, 2],
            avg_loss=0.5,
            mse=0.01,
            mae=0.1,
            flatten_mse=0.01,
            flatten_mae=0.1,
        )

        assert result.avg_loss == 0.5
        assert result.mse == 0.01
        assert result.mae == 0.1
        assert len(result.predictions) == 1
        assert len(result.flatten_predictions) == 2

    def test_with_metrics(self):
        """Test validation result with additional metrics."""
        result = ValidationResult(
            predictions=[],
            actuals=[],
            flatten_predictions=[],
            flatten_actuals=[],
            skus=[],
            avg_loss=0.5,
            mse=0.01,
            mae=0.1,
            flatten_mse=0.01,
            flatten_mae=0.1,
            metrics={"r2": 0.95, "mape": 5.0},
        )

        assert result.metrics["r2"] == 0.95
        assert result.metrics["mape"] == 5.0


class TestEvaluator:
    """Tests for Evaluator class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return MockModel(output_value=1.0)

    @pytest.fixture
    def mock_batch(self):
        """Create a mock batch of data."""
        batch_size = 4
        window = 10
        n_out = 4

        return {
            "qty": torch.randn(batch_size, window, 1),
            "past_time": torch.randn(batch_size, window, 4),
            "future_time": torch.randn(batch_size, n_out, 4),
            "sku": torch.randint(0, 10, (batch_size,)),
            "y": torch.randn(batch_size, n_out),
            "cats": {
                "category": torch.randint(0, 5, (batch_size,)),
            },
        }

    @pytest.fixture
    def evaluator(self, mock_model):
        """Create an evaluator for testing."""
        return Evaluator(
            model=mock_model,
            criterion=nn.MSELoss(),
            batch_size=4,
            total_examples=8,
            flatten_loss=True,
        )

    def test_evaluator_creation(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.batch_size == 4
        assert evaluator.total_examples == 8
        assert evaluator.flatten_loss is True

    def test_evaluator_with_plot_dir(self, mock_model, tmp_path: Path):
        """Test evaluator with plot directory."""
        evaluator = Evaluator(
            model=mock_model,
            criterion=nn.MSELoss(),
            batch_size=4,
            total_examples=8,
            plot_dir=tmp_path / "plots",
        )

        assert evaluator.plot_dir == tmp_path / "plots"

    def test_evaluator_with_metrics(self, mock_model):
        """Test evaluator with custom metrics."""
        from torchmetrics import MeanAbsoluteError

        metrics = {"mae": MeanAbsoluteError()}

        evaluator = Evaluator(
            model=mock_model,
            criterion=nn.MSELoss(),
            batch_size=4,
            total_examples=8,
            metrics=metrics,
        )

        assert "mae" in evaluator.metrics


class TestEvaluatorValidate:
    """Tests for Evaluator.validate method."""

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock round-robin dataloader."""
        batch_size = 4
        n_out = 4

        def mock_batch_generator():
            for i in range(2):  # 2 batches
                yield {
                    "0": {
                        "qty": torch.randn(batch_size, 10, 1),
                        "past_time": torch.randn(batch_size, 10, 4),
                        "future_time": torch.randn(batch_size, n_out, 4),
                        "sku": torch.randint(0, 10, (batch_size,)),
                        "y": torch.randn(batch_size, n_out),
                        "cats": {"category": torch.randint(0, 5, (batch_size,))},
                    }
                }

        return mock_batch_generator()

    def test_validate_returns_result(self, mock_dataloader):
        """Test that validate returns a ValidationResult."""
        model = MockModel()
        evaluator = Evaluator(
            model=model,
            criterion=nn.MSELoss(),
            batch_size=4,
            total_examples=8,
        )

        result = evaluator.validate(mock_dataloader, plot=False)

        assert isinstance(result, ValidationResult)
        assert result.avg_loss >= 0
        assert result.mse >= 0
        assert result.mae >= 0

    def test_validate_stores_predictions(self, mock_dataloader):
        """Test that validate stores predictions and actuals."""
        model = MockModel()
        evaluator = Evaluator(
            model=model,
            criterion=nn.MSELoss(),
            batch_size=4,
            total_examples=8,
        )

        result = evaluator.validate(mock_dataloader, plot=False)

        assert len(result.flatten_predictions) > 0
        assert len(result.flatten_actuals) > 0
        assert len(result.skus) > 0
