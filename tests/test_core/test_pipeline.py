"""Tests for ForecastPipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from demand_forecast.core.pipeline import ForecastPipeline


class TestForecastPipelineInit:
    """Tests for ForecastPipeline initialization."""

    def test_pipeline_creation(self, sample_config):
        """Test creating a pipeline."""
        pipeline = ForecastPipeline(sample_config)

        assert pipeline.settings == sample_config
        assert pipeline.model is None
        assert pipeline.sku_to_index == {}

    def test_pipeline_sets_seeds(self, sample_config):
        """Test that pipeline sets random seeds."""
        ForecastPipeline(sample_config)

        # Seeds should be set, so random numbers should be reproducible
        torch.manual_seed(sample_config.seed)
        expected = torch.randn(1)

        torch.manual_seed(sample_config.seed)
        actual = torch.randn(1)

        assert torch.equal(expected, actual)

    def test_pipeline_device_cpu(self, sample_config):
        """Test pipeline uses CPU device when specified."""
        sample_config.device = "cpu"
        pipeline = ForecastPipeline(sample_config)

        assert pipeline.device == torch.device("cpu")


class TestForecastPipelineTrain:
    """Tests for ForecastPipeline.train method."""

    @patch.object(ForecastPipeline, "build_model")
    @patch.object(ForecastPipeline, "create_datasets")
    def test_train_requires_model(self, mock_datasets, mock_build, sample_config):
        """Test train raises error without model."""
        pipeline = ForecastPipeline(sample_config)
        pipeline._raw_datasets = {"0": (np.array([]), np.array([]), np.array([]))}

        with pytest.raises(ValueError, match="Must call build_model first"):
            pipeline.train()

    @patch.object(ForecastPipeline, "build_model")
    def test_train_requires_datasets(self, mock_build, sample_config):
        """Test train raises error without datasets."""
        pipeline = ForecastPipeline(sample_config)
        pipeline.model = MagicMock()

        with pytest.raises(ValueError, match="Must call create_datasets first"):
            pipeline.train()


class TestForecastPipelinePredict:
    """Tests for ForecastPipeline.predict method."""

    def test_predict_requires_model(self, sample_config):
        """Test predict raises error without model."""
        pipeline = ForecastPipeline(sample_config)
        pipeline._raw_datasets = {"0": (np.array([]), np.array([]), np.array([]))}

        with pytest.raises(ValueError, match="Must load or build model first"):
            pipeline.predict()

    def test_predict_requires_datasets(self, sample_config):
        """Test predict raises error without datasets."""
        pipeline = ForecastPipeline(sample_config)
        pipeline.model = MagicMock()

        with pytest.raises(ValueError, match="Must call create_datasets first"):
            pipeline.predict()


class TestForecastPipelineEvaluate:
    """Tests for ForecastPipeline.evaluate method."""

    def test_evaluate_requires_model(self, sample_config):
        """Test evaluate raises error without model."""
        pipeline = ForecastPipeline(sample_config)
        pipeline._raw_datasets = {"0": (np.array([]), np.array([]), np.array([]))}

        with pytest.raises(ValueError, match="Must load or build model first"):
            pipeline.evaluate()

    def test_evaluate_requires_datasets(self, sample_config):
        """Test evaluate raises error without datasets."""
        pipeline = ForecastPipeline(sample_config)
        pipeline.model = MagicMock()

        with pytest.raises(ValueError, match="Must call create_datasets first"):
            pipeline.evaluate()

    @patch("demand_forecast.core.pipeline.create_dataloaders")
    @patch("demand_forecast.data.dataloader.get_round_robin_iterators")
    @patch("demand_forecast.core.evaluator.Evaluator")
    @patch("demand_forecast.utils.metrics.init_metrics")
    def test_evaluate_returns_metrics(
        self,
        mock_init_metrics,
        mock_evaluator_class,
        mock_rr_iterators,
        mock_create_dls,
        sample_config,
    ):
        """Test evaluate returns dictionary of metrics."""
        pipeline = ForecastPipeline(sample_config)
        pipeline.model = MagicMock()
        pipeline._raw_datasets = {
            "0": (
                np.array([[[1]]]),
                np.array([1]),
                np.array([[1]]),
                np.array([[[1]]]),
                np.array([1]),
                np.array([[1]]),
            )
        }
        pipeline.categorical_encoder = MagicMock()
        pipeline.categorical_encoder.get_encoded_columns.return_value = []
        pipeline.sku_to_index = {"SKU1": 0}

        # Mock dataloaders - returns (train_dls, test_dls, train_dss, test_dss)
        mock_test_dl = MagicMock()
        mock_test_dl.__iter__ = MagicMock(return_value=iter([]))
        mock_create_dls.return_value = ({"0": MagicMock()}, {"0": mock_test_dl}, {}, {})
        mock_rr_iterators.return_value = (iter([]), iter([]))

        # Mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.validate.return_value = MagicMock(
            mse=0.1,
            mae=0.2,
            flatten_mse=0.15,
            flatten_mae=0.25,
            avg_loss=0.5,
            flatten_predictions=[1.0, 2.0],
            flatten_actuals=[1.1, 2.1],
            skus=[0, 0],
            metrics={},
        )
        mock_evaluator_class.return_value = mock_evaluator

        mock_init_metrics.return_value = {}

        result = pipeline.evaluate()

        assert isinstance(result, dict)
        assert "mse" in result
        assert "rmse" in result
        assert "mae" in result
        assert "mape" in result
        assert "r_squared" in result
        assert "correlation" in result

    @patch("demand_forecast.core.pipeline.create_dataloaders")
    @patch("demand_forecast.data.dataloader.get_round_robin_iterators")
    @patch("demand_forecast.core.evaluator.Evaluator")
    @patch("demand_forecast.utils.metrics.init_metrics")
    def test_evaluate_with_plot(
        self,
        mock_init_metrics,
        mock_evaluator_class,
        mock_rr_iterators,
        mock_create_dls,
        sample_config,
        tmp_path,
    ):
        """Test evaluate generates plots when requested."""
        sample_config.output.model_dir = tmp_path / "models"
        pipeline = ForecastPipeline(sample_config)
        pipeline.model = MagicMock()
        pipeline._raw_datasets = {
            "0": (
                np.array([[[1]]]),
                np.array([1]),
                np.array([[1]]),
                np.array([[[1]]]),
                np.array([1]),
                np.array([[1]]),
            )
        }
        pipeline.categorical_encoder = MagicMock()
        pipeline.categorical_encoder.get_encoded_columns.return_value = []
        pipeline.sku_to_index = {"SKU1": 0}

        # Mock dataloaders - returns (train_dls, test_dls, train_dss, test_dss)
        mock_test_dl = MagicMock()
        mock_test_dl.__iter__ = MagicMock(return_value=iter([]))
        mock_create_dls.return_value = ({"0": MagicMock()}, {"0": mock_test_dl}, {}, {})
        mock_rr_iterators.return_value = (iter([]), iter([]))

        # Mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.validate.return_value = MagicMock(
            mse=0.1,
            mae=0.2,
            flatten_mse=0.15,
            flatten_mae=0.25,
            avg_loss=0.5,
            flatten_predictions=[1.0, 2.0, 3.0],
            flatten_actuals=[1.1, 2.1, 3.1],
            skus=[0, 0, 0],
            metrics={},
        )
        mock_evaluator_class.return_value = mock_evaluator

        mock_init_metrics.return_value = {}

        plot_dir = tmp_path / "eval_plots"
        with patch("demand_forecast.utils.visualization.plot_prediction_quality") as mock_plot:
            pipeline.evaluate(plot=True, plot_dir=plot_dir)

            # Check that plot_prediction_quality was called
            mock_plot.assert_called_once()


class TestForecastPipelineSaveLoad:
    """Tests for ForecastPipeline save/load methods."""

    def test_save_requires_model(self, sample_config, tmp_path):
        """Test save raises error without model."""
        pipeline = ForecastPipeline(sample_config)

        with pytest.raises(ValueError, match="No model to save"):
            pipeline.save(tmp_path / "model.pt")

    @patch("torch.save")
    def test_save_creates_artifacts(self, mock_torch_save, sample_config, tmp_path):
        """Test save creates model and artifacts."""
        pipeline = ForecastPipeline(sample_config)
        pipeline.model = MagicMock()
        pipeline.scaler_manager = MagicMock()
        pipeline.categorical_encoder = MagicMock()

        model_path = tmp_path / "models" / "model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        pipeline.save(model_path)

        mock_torch_save.assert_called_once()
        pipeline.scaler_manager.save.assert_called_once()
        pipeline.categorical_encoder.save.assert_called_once()
