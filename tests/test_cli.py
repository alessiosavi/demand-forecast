"""Tests for CLI commands."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from demand_forecast.cli import app

runner = CliRunner()


class TestVersionCommand:
    """Tests for version command."""

    def test_version_output(self):
        """Test version command outputs version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "demand-forecast version" in result.stdout


class TestGenerateDataCommand:
    """Tests for generate-data command."""

    def test_generate_data_creates_file(self, tmp_path: Path):
        """Test that generate-data creates a CSV file."""
        output_file = tmp_path / "test_data.csv"

        result = runner.invoke(
            app,
            [
                "generate-data",
                str(output_file),
                "--products",
                "5",
                "--stores",
                "2",
                "--days",
                "30",
                "--seed",
                "42",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_generate_data_with_defaults(self, tmp_path: Path):
        """Test generate-data with default parameters."""
        output_file = tmp_path / "default_data.csv"

        result = runner.invoke(
            app,
            ["generate-data", str(output_file), "--days", "10", "--products", "2", "--stores", "1"],
        )

        assert result.exit_code == 0
        assert output_file.exists()


class TestTrainCommand:
    """Tests for train command."""

    def test_train_missing_config(self):
        """Test train command fails without config."""
        result = runner.invoke(app, ["train"])
        assert result.exit_code != 0

    def test_train_config_not_found(self, tmp_path: Path):
        """Test train command fails with non-existent config."""
        result = runner.invoke(
            app,
            ["train", "--config", str(tmp_path / "nonexistent.yaml")],
        )
        assert result.exit_code != 0

    @patch("demand_forecast.cli.ForecastPipeline")
    @patch("demand_forecast.cli.Settings")
    def test_train_with_plot_flag(self, mock_settings, mock_pipeline, tmp_path: Path):
        """Test train command with --plot flag."""
        # Create mock config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("data:\n  input_path: test.csv\n")

        mock_settings_instance = MagicMock()
        mock_settings_instance.output.model_dir = tmp_path / "models"
        mock_settings.from_yaml.return_value = mock_settings_instance

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        runner.invoke(
            app,
            ["train", "--config", str(config_path), "--plot"],
        )

        # Check that train was called with plot=True
        mock_pipeline_instance.train.assert_called_once()
        call_kwargs = mock_pipeline_instance.train.call_args
        assert call_kwargs[1]["plot"] is True


class TestEvaluateCommand:
    """Tests for evaluate command."""

    def test_evaluate_missing_args(self):
        """Test evaluate command fails without required args."""
        result = runner.invoke(app, ["evaluate"])
        assert result.exit_code != 0

    @patch("demand_forecast.cli.ForecastPipeline")
    @patch("demand_forecast.cli.Settings")
    def test_evaluate_saves_metrics(self, mock_settings, mock_pipeline, tmp_path: Path):
        """Test evaluate command saves metrics to file."""
        # Create mock config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("data:\n  input_path: test.csv\n")

        mock_settings_instance = MagicMock()
        mock_settings_instance.output.model_dir = tmp_path / "models"
        mock_settings.from_yaml.return_value = mock_settings_instance

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.evaluate.return_value = {
            "mse": 0.1,
            "rmse": 0.316,
            "mae": 0.2,
            "mape": 5.0,
            "r_squared": 0.95,
            "correlation": 0.98,
            "total_samples": 100,
        }
        mock_pipeline.return_value = mock_pipeline_instance

        output_metrics = tmp_path / "metrics.json"
        model_path = tmp_path / "model.pt"
        model_path.touch()
        data_path = tmp_path / "data.csv"
        data_path.touch()

        result = runner.invoke(
            app,
            [
                "evaluate",
                str(model_path),
                str(data_path),
                "--config",
                str(config_path),
                "--output",
                str(output_metrics),
            ],
        )

        assert result.exit_code == 0
        assert output_metrics.exists()

        # Check metrics content
        with open(output_metrics) as f:
            metrics = json.load(f)
        assert "mse" in metrics
        assert "rmse" in metrics

    @patch("demand_forecast.cli.ForecastPipeline")
    @patch("demand_forecast.cli.Settings")
    def test_evaluate_with_plot(self, mock_settings, mock_pipeline, tmp_path: Path):
        """Test evaluate command with --plot flag."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("data:\n  input_path: test.csv\n")

        mock_settings_instance = MagicMock()
        mock_settings_instance.output.model_dir = tmp_path / "models"
        mock_settings.from_yaml.return_value = mock_settings_instance

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.evaluate.return_value = {
            "mse": 0.1,
            "rmse": 0.316,
            "mae": 0.2,
            "mape": 5.0,
            "r_squared": 0.95,
            "correlation": 0.98,
            "total_samples": 100,
        }
        mock_pipeline.return_value = mock_pipeline_instance

        model_path = tmp_path / "model.pt"
        model_path.touch()
        data_path = tmp_path / "data.csv"
        data_path.touch()

        runner.invoke(
            app,
            [
                "evaluate",
                str(model_path),
                str(data_path),
                "--config",
                str(config_path),
                "--plot",
            ],
        )

        # Check that evaluate was called with plot=True
        mock_pipeline_instance.evaluate.assert_called_once()
        call_kwargs = mock_pipeline_instance.evaluate.call_args
        assert call_kwargs[1]["plot"] is True


class TestPredictCommand:
    """Tests for predict command."""

    def test_predict_missing_args(self):
        """Test predict command fails without required args."""
        result = runner.invoke(app, ["predict"])
        assert result.exit_code != 0

    @patch("demand_forecast.cli.ForecastPipeline")
    @patch("demand_forecast.cli.Settings")
    def test_predict_saves_csv(self, mock_settings, mock_pipeline, tmp_path: Path):
        """Test predict command saves predictions to CSV."""
        import pandas as pd

        config_path = tmp_path / "config.yaml"
        config_path.write_text("data:\n  input_path: test.csv\n")

        mock_settings_instance = MagicMock()
        mock_settings_instance.output.model_dir = tmp_path / "models"
        mock_settings.from_yaml.return_value = mock_settings_instance

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.predict.return_value = pd.DataFrame(
            {
                "sku": ["SKU1", "SKU2"],
                "cluster": [0, 1],
                "prediction": [100.0, 200.0],
                "actual": [95.0, 205.0],
                "lower_bound": [90.0, 190.0],
                "upper_bound": [110.0, 210.0],
            }
        )
        mock_pipeline.return_value = mock_pipeline_instance

        model_path = tmp_path / "model.pt"
        model_path.touch()
        data_path = tmp_path / "data.csv"
        data_path.touch()
        output_csv = tmp_path / "predictions.csv"

        result = runner.invoke(
            app,
            [
                "predict",
                str(model_path),
                str(data_path),
                "--config",
                str(config_path),
                "--output",
                str(output_csv),
            ],
        )

        assert result.exit_code == 0
        assert output_csv.exists()

    @patch("demand_forecast.cli.ForecastPipeline")
    @patch("demand_forecast.cli.Settings")
    def test_predict_with_confidence(self, mock_settings, mock_pipeline, tmp_path: Path):
        """Test predict command with custom confidence level."""
        import pandas as pd

        config_path = tmp_path / "config.yaml"
        config_path.write_text("data:\n  input_path: test.csv\n")

        mock_settings_instance = MagicMock()
        mock_settings_instance.output.model_dir = tmp_path / "models"
        mock_settings.from_yaml.return_value = mock_settings_instance

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.predict.return_value = pd.DataFrame(
            {
                "sku": ["SKU1"],
                "prediction": [100.0],
            }
        )
        mock_pipeline.return_value = mock_pipeline_instance

        model_path = tmp_path / "model.pt"
        model_path.touch()
        data_path = tmp_path / "data.csv"
        data_path.touch()
        output_csv = tmp_path / "predictions.csv"

        runner.invoke(
            app,
            [
                "predict",
                str(model_path),
                str(data_path),
                "--config",
                str(config_path),
                "--output",
                str(output_csv),
                "--confidence",
                "0.90",
            ],
        )

        # Check that predict was called with confidence=0.90
        mock_pipeline_instance.predict.assert_called_once()
        call_kwargs = mock_pipeline_instance.predict.call_args
        assert call_kwargs[1]["confidence"] == 0.90

    @patch("demand_forecast.cli.ForecastPipeline")
    @patch("demand_forecast.cli.Settings")
    def test_predict_with_plot(self, mock_settings, mock_pipeline, tmp_path: Path):
        """Test predict command with --plot flag."""
        import pandas as pd

        config_path = tmp_path / "config.yaml"
        config_path.write_text("data:\n  input_path: test.csv\n")

        mock_settings_instance = MagicMock()
        mock_settings_instance.output.model_dir = tmp_path / "models"
        mock_settings.from_yaml.return_value = mock_settings_instance

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.predict.return_value = pd.DataFrame(
            {
                "sku": ["SKU1"],
                "prediction": [100.0],
            }
        )
        mock_pipeline.return_value = mock_pipeline_instance

        model_path = tmp_path / "model.pt"
        model_path.touch()
        data_path = tmp_path / "data.csv"
        data_path.touch()
        output_csv = tmp_path / "predictions.csv"

        runner.invoke(
            app,
            [
                "predict",
                str(model_path),
                str(data_path),
                "--config",
                str(config_path),
                "--output",
                str(output_csv),
                "--plot",
            ],
        )

        # Check that predict was called with plot=True
        mock_pipeline_instance.predict.assert_called_once()
        call_kwargs = mock_pipeline_instance.predict.call_args
        assert call_kwargs[1]["plot"] is True


class TestPreprocessCommand:
    """Tests for preprocess command."""

    def test_preprocess_missing_args(self):
        """Test preprocess command fails without required args."""
        result = runner.invoke(app, ["preprocess"])
        assert result.exit_code != 0
