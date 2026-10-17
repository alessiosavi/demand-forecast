"""Tests for CLI commands."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
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


class TestGenerateSalesCommand:
    """Exercise plain/gzipped catalogues and the store's exact CSV contract."""

    @pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
    def test_generate_sales(self, tmp_path, suffix):
        products = pd.DataFrame(
            {
                "id": ["catalogue:001", "catalogue:002"],
                "category": ["outerwear", "clothing"],
                "subcategory": ["jacket", "shorts"],
                "price": [100.0, 20.0],
                "embedding": ["unused huge payload", "unused huge payload"],
            }
        )
        product_file = tmp_path / f"products{suffix}"
        products.to_csv(product_file, index=False)
        output = tmp_path / "data" / "sales.csv"
        with patch("pandas.read_csv", wraps=pd.read_csv) as read_csv:
            result = runner.invoke(
                app,
                [
                    "generate-sales",
                    "--products",
                    str(product_file),
                    "--output",
                    str(output),
                    "--start",
                    "2024-10-17",
                    "--end",
                    "2024-11-16",
                    "--seed",
                    "7",
                    "--promo-share",
                    "0",
                    "-v",
                ],
            )
        assert result.exit_code == 0, result.output
        assert set(read_csv.call_args.kwargs["usecols"]) == {
            "id",
            "price",
            "category",
            "subcategory",
        }
        sales = pd.read_csv(output)
        assert list(sales.columns) == ["date", "product_id", "qty", "unit_price"]
        assert sales["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
        assert sales["date"].between("2024-10-17", "2024-11-16").all()
        assert sales["qty"].ge(1).all()
        expected_prices = sales["product_id"].map(products.set_index("id")["price"])
        pd.testing.assert_series_equal(sales["unit_price"], expected_prices, check_names=False)
        assert all(
            len(line.rsplit(",", 1)[1].split(".")[1]) == 2
            for line in output.read_text().splitlines()[1:]
        )

    def test_generate_sales_defaults(self, tmp_path):
        catalogue = tmp_path / "products.csv"
        catalogue.write_text("id,category,subcategory,price\nreal:001,clothing,shorts,25.00\n")
        output = tmp_path / "sales.csv"
        result = runner.invoke(
            app, ["generate-sales", "--products", str(catalogue), "--output", str(output)]
        )
        assert result.exit_code == 0, result.output
        sales = pd.read_csv(output)
        assert sales["date"].min() == "2024-10-17"
        assert sales["date"].max() == "2026-10-16"

    @pytest.mark.parametrize("arguments", [["--start", "2026-10-17"], ["--promo-share", "1.1"]])
    def test_invalid_generate_sales_arguments(self, tmp_path, arguments):
        catalogue = tmp_path / "products.csv"
        catalogue.write_text("id,category,subcategory,price\nreal:001,clothing,shorts,25.00\n")
        output = tmp_path / "sales.csv"
        result = runner.invoke(
            app,
            ["generate-sales", "--products", str(catalogue), "--output", str(output), *arguments],
        )
        assert result.exit_code != 0
        assert not output.exists()


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

    @patch("demand_forecast.core.pipeline.ForecastPipeline")
    @patch("demand_forecast.config.settings.Settings")
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

    @patch("demand_forecast.core.pipeline.ForecastPipeline")
    @patch("demand_forecast.config.settings.Settings")
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

    @patch("demand_forecast.core.pipeline.ForecastPipeline")
    @patch("demand_forecast.config.settings.Settings")
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

    @patch("demand_forecast.core.pipeline.ForecastPipeline")
    @patch("demand_forecast.config.settings.Settings")
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

    @patch("demand_forecast.core.pipeline.ForecastPipeline")
    @patch("demand_forecast.config.settings.Settings")
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

    @patch("demand_forecast.core.pipeline.ForecastPipeline")
    @patch("demand_forecast.config.settings.Settings")
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
