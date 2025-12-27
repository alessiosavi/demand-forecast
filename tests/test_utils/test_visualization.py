"""Tests for visualization utilities."""

from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast.utils.visualization import (
    plot_prediction_quality,
    plot_training_history,
    plot_validation_results,
    save_prediction_report,
)


class TestPlotPredictionQuality:
    """Tests for plot_prediction_quality function."""

    def test_basic_plot(self, tmp_path: Path):
        """Test basic plotting without errors."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.array([1.1, 2.2, 2.9, 4.1, 5.2])

        save_path = tmp_path / "quality.png"
        plot_prediction_quality(
            actuals=actuals,
            predictions=predictions,
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()

    def test_with_skus(self, tmp_path: Path):
        """Test plotting with SKU labels."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
        skus = ["SKU1", "SKU2", "SKU3", "SKU4", "SKU5"]

        save_path = tmp_path / "quality_skus.png"
        plot_prediction_quality(
            actuals=actuals,
            predictions=predictions,
            skus=skus,
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()

    def test_large_dataset(self, tmp_path: Path):
        """Test plotting with large dataset (should sample)."""
        np.random.seed(42)
        actuals = np.random.randn(500)
        predictions = actuals + np.random.randn(500) * 0.1

        save_path = tmp_path / "quality_large.png"
        plot_prediction_quality(
            actuals=actuals,
            predictions=predictions,
            max_samples=100,
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()

    def test_custom_title(self, tmp_path: Path):
        """Test plotting with custom title."""
        actuals = np.array([1.0, 2.0, 3.0])
        predictions = np.array([1.1, 2.2, 2.9])

        save_path = tmp_path / "quality_custom.png"
        plot_prediction_quality(
            actuals=actuals,
            predictions=predictions,
            title="Custom Title Test",
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()


class TestPlotValidationResults:
    """Tests for plot_validation_results function."""

    def test_basic_plot(self, tmp_path: Path):
        """Test basic validation plotting."""
        actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
        predictions = [1.1, 2.2, 2.9, 4.1, 5.2]

        save_path = tmp_path / "validation.png"
        plot_validation_results(
            flatten_actuals=actuals,
            flatten_predictions=predictions,
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()

    def test_with_title(self, tmp_path: Path):
        """Test validation plot with custom title."""
        actuals = [1.0, 2.0, 3.0]
        predictions = [1.1, 2.2, 2.9]

        save_path = tmp_path / "validation_titled.png"
        plot_validation_results(
            flatten_actuals=actuals,
            flatten_predictions=predictions,
            title="Epoch 5 Validation",
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()


class TestPlotTrainingHistory:
    """Tests for plot_training_history function."""

    def test_train_only(self, tmp_path: Path):
        """Test plotting with only training losses."""
        train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]

        save_path = tmp_path / "history_train.png"
        plot_training_history(
            train_losses=train_losses,
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()

    def test_train_and_val(self, tmp_path: Path):
        """Test plotting with both training and validation losses."""
        train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
        val_losses = [1.1, 0.9, 0.7, 0.6, 0.5]

        save_path = tmp_path / "history_both.png"
        plot_training_history(
            train_losses=train_losses,
            val_losses=val_losses,
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()


class TestSavePredictionReport:
    """Tests for save_prediction_report function."""

    def test_basic_report(self, tmp_path: Path):
        """Test generating basic prediction report."""
        predictions_df = pd.DataFrame(
            {
                "sku": ["SKU1", "SKU2", "SKU3", "SKU4", "SKU5"],
                "cluster": [0, 0, 1, 1, 2],
                "prediction": [100.0, 150.0, 200.0, 250.0, 300.0],
                "lower_bound": [90.0, 140.0, 190.0, 240.0, 290.0],
                "upper_bound": [110.0, 160.0, 210.0, 260.0, 310.0],
            }
        )

        output_dir = tmp_path / "report"
        saved_files = save_prediction_report(
            predictions_df=predictions_df,
            output_dir=output_dir,
        )

        assert output_dir.exists()
        assert "summary" in saved_files
        assert saved_files["summary"].exists()
        assert "distribution" in saved_files
        assert saved_files["distribution"].exists()

    def test_report_without_confidence(self, tmp_path: Path):
        """Test report without confidence intervals."""
        predictions_df = pd.DataFrame(
            {
                "sku": ["SKU1", "SKU2", "SKU3"],
                "prediction": [100.0, 150.0, 200.0],
            }
        )

        output_dir = tmp_path / "report_no_ci"
        saved_files = save_prediction_report(
            predictions_df=predictions_df,
            output_dir=output_dir,
        )

        assert output_dir.exists()
        assert "summary" in saved_files
        # Should not have confidence intervals plot
        assert "confidence_intervals" not in saved_files

    def test_custom_prefix(self, tmp_path: Path):
        """Test report with custom file prefix."""
        predictions_df = pd.DataFrame(
            {
                "sku": ["SKU1"],
                "prediction": [100.0],
            }
        )

        output_dir = tmp_path / "report_custom"
        saved_files = save_prediction_report(
            predictions_df=predictions_df,
            output_dir=output_dir,
            prefix="test_pred",
        )

        # Check that files use the custom prefix
        assert any("test_pred" in str(path) for path in saved_files.values())

    def test_summary_content(self, tmp_path: Path):
        """Test that summary contains expected statistics."""
        predictions_df = pd.DataFrame(
            {
                "sku": ["SKU1", "SKU2", "SKU3"],
                "cluster": [0, 1, 2],
                "prediction": [100.0, 200.0, 300.0],
            }
        )

        output_dir = tmp_path / "report_summary"
        saved_files = save_prediction_report(
            predictions_df=predictions_df,
            output_dir=output_dir,
        )

        # Read and check summary content
        summary_content = saved_files["summary"].read_text()
        assert "total_skus: 3" in summary_content
        assert "mean_prediction" in summary_content
        assert "num_clusters: 3" in summary_content
