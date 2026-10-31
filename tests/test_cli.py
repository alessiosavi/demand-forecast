"""Tests for CLI commands."""

from pathlib import Path
from unittest.mock import patch

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

    def test_generate_long_data(self, tmp_path: Path):
        from demand_forecast.config.settings import DatasetConfig
        from demand_forecast.data.loader_long import load_long

        output_dir = tmp_path / "panel"
        result = runner.invoke(
            app,
            [
                "generate-data",
                str(output_dir),
                "--format",
                "long",
                "--products",
                "4",
                "--days",
                "30",
                "--freq",
                "W-SUN",
                "--start",
                "2023-01-01",
                "--seed",
                "7",
            ],
        )
        assert result.exit_code == 0, result.output
        assert {path.name for path in output_dir.iterdir()} == {"long.csv", "static.csv"}
        long = pd.read_csv(output_dir / "long.csv")
        static = pd.read_csv(output_dir / "static.csv")
        assert list(long.columns) == ["id", "timestamp", "target", "promo", "oil"]
        assert list(static.columns) == ["id", "cat"]
        panel = load_long(
            DatasetConfig(
                source="long",
                path=output_dir / "long.csv",
                id_columns=["id"],
                timestamp_column="timestamp",
                target="target",
                target_support="nonnegative",
                known_dynamic=["promo"],
                past_dynamic=["oil"],
                static_categorical=["cat"],
                static_path=output_dir / "static.csv",
                freq="W-SUN",
            )
        )
        assert len(panel) == 4
        assert all(series.length == 30 for series in panel.series)

    def test_generate_data_rejects_unknown_format(self, tmp_path: Path):
        output = tmp_path / "output"
        result = runner.invoke(app, ["generate-data", str(output), "--format", "unknown"])
        assert result.exit_code == 2, result.output
        assert "format" in result.output and not output.exists()

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
        assert sales["qty"].ge(0).all()
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
