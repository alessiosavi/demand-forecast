"""Tests for configuration module."""

from pathlib import Path

from demand_forecast.config import (
    DataConfig,
    ModelConfig,
    Settings,
    TimeSeriesConfig,
    TrainingConfig,
)


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self, tmp_path: Path):
        """Test default configuration values."""
        config = DataConfig(input_path=tmp_path / "data.csv")

        assert config.resample_period == "1W"
        assert config.max_zeros_ratio == 0.7
        assert config.date_column == "date"
        assert config.sku_column == "sku"

    def test_custom_values(self, tmp_path: Path):
        """Test custom configuration values."""
        config = DataConfig(
            input_path=tmp_path / "data.csv",
            resample_period="1D",
            max_zeros_ratio=0.5,
        )

        assert config.resample_period == "1D"
        assert config.max_zeros_ratio == 0.5


class TestTimeSeriesConfig:
    """Tests for TimeSeriesConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TimeSeriesConfig()

        assert config.window == 52
        assert config.n_out == 16
        assert config.test_size == 0.2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TimeSeriesConfig(window=26, n_out=8, test_size=0.3)

        assert config.window == 26
        assert config.n_out == 8
        assert config.test_size == 0.3


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()

        assert config.d_model == 256
        assert config.nhead == 8
        assert config.num_encoder_layers == 4

    def test_d_model_divisible_by_nhead(self):
        """Test that d_model is divisible by nhead."""
        config = ModelConfig(d_model=256, nhead=8)
        assert config.d_model % config.nhead == 0


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.num_epochs == 10
        assert config.batch_size == 128
        assert config.learning_rate == 1e-5


class TestSettings:
    """Tests for Settings."""

    def test_from_dict(self, tmp_path: Path):
        """Test creating settings from dictionary."""
        data = {
            "data": {"input_path": str(tmp_path / "data.csv")},
            "seed": 123,
        }
        settings = Settings.model_validate(data)

        assert settings.seed == 123
        assert settings.data.input_path == tmp_path / "data.csv"

    def test_default_device(self, tmp_path: Path):
        """Test default device detection."""
        settings = Settings(
            data=DataConfig(input_path=tmp_path / "data.csv"),
        )

        # Device should be None (auto-detect) by default
        assert settings.device is None

    def test_yaml_roundtrip(self, tmp_path: Path):
        """Test saving and loading from YAML."""
        original = Settings(
            data=DataConfig(input_path=tmp_path / "data.csv"),
            seed=999,
        )

        yaml_path = tmp_path / "config.yaml"
        original.to_yaml(yaml_path)

        loaded = Settings.from_yaml(yaml_path)

        assert loaded.seed == original.seed
        assert loaded.data.resample_period == original.data.resample_period
