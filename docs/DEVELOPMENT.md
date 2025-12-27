# Development Guide

Guide for contributing to and developing the demand forecasting system.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) CUDA-compatible GPU

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/demand-forecast.git
cd demand-forecast

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Installation

```bash
# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy demand_forecast/

# Check CLI works
demand-forecast --help
```

## Project Structure

```text
demand-forecast/
├── demand_forecast/           # Main package
│   ├── __init__.py           # Package init, logging config
│   ├── __main__.py           # python -m entry point
│   ├── cli.py                # Typer CLI commands
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py       # Pydantic configuration
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py     # Custom exceptions
│   │   ├── pipeline.py       # Main orchestration
│   │   ├── trainer.py        # Training loop
│   │   └── evaluator.py      # Evaluation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py         # Data loading
│   │   ├── preprocessor.py   # Preprocessing
│   │   ├── feature_engineering.py
│   │   ├── dataset.py        # PyTorch Dataset
│   │   └── dataloader.py     # DataLoader factory
│   ├── models/
│   │   ├── __init__.py
│   │   ├── components.py     # PositionalEncoding
│   │   ├── transformer.py    # Main model
│   │   ├── wrapper.py        # Multi-model wrapper
│   │   └── losses.py         # Loss functions
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py      # Prediction wrapper
│   │   └── confidence.py     # CI calculation
│   ├── synthetic/
│   │   ├── __init__.py
│   │   └── generator.py      # Data generation
│   └── utils/
│       ├── __init__.py
│       ├── clustering.py
│       ├── memory.py
│       ├── metrics.py
│       ├── outliers.py
│       ├── time_features.py
│       ├── timeseries.py
│       └── visualization.py
├── tests/                     # Test suite
│   ├── conftest.py           # Pytest fixtures
│   ├── test_config.py
│   ├── test_data/
│   ├── test_models/
│   └── test_utils/
├── docs/                      # Documentation
├── config.example.yaml        # Example configuration
├── pyproject.toml            # Project metadata
├── Makefile                  # Dev commands
└── README.md
```

## Code Style

### Formatting

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Type Hints

All functions should have type hints:

```python
def create_timeseries(
    X: NDArray[np.float32],
    cat: NDArray[np.float32],
    y: NDArray[np.float32],
    window: int = 10,
    n_out: int = 1,
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Create sliding window sequences."""
    ...
```

Run type checking:

```bash
mypy demand_forecast/
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_confidence_intervals(
    predictions: np.ndarray,
    actuals: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate confidence intervals for predictions.

    Uses the standard deviation of prediction errors to compute
    confidence bands around predictions.

    Args:
        predictions: Predicted values.
        actuals: Actual values.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bounds, upper_bounds).

    Raises:
        ValueError: If predictions and actuals have different shapes.

    Example:
        >>> preds = np.array([1.0, 2.0, 3.0])
        >>> actuals = np.array([1.1, 1.9, 3.2])
        >>> lower, upper = calculate_confidence_intervals(preds, actuals)
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=demand_forecast --cov-report=term-missing

# Run specific test file
pytest tests/test_utils/test_clustering.py

# Run specific test
pytest tests/test_utils/test_clustering.py::test_find_best_k

# Run with verbose output
pytest -v

# Run in parallel
pytest -n auto
```

### Writing Tests

Tests live in the `tests/` directory, mirroring the package structure:

```python
# tests/test_utils/test_outliers.py
import pytest
import pandas as pd
from demand_forecast.utils.outliers import remove_outliers


class TestRemoveOutliers:
    def test_no_outliers(self):
        """Test with data that has no outliers."""
        data = pd.Series([1, 2, 3, 4, 5])
        clipped, has_outliers = remove_outliers(data, n=3)

        assert not has_outliers
        assert clipped.equals(data)

    def test_with_outliers(self):
        """Test with data containing outliers."""
        data = pd.Series([1, 2, 3, 100, 4, 5])
        clipped, has_outliers = remove_outliers(data, n=2)

        assert has_outliers
        assert clipped.max() < 100

    def test_empty_series(self):
        """Test with empty series."""
        data = pd.Series([], dtype=float)
        clipped, has_outliers = remove_outliers(data)

        assert not has_outliers
```

### Test Fixtures

Common fixtures are defined in `tests/conftest.py`:

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_sales_data():
    """Generate sample sales data for testing."""
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    return pd.DataFrame({
        "date": dates,
        "sku": ["SKU001"] * 100 + ["SKU002"] * 100,
        "store_id": ["store_1"] * 200,
        "qty": np.random.randint(0, 100, 200),
        "category": ["shoes"] * 200,
    })

@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    from demand_forecast.config import Settings, DataConfig
    return Settings(
        data=DataConfig(input_path=Path("test.csv")),
    )
```

## Adding New Features

### 1. Create a New Utility

```python
# demand_forecast/utils/new_feature.py
"""New feature utilities."""

import logging
from typing import List

logger = logging.getLogger(__name__)


def my_new_function(data: List[float], param: int = 10) -> float:
    """Compute something useful.

    Args:
        data: Input data.
        param: Some parameter.

    Returns:
        Computed result.
    """
    logger.info(f"Processing {len(data)} items with param={param}")
    # Implementation
    return result
```

### 2. Export in `__init__.py`

```python
# demand_forecast/utils/__init__.py
from demand_forecast.utils.new_feature import my_new_function

__all__ = [
    # ... existing exports
    "my_new_function",
]
```

### 3. Write Tests

```python
# tests/test_utils/test_new_feature.py
import pytest
from demand_forecast.utils import my_new_function


class TestMyNewFunction:
    def test_basic_usage(self):
        result = my_new_function([1, 2, 3])
        assert result > 0

    def test_with_param(self):
        result = my_new_function([1, 2, 3], param=20)
        assert result > 0
```

### 4. Add CLI Command (if applicable)

```python
# demand_forecast/cli.py
@app.command()
def new_command(
    input_file: Path = typer.Argument(...),
    param: int = typer.Option(10, "--param", "-p"),
):
    """Description of new command."""
    from demand_forecast.utils import my_new_function

    # Implementation
    result = my_new_function(data, param)
    typer.echo(f"Result: {result}")
```

## Debugging

### Enable Debug Logging

```python
from demand_forecast import configure_logging
configure_logging("DEBUG")
```

Or via CLI:

```bash
demand-forecast train --config config.yaml --verbose
```

### Using Debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use debugpy for VS Code
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def my_function():
    # Function to profile
    pass
```

## Common Development Tasks

### Update Dependencies

```bash
# Update pyproject.toml, then:
pip install -e ".[dev]"
```

### Generate Documentation

```bash
# If using Sphinx
cd docs
make html
```

### Release Checklist

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite: `pytest`
4. Run linting: `ruff check .`
5. Run type checking: `mypy demand_forecast/`
6. Create git tag: `git tag v1.0.0`
7. Push with tags: `git push --tags`

## Architecture Decisions

### Why Pydantic for Configuration?

- Automatic validation with helpful error messages
- Type coercion (string "42" → int 42)
- YAML/JSON serialization built-in
- IDE autocomplete support

### Why Separate Model per Cluster?

- Different demand patterns need different parameters
- Improves accuracy for heterogeneous catalogs
- Allows different learning rates per cluster

### Why RoundRobin Iterator?

- Balances training across clusters
- Prevents overfitting to larger clusters
- Ensures all clusters get equal training attention

### Why Not Global State?

- Testability: functions can be tested in isolation
- Thread safety: no shared mutable state
- Serialization: all state is explicit and saveable

## Getting Help

- Check existing issues on GitHub
- Read the [Architecture Guide](ARCHITECTURE.md)
- Review the [API Reference](API.md)
- Open a new issue for bugs or feature requests
