# Contributing to Demand Forecast

Thank you for your interest in contributing to Demand Forecast! This document provides guidelines and information for contributors.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Python version and OS
   - Package version (`demand-forecast version`)
   - Minimal reproducible example
   - Expected vs actual behavior
   - Full error traceback

### Suggesting Features

1. Check existing issues and discussions
2. Use the feature request template
3. Describe the use case and proposed solution
4. Consider backward compatibility

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Write/update tests
5. Update documentation if needed
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/demand-forecast.git
cd demand-forecast

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify setup
pytest
ruff check .
mypy demand_forecast/
```

## Coding Standards

### Style Guide

- Follow PEP 8
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Maximum line length: 100 characters
- Use descriptive variable names

### Type Hints

All public functions must have type hints:

```python
def process_data(
    data: pd.DataFrame,
    window: int = 52,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process data for training."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(arg1: str, arg2: int = 10) -> bool:
    """Short description of function.

    Longer description if needed, explaining the purpose
    and behavior of the function.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When arg1 is empty.

    Example:
        >>> my_function("hello", 20)
        True
    """
```

### Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use pytest fixtures for shared setup
- Test edge cases and error conditions

```bash
# Run tests with coverage
pytest --cov=demand_forecast --cov-report=term-missing

# Run specific tests
pytest tests/test_utils/test_clustering.py -v
```

## Pull Request Process

### Before Submitting

1. **Run all checks:**

   ```bash
   ruff check .
   ruff format .
   mypy demand_forecast/
   pytest
   ```

2. **Update documentation** if you changed:
   - Public API
   - Configuration options
   - CLI commands

3. **Add changelog entry** for user-facing changes

### PR Requirements

- Clear title describing the change
- Reference any related issues
- Include tests for new functionality
- Pass all CI checks
- One approval from maintainer

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code change that doesn't fix a bug or add a feature
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:

```
feat(models): add attention dropout parameter
fix(data): handle empty categorical columns
docs(readme): update installation instructions
```

## Project Structure

```
demand_forecast/
├── config/         # Configuration (Pydantic models)
├── core/           # Orchestration, training, evaluation
├── data/           # Data loading and preprocessing
├── models/         # Neural network models
├── inference/      # Prediction and confidence intervals
├── synthetic/      # Synthetic data generation
└── utils/          # Utility functions
```

### Adding New Modules

1. Create module in appropriate directory
2. Add exports to `__init__.py`
3. Write comprehensive tests
4. Update API documentation

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release commit
4. Tag release: `git tag v1.0.0`
5. Push with tags
6. GitHub Actions publishes to PyPI

## Getting Help

- Open a GitHub issue for bugs or features
- Check [Development Guide](docs/DEVELOPMENT.md) for setup help
- Review [Architecture Guide](docs/ARCHITECTURE.md) for design questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in:

- GitHub contributors list
- CHANGELOG.md for significant contributions
- README acknowledgments for major features

Thank you for contributing!
