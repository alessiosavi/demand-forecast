.PHONY: help install install-dev lint format typecheck test test-cov clean build docs \
        docker-build docker-test docker-lint docker-dev docker-clean

PYTHON := python
PIP := pip

# Default target
help:
	@echo "Demand Forecast - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install package"
	@echo "  make install-dev   Install with dev dependencies"
	@echo ""
	@echo "Quality:"
	@echo "  make lint          Run linter (ruff check)"
	@echo "  make format        Format code (ruff format)"
	@echo "  make typecheck     Run type checker (mypy)"
	@echo "  make check         Run all checks (lint + typecheck)"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make test-fast     Run tests in parallel"
	@echo ""
	@echo "Build:"
	@echo "  make build         Build package"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Data:"
	@echo "  make sample-data   Generate sample data"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-test   Run tests in Docker"
	@echo "  make docker-lint   Run linting in Docker"
	@echo "  make docker-dev    Start interactive dev container"
	@echo "  make docker-clean  Remove Docker images"
	@echo ""
	@echo "Docs:"
	@echo "  make docs          Build documentation"

# Installation
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

# Code quality
lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy demand_forecast/

check: lint typecheck

# Testing
test:
	pytest

test-cov:
	pytest --cov=demand_forecast --cov-report=term-missing --cov-report=html

test-fast:
	pytest -n auto

# Build
build: clean
	$(PYTHON) -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Data generation
sample-data:
	demand-forecast generate-data sample_sales.csv --products 50 --stores 10 --days 730

# Documentation
docs:
	@echo "Documentation is in docs/ folder (Markdown format)"
	@echo "For Sphinx docs, run: cd docs && make html"

# Docker targets
docker-build:
	docker build -t demand-forecast:latest .

docker-test:
	docker-compose run --rm test

docker-test-cov:
	docker-compose run --rm test-cov

docker-lint:
	docker-compose run --rm lint

docker-typecheck:
	docker-compose run --rm typecheck

docker-dev:
	docker-compose run --rm dev

docker-clean:
	docker-compose down --rmi local -v
	docker rmi demand-forecast:latest 2>/dev/null || true
