# Demand Forecast - Development/Testing Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY demand_forecast/ ./demand_forecast/
COPY tests/ ./tests/
COPY config.example.yaml ./

# Install the package with dev dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip install -e ".[dev]"
RUN --mount=type=cache,target=/root/.cache/pip pip install dill

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command: run tests
CMD ["pytest", "-v", "--tb=short"]
