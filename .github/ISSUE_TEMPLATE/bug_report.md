---
name: Bug Report
about: Report a bug or unexpected behavior
title: "[BUG] "
labels: bug
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## Environment

- **OS:** (e.g., macOS 14.0, Ubuntu 22.04, Windows 11)
- **Python version:** (output of `python --version`)
- **Package version:** (output of `demand-forecast version`)
- **PyTorch version:** (output of `python -c "import torch; print(torch.__version__)"`)
- **CUDA version (if applicable):** (output of `nvcc --version`)

## Steps to Reproduce

1. Step one
2. Step two
3. Step three

## Minimal Reproducible Example

```python
# Paste minimal code that reproduces the issue
from demand_forecast import ...

```

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

A clear and concise description of what actually happened.

## Error Traceback

```
Paste the full error traceback here
```

## Configuration

If applicable, paste your `config.yaml`:

```yaml
# config.yaml contents
```

## Additional Context

Add any other context about the problem here (screenshots, logs, etc.)

## Checklist

- [ ] I have searched existing issues for duplicates
- [ ] I have included a minimal reproducible example
- [ ] I have included the full error traceback
- [ ] I am using a supported Python version (3.10+)
