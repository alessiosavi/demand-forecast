# Development

Start with the [README quick start](../README.md#quick-start) for a working CPU
example and its [development checks](../README.md#development-and-latency) for
tests, linting and formatting. Benchmark tests require the optional `bench` extra.

Use these sources when changing the current implementation:

- The [Settings schema](../demand_forecast/config/settings.py) for configuration
  fields and validation.
- The [design specification](superpowers/specs/2026-10-30-fev-bench-v2-design.md)
  for data, model, training and migration contracts.
- The [phase-0–3 implementation plan](superpowers/plans/2026-10-31-fev-core-v2.md)
  for task boundaries and acceptance checks.

This page replaces the retired pipeline development guide with current pointers;
the full documentation rewrite is phase 5.
