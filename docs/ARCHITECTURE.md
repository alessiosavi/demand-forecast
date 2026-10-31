# Architecture

The current path is long-format data or a fev window → `Panel` → masked sample
batches → shared `ForecastModel` with a `V2Backbone` → trainer → inference bundle
→ `Predictor`. Quantile predictions are returned in original units.

Use these sources for the current contract:

- [Design specification](superpowers/specs/2026-10-30-fev-bench-v2-design.md),
  especially §§4–6 for data, model and training contracts and §8 for migration.
- [Implementation plan](superpowers/plans/2026-10-31-fev-core-v2.md) for the exact
  phase-0–3 scope, defaults and acceptance checks.
- [Settings schema](../demand_forecast/config/settings.py) for the implemented
  configuration, and the [README module inventory](../README.md#package-modules)
  for current package paths.
- [Frozen evaluation protocol](protocol.md) and [Colab workflow](colab.md).

`models/wrapper.py`, `models/transformer.py`, `models/transformer_v2.py`,
`models/lightweight.py`, `core/tuning.py`, and `utils/clustering.py` remain on the
legacy contract until the phase-4 plan. The current v2 backbone is `models/v2.py`;
legacy imports, feature-engineering helpers and loss classes remain available.

The store-specific pipeline and its old architecture walkthrough have been
retired. This page provides migration pointers until the full phase-5 rewrite.
