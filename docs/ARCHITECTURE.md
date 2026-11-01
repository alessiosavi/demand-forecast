# Architecture

The current path is long-format data or a fev window → `Panel` → masked sample
batches → shared `ForecastModel` with a `V2Backbone` → trainer → inference bundle
→ `Predictor`. Quantile predictions are returned in original units.

Use these sources for the current contract:

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
