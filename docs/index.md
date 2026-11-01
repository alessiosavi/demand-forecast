# Demand Forecast

Demand Forecast trains quantile forecasts for long-format panels of time series.
The current v2 pipeline combines explicit data roles, masked context/future
inputs, an encoder-decoder Transformer, portable inference bundles and optional
fev evaluation. Predictions contain ordered quantiles in original units and a
median point forecast.

## What the current pipeline supports

- **Heterogeneous panels:** categorical statics and series IDs have learned
  embeddings; numeric statics and dynamic covariates use numeric projections.
  Known-future inputs are separate from measurements available only in the past.
- **Missing and short histories:** observation and padding masks retain the
  difference between a missing value and a zero. Empirical-context or saved-level
  fallbacks keep every input series in the prediction output.
- **Quantile training:** full-past scale normalization, doubled pinball loss,
  inner SQL model selection and a monotone quantile head. Nonnegative target
  support clamps forecasts at zero; signed targets are also supported.
- **Reusable runs:** bundles preserve weights and fitted preprocessing;
  identity-checked checkpoints support training resume. The fev runner trains
  once per task, forecasts each item at its own origin and saves completed
  prediction windows and official evaluation summaries.

Only `v2` is accepted by the current model configuration. Standard, advanced,
TCN and mixer models remain in a separate legacy registry; they are not yet on
this trainer, bundle or fev path. Clustering, tuning, export and DHF are not
integrated into this pipeline. Full refit and per-window training are rejected.
The [architecture guide](ARCHITECTURE.md) explains the family boundaries, data
flow, masks, covariate guards and artifacts.

## Get started

From a repository checkout, use Python 3.10 or newer:

```bash
pip install -e ".[dev]"
demand-forecast --help
demand-forecast version
```

Follow the [complete CPU quick start](https://github.com/alessiosavi/demand-forecast#quick-start)
to generate a small weekly panel, save its settings, train and forecast.
The commands are `train --config … --out …` and
`forecast --bundle … --config … --origin … --out …`.
See [configuration](CONFIGURATION.md) for long-format roles and
the [CLI command table](ARCHITECTURE.md#command-line-interface) for every command.

## Evaluate with fev

```bash
pip install -e ".[dev,bench]"
demand-forecast fev --help
```

The benchmark extra pins fev 0.10.0 and statsforecast 2.0.1. Seven bundled weekly
tasks use the same v2 pipeline with task-specific roles, horizons and quantiles.
`fev run` takes a settings file through `--config`; `fev baseline` runs seasonal
naive, and `fev leaderboard` compares saved summaries.

Use the [frozen protocol](protocol.md), [Colab workflow](colab.md) and
[EC2 runbook](ec2.md) for benchmark execution. The
[architecture guide](ARCHITECTURE.md#fev-orchestration-identity-and-resume)
describes per-item origins, resume behavior and timing diagnostics.

## Documentation

- [Quick start guide](QUICKSTART.md)
- [Configuration](CONFIGURATION.md)
- [Architecture and data flow](ARCHITECTURE.md)
- [API reference](API.md)
- [Development](DEVELOPMENT.md)
- [Package module inventory](https://github.com/alessiosavi/demand-forecast#package-modules)

MIT License; see the [license](https://github.com/alessiosavi/demand-forecast/blob/main/LICENSE)
and [third-party notices](https://github.com/alessiosavi/demand-forecast/blob/main/THIRD_PARTY_NOTICES.md).
