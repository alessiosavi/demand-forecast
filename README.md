# Demand Forecast

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Demand Forecast provides quantile forecasting for long-format time-series panels:
explicit data roles, frequency-aware sampling, a shared `v2` model and trainer,
portable bundles, and optional fev evaluation. Predictions include every input
series, with fallbacks for missing context or unusable scale, ordered quantiles
in original units, and a median point forecast.

The current pipeline supports **v2 only**. Standard, advanced, TCN and mixer
models remain in a separate legacy registry and are not yet integrated with the
current trainer, bundle or fev path.

## How it works

Long-format tables or fev windows become a `Panel` with declared target,
known-future, past-only and static roles. An inner validation split determines
the permitted training history; fitted static vocabularies and masked sample
batches feed `ForecastModel` and its `V2Backbone`. Training uses doubled pinball
loss and selects weights by inner SQL. A saved bundle carries the model and
preprocessing to `Predictor` for inference.

Categorical statics and series IDs have learned embeddings, including unknown
and missing slots. Numeric statics and continuous covariates are projected with
their masks. Full-past target scaling, guarded covariate normalization and a
bidirectional known-future decoder support heterogeneous inputs. A monotone
quantile head keeps knots ordered, and nonnegative target support clamps
forecasts at zero. The [architecture guide](docs/ARCHITECTURE.md) includes the
data-flow diagram, artifact table, fallback rules and per-item-origin semantics.

## Quick start

Install from this checkout with Python 3.10+ and generate a small weekly panel:

```bash
pip install -e ".[dev]"
demand-forecast generate-data /tmp/dfc-example --format long \
  --products 6 --days 60 --freq W-SUN --start 2022-01-02 --seed 0
```

Save this as `/tmp/dfc-example/config.yaml`. This is a tiny CPU smoke configuration
with a 16-wide model and at most 30 optimizer steps.

```yaml
data:
  source: long
  path: /tmp/dfc-example/long.csv
  static_path: /tmp/dfc-example/static.csv
  id_columns: [id]
  timestamp_column: timestamp
  target: target
  target_support: nonnegative
  known_dynamic: [promo]
  past_dynamic: [oil]
  static_categorical: [cat]
  freq: W-SUN
forecast:
  horizon: 4
  max_context: 64
model:
  family: v2
  d_model: 16
  nhead: 2
  num_encoder_layers: 1
  num_decoder_layers: 1
  dim_feedforward: 32
  hidden_continuous_size: 16
  cat_emb_dim: 4
training:
  base_steps: 30
  batch_size: 8
  warmup_steps: 2
  amp: false
seed: 3
device: cpu
```

Train, then forecast from the first timestamp after the generated history:

```bash
demand-forecast train --config /tmp/dfc-example/config.yaml --out /tmp/dfc-example/run
demand-forecast forecast --bundle /tmp/dfc-example/run/bundle \
  --config /tmp/dfc-example/config.yaml --origin 2023-02-26 \
  --out /tmp/dfc-example/forecasts.csv
demand-forecast version
```

This training example writes `bundle/`, `checkpoints/`, and `run.json`. Add `--resume` to the
same training command to reuse a matching run; changed settings or inputs fail
identity checks. Forecast CSVs contain `series_id`, `origin`, `timestamp`,
`q0.1` through `q0.9`, and `point` (the median, `q0.5`). Supply known covariates
for future timestamps in the long CSV when available; missing values are masked.
Set `target_support: real` for signed targets. Paths in YAML resolve from the
working directory.

## fev evaluation

Task definitions can be read without the benchmark dependencies. Running the
benchmark commands requires the optional extra:

```bash
pip install -e ".[dev,bench]"
demand-forecast fev --help
```

For the seven bundled weekly tasks, use the [frozen protocol](docs/protocol.md),
[Colab workflow](docs/colab.md) or [EC2 runbook](docs/ec2.md). The checked-in
[standard configuration](configs/fev-v2-standard.yaml) defines the model and
training controls; the runner derives data roles, horizon and quantiles from
each task. On the benchmark host:

```bash
demand-forecast fev baseline --tasks weekly --out-dir fev-results
demand-forecast fev run --tasks weekly --config configs/fev-v2-standard.yaml \
  --policy first_window --out-dir fev-results --resume
demand-forecast fev leaderboard fev-results/summaries.csv
```

`--tasks` also accepts comma-separated task names or a local task YAML. Baselines
skip completed tasks; model runs with `--resume` reuse matching bundles and saved
windows. Each task trains on its first window, then forecasts later windows at
each item's own origin. Identity checks cover configuration, data and the
resolved training origin. Training progress, checkpoints, window predictions and
summaries retain timing and diagnostic information. `training.refit: full`
and `--policy per_window` are currently unimplemented and rejected.

## Commands

| Command | Purpose |
| --- | --- |
| `train --config ... --out ... [--resume]` | Train and save a bundle |
| `forecast --bundle ... --config ... --origin ... --out ...` | Export quantile forecasts |
| `fev baseline` / `fev run` / `fev leaderboard` | Evaluate and compare fev summaries |
| `generate-data` | Generate synthetic sales or long-format panels |
| `generate-sales` | Generate sales from a product catalogue |
| `version` | Print the installed package version |

Use `demand-forecast --help` or a command's `--help` for options. The
[complete CLI table](docs/ARCHITECTURE.md#command-line-interface) lists required
and optional arguments. Tuning, clustering, export and DHF are not integrated
into the current pipeline.

## Package modules

The current implementation is organized as follows (paths are under `demand_forecast/`):

| Modules | Responsibility |
| --- | --- |
| `cli.py`, `config/settings.py` | Commands and validated settings |
| `data/panel.py`, `data/frequency.py` | Ragged series, roles, grids and calendar features |
| `data/loader_long.py`, `data/loader_fev.py` | Long-format and fev input adapters |
| `data/encoders.py`, `data/scaling.py`, `data/sampler.py` | Static encoding, sample normalization and masked batches |
| `models/base.py`, `models/heads.py`, `models/v2.py`, `models/registry.py` | Shared model contract, monotone quantiles and the v2 backbone |
| `models/components.py`, `models/losses.py` | Building blocks and losses, including retained legacy classes |
| `core/validation.py`, `core/training.py`, `core/evaluation.py`, `metrics.py` | Inner validation, training and quantile metrics |
| `core/run.py`, `core/bundle.py`, `core/exceptions.py` | Run orchestration, persistence and errors |
| `inference/predictor.py` | Bundle-based forecasts in original units |
| `bench/specs.py`, `bench/baselines.py`, `bench/run.py` | Optional fev task definitions, baseline and runner |
| `synthetic/generator.py` | Synthetic panels and sales |
| `utils/checkpoint.py`, `utils/device.py`, `utils/memory.py`, `utils/outliers.py`, `utils/timeseries.py`, `utils/visualization.py` | Shared utilities |

`models/wrapper.py`, `models/transformer.py`, `models/transformer_v2.py`,
`models/lightweight.py`, `core/tuning.py`, and `utils/clustering.py` remain on the
legacy contract. Their imports and tests are retained;
`data/feature_engineering.py` and legacy loss classes also remain available.
The current `v2` CLI uses `models/v2.py`.

## Development and latency

```bash
python -m pytest -q --no-cov
ruff check .
ruff format --check .
python scripts/latency.py --bundle /tmp/dfc-example/run/bundle
```

CI runs the full suite both with `.[dev,bench]` and in a fresh `.[dev]` environment,
where bench tests skip. It also installs a built wheel outside the checkout and
checks CLI help and the packaged weekly task definitions without the bench extra.

The latency script generates 512 synthetic series matching the bundle's roles
and frequency, prepares one batch, performs five warmups and averages 20
`model.predict` calls on one CPU thread. It prints ms/batch and ms/series; loading
and batch preparation are excluded. Override `--n-series` or `--repeats` as needed.
Tiny-bundle timings are smoke measurements, not standard-model benchmark results.

## Documentation

- [Documentation home](docs/index.md)
- [Architecture and data flow](docs/ARCHITECTURE.md)
- [Configuration](docs/CONFIGURATION.md)
- [API reference](docs/API.md)
- [Frozen evaluation protocol](docs/protocol.md)
- [Settings schema](demand_forecast/config/settings.py)

MIT License; see [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
