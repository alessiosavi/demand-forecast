# Zero-shot reference models on fev

Run these models on the GPU benchmark host. Do not run real datasets on the Mac.
The runner consumes native fev YAML, including shifted definitions, multiple
simultaneous targets, and names such as `uk_covid_nation_1W/new`. No training role
profile is needed. `--names @names.txt` or `--names name1,name2` selects tasks.

## Separate host environment

For an NVIDIA L4 host with an NVIDIA driver supporting CUDA 12.8 (check
`nvidia-smi`), use Python 3.12 and the official CUDA 12.8 PyTorch wheel:

```bash
python3.12 -m venv ~/venvs/yardsticks
source ~/venvs/yardsticks/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-yardsticks.txt
python -m pip install -e '.[bench]'
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
python -m pip freeze > yardsticks-environment.txt
```

Use the `cu126` wheel index instead if the installed driver supports CUDA 12.6
but not 12.8. The wheel supplies its CUDA runtime; match the driver's capability,
not a separately installed CUDA toolkit. Wheel commands are from
[PyTorch's versioned installation instructions](https://pytorch.org/get-started/previous-versions/#v280).
Requirements pin the inference packages and numerical stack; `pip freeze` records
remaining transitive dependencies. Keep this environment separate from our model.

```bash
python scripts/fev_yardstick.py --model chronos2 --weights amazon/chronos-2 \
  --revision 29ec3766d36d6f73f0696f85560a422f50e8498c \
  --tasks shifted.yaml --model-name chronos2 --out-dir results/chronos2 --resume --keep-going
python scripts/fev_yardstick.py --model timestral --weights mistralai/Timestral-5M \
  --revision bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f \
  --tasks shifted.yaml --model-name timestral5m --out-dir results/timestral5m --resume --keep-going
```

## Published settings and ablations

Settings follow the public fev adapters pinned at
[fev@9cf943c](https://github.com/autogluon/fev/tree/9cf943c388e5cfaf352175ea3c69e408b197d997/models).
Chronos uses `chronos-forecasting==2.2.2`; Timestral uses
[mistralai/timestral@a534bdb](https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff).
Both use float32, joint targets, median point predictions and the nine task
quantiles directly from their native quantile heads. Chronos uses its published
quantile API; Timestral selects exact levels and uses linear interpolation only
for non-native levels, as its published script does. Neither uses static columns.

| Switch | Chronos-2 | Timestral-5M |
|---|---|---|
| default context | checkpoint cap: 8192 for pinned weights | checkpoint cap: 32768 for pinned weights |
| `--max-context N` | runner retains last N target and dynamic history values; passes N to pipeline context limit, capped by checkpoint | runner retains last N history values; forecaster's rolling cap becomes min(N, checkpoint cap) |
| `--covariates all` (default) | past-only and known history plus known future; nonnumeric/bool dynamics cast to strings as upstream | numeric, boolean and string dynamics; dataset-wide type/vocabulary inference; categorical codes target-encoded by the forecaster using only current context (joint multivariate targets retain codes) |
| `--covariates none` | targets only, no covariate history or future | targets only; skips type/vocabulary inference |
| `--cross-series on` (default) | cross-learning within each pipeline batch, including items, targets and covariates | joint targets within each item; separate batch items do not attend to each other |
| `--cross-series off` | splits each item into single-target groups and sets `cross_learning=False`; each target retains its own conditioning covariates | rejected: this ablation is registered for Chronos only |
| `--batch-size N` | variate budget, including covariates; raised to fit one group; default 100 | item limit, default 1024; length-sorted batches across all windows, budget 2^23 rows × padded steps; halves batches on OOM |

Future covariates keep the full forecast horizon. History trimming never shortens
the requested horizon. Context ablations bound information available to fixed
pretrained weights; they do not simulate retraining on shorter histories.
Timestral retains its forecaster defaults: symmetric averaging, nonnegative
clipping, block autoregression and forecast caps. Chronos retains its pipeline's
normalization, category encoding and long-horizon decoding defaults.

## Evidence and resume

`summaries.csv` uses fev's summary schema plus `task`, the adapter name, every CLI
inference setting, resolved weights revision and run identity. Training time is
zero and the trained-on-this-dataset flag is false (zero-shot use, not a new
claim about pretraining contamination). `inference_time_s` covers adapter
conversion and inference, excluding model loading, data loading and scoring.
Timestral batches windows together, so each NPZ receives an equal share of the
task's measured inference time; it is not a separately measured window latency.

Each task has `run.json` and `window_N.npz`: item IDs, target names, quantiles,
original-unit predictions `(items, targets, horizon, quantiles)`, identity and
time. Item and target order is fev's order. Failures, including fail-fast errors,
are recorded in `failures.json`; successful retries clear them. `--keep-going`
returns exit 1 if any selected task remains incomplete. Use `fev_report.py
--model-name LABEL` to verify identities, file hashes, coverage and recomputed
fev scores. Yardstick audits do not require synthetic training/bundle artifacts
or a baseline. `fev_wins.py standing` accepts these summaries directly. Isolated
per-task runs can be joined with `fev_merge_runs.py`, including names with `/`.

One output directory holds one yardstick configuration. Resume checks weights,
all switches, package versions, task definition, dataset fingerprint and window
hashes. Changed configurations are refused: choose a new output directory.
Missing/damaged window files trigger inference again; a missing/stale summary is
rebuilt from intact recorded windows. A model is never loaded for complete tasks.
Local weight directories are content-hashed; Hub aliases resolve to commit SHAs.
The default IDs above use the pinned SHAs even when `--revision` is omitted.

## Reproduction limits

The published Chronos CSV records fev 0.8.0, and Timestral records 0.11.0.dev0;
our harness deliberately uses fev 0.10.0. Their CSVs do not record weight SHAs.
These defaults reproduce the public adapter settings; numerical parity remains
unverified until the preregistered five-task host check passes. Tests use only
synthetic local data and recording fakes; they download no model weights.

The optional Simons adapter is deferred. Its public fev wrapper additionally
requires the Simons inference package, grouped targets (at most 32), covariate
limits (16 known-future columns), category handling and an independently pinned
checkpoint/API contract with synthetic tests.
