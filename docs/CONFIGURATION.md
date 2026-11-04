# Configuration

The [Settings schema](https://github.com/alessiosavi/demand-forecast/blob/main/demand_forecast/config/settings.py) is the source of
truth for current YAML fields and validation. Its top-level sections are `data`,
`forecast`, `model`, `training`, and `tuning`, with `seed`, `device`, and
`log_level`. `data` and `forecast` are required; other sections have defaults.

Start with the [tiny CPU example](https://github.com/alessiosavi/demand-forecast/blob/main/README.md#quick-start). Declare the ID
columns, target and `target_support` (`nonnegative` or `real`), then the forecast
horizon. Known-future, past-only and static columns have explicit roles. Paths
resolve from the working directory. The CLI takes output paths through `--out`
or `--out-dir`.

The [frozen protocol](protocol.md) defines the benchmark configuration and the
data, model, training and evaluation contracts.
`model.family: v2`, `training.refit: none`, and fev `--policy first_window` are
the official defaults. Full refitting and per-window training are available as
ablation options. Other model families and legacy tuning/clustering integration
remain outside the shared pipeline; the retained `tuning` schema does not enable them.

`models/wrapper.py`, `models/transformer.py`, `models/transformer_v2.py`,
`models/lightweight.py`, `core/tuning.py`, and `utils/clustering.py` remain on the
legacy contract and are not integrated into the shared pipeline. Old hackathon examples and their
`timeseries`/`output` sections do not describe the current `Settings` API.

Use the linked README, Settings schema and frozen protocol for current commands and contracts.

## Fine-tune ablations

| Setting | Default | Effect |
| --- | --- | --- |
| `training.refit` | `none` | `full` repeats training from the same initial weights and sampler seed on history through the grid step before the forecast origin, for exactly the selected `best_step`. It retains the selection run's LR schedule length and warmup, with no validation or selection in the refit phase. |
| `training.trainable` | `all` | `heads_only` freezes every backbone parameter. Input projections, static embeddings/gates and the quantile head remain trainable; the optimizer contains only trainable parameters. |
| `training.series_embedding` | `true` | `false` omits the `__series__` vocabulary and embedding. Task categorical and numeric statics are retained. Bundles with either form load normally, including tasks without any statics. |
| `forecast.scale_floor_frac` | `0.0` | A fraction in `[0, 1)`. When positive, eligible target scales are at least this fraction times the mean absolute observed value in the sample's cropped context, in addition to `1e-6`. Training, validation and inference share the transform. Zero/undefined seasonal scales retain their existing eligibility and fallback behavior. |
| CLI `fev run --policy` | `first_window` | `per_window` trains/selects a separate model on each window's own available past, using the same settings; optional refit applies to each window. |

Refit is skipped when selection chooses step zero (`best_step_zero`) or there
is no inner validation (`no_inner_validation`, since that path already trains
on all permitted history). `zero_shot: true` with `refit: full` is rejected.
Refit retains the selection encoder and model dimensions so initialization is
identical; series first observed in the added history use the unknown ID code.
The saved bundle uses the refit model and full training cutoff. Selection scores,
`best_step`, validation history and stop reason still describe selection.

`run.json` records `refit` (`mode`, `performed`, `steps`, `train_upto`, `reason`),
`trainable` (`mode`, `params_trainable`, `params_total`), `policy`, and
`validation_history`. Cumulative timings sum both training phases; peak memory
is their maximum. Selection telemetry remains unchanged.
`floored_target_samples` counts samples whose positive relative floor raised the
scale beyond the existing floor, in cumulative training diagnostics and prediction
window diagnostics. Cached validation batches use the same floor but do not add
to the training-sample counter. Optional prediction counters are unknown in old
archives that lack them; summaries sum the available window counters.

Full refit persists `checkpoints/selection.json` before starting and uses
`checkpoints/refit/{latest.pt,progress.json}`. Resume continues that phase without
rerunning selection. `--stop-after` counts optimizer-step attempts across selection
and refit in the current training call, including a checkpointed pause at the phase
boundary. As before, non-finite losses and AMP skips can prevent an actual update.

With `per_window`, training artifacts live at `<out>/<task>/window_<w>/` and
predictions at `<out>/<task>/window_<w>.npz`. Each archive carries its own window
identity. The summary identity is SHA256 of the compact JSON list of ordered
window identities (`json.dumps(ids, separators=(",", ":")).encode("utf-8")`).
`training_time_s` sums persisted train/evaluation/checkpoint durations over all
windows and both phases. Session/non-finite/scaler-skip counts sum across windows;
`resolved_max_steps`, `n_fit_ids` and `n_val_items` describe the last window.
The summary `policy` column distinguishes the two policies; older model summaries
without that column load as `first_window`. Each window uses its own latest origin
for training and earliest item origin for the bundle's forecast floor, with fev's
item-specific truncation unchanged.

Defaults for `trainable`, `series_embedding` and `scale_floor_frac` are omitted
from canonical identity settings; only `per_window` adds `policy` to the identity
payload. `refit` was already an identity field. Existing default run identities
and numerical frames therefore remain compatible.

`configs/fev-v2-pretrained-finetune-tuned.yaml` is the tuned recipe from fev tune
trial 11 (2026-11-03) for the post-hoc confirmation sweep. Supply the pretrained
bundle with `--init-from`; it leaves the ablation switches at their defaults.
These options and the tuned recipe do not replace the frozen official row.
