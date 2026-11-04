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
implemented. Full refitting, other model families, tuning and clustering integration are not implemented in the shared pipeline; the retained `tuning` schema does not enable them.

`models/wrapper.py`, `models/transformer.py`, `models/transformer_v2.py`,
`models/lightweight.py`, `core/tuning.py`, and `utils/clustering.py` remain on the
legacy contract and are not integrated into the shared pipeline. Old hackathon examples and their
`timeseries`/`output` sections do not describe the current `Settings` API.

Use the linked README, Settings schema and frozen protocol for current commands and contracts.

## Pretraining corpus configuration

Pretraining uses its own `PretrainConfig`. The existing
`configs/pretrain-v2-first-try.yaml` retains its legacy shard lists, hold-outs,
`kernel_synth_every` schedule and run identity. Optional `corpus.sources` maps
registered kind names to `{shards: [...], holdout: [...], weight: ...}` for
deterministic weighted mixing. It must include `tsmixup`; weights must be finite
and positive and hold-outs disjoint from training shards. Omit the legacy
controls in weighted configs; conflicting specifications are rejected.

The separate `configs/corpus-v2.yaml` build recipe pins source revisions and
records include/exclude lists, attribution, filters and shard limits. It loads
offline and rejects unknown fields. `local_path` on a build source enables
offline conversion from a saved snapshot. See [Corpus v2](corpus.md) for build
verify and content-audit commands, a weighted pretraining example, reproducibility limits,
and the research-only licence position.

`demand-forecast pretrain corpus audit --root /data/corpus --out /data/audit
--tasks fev100 --workers 4 --resume` compares all numeric benchmark sequences
against `chronos_real,gifteval_pretrain` by default. `--tasks` also accepts
`weekly`, `heldout`, or a task YAML with optional per-task `local_path` parquet
overrides (relative to the YAML). No model role profiles are needed. `--kinds`
selects a comma-separated list of corpus kinds. `--controls` runs the offline
40 matcher controls and ten rule self-tests without a corpus. Constant overrides
are restricted to `--controls --control-constants <json>`; production audits
always use the documented run-11 defaults. See [Audit](corpus.md#audit) for the
copy definition, declaration output, limits and resource requirements.
