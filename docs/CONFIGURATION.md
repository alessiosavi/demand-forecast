# Configuration

The [Settings schema](../demand_forecast/config/settings.py) is the source of
truth for current YAML fields and validation. Its top-level sections are `data`,
`forecast`, `model`, `training`, and `tuning`, with `seed`, `device`, and
`log_level`. `data` and `forecast` are required; other sections have defaults.

Start with the [tiny CPU example](../README.md#quick-start). Declare the ID
columns, target and `target_support` (`nonnegative` or `real`), then the forecast
horizon. Known-future, past-only and static columns have explicit roles. Paths
resolve from the working directory. The CLI takes output paths through `--out`
or `--out-dir`.

For semantics and planned extensions, see the
[design specification](superpowers/specs/2026-10-30-fev-bench-v2-design.md)
(§4 for data, §5 for models, §6 for training) and the
[phase-0–3 implementation plan](superpowers/plans/2026-10-31-fev-core-v2.md).
`model.family: v2`, `training.refit: none`, and fev `--policy first_window` are
implemented. Full refitting, other families, tuning and clustering integration
are deferred to phase 4; the retained `tuning` schema does not enable them yet.

`models/wrapper.py`, `models/transformer.py`, `models/transformer_v2.py`,
`models/lightweight.py`, `core/tuning.py`, and `utils/clustering.py` remain on the
legacy contract until the phase-4 plan. Old hackathon examples and their
`timeseries`/`output` sections do not describe the current `Settings` API.

This page replaces the old configuration walkthrough with current pointers;
the full documentation rewrite is phase 5.
