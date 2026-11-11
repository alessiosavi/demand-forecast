# Dataset wins

This is the protocol fixed on 2026-11-11 before GPU runs. Choose recipes only
on shifted backtests. Never use official test scores to choose an arm, a seed,
or whether to ensemble. Nothing is published by this workflow.

## Fixed protocol

- Use the surviving first-try pretrained bundle:
  `s3://veliu-public-dataset/demand-forecast/pretrain-first-try/pretrain-9238837-20261102T074107Z/pretrained/bundle/`.
  The corpus-v2 bundle is unavailable.
- For an official task with window count `W`, initial cutoff `C`, horizon `H`,
  and step `S` (default `H`), set `W' = min(max(W, 4), 8)` and
  `C' = C - ((W' - 1) * S + H)`. Resolve an omitted `C` using the original
  definition: `C = -H - (W - 1) * S`. Names and other settings stay the same.
  The last backtest horizon ends at the first official cutoff, even when `S < H`.
  Reduce windows explicitly only when history is too short; record infeasibility.
- Run four arms on shifted tasks, using history before the first backtest cutoff:
  Z = zero-shot; A0 = `fev-v2-pretrained-finetune-tuned.yaml`, seed 0;
  A1 = `fev-v2-pretrained-finetune-hygiene.yaml`; A2 = per-dataset `fev tune`,
  limited to 10 trials or 40 minutes, using shifted tasks and inner validation only.
  A1 adds `warmup_frac: 0.1`, `select_min_gain: 0.05`,
  `exception_policy: pretrained`, and `weight_average: tail` to A0.
- Choose the lowest backtest SQL only if it is at most `0.99 * A0`; otherwise
  retain A0. An absent/non-finite arm row is ineligible. Without a finite A0 row,
  selection records `chosen: null`; operationally retain A0 and record why.
  Commit the selection JSON to the private plans directory before official runs.
- For each chosen recipe, run seeds 0, 1, 2 on the official task. Average quantile
  forecasts and re-sort along the quantile axis. This ensemble is mandatory.
  Report single-seed official A0 alongside it as reference only.
- Every final run uses `--policy first_window` and `training.refit: none`.
  Report all 21 targets, wins and losses, against every published non-leaking
  cell in the locally pinned `autogluon/fev@9cf943c` snapshot. First means strictly
  lower than every eligible published cell.
- Stop when this plan consumes $45, or setup cannot complete in two sessions.
  The operator maintains the cost ledger (cap $91, enforce $86 on a new ledger;
  previous ledger closed at $134.917 committed). Training and real-data runs
  belong on the guest, never on the Mac. Ensure shutdown still works if an AWS
  token expires; see the [EC2 runbook](ec2.md).

The 21 targets are:

```text
rohlik_sales_1W
m5_1W
uk_covid_nation_1W/new
hospital_admissions_1D
kdd_cup_2022_1D
rohlik_sales_1D
m5_1M
hermes
hierarchical_sales_1D
restaurant
boomlet_963
hierarchical_sales_1W
favorita_stores_1M
hospital_admissions_1W
boomlet_1062
SZ_TAXI_15T
m5_1D
hospital
favorita_stores_1D
boomlet_1225
boomlet_1282
```

## Derive task files

Run scripts from the repository root with the bench extra installed and
`PYTHONPATH=$PWD`. `--names` accepts comma-separated names or `@file`, with one
name per line (commas also work inside the file). Single-name selections produce
single-dataset YAML. Unknown and duplicate names fail.

```bash
export PYTHONPATH="$PWD"
python scripts/fev_shift_tasks.py --names m5_1W --out /tmp/bt-m5.yaml
python scripts/fev_shift_tasks.py --names m5_1W --mode official --out /tmp/official-m5.yaml
python scripts/fev_shift_tasks.py --names m5_1W --windows m5_1W=2 --out /tmp/bt-short.yaml
```

`--tasks custom.yaml` selects a custom source; the default is the packaged
`fev100` task list. Official mode copies definitions without changing fields and
rejects `--windows`. Keep relative local dataset paths valid from the output
YAML's directory, or use absolute paths in the source.

In shifted mode, tasks with an official timestamp cutoff require both `--cutoff NAME=TIMESTAMP`
(the first shifted cutoff) and `--step NAME=OFFSET` (a positive pandas offset).
Repeat each option for multiple tasks. `--windows NAME=N` overrides the default
window count `W' = min(max(W, 4), 8)`. Under the assumption that the supplied step
spans exactly one horizon, the script requires
`Timestamp(cutoff') + W' * to_offset(step) == Timestamp(official cutoff)` exactly;
a mismatch reports both dates. The operator must verify the step against the
dataset frequency: the spec loader exposes no frequency and the script does not
infer one from a task name.

For Rohlik's eight-week and fourteen-day horizons, respectively:

```bash
python scripts/fev_shift_tasks.py --names rohlik_sales_1W,rohlik_sales_1D \
  --cutoff rohlik_sales_1W=2023-05-05 --step rohlik_sales_1W=56D \
  --cutoff rohlik_sales_1D=2023-10-20 --step rohlik_sales_1D=14D \
  --out /tmp/bt-rohlik.yaml
```

Both tasks use four shifted windows ending at the official 2023-12-15 cutoff.
The YAML retains string cutoffs and steps, which `load_task_specs` and fev 0.10.0
accept with multiple windows. `--cutoff` and `--step` fail for integer-cutoff
tasks and in official mode. Integer-cutoff tasks with offset-string steps still
fail with the task name; use a verified integer step instead. The script never
downloads data or silently reduces window counts.

## Backtest, select, and run the official tasks

The commands below use guest-local bundle and output paths. Run each arm against
the shifted YAML. For Z, supply a config with `training.zero_shot: true`; for A1,
use the hygiene config. Supply the shifted YAML to `fev tune` for A2 and respect
the trial/time limits above.

```bash
demand-forecast fev run --tasks /tmp/bt-m5.yaml \
  --config configs/fev-v2-pretrained-finetune-tuned.yaml \
  --policy first_window --model-name A0 --out-dir runs/bt/A0 \
  --init-from /srv/pretrained/bundle --resume --keep-going
python scripts/fev_wins.py select \
  --arm Z=runs/bt/Z --arm A0=runs/bt/A0 --arm A1=runs/bt/A1 --arm A2=runs/bt/A2 \
  --reference A0 --min-gain 0.01 --out runs/selection.json --names m5_1W
```

Each arm accepts a run directory or CSV and must contain at most one row per task.
Missing task rows within an existing arm file are ineligible; nonexistent files
are input errors. `selection.json` records scores, decisions and reasons, and
`selection.csv` is written beside it. These scripts trust the supplied backtest
paths; the operator must not supply official rows to selection.

Create three copies of the chosen config with top-level `seed: 0`, `seed: 1`,
and `seed: 2`. Run that recipe on `/tmp/official-m5.yaml` into separate
`runs/official/seed0`, `seed1`, and `seed2` directories, then average:

```bash
python scripts/fev_ensemble_runs.py \
  --runs runs/official/seed0 runs/official/seed1 runs/official/seed2 \
  --out runs/official/ensemble --model-name demand-forecast-ensemble
```

Ensemble inputs require identical task definitions, quantile levels, window
counts, series IDs in the same order, and array shapes. Output must be new or
empty. The script writes averaged `window_<i>.npz`, provenance in `ensemble.json`,
and `summaries.csv` via the runner's official fev scorer. Scoring needs the
original datasets, available locally or through fev's configured dataset source;
the run directories do not contain truth labels. One monotone input run reproduces
its `test_error` exactly. Training and inference times remain unknown for the new
entry; input timings are not presented as ensemble measurements.

## Standing

Use a local directory containing the published model CSVs from the pinned
snapshot. The script performs no download and does not verify the snapshot's
Git revision. Keep a separate manifest with the revision when preparing it.

```bash
python scripts/fev_wins.py standing --entry runs/official/ensemble \
  --published /srv/fev-9cf943c/results --out-dir runs/standing --names @targets.txt
```

Repeat `--entry` for disjoint per-dataset runs; overlapping task rows fail so
official scores cannot silently choose among entries. Run a separate standing
command for the A0 reference. Without `--names`, report the union of our and
published task names; pass the 21-target file to retain every intended target.

`standing.csv`, `standing.json`, and `standing.md` contain our SQL, seasonal-naive
SQL, best published model/SQL, `gap_percent = 100 * (ours / best - 1)`, rank,
strict `first`, and the number of published cells within 5% of the best.
Negative gaps mean improvement. Ranks use `1 + count(published SQL < ours)`;
ties share rank but are not strict firsts. Missing/non-finite cells do not rank;
tasks with no comparison retain a reason. A zero best with positive ours has
an undefined percentage gap, recorded as null. Totals count strict firsts, top 3,
and top 5 across comparable tasks. Missing published tasks are evaluated using
the cells available for that task. Leak flags must be valid; flagged cells are
excluded from all published comparisons unless `--include-leaking` is explicit.
All scripts return 0 on success and 2 on input errors.
