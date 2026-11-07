# Kaggle forecasting-competitions bench

The Kaggle bench runs the frozen demand-forecast models against Kaggle
forecasting competitions with adapter-only wrappers: each competition's data is
converted into the existing long/static input, and the model architecture,
training recipe and frozen artifacts are never changed for a competition.
`store-sales-time-series-forecasting` is the reference implementation; every
other competition adds a compact converter module on the same runner
(`demand_forecast/kaggle/runner.py`).

## Prerequisites

- **CLI**: `kaggle` CLI 2.2.5 or later, authenticated (`kaggle competitions
list` must work; API token at `~/.kaggle/kaggle.json`). The CLI cannot accept
  competition rules.
- **Join clicks (owner)**: the owner must click **Join** on each competition's
  page before its data downloads or a submission succeeds. This is a browser
  action; the runner fails loudly until it is done.
- **GPU quota**: kernel runs use T4 ×2 (free tier: ≈30 GPU-h/week floating,
  ≈14.6 GiB usable per card, 12 h max session). Every run budgets against the
  weekly quota; the runner pushes with the default 6 h timeout (`-t 21600`).
- **Pushed commit**: the kernel driver pip-installs
  `git+https://github.com/alessiosavi/demand-forecast@<sha>`; the commit the
  operator script pins (the repo's local `HEAD`) must be pushed to GitHub
  first.
- **Private staging**: prepared competition data is uploaded only as a private
  Kaggle dataset (`alessiosavi/kb-<slug>`); the pretrained bundle rides in
  `alessiosavi/kb-assets` as `bundle-L3-first-try/` and is identity-checked in
  the kernel. Competition data and derived artifacts are never published.
- **Evidence**: runs live under
  `.superpowers/sdd/2026-11-07-kaggle-forecasting-competitions/runs/`, which is
  never committed.

## Stages

Run everything through the operator script (defaults to store-sales):

```bash
# what would run, without touching the network or Kaggle
.venv/bin/python scripts/kaggle/run.py --dry-run --stage all

# full sequence: download → prepare → upload → kernel → fetch → score → submit
.venv/bin/python scripts/kaggle/run.py --stage all

# everything except the leaderboard submission
.venv/bin/python scripts/kaggle/run.py --stage all-until-submit

# a single stage (e.g. resume: just fetch after the kernel finished)
.venv/bin/python scripts/kaggle/run.py --stage fetch
```

| stage      | what it does                                                                                                                                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `download` | `kaggle competitions download -c <slug> -p raw/ --unzip` into the run dir                                                                                                                                                 |
| `prepare`  | converts `raw/` with the competition module into `prepared/` (`official/` and `mirror/` variants, settings, `prepare.json`)                                                                                               |
| `upload`   | writes `dataset-metadata.json` and creates the private prepared dataset, or versions it when it already exists (always with a changelog message)                                                                          |
| `kernel`   | writes the kernel dir (metadata + generated driver), pushes `alessiosavi/kb-<slug>-run`, then polls `kaggle kernels status` until `COMPLETE`; `ERROR`, `CANCEL*`, CLI failures and unparseable status output abort loudly |
| `fetch`    | clears `fetched/`, downloads the kernel output tree (`kaggle kernels output`) and requires the `workload.json` completeness marker                                                                                        |
| `score`    | scores the mirror window (`scores.json`), assembles `submission.csv` from the fetched official fine-tune predictions and validates it against the prepared `sample_submission.csv`, then writes `REPORT.md`               |
| `submit`   | re-validates `submission.csv` against the sample, submits once — `-m "demand-forecast L3"` — and records the `kaggle competitions submissions` listing and submission ref; it never retries                               |

`all-until-submit` runs the first six stages and stops before the submission,
so the candidate can be inspected first.

## Evidence layout

```
runs/<slug>/
  raw/               downloaded competition files (never published)
  prepared/          official/ + mirror/ converter output, settings, prepare.json
  kernel/            pushable kernel dir (kernel-metadata.json + generated main.py)
  fetched/           kernel output tree; workload.json is the completeness marker
  scores.json        mirror-window scores (baseline + zero-shot + fine-tune)
  submission.csv     validated candidate in the competition's sample order
  submissions.txt    recorded `kaggle competitions submissions` listing
  REPORT.md          mirror scores (mirror (non-official)), corpus_overlap: pending,
                     and the submission ref once recorded
```

The fetched kernel output tree is anchored by `workload.json`, written at the
workload's out root; per-run `run.json` files exist even for failed or partial
trees and are not a completeness signal. A fetched tree without `workload.json`
raises `RuntimeError("no kernel output")`.

## Rules

- **No tuning, no model changes.** One protocol pass per competition: the same
  standard trio (seasonal-naive baseline, pretrained L3 zero-shot, fine-tuned
  L3) with the pinned recipe; adapters only, and evaluation windows never drive
  fitting, recipe, or selection.
- **No reshuffling submissions.** Submission slots are limited (daily and
  total). The runner submits the validated headline (fine-tune) candidate
  once; a failed submit raises instead of retrying, and a submission is never
  re-uploaded to chase placement.
- **Mirror numbers stay labelled.** Local scores come from the mirror window
  cut out of permitted history and are always reported as
  `mirror (non-official)`; the official number comes from the leaderboard.
  `corpus_overlap: pending` flips only after the corpus-overlap audit for that
  competition runs.
- **Late submissions are probed, not assumed.** For closed competitions the
  late-submission status is verified at the first submission attempt and
  recorded in that competition's report either way.
