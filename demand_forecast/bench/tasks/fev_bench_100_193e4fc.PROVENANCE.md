# fev-bench 100-task snapshot

- Source: <https://raw.githubusercontent.com/autogluon/fev/193e4fc/benchmarks/fev_bench/tasks.yaml>
- fev commit: `193e4fcdbafe37bd5f676a00db65cf02ca1f737e`
- SHA-256: `c7160f61a5e1ded66a3954ef1c514d55d13be18534b34fca817356312a6520a9`
- File size: 48,443 bytes; fetched 2026-11-04, preserved byte-identically.
- Dataset repository: `autogluon/fev_datasets`
- Pinned dataset revision: `d3e7e720f4158eab03ebcb021f1cce779cd92acc`

The YAML has 100 tasks and 96 distinct dataset/config pairs. Four configs have
two tasks each: `uk_covid_nation_1D` and `uk_covid_nation_1W` each have `/new`
and `/cumulative` target groups; `fred_md_2025` and `fred_qd_2025` each have
`/cee` and `/macro` target groups. Preserve these explicit `task_name` values
in declarations. Other names default to `dataset_config`.

The upstream target field is `target`, either a string or a list of columns;
`known_dynamic_columns` and `past_dynamic_columns` identify dynamic roles.
`id_column` and `timestamp_column` are omitted here and default to `id` and
`timestamp`. Horizon, windows, metrics and quantiles remain in the unchanged
YAML; a content audit uses the full train split, independent of evaluation
cutoffs and model role profiles. Hub metadata at the pinned revision confirms
numeric sequence features (including multivariate targets) and timestamp
sequences, for example `ETT_15T`: `id`, `timestamp`, HUFL/HULL/MUFL/MULL/LUFL/LULL/OT.
