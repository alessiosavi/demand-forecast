"""Pinned fev task definitions with explicit target support and static roles.

Reading specs needs only the base dependencies; constructing a fev task needs
the benchmark extra. Omitted task settings follow fev 0.10.0 defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml

FEV_DATASETS_REVISION = "d3e7e720f4158eab03ebcb021f1cce779cd92acc"


@dataclass(frozen=True)
class FevDatasetTask:
    """Dataset identity and column roles for content audits; no model profiles."""

    name: str
    dataset_path: str
    dataset_config: str | None = None
    target: str | list[str] = "target"
    known_dynamic_columns: list[str] = field(default_factory=list)
    past_dynamic_columns: list[str] = field(default_factory=list)
    id_column: str = "id"
    timestamp_column: str = "timestamp"
    local_path: str | None = None
    freq: str | None = None

    @property
    def targets(self) -> list[str]:
        return [self.target] if isinstance(self.target, str) else self.target


def fev_bench_tasks_path(selector: str | Path = "fev100"):
    """Locate the byte-identical upstream YAML or an explicitly selected subset."""
    filenames = {
        "fev100": "fev_bench_100_193e4fc.yaml",
        "weekly": "fev_bench_weekly.yaml",
        "heldout": "fev_bench_weekly_heldout.yaml",
    }
    if str(selector) in filenames:
        return files("demand_forecast.bench").joinpath("tasks", filenames[str(selector)])
    return Path(selector)


def fev_bench_tasks(selector: str | Path = "fev100") -> list[FevDatasetTask]:
    """Read all 100 upstream tasks, a subset, or local YAML without role profiles.

    Upstream calls the target field ``target`` (including multivariate lists).
    A local_path override points to a parquet file, relative to custom YAML.
    Explicit task names distinguish different targets of the same dataset.
    """
    path = fev_bench_tasks_path(selector)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
        raise ValueError("task YAML must contain a tasks list")
    tasks = []
    for definition in data["tasks"]:
        name = definition.get("name") or definition.get("task_name")
        name = name or definition.get("dataset_config") or Path(definition["dataset_path"]).stem
        values = {
            key: definition[key]
            for key in FevDatasetTask.__dataclass_fields__
            if key != "name" and key in definition
        }
        local = values.get("local_path") or _local_path(values["dataset_path"])
        if local is not None:
            local = Path(local).expanduser()
            if not local.is_absolute():
                local = Path(str(path)).parent / local
            values["local_path"] = str(local.resolve())
        tasks.append(FevDatasetTask(name=name, **values))
    if len({task.name for task in tasks}) != len(tasks):
        raise ValueError("duplicate task names in audit YAML")
    if not tasks:
        raise ValueError("task YAML must not be empty")
    return tasks


@dataclass
class FevTaskSpec:
    """A task and its data roles, independent of optional fev imports."""

    name: str
    dataset_path: str | Path
    dataset_config: str | None = None
    horizon: int = 1
    num_windows: int = 1
    seasonality: int = 1
    quantile_levels: list[float] = field(default_factory=list)
    target: str | list[str] = "target"
    known_dynamic: list[str] = field(default_factory=list)
    past_dynamic: list[str] = field(default_factory=list)
    static_columns: list[str] = field(default_factory=list)
    categorical_statics: list[str] = field(default_factory=list)
    target_support: Literal["nonnegative", "real"] | dict[str, str] = field(kw_only=True)
    initial_cutoff: int | str | None = None
    min_context_length: int = 1
    eval_metric: str | dict[str, Any] = "MASE"
    extra_metrics: list[str | dict[str, Any]] = field(default_factory=list)
    window_step_size: int | str | None = None
    id_column: str = "id"
    timestamp_column: str = "timestamp"

    def __post_init__(self):
        if isinstance(self.target, list) and (
            not self.target or len(set(self.target)) != len(self.target)
        ):
            raise ValueError("target list must be nonempty and unique")
        supports = self.target_support
        if isinstance(supports, dict):
            targets = [self.target] if isinstance(self.target, str) else self.target
            if set(supports) != set(targets):
                raise ValueError("target_support mapping must cover exactly the target columns")
            supports = list(supports.values())
        else:
            supports = [supports]
        if not supports or any(value not in ("nonnegative", "real") for value in supports):
            raise ValueError("target_support must be 'nonnegative' or 'real'")
        if not set(self.categorical_statics) <= set(self.static_columns):
            raise ValueError("categorical_statics must be a subset of static_columns")

    @property
    def targets(self) -> list[str]:
        return [self.target] if isinstance(self.target, str) else self.target

    def task_definition(self) -> dict:
        """Return the evaluation boundaries included in a model run's identity."""
        return {
            "name": self.name,
            "horizon": self.horizon,
            "num_windows": self.num_windows,
            "initial_cutoff": self.initial_cutoff,
            "window_step_size": self.window_step_size,
            "min_context_length": self.min_context_length,
            "seasonality": self.seasonality,
            "quantile_levels": list(self.quantile_levels),
            **({"targets": list(self.target)} if isinstance(self.target, list) else {}),
        }

    def to_task(self) -> fev.Task:  # noqa: F821 -- optional import stays local.
        """Construct the pinned fev task without loading its dataset."""
        try:
            import fev
        except ImportError as exc:
            raise ImportError("install demand-forecast[bench]") from exc

        values = asdict(self)
        values["task_name"] = values.pop("name")
        values["known_dynamic_columns"] = values.pop("known_dynamic")
        values["past_dynamic_columns"] = values.pop("past_dynamic")
        del values["categorical_statics"], values["target_support"]
        local_path = _local_path(self.dataset_path)
        values["dataset_path"] = str(local_path or self.dataset_path)
        if local_path is not None:
            values["dataset_config"] = None
        return fev.Task(**values)


def dataset_fingerprint(task) -> str:
    """Isolate fev 0.10.0's private fingerprint attribute behind one adapter."""
    task.load_full_dataset()
    return task._dataset_fingerprint


def _local_path(dataset_path: str | Path) -> Path | None:
    """Recognize parquet paths without treating HF repo IDs as local paths."""
    path = Path(dataset_path).expanduser()
    if (
        isinstance(dataset_path, Path)
        or path.is_absolute()
        or path.suffix == ".parquet"
        or str(dataset_path).startswith(("./", "../", "~"))
        or path.is_file()
    ):
        return path
    return None


def load_task_specs(selector: str | Path, profiles: Path | None = None) -> list[FevTaskSpec]:
    """Read ``weekly``, ``heldout``, ``fev100``, comma-separated task names, or task YAML.

    Profiles map task names to ``target_support`` and ``categorical_statics``.
    Custom YAML may supply ``name`` (or fev's ``task_name``); otherwise the
    dataset config, or local file stem, identifies its profile. Relative local
    parquet paths are resolved against the YAML file's directory.
    """
    resources = files("demand_forecast.bench").joinpath("tasks")
    path = Path(selector)
    custom_yaml = isinstance(selector, Path) or path.suffix in (".yaml", ".yml") or path.is_file()
    if custom_yaml:
        definitions = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        filename = {
            "weekly": "fev_bench_weekly.yaml",
            "heldout": "fev_bench_weekly_heldout.yaml",
        }.get(str(selector), "fev_bench_100_193e4fc.yaml")
        definitions = yaml.safe_load(resources.joinpath(filename).read_text())
        if selector not in ("weekly", "heldout", "fev100"):
            # Historical aliases (notably uk_covid_utla_1W) remain selectable.
            existing = {d.get("task_name", d["dataset_config"]) for d in definitions["tasks"]}
            for legacy in ("fev_bench_weekly.yaml", "fev_bench_weekly_heldout.yaml"):
                for definition in yaml.safe_load(resources.joinpath(legacy).read_text())["tasks"]:
                    if definition.get("task_name", definition["dataset_config"]) not in existing:
                        definitions["tasks"].append(definition)
    if not isinstance(definitions, dict) or not isinstance(definitions.get("tasks"), list):
        raise ValueError("task YAML must contain a tasks list")

    if profiles is None:
        profile_data = yaml.safe_load(resources.joinpath("profiles.yaml").read_text())
        profile_data.update(yaml.safe_load(resources.joinpath("profiles_fev100.yaml").read_text()))
    else:
        profile_data = yaml.safe_load(Path(profiles).read_text(encoding="utf-8"))
    if profile_data is None:
        profile_data = {}
    if not isinstance(profile_data, dict):
        raise ValueError("profiles must map task names to profiles")

    named_tasks = {}
    for definition in definitions["tasks"]:
        values = dict(definition)
        task_name = values.pop("task_name", None)
        name = values.pop("name", None) or task_name or values.get("dataset_config")
        name = name or Path(values["dataset_path"]).stem
        if name in named_tasks:
            raise ValueError(f"duplicate task name: {name}")
        named_tasks[name] = values

    names = (
        list(named_tasks)
        if custom_yaml or selector in ("weekly", "heldout", "fev100")
        else [name.strip() for name in str(selector).split(",")]
    )
    specs = []
    for name in names:
        if name not in named_tasks:
            raise ValueError(f"unknown task: {name}")
        profile = profile_data.get(name)
        if not isinstance(profile, dict):
            raise ValueError("profile missing")
        if "target_support" not in profile:
            raise ValueError(f"target_support missing from profile: {name}")
        values = named_tasks[name].copy()
        for upstream, local in (
            ("known_dynamic_columns", "known_dynamic"),
            ("past_dynamic_columns", "past_dynamic"),
        ):
            if upstream in values:
                values[local] = values.pop(upstream)
        local_path = _local_path(values["dataset_path"])
        if custom_yaml and local_path is not None and not local_path.is_absolute():
            values["dataset_path"] = str((path.parent / local_path).resolve())
        specs.append(
            FevTaskSpec(
                name=name,
                **values,
                target_support=profile["target_support"],
                categorical_statics=profile.get("categorical_statics", []),
            )
        )
    return specs
