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
    target_support: Literal["nonnegative", "real"] = field(kw_only=True)
    initial_cutoff: int | str | None = None
    min_context_length: int = 1
    eval_metric: str | dict[str, Any] = "MASE"
    extra_metrics: list[str | dict[str, Any]] = field(default_factory=list)
    window_step_size: int | str | None = None
    id_column: str = "id"
    timestamp_column: str = "timestamp"

    def __post_init__(self):
        if self.target_support not in ("nonnegative", "real"):
            raise ValueError("target_support must be 'nonnegative' or 'real'")
        if not set(self.categorical_statics) <= set(self.static_columns):
            raise ValueError("categorical_statics must be a subset of static_columns")

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
    """Read ``weekly``, ``heldout``, comma-separated weekly names, or task YAML.

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
        filename = (
            "fev_bench_weekly_heldout.yaml" if selector == "heldout" else "fev_bench_weekly.yaml"
        )
        definitions = yaml.safe_load(resources.joinpath(filename).read_text())
    if not isinstance(definitions, dict) or not isinstance(definitions.get("tasks"), list):
        raise ValueError("task YAML must contain a tasks list")

    if profiles is None:
        profile_data = yaml.safe_load(resources.joinpath("profiles.yaml").read_text())
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
        if custom_yaml or selector in ("weekly", "heldout")
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
