"""In-kernel workload: train and forecast every mode/window combination.

A Kaggle kernel mounts the prepared dataset and the pinned pretrained bundle at
paths different from the laptop paths stored inside the prepared settings. This
module reloads each ``<window>/settings-<mode>.yaml``, rewrites dataset paths
that are set but missing on the mount (a ``None`` path — e.g. a statics-free
competition — stays ``None``), injects ``training.init_from``, and runs the
shared ``demand_forecast.core.run`` train/forecast path unchanged (CPU or GPU).
Each combination writes ``<out_root>/<mode>/<window>/`` with its bundle,
predictions and a small ``run.json``; the top-level ``workload.json`` is the
completion marker the host-side fetch stage checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]  # repo venv carries no pandas stubs

from demand_forecast.config import Settings
from demand_forecast.core.run import forecast_from_bundle, train_from_settings
from demand_forecast.kaggle.assets import verify_bundle_identity

__all__ = ["run_workload"]


def _window_origin(prepared_root: Path, window: str) -> str:
    """Read the window's forecast origin from its ``prepare.json``."""
    report = json.loads((prepared_root / window / "prepare.json").read_text(encoding="utf-8"))
    return str(report[f"{window}_origin"])


def _rewrite_data_paths(settings: Settings, window_root: Path) -> None:
    """Point stored dataset paths at the mounted window tree when they are missing.

    Only a non-``None`` stored path that does not exist is replaced: the laptop
    paths baked into the prepared settings never exist on the kernel mount,
    while a ``None`` path (a statics-free competition) must stay ``None`` —
    synthesizing a ``static.parquet`` path there would point the loader at a
    file no converter wrote.
    """
    if settings.data.path is not None and not settings.data.path.exists():
        settings.data.path = window_root / "long.parquet"
    if settings.data.static_path is not None and not settings.data.static_path.exists():
        settings.data.static_path = window_root / "static.parquet"


def run_workload(
    prepared_root: Path,
    out_root: Path,
    *,
    init_from: Path | None,
    bundle_identity: str | None,
    modes: tuple[str, ...] = ("zero_shot", "finetune"),
    windows: tuple[str, ...] = ("mirror", "official"),
) -> dict[str, Any]:
    """Train and forecast each mode/window combination under ``out_root``.

    ``prepared_root/<window>`` must hold the converter's long/static parquet,
    settings files and ``prepare.json``. Stored dataset paths that are set but
    missing are replaced with the mounted locations; a ``None`` path stays
    ``None`` (statics-free competitions). ``init_from`` (required for
    ``zero_shot``) is injected into every combination's training settings and
    ``bundle_identity``, when given, is verified before any run starts. All
    written paths are absolute; the returned mapping equals ``workload.json``.
    """
    if "zero_shot" in modes and init_from is None:
        raise ValueError("zero_shot requires init_from")
    prepared_root, out_root = Path(prepared_root).resolve(), Path(out_root).resolve()
    bundle_path = None if init_from is None else Path(init_from).resolve()
    if bundle_identity is not None:
        if bundle_path is None:
            raise ValueError("bundle_identity requires init_from")
        verify_bundle_identity(bundle_path, bundle_identity)

    runs: list[dict[str, str | None]] = []
    for mode in modes:
        for window in windows:
            out_dir = out_root / mode / window
            out_dir.mkdir(parents=True, exist_ok=True)
            window_root = prepared_root / window
            settings = Settings.from_yaml(window_root / f"settings-{mode.replace('_', '-')}.yaml")
            _rewrite_data_paths(settings, window_root)
            if bundle_path is not None:
                settings.training.init_from = str(bundle_path)
            bundle = train_from_settings(settings, out_dir)
            origin = _window_origin(prepared_root, window)
            forecast_from_bundle(
                bundle, settings, pd.Timestamp(origin), out_dir / "predictions.csv"
            )
            run_info: dict[str, str | None] = {
                "mode": mode,
                "window": window,
                "origin": origin,
                "init_from": settings.training.init_from,
            }
            (out_dir / "run.json").write_text(
                json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
            )
            runs.append(run_info)

    workload = {"modes": list(modes), "windows": list(windows), "runs": runs}
    (out_root / "workload.json").write_text(json.dumps(workload, indent=2) + "\n", encoding="utf-8")
    return workload
