"""Kernel workload end to end on a tiny prepared tree (CPU, real train/forecast)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from demand_forecast.config import DatasetConfig, Settings
from demand_forecast.core.run import train_from_settings
from demand_forecast.kaggle.kernel_workload import _rewrite_data_paths, run_workload
from demand_forecast.kaggle.settings import build_settings, load_template, write_settings

# The dev-only CI lap installs no parquet engine: skip this module, never fail it.
pytest.importorskip("pyarrow")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES = {
    "finetune": _REPO_ROOT / "configs/kaggle-v2-l3-finetune.yaml",
    "zero_shot": _REPO_ROOT / "configs/kaggle-v2-l3-zero-shot.yaml",
}
_HORIZON = 2
_STAMP_IDENTITY = "tiny-bootstrap-identity"
# Tiny CPU overrides pinned by the task brief; zero_shot=False keeps every
# combination on the fast trainer path (the zero-shot branch of core is covered
# by demand_forecast's own tests).
_MODEL_OVERRIDES = {
    "d_model": 8,
    "nhead": 2,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "dim_feedforward": 16,
    "hidden_continuous_size": 8,
}
_TRAINING_OVERRIDES = {
    "base_steps": 2,
    "eval_every": 1,
    "checkpoint_every": 1,
    "refit": "none",
    "zero_shot": False,
}
# Timeline: 6 train days (2017-08-01..06) + 2 test days (07..08); the mirror
# window clips the last horizon train days (mirror origin 2017-08-05).
_DAYS = pd.date_range("2017-08-01", periods=8, freq="D")
_MIRROR_ORIGIN = "2017-08-05"
_OFFICIAL_ORIGIN = "2017-08-07"


def _dataset_config(directory: Path) -> DatasetConfig:
    """The long/static roles the store-sales templates consume (as Task 4 pins)."""
    return DatasetConfig(
        source="long",
        path=directory / "long.parquet",
        id_columns=["store_nbr", "family"],
        timestamp_column="date",
        target="sales",
        target_support="nonnegative",
        known_dynamic=["onpromotion", "is_national_holiday"],
        past_dynamic=["oil_price", "transactions"],
        static_path=directory / "static.parquet",
        static_categorical=["store_nbr", "family", "city", "state", "type", "cluster"],
        freq="D",
        missing_target="zero",
        duplicates="error",
    )


def _long_frame(history_days: int) -> pd.DataFrame:
    """Two series on the daily grid; target and past dynamics end at ``history_days``."""
    rows = []
    for index, day in enumerate(_DAYS):
        for store_nbr, family in ((1, "AUTOMOTIVE"), (2, "BABY CARE")):
            observed = index < history_days
            rows.append(
                {
                    "store_nbr": store_nbr,
                    "family": family,
                    "date": day,
                    "sales": float(store_nbr * 10 + index + 1) if observed else float("nan"),
                    "onpromotion": int(day.day % 2 == 0),
                    "is_national_holiday": 0,
                    "oil_price": 40.0 + index if observed else float("nan"),
                    "transactions": float(100 * store_nbr) if observed else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _static_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "family": ["AUTOMOTIVE", "BABY CARE"],
            "city": ["Quito", "Guayaquil"],
            "state": ["Pichincha", "Guayas"],
            "type": ["A", "B"],
            "cluster": [3, 7],
        }
    )


def _tiny_settings(settings: Settings) -> Settings:
    """Shrink the real template settings to a CPU-second training run."""
    return settings.model_copy(
        update={
            "model": settings.model.model_copy(update=_MODEL_OVERRIDES),
            "training": settings.training.model_copy(update=_TRAINING_OVERRIDES),
        }
    )


def _tiny_prepared_tree(root: Path) -> Path:
    """Write the two-window prepared tree with settings built from the real templates."""
    mirror, official = root / "mirror", root / "official"
    for directory in (mirror, official):
        directory.mkdir(parents=True)
    _long_frame(history_days=4).to_parquet(mirror / "long.parquet", index=False)
    _long_frame(history_days=6).to_parquet(official / "long.parquet", index=False)
    for directory in (mirror, official):
        _static_frame().to_parquet(directory / "static.parquet", index=False)
    report = {
        "slug": "store-sales-time-series-forecasting",
        "series": 2,
        "train_days": 6,
        "test_days": 2,
        "mirror_origin": _MIRROR_ORIGIN,
        "official_origin": _OFFICIAL_ORIGIN,
        "quirks": {"missing_target": "zero", "negative_sales": 0, "zero_filled_rows": 0},
    }
    for directory in (mirror, official):
        for mode, template_path in _TEMPLATES.items():
            settings = build_settings(
                load_template(template_path),
                data=_dataset_config(directory),
                horizon=_HORIZON,
            )
            write_settings(
                _tiny_settings(settings), directory / f"settings-{mode.replace('_', '-')}.yaml"
            )
        (directory / "prepare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return root


def _stamp_pretrain_identity(bundle_dir: Path, identity: str) -> None:
    """Mark a bootstrapped bundle as pretrained; core requires the manifest key."""
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pretrain_identity"] = identity
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _bootstrap_bundle(reference: Settings, work_dir: Path) -> Path:
    """Train a tiny pretrain-shaped source bundle and stamp its identity.

    ``init_from_pretrained`` accepts only the calendar-off, statics-free
    pretraining shape: exactly one categorical vocabulary (``__series__``) and
    no numeric statics. The pinned L3 bundle has that shape, so the tiny source
    is trained on a single statics-free series with the reference settings'
    model/forecast/training blocks. ``train_from_settings`` does not write
    ``pretrain_identity`` (the pretraining pipeline does), and core requires the
    key on any ``init_from`` bundle, so it is stamped after training.
    """
    days = pd.date_range("2017-08-01", periods=8, freq="D")
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "pretrain-long.parquet"
    pd.DataFrame(
        {
            "series_id": ["pretrain-series"] * len(days),
            "timestamp": days,
            "sales": [float(index + 1) for index in range(len(days))],
        }
    ).to_parquet(path, index=False)
    settings = reference.model_copy(
        update={
            "data": DatasetConfig(
                source="long",
                path=path,
                id_columns=["series_id"],
                timestamp_column="timestamp",
                target="sales",
                target_support="nonnegative",
                freq="D",
                missing_target="zero",
                duplicates="error",
            )
        }
    )
    bundle = train_from_settings(settings, work_dir / "run")
    _stamp_pretrain_identity(bundle, _STAMP_IDENTITY)
    return bundle


def test_run_workload_end_to_end_tiny(tmp_path: Path):
    """Bootstrap a tiny bundle, then run the workload with it as init_from."""
    prepared = _tiny_prepared_tree(tmp_path / "prepared")

    # Bootstrap: train once without init_from from the official settings' recipe.
    bootstrap_settings = Settings.from_yaml(prepared / "official" / "settings-finetune.yaml")
    bundle_a = _bootstrap_bundle(bootstrap_settings, tmp_path / "bootstrap")

    out = tmp_path / "out"
    result = run_workload(prepared, out, init_from=bundle_a, bundle_identity=None)

    expected_runs = []
    for mode in ("zero_shot", "finetune"):
        for window, origin in (("mirror", _MIRROR_ORIGIN), ("official", _OFFICIAL_ORIGIN)):
            run_dir = out / mode / window
            assert (run_dir / "bundle/manifest.json").is_file()
            predictions = pd.read_csv(run_dir / "predictions.csv")
            assert {"series_id", "timestamp", "point", "q0.5"} <= set(predictions.columns)
            assert len(predictions) == 2 * _HORIZON
            assert predictions["point"].notna().all()
            stamps = pd.to_datetime(predictions["timestamp"]).drop_duplicates().sort_values()
            assert stamps.tolist() == pd.date_range(origin, periods=_HORIZON, freq="D").tolist()
            expected_runs.append(
                {"mode": mode, "window": window, "origin": origin, "init_from": str(bundle_a)}
            )
            assert (
                json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
                == (expected_runs[-1])
            )

    assert result == {
        "modes": ["zero_shot", "finetune"],
        "windows": ["mirror", "official"],
        "runs": expected_runs,
    }
    assert json.loads((out / "workload.json").read_text(encoding="utf-8")) == result


def test_run_workload_requires_init_from_for_zero_shot(tmp_path: Path):
    with pytest.raises(ValueError, match="zero_shot requires init_from"):
        run_workload(tmp_path / "prepared", tmp_path / "out", init_from=None, bundle_identity=None)


def test_run_workload_rejects_mismatched_bundle_identity(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"pretrain_identity": "some-other-identity"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="pretrain_identity"):
        run_workload(
            tmp_path / "prepared",
            tmp_path / "out",
            init_from=bundle,
            bundle_identity="the-expected-identity",
            modes=("finetune",),
            windows=("mirror",),
        )


def test_run_workload_rewrites_missing_dataset_paths(tmp_path: Path):
    """Kernel mounts move the prepared files; stale stored paths must be replaced."""
    prepared = _tiny_prepared_tree(tmp_path / "prepared")
    settings_path = prepared / "mirror" / "settings-finetune.yaml"
    settings = Settings.from_yaml(settings_path)
    settings.data.path = tmp_path / "elsewhere" / "long.parquet"
    settings.data.static_path = tmp_path / "elsewhere" / "static.parquet"
    write_settings(settings, settings_path)

    result = run_workload(
        prepared,
        tmp_path / "out",
        init_from=None,
        bundle_identity=None,
        modes=("finetune",),
        windows=("mirror",),
    )

    run_dir = tmp_path / "out" / "finetune" / "mirror"
    assert (run_dir / "bundle/manifest.json").is_file()
    assert (run_dir / "predictions.csv").is_file()
    assert result["runs"] == [
        {"mode": "finetune", "window": "mirror", "origin": _MIRROR_ORIGIN, "init_from": None}
    ]


def test_rewrite_data_paths_keeps_none_and_rewrites_missing(tmp_path: Path):
    """A static-less competition keeps ``None``; only set-but-missing paths move."""
    window_root = tmp_path / "window"
    window_root.mkdir()
    settings = build_settings(
        load_template(_TEMPLATES["finetune"]),
        data=_dataset_config(window_root),
        horizon=_HORIZON,
    )
    settings.data.path = None
    settings.data.static_path = None

    _rewrite_data_paths(settings, window_root)

    assert settings.data.path is None  # statics-free: never synthesized into a file path
    assert settings.data.static_path is None

    existing = tmp_path / "present.parquet"
    existing.write_bytes(b"present")
    settings.data.path = tmp_path / "elsewhere" / "long.parquet"
    settings.data.static_path = existing

    _rewrite_data_paths(settings, window_root)

    assert settings.data.path == window_root / "long.parquet"
    assert settings.data.static_path == existing  # an existing path is left alone
