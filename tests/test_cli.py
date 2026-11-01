"""Tests for CLI commands."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from demand_forecast.cli import app

runner = CliRunner()


def test_pretrain_command_set_and_help():
    from typer.main import get_command

    assert set(get_command(app).commands) == {
        "train",
        "forecast",
        "fev",
        "pretrain",
        "generate-data",
        "generate-sales",
        "version",
    }
    assert set(get_command(app).commands["pretrain"].commands) == {"run", "probe", "probe-task"}
    for command in ("run", "probe", "probe-task"):
        result = runner.invoke(app, ["pretrain", command, "--help"])
        assert result.exit_code == 0, result.output


class TestVersionCommand:
    """Tests for version command."""

    def test_version_output(self):
        """Test version command outputs version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "demand-forecast version" in result.stdout


class TestGenerateDataCommand:
    """Tests for generate-data command."""

    def test_generate_data_creates_file(self, tmp_path: Path):
        """Test that generate-data creates a CSV file."""
        output_file = tmp_path / "test_data.csv"

        result = runner.invoke(
            app,
            [
                "generate-data",
                str(output_file),
                "--products",
                "5",
                "--stores",
                "2",
                "--days",
                "30",
                "--seed",
                "42",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_generate_long_data(self, tmp_path: Path):
        from demand_forecast.config.settings import DatasetConfig
        from demand_forecast.data.loader_long import load_long

        output_dir = tmp_path / "panel"
        result = runner.invoke(
            app,
            [
                "generate-data",
                str(output_dir),
                "--format",
                "long",
                "--products",
                "4",
                "--days",
                "30",
                "--freq",
                "W-SUN",
                "--start",
                "2023-01-01",
                "--seed",
                "7",
            ],
        )
        assert result.exit_code == 0, result.output
        assert {path.name for path in output_dir.iterdir()} == {"long.csv", "static.csv"}
        long = pd.read_csv(output_dir / "long.csv")
        static = pd.read_csv(output_dir / "static.csv")
        assert list(long.columns) == ["id", "timestamp", "target", "promo", "oil"]
        assert list(static.columns) == ["id", "cat"]
        panel = load_long(
            DatasetConfig(
                source="long",
                path=output_dir / "long.csv",
                id_columns=["id"],
                timestamp_column="timestamp",
                target="target",
                target_support="nonnegative",
                known_dynamic=["promo"],
                past_dynamic=["oil"],
                static_categorical=["cat"],
                static_path=output_dir / "static.csv",
                freq="W-SUN",
            )
        )
        assert len(panel) == 4
        assert all(series.length == 30 for series in panel.series)

    def test_generate_data_rejects_unknown_format(self, tmp_path: Path):
        output = tmp_path / "output"
        result = runner.invoke(app, ["generate-data", str(output), "--format", "unknown"])
        assert result.exit_code == 2, result.output
        assert "format" in result.output and not output.exists()

    def test_generate_data_with_defaults(self, tmp_path: Path):
        """Test generate-data with default parameters."""
        output_file = tmp_path / "default_data.csv"

        result = runner.invoke(
            app,
            ["generate-data", str(output_file), "--days", "10", "--products", "2", "--stores", "1"],
        )

        assert result.exit_code == 0
        assert output_file.exists()


class TestGenerateSalesCommand:
    """Exercise plain/gzipped catalogues and the store's exact CSV contract."""

    @pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
    def test_generate_sales(self, tmp_path, suffix):
        products = pd.DataFrame(
            {
                "id": ["catalogue:001", "catalogue:002"],
                "category": ["outerwear", "clothing"],
                "subcategory": ["jacket", "shorts"],
                "price": [100.0, 20.0],
                "embedding": ["unused huge payload", "unused huge payload"],
            }
        )
        product_file = tmp_path / f"products{suffix}"
        products.to_csv(product_file, index=False)
        output = tmp_path / "data" / "sales.csv"
        with patch("pandas.read_csv", wraps=pd.read_csv) as read_csv:
            result = runner.invoke(
                app,
                [
                    "generate-sales",
                    "--products",
                    str(product_file),
                    "--output",
                    str(output),
                    "--start",
                    "2024-10-17",
                    "--end",
                    "2024-11-16",
                    "--seed",
                    "7",
                    "--promo-share",
                    "0",
                    "-v",
                ],
            )
        assert result.exit_code == 0, result.output
        assert set(read_csv.call_args.kwargs["usecols"]) == {
            "id",
            "price",
            "category",
            "subcategory",
        }
        sales = pd.read_csv(output)
        assert list(sales.columns) == ["date", "product_id", "qty", "unit_price"]
        assert sales["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
        assert sales["date"].between("2024-10-17", "2024-11-16").all()
        assert sales["qty"].ge(0).all()
        expected_prices = sales["product_id"].map(products.set_index("id")["price"])
        pd.testing.assert_series_equal(sales["unit_price"], expected_prices, check_names=False)
        assert all(
            len(line.rsplit(",", 1)[1].split(".")[1]) == 2
            for line in output.read_text().splitlines()[1:]
        )

    def test_generate_sales_defaults(self, tmp_path):
        catalogue = tmp_path / "products.csv"
        catalogue.write_text("id,category,subcategory,price\nreal:001,clothing,shorts,25.00\n")
        output = tmp_path / "sales.csv"
        result = runner.invoke(
            app, ["generate-sales", "--products", str(catalogue), "--output", str(output)]
        )
        assert result.exit_code == 0, result.output
        sales = pd.read_csv(output)
        assert sales["date"].min() == "2024-10-17"
        assert sales["date"].max() == "2026-10-16"

    @pytest.mark.parametrize("arguments", [["--start", "2026-10-17"], ["--promo-share", "1.1"]])
    def test_invalid_generate_sales_arguments(self, tmp_path, arguments):
        catalogue = tmp_path / "products.csv"
        catalogue.write_text("id,category,subcategory,price\nreal:001,clothing,shorts,25.00\n")
        output = tmp_path / "sales.csv"
        result = runner.invoke(
            app,
            ["generate-sales", "--products", str(catalogue), "--output", str(output), *arguments],
        )
        assert result.exit_code != 0
        assert not output.exists()


def test_train_then_forecast_cli(tmp_path):
    from demand_forecast.synthetic.generator import generate_long_panel
    from tests.helpers.run_config import write_cfg

    long, static = generate_long_panel(
        6, 60, "W-SUN", "2022-01-02", ("promo",), ("oil",), ("cat",), 0
    )
    write_cfg(tmp_path, long, static, horizon=4, base_steps=30, d_model=16)
    result = runner.invoke(
        app, ["train", "--config", str(tmp_path / "cfg.yaml"), "--out", str(tmp_path / "run")]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert (tmp_path / "run/bundle/manifest.json").exists()
    result = runner.invoke(
        app,
        [
            "forecast",
            "--bundle",
            str(tmp_path / "run/bundle"),
            "--config",
            str(tmp_path / "cfg.yaml"),
            "--origin",
            "2023-02-26",
            "--out",
            str(tmp_path / "f.csv"),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    df = pd.read_csv(tmp_path / "f.csv")
    assert len(df) == 6 * 4 and list(df.columns)[:3] == ["series_id", "origin", "timestamp"]
    assert df["point"].equals(df["q0.5"])
    assert df["origin"].eq("2023-02-26").all()


@pytest.fixture
def run_settings(tmp_path):
    from demand_forecast.config import Settings
    from demand_forecast.synthetic.generator import generate_long_panel
    from tests.helpers.run_config import write_cfg

    long, static = generate_long_panel(
        3, 32, "W-SUN", "2022-01-02", ("promo",), ("oil",), ("cat",), 0
    )
    return Settings.from_yaml(write_cfg(tmp_path, long, static, horizon=4, base_steps=8, d_model=8))


def test_run_identity_canonical_json(run_settings):
    import hashlib
    import json

    from demand_forecast.core.run import run_identity

    origin = pd.Timestamp("2022-08-14")
    task = {"name": "fixture", "horizon": 4}
    settings = run_settings.model_dump(mode="json")
    for key in ("max_steps_cap", "val_items_max"):
        if settings["training"][key] is None:
            del settings["training"][key]
    for key in ("init_from", "zero_shot", "eval_at_start"):
        del settings["training"][key]
    payload = {
        "settings": settings,
        "seed": 3,
        "dataset_fingerprint": "data-hash",
        "package_version": "2.0.0",
        "origin": origin.isoformat(),
        "task_definition": task,
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert run_identity(run_settings, 3, "data-hash", "2.0.0", origin, task) == expected
    assert (
        run_identity(
            run_settings,
            3,
            "data-hash",
            "2.0.0",
            origin.isoformat(),
            dict(reversed(list(task.items()))),
        )
        == expected
    )
    assert run_identity(run_settings, 3, "other", "2.0.0", origin, task) != expected
    assert run_identity(run_settings, 3, "data-hash", "2.0.0", origin, None) != expected
    assert (
        run_identity(run_settings, 3, "data-hash", "2.0.0", origin + pd.Timedelta(weeks=1), task)
        != expected
    )


def test_train_seeds_before_build_and_saves_run_metadata(tmp_path, run_settings, monkeypatch):
    import json
    import random

    import numpy as np
    import torch

    from demand_forecast.core import run
    from demand_forecast.core.bundle import load_bundle

    seen = []
    build = run.build_model

    def capture(*args, **kwargs):
        seen.append((torch.rand(1).item(), np.random.rand(), random.random()))
        return build(*args, **kwargs)

    monkeypatch.setattr(run, "build_model", capture)
    bundle = run.train_from_settings(run_settings, tmp_path / "one")
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    second = run.train_from_settings(run_settings, tmp_path / "two")
    assert seen[0] == seen[1]
    loaded = load_bundle(bundle)
    repeated = load_bundle(second)
    for name, value in loaded.model.state_dict().items():
        assert torch.equal(value, repeated.model.state_dict()[name]), name
    info = json.loads((tmp_path / "one/run.json").read_text())
    assert info["run_identity"] == loaded.manifest["run_identity"]
    assert info["seed"] == run_settings.seed and info["resumed_from_step"] == 0
    assert info["origin"] == "2022-08-14T00:00:00"
    assert info["plan"]["train_upto"] == loaded.panel_metadata["training_cutoff"]
    assert info["fit_result"]["steps"] == 8
    assert info["timings"]["training_time_s"] > 0


def test_train_encoder_uses_only_fit_ids(tmp_path, run_settings):
    from demand_forecast.core.bundle import load_bundle
    from demand_forecast.core.run import train_panel
    from demand_forecast.data.loader_long import load_long

    panel = load_long(run_settings.data)
    # Only the validation block contains this series' target/past observations.
    held_out = panel.series[-1]
    held_out.target[:-4] = float("nan")
    held_out.dynamic["oil"][:-4] = float("nan")
    held_out.statics["cat"] = "validation-only"
    bundle = load_bundle(
        train_panel(
            panel,
            pd.Timestamp("2022-08-14"),
            run_settings,
            tmp_path / "run",
            dataset_fingerprint="fit-only",
        )
    )
    assert held_out.series_id not in bundle.encoder.vocabularies["__series__"]
    assert "validation-only" not in bundle.encoder.vocabularies["cat"]


@pytest.mark.parametrize("resume_at", [0, 3])
def test_train_panel_writes_bundle_when_stop_after_completes_budget(
    tmp_path, run_settings, resume_at
):
    import json

    import torch

    from demand_forecast.core.bundle import load_bundle
    from demand_forecast.core.run import train_panel
    from demand_forecast.core.training import InterruptedRun
    from demand_forecast.data.loader_long import load_long

    # Keep this check focused on budget completion, independently of validation scores.
    run_settings.training.patience = 20
    panel = load_long(run_settings.data)
    out = tmp_path / "run"
    args = (panel, pd.Timestamp("2022-08-14"), run_settings, out)
    if resume_at:
        with pytest.raises(InterruptedRun):
            train_panel(*args, dataset_fingerprint="fixture", stop_after=resume_at)
    bundle = train_panel(
        *args,
        dataset_fingerprint="fixture",
        resume=bool(resume_at),
        stop_after=8 - resume_at,
    )
    assert bundle == out / "bundle"
    loaded = load_bundle(bundle)
    report = json.loads((out / "run.json").read_text())
    assert report["fit_result"]["stop_reason"] == "max_steps"
    assert report["fit_result"]["steps"] == 8
    assert report["resumed_from_step"] == resume_at
    assert loaded.manifest["run"] == report
    assert torch.load(out / "checkpoints/latest.pt", weights_only=False)["step"] == 8


def test_interrupted_run_resumes_exactly_and_reuses_bundle(tmp_path, run_settings, monkeypatch):
    import json

    import torch

    from demand_forecast.core import run
    from demand_forecast.core.bundle import load_bundle
    from demand_forecast.core.training import InterruptedRun
    from demand_forecast.data.loader_long import load_long

    panel = load_long(run_settings.data)
    origin = pd.Timestamp("2022-08-14")
    options = {"dataset_fingerprint": "fixture"}
    with pytest.raises(InterruptedRun):
        run.train_panel(panel, origin, run_settings, tmp_path / "split", stop_after=3, **options)
    assert not (tmp_path / "split/bundle").exists()
    checkpoint = torch.load(tmp_path / "split/checkpoints/latest.pt", weights_only=False)
    assert checkpoint["step"] == 3
    resumed = run.train_panel(
        panel, origin, run_settings, tmp_path / "split", resume=True, **options
    )
    full = run.train_panel(panel, origin, run_settings, tmp_path / "full", **options)
    resumed_bundle, full_bundle = load_bundle(resumed), load_bundle(full)
    assert resumed_bundle.manifest["resumed_from_step"] == 3
    for name, value in full_bundle.model.state_dict().items():
        assert torch.equal(value, resumed_bundle.model.state_dict()[name]), name
    metadata = json.loads((tmp_path / "split/run.json").read_text())
    assert metadata["resumed_from_step"] == 3
    before = {p: p.read_bytes() for p in (tmp_path / "split").rglob("*") if p.is_file()}
    with patch.object(run.Trainer, "fit", side_effect=AssertionError("already trained")):
        assert (
            run.train_panel(panel, origin, run_settings, tmp_path / "split", resume=True, **options)
            == resumed
        )
    assert all(p.read_bytes() == content for p, content in before.items())
    (tmp_path / "split/run.json").unlink()
    with patch.object(run.Trainer, "fit", side_effect=AssertionError("already trained")):
        run.train_panel(panel, origin, run_settings, tmp_path / "split", resume=True, **options)
    assert json.loads((tmp_path / "split/run.json").read_text()) == metadata


@pytest.mark.parametrize("artifact", ["checkpoint", "bundle", "run"])
def test_resume_checks_all_present_artifact_identities(tmp_path, run_settings, artifact):
    import json

    import torch

    from demand_forecast.core.run import train_from_settings

    out = tmp_path / "run"
    train_from_settings(run_settings, out)
    if artifact == "checkpoint":
        path = out / "checkpoints/latest.pt"
        content = torch.load(path, weights_only=False)
        content["run_identity"] = "wrong"
        torch.save(content, path)
    else:
        path = out / ("bundle/manifest.json" if artifact == "bundle" else "run.json")
        content = json.loads(path.read_text())
        content["run_identity"] = "wrong"
        path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="run identity mismatch"):
        train_from_settings(run_settings, out, resume=True)


@pytest.mark.parametrize("change", ["settings", "long", "static"])
def test_resume_rejects_changed_inputs(tmp_path, run_settings, change):
    from demand_forecast.core.run import train_from_settings

    out = tmp_path / "run"
    train_from_settings(run_settings, out)
    if change == "settings":
        run_settings.seed += 1
    else:
        path = run_settings.data.path if change == "long" else run_settings.data.static_path
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="run identity mismatch"):
        train_from_settings(run_settings, out, resume=True)


@pytest.mark.parametrize("missing", [False, True])
def test_train_no_training_data_still_saves_fallback_bundle(tmp_path, run_settings, missing):
    import json

    import numpy as np

    from demand_forecast.core.run import forecast_from_bundle, train_from_settings

    frame = pd.read_csv(run_settings.data.path).groupby("id", sort=False).head(1)
    frame["target"] = np.nan if missing else 6.0
    frame["oil"] = np.nan if missing else 1.0
    frame.to_csv(run_settings.data.path, index=False)
    out = tmp_path / "run"
    bundle = train_from_settings(run_settings, out)
    report = json.loads((out / "run.json").read_text())
    assert report["fit_result"]["steps"] == 0
    assert report["fit_result"]["stop_reason"] == "no_training_data"
    forecasts = forecast_from_bundle(
        bundle, run_settings, pd.Timestamp("2022-01-09"), tmp_path / "nested/f.csv"
    )
    assert (forecasts.filter(regex="^q").to_numpy() == (0.0 if missing else 6.0)).all()
    assert len(forecasts) == 12


def test_train_cli_resume_and_errors(tmp_path, run_settings):
    args = ["train", "--config", str(tmp_path / "cfg.yaml"), "--out", str(tmp_path / "run")]
    assert runner.invoke(app, args).exit_code == 0
    with patch(
        "demand_forecast.core.run.Trainer.fit", side_effect=AssertionError("already trained")
    ):
        result = runner.invoke(app, [*args, "--resume"])
    assert result.exit_code == 0, result.output
    run_settings.training.refit = "full"
    run_settings.to_yaml(tmp_path / "cfg.yaml")
    result = runner.invoke(app, args)
    assert result.exit_code != 0 and "phase-4" in result.output
    result = runner.invoke(
        app,
        [
            "forecast",
            "--config",
            str(tmp_path / "cfg.yaml"),
            "--bundle",
            str(tmp_path / "run/bundle"),
            "--origin",
            "2022-08-15",
            "--out",
            str(tmp_path / "bad.csv"),
        ],
    )
    assert result.exit_code != 0 and "off the" in result.output
    assert not (tmp_path / "bad.csv").exists()


def test_stop_after_writes_progress_json_and_resume_accumulates_timings(tmp_path, run_settings):
    import json

    from demand_forecast.core.run import train_panel
    from demand_forecast.core.training import InterruptedRun
    from demand_forecast.data.loader_long import load_long

    panel = load_long(run_settings.data)
    origin = panel.timestamps(0)[-1] + pd.tseries.frequencies.to_offset(panel.freq)
    out = tmp_path / "timed"
    with pytest.raises(InterruptedRun):
        train_panel(panel, origin, run_settings, out, dataset_fingerprint="fixture", stop_after=3)
    progress = json.loads((out / "checkpoints/progress.json").read_text())
    assert progress["step"] == 3 and progress["timings"]["sessions"] == 1
    assert not (out / "run.json").exists()
    train_panel(panel, origin, run_settings, out, dataset_fingerprint="fixture", resume=True)
    report = json.loads((out / "run.json").read_text())
    assert report["resolved_budget"] == {"max_steps": 8, "eval_every": 1}
    assert report["timings"]["cumulative"]["sessions"] == 2
    assert report["timings"]["cumulative"]["train_s"] >= progress["timings"]["train_s"] > 0
    assert len(report["telemetry"]) == report["fit_result"]["steps"]
    assert report["n_fit_ids"] == 3 and report["n_val_items"] == 3
    assert report["amp_dtype"] is None
    assert report["fit_result"]["nonfinite_losses"] == 0


def test_run_json_serialises_nonfinite_as_null(tmp_path):
    import json

    from demand_forecast.core.run import _write_run

    path = tmp_path / "run.json"
    _write_run(
        path,
        {
            "telemetry": [{"score": float("nan"), "duration": float("inf")}],
            "negative": -float("inf"),
            "finite": 1.0,
        },
    )
    assert json.loads(path.read_text()) == {
        "telemetry": [{"score": None, "duration": None}],
        "negative": None,
        "finite": 1.0,
    }


def test_train_cli_nonfinite_training_exits_1_with_stderr_and_no_bundle(
    tmp_path, run_settings, monkeypatch
):
    from demand_forecast.core import training

    run_settings.training.base_steps = 20
    run_settings.training.patience = 100
    run_settings.to_yaml(tmp_path / "cfg.yaml")
    monkeypatch.setattr(training, "pinball_loss", lambda pred, *a: (pred.sum() * float("nan"), 1))
    out = tmp_path / "failed"
    args = ["train", "--config", str(tmp_path / "cfg.yaml"), "--out", str(out)]
    with pytest.warns(UserWarning, match=r"lr_scheduler.step\(\).*before.*optimizer.step"):
        result = runner.invoke(app, args)
    expected = (
        "Training stopped after 20 consecutive non-finite losses at step 20; "
        f"checkpoint {out / 'checkpoints/latest.pt'}"
    )
    assert result.exit_code == 1 and expected in result.stderr
    assert "Bundle saved" not in result.stdout
    assert not (out / "bundle").exists() and not (out / "run.json").exists()
    assert (out / "checkpoints/latest.pt").exists()
    resumed = runner.invoke(app, [*args, "--resume"])
    assert resumed.exit_code == 1 and expected in resumed.stderr
    assert not (out / "bundle").exists() and not (out / "run.json").exists()


@pytest.mark.parametrize("command", ["train", "fev"])
def test_cli_init_from_merge_rejects_empty_and_list_roots_and_handles_missing_training(
    tmp_path, monkeypatch, command
):
    import yaml

    from tests.helpers.settings import tiny_settings

    config = tmp_path / "config.yaml"
    source = tmp_path / "pretrained"
    source.mkdir()
    args = ["train", "--out"] if command == "train" else ["fev", "run", "--out-dir"]
    args += [str(tmp_path / "out"), "--config", str(config)]
    target = (
        "demand_forecast.core.run.train_from_settings"
        if command == "train"
        else "demand_forecast.bench.run.run_fev"
    )
    captured = []

    def capture(*args, **kwargs):
        captured.append(args[0] if command == "train" else args[1])
        return tmp_path / "out/bundle"

    monkeypatch.setattr(target, capture)
    for raw, message in [
        (None, "config root must be a mapping"),
        ([], "config root must be a mapping"),
        ({"training": []}, "training must be a mapping"),
        ({"training": None}, "training must be a mapping"),
    ]:
        config.write_text(yaml.safe_dump(raw))
        for override in ([], ["--init-from", str(source)]):
            result = runner.invoke(app, [*args, *override])
            assert result.exit_code == 2 and message in result.output, result.output
    assert not captured
    raw = tiny_settings().model_dump(mode="json")
    del raw["training"]
    config.write_text(yaml.safe_dump(raw))
    result = runner.invoke(app, [*args, "--init-from", str(source)])
    assert result.exit_code == 0, result.output
    assert captured[-1].training.init_from == str(source)
    raw["training"] = {"init_from": "yaml-source", "learning_rate": 0.0001}
    config.write_text(yaml.safe_dump(raw))
    result = runner.invoke(app, [*args, "--init-from", str(source)])
    assert result.exit_code == 0, result.output
    assert captured[-1].training.init_from == str(source)
    assert captured[-1].training.learning_rate == 0.0001
    result = runner.invoke(app, args)
    assert result.exit_code == 0 and captured[-1].training.init_from == "yaml-source"


def test_cli_shipped_zero_shot_yaml_resolves(tmp_path, monkeypatch):
    from demand_forecast.config import Settings

    config = Path(__file__).resolve().parents[1] / "configs/fev-v2-pretrained-zero-shot.yaml"
    settings = Settings.from_yaml(config)
    assert settings.training.zero_shot and settings.training.init_from is None
    source = tmp_path / "pretrained"
    source.mkdir()
    captured = []

    def capture(specs, settings, *args, **kwargs):
        captured.append(settings)

    monkeypatch.setattr("demand_forecast.bench.run.run_fev", capture)
    result = runner.invoke(
        app,
        [
            "fev",
            "run",
            "--config",
            str(config),
            "--out-dir",
            str(tmp_path / "out"),
            "--init-from",
            str(source),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured[0].training.zero_shot and captured[0].training.init_from == str(source)
    assert captured[0].model.d_model == 256
