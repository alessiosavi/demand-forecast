"""Offline checks of the v5 export, parity replay and command line."""

import importlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, fields, replace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from demand_forecast.data.episode import (
    CH_OBSERVED,
    CH_VALUE,
    CH_VALUE_STATS,
    EpisodeBatch,
    EpisodeRequest,
    Item,
    Panel,
)
from demand_forecast.data.episode import builder as episode_builder
from demand_forecast.models.v5 import V5Config, V5Model, forecast, forecast_items
from demand_forecast.pretrain.v5 import (
    OptimConfig,
    SamplerConfig,
    SourceConfig,
    V5TrainConfig,
    file_digest,
    identity,
    load_model,
    save_checkpoint,
    sources,
)
from demand_forecast.synthetic import typed

TINY = V5Config(
    d_model=16,
    nhead=2,
    enc_layers=1,
    dec_layers=1,
    expert_enc_layers=1,
    expert_dec_layers=1,
    dim_feedforward=16,
    d_calendar=4,
    d_step=4,
    d_out=4,
)


@pytest.fixture(scope="module")
def export_api():
    return importlib.import_module("demand_forecast.models.v5.export")


def _checkpoint(directory, weights="model", model=None, *, sampler_options=None):
    model = TINY if model is None else model
    config = V5TrainConfig(
        model,
        (SourceConfig("corpus", 1.0, root="/unused/local/corpus"),),
        SamplerConfig(screen_no_evidence=0.2, **(sampler_options or {})),
        OptimConfig(max_steps=1, warmup_steps=0, ema_decay=0.999 if weights == "ema" else None),
        device="cpu",
        workers=1,
    )
    raw = V5Model(model).state_dict()
    ema = None
    if weights == "ema":
        ema = {name: value.clone() for name, value in raw.items()}
        ema["head.projection.bias"][0] += 0.1
    return save_checkpoint(
        directory,
        {
            "config": config._serialized(),
            "identity": identity(config, code="fixture", corpus=None, init=None, env=None),
            "model": raw,
            "ema": ema,
            "export": weights,
            "counters": {"batch": 7, "updates": 4, "saves": 1},
        },
        counter=1,
    )[0]


@pytest.fixture(scope="module")
def reference_export(tmp_path_factory, export_api):
    directory = tmp_path_factory.mktemp("v5-export")
    out = directory / "export"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        export_api.write_export(_checkpoint(directory), out)
    return out


@pytest.fixture
def artifact(tmp_path, reference_export):
    return shutil.copytree(reference_export, tmp_path / "export")


@pytest.mark.parametrize("weights", ["raw", "ema"])
def test_round_trip_of_the_whole_state_dict(tmp_path, monkeypatch, export_api, weights):
    # V5 currently has no persistent buffers. Exercise the state-dict contract
    # with a non-floating buffer on both sides of serialization.
    original_init = V5Model.__init__

    def with_buffer(self, config):
        original_init(self, config)
        self.register_buffer("export_test_buffer", torch.tensor([13, 21], dtype=torch.int64))

    monkeypatch.setattr(V5Model, "__init__", with_buffer)
    checkpoint = _checkpoint(tmp_path, "model" if weights == "raw" else "ema")
    out = tmp_path / "existing-empty"
    out.mkdir()
    written = export_api.write_export(checkpoint, out)
    loaded, metadata = export_api.load_export(out)
    oracle, config = load_model(checkpoint, weights="export")

    assert set(path.name for path in out.iterdir()) == {
        "config.json",
        "model.safetensors",
        "parity.npz",
    }
    assert written == metadata
    assert set(metadata) == {"format", "model", "training", "source", "files"}
    assert metadata["format"] == "v5-export-1"
    # The stored model mapping leaves the default expert mode out, as it did before the mode existed.
    stored = json.loads(json.dumps(asdict(TINY)))
    assert stored.pop("expert_mode") == "routed"
    assert metadata["model"] == stored
    assert metadata["training"] == config.canonical()
    assert metadata["source"] == {
        "sha256": file_digest(checkpoint),
        "identity": identity(config, code="fixture", corpus=None, init=None, env=None),
        "weights": weights,
        "updates": 4,
    }
    assert metadata["files"] == {
        name: file_digest(out / name) for name in ("parity.npz", "model.safetensors")
    }
    assert export_api.export_digest(out) == file_digest(out / "config.json")
    assert not loaded.training
    for state in (loaded.state_dict(), load_file(out / "model.safetensors")):
        expected = oracle.state_dict()
        assert state.keys() == expected.keys()
        for name, value in state.items():
            assert value.dtype == expected[name].dtype
            torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
        assert state["export_test_buffer"].tolist() == [13, 21]

    item = Item("one", 0, {"y": np.arange(48, dtype=np.float64) ** 0.5}, {}, {})
    panel = Panel("forecast", (item,), {}, None, None, False)
    request = EpisodeRequest(0, (32,), 4, context=32)
    expected_forecast = forecast(oracle, panel, request)
    actual_forecast = forecast(loaded, panel, request)
    # Equal weights give equal forecasts only up to float32 kernel noise: on x86-64 two models at
    # different addresses differed by 2e-8 (the weights above are compared exactly).
    np.testing.assert_allclose(
        actual_forecast.quantiles, expected_forecast.quantiles, rtol=1e-6, atol=1e-6
    )
    assert actual_forecast.names == expected_forecast.names
    assert actual_forecast.gates == expected_forecast.gates


def test_dense_export_round_trips_with_parity(tmp_path, export_api):
    out = tmp_path / "export"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        dense = replace(TINY, expert_mode="dense")
        metadata = export_api.write_export(_checkpoint(tmp_path, model=dense), out)
    assert metadata["model"]["expert_mode"] == "dense"
    loaded, _ = export_api.load_export(out)
    assert loaded.config.expert_mode == "dense"
    assert not any("router" in name for name, _ in loaded.named_parameters())
    export_api.require_parity(out)


def _read_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def test_canonical_serialized_and_export_omit_an_empty_derive(export_api, reference_export):
    _, metadata = export_api.load_export(reference_export)
    assert "derive" not in metadata["training"]["sampler"]
    arrays = _read_archive(reference_export / "parity.npz")
    for name, value in arrays.items():
        if name.endswith("/requests"):
            assert all("derive" not in request for request in json.loads(value.item()))
            assert all(
                request.derive == frozenset() for request in export_api._read_requests(value)
            )


@pytest.mark.parametrize(
    "derive",
    [
        ("increment_spread",),
        ("lag_rows",),
        ("naive_rows",),
        ("onset_rows",),
        ("onset_rows", "naive_rows", "increment_spread", "lag_rows"),
    ],
)
def test_enabled_names_round_trip_sorted(tmp_path, export_api, derive):
    checkpoint = _checkpoint(tmp_path, sampler_options={"derive": derive})
    payload = export_api.read_payload(checkpoint)
    assert payload["config"]["sampler"]["derive"] == sorted(derive)
    _, config = load_model(checkpoint)
    assert config.sampler.derive == tuple(sorted(derive))
    out = tmp_path / "export"
    export_api.write_export(checkpoint, out)
    _, metadata = export_api.load_export(out)
    assert metadata["training"]["sampler"]["derive"] == sorted(derive)
    arrays = _read_archive(out / "parity.npz")
    for name, value in arrays.items():
        if name.endswith("/requests"):
            assert all(request["derive"] == sorted(derive) for request in json.loads(value.item()))
            assert all(
                request.derive == frozenset(derive) for request in export_api._read_requests(value)
            )


def test_derive_reaches_every_request(tmp_path, export_api, monkeypatch):
    derive = ("increment_spread", "lag_rows")
    checkpoint = _checkpoint(tmp_path, sampler_options={"derive": derive})
    seen = []
    validate = episode_builder.EpisodeBuilder._validate

    def observe(builder, request):
        seen.append(request.derive)
        return validate(builder, request)

    monkeypatch.setattr(episode_builder.EpisodeBuilder, "_validate", observe)
    export_api.write_export(checkpoint, tmp_path / "export")
    assert len(seen) > 17 and set(seen) == {frozenset(derive)}


def test_an_enabled_export_passes_parity(tmp_path, export_api, reference_export):
    out = tmp_path / "export"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        checkpoint = _checkpoint(tmp_path, sampler_options={"derive": ("increment_spread",)})
    export_api.write_export(checkpoint, out)
    export_api.require_parity(out)
    original = _read_archive(reference_export / "parity.npz")
    enabled = _read_archive(out / "parity.npz")
    spread = CH_VALUE_STATS + 1
    for field in ("x", "xh"):
        key = f"case_6/batch/{field}"
        before, after = original[key], enabled[key]
        assert before.dtype == after.dtype
        np.testing.assert_array_equal(before[..., :spread], after[..., :spread])
        np.testing.assert_array_equal(before[..., spread + 1 :], after[..., spread + 1 :])
        assert np.any(before[..., spread] != after[..., spread])


def test_enabled_export_holds_the_rows(tmp_path, export_api):
    out = tmp_path / "export"
    checkpoint = _checkpoint(tmp_path, sampler_options={"derive": ("naive_rows",)})
    export_api.write_export(checkpoint, out)
    model, metadata = export_api.load_export(out)
    export_api.require_parity(out)

    arrays = _read_archive(out / "parity.npz")
    assert sum(key.endswith("/panel") for key in arrays) == 9
    prefix = "case_9"
    panel = export_api._read_panel(arrays, prefix)
    (request,) = export_api._read_requests(arrays[f"{prefix}/requests"])
    assert panel.season == 24 and panel.long_lag == 168
    item = panel.items[request.item_index]
    (target,) = item.targets
    origin = request.origins[0]
    local = origin - item.start
    cells = np.arange(local, local + request.horizon)
    scaled = (item.targets[target] - arrays[f"{prefix}/batch/loc"][0, 0]) / arrays[
        f"{prefix}/batch/scale"
    ][0, 0]
    assert len(scaled) >= local >= 168
    previous_season = local - 1 - (local - 1 - cells) % 24
    previous_week = local - 1 - (local - 1 - cells) % 168
    drift = (scaled[local - 24 : local] - scaled[local - 48 : local - 24]).mean() / 24
    expected = {
        "last": np.full(request.horizon, scaled[local - 1]),
        "season": scaled[previous_season],
        "season2": scaled[previous_week],
        "drift": scaled[previous_season] + drift * (cells - previous_season),
    }
    with torch.no_grad():
        model.gate_net.net[-1].weight.zero_()
        model.gate_net.net[-1].bias.fill_(-1000)
    evidence = forecast_items(
        model,
        panel,
        [(request.item_index, origin)],
        request.horizon,
        context=request.context,
        screening=export_api._screening(metadata["training"]),
        derive=request.derive,
        pruned=True,
    ).evidence
    (names,) = json.loads(arrays[f"{prefix}/batch/row_names"].item())
    known = arrays[f"{prefix}/batch/known_rows"][0].tolist()
    horizon = arrays[f"{prefix}/batch/xh"][0, 0]
    assert set(names) == {target, *(f"{target}~{suffix}" for suffix in expected)}
    for suffix, values in expected.items():
        name = f"{target}~{suffix}"
        row = horizon[known.index(names.index(name))].reshape(-1, horizon.shape[-1])
        np.testing.assert_allclose(
            row[: request.horizon, CH_VALUE], np.arcsinh(values), atol=1e-6, rtol=1e-6
        )
        assert row[: request.horizon, CH_OBSERVED].all()
        assert not row[request.horizon :].any()
        index = evidence.candidates.index(f"derived:{name}")
        assert evidence.status[0, index] == 1
        assert evidence.gate[0, index] == pytest.approx(model.gate_net.known_gate_floor)
        assert evidence.score[0, index] == pytest.approx(1.0)


def _rehash(out, filename):
    config = out / "config.json"
    metadata = json.loads(config.read_text())
    metadata["files"][filename] = file_digest(out / filename)
    config.write_text(json.dumps(metadata))


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_load_export_requires_exact_state_keys(export_api, artifact, mutation):
    weights = artifact / "model.safetensors"
    state = load_file(weights)
    if mutation == "missing":
        del state["head.projection.bias"]
        message = r"Missing key.*head\.projection\.bias"
    else:
        state["extra.weight"] = torch.zeros(1)
        message = r"Unexpected key.*extra\.weight"
    save_file(state, weights)
    _rehash(artifact, weights.name)

    with pytest.raises(RuntimeError, match=message):
        export_api.load_export(artifact)


def test_parity_passes_where_written(export_api, artifact):
    report = export_api.check_parity(artifact)
    assert report
    assert all(value is True or (type(value) is float and value <= 1) for value in report.values())
    export_api.require_parity(artifact)

    arrays = _read_archive(artifact / "parity.npz")
    assert sum(key.endswith("/panel") for key in arrays) == 8
    assert all(not value.dtype.hasobject for value in arrays.values())
    requests = []
    for i in range(1, 9):
        prefix = f"case_{i}/"
        assert {field.name for field in fields(EpisodeBatch)} == {
            key.removeprefix(prefix + "batch/")
            for key in arrays
            if key.startswith(prefix + "batch/")
        }
        requests.append(json.loads(arrays[prefix + "requests"].item()))
    assert [len(requests[i]) for i in (2, 3, 4, 7)] == [1, 2, 17, 3]
    assert requests[2][0]["target"] is None
    assert len({request["target"] for request in requests[4]}) == 17
    names = json.loads(arrays["case_1/batch/row_names"].item())
    assert "group=1" in names[0]
    assert arrays["case_1/batch/cal_ids"].max() >= 0
    assert np.all(arrays["case_2/batch/cal_ids"] == -1)
    assert arrays["case_3/batch/keep"].sum() == 5
    assert np.all(arrays["case_4/batch/keep"].sum(axis=1) == 1)
    assert arrays["case_6/item_0/dynamic_0"].dtype.kind == "U"
    assert arrays["case_6/item_0/dynamic_0_missing"][48:].tolist() == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    assert len({request["origins"][0] for request in requests[7]}) == 3
    assert arrays["case_7/batch/static_num_flags"].any()
    assert arrays["case_8/batch/fallback"].sum() == 3
    assert all(arrays[f"case_{i}/output/quantiles"].dtype == np.float32 for i in range(1, 9))


@pytest.mark.parametrize("case", ["case_8", "case_9"], ids=["missing", "extra"])
def test_parity_requires_exact_case_prefixes(export_api, artifact, case):
    path = artifact / "parity.npz"
    arrays = _read_archive(path)
    if case == "case_8":
        arrays = {key: value for key, value in arrays.items() if not key.startswith(f"{case}/")}
    else:
        arrays[f"{case}/requests"] = arrays["case_1/requests"]
    np.savez_compressed(path, **arrays)
    _rehash(artifact, path.name)

    with pytest.raises(ValueError, match=case):
        export_api.check_parity(artifact)


@pytest.mark.parametrize(
    ("drift", "entry"),
    [
        ("weight", r"\.output\.quantiles"),
        ("scale", r"\.batch\.loc"),
        ("mask", r"\.batch\.row_mask"),
        ("calendar", r"\.batch\.cal_ids"),
        ("static_type", r"\.batch\.row_names"),
    ],
)
def test_parity_detects(export_api, artifact, monkeypatch, drift, entry):
    if drift == "weight":
        weights = artifact / "model.safetensors"
        state = load_file(weights)
        state["head.projection.bias"][TINY.quantiles.index(0.5)] += 1e-2
        save_file(state, weights)
        _rehash(artifact, weights.name)
    else:
        path = artifact / "parity.npz"
        arrays = _read_archive(path)
        if drift == "scale":
            original = episode_builder.fit_scale

            def changed_scale(values, cutoff, season):
                fitted = original(values, cutoff, season)
                if not np.isfinite(values[:cutoff]).any():
                    return replace(fitted, loc=fitted.loc + 1e-3)
                return fitted

            monkeypatch.setattr(episode_builder, "fit_scale", changed_scale)
        elif drift == "mask":
            mask = arrays["case_1/batch/row_mask"]
            mask.flat[0] = not mask.flat[0]
        elif drift == "calendar":
            arrays["case_1/batch/cal_ids"].flat[0] += 1
        else:
            key = "case_1/item_0/static_values"
            assert json.loads(arrays[key][0]) == 1
            arrays[key] = np.asarray(["1.0", *arrays[key][1:]])
        np.savez_compressed(path, **arrays)
        _rehash(artifact, path.name)

    with pytest.raises(ValueError, match=entry):
        export_api.require_parity(artifact)


def test_parity_needs_no_generator_and_no_pickle(export_api, artifact, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("parity replay tried to regenerate a panel")

    _read_archive(artifact / "parity.npz")
    monkeypatch.setattr(typed, "generate_typed_panel", unexpected)
    monkeypatch.setattr(sources, "generate_typed_panel", unexpected)
    monkeypatch.setattr(np.random, "default_rng", unexpected)
    load = np.load

    def no_pickle(*args, **kwargs):
        assert kwargs.get("allow_pickle") is False
        return load(*args, **kwargs)

    monkeypatch.setattr(np, "load", no_pickle)
    export_api.require_parity(artifact)


def test_export_is_one_unit(tmp_path, export_api, monkeypatch):
    checkpoint = _checkpoint(tmp_path)
    out = tmp_path / "export"
    attempted = []

    def fail_after_weights(path, **arrays):
        staged = path.parent
        assert staged.parent == out.parent and staged != out
        assert (staged / "model.safetensors").is_file()
        attempted.append(staged)
        raise OSError("cannot write parity.npz")

    monkeypatch.setattr(np, "savez_compressed", fail_after_weights)
    with pytest.raises(OSError, match="cannot write parity"):
        export_api.write_export(checkpoint, out)
    assert not out.exists()
    assert len(attempted) == 1 and not attempted[0].exists()

    out.mkdir()
    marker = out / "keep"
    marker.write_bytes(b"original")
    with pytest.raises(ValueError, match="empty"):
        export_api.write_export(checkpoint, out)
    assert marker.read_bytes() == b"original"
    assert list(out.iterdir()) == [marker]


@pytest.mark.parametrize("filename", ["model.safetensors", "parity.npz"])
def test_digests_in_config_are_checked_on_load(export_api, artifact, filename):
    with (artifact / filename).open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match=filename.replace(".", r"\.")):
        export_api.load_export(artifact)


def test_command_line(tmp_path):
    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "demand_forecast.models.v5.export", *map(str, args)],
            capture_output=True,
            text=True,
            check=False,
        )

    first, second = tmp_path / "selftest", tmp_path / "written"
    result = run("selftest", first)
    assert result.returncode == 0, result.stderr
    result = run("write", "--checkpoint", _checkpoint(tmp_path), "--out", second)
    assert result.returncode == 0, result.stderr
    result = run("check", second, "--device", "cpu")
    assert result.returncode == 0, result.stderr
    assert "dated-peers" in result.stdout and "ragged-fallback" in result.stdout

    path = second / "parity.npz"
    arrays = _read_archive(path)
    arrays["case_1/batch/cal_ids"].flat[0] += 1
    np.savez_compressed(path, **arrays)
    _rehash(second, path.name)
    result = run("check", second)
    assert result.returncode == 1
    assert "cal_ids" in result.stdout
