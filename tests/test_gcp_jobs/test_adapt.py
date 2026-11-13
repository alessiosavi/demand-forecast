"""Offline task-local adaptation acceptance: no model or dataset downloads."""

import copy
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def adapt(upstream):
    from simons.training import adapt

    return adapt


@pytest.fixture
def tiny_model(adapt):
    import torch
    from simons import Simons, SimonsConfig

    torch.set_num_threads(1)
    torch.manual_seed(100)
    return Simons(
        SimonsConfig(
            context_length=64,
            near_context=32,
            prediction_length=8,
            patch_length=4,
            far_pool=2,
            width=16,
            layers=2,
            heads=2,
            period_slots=1,
            covariates=True,
            binary_rows=True,
            read_after=0,
        )
    )


@pytest.mark.parametrize(
    "scores,selected,stop,reason",
    [
        ([(1, 1), (1, 1), (1, 1)], 0, True, "no_qualifying_gain"),
        ([(1, 1), (0.9, 1.01), (0.89, 1)], 0, False, "no_qualifying_gain"),
        ([(1, 1), (0.97, 0.96), (0.99, 0.99), (1.1, 1.1)], 50, True, "inner_gain"),
        ([(1, 1), (0.98, 0.98), (0.97, 0.97)], 100, False, "inner_gain"),
        ([(1, 1), (0.9, 0.9), (np.nan, 0.8)], 0, True, "non_finite_evaluation"),
        ([(0, 0), (0, 0)], 0, False, "no_qualifying_gain"),
    ],
)
def test_selection(adapt, scores, selected, stop, reason):
    evaluations = [
        {"update": n * 50, "windows": [{"score": s} for s in pair]} for n, pair in enumerate(scores)
    ]
    decision = adapt.select(evaluations)
    assert decision["selected_update"] == selected
    assert decision["stop"] is stop
    assert decision["reason"] == reason


def test_metric_hand_case(adapt):
    from simons.training.adapt_eval import sql

    # Seasonal error = 2, |target - prediction| = 2; median's doubled pinball = 2.
    truth = np.array([[[6.0]]])
    past = [np.array([[0.0, 2.0, 4.0]])]
    prediction = np.array([[[[4.0]]]])
    np.testing.assert_array_equal(sql(truth, prediction, past, [0.5], 1), [[1.0]])


def test_metric_fev_source_parity(adapt):
    from conftest import module_at
    from simons.training.adapt_eval import sql

    # The supplied reference clone needs only NumPy, even when fev is not installed.
    source = Path(
        os.environ.get(
            "FEV_METRICS_SOURCE", "/Users/asavi/df-wins-2026-11-11/published/fev/src/fev/metrics.py"
        )
    )
    if source.is_file():
        fev = module_at("reference_fev_metrics", source)
    else:
        fev = pytest.importorskip("fev.metrics", reason="fev source/package unavailable")
    rng = np.random.default_rng(9)
    truth = rng.normal(size=(3, 5, 2))
    prediction = rng.normal(size=(3, 5, 2, 3))
    past = [rng.normal(size=(2, n)) for n in (12, 11, 4)]
    past[0][1] = 0  # zero scale is excluded, not treated as epsilon by default
    past[1][0, 3] = np.nan
    truth[1, 2, 1] = np.nan
    levels = [0.1, 0.5, 0.9]
    for epsilon in (0, 0.01):
        expected = fev.SQL(epsilon=epsilon).compute(
            y_true=truth,
            y_pred=prediction[..., 1],
            y_past=np.concatenate([p.T for p in past]),
            y_past_lengths=np.array([p.shape[-1] for p in past]),
            q_pred=prediction,
            seasonality=3,
            quantile_levels=levels,
        )
        np.testing.assert_allclose(
            sql(truth, prediction, past, levels, 3, epsilon).mean(-1)
            if np.ndim(expected)
            else np.mean(sql(truth, prediction, past, levels, 3, epsilon)),
            expected,
            rtol=1e-14,
        )


@pytest.mark.parametrize("params", ["head", "all"])
def test_gradient_freezing_and_masking(adapt, tiny_model, params):
    import torch
    from simons.training.adapt_data import horizon_loss
    from simons.training.optim import NorMuonAdamW

    model = tiny_model
    info = adapt.parameters(model, params)
    assert info["trainable_count"] + info["frozen_count"] == sum(
        p.numel() for p in model.parameters()
    )
    if params == "head":
        assert info["trainable_names"] == ["norm.weight", "head.weight", "head.bias"]
        assert info["trainable_count"] == 16 + 36 * 16 + 36
    before = {k: v.clone() for k, v in model.state_dict().items()}
    values = torch.randn(1, 2, 40)
    covs = torch.randn(1, 1, 40)
    optimizer = NorMuonAdamW(model, muon_lr=0.0002, adam_lr=0.00005)
    # Only [32,35) is supervised; trailing slots must be hidden and have no loss.
    loss, count = horizon_loss(model, values, covs, 32, 3)
    changed = values.clone()
    changed[..., 35:] += 1000
    changed_covs = covs.clone()
    changed_covs[..., 35:] += 1000
    other, _ = horizon_loss(model, changed, changed_covs, 32, 3)
    torch.testing.assert_close(loss, other, rtol=0, atol=0)
    assert count == 6
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    optimizer.step()
    for name, value in model.state_dict().items():
        if name in info["frozen_names"]:
            assert torch.equal(value, before[name]), name
    assert any(not torch.equal(model.state_dict()[n], before[n]) for n in grads)


@pytest.fixture
def bundle_factory(tmp_path):
    pytest.importorskip("fev", reason="AD1 bundles require fev; use yardsticks-venv")
    from conftest import ROOT, module_at

    inputs = module_at(
        "adapt_input_test_support", ROOT / "tests/test_bench/test_adaptation_data.py"
    )
    from demand_forecast.bench.adaptation.data import build_manifest

    def build(lengths=(104, 100), name="bundle", *, formatter=None, known=True, precise=False):
        import datasets

        rows, targets = inputs.task_rows(
            lengths=lengths, starts=(0,) * len(lengths), channels=2, known=known
        )
        for row in rows:
            n = len(row["timestamp"])
            if known:
                row["promo"] = (np.arange(n) % 2).astype(np.bool_)
                row["event"] = (["a", None, "None", "b"] * n)[:n]
            if precise:
                row["y0"] = 1e8 + np.arange(n, dtype=np.float64) * 0.01
                row["y0"][0] = np.nan
        task = inputs.SyntheticTask(rows, targets, cutoff=-8, known=known)
        task.load_full_dataset = lambda: datasets.Dataset.from_list(rows).with_format(formatter)
        task.horizon = 3
        task.quantile_levels = [0.1, 0.5, 0.9]
        yaml = tmp_path / "task.yaml"
        yaml.write_text("task_name: family/task\n")
        root = tmp_path / name
        build_manifest(task, task_yaml=yaml, output=root, max_context=64)
        return root

    return build


@pytest.fixture
def bundle(bundle_factory):
    return bundle_factory()


@pytest.fixture
def base(tiny_model, tmp_path):
    path = tmp_path / "base"
    tiny_model.save_pretrained(path)
    return path


def short_recipe(adapt, **overrides):
    return adapt.Recipe(
        **{
            "max_updates": 5,
            "eval_updates": (0, 2, 4, 5),
            "batch_size": 3,
            "microbatch_size": 2,
            "context": 32,
            "checkpoint_updates": 1,
            **overrides,
        }
    )


def report_equal(left, right):
    left, right = copy.deepcopy(left), copy.deepcopy(right)
    for report in (left, right):
        report.pop("wall_seconds")
        report.pop("training_wall_seconds")
        report.pop("peak_cuda_memory_bytes")
        for evaluation in report["evaluations"]:
            evaluation.pop("wall_seconds")
        if report["last_update"] is not None:
            report["last_update"].pop("wall_seconds")
    assert left == right


def test_native_sampling_deterministic_and_causal(adapt, bundle, tiny_model):
    import torch
    from simons.training.adapt_data import Bundle

    loaded = Bundle(bundle)
    sampler = loaded.sampler(tiny_model.config, 32)
    left = sampler.sample(np.random.default_rng(31))
    right = sampler.sample(np.random.default_rng(31))
    torch.testing.assert_close(left[0], right[0], rtol=0, atol=0, equal_nan=True)
    assert left[2] == right[2]
    assert left[0].shape == (1, 2, 40)
    assert left[1].shape == (1, 2, 40)
    for _ in range(10):
        values, _, evidence = sampler.sample(np.random.default_rng(_))
        assert evidence["origin"] + loaded.horizon <= 86
        assert int(torch.isfinite(values[..., :32]).sum()) == 64
        assert torch.isnan(values[..., 35:]).all()


def test_actual_evaluation_and_adapter_parity(adapt, bundle, tiny_model, monkeypatch):
    from simons import forecast_series
    from simons.training.adapt_data import Bundle, raw_model
    from simons.training.adapt_eval import evaluate

    loaded = Bundle(bundle)
    result = evaluate(tiny_model, loaded, batch_size=2)
    assert len(result) == 2 and all(np.isfinite(w["score"]) for w in result)
    assert all(len(w["groups"]) == 2 for w in result)
    # Evaluate really goes through the scoring adapter's public entry point.
    from simons.training import adapt_eval

    calls = []

    def forward(model, series, horizon, **kwargs):
        calls.append(kwargs)
        assert all(getattr(model.config, k) is False for k in ("views", "fit_row", "xreg"))
        return forecast_series(model, series, horizon, **kwargs)

    monkeypatch.setattr(adapt_eval, "forecast_series", forward)
    assert evaluate(tiny_model, loaded, batch_size=2) == result
    assert all(c["past_covariates"] is None and c["shared_categories"] is False for c in calls)
    assert tiny_model.config.views is True
    with raw_model(tiny_model):
        from demand_forecast.bench.yardsticks.adapters import SimonsAdapter
        from demand_forecast.bench.yardsticks.runner import Options

        # The exact adapter transformation is also exercised directly below.
        adapter = object.__new__(SimonsAdapter)
        adapter.model, adapter.forecast_series = tiny_model, forecast_series
        adapter.options = Options(
            model="simons",
            model_name="tiny",
            weights="unused",
            device="cpu",
            covariates="all",
            batch_size=2,
        )
        from types import SimpleNamespace

        inner = loaded.inners[0]
        items = []
        for item in inner["items"]:
            block = item["model_input"][0]
            target = np.asarray(block["target"], dtype=np.float32)
            known = {
                c: block["known_covariates"][c]
                for c in loaded.manifest["covariates"]["known_future"]
            }
            length = target.shape[-1]
            items.append(
                SimpleNamespace(
                    target=target,
                    past={k: v[:length] for k, v in known.items()},
                    future={k: v[length:] for k, v in known.items()},
                )
            )
        # WindowInput's other metadata is irrelevant to adapter.predict.
        outputs = adapter.predict([SimpleNamespace(items=items)], loaded.horizon, [0.1, 0.5, 0.9])
        direct = forecast_series(
            tiny_model,
            [np.asarray(i["model_input"][0]["target"], dtype=np.float32) for i in inner["items"]],
            loaded.horizon,
            future_covariates=[
                {
                    c: np.asarray(i["model_input"][0]["known_covariates"][c], dtype=object)
                    for c in loaded.manifest["covariates"]["known_future"]
                }
                for i in inner["items"]
            ],
            shared_categories=False,
            past_covariates=None,
            batch_size=2,
        )
        for expected, actual in zip(direct, outputs[0], strict=True):
            np.testing.assert_array_equal(actual, np.asarray(expected)[..., [0, 4, 8]])


def test_inner_labels_change_scores_never_training(adapt, bundle, base, tiny_model):
    import torch
    from simons.training.adapt_data import Bundle
    from simons.training.adapt_eval import evaluate
    from simons.training.optim import NorMuonAdamW

    loaded = Bundle(bundle)
    before = evaluate(tiny_model, loaded)
    altered = copy.deepcopy(loaded)
    for inner in altered.inners:
        for item in inner["items"]:
            item["score_target"] = (np.asarray(item["score_target"]) + 100).tolist()
    assert evaluate(tiny_model, altered) != before
    recipe = short_recipe(adapt)
    states = []
    for data in (loaded, altered):
        model = copy.deepcopy(tiny_model)
        adapt.parameters(model, "all")
        optimizer = NorMuonAdamW(model)
        measured = adapt.train_update(
            model, optimizer, data.sampler(model.config, 32), recipe, 11, 1
        )
        assert np.isfinite(measured["gradient_norm"])
        states.append({n: (p.detach().clone(), p.grad) for n, p in model.named_parameters()})
    for name in states[0]:
        for left, right in zip(states[0][name], states[1][name], strict=True):
            if left is not None:
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            else:
                assert right is None


def force_gains(monkeypatch, adapt):
    original = adapt.evaluate
    baseline = None

    def evaluate(model, bundle, batch_size):
        nonlocal baseline
        measured = original(model, bundle, batch_size)
        if baseline is None:
            baseline = float(model.head.weight.detach().sum())
        changed = float(model.head.weight.detach().sum()) != baseline
        return [{**w, "score": (0.9 if changed else 1.0) * (i + 1)} for i, w in enumerate(measured)]

    monkeypatch.setattr(adapt, "evaluate", evaluate)


@pytest.mark.parametrize("interrupt_event,step", [("evaluation", 2), ("checkpoint", 1)])
@pytest.mark.parametrize("qualify", [False, True])
def test_resume_identical(
    adapt, bundle, base, tmp_path, monkeypatch, interrupt_event, step, qualify
):
    from simons.training.reliability import latest_checkpoint
    from simons.training.trainer import read_checkpoint

    if qualify:
        force_gains(monkeypatch, adapt)
    recipe = short_recipe(adapt)
    full = tmp_path / "full"
    expected = adapt.run(bundle, base, full, recipe, seed=29)
    resumed = tmp_path / "resumed"

    def interrupt(event, state):
        if event == interrupt_event and state["step"] == step:
            raise InterruptedError("offline kill")

    with pytest.raises(InterruptedError):
        adapt.run(bundle, base, resumed, recipe, seed=29, observer=interrupt)
    committed = read_checkpoint(latest_checkpoint(resumed / "checkpoints"))["adaptation"]
    actual = adapt.run(bundle, base, resumed, recipe, seed=29)
    report_equal(expected, actual)
    assert actual["training_wall_seconds"] >= committed["training_wall_seconds"] > 0
    assert actual["evaluations"][: len(committed["evaluations"])] == committed["evaluations"]
    for name in ("config.json", "model.safetensors"):
        assert (resumed / name).read_bytes() == (full / name).read_bytes()
    assert actual["exposure"]["updates"] == actual["updates_run"]
    assert actual["exposure"]["grouped_windows"] == actual["updates_run"] * recipe.batch_size
    if qualify:
        assert actual["decision"]["selected"] == "adapted"
    again = adapt.run(bundle, base, resumed, recipe, seed=29)
    report_equal(actual, again)


def test_frozen_fallback_and_identity(adapt, bundle, base, tmp_path):
    recipe = short_recipe(adapt, max_updates=0)
    out = tmp_path / "out"
    report = adapt.run(bundle, base, out, recipe)
    assert report["decision"]["selected"] == "frozen_base"
    for name in ("config.json", "model.safetensors"):
        assert (out / name).read_bytes() == (base / name).read_bytes()
    for changed in (replace(recipe, params="all"), replace(recipe, lr_multiplier=0.1)):
        with pytest.raises(ValueError, match="identity"):
            adapt.run(bundle, base, out, changed)
    with pytest.raises(ValueError, match="identity"):
        adapt.run(bundle, base, out, recipe, seed=29)


@pytest.mark.parametrize("formatter", [None, "numpy"])
def test_checkpoint_nonce_and_export_parity(
    adapt, bundle_factory, base, tmp_path, monkeypatch, formatter
):
    from simons import Simons
    from simons.training.adapt_data import Bundle
    from simons.training.adapt_eval import evaluate
    from simons.training.trainer import read_checkpoint

    bundle = bundle_factory(formatter=formatter)
    force_gains(monkeypatch, adapt)
    output = tmp_path / "out"
    run_dir = tmp_path / "runner"
    run_dir.mkdir()
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    (run_dir / "checkpoint-request").write_text("first-nonce\n")
    seen = []

    def hook(event, state):
        if event == "update" and state["step"] == 1:
            assert (run_dir / "checkpoint-done").read_text().strip() == "first-nonce"
            (run_dir / "checkpoint-request").write_text("second-nonce\n")
        if event == "checkpoint":
            seen.append(state["acknowledged_nonce"])

    report = adapt.run(bundle, base, output, short_recipe(adapt), observer=hook)
    assert (run_dir / "checkpoint-done").read_text().strip() == "second-nonce"
    assert "first-nonce" in seen and "second-nonce" in seen
    saved = read_checkpoint(output / "checkpoints/checkpoint-final.pt")
    assert saved["adaptation"]["acknowledged_nonce"] == "second-nonce"
    selected = Simons.from_pretrained(output)
    in_memory = Simons.from_pretrained(base)
    in_memory.load_state_dict(saved["adaptation"]["best_model"])
    assert evaluate(selected, Bundle(bundle)) == evaluate(in_memory, Bundle(bundle))
    from simons import forecast_series
    from simons.training.adapt_data import raw_model

    loaded = Bundle(bundle)
    inner = loaded.inners[0]
    series = [
        np.asarray(item["model_input"][0]["target"], dtype=np.float32) for item in inner["items"]
    ]
    known = [
        {
            c: np.asarray(item["model_input"][0]["known_covariates"][c], dtype=object)
            for c in loaded.manifest["covariates"]["known_future"]
        }
        for item in inner["items"]
    ]
    predictions = []
    for model in (selected, in_memory):
        with raw_model(model):
            predictions.append(
                forecast_series(
                    model,
                    series,
                    3,
                    future_covariates=known,
                    shared_categories=False,
                    past_covariates=None,
                    batch_size=2,
                )
            )
    for left, right in zip(*predictions, strict=True):
        np.testing.assert_array_equal(left, right)
    assert report["parameters"]["frozen_count"] > 0


def rewrite_bundle(bundle, mutate):
    from simons.training.adapt_data import json_bytes, sha

    manifest = json.loads((bundle / "manifest.json").read_bytes())
    mutate(manifest)
    for name, spec in manifest["files"].items():
        spec.update(sha256=sha(bundle / name), bytes=(bundle / name).stat().st_size)
    import hashlib

    manifest.pop("fingerprint")
    manifest["fingerprint"] = hashlib.sha256(json_bytes(manifest)).hexdigest()
    (bundle / "manifest.json").write_bytes(json_bytes(manifest))


@pytest.mark.parametrize("version", [None, 1, 3, True, 2.0, "2"])
@pytest.mark.parametrize("payload", ["manifest.json", "inner-0.json", "inner-1.json"])
def test_bundle_refuses_other_schema_versions(adapt, bundle, version, payload):
    from simons.training.adapt_data import Bundle

    if payload == "manifest.json":
        rewrite_bundle(bundle, lambda m: m.update(schema_version=version))
    else:
        path = bundle / payload
        inner = json.loads(path.read_bytes())
        inner["schema_version"] = version
        path.write_text(json.dumps(inner))
        rewrite_bundle(bundle, lambda m: None)
    with pytest.raises(ValueError, match=r"schema_version.*2"):
        Bundle(bundle)


@pytest.mark.parametrize(
    "folder,column",
    [("univariate", "target"), ("covariates", "target"), ("covariates", "covariates")],
)
def test_native_schema_does_not_silently_cast_float64(adapt, bundle, tiny_model, folder, column):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from simons.training.adapt_data import Bundle

    path = bundle / folder / "train-00000.parquet"
    table = pq.read_table(path)
    value = pa.list_(pa.float64())
    dtype = value if (folder, column) == ("covariates", "target") else pa.list_(value)
    schema = table.schema.set(table.schema.get_field_index(column), pa.field(column, dtype))
    pq.write_table(table.cast(schema), path)
    rewrite_bundle(bundle, lambda m: None)
    with pytest.raises(ValueError, match="native.*schema"):
        Bundle(bundle).sampler(tiny_model.config, 32)


@pytest.mark.parametrize("formatter", [None, "numpy"])
@pytest.mark.parametrize("known", [False, True])
def test_schema2_evaluation_preserves_scorer_inputs_and_precision(
    adapt, bundle_factory, tiny_model, monkeypatch, formatter, known
):
    import fev.metrics
    from simons.covariates import prepare_covariates
    from simons.training import adapt_eval
    from simons.training.adapt_data import Bundle

    loaded = Bundle(bundle_factory(formatter=formatter, known=known, precise=True))
    sampler = loaded.sampler(tiny_model.config, 32)
    assert sampler.sample(np.random.default_rng(11))[0].numpy().dtype == np.float32
    if known:
        # Native rows keep training-only codes, in the manifest's column order.
        covs = sampler.items["item-0"][0][2]
        np.testing.assert_array_equal(covs[0, :4], [0, 1, 0, 1])
        np.testing.assert_equal(covs[1, :4], [1, np.nan, 0, 2])
    forecasts = []
    original = adapt_eval.forecast_series

    def record(model, series, horizon, **kwargs):
        assert all(s.dtype == np.float32 for s in series)
        for target, cov in zip(series, kwargs["future_covariates"], strict=True):
            if not known:
                assert cov is None
                continue
            assert all(v.dtype == object for v in cov.values())
            assert None in cov["event"] and "None" in cov["event"]
            assert all(isinstance(v, bool) for v in cov["promo"])
            # Simons must see a missing value, distinct from the literal category.
            encoded = prepare_covariates(
                target.shape[-1], horizon, future=cov, shared_categories=False
            )
            np.testing.assert_array_equal(encoded[0], np.asarray(cov["promo"], dtype=np.float32))
            assert np.isnan(encoded[1]).any()
        outputs = original(model, series, horizon, **kwargs)
        forecasts.append(outputs)
        return outputs

    monkeypatch.setattr(adapt_eval, "forecast_series", record)
    measured = adapt_eval.evaluate(tiny_model, loaded)
    for inner, outputs, actual in zip(loaded.inners, forecasts, measured, strict=True):
        items = inner["items"]
        truth = np.stack([np.asarray(i["score_target"], dtype=np.float64).T for i in items])
        past = [np.asarray(i["score_past"], dtype=np.float64).T for i in items]
        # A float32 cast would erase this channel's seasonal scale completely.
        observed = past[0][np.isfinite(past[0][:, 0]), 0]
        assert np.ptp(observed) > 0
        assert np.ptp(observed.astype(np.float32)) == 0
        q = np.stack(
            [np.sort(np.asarray(p), axis=-1)[..., [0, 4, 8]].transpose(1, 0, 2) for p in outputs]
        )
        expected = fev.metrics.SQL().compute(
            y_true=truth,
            y_pred=q[..., 1],
            q_pred=q,
            y_past=np.concatenate(past),
            y_past_lengths=np.array([len(p) for p in past]),
            quantile_levels=[0.1, 0.5, 0.9],
            seasonality=2,
        )
        assert actual["score"] == pytest.approx(float(np.mean(expected)), rel=1e-14)


def test_some_excluded_items_use_eligible_task_weights(
    adapt, bundle_factory, base, tmp_path, monkeypatch
):
    force_gains(monkeypatch, adapt)
    recipe = short_recipe(adapt, max_updates=2)
    eligible = adapt.run(bundle_factory(), base, tmp_path / "eligible", recipe)
    ragged_bundle = bundle_factory((104, 100, 12), "ragged-bundle")
    ragged = adapt.run(ragged_bundle, base, tmp_path / "ragged", recipe)
    assert ragged["decision"]["selected"] == "adapted"
    assert ragged["updates_run"] == 2
    assert ragged["excluded_items"] == [
        {
            "item_id": "item-2",
            "reason": "insufficient_inner_history",
            "fallback": "frozen_model",
        }
    ]
    assert ragged["exposure"] == eligible["exposure"]
    assert [e["windows"] for e in ragged["evaluations"]] == [
        e["windows"] for e in eligible["evaluations"]
    ]
    assert all(
        {g["item_id"] for g in window["groups"]} == {"item-0", "item-1"}
        for evaluation in ragged["evaluations"]
        for window in evaluation["windows"]
    )
    for name in ("config.json", "model.safetensors"):
        assert (tmp_path / "ragged" / name).read_bytes() == (
            tmp_path / "eligible" / name
        ).read_bytes()
    assert (
        json.loads((tmp_path / "ragged/adaptation.json").read_bytes())["excluded_items"]
        == (ragged["excluded_items"])
    )


def test_all_excluded_items_keep_frozen_base(adapt, bundle_factory, base, tmp_path):
    bundle = bundle_factory((12, 10))
    result = adapt.run(bundle, base, tmp_path / "out", short_recipe(adapt))
    assert result["decision"]["selected"] == "frozen_base"
    assert result["decision"]["reason"] == "all_items_excluded"
    assert result["updates_run"] == 0
    assert result["training_wall_seconds"] == 0
    assert [item["item_id"] for item in result["excluded_items"]] == ["item-0", "item-1"]
    assert all(item["reason"] == "insufficient_inner_history" for item in result["excluded_items"])
    for name in ("config.json", "model.safetensors"):
        assert (tmp_path / "out" / name).read_bytes() == (base / name).read_bytes()


def test_cost_receipt_separates_training_and_evaluations(adapt, bundle, base, tmp_path):
    from simons.training.trainer import read_checkpoint

    output = tmp_path / "out"
    result = adapt.run(bundle, base, output, short_recipe(adapt, max_updates=2))
    assert result["training_wall_seconds"] > 0
    assert 0 < result["last_update"]["wall_seconds"] <= result["training_wall_seconds"]
    assert all(e["wall_seconds"] > 0 for e in result["evaluations"])
    assert (
        result["training_wall_seconds"] + sum(e["wall_seconds"] for e in result["evaluations"])
        <= result["wall_seconds"]
    )
    assert result["peak_cuda_memory_bytes"] is None
    assert (
        json.loads((output / "adaptation.json").read_bytes())["training_wall_seconds"]
        == (result["training_wall_seconds"])
    )
    saved = read_checkpoint(output / "checkpoints/checkpoint-final.pt")["adaptation"]
    assert saved["training_wall_seconds"] == result["training_wall_seconds"]
    resumed = adapt.run(bundle, base, output, short_recipe(adapt, max_updates=2))
    assert resumed["training_wall_seconds"] == result["training_wall_seconds"]
    assert resumed["evaluations"] == result["evaluations"]


def test_cuda_cost_clock_waits_for_work_and_preserves_peak(adapt, monkeypatch):
    import torch

    # Exercise the CUDA measurement boundary without requiring a GPU on this host.
    pending = [30.0, 70.0]
    clock = 0.0

    def synchronize(device):
        nonlocal clock
        assert device == torch.device("cuda:1")
        clock = pending.pop(0)

    def reset(device):
        assert device == torch.device("cuda:1")
        pending[:] = [3.0, 7.0]

    def peak(device):
        assert device == torch.device("cuda:1")
        return 256

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", peak)
    monkeypatch.setattr(adapt.time, "monotonic", lambda: clock)
    meter = adapt.CostMeter("cuda:1")
    start = meter.now()
    assert meter.now() - start == 4.0
    assert meter.peak_bytes(None) == 256
    assert meter.peak_bytes(512) == 512


@pytest.mark.parametrize("failure", ["empty", "nonfinite"])
def test_coverage_fallback(adapt, bundle, base, tmp_path, monkeypatch, failure):
    if failure == "empty":
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = bundle / "univariate/train-00000.parquet"
        table = pq.read_table(path)
        pq.write_table(pa.Table.from_batches([], schema=table.schema), path)
        rewrite_bundle(bundle, lambda m: None)
    else:
        from simons.training import adapt_eval

        monkeypatch.setattr(
            adapt_eval,
            "forecast_series",
            lambda model, series, horizon, **kw: [
                np.full((len(s), horizon, 9), np.nan) for s in series
            ],
        )
    result = adapt.run(bundle, base, tmp_path / "out", short_recipe(adapt))
    assert result["decision"]["selected"] == "frozen_base"
    assert result["updates_run"] == 0
    assert (
        result["decision"]["reason"]
        == {
            "empty": "empty_training_set",
            "nonfinite": "non_finite_forecast",
        }[failure]
    )
    assert (tmp_path / "out/model.safetensors").read_bytes() == (
        base / "model.safetensors"
    ).read_bytes()


def test_payload_tampering_refused(adapt, bundle):
    from simons.training.adapt_data import Bundle

    (bundle / "inner-1.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        Bundle(bundle)


def test_sql_invalid_channel_propagates_like_fev(adapt, bundle, tiny_model):
    from simons.training.adapt_data import Bundle
    from simons.training.adapt_eval import evaluate

    loaded = Bundle(bundle)
    for inner in loaded.inners:
        for item in inner["items"]:
            item["score_past"][1] = [0.0] * len(item["score_past"][1])
    # fev averages channel scores with np.mean, so an entirely undefined channel
    # must invalidate the overall SQL instead of silently narrowing coverage.
    assert all(window["score"] is None for window in evaluate(tiny_model, loaded))


def test_resume_rejects_changed_inputs(adapt, bundle, base, tmp_path):
    recipe = short_recipe(adapt, max_updates=0)
    output = tmp_path / "out"
    adapt.run(bundle, base, output, recipe)
    rewrite_bundle(bundle, lambda m: m.update(phase="confirmation"))
    with pytest.raises(ValueError, match="identity"):
        adapt.run(bundle, base, output, recipe)
    # A changed config byte is an identity change even if parsed settings match.
    (base / "config.json").write_bytes((base / "config.json").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="identity"):
        adapt.run(bundle, base, output, recipe)


def test_checkpoint_ack_recovers_after_interruption(adapt, bundle, base, tmp_path, monkeypatch):
    output = tmp_path / "out"
    run_dir = tmp_path / "runner"
    run_dir.mkdir()
    (run_dir / "checkpoint-request").write_text("nonce\n")
    monkeypatch.setenv("RUN_DIR", str(run_dir))

    def interrupt(event, state):
        if event == "checkpoint" and state["acknowledged_nonce"] == "nonce":
            raise InterruptedError("killed between durable save and ack")

    recipe = short_recipe(adapt)
    with pytest.raises(InterruptedError):
        adapt.run(bundle, base, output, recipe, observer=interrupt)
    assert not (run_dir / "checkpoint-done").exists()
    adapt.run(bundle, base, output, recipe)
    assert (run_dir / "checkpoint-done").read_text() == "nonce\n"


def test_partial_publication_repairs_from_final(adapt, bundle, base, tmp_path, monkeypatch):
    output = tmp_path / "out"
    recipe = short_recipe(adapt, max_updates=0)
    original = adapt.atomic_bytes

    def fail(path, data):
        if path.name == "model.safetensors":
            raise InterruptedError("killed during export")
        original(path, data)

    monkeypatch.setattr(adapt, "atomic_bytes", fail)
    with pytest.raises(InterruptedError):
        adapt.run(bundle, base, output, recipe)
    monkeypatch.setattr(adapt, "atomic_bytes", original)
    report = adapt.run(bundle, base, output, recipe)
    assert report["decision"]["selected"] == "frozen_base"
    assert (output / "model.safetensors").read_bytes() == (base / "model.safetensors").read_bytes()


def test_cli_zero_update_export(adapt, upstream, bundle, base, tmp_path):
    import subprocess
    import sys

    output = tmp_path / "cli-output"
    environment = {**os.environ, "PYTHONPATH": str(upstream / "src"), "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "simons.training.adapt",
            "--bundle",
            str(bundle),
            "--base",
            str(base),
            "--output",
            str(output),
            "--params",
            "head",
            "--lr-multiplier",
            "0.10",
            "--max-updates",
            "0",
            "--seed",
            "29",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["selected"] == "frozen_base"
    report = json.loads((output / "adaptation.json").read_bytes())
    assert report["seed"] == 29 and report["recipe"]["lr_multiplier"] == 0.10
    assert (output / "model.safetensors").read_bytes() == (base / "model.safetensors").read_bytes()
