#!/usr/bin/env python3
"""EC2-only real-data parity gate; --self-test uses a tiny offline fixture, no training."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

# The reference oracle intentionally lives in this checkout, not an installed test package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demand_forecast.bench.run import resolved_origin, settings_for_task, window_origins  # noqa: E402
from demand_forecast.bench.specs import load_task_specs  # noqa: E402
from demand_forecast.config import Settings  # noqa: E402
from demand_forecast.core.evaluation import evaluate_prepared, prepare_validation  # noqa: E402
from demand_forecast.core.validation import plan_inner_validation  # noqa: E402
from demand_forecast.data.encoders import StaticEncoder  # noqa: E402
from demand_forecast.data.frequency import steps_between  # noqa: E402
from demand_forecast.data.loader_fev import panel_from_window  # noqa: E402
from demand_forecast.data.panel import Panel  # noqa: E402
from demand_forecast.data.sampler import BatchBuilder, WindowSampler, align_to_origin, build_origins  # noqa: E402
from demand_forecast.models.base import ModelSpec  # noqa: E402
from demand_forecast.models.registry import build_model  # noqa: E402
from tests.helpers.reference_batch import (  # noqa: E402
    assert_batches_equal,
    reference_align_to_origin,
    reference_batches,
    reference_evaluate,
    reference_make_batch,
)


def check_task(spec, settings, batches, device, emit, *, batch_size=128, inference_size=512):
    task = spec.to_task()
    window = next(iter(task.iter_windows()))
    past, future = window.get_input_data()
    panel = panel_from_window(past, future, spec, task.freq)  # exactly one panel build
    settings = settings_for_task(spec, settings, task.freq)
    cfg = settings.forecast
    origins = window_origins(future, spec.id_column, spec.timestamp_column)
    origin = resolved_origin(origins)
    plan = plan_inner_validation(panel, origin, cfg.horizon, settings.training.min_train_history)
    encoder = StaticEncoder().fit(panel, plan.fit_ids)
    facts = dict(
        distinct=len(set(origins.values())),
        earliest=str(min(origins.values())),
        latest=str(origin),
        train_upto=str(plan.train_upto),
        val_origin=str(plan.val_origin),
        n_fit_ids=len(plan.fit_ids),
        n_val_items=len(plan.val_items),
    )
    if spec.name == "walmart":
        assert facts["distinct"] == 11, facts
        for key, expected in (
            ("earliest", "2011-10-14"),
            ("latest", "2012-02-03"),
            ("train_upto", "2011-04-29"),
            ("val_origin", "2011-05-06"),
        ):
            assert facts[key][:10] == expected, facts
    emit("origin facts", facts)

    candidates = build_origins(panel, cfg.horizon, cfg.min_context, upto=plan.train_upto)
    fit_ids = set(plan.fit_ids)
    fit_indices = [i for i, series in enumerate(panel.series) if series.series_id in fit_ids]
    candidates = candidates[np.isin(candidates[:, 0], fit_indices)]
    assert len(candidates), "no training origins to compare"
    samplers = [
        WindowSampler(
            panel,
            candidates,
            cfg.horizon,
            cfg.max_context,
            encoder,
            seed=0,
            label_upto=plan.train_upto,
        )
        for _ in range(2)
    ]
    compared = 0
    while compared < batches:
        for actual, expected in zip(
            samplers[0].batches(batch_size), reference_batches(samplers[1], batch_size), strict=True
        ):
            assert_batches_equal(actual, expected)
            assert samplers[0].state() == samplers[1].state()
            compared += 1
            if compared == batches:
                break
        for sampler in samplers:
            sampler.set_epoch(sampler.epoch + 1)
    emit("training batches", {"batches": compared, "batch_size": batch_size})

    assert len(plan.val_items), "no validation items to compare"
    prepared = prepare_validation(
        panel,
        plan.val_items,
        plan.val_origin,
        cfg.horizon,
        cfg.max_context,
        encoder,
        inference_size,
    )
    positions = [
        (int(i), steps_between(panel.series[i].start, plan.val_origin, panel.freq))
        for i in plan.val_items
    ]
    aligned = reference_align_to_origin(panel, positions)
    for start, batch in zip(
        range(0, len(positions), inference_size), prepared.batches, strict=True
    ):
        assert_batches_equal(
            batch,
            reference_make_batch(
                aligned,
                positions[start : start + inference_size],
                cfg.horizon,
                cfg.max_context,
                encoder,
            ),
        )
    emit(
        "prepared validation",
        {"items": len(positions), "batches": len(prepared.batches), "nbytes": prepared.nbytes},
    )
    del aligned

    # Match Predictor.forecast_per_item_origins: group first, then chunk in item order.
    groups = {}
    for series in panel.series:
        groups.setdefault(origins[series.series_id], []).append(series)
    inference_batches = 0
    for group_origin, series in groups.items():
        group = Panel(series, panel.roles, panel.freq, panel.seasonal_period, panel.target_support)
        items = [
            (i, max(steps_between(s.start, group_origin, panel.freq), 0))
            for i, s in enumerate(series)
        ]
        builder = BatchBuilder(align_to_origin(group, items), cfg.horizon, cfg.max_context, encoder)
        reference_panel = reference_align_to_origin(group, items)
        for start in range(0, len(items), inference_size):
            chunk = items[start : start + inference_size]
            assert_batches_equal(
                builder.build(chunk),
                reference_make_batch(reference_panel, chunk, cfg.horizon, cfg.max_context, encoder),
            )
            inference_batches += 1
    emit("inference batches", {"batches": inference_batches, "origins": len(groups)})

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    model_spec = ModelSpec.from_panel(
        panel, encoder, cfg, settings.model.d_model, settings.model.cat_emb_dim
    )
    model = build_model(settings.model.family, model_spec, settings.model.backbone_kwargs()).to(
        device
    )
    expected = reference_evaluate(
        model,
        panel,
        plan.val_items,
        plan.val_origin,
        cfg.horizon,
        cfg.max_context,
        encoder,
        cfg.quantiles,
        inference_size,
    )
    actual = evaluate_prepared(model, prepared, cfg.quantiles)
    np.testing.assert_equal(actual, expected)  # exact, including NaN == NaN
    emit("evaluation", {"device": str(device), "metrics": len(actual)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="m5_1W,hermes,walmart")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.batches < 1:
        parser.error("--batches must be positive")
    device = torch.device("cpu" if args.self_test else "cuda")
    report = {"self_test": args.self_test, "device": str(device), "checks": []}
    status = 0
    with TemporaryDirectory(prefix="fev-parity-") as temporary:
        try:
            if args.self_test:
                from tests.helpers.fev_fixture import write_fev_fixture
                from tests.helpers.fev_spec import fixture_spec
                from tests.helpers.settings import tiny_settings

                specs = [
                    fixture_spec(
                        write_fev_fixture(
                            Path(temporary), length=80, lengths=[80, 80, 80, 75, 72, 60]
                        )
                    )
                ]
                settings = tiny_settings(base_steps=2, d_model=8)
            else:
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "real-data parity requires EC2 CUDA; use --self-test locally"
                    )
                settings = Settings.from_yaml(
                    Path(__file__).resolve().parents[2] / "configs/fev-v2-standard.yaml"
                )
                specs = load_task_specs(args.tasks)
            for spec in specs:

                def emit(check, details):
                    row = {"task": spec.name, "check": check, "status": "pass", **details}
                    report["checks"].append(row)
                    print(json.dumps(row), flush=True)

                try:
                    check_task(
                        spec,
                        settings,
                        args.batches,
                        device,
                        emit,
                        **({"batch_size": 2, "inference_size": 2} if args.self_test else {}),
                    )
                except Exception as exc:
                    status = 1
                    row = {
                        "task": spec.name,
                        "check": "parity",
                        "status": "fail",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    report["checks"].append(row)
                    print(json.dumps(row), flush=True)
        except Exception as exc:
            status = 1
            report["checks"].append({"check": "setup", "status": "fail", "error": str(exc)})
            print(f"FAIL setup: {exc}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
