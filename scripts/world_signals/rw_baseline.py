"""Zero-return random walk: every quantile is degenerate at zero."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast.bench.run import _save_predictions, _save_summaries, predictions_for_fev
from demand_forecast.bench.specs import dataset_fingerprint, load_task_specs


def rw_window_predictions(item_ids, horizon, quantiles):
    """Return fev rows with predictions and every (degenerate) quantile at 0.0.

    IDs specify the output order. These are zero returns, equivalent to a flat
    last-close price path; they are not calibrated probabilistic forecasts.
    """
    if horizon <= 0 or len(set(item_ids)) != len(item_ids):
        raise ValueError("Positive horizon and unique item IDs required")
    return [
        {"predictions": [0.0] * horizon, **{str(q): [0.0] * horizon for q in quantiles}}
        for _ in item_ids
    ]


def run_baseline(specs, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summaries.csv"
    summaries = (
        pd.read_csv(path, float_precision="round_trip")
        if path.exists()
        else pd.DataFrame(columns=["task", "model_name"])
    )
    for spec in specs:
        task = spec.to_task()
        identity = hashlib.sha256(
            json.dumps(
                {
                    "model": "random_walk",
                    "task": spec.task_definition(),
                    "dataset": dataset_fingerprint(task),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        directory = out_dir / spec.name
        directory.mkdir(exist_ok=True)
        expected = {f"window_{i}.npz" for i in range(spec.num_windows)}
        if {path.name for path in directory.glob("window_*.npz")} - expected:
            raise ValueError(f"Unexpected random-walk prediction windows: {directory}")
        completed = summaries.loc[
            (summaries.task == spec.name) & (summaries.model_name == "random_walk")
        ]
        if len(completed) > 1:
            raise ValueError(f"Duplicate random_walk rows: {spec.name}")
        if not completed.empty and completed.iloc[0].get("run_identity") != identity:
            raise ValueError(
                f"Random-walk identity mismatch: {spec.name}; use a new output directory"
            )
        predictions = []
        all_saved = True
        for index, window in enumerate(task.iter_windows(num_proc=1)):
            ids = [str(row[task.id_column]) for row in window.get_ground_truth()]
            target = directory / f"window_{index}.npz"
            if target.exists():
                with np.load(target, allow_pickle=False) as saved:
                    if saved["identity"].item() != identity:
                        raise ValueError(f"Random-walk identity mismatch: {target}")
                    aligned = predictions_for_fev(
                        saved["ids"].tolist(), saved["arrays"], ids, task.quantile_levels
                    )
                    if saved["arrays"].shape != (
                        len(ids),
                        task.horizon,
                        len(task.quantile_levels),
                    ) or np.any(saved["arrays"] != 0):
                        raise ValueError(f"Invalid random-walk predictions: {target}")
            else:
                all_saved = False
                aligned = rw_window_predictions(ids, task.horizon, task.quantile_levels)
                _save_predictions(
                    target,
                    ids,
                    np.zeros((len(ids), task.horizon, len(task.quantile_levels))),
                    identity,
                    {
                        "inference_time_s": 0.0,
                        "fallback_no_context": 0,
                        "fallback_undefined_scale": 0,
                        "device": "cpu",
                        "cold_covariate_columns": 0,
                        "clipped_covariate_values": 0,
                        "floored_covariate_columns": 0,
                    },
                )
            predictions.append(aligned)
        if len(completed) == 1 and all_saved:
            print(f"Skipping completed task {spec.name} / random_walk")
            continue
        summary = {
            "task": spec.name,
            **task.evaluation_summary(
                predictions,
                model_name="random_walk",
                training_time_s=0,
                inference_time_s=0,
                extra_info={"run_identity": identity, "quantiles_degenerate": True},
            ),
        }
        summaries = summaries.loc[
            ~((summaries.task == spec.name) & (summaries.model_name == "random_walk"))
        ]
        row = pd.DataFrame([summary])
        summaries = row if summaries.empty else pd.concat([summaries, row], ignore_index=True)
        _save_summaries(summaries, path)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_baseline(load_task_specs(args.tasks, args.profiles), args.out_dir)


if __name__ == "__main__":
    main()
