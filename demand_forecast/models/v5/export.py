"""Publish v5 weights and replayable CPU references for scoring environments."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping
from dataclasses import asdict, fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from demand_forecast.data.episode import (
    DERIVABLE,
    EpisodeBatch,
    EpisodeBuilder,
    EpisodeRequest,
    Item,
    Panel,
    ScreeningConfig,
    collate,
    is_missing,
    label_of,
)
from demand_forecast.pretrain.v5 import (
    OptimConfig,
    SamplerConfig,
    V5TrainConfig,
    file_digest,
    identity,
    read_payload,
    save_checkpoint,
)
from demand_forecast.pretrain.v5.loop import _configuration

from .config import V5Config
from .model import V5Model

_FORMAT = "v5-export-1"
_WEIGHTS = "model.safetensors"
_PARITY = "parity.npz"
_CONFIG = "config.json"
_CASE_COUNT = 8
_BATCH_TOLERANCE = 1e-6
_CPU_TOLERANCE = 1e-4
_DEVICE_TOLERANCE = 2e-3

_Arrays = Mapping[str, np.ndarray]
_Differences = dict[str, float | bool]


def _json_array(value: object) -> np.ndarray:
    return np.asarray(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _read_json(array: np.ndarray):
    return json.loads(array.item())


def _store_panel(arrays: dict[str, np.ndarray], prefix: str, panel: Panel) -> None:
    items = []
    for i, item in enumerate(panel.items):
        base = f"{prefix}/item_{i}"
        for j, values in enumerate(item.targets.values()):
            arrays[f"{base}/target_{j}"] = values
        for j, values in enumerate(item.dynamic.values()):
            key = f"{base}/dynamic_{j}"
            if values.dtype == object:
                missing = [is_missing(value) for value in values]
                arrays[key] = np.asarray(
                    ["" if absent else label_of(value) for value, absent in zip(values, missing)],
                    dtype=np.str_,
                )
                arrays[f"{key}_missing"] = np.asarray(missing, dtype=bool)
            else:
                arrays[key] = values

        # JSON scalars preserve bool, int, float and str inside Unicode arrays.
        # Offsets delimit cells, including empty tuples, in the flat values array.
        offsets, tuples, values = [0], [], []
        for cell in item.static.values():
            is_tuple = isinstance(cell, tuple)
            tuples.append(is_tuple)
            values.extend(
                json.dumps(value, allow_nan=False) for value in (cell if is_tuple else (cell,))
            )
            offsets.append(len(values))
        arrays[f"{base}/static_offsets"] = np.asarray(offsets, dtype=np.int64)
        arrays[f"{base}/static_is_tuple"] = np.asarray(tuples, dtype=bool)
        arrays[f"{base}/static_values"] = np.asarray(values, dtype=np.str_)
        items.append(
            {
                "id": item.item_id,
                "start": item.start,
                "targets": list(item.targets),
                "dynamic": list(item.dynamic),
                "static": list(item.static),
            }
        )
    arrays[f"{prefix}/panel"] = _json_array(
        {
            "name": panel.name,
            "freq": panel.freq,
            "epoch": panel.epoch.isoformat() if panel.epoch is not None else None,
            "aligned": panel.aligned,
            "roles": panel.roles,
            "items": items,
        }
    )


def _read_panel(arrays: _Arrays, prefix: str) -> Panel:
    data = _read_json(arrays[f"{prefix}/panel"])
    items = []
    for i, item in enumerate(data["items"]):
        base = f"{prefix}/item_{i}"
        targets = {name: arrays[f"{base}/target_{j}"] for j, name in enumerate(item["targets"])}
        dynamic = {}
        for j, name in enumerate(item["dynamic"]):
            key = f"{base}/dynamic_{j}"
            values = arrays[key]
            if values.dtype.kind == "U":
                values = values.astype(object)
                values[arrays[f"{key}_missing"]] = None
            dynamic[name] = values

        offsets = arrays[f"{base}/static_offsets"]
        tuples = arrays[f"{base}/static_is_tuple"]
        values = arrays[f"{base}/static_values"]
        static = {}
        for j, name in enumerate(item["static"]):
            cell = tuple(json.loads(value) for value in values[offsets[j] : offsets[j + 1]])
            static[name] = cell if tuples[j] else cell[0]
        items.append(Item(item["id"], item["start"], targets, dynamic, static))
    epoch = None if data["epoch"] is None else pd.Timestamp(data["epoch"])
    return Panel(data["name"], tuple(items), data["roles"], data["freq"], epoch, data["aligned"])


def _store_requests(requests: list[EpisodeRequest]) -> np.ndarray:
    stored = []
    for request in requests:
        data = asdict(request)
        del data["derive"]
        data["drop"] = sorted(request.drop)
        if request.keep_columns is not None:
            data["keep_columns"] = sorted(request.keep_columns)
        if request.derive:
            data["derive"] = sorted(request.derive)
        stored.append(data)
    return _json_array(stored)


def _read_requests(array: np.ndarray) -> list[EpisodeRequest]:
    requests = []
    for request in _read_json(array):
        request["origins"] = tuple(request["origins"])
        request["drop"] = frozenset(request["drop"])
        request["derive"] = frozenset(request.get("derive", ()))
        if request["keep_columns"] is not None:
            request["keep_columns"] = frozenset(request["keep_columns"])
        requests.append(EpisodeRequest(**request))
    return requests


def _parity_cases(
    screening: ScreeningConfig,
    derive: frozenset[str],
) -> Iterator[tuple[Panel, list[EpisodeRequest]]]:
    """Author small inputs that exercise the scoring builder's distinct paths."""
    rng = np.random.default_rng(2026)
    steps = np.arange(56, dtype=np.float64)
    epoch = pd.Timestamp("2024-01-15")

    def item(i, count=1, *, dynamic=None, static=None):
        targets = {
            f"y{j:02d}": (i + 1) * np.sin(steps / (2.3 + i / 5 + j)) + rng.normal(size=len(steps))
            for j in range(count)
        }
        return Item(f"item-{i}", 0, targets, dynamic or {}, static or {})

    def regular(panel: Panel) -> tuple[Panel, list[EpisodeRequest]]:
        return panel, EpisodeBuilder(panel, screening).plan_requests(
            0, (48,), 8, context=48, derive=derive
        )

    # Both spellings have peers. Changing the first cell from 1 to 1.0 changes its group.
    groups = (1, "1", "1", "1", "1.0", "1.0", "1.0", "1.0", "two", "two", "two", None)
    # The two observed flag levels tie at the cutoff, with 1 encountered first.
    flag = 1 - steps % 2
    flag[[2, 3]] = np.nan
    peers = tuple(
        item(i, dynamic={"flag": flag}, static={"group": group}) for i, group in enumerate(groups)
    )
    yield regular(Panel("dated-peers", peers, {"flag": "known"}, "D", epoch, True))

    numbers = {"known": np.cos(steps / 3), "past": np.sin(steps / 6)}
    numbers["known"][3:7] = numbers["known"][2]
    numbers["known"][[11, 49]] = np.nan
    numbers["past"][13] = np.nan
    yield regular(
        Panel(
            "undated-numbers",
            (item(0, dynamic=numbers),),
            {"known": "known", "past": "past"},
            None,
            None,
            False,
        )
    )
    yield regular(Panel("joint-five", (item(0, 5),), {}, None, None, False))

    with_static = tuple(item(i, 2, static={"group": "odd" if i % 2 else "even"}) for i in range(8))
    yield regular(Panel("typed-two", with_static, {}, None, None, True))
    yield regular(Panel("split-seventeen", (item(0, 17),), {}, None, None, False))

    known = np.resize(np.array(["work", "special", "off", "work", "off"], dtype=object), len(steps))
    known[[5, 7, 48, 52, 55]] = None
    past = np.resize(np.array(["low", "mid", "mid", "high"], dtype=object), len(steps))
    past[[6, 27]] = None
    past[48:] = None
    dated_item = item(0, dynamic={"event": known, "state": past})
    dated_item.targets["y00"][[4, 11, 19]] = np.nan
    dated_panel, dated_requests = regular(
        Panel(
            "dated-categories",
            (dated_item,),
            {"event": "known", "state": "past"},
            "M",
            pd.Timestamp("2024-01-31"),
            True,
        )
    )
    expanded_requests = [
        replace(request, drop=frozenset({"value_stats"})) for request in dated_requests
    ]
    yield dated_panel, dated_requests + expanded_requests

    mixed = tuple(
        item(
            i,
            static={
                # Keep at least eight co-observations after excluding missing cells.
                "size": None if i == 9 else i // 3 + 0.25,
                "labels": () if i == 8 else ("shared", "odd" if i % 2 else "even"),
                "tier": i % 3,
                "active": i % 3 == 0,
            },
        )
        for i in range(10)
    )
    yield regular(Panel("mixed-statics", mixed, {}, None, None, False))

    ragged = (
        Item("constant", 0, {"y": np.full(56, 6.0)}, {}, {}),
        Item("missing", 2, {"y": np.full(41, np.nan)}, {}, {}),
        Item("one-step", 11, {"y": np.array([3.0] + [np.nan] * 8)}, {}, {}),
    )
    yield (
        Panel("ragged-fallback", ragged, {}, None, None, True),
        [
            EpisodeRequest(i, (origin,), 8, context=context, derive=derive)
            for i, (origin, context) in enumerate(((48, 48), (35, 32), (12, 16)))
        ],
    )

    if derive.intersection(("naive_rows", "naive_anchor")):
        steps = np.arange(240, dtype=np.float64)
        values = steps / 30 + np.sin(2 * np.pi * steps / 24) + np.sin(2 * np.pi * steps / 168)
        hourly = Item("hourly", 13, {"y": values}, {}, {})
        yield (
            Panel("hourly-naive", (hourly,), {}, "h", epoch, True),
            [EpisodeRequest(0, (hourly.start + 192,), 37, context=64, derive=derive)],
        )


def _screening(training: dict) -> ScreeningConfig:
    sampler = training["sampler"]
    return ScreeningConfig(
        no_evidence=sampler.get("screen_no_evidence", 0.0),
        fair_static_spread=sampler.get("screen_fair_spread", False),
        scale_floor=sampler.get("scale_floor"),
    )


def _batch_arrays(batch: EpisodeBatch) -> Iterator[tuple[str, np.ndarray]]:
    for field in fields(batch):
        value = getattr(batch, field.name)
        yield field.name, value.numpy() if isinstance(value, torch.Tensor) else _json_array(value)


@torch.inference_mode()
def _output_arrays(model: V5Model, batch: EpisodeBatch) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    with torch.autocast(device_type=device.type, enabled=False):
        output = model(batch.to(device))
    return {
        "quantiles": output.quantiles.cpu().numpy(),
        **{
            f"gates.{field.name}": getattr(output.gates, field.name).cpu().numpy()
            for field in fields(output.gates)
        },
    }


def _write_parity(
    path: Path, model: V5Model, screening: ScreeningConfig, derive: frozenset[str]
) -> None:
    arrays = {}
    for i, (panel, requests) in enumerate(_parity_cases(screening, derive), start=1):
        prefix = f"case_{i}"
        _store_panel(arrays, prefix, panel)
        arrays[f"{prefix}/requests"] = _store_requests(requests)
        builder = EpisodeBuilder(panel, screening)
        batch = collate([builder.build(request) for request in requests])
        for name, value in _batch_arrays(batch):
            arrays[f"{prefix}/batch/{name}"] = value
        for name, value in _output_arrays(model, batch).items():
            arrays[f"{prefix}/output/{name}"] = value
    if any(value.dtype.hasobject for value in arrays.values()):
        raise TypeError("v5 parity cannot store object arrays")
    np.savez_compressed(path, **arrays)


def _load_model(config: dict, state: dict[str, torch.Tensor]) -> V5Model:
    with torch.random.fork_rng(devices=[]):
        model = V5Model(V5Config(**config))
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def write_export(checkpoint: Path, out: Path) -> dict:
    """Publish one verified export directory from the checkpoint's selected weights."""
    checkpoint, out = Path(checkpoint), Path(out)
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise ValueError(f"export destination must be absent or an empty directory: {out}")

    before = checkpoint.stat()
    saved = read_payload(checkpoint)
    source_digest = file_digest(checkpoint)
    after = checkpoint.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"checkpoint changed while being read: {checkpoint}")

    key = saved["export"]
    if key not in ("model", "ema") or saved[key] is None:
        raise ValueError("unavailable checkpoint weights: export")
    weights = saved[key]
    training = _configuration(saved["config"]).canonical()
    metadata = {
        "format": _FORMAT,
        "model": training["model"],
        "training": training,
        "source": {
            "sha256": source_digest,
            "identity": saved["identity"],
            "updates": saved["counters"]["updates"],
            "weights": "ema" if key == "ema" else "raw",
        },
    }
    del saved

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as temporary:
        staging = Path(temporary)
        save_file(
            {
                name: value.detach().clone(memory_format=torch.contiguous_format)
                for name, value in weights.items()
            },
            staging / _WEIGHTS,
        )
        model = _load_model(metadata["model"], weights).float()
        _write_parity(
            staging / _PARITY,
            model,
            _screening(training),
            frozenset(training["sampler"].get("derive", ())),
        )
        metadata["files"] = {name: file_digest(staging / name) for name in (_WEIGHTS, _PARITY)}
        (staging / _CONFIG).write_text(
            json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        require_parity(staging)
        staging.rename(out)
    return metadata


def load_export(directory: Path, device: str = "cpu") -> tuple[V5Model, dict]:
    """Validate the format and both digests before strictly loading the weights."""
    directory = Path(directory)
    metadata = json.loads((directory / _CONFIG).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("format") != _FORMAT:
        raise ValueError(f"expected {_FORMAT} in {directory / _CONFIG}")
    digests = metadata.get("files")
    if not isinstance(digests, dict):
        raise ValueError(f"missing file digests in {directory / _CONFIG}")
    for name in (_WEIGHTS, _PARITY):
        path = directory / name
        if not path.is_file() or digests.get(name) != file_digest(path):
            raise ValueError(f"missing file or SHA-256 mismatch: {path}")
    state = load_file(directory / _WEIGHTS)
    return _load_model(metadata["model"], state).to(device), metadata


def export_digest(directory: Path) -> str:
    """Return the SHA-256 of config.json, which binds both payload files."""
    return file_digest(Path(directory) / _CONFIG)


def _difference(actual: np.ndarray, expected: np.ndarray, tolerance: float) -> float | bool:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    if actual.dtype.kind != "f":
        return bool(np.array_equal(actual, expected))

    finite = np.isfinite(actual) & np.isfinite(expected)
    if not np.array_equal(actual[~finite], expected[~finite], equal_nan=True):
        return float("inf")
    actual_finite = actual[finite].astype(np.float64, copy=False)
    expected_finite = expected[finite].astype(np.float64, copy=False)
    with np.errstate(over="ignore"):
        delta = np.abs(actual_finite - expected_finite)
        allowed = tolerance + tolerance * np.abs(expected_finite)
        return float(np.max(delta / allowed, initial=0.0))


def check_parity(directory: Path, device: str = "cpu") -> dict[str, float | bool]:
    """Replay stored cases and report exact checks and maximum fractions of tolerance."""
    directory = Path(directory)
    model, metadata = load_export(directory, device)
    model.float()
    training = metadata["training"]
    screening = _screening(training)
    case_count = _CASE_COUNT
    if {"naive_rows", "naive_anchor"}.intersection(training["sampler"].get("derive", ())):
        case_count += 1
    tolerance = _CPU_TOLERANCE if torch.device(device).type == "cpu" else _DEVICE_TOLERANCE
    differences = {}
    with np.load(directory / _PARITY, allow_pickle=False) as arrays:
        prefixes = tuple(f"case_{i}" for i in range(1, case_count + 1))
        actual = {key.split("/", 1)[0] for key in arrays.files if key.startswith("case_")}
        missing = set(prefixes) - actual
        extra = actual.difference(prefixes)
        if missing or extra:
            raise ValueError(
                f"v5 parity case prefixes differ: missing {sorted(missing)}, extra {sorted(extra)}"
            )
        for prefix in prefixes:
            panel = _read_panel(arrays, prefix)
            requests = _read_requests(arrays[f"{prefix}/requests"])
            builder = EpisodeBuilder(panel, screening)
            batch = collate([builder.build(request) for request in requests])
            for name, value in _batch_arrays(batch):
                differences[f"{panel.name}.batch.{name}"] = _difference(
                    value, arrays[f"{prefix}/batch/{name}"], _BATCH_TOLERANCE
                )
            for name, value in _output_arrays(model, batch).items():
                differences[f"{panel.name}.output.{name}"] = _difference(
                    value, arrays[f"{prefix}/output/{name}"], tolerance
                )
    return differences


def _severity(value: float | bool) -> float:
    if isinstance(value, bool):
        return 0.0 if value else float("inf")
    return value


def _parity_error(differences: _Differences) -> str | None:
    name = max(differences, key=lambda entry: _severity(differences[entry]))
    worst = differences[name]
    if worst is False:
        return f"v5 parity failed at {name}: exact comparison failed"
    if _severity(worst) > 1:
        return f"v5 parity failed at {name}: {worst:.6g} times the allowed error"
    return None


def require_parity(directory: Path, device: str = "cpu") -> None:
    """Raise ValueError naming the worst difference if parity fails."""
    error = _parity_error(check_parity(directory, device))
    if error is not None:
        raise ValueError(error)


def _print_table(differences: _Differences) -> None:
    print(f"{'case':<18} {'batch ratio':>12} {'output ratio':>12}  exact")
    cases = dict.fromkeys(key.split(".", 1)[0] for key in differences)
    for case in cases:
        batch, output, mismatches = [], [], []
        for key, value in differences.items():
            name, group, field = key.split(".", 2)
            if name != case:
                continue
            if isinstance(value, float):
                (batch if group == "batch" else output).append(value)
            elif not value:
                mismatches.append(f"{group}.{field}")
        exact = ", ".join(mismatches) if mismatches else "ok"
        print(f"{case:<18} {max(batch, default=0):12.6g} {max(output, default=0):12.6g}  {exact}")


def _selftest(out: Path, derive: tuple[str, ...] = ()) -> None:
    config = V5TrainConfig(
        V5Config(
            d_model=16,
            nhead=2,
            enc_layers=1,
            dec_layers=1,
            dim_feedforward=16,
            expert_enc_layers=1,
            expert_dec_layers=1,
            d_calendar=4,
            d_step=4,
            d_out=4,
        ),
        (),
        SamplerConfig(derive=derive),
        OptimConfig(max_steps=1, warmup_steps=0, ema_decay=None),
        seed=2026,
        device="cpu",
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        model = V5Model(config.model)
    # Retain the selftest's original checkpoint representation and source digest.
    checkpoint_config = asdict(config)
    if not config.model.horizon_seed:
        del checkpoint_config["model"]["horizon_seed"]
    if not config.model.naive_mix:
        del checkpoint_config["model"]["naive_mix"]
    if not config.model.mix_evidence:
        del checkpoint_config["model"]["mix_evidence"]
    if config.sampler.derive:
        checkpoint_config["sampler"]["derive"] = list(config.sampler.derive)
    else:
        del checkpoint_config["sampler"]["derive"]
    if config.sampler.scale_floor is None:
        del checkpoint_config["sampler"]["scale_floor"]
    with TemporaryDirectory(prefix="v5-selftest-") as directory:
        checkpoint, _ = save_checkpoint(
            directory,
            {
                "model": model.state_dict(),
                "export": "model",
                "config": checkpoint_config,
                "identity": identity(config, code="selftest", corpus=None, init=None, env=None),
                "counters": {"batch": 0, "updates": 0, "saves": 1},
            },
            counter=1,
        )
        write_export(checkpoint, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    verbs = parser.add_subparsers(dest="verb", required=True)
    write = verbs.add_parser("write", help="export a training checkpoint")
    write.add_argument("--checkpoint", required=True, type=Path)
    write.add_argument("--out", required=True, type=Path)
    check = verbs.add_parser("check", help="replay the export's parity cases")
    check.add_argument("directory", type=Path)
    check.add_argument("--device", default="cpu")
    selftest = verbs.add_parser("selftest", help="export a tiny seeded untrained model")
    selftest.add_argument("directory", type=Path)
    selftest.add_argument("--derive", action="append", default=[], choices=DERIVABLE)
    args = parser.parse_args()
    if args.verb == "write":
        write_export(args.checkpoint, args.out)
    elif args.verb == "selftest":
        _selftest(args.directory, tuple(args.derive))
    else:
        differences = check_parity(args.directory, args.device)
        _print_table(differences)
        error = _parity_error(differences)
        if error is not None:
            print(error, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
