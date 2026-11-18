"""One resumable v5 unit under the cloud guest's state contract."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import signal
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from . import loop, measure, t1
from .checkpoint import _atomic_write, file_digest, read_manifest, write_manifest
from .config import V5TrainConfig, identity
from .sources import SourceMix

T1_EVAL_DEFAULTS = (200, 1000)


def _name(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value):
        raise argparse.ArgumentTypeError("expected a safe unit name")
    return value


def _hex(value: str, length: int) -> str:
    if not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
        raise argparse.ArgumentTypeError(f"expected {length} lowercase hexadecimal characters")
    return value


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("train", "qualify", "t1"):
        sub = actions.add_parser(action)
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.add_argument("--code", type=lambda v: _hex(v, 40), required=True)
        for key in ("job-sha256", "env-sha256"):
            sub.add_argument("--" + key, type=lambda v: _hex(v, 64), required=True)
        if action == "t1":
            sub.add_argument("--base", type=Path, required=True)
            sub.add_argument("--path", choices=t1.PATHS, required=True)
            sub.add_argument("--arm", type=_name, required=True)
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--episodes", type=_positive, default=T1_EVAL_DEFAULTS[0])
            sub.add_argument("--max-panels", type=_positive, default=T1_EVAL_DEFAULTS[1])
        else:
            sub.add_argument("--config", type=Path, required=True)
            sub.add_argument("--name", type=_name, required=True)
            sub.add_argument("--corpus-root", type=Path)
        if action in ("train", "t1"):
            sub.add_argument("--stop-after", type=_positive)
        if action == "train":
            sub.add_argument("--init-file", type=Path)
            sub.add_argument("--init-sha256", type=lambda v: _hex(v, 64))
    sub = actions.add_parser("t1-compare")
    sub.add_argument("--base", type=Path, required=True)
    sub.add_argument("--code", type=lambda v: _hex(v, 40), required=True)
    sub.add_argument("--path", choices=t1.PATHS, required=True)
    sub.add_argument("--seed", type=int, required=True)
    sub.add_argument("--episodes", type=_positive, required=True)
    sub.add_argument("--max-panels", type=_positive, required=True)
    sub.add_argument("--arm", action="append", default=[])
    sub.add_argument("--out", type=Path, required=True)
    return parser


def _json(path: Path, record: dict) -> None:
    encoded = json.dumps(record, sort_keys=True, allow_nan=False).encode()
    _atomic_write(path, lambda stream: stream.write(encoded))


def _copy(source: Path, destination: Path, staging: Path) -> None:
    def write(stream):
        with source.open("rb") as incoming:
            shutil.copyfileobj(incoming, stream)

    _atomic_write(destination, write, staging=staging)


def _complete(unit: Path, expected: dict) -> bool:
    saved = read_manifest(unit / "unit.json")
    if saved is not None and all(saved.get(k) == v for k, v in expected.items()):
        return True
    (unit / "unit.json").unlink(missing_ok=True)
    return False


def _comparison_arm(directory: Path, expected: dict, config: V5TrainConfig) -> tuple[dict, dict]:
    """Read a copied unit without the cloud completion helper's invalidation writes."""
    unit = read_manifest(directory / "unit.json")
    if unit is None or "result.json" not in unit["files"]:
        raise ValueError("unit.json does not validate its artifacts, including result.json")
    for key, value in expected.items():
        if unit.get(key) != value:
            raise ValueError(f"{key} mismatch in unit.json")
    for key in ("env_sha256", "eval_digest"):
        if not isinstance(unit.get(key), str):
            raise ValueError(f"missing or invalid {key} in unit.json")
        try:
            _hex(unit[key], 64)
        except argparse.ArgumentTypeError as error:
            raise ValueError(f"invalid {key} in unit.json: {error}") from error
    requested = identity(
        config, code=expected["code"], corpus=None, init=None, env=unit["env_sha256"]
    )
    if unit.get("identity") != requested:
        raise ValueError("training identity mismatch in unit.json")
    result = json.loads((directory / "result.json").read_text())
    if not isinstance(result, dict):
        raise ValueError("result.json must hold an evaluation object")
    if result.get("eval_digest") != unit["eval_digest"]:
        raise ValueError("eval_digest mismatch between result.json and unit.json")
    losses = result.get("loss")
    if not isinstance(losses, list) or any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in losses
    ):
        raise ValueError("result.json losses must be a list of finite nonnegative numbers")
    return unit, result


def _t1_compare(args) -> int:
    entry = t1.PATHS[args.path]
    directories = {}
    for argument in args.arm:
        arm, separator, directory = argument.partition("=")
        if not separator or not directory:
            raise ValueError(f"{arm}: expected --arm NAME=DIR")
        if arm in directories:
            raise ValueError(f"{arm}: repeated arm")
        if arm not in entry["arms"]:
            raise ValueError(f"{arm}: extra arm for {args.path}")
        directory = Path(directory)
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"{arm}: requires an existing real directory without a symlink")
        directories[arm] = directory
    for arm in entry["arms"]:
        if arm not in directories:
            raise ValueError(f"{arm}: missing arm for {args.path}")

    base = V5TrainConfig.from_yaml(args.base)
    expected = dict(
        path=args.path,
        seed=args.seed,
        code=args.code,
        base_sha256=file_digest(args.base),
        episodes=args.episodes,
        max_panels=args.max_panels,
    )
    units, results, hashes = {}, {}, {}
    for arm in entry["arms"]:
        try:
            config = t1.t1_config(args.path, arm, seed=args.seed, base=base)
            unit, result = _comparison_arm(directories[arm], {**expected, "arm": arm}, config)
            if units:
                reference = next(iter(units.values()))
                for key in ("env_sha256", "eval_digest"):
                    if unit[key] != reference[key]:
                        raise ValueError(f"{key} differs across arms")
            hashes[arm] = file_digest(directories[arm] / "unit.json")
            units[arm], results[arm] = unit, result
        except Exception as error:
            raise ValueError(f"{arm}: {error}") from error

    episodes = t1.eval_set(args.path, base=base, episodes=args.episodes, max_panels=args.max_panels)
    digest = t1.eval_digest(episodes)
    for arm, unit in units.items():
        if unit["eval_digest"] != digest:
            raise ValueError(f"{arm}: eval_digest differs from rebuilt evaluation set")
        if len(results[arm]["loss"]) != len(episodes):
            raise ValueError(f"{arm}: result.json losses do not align with evaluation episodes")

    comparisons = []
    for true, null, decides in entry["comparisons"]:
        comparison = t1.compare(
            results[true]["loss"],
            results[null]["loss"],
            episodes=episodes,
            required=entry["required"],
        )
        comparisons.append({"true": true, "null": null, "decides": decides, **comparison})
    outcomes = [c["outcome"] for c in comparisons if c["decides"]]
    report = {
        **expected,
        "env_sha256": next(iter(units.values()))["env_sha256"],
        "eval_digest": digest,
        "units": hashes,
        "comparisons": comparisons,
    }
    if args.path == "selection":
        try:
            report["selection_report"] = t1.selection_report(
                results["true"], results["null"], results["gates_one"], episodes
            )
            outcomes.append(report["selection_report"]["outcome"])
        except Exception as error:
            raise ValueError(f"selection arms true/null/gates_one: {error}") from error
    report["outcome"] = t1._outcome(outcomes)
    _json(args.out / "compare.json", report)
    return 0


def _run(args, unit: Path, training: Path) -> int:
    base = V5TrainConfig.from_yaml(args.base if args.action == "t1" else args.config)
    base_digest = file_digest(args.base) if args.action == "t1" else None
    config = base
    if args.action == "t1":
        config = t1.t1_config(args.path, args.arm, seed=args.seed, base=base)
    init = getattr(args, "init_file", None)
    init_digest = getattr(args, "init_sha256", None)
    if (init is None) != (init_digest is None):
        raise ValueError("init-file and init-sha256 must be supplied together")
    if init is not None and file_digest(init) != init_digest:
        raise ValueError("init SHA-256 mismatch")
    corpus_root = getattr(args, "corpus_root", None)
    corpus_digest = SourceMix(config.sources, corpus_root=corpus_root).corpus_digest
    requested = identity(
        config, code=args.code, corpus=corpus_digest, init=init_digest, env=args.env_sha256
    )
    expected = dict(
        code=args.code, job_sha256=args.job_sha256, env_sha256=args.env_sha256, identity=requested
    )
    if args.action == "t1":
        expected.update(
            path=args.path,
            arm=args.arm,
            seed=args.seed,
            base_sha256=base_digest,
            episodes=args.episodes,
            max_panels=args.max_panels,
        )

    # Identity is checked even for complete units; stale completion must not
    # survive a failed attempt under another environment or configuration.
    cached = _complete(unit, expected)
    if args.action == "qualify":
        sidecar = unit / "identity.json"
        if sidecar.exists():
            if json.loads(sidecar.read_text())["identity"] != requested:
                raise loop.IdentityError("qualification identity mismatch")
        else:
            _json(sidecar, {"identity": requested})
        if cached:
            return 0
        # qualify has no corpus_root argument; roots are machine-local and do
        # not participate in canonical configuration identity.
        config = replace(
            config,
            sources=tuple(
                replace(s, root=str(corpus_root))
                if s.kind == "corpus" and s.root is None and corpus_root is not None
                else s
                for s in config.sources
            ),
        )
        staging = args.run_dir / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        # P5 creates temporary training states beneath out. Keep those outside
        # the guest's copied tree and publish only its completed report.
        with TemporaryDirectory(prefix="qualify-", dir=staging) as temporary:
            output = Path(temporary)
            try:
                measure.qualify(config, output)
            except RuntimeError as error:
                # P5 exposes an inner trainer's graceful TERM as this exception,
                # rather than returning a TrainResult to its caller.
                if str(error) == "qualification training stopped with terminated":
                    return 75
                raise
            json.loads((output / "qualify.json").read_text())
            _copy(output / "qualify.json", unit / "qualify.json", staging)
        files = ["qualify.json"]
    else:
        result = loop.train(
            config,
            training,
            code=args.code,
            env=args.env_sha256,
            run_dir=args.run_dir,
            staging=args.run_dir / "staging",
            corpus_root=corpus_root,
            stop_after=args.stop_after,
            init=init,
        )
        if result.stop_reason in ("interrupted", "terminated"):
            return 75
        if result.stop_reason != "max_steps":
            raise RuntimeError(f"training stopped: {result.stop_reason}")
        manifest = read_manifest(training / "train.json")
        if (
            manifest is None
            or manifest.get("identity") != requested
            or result.batch != config.optim.max_steps
        ):
            raise RuntimeError("training completion does not validate")
        if cached:
            return 0
        files = ["checkpoint-final.pt", "train.json"]
        if args.action == "t1":
            episodes = t1.eval_set(
                args.path, base=base, episodes=args.episodes, max_panels=args.max_panels
            )
            evaluation = t1.evaluate(
                training / "checkpoint-final.pt",
                args.path,
                args.arm,
                episodes,
                forced=args.path == "selection" and args.arm == "true",
            )
            expected["eval_digest"] = t1.eval_digest(episodes)
            if evaluation.get("eval_digest") != expected["eval_digest"]:
                raise RuntimeError("evaluation digest mismatch")
            _json(unit / "result.json", evaluation)
            for name in files:
                _copy(training / name, unit / name, args.run_dir / "staging")
            if read_manifest(unit / "train.json") is None:
                raise RuntimeError("copied training manifest does not validate")
            files.append("result.json")
    write_manifest(unit / "unit.json", {**expected, "files": files})
    if read_manifest(unit / "unit.json") is None:
        raise RuntimeError("unit publication does not validate")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "t1-compare":
        try:
            return _t1_compare(args)
        except Exception as error:
            print(f"t1-compare: {error}", file=sys.stderr)
            return 1
    if not args.run_dir.is_absolute():
        raise SystemExit("run-dir must be absolute")
    if args.action == "t1":
        root = args.run_dir / "state/t1" / f"{args.path}-{args.arm}-s{args.seed}"
        training = root / "train"
        unit = root / f"eval-{args.episodes}-{args.max_panels}"
    else:
        unit = args.run_dir / "state" / args.action / args.name
        training = unit
    unit.mkdir(parents=True, exist_ok=True)
    (unit / "failure.json").unlink(missing_ok=True)

    def stop(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        return _run(args, unit, training)
    except KeyboardInterrupt:
        (unit / "unit.json").unlink(missing_ok=True)
        return 75
    except Exception as error:
        (unit / "unit.json").unlink(missing_ok=True)
        trace = traceback.format_exc()
        print(trace, file=sys.stderr, flush=True)  # The guest copies the job's log to the bucket.
        _json(
            unit / "failure.json",
            {"reason": f"{type(error).__name__}: {error}", "traceback": trace},
        )
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
