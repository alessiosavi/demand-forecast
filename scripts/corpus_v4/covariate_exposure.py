"""Measure real-window exposure with the patched upstream sampler; train nothing."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib
import inspect
import json
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.gcp.jobs.make_simons_job import canonical_condition

REPLAY = "real-only statistical"
STAGES = ("stage3", "stage4", "stage4-long")

# sha256(inspect.getsource(function).encode()), including indentation and line layout.
# Recorded after third_party/simons/prepare.py applied its full series to this pin.
SAMPLER_PIN = "8e338e915d26c8960713889ef12ab1cf5c08d591"
SUPPORTED_SAMPLER_SOURCE_SHA256 = {
    "trainer.covariate_passes": "5ef3d60498ddaafca21dd21330343255db84bc599e7a1a165bfdd8d4f685de71",
    "trainer.config_module.weights": "185f937829151aea752f729204a9511342c10f44d764e0abb46a2052c3350ff2",
    "trainer.network.future_steps": "3280e135ca10394f873c4a5aea36d1e52e69943e3e60bed6e8dfa2f04ff75f8a",
    "covariates.stream.long_spec": "ac5f35150749c5b4c607182811905b6fa27ccbb25bfdcce8321516d0d038d260",
    "covariates.stream.Batches.__iter__": "c499fbf58496adfb6f64aa37b21192d7884b7cc925243ab7ceb104dc5beb4f3b",
    "covariates.stream.CovariateStream.__init__": "d894299c9f2fb9304e5bc30d8da8c8abc1e65c071353257c190050cce74e96da",
    "covariates.stream.make_batch": "f95e533e4a544b05412507c722b50a82351805499f96213fad16bf3ff4f4c868",
    "covariates.stream.tokens_per_window": "69b5ab68d4496c0fefb5d9cf9dd0ea2a97c146abee8c12382fcb2d4e6fc6d50a",
    "covariates.stream.real_corpus": "dcb844f86fc5a8762612aa216b459d96a9794496af8cd450f1a0119be1886a58",
    "covariates.windows.CovariateCorpus.__init__": "bde63d3bc050fbfb7a9ff30282ef16671efaf16df7c6a608871b46961d108cfe",
    "covariates.windows.CovariateCorpus.train_sample": "f82dbd9d184946a3830400e4faa2a01ba4219e6316756cf73477c5cf81902a52",
    "covariates.windows.horizon_for": "12f45381b12693478e955a4a2aab3703baa3f799d55cf80b6a817a478fe40472",
    "covariates.windows.source_weights": "5fef15bcf4b8987562f2a7a5f561c4fc0f230c438e033f453b7a268df3b5b75d",
    "covariates.windows.load_items": "86497d738282bf96a609ca9fd67d3fa6b41aaa5ec80d391331e2269e55d930e3",
    "covariates.windows._list_arrays": "317478a48635eb167a1a6c90c98943086bc6f1dc9fee858da675e3e215910e03",
    "covariates.windows._matrix_arrays": "9081fd43590bbeb969c48e0174763ab3da9619806d15ae61d59e687b12e8bf2e",
    "covariates.windows.corpus_files": "c56f9a68e5d107625dee9001dcf5d9f955eaa2ad8a452e261ccc29f9a06a5e03",
    "covariates.windows.resolve": "014bac3918d2601270cf6ba83e2225c9d74fd1ba3d550db7f4f972e628a85142",
}


def condition(overrides):
    return canonical_condition("exposure", overrides) if overrides else None


@contextmanager
def upstream_modules(upstream):
    """Import only the requested, already patched checkout, including in test processes."""
    upstream = Path(upstream).resolve()
    sys.path.insert(0, str(upstream / "src"))
    try:
        modules = [
            importlib.import_module("simons.training." + name)
            for name in ("spot_job", "config", "trainer", "covariates.stream", "covariates.windows")
        ]
        for module in modules:
            if not Path(module.__file__).resolve().is_relative_to(upstream / "src"):
                raise RuntimeError(f"wrong upstream module: {module.__file__}; use a fresh process")
        yield modules
    finally:
        sys.path.remove(str(upstream / "src"))


def _no_base(*args, **kwargs):
    raise RuntimeError("real-only audit attempted to load base/univariate series")


def verify_sampler_source(modules):
    """Fail before instrumentation or counting if any supported sampler body changed."""
    _, _, trainer, stream, windows = modules
    roots = {"trainer": trainer, "covariates": SimpleNamespace(stream=stream, windows=windows)}
    verified = {}
    for name, expected in SUPPORTED_SAMPLER_SOURCE_SHA256.items():
        parts = name.split(".")
        function = roots[parts[0]]
        try:
            for part in parts[1:]:
                function = getattr(function, part)
            actual = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        except (AttributeError, OSError, TypeError) as error:
            raise RuntimeError(f"unsupported sampler source: {name}; cannot read source") from error
        if actual != expected:
            raise RuntimeError(
                f"unsupported sampler source: {name}; prepared pin {SAMPLER_PIN} "
                f"requires sha256 {expected}, found {actual}"
            )
        verified[name] = actual
    return verified


def sampling_spec(modules, upstream, corpus, stage, pass_name, size, seed, overrides):
    job, config, trainer, stream, _ = modules
    # Only paths are needed from the predecessor; no checkpoint or model is loaded.
    with tempfile.TemporaryDirectory(prefix="exposure-recipe-") as temporary:
        layout = Path(temporary)
        (layout / "covariates").symlink_to(corpus / "covariates", target_is_directory=True)
        (layout / "univariate").mkdir()
        with patch.object(job, "predecessor", lambda *a: layout / "unused-checkpoint.pt"):
            raw = job.recipe(
                upstream, layout, layout, size, stage, seed, 1, condition=condition(overrides)
            )
        model, settings = config.load(job.write_recipe(raw, layout))

        def record(spec, *args):
            unknown = set(spec) - set(stream.DEFAULTS)
            if unknown:
                raise RuntimeError(f"unsupported upstream sampling fields: {sorted(unknown)}")
            return copy.deepcopy({**stream.DEFAULTS, **spec})

        with patch.object(stream, "CovariateStream", record):
            passes = dict(
                trainer.covariate_passes(settings, model, SimpleNamespace(world=1), seed, 0, "cpu")
            )
        name = "covariate" if pass_name == "main" else pass_name
        if name not in passes:
            raise ValueError(f"{pass_name} pass is not configured for {stage}")
        spec = passes[name]
        # Stable provenance paths: the temporary layout is never an input dataset.
        spec["base"] = str(corpus / "univariate") if spec["base"] else None
        spec["real"] = str(corpus / "covariates")
        return spec


def _syntax(function):
    lines, first = inspect.getsourcelines(function)
    return ast.parse(textwrap.dedent("".join(lines))).body[0], first


class _CapturedCountError(Exception):
    pass


def batch_sizes(stream, spec, hw):
    """Read `count` from make_batch itself, stopping before its first sample."""
    result = []
    code = stream.make_batch.__code__
    for hist, probability in zip(spec["hists"], hw):
        captured = {}

        def trace(frame, event, arg):
            if frame.f_code is not code:
                return None
            if event == "line" and "count" in frame.f_locals:
                captured["windows"] = int(frame.f_locals["count"])
                raise _CapturedCountError
            return trace

        rng = SimpleNamespace(choice=lambda *a, **k: hist)
        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            stream.make_batch(spec, rng, None, None, ["real"], [1.0], hw)
        except _CapturedCountError:
            pass
        finally:
            sys.settrace(previous)
        if not captured:
            raise RuntimeError("cannot obtain make_batch's window count without sampling")
        result.append(dict(history=int(hist), probability=float(probability), **captured))
    return result


class _ProbeDoneError(Exception):
    pass


def usable_histories(real, windows, spec):
    """Exact row feasibility using upstream train_sample on representative endpoints.

    Its rejection guards depend on length and finite target counts only. Endpoints
    with the same history/future finite counts are equivalent for those guards.
    Probe the first legal endpoint, then each count change, until upstream accepts.
    This does not consume the audit's random stream or duplicate a rejection rule.
    """
    node, _ = _syntax(windows.CovariateCorpus.train_sample)
    guards = [
        n.test
        for n in ast.walk(node)
        if isinstance(n, ast.If) and any(isinstance(c, ast.Continue) for c in n.body)
    ]
    allowed = {"n", "horizon", "low", "min_observed", "hist", "np", "y"}
    if any({n.id for n in ast.walk(g) if isinstance(n, ast.Name)} - allowed for g in guards):
        raise RuntimeError("unsupported train_sample rejection guard; cannot count usable rows")
    y_uses = [
        ast.unparse(n)
        for g in guards
        for n in ast.walk(g)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "isfinite"
    ]
    if set(y_uses) != {"np.isfinite(y[:-horizon])", "np.isfinite(y[-horizon:])"}:
        raise RuntimeError("unsupported train_sample observation rules")
    usable = {source: Counter() for source in real.by_source}
    usable_rows = set()
    probe = copy.copy(real)
    probe.p = None
    for index, row in enumerate(real.recs):
        if row["source"] not in real.sources:
            continue
        probe.sources = [row["source"]]
        probe.by_source = {row["source"]: [index]}
        options = []
        windows.horizon_for(
            row["freq"], SimpleNamespace(choice=lambda values: options.extend(values) or values[0])
        )
        prefix = np.concatenate(([0], np.cumsum(np.isfinite(row["target"]))))
        for hist in spec["hists"]:
            for patches in sorted(set(options)):
                endpoints = None
                cursor = 0

                def endpoint(low, high=None):
                    nonlocal endpoints, cursor
                    if high is None:
                        return 0  # the sole source or row
                    if endpoints is None:
                        endpoints = [int(low)]
                        cursor = 1
                        return int(low)
                    if cursor == 1:
                        ends = np.arange(low, high)
                        boundary = np.maximum(ends - patches * windows.PATCH, 0)
                        future = prefix[ends] - prefix[boundary]
                        history = prefix[boundary] - prefix[np.maximum(boundary - hist, 0)]
                        changes = (
                            np.flatnonzero((np.diff(future) != 0) | (np.diff(history) != 0)) + 1
                        )
                        endpoints.extend(ends[changes].tolist())
                    if cursor >= len(endpoints):
                        raise _ProbeDoneError
                    result = endpoints[cursor]
                    cursor += 1
                    return result

                rng = SimpleNamespace(integers=endpoint, choice=lambda *a, **k: patches)
                try:
                    got = None
                    while got is None:
                        before = cursor
                        got = probe.train_sample(rng, int(hist), float(spec["min_observed"]))
                        if cursor == before:
                            break  # Upstream rejected length before choosing any endpoint.
                except _ProbeDoneError:
                    got = None
                if got is not None:
                    usable[row["source"]][str(hist)] += 1
                    usable_rows.add(index)
                    break
    return usable, usable_rows


def _counter():
    return dict(
        rows_loaded=0,
        rows_retained=0,
        rows_usable=0,
        rows_usable_by_history={},
        draws=0,
        accepted=0,
        rejected=0,
        rejected_by_reason=Counter(),
        horizon_histogram=Counter(),
        history_lengths=Counter(),
        channels_before_dropout=Counter(),
        channels_after_dropout=Counter(),
        windows_no_channels=0,
        windows_with_finite_future=0,
        finite_future_share_sum=0.0,
        distinct_rows_drawn=set(),
        distinct_row_endpoints=set(),
    )


class InstrumentedCorpus:
    """Wrap train_sample and trace its actual decisions, including internal retries."""

    def __init__(self, real, windows, stream, stats, spec):
        self.real, self.stats, self.spec = real, stats, spec
        self.sample_code = windows.CovariateCorpus.train_sample.__code__
        self.batch_code = stream.make_batch.__code__
        self.row_ids = {id(row): index for index, row in enumerate(real.recs)}
        sample, first = _syntax(windows.CovariateCorpus.train_sample)
        self.reasons = {}
        for node in ast.walk(sample):
            if isinstance(node, ast.If):
                for child in node.body:
                    if isinstance(child, ast.Continue):
                        self.reasons[first + child.lineno - 1] = ast.unparse(node.test)
        batch, first = _syntax(stream.make_batch)
        appends = [
            n
            for n in ast.walk(batch)
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "samples.append"
        ]
        if len(appends) != 1 or not self.reasons:
            raise RuntimeError("cannot instrument upstream sampler; unsupported source")
        self.append_line = first + appends[0].lineno - 1
        self.accepted = None

    def trace(self, frame, event, arg):
        if frame.f_code is self.sample_code:
            values = frame.f_locals
            if event == "line" and frame.f_lineno in self.reasons:
                self.draw(values, self.reasons[frame.f_lineno])
            elif event == "return" and arg is not None:
                self.draw(values, None)
                self.accepted = values["r"]["source"]
            return self.trace
        if frame.f_code is self.batch_code:
            if event == "line" and frame.f_lineno == self.append_line:
                if self.accepted is None:
                    raise RuntimeError("upstream batch has no recorded real window")
                row = self.stats[self.accepted]
                c, horizon = frame.f_locals["c"], frame.f_locals["H"]
                finite = np.isfinite(c[:, -horizon:])
                row["channels_after_dropout"][str(len(c))] += 1
                row["windows_no_channels"] += int(len(c) == 0)
                row["windows_with_finite_future"] += int(finite.any())
                row["finite_future_share_sum"] += float(finite.mean()) if finite.size else 0.0
                self.accepted = None
            return self.trace
        return None

    def draw(self, values, reason):
        record = values["r"]
        row = self.stats[record["source"]]
        index = self.row_ids[id(record)]
        row["draws"] += 1
        row["distinct_rows_drawn"].add(index)
        # Early length rejections have no endpoint (locals from prior retries can persist).
        if reason is None or "y" in {
            n.id for n in ast.walk(ast.parse(reason)) if isinstance(n, ast.Name)
        }:
            row["distinct_row_endpoints"].add((index, int(values["end"])))
        if reason is not None:
            row["rejected"] += 1
            row["rejected_by_reason"][reason] += 1
        else:
            row["accepted"] += 1
            row["horizon_histogram"][str(values["horizon"])] += 1
            row["history_lengths"][str(values["hist"])] += 1
            row["channels_before_dropout"][str(min(len(values["c"]), self.spec["max_cov"]))] += 1

    def train_sample(self, *args):
        return self.real.train_sample(*args)


def _finished(row, accepted):
    result = {
        k: (dict(sorted(v.items())) if isinstance(v, dict) else len(v) if isinstance(v, set) else v)
        for k, v in row.items()
        if k != "finite_future_share_sum"
    }
    result["accepted_share"] = row["accepted"] / accepted if accepted else 0.0
    result["mean_finite_future_share"] = (
        row["finite_future_share_sum"] / row["accepted"] if row["accepted"] else 0.0
    )
    return result


def audit(*, corpus, upstream, stage, pass_name, size="25m", batches, seed, overrides=()):
    corpus, upstream = Path(corpus).resolve(), Path(upstream).resolve()
    if not (corpus / "covariates").is_dir():
        raise FileNotFoundError(f"missing covariates/ in {corpus}")
    if stage not in STAGES or pass_name not in ("main", "long") or size != "25m":
        raise ValueError("audit requires a 25m stage3/stage4/stage4-long main or long pass")
    if type(batches) is not int or batches < 1 or type(seed) is not int or seed < 0:
        raise ValueError("batches must be positive and seed nonnegative")
    condition(overrides)
    with upstream_modules(upstream) as modules, ExitStack() as stack:
        _, _, _, stream, windows = modules
        sampler_source_sha256 = verify_sampler_source(modules)
        stack.enter_context(patch.object(windows, "BaseSeries", _no_base))
        stack.enter_context(patch.object(windows, "base_files", _no_base))
        configured = sampling_spec(
            modules, upstream, corpus, stage, pass_name, size, seed, overrides
        )
        spec = copy.deepcopy(configured)
        spec.update(base=None, mix={"real": 1.0})
        stats = {}
        read_table = windows.pq.read_table

        def read(path, *args, **kwargs):
            if not Path(path).resolve().is_relative_to((corpus / "covariates").resolve()):
                _no_base()
            table = read_table(path, *args, **kwargs)
            for source, count in Counter(table.column("source").to_pylist()).items():
                stats.setdefault(source, _counter())["rows_loaded"] += count
            return table

        with patch.object(windows.pq, "read_table", read):
            real = stream.real_corpus(spec)
        if real is None:
            raise RuntimeError("upstream real_corpus returned no corpus")
        for source, indices in real.by_source.items():
            stats[source]["rows_retained"] = len(indices)
        usable, usable_rows = usable_histories(real, windows, spec)
        for source, counts in usable.items():
            stats[source]["rows_usable_by_history"] = dict(counts)
        for index in usable_rows:
            stats[real.recs[index]["source"]]["rows_usable"] += 1
        hw = np.array(spec["hist_weights"], float)
        hw /= hw.sum()  # Exactly Batches.__iter__'s history probabilities.
        sizes = batch_sizes(stream, spec, hw)
        observed_batches = []
        wrapper = InstrumentedCorpus(real, windows, stream, stats, spec)
        rng = np.random.default_rng([seed, int(spec["stream_tag"]), 0, 0])
        previous = sys.gettrace()
        try:
            sys.settrace(wrapper.trace)
            for _ in range(batches):
                batch = stream.make_batch(spec, rng, None, wrapper, ["real"], [1.0], hw)
                observed_batches.append(
                    dict(
                        hist=int(batch["hist"]),
                        **{
                            k: int(batch["kinds"].get(k, 0))
                            for k in ("real", "sibling", "synthetic")
                        },
                    )
                )
        finally:
            sys.settrace(previous)
        totals = _counter()
        for row in stats.values():
            for key, value in row.items():
                if isinstance(value, set):
                    totals[key].update(value)
                elif isinstance(value, dict):
                    totals[key] = dict(Counter(totals[key]) + Counter(value))
                else:
                    totals[key] += value
        mean = sum(s["windows"] * s["probability"] for s in sizes)
        mix = configured["mix"]
        real_share = mix.get("real", 0) / sum(mix.values())
        revision = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip()
        return dict(
            replay=REPLAY,
            inputs=dict(
                corpus=str(corpus),
                upstream=str(upstream),
                stage=stage,
                size=size,
                pass_name=pass_name,
                batches=batches,
                seed=seed,
                overrides=list(overrides),
            ),
            upstream_revision=revision,
            sampler_source_sha256=sampler_source_sha256,
            spec_entry_point="spot_job.recipe -> config.load -> trainer.covariate_passes (recording CovariateStream)",
            configured_spec=configured,
            configured_mix=mix,
            effective_spec=spec,
            loss_weight=float(configured["weight"]),
            rank_world=1,
            random_stream="one generator seeded [seed, stream_tag, 0, 0]; no loader workers",
            windows_per_batch_by_history=sizes,
            mean_windows_per_batch=mean,
            expected_real_windows_per_update=mean * real_share,
            counter_notes=dict(
                draws="internal train_sample attempts, including retries",
                rows_usable="can accept at least one endpoint at a configured history and upstream horizon",
                finite_future_share="mean per accepted window over retained channels and its horizon; empty channels contribute zero",
                channels_before_dropout="after upstream channel caps, before dropout",
                distinct_row_endpoints="accepted and rejected attempts that chose an endpoint; row identity is loaded record index",
            ),
            totals=_finished(totals, totals["accepted"]),
            per_source={s: _finished(stats[s], totals["accepted"]) for s in sorted(stats)},
            batches=observed_batches,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--size", choices=["25m"], default="25m")
    parser.add_argument("--pass", dest="pass_name", choices=["main", "long"], required=True)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--override", dest="overrides", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    output = args.pop("output").resolve()
    if output.is_relative_to(args["corpus"].resolve()):
        raise ValueError("output must be outside the read-only corpus")
    report = audit(**args)
    output.mkdir(parents=True, exist_ok=True)
    name = args["pass_name"]
    (output / f"{name}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    metadata = {
        k: report[k]
        for k in (
            "replay",
            "configured_mix",
            "loss_weight",
            "windows_per_batch_by_history",
            "mean_windows_per_batch",
            "expected_real_windows_per_update",
        )
    }
    fields = [*metadata, "source", *report["totals"]]
    with (output / f"{name}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for source, values in report["per_source"].items():
            writer.writerow(
                dict(
                    source=source,
                    **{
                        k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                        for k, v in {**metadata, **values}.items()
                    },
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
