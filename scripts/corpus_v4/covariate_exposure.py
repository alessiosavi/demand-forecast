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
from types import CodeType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.gcp.jobs.make_simons_job import canonical_condition

REPLAY = "real-only statistical"
STAGES = ("stage3", "stage4", "stage4-long")

# sha256(file bytes), covering the loaded-module union for every supported stage/pass.
# Recorded after third_party/simons/prepare.py applied its full series to this pin.
SAMPLER_PIN = "8e338e915d26c8960713889ef12ab1cf5c08d591"
SUPPORTED_SOURCE_SHA256 = {
    "simons/__init__.py": "4b46dc41e3cad66cfa349be82d0b90df1bfa25cbc4aa36ccd7e1c172db93ca9d",
    "simons/config.py": "e37c0f0888d7ef5017f3b48198d6ae9ccfe478b1dbad2b04de90874a9f6f202b",
    "simons/covariates.py": "4f172b5a26cb9eb46129ddf81777e127945bba1c484d19596267e9812cf412ef",
    "simons/forecast.py": "ae84c4408915f6b6126632efcfdbbf5b7b4f816ca2aaa848479d081d6dc90aca",
    "simons/model.py": "6b87b46f6f56267b9893fe224c959b24591a8a2588d4eec4194494e5435e95df",
    "simons/periodicity.py": "8b44693f6561c0e99acb6d5bec0b977ac5fe1e2028c8ab8fa995d842ce93da9b",
    "simons/synth/__init__.py": "8d18ab3612d4ee30be91f4c29df8d1d0d369d55b85636b6e81b2bbd37a39f802",
    "simons/synth/augment.py": "7aff41774cb8db54146a41d4b2ead85578d1216a2e3457b699811cf372c0d59e",
    "simons/synth/base.py": "22ae20d59db6f45cbee1ca0eaa34308c897b149c4cc078eec79df98c21367c97",
    "simons/synth/covariates/__init__.py": "d3d9b719381643417cab5e705ba821667cf26c228f2edfd5a2132eb2ef797eae",
    "simons/synth/covariates/effects.py": "860964551372a4d3f0ca371ce00b0271f87e875756dafa87a04b8765264500b3",
    "simons/synth/mech.py": "228928cc9d0d23c70201ff238d9e48fa19b53c3d2f210fa99d0f9e5370406550",
    "simons/synth/multi.py": "91bf4f23deafb161214bf8899fc6fe936cc27b7b3474a72d79e54a9070fdbcc6",
    "simons/synth/observe.py": "1b435accb6baa41bab3a53993b4f1fe3d90278868f5f21dba3cc723449ce0556",
    "simons/training/__init__.py": "cca66ebd098c78b859974b42899dfb686007b17e0dadcfbc4c1e6ca814d88c49",
    "simons/training/augment.py": "0ab923b9a79478a25e10d1eba9506f026526c26da36c4cdc4f903a7528b1f6a9",
    "simons/training/config.py": "d768e48c5dd2bd6ae36d1fa3f33531bff173a71fce3202fdd8b96b5bf8123a14",
    "simons/training/covariates/__init__.py": "f45b58674bdbb9b54845c4d5b558f3b36ea387211a7d2d6dbb57fdff3fe53481",
    "simons/training/covariates/roles.py": "5350125021ead60c3953427d9db3a80d30cb9577d92e4b724efc2f8a7ae2ea7a",
    "simons/training/covariates/stream.py": "075444db876a93089c4c0d658cd212ff74135a7a4e2f171a039cc83b43bc138b",
    "simons/training/covariates/windows.py": "8d1113a846a6b3e23ca6abe474a45df99b846fabaacc3a577f8aa741bf078cb1",
    "simons/training/data.py": "3e61f0bce8e928699d0759ecc05c1880df5239f8c747e43636e7250a55fdcbbe",
    "simons/training/distributed.py": "394682806caeff50774e090c387b27b8002b3004d9d9363841c7e4b9356aff3e",
    "simons/training/exposure.py": "01615fed6acfc71f23f9e1dd27f32dd0a943b73b95b930c6eb8b12da4e0c7077",
    "simons/training/fork_conditions.py": "bbbaaeec6c884f538bf3e508315f864c8b2f138ed959fbbe360caca495674b36",
    "simons/training/network.py": "f76f41991509b5f8b5a42eda5681551ae3a570157ba9555d2fc4ceec842457bc",
    "simons/training/objective.py": "fe934e67203e59c197511fe0b2dbd500bd4d582d2150ea4f80bb613a13e75173",
    "simons/training/optim.py": "7fdc5a369755d089a8d41f1f013dca2f1120175ec3b559aece0f9e70807b74c0",
    "simons/training/packing.py": "b8e80e1d8ffd623cf6b9dd3920c4117a987f23bd5137cecd7d5d45a5b979ca06",
    "simons/training/reliability.py": "d3bac409c444de7fc770a5787001c7f215df76c7344672a2ca93d4cbb0eb5ba8",
    "simons/training/schedule.py": "31b6db177d4f161f43c849de37acda23d133fc32b8e5982313bb617cfb63c5e2",
    "simons/training/screen.py": "b4df5718f5b05fef1b67bc63b1541253c6e9a571752fe89b66ab62d9ea11768b",
    "simons/training/spot_job.py": "fdec6013bff644cde2251d91ed5537354f5f17c77e3fbc4c72a807637cbcc80b",
    "simons/training/trainer.py": "8fd976343799fa5ffc38a74b69def9ab22002903d0f7ee0200965cc9a785ce0d",
    "simons/training/warp.py": "bd80855d44c62273061ab1337708b6cffa51a7911c491db804207b2de40adffc",
}
SAMPLER_FUNCTIONS = (
    "trainer.covariate_passes",
    "trainer.config_module.weights",
    "trainer.network.future_steps",
    "covariates.stream.long_spec",
    "covariates.stream.Batches.__init__",
    "covariates.stream.Batches.__iter__",
    "covariates.stream.CovariateStream.__init__",
    "covariates.stream.make_batch",
    "covariates.stream.tokens_per_window",
    "covariates.stream.real_corpus",
    "covariates.windows.CovariateCorpus.__init__",
    "covariates.windows.CovariateCorpus.train_sample",
    "covariates.windows.horizon_for",
    "covariates.windows.source_weights",
    "covariates.windows.load_items",
    "covariates.windows._list_arrays",
    "covariates.windows._matrix_arrays",
    "covariates.windows.corpus_files",
    "covariates.windows.resolve",
)


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


def loaded_sources(upstream):
    """Snapshot every loaded file under this checkout, including lazy dependencies."""
    root = Path(upstream).resolve() / "src"
    sources = {}
    for module in tuple(sys.modules.values()):
        if not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        if path.is_relative_to(root):
            name = path.relative_to(root).as_posix()
            try:
                sources[name] = path.read_bytes()
            except OSError as error:
                raise RuntimeError(
                    f"unsupported upstream source: {name}; cannot read file"
                ) from error
    return sources


def verify_sampler_source(modules, upstream):
    """Verify dependency files, then bind the traced functions to their compiled source."""
    root = Path(upstream).resolve() / "src"
    for name in SUPPORTED_SOURCE_SHA256:
        module_name = name.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        module = sys.modules.get(module_name)
        if module is not None and (
            not getattr(module, "__file__", None) or Path(module.__file__).resolve() != root / name
        ):
            raise RuntimeError(f"unsupported upstream source: {name}; loaded from elsewhere")
    sources = loaded_sources(upstream)
    verified = {}
    for name, source in sorted(sources.items()):
        actual = hashlib.sha256(source).hexdigest()
        expected = SUPPORTED_SOURCE_SHA256.get(name)
        if actual != expected:
            raise RuntimeError(
                f"unsupported upstream source: {name}; prepared pin {SAMPLER_PIN} "
                f"requires sha256 {expected}, found {actual}"
            )
        verified[name] = actual

    _, _, trainer, stream, windows = modules
    roots = {"trainer": trainer, "covariates": SimpleNamespace(stream=stream, windows=windows)}
    compiled = {}
    for name in SAMPLER_FUNCTIONS:
        parts = name.split(".")
        function = roots[parts[0]]
        try:
            for part in parts[1:]:
                owner, function = function, getattr(function, part)
            if hasattr(function, "__wrapped__"):
                raise ValueError("wrapped function")
            if inspect.ismethod(function):
                function = function.__func__
            if not inspect.isfunction(function) or hasattr(function, "__wrapped__"):
                raise ValueError("not a plain function or method")
            qualified_name = (
                owner.__qualname__ + "." + parts[-1] if inspect.isclass(owner) else parts[-1]
            )
            if function.__qualname__ != qualified_name:
                raise ValueError("function has a different qualified name")
            module = sys.modules[function.__module__]
            if function.__globals__ is not vars(module):
                raise ValueError("function is not defined in its module")
            path = Path(module.__file__).resolve()
            relative = path.relative_to(root).as_posix()
            if Path(function.__code__.co_filename).resolve() != path:
                raise ValueError("function code is not from its module file")
            if relative not in compiled:
                pending = [compile(sources[relative], str(path), "exec", dont_inherit=True)]
                codes = {}
                while pending:
                    code = pending.pop()
                    # co_name, not co_qualname: the cloud guests run Python 3.10.
                    codes[code.co_name, code.co_firstlineno] = code
                    pending.extend(c for c in code.co_consts if isinstance(c, CodeType))
                compiled[relative] = codes
            expected = compiled[relative].get(
                (function.__code__.co_name, function.__code__.co_firstlineno)
            )
            pending = [(function.__code__, expected)]
            while pending:
                actual_code, expected_code = pending.pop()
                # Python 3.10 code equality omits the line tables used by the tracer.
                if actual_code != expected_code or tuple(actual_code.co_lines()) != tuple(
                    expected_code.co_lines()
                ):
                    raise ValueError("executed code differs from verified source")
                pending.extend(
                    (actual, expected)
                    for actual, expected in zip(actual_code.co_consts, expected_code.co_consts)
                    if isinstance(actual, CodeType)
                )
        except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
            raise RuntimeError(f"unsupported sampler code: {name}; {error}") from error
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
        stack.enter_context(patch.object(windows, "BaseSeries", _no_base))
        stack.enter_context(patch.object(windows, "base_files", _no_base))
        configured = sampling_spec(
            modules, upstream, corpus, stage, pass_name, size, seed, overrides
        )
        spec = copy.deepcopy(configured)
        spec.update(base=None, mix={"real": 1.0})
        loaded_source_names = []
        read_table = windows.pq.read_table

        def read(path, *args, **kwargs):
            if not Path(path).resolve().is_relative_to((corpus / "covariates").resolve()):
                _no_base()
            table = read_table(path, *args, **kwargs)
            loaded_source_names.extend(table.column("source").to_pylist())
            return table

        with patch.object(windows.pq, "read_table", read):
            real = stream.real_corpus(spec)
        if real is None:
            raise RuntimeError("upstream real_corpus returned no corpus")
        sampler_source_sha256 = verify_sampler_source(modules, upstream)
        stats = {}
        for source, count in Counter(loaded_source_names).items():
            stats[source] = _counter()
            stats[source]["rows_loaded"] = count
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
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--size", choices=["25m"], default="25m")
    parser.add_argument("--pass", dest="pass_name", choices=["main", "long"])
    parser.add_argument("--batches", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--override", dest="overrides", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-source-hashes", action="store_true")
    args = vars(parser.parse_args(argv))
    args["upstream"] = args["upstream"].resolve()
    if args.pop("print_source_hashes"):
        with upstream_modules(args["upstream"]) as modules:
            with tempfile.TemporaryDirectory(prefix="exposure-source-hashes-") as temporary:
                corpus = Path(temporary)
                (corpus / "covariates").mkdir()
                for stage in STAGES:
                    for pass_name in ("main", "long") if stage != "stage3" else ("main",):
                        sampling_spec(
                            modules, args["upstream"], corpus, stage, pass_name, "25m", 11, ()
                        )
            hashes = {
                name: hashlib.sha256(source).hexdigest()
                for name, source in loaded_sources(args["upstream"]).items()
            }
        print(json.dumps(hashes, indent=2, sort_keys=True))
        return 0
    missing = [
        "--" + name.replace("_", "-").replace("pass-name", "pass")
        for name in ("corpus", "stage", "pass_name", "batches", "seed", "output")
        if args[name] is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
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
