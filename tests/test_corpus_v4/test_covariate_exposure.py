"""Real upstream sampling, tiny local corpora; no downloads or training."""

import ast
import csv
import functools
import hashlib
import inspect
import json
import linecache
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Collected in the environment without the bench extra too: skip there instead of failing collection.
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from tests.test_gcp_jobs.conftest import upstream as upstream_fixture  # noqa: E402

upstream = upstream_fixture


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    folder = root / "covariates/tiny"
    folder.mkdir(parents=True)
    rows = []
    for source, count in (("a", 1), ("b", 4), ("c", 11), ("short", 1), ("sparse", 1)):
        for index in range(count):
            length = 64 if source == "short" else 16480
            values = np.ones(length, dtype=np.float32)
            target = np.full(length, np.nan) if source == "sparse" else values
            cov = np.full(length, np.nan) if source == "c" else values
            rows.append(
                dict(
                    item_id=f"{source}-{index}",
                    source=source,
                    freq="D",
                    target=target.tolist(),
                    covariates=[cov.tolist(), cov.tolist()],
                )
            )
    pq.write_table(pa.Table.from_pylist(rows), folder / "train-000.parquet")
    return root


def run(corpus, upstream, **changes):
    from scripts.corpus_v4.covariate_exposure import audit

    return audit(
        corpus=corpus,
        upstream=upstream,
        stage="stage3",
        pass_name="main",
        size="25m",
        batches=8,
        seed=11,
        **changes,
    )


def test_real_only_counts_rejections_uniform_sources_and_determinism(corpus, upstream):
    report = run(corpus, upstream)
    assert report == run(corpus, upstream)
    assert report["replay"] == "real-only statistical"
    assert report["configured_mix"] == {"synthetic": 0.63, "sibling": 0.25, "real": 0.12}
    assert report["effective_spec"]["base"] is None
    assert report["effective_spec"]["mix"] == {"real": 1.0}
    assert report["loss_weight"] == 0.025
    expected_hashes = {}
    for module in tuple(sys.modules.values()):
        if getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if path.is_relative_to(upstream / "src"):
                expected_hashes[path.relative_to(upstream / "src").as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    assert report["sampler_source_sha256"] == expected_hashes
    sources = report["per_source"]
    assert {k: v["rows_loaded"] for k, v in sources.items()} == dict(
        a=1, b=4, c=11, short=1, sparse=1
    )
    assert sources["short"]["rows_usable"] == sources["sparse"]["rows_usable"] == 0
    for name in ("a", "b", "c"):
        assert sources[name]["rows_usable"] == sources[name]["rows_loaded"]
        assert sources[name]["accepted_share"] == pytest.approx(1 / 3, abs=0.08)
    totals = report["totals"]
    assert totals["draws"] == totals["accepted"] + totals["rejected"]
    assert totals["rejected"] > 0
    assert sum(totals["rejected_by_reason"].values()) == totals["rejected"]
    assert totals["distinct_rows_drawn"] == 18
    assert totals["distinct_row_endpoints"] >= totals["accepted"] * 0.95
    assert sources["c"]["windows_with_finite_future"] == 0
    assert sources["c"]["mean_finite_future_share"] == 0
    assert totals["windows_no_channels"] > 0
    assert totals["windows_with_finite_future"] > 0
    assert sum(totals["history_lengths"].values()) == totals["accepted"]
    assert sum(totals["horizon_histogram"].values()) == totals["accepted"]
    assert all(b["sibling"] == b["synthetic"] == 0 and b["real"] > 0 for b in report["batches"])
    sizes = {s["history"]: s["windows"] for s in report["windows_per_batch_by_history"]}
    assert all(b["real"] == sizes[b["hist"]] for b in report["batches"])
    mean = sum(x["windows"] * x["probability"] for x in report["windows_per_batch_by_history"])
    assert report["expected_real_windows_per_update"] == pytest.approx(mean * 0.12)
    assert not (corpus / "univariate").exists()


@pytest.mark.parametrize(
    "stage,pass_name,weight",
    [("stage4", "main", 0.05), ("stage4", "long", 0.2), ("stage4-long", "long", 0.2)],
)
def test_recipe_specs_and_long_observation_rules(corpus, upstream, stage, pass_name, weight):
    from scripts.corpus_v4.covariate_exposure import audit

    report = audit(
        corpus=corpus,
        upstream=upstream,
        stage=stage,
        pass_name=pass_name,
        size="25m",
        batches=3,
        seed=22,
        overrides=["covariate_weight=0.05", "covariate_mix=0,0.25,0.75", "long_weight=0.2"],
    )
    assert report["loss_weight"] == weight
    if pass_name == "long":
        assert report["configured_mix"] == {"real": 1.0}
        assert report["totals"]["rows_retained"] == 17
        assert report["totals"]["rows_usable"] == 16
        assert report["effective_spec"]["min_observed"] == 0.75
        assert {x["history"] for x in report["windows_per_batch_by_history"]} == {2048, 4096}
        assert report["expected_real_windows_per_update"] == report["mean_windows_per_batch"]


@pytest.mark.parametrize("drop_all,drop_each", [(1.0, 0.15), (0.0, 1.0)])
def test_dropout_counts_channels_even_when_all_values_are_missing(
    corpus, upstream, monkeypatch, drop_all, drop_each
):
    # Import through the validated checkout first.
    run(corpus, upstream)
    from simons.training.covariates import stream

    monkeypatch.setitem(stream.DEFAULTS, "drop_all", drop_all)
    monkeypatch.setitem(stream.DEFAULTS, "drop_each", drop_each)
    report = run(corpus, upstream)
    assert report["totals"]["windows_no_channels"] == report["totals"]["accepted"]
    assert report["totals"]["windows_with_finite_future"] == 0
    assert report["totals"]["channels_before_dropout"] == {"2": report["totals"]["accepted"]}


def test_cli_outputs_and_missing_covariates(corpus, upstream, tmp_path, monkeypatch):
    from scripts.corpus_v4.covariate_exposure import main

    previous_trace = sys.gettrace()
    out = tmp_path / "out"
    args = [
        "--corpus",
        str(corpus),
        "--upstream",
        str(upstream),
        "--stage",
        "stage3",
        "--size",
        "25m",
        "--pass",
        "main",
        "--batches",
        "1",
        "--seed",
        "3",
        "--output",
        str(out),
    ]
    assert main(args) == 0
    report = json.loads((out / "main.json").read_text())
    with (out / "main.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(report["per_source"])
    assert all(row["replay"] == "real-only statistical" for row in rows)
    args[args.index(str(corpus))] = str(tmp_path / "absent")
    with pytest.raises((ValueError, FileNotFoundError), match="covariates"):
        main(args)
    assert sys.gettrace() is previous_trace


def test_unavailable_pass_fails_loudly(corpus, upstream):
    from scripts.corpus_v4.covariate_exposure import audit

    with pytest.raises(ValueError, match="long.*stage3"):
        audit(
            corpus=corpus,
            upstream=upstream,
            stage="stage3",
            pass_name="long",
            size="25m",
            batches=1,
            seed=1,
        )


def test_usable_rows_match_exhaustive_upstream_endpoints(corpus, upstream):
    from scripts.corpus_v4.covariate_exposure import upstream_modules, usable_histories

    with upstream_modules(upstream) as modules:
        windows = modules[-1]
        real = windows.CovariateCorpus(corpus / "covariates")
        # More than twenty rejected endpoint candidates precede a valid window.
        target = np.full(340, np.nan, dtype=np.float32)
        target[210:290] = 1
        intermittent = np.ones(340, dtype=np.float32)
        intermittent[::3] = np.nan
        for row, values in zip(real.recs, (target, intermittent, np.full(340, np.nan))):
            row.update(target=values, covs=np.ones((1, len(values))), freq="D")
        real.recs = real.recs[:3]
        real.sources = ["a"]
        real.by_source = {"a": list(range(3))}
        for row in real.recs:
            row["source"] = "a"
        spec = dict(hists=[64, 128, 256], min_observed=0.75)
        counts, usable = usable_histories(real, windows, spec)
        expected, expected_rows = {}, set()
        for hist in spec["hists"]:
            found = set()
            for index, row in enumerate(real.recs):
                for horizon in (32, 64):
                    for end in range(len(row["target"]) + 1):

                        def integers(low, high=None):
                            if high is None:
                                return index if low == 3 else 0
                            if not low <= end < high:
                                raise StopIteration
                            return end

                        rng = SimpleNamespace(
                            integers=integers, choice=lambda *a, **k: horizon // 32
                        )
                        try:
                            got = real.train_sample(rng, hist, spec["min_observed"])
                        except StopIteration:
                            continue
                        if got is not None:
                            found.add(index)
                            break
                    if index in found:
                        break
            expected[str(hist)] = len(found)
            expected_rows.update(found)
        assert {str(h): counts["a"][str(h)] for h in spec["hists"]} == expected
        assert usable == expected_rows


def test_base_loader_guard_and_sampler_trace_restoration(corpus, upstream, monkeypatch):
    from scripts.corpus_v4.covariate_exposure import upstream_modules

    previous_trace = sys.gettrace()
    with upstream_modules(upstream) as modules:
        windows = modules[-1]

        def unexpected_base(*args, **kwargs):
            windows.BaseSeries([])

        monkeypatch.setattr(windows.pq, "read_table", unexpected_base)
        with pytest.raises(RuntimeError, match="base/univariate"):
            run(corpus, upstream)
    assert sys.gettrace() is previous_trace


def test_cli_in_a_fresh_process(corpus, upstream, tmp_path):
    out = tmp_path / "fresh-output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.corpus_v4.covariate_exposure",
            "--corpus",
            str(corpus),
            "--upstream",
            str(upstream),
            "--stage",
            "stage4-long",
            "--pass",
            "main",
            "--batches",
            "1",
            "--seed",
            "11",
            "--output",
            str(out),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "main.json").read_text())
    assert report["totals"]["accepted"] > 0
    assert report["configured_mix"]["real"] == 0.12


def test_print_source_hashes_needs_only_upstream(upstream):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.corpus_v4.covariate_exposure",
            "--upstream",
            str(upstream),
            "--print-source-hashes",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    hashes = json.loads(result.stdout)
    assert "simons/training/spot_job.py" in hashes
    assert "simons/training/config.py" in hashes
    assert "simons/training/covariates/stream.py" in hashes
    for name, digest in hashes.items():
        assert hashlib.sha256((upstream / "src" / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("module_name", ["spot_job", "config"])
def test_dependency_file_drift_refused_before_counting(corpus, upstream, monkeypatch, module_name):
    from scripts.corpus_v4 import covariate_exposure as exposure

    with exposure.upstream_modules(upstream) as modules:
        module = modules[0 if module_name == "spot_job" else 1]
        path = Path(module.__file__)
        original = path.read_bytes()

        def no_counting():
            pytest.fail("counting started before source verification")

        monkeypatch.setattr(exposure, "_counter", no_counting)
        try:
            path.write_bytes(original + b"\n# unrecorded recipe/config change\n")
            with pytest.raises(RuntimeError, match="unsupported.*" + module_name + r"\.py"):
                run(corpus, upstream)
        finally:
            path.write_bytes(original)


def test_unrecorded_module_loaded_during_corpus_build_refused(corpus, upstream, monkeypatch):
    from scripts.corpus_v4 import covariate_exposure as exposure

    with exposure.upstream_modules(upstream) as modules:
        windows = modules[-1]
        original_read = windows.pq.read_table
        path = upstream / "src/exposure_unrecorded.py"

        def read(*args, **kwargs):
            monkeypatch.setitem(sys.modules, "exposure_unrecorded", SimpleNamespace(__file__=path))
            return original_read(*args, **kwargs)

        monkeypatch.setattr(windows.pq, "read_table", read)
        path.write_text("# a newly loaded dependency\n")
        try:
            with pytest.raises(RuntimeError, match="unsupported.*exposure_unrecorded.py"):
                run(corpus, upstream)
        finally:
            path.unlink()


def test_recorded_dependency_loaded_elsewhere_refused(corpus, upstream, monkeypatch, tmp_path):
    from scripts.corpus_v4 import covariate_exposure as exposure

    with exposure.upstream_modules(upstream):
        module = sys.modules["simons.training.schedule"]
        monkeypatch.setattr(module, "__file__", str(tmp_path / "schedule.py"))
        with pytest.raises(RuntimeError, match="unsupported.*schedule.py"):
            run(corpus, upstream)


@pytest.mark.parametrize("target", ["folded_guard", "batches_constructor"])
def test_executed_code_with_pinned_source_lookup_refused(corpus, upstream, monkeypatch, target):
    from scripts.corpus_v4 import covariate_exposure as exposure

    with exposure.upstream_modules(upstream) as modules:
        stream, windows = modules[-2:]
        if target == "folded_guard":
            module, owner, attribute = windows, windows.CovariateCorpus, "train_sample"
            old = "if n < horizon + 64:\n                continue"
            new = "if n < horizon + 64: continue\n"
        else:
            module, owner, attribute = stream, stream.Batches, "__init__"
            old = "self.spec, self.seed, self.first_step = spec, seed, first_step"
            new = old + '; self.spec = {**spec, "hist_weights": [1, 0, 0, 0, 0, 0]}'
        original = getattr(owner, attribute)
        source = Path(module.__file__).read_text()
        changed = source.replace(old, new)
        assert changed != source
        namespace = {}
        exec(compile(changed, module.__file__, "exec", dont_inherit=True), vars(module), namespace)
        edited_owner = namespace[owner.__name__]
        pinned_source = inspect.getsource(original)
        monkeypatch.setattr(original, "__code__", getattr(edited_owner, attribute).__code__)
        assert inspect.getsource(original) == pinned_source
        with pytest.raises(RuntimeError, match="unsupported.*" + owner.__name__ + "." + attribute):
            run(corpus, upstream)


def test_wrapped_horizon_with_pinned_source_lookup_refused(corpus, upstream, monkeypatch):
    from scripts.corpus_v4 import covariate_exposure as exposure

    with exposure.upstream_modules(upstream) as modules:
        windows = modules[-1]
        original = windows.horizon_for

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return 2 * original(*args, **kwargs)

        assert inspect.getsource(wrapped) == inspect.getsource(original)
        monkeypatch.setattr(windows, "horizon_for", wrapped)
        with pytest.raises(RuntimeError, match="unsupported.*horizon_for"):
            run(corpus, upstream)


@pytest.mark.parametrize("replacement", ["alias", "partial"])
def test_horizon_function_binding_refused(corpus, upstream, monkeypatch, replacement):
    from scripts.corpus_v4 import covariate_exposure as exposure

    with exposure.upstream_modules(upstream) as modules:
        windows = modules[-1]
        substitute = (
            windows.source_weights
            if replacement == "alias"
            else functools.partial(windows.horizon_for)
        )
        monkeypatch.setattr(windows, "horizon_for", substitute)
        with pytest.raises(RuntimeError, match="unsupported.*horizon_for"):
            run(corpus, upstream)


def replace_source(monkeypatch, owner, attribute, source):
    """Give an in-memory upstream edit inspectable source, without changing its checkout."""
    original = getattr(owner, attribute)
    filename = str(original.__code__.co_filename) + ".exposure-test-" + attribute
    monkeypatch.setitem(
        linecache.cache, filename, (len(source), None, source.splitlines(True), filename)
    )
    namespace = {}
    exec(compile(source, filename, "exec"), original.__globals__, namespace)
    monkeypatch.setattr(owner, attribute, namespace[original.__name__])


@pytest.mark.parametrize("change", ["one_line_guard", "extra_observation_rule"])
def test_sampler_guard_drift_refused_before_counting(corpus, upstream, monkeypatch, change):
    from scripts.corpus_v4 import covariate_exposure as exposure

    with exposure.upstream_modules(upstream) as modules:
        windows = modules[-1]
        source = textwrap.dedent(inspect.getsource(windows.CovariateCorpus.train_sample))
        if change == "one_line_guard":
            changed = source.replace(
                "if n < horizon + 64:\n            continue", "if n < horizon + 64: continue"
            )
        else:
            changed = source.replace(
                "        return y, c, horizon",
                "        if np.nansum(y[-horizon:]) <= 0:\n            continue\n        return y, c, horizon",
            )
        assert changed != source
        replace_source(monkeypatch, windows.CovariateCorpus, "train_sample", changed)
        counted = []
        original = exposure._counter

        def counter():
            counted.append(True)
            return original()

        monkeypatch.setattr(exposure, "_counter", counter)
        with pytest.raises(RuntimeError, match="unsupported.*CovariateCorpus.train_sample"):
            run(corpus, upstream)
        assert not counted


@pytest.mark.parametrize(
    "module_index,attribute",
    [
        (2, "covariate_passes"),
        (2, "config_module.weights"),
        (2, "network.future_steps"),
        (3, "long_spec"),
        (3, "Batches.__init__"),
        (3, "Batches.__iter__"),
        (3, "CovariateStream.__init__"),
        (3, "make_batch"),
        (3, "tokens_per_window"),
        (3, "real_corpus"),
        (4, "CovariateCorpus.__init__"),
        (4, "horizon_for"),
        (4, "source_weights"),
        (4, "load_items"),
        (4, "_list_arrays"),
        (4, "_matrix_arrays"),
        (4, "corpus_files"),
        (4, "resolve"),
    ],
)
def test_sampler_helper_drift_refused(corpus, upstream, monkeypatch, module_index, attribute):
    from scripts.corpus_v4 import covariate_exposure as exposure

    with exposure.upstream_modules(upstream) as modules:
        owner = modules[module_index]
        path = attribute.split(".")
        for part in path[:-1]:
            owner = getattr(owner, part)
        source = textwrap.dedent(inspect.getsource(getattr(owner, path[-1])))
        # A different docstring changes the exact body, without changing its signature or rules.
        first = ast.parse(source).body[0].body[0]
        lines = source.splitlines(True)
        lines.insert(first.lineno - 1, "    'unsupported source revision'\n")
        replace_source(monkeypatch, owner, path[-1], "".join(lines))
        with pytest.raises(RuntimeError, match="unsupported.*" + attribute):
            run(corpus, upstream)


def test_nonuniform_and_zero_source_weights(corpus, upstream, monkeypatch):
    from scripts.corpus_v4 import covariate_exposure as exposure

    original = exposure.sampling_spec

    def weighted_spec(*args, **kwargs):
        spec = original(*args, **kwargs)
        spec["source_weights"] = {"a": 1, "b": 3, "c": 0, "short": 0, "sparse": 0}
        return spec

    monkeypatch.setattr(exposure, "sampling_spec", weighted_spec)
    report = run(corpus, upstream)
    sources = report["per_source"]
    for name in ("c", "short", "sparse"):
        assert sources[name]["draws"] == sources[name]["accepted"] == 0
        assert sources[name]["accepted_share"] == 0
    assert sources["a"]["accepted_share"] == pytest.approx(1 / 4, abs=0.06)
    assert sources["b"]["accepted_share"] == pytest.approx(3 / 4, abs=0.06)
    assert report["totals"]["draws"] == report["totals"]["accepted"]


@pytest.mark.parametrize("max_cov,expected_channels", [(3, 3), (32, 16)])
def test_channel_caps_and_fractional_finite_future_coverage(
    tmp_path, upstream, monkeypatch, max_cov, expected_channels
):
    from scripts.corpus_v4.covariate_exposure import upstream_modules

    corpus = tmp_path / "fractional-corpus"
    folder = corpus / "covariates/tiny"
    folder.mkdir(parents=True)
    length = 512
    # Every daily horizon is 32 or 64 steps: alternating finite/NaN values cover exactly half.
    partial = np.ones(length, dtype=np.float32)
    partial[::2] = np.nan
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    item_id="wide",
                    source="wide",
                    freq="D",
                    target=[1.0] * length,
                    covariates=[partial.tolist()] * 20,
                )
            ]
        ),
        folder / "train-000.parquet",
    )
    with upstream_modules(upstream) as modules:
        stream = modules[-2]
        monkeypatch.setitem(stream.DEFAULTS, "drop_all", 0.0)
        monkeypatch.setitem(stream.DEFAULTS, "drop_each", 0.0)
        # Exercise both train_sample's MAX_COV=16 and make_batch's configurable cap.
        monkeypatch.setitem(stream.DEFAULTS, "max_cov", max_cov)
        report = run(corpus, upstream)
    totals = report["totals"]
    assert report["effective_spec"]["max_cov"] == max_cov
    assert (
        totals["channels_before_dropout"]
        == totals["channels_after_dropout"]
        == {str(expected_channels): totals["accepted"]}
    )
    assert totals["windows_with_finite_future"] == totals["accepted"]
    assert totals["windows_no_channels"] == 0
    # Each retained channel has 16 observed steps per 32-step horizon (or 32 per 64).
    expected_share = (expected_channels * 16) / (expected_channels * 32)
    assert 0 < totals["mean_finite_future_share"] < 1
    assert totals["mean_finite_future_share"] == expected_share
    assert report["per_source"]["wide"]["mean_finite_future_share"] == expected_share
