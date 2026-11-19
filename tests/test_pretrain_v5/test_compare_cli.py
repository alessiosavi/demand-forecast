"""P7b compares copied, validated T1 units without touching training state."""

import copy
import json
import shutil
from dataclasses import replace

import pytest

from demand_forecast.pretrain.v5 import job, t1
from demand_forecast.pretrain.v5.checkpoint import file_digest, read_manifest, write_manifest
from demand_forecast.pretrain.v5.config import V5TrainConfig, identity
from tests.helpers.v5_loop import config

CODE = "a" * 40
ENV = "b" * 64
SEED = 7
EPISODES = 2
MAX_PANELS = 12


@pytest.fixture(scope="module")
def tiny_units(tmp_path_factory):
    """Train each requested arm once; retain only independent evaluation copies."""
    root = tmp_path_factory.mktemp("compare-units")
    base = config(steps=1, workers=0)
    base = replace(
        base, sampler=replace(base.sampler, designate_target=True, contexts=(64,), horizon=(16, 24))
    )
    base_file = root / "base.yaml"
    base.to_yaml(base_file)
    paths = copy.deepcopy(t1.PATHS)
    for name, entry in paths.items():
        source = entry["both"].setdefault("source", {})
        source.update(max_items=4, length=(128, 160))
        if name == "bank":
            source.update(freq="h", length=(400, 480))
        if name == "routing":
            entry["both"].setdefault("sampler", {})["episodes_per_batch"] = 2
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(t1, "T1_BASE", base)
        patch.setattr(t1, "PATHS", paths)
        saved = {}

        def build(path):
            if path not in saved:
                run = root / f"run-{path}"
                arms = {}
                for arm in t1.PATHS[path]["arms"]:
                    command = [
                        "t1",
                        "--run-dir",
                        str(run),
                        "--code",
                        CODE,
                        "--job-sha256",
                        "c" * 64,
                        "--env-sha256",
                        ENV,
                        "--base",
                        str(base_file),
                        "--path",
                        path,
                        "--arm",
                        arm,
                        "--seed",
                        str(SEED),
                        "--episodes",
                        str(EPISODES),
                        "--max-panels",
                        str(MAX_PANELS),
                    ]
                    assert job.main(command) == 0
                    source = run / f"state/t1/{path}-{arm}-s{SEED}/eval-{EPISODES}-{MAX_PANELS}"
                    destination = root / path / arm
                    shutil.copytree(source, destination)
                    assert read_manifest(destination / "unit.json") is not None
                    arms[arm] = destination
                shutil.rmtree(run)
                saved[path] = arms
            return base_file, saved[path]

        yield build


@pytest.fixture
def copied(tiny_units, tmp_path):
    def make(path="static_category"):
        base, originals = tiny_units(path)
        arms = {}
        for name, source in originals.items():
            arms[name] = tmp_path / "copies" / path / name
            shutil.copytree(source, arms[name])
        out = tmp_path / "output"
        command = [
            "t1-compare",
            "--base",
            str(base),
            "--code",
            CODE,
            "--path",
            path,
            "--seed",
            str(SEED),
            "--episodes",
            str(EPISODES),
            "--max-panels",
            str(MAX_PANELS),
            "--out",
            str(out),
        ]
        return base, arms, out, command

    return make


def invoke(command, arms):
    return job.main([*command, *(part for a, d in arms.items() for part in ("--arm", f"{a}={d}"))])


def rewrite_unit(directory, **changes):
    path = directory / "unit.json"
    record = json.loads(path.read_text())
    record.update(changes)
    path.write_text(json.dumps(record))


def rewrite_result(directory, **changes):
    path = directory / "result.json"
    record = json.loads(path.read_text())
    record.update(changes)
    path.write_text(json.dumps(record))
    manifest = json.loads((directory / "unit.json").read_text())
    write_manifest(directory / "unit.json", manifest)


def snapshot(arms):
    return {str(p): p.read_bytes() for d in arms.values() for p in d.rglob("*") if p.is_file()}


def rejected(command, arms, out, capsys, arm, reason):
    before = snapshot(arms)
    assert invoke(command, arms) == 1
    error = capsys.readouterr().err.lower()
    assert arm.lower() in error
    assert reason.lower() in error
    assert not out.exists()
    assert snapshot(arms) == before
    return error


@pytest.mark.parametrize("path", ["static_category", "category_mean"])
def test_copied_units_publish_figures_slices_hashes_and_deterministic_bytes(
    copied, path, monkeypatch
):
    base, arms, out, command = copied(path)
    eval_set = t1.eval_set
    observed = []

    def observe(*args, **kwargs):
        observed.append(kwargs.get("base"))
        return eval_set(*args, **kwargs)

    monkeypatch.setattr(t1, "eval_set", observe)
    before = snapshot(arms)
    assert invoke(command, arms) == 0
    assert observed == [V5TrainConfig.from_yaml(base)]
    assert [p.name for p in out.iterdir()] == ["compare.json"]
    payload = (out / "compare.json").read_bytes()
    report = json.loads(payload)
    assert report["path"] == path
    assert report["outcome"] == "inconclusive"
    assert "removal" not in report
    assert report["units"] == {a: file_digest(d / "unit.json") for a, d in arms.items()}
    assert len(report["comparisons"]) == 1
    comparison = report["comparisons"][0]
    assert (comparison["true"], comparison["null"], comparison["decides"]) == ("true", "null", True)
    true = json.loads((arms["true"] / "result.json").read_text())["loss"]
    null = json.loads((arms["null"] / "result.json").read_text())["loss"]
    assert comparison["all"]["n"] == len(true) == EPISODES
    assert comparison["all"]["gain"] == pytest.approx((sum(null) - sum(true)) / sum(null))
    assert comparison["all"]["low"] is None and comparison["all"]["high"] is None
    assert comparison["outcome"] == "inconclusive"
    if path == "category_mean":
        assert comparison["slices"]
        assert set(comparison["slices"]) <= {"known", "past", "rare", "late_first"}
    else:
        assert comparison["slices"] == {}
    assert invoke(command, dict(reversed(list(arms.items())))) == 0
    assert (out / "compare.json").read_bytes() == payload
    other = out.with_name("other-output")
    command[command.index(str(out))] = str(other)
    assert invoke(command, arms) == 0
    assert (other / "compare.json").read_bytes() == payload
    assert snapshot(arms) == before


@pytest.mark.parametrize(
    "path,missing",
    [
        ("static_category", "null"),
        ("selection", "gates_one"),
        ("bank", "null_steps"),
        ("routing", "null_rolled"),
    ],
)
def test_missing_arm_is_named(copied, capsys, path, missing):
    _, arms, out, command = copied(path)
    del arms[missing]
    rejected(command, arms, out, capsys, missing, "missing")


def test_extra_arm_is_named(copied, capsys):
    _, arms, out, command = copied()
    arms["unexpected"] = arms["true"]
    rejected(command, arms, out, capsys, "unexpected", "extra")


def test_repeated_arm_is_named(copied, capsys):
    _, arms, out, command = copied()
    command += ["--arm", f"null={arms['null']}"]
    rejected(command, arms, out, capsys, "null", "repeat")


def test_malformed_arm_is_named(copied, capsys):
    _, arms, out, command = copied()
    command += ["--arm", "broken"]
    rejected(command, arms, out, capsys, "broken", "name=dir")


@pytest.mark.parametrize("kind", ["missing", "file", "symlink", "broken_symlink"])
def test_arm_requires_an_existing_real_directory(copied, tmp_path, capsys, kind):
    _, arms, out, command = copied()
    directory = tmp_path / "invalid-arm"
    if kind == "file":
        directory.write_text("not a directory")
    elif kind in ("symlink", "broken_symlink"):
        directory.symlink_to(arms["null"] if kind == "symlink" else tmp_path / "absent")
    arms["null"] = directory
    rejected(command, arms, out, capsys, "null", "directory")


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "known_driver"),
        ("arm", "true"),
        ("seed", 8),
        ("code", "d" * 40),
        ("base_sha256", "e" * 64),
        ("episodes", 3),
        ("max_panels", 13),
        ("identity", "f" * 64),
        ("env_sha256", None),
        ("eval_digest", "0" * 64),
    ],
)
def test_unit_metadata_mismatch_is_named(copied, capsys, field, value):
    _, arms, out, command = copied()
    rewrite_unit(arms["null"], **{field: value})
    rejected(command, arms, out, capsys, "null", field)


@pytest.mark.parametrize("shared", [False, True])
def test_environment_must_match_even_when_identity_is_valid(copied, capsys, shared):
    base_file, arms, out, command = copied()
    base = V5TrainConfig.from_yaml(base_file)
    for arm in arms if shared else ("null",):
        env = "d" * 64
        rewrite_unit(
            arms[arm],
            env_sha256=env,
            identity=identity(
                t1.t1_config("static_category", arm, seed=SEED, base=base),
                code=CODE,
                corpus=None,
                init=None,
                env=env,
            ),
        )
    if shared:
        assert invoke(command, arms) == 0
        assert json.loads((out / "compare.json").read_text())["env_sha256"] == "d" * 64
    else:
        rejected(command, arms, out, capsys, "null", "env_sha256")


@pytest.mark.parametrize("shared", [False, True])
def test_eval_digest_must_match_results_arms_and_rebuilt_set(copied, capsys, shared):
    _, arms, out, command = copied()
    for arm in arms if shared else ("null",):
        rewrite_result(arms[arm], eval_digest="d" * 64)
        rewrite_unit(arms[arm], eval_digest="d" * 64)
    rejected(command, arms, out, capsys, "true" if shared else "null", "eval_digest")


@pytest.mark.parametrize(
    "damage", ["hash", "size", "missing", "unlisted", "manifest", "checkpoint"]
)
def test_invalid_unit_or_artifact_is_rejected_without_mutation(copied, capsys, damage):
    _, arms, out, command = copied()
    directory = arms["null"]
    result = directory / "result.json"
    if damage == "hash":
        result.write_text(result.read_text().replace('"loss"', '"lost"'))
    elif damage == "size":
        result.write_text(result.read_text() + " ")
    elif damage == "missing":
        result.unlink()
    elif damage == "unlisted":
        manifest = json.loads((directory / "unit.json").read_text())
        del manifest["files"]["result.json"]
        rewrite_unit(directory, files=manifest["files"])
    elif damage == "checkpoint":
        (directory / "checkpoint-final.pt").write_bytes(b"damaged")
    else:
        (directory / "unit.json").write_text("not json")
    rejected(command, arms, out, capsys, "null", "unit.json")


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"eval_digest": "f" * 64}, "eval_digest"),
        ({"loss": [1.0]}, "loss"),
        ({"loss": [-1.0, 1.0]}, "loss"),
    ],
)
def test_resigned_invalid_result_names_arm(copied, capsys, change, reason):
    _, arms, out, command = copied()
    rewrite_result(arms["null"], **change)
    rejected(command, arms, out, capsys, "null", reason)


def controlled_episodes(monkeypatch, arms, episodes, losses):
    """Keep real unit artifacts/identities; isolate aggregation with paired data."""
    monkeypatch.setattr(t1, "eval_set", lambda *a, **kw: episodes)
    digest = t1.eval_digest(episodes)
    for arm, directory in arms.items():
        rewrite_result(directory, loss=losses[arm], eval_digest=digest)
        rewrite_unit(directory, eval_digest=digest)


@pytest.mark.parametrize("bad_null", [None, "null_uniform", "null_rolled"])
def test_routing_both_nulls_decide(copied, monkeypatch, bad_null):
    _, arms, out, command = copied("routing")
    episodes = [{"slices": ()} for _ in range(30)]
    losses = {"true": [0.9] * 30, "null_uniform": [1.0] * 30, "null_rolled": [1.0] * 30}
    if bad_null:
        losses[bad_null] = [0.9] * 30
    controlled_episodes(monkeypatch, arms, episodes, losses)
    assert invoke(command, arms) == 0
    report = json.loads((out / "compare.json").read_text())
    assert report["outcome"] == ("fail" if bad_null else "pass")
    assert {c["null"] for c in report["comparisons"] if c["decides"]} == {
        "null_uniform",
        "null_rolled",
    }


@pytest.mark.parametrize(
    "slice_state,want", [("pass", "pass"), ("fail", "fail"), ("short", "inconclusive")]
)
def test_bank_null_all_and_every_required_slice_decide(copied, monkeypatch, slice_state, want):
    _, arms, out, command = copied("bank")
    episodes = [
        {"slices": (name,)}
        for name in ("seasonal_lag", "intermittent", "level_shift")
        for _ in range(29 if slice_state == "short" and name == "level_shift" else 30)
    ]
    true = [
        0.9 if e["slices"] != ("level_shift",) or slice_state != "fail" else 1.0 for e in episodes
    ]
    controlled_episodes(
        monkeypatch,
        arms,
        episodes,
        {
            "true": true,
            "null_all": [1.0] * len(episodes),
            "null_steps": true,
        },
    )
    assert invoke(command, arms) == 0
    report = json.loads((out / "compare.json").read_text())
    comparisons = {c["null"]: c for c in report["comparisons"]}
    assert report["outcome"] == want
    assert comparisons["null_all"]["decides"] is True
    assert comparisons["null_steps"]["decides"] is False
    assert comparisons["null_steps"]["outcome"] == "fail"
    assert comparisons["null_all"]["all"]["outcome"] == "pass"
    assert comparisons["null_all"]["slices"]["level_shift"]["outcome"] == (
        "inconclusive" if slice_state == "short" else slice_state
    )
    assert set(comparisons["null_all"]["slices"]) == {"seasonal_lag", "intermittent", "level_shift"}


def test_reported_slice_failure_does_not_decide(copied, monkeypatch):
    _, arms, out, command = copied("category_mean")
    episodes = [{"slices": ("rare",) if i < 30 else ("known",)} for i in range(90)]
    controlled_episodes(
        monkeypatch,
        arms,
        episodes,
        {
            "true": [1.0] * 30 + [0.8] * 60,
            "null": [1.0] * 90,
        },
    )
    assert invoke(command, arms) == 0
    report = json.loads((out / "compare.json").read_text())
    assert report["outcome"] == "pass"
    assert report["comparisons"][0]["slices"]["rare"]["outcome"] == "fail"


@pytest.mark.parametrize("damage", [None, "eval_digest", "loss", "derived"])
def test_category_comparison_and_removal_use_known_cohort(copied, monkeypatch, capsys, damage):
    _, arms, out, command = copied("category_timing")
    episodes = [{"slices": ("known",)}] * 30 + [{"slices": ("past",)}] * 30
    controlled_episodes(
        monkeypatch,
        arms,
        episodes,
        {"true": [0.8] * 30 + [1.2] * 30, "null": [1.0] * 60},
    )
    true = arms["true"]
    off = true / "result-derive-off.json"
    result = {
        "eval_digest": t1.eval_digest(episodes),
        "loss": [0.7] * 30 + [2.4] * 30,
        "derive": [],
        "derived": [[] for _ in episodes],
    }
    if damage == "eval_digest":
        result["eval_digest"] = "d" * 64
    if damage == "loss":
        result["loss"].pop()
        result["derived"].pop()
    if damage == "derived":
        result["derived"][0] = ["category~onset"]
    manifest = read_manifest(true / "unit.json")
    off.write_text(json.dumps(result))
    manifest["files"][off.name] = {}
    write_manifest(true / "unit.json", manifest)
    if damage:
        assert damage in rejected(command, arms, out, capsys, "true", off.name)
        return

    assert invoke(command, arms) == 0
    report = json.loads((out / "compare.json").read_text())
    assert report["comparisons"][0]["overall"]["outcome"] == "fail"
    assert report["comparisons"][0]["slices"]["known"]["n"] == 30
    assert report["outcome"] == "pass"
    assert report["removal"]["n"] == 30
    for key in ("gain", "low", "high"):
        assert report["removal"][key] == pytest.approx((0.7 - 0.8) / 0.7)


@pytest.mark.parametrize("damage", [None, "copy", "true", "null"])
def test_category_spread_requires_matching_presentations(copied, capsys, damage):
    _, arms, out, command = copied("category_spread")
    true = arms["true"]
    off = true / "result-derive-off.json"
    if damage == "copy":
        manifest = read_manifest(true / "unit.json")
        shutil.copyfile(true / "result.json", off)
        write_manifest(true / "unit.json", manifest)
        assert "derive must be empty" in rejected(command, arms, out, capsys, "true", off.name)
    elif damage:
        rewrite_result(arms[damage], derive=[])
        assert "derive" in rejected(command, arms, out, capsys, damage, "result.json")
    else:
        assert invoke(command, arms) == 0
        assert json.loads((out / "compare.json").read_text())["removal"]["n"] > 0


def test_selection_publishes_all_r2_decisions(copied):
    _, arms, out, command = copied("selection")
    assert invoke(command, arms) == 0
    report = json.loads((out / "compare.json").read_text())
    selection = report["selection_report"]
    assert set(selection) == {"pruning", "forced", "shuffle", "gates_one", "pruned", "outcome"}
    assert selection["outcome"] == "inconclusive"
    assert selection["shuffle"]["all"]["n"] == EPISODES
    assert selection["gates_one"]["n"] == EPISODES
    assert selection["pruned"] == json.loads((arms["true"] / "result.json").read_text())["pruned"]
    assert set(report["units"]) == {"true", "null", "gates_one"}
    assert [p.name for p in out.iterdir()] == ["compare.json"]


@pytest.mark.parametrize(
    "comparison_outcome,selection_outcome,want",
    [
        ("pass", "pass", "pass"),
        ("pass", "inconclusive", "inconclusive"),
        ("pass", "fail", "fail"),
        ("inconclusive", "pass", "inconclusive"),
        ("inconclusive", "inconclusive", "inconclusive"),
        ("inconclusive", "fail", "fail"),
        ("fail", "pass", "fail"),
        ("fail", "inconclusive", "fail"),
        ("fail", "fail", "fail"),
    ],
)
def test_selection_verdict_combines_comparison_and_r2(
    copied, monkeypatch, comparison_outcome, selection_outcome, want
):
    _, arms, out, command = copied("selection")
    monkeypatch.setattr(t1, "compare", lambda *a, **kw: {"outcome": comparison_outcome})
    monkeypatch.setattr(t1, "selection_report", lambda *a, **kw: {"outcome": selection_outcome})
    assert invoke(command, arms) == 0
    report = json.loads((out / "compare.json").read_text())
    assert report["comparisons"][0]["outcome"] == comparison_outcome
    assert report["selection_report"]["outcome"] == selection_outcome
    assert report["outcome"] == want


def test_failure_preserves_existing_output(copied, capsys):
    _, arms, out, command = copied()
    out.mkdir()
    (out / "compare.json").write_bytes(b"previous report")
    rewrite_unit(arms["null"], seed=99)
    assert invoke(command, arms) == 1
    assert "null" in capsys.readouterr().err
    assert (out / "compare.json").read_bytes() == b"previous report"
    assert [p.name for p in out.iterdir()] == ["compare.json"]
