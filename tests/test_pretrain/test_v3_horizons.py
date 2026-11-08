"""Full-window feasibility, exact v3 continuation and realized batch exposure."""

import json
from collections import Counter
from contextlib import ExitStack, closing
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError

pytest.importorskip("pyarrow")
pytestmark = pytest.mark.bench

from demand_forecast.data.encoders import StaticEncoder  # noqa: E402
from demand_forecast.models.base import ModelSpec  # noqa: E402
from demand_forecast.models.registry import build_model  # noqa: E402
from demand_forecast.pretrain.config import (  # noqa: E402
    FEV_HORIZONS,
    PretrainConfig,
    pretrain_identity,
)
from demand_forecast.pretrain.corpus import (  # noqa: E402
    ShardRef,
    corpus_manifest,
    shard_panel,
    write_manifest,
)
from demand_forecast.pretrain.sampler import (  # noqa: E402
    CorpusSampler,
    V3ShardStream,
    sampler_state_from_dict,
)
from tests.helpers.corpus_fixture import write_corpus_fixture  # noqa: E402
from tests.helpers.reference_batch import assert_batches_equal  # noqa: E402
from tests.test_pretrain.test_recipe_grids import _row, _shard  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _ref(tmp_path, rows, index=0):
    folder = tmp_path / str(index)
    folder.mkdir()
    path = _shard(folder, rows)
    return ShardRef(path, "chronos_real", index, "unused", path.stat().st_size)


def _stream(refs, **kwargs):
    return V3ShardStream(
        refs,
        "chronos_real",
        **(
            dict(
                horizon=288,
                max_context=32,
                min_context=8,
                batch_size=4,
                seed=11,
                encoder=StaticEncoder().fit(shard_panel([]), []),
                prefetch=0,
                period_policy="frequency",
                grid_policy="source",
                scale_floor_frac=0.0,
                horizon_policy="fev_set",
            )
            | kwargs
        ),
    )


def test_pinned_set_and_200_batches_cover_every_feasible_horizon(tmp_path, record_property):
    tasks = yaml.safe_load(
        (ROOT / "demand_forecast/bench/tasks/fev_bench_100_193e4fc.yaml").read_text()
    )
    assert set(FEV_HORIZONS) == {task["horizon"] for task in tasks["tasks"]}
    ref = _ref(tmp_path, [_row(0, target=np.arange(700) ** 2)])
    histogram = Counter()
    with closing(_stream([ref], shard_visit_quota=800)) as stream:
        for _ in range(200):
            batch = next(stream)
            horizon = batch.labels.shape[1]
            histogram[horizon] += 1
            assert batch.label_mask.all()
            assert (batch.origin_pos >= 8).all()
            assert (batch.origin_pos + horizon <= 700).all()
            assert batch.target.shape[1] <= 32
        assert stream.state().visit_cursor == stream.state().visit_quota == 800
    assert set(histogram) == set(FEV_HORIZONS)
    record_property("realized_horizon_histogram", json.dumps(dict(sorted(histogram.items()))))


def test_pairs_sample_series_uniformly_despite_unequal_lengths(tmp_path, record_property):
    lengths = np.array([1400, 26])
    ref = _ref(tmp_path, [_row(i, target=np.arange(n)) for i, n in enumerate(lengths)])
    draws = Counter()
    with closing(
        _stream([ref], horizon=5, horizon_policy="fixed", shard_visit_quota=400)
    ) as stream:
        for _ in range(100):
            batch = next(stream)
            draws.update(batch.series_index.tolist())
            assert (batch.origin_pos >= 8).all()
            assert (batch.origin_pos.numpy() + 5 <= lengths[batch.series_index.numpy()]).all()
        assert stream.state().visit_cursor == stream.state().visit_quota == 400
    frequencies = {row: draws[row] / 400 for row in range(2)}
    record_property("realized_series_frequencies", json.dumps(frequencies))
    assert sum(draws.values()) == 400
    assert all(0.35 <= frequency <= 0.65 for frequency in frequencies.values()), frequencies


def test_group_feasibility_skips_long_horizons_and_respects_period(tmp_path):
    ref = _ref(
        tmp_path,
        [
            _row(0, target=np.arange(13)),  # D, minimum 8: exactly H=5 at origin 8.
            _row(1, "h", target=np.arange(33)),  # m=24, origins >=25, only H=5/8.
            _row(2, "5min", target=np.arange(293)),  # m=288: no complete H=5 window.
        ],
    )
    seen = {}
    with ExitStack() as stack:
        a, b = [
            stack.enter_context(closing(_stream([ref], shard_visit_quota=80))) for _ in range(2)
        ]
        for _ in range(100):
            first, second = next(a), next(b)
            assert_batches_equal(first, second)
            assert len(set(first.series_index.tolist())) == 1
            row = first.series_index[0].item()
            horizon = first.labels.shape[1]
            seen.setdefault(row, set()).add(horizon)
            assert row in (0, 1)
            assert (first.origin_pos >= (8 if row == 0 else 25)).all()
            assert (first.origin_pos + horizon <= (13 if row == 0 else 33)).all()
        assert a.state() == b.state()
    assert seen == {0: {5}, 1: {5, 8}}


@pytest.mark.parametrize("prefetch", [0, 1])
def test_empty_source_raises_and_empty_shards_are_skipped(tmp_path, prefetch):
    short = _ref(tmp_path, [_row(0, target=np.arange(12))], 0)
    empty = _ref(tmp_path, [], 1)
    usable = _ref(tmp_path, [_row(0, target=np.arange(13))], 2)
    with closing(_stream([short, empty], prefetch=prefetch)) as stream:
        with pytest.raises(ValueError, match="chronos_real source has no usable windows"):
            next(stream)
    with closing(_stream([short, usable, empty], prefetch=prefetch)) as stream:
        for _ in range(5):
            batch = next(stream)
            assert batch.size == 1 and batch.labels.shape[1] == 5


@pytest.mark.parametrize("prefetch", [0, 1])
@pytest.mark.parametrize("saved_after", [0, 1, 3, 4, 7, 12])
def test_v3_all_tensors_resume_at_pair_tail_shard_and_pass_boundaries(
    tmp_path, prefetch, saved_after
):
    refs = [
        _ref(
            tmp_path, [_row(0, target=np.arange(700) ** 2), _row(1, "h", target=np.arange(90))], 0
        ),
        _ref(tmp_path, [_row(2, target=np.arange(12))], 1),
        _ref(tmp_path, [_row(3, "MS", target=np.arange(420) ** 2)], 2),
    ]
    with ExitStack() as stack:
        full, original, restored = [
            stack.enter_context(
                closing(
                    _stream(
                        refs, prefetch=(1 - prefetch if i == 0 else prefetch), shard_visit_quota=10
                    )
                )
            )
            for i in range(3)
        ]
        expected = [next(full) for _ in range(28)]
        first = [next(original) for _ in range(saved_after)]
        snapshot = original.state()
        frozen = deepcopy(snapshot)
        next(original)
        assert frozen == snapshot
        next(restored)  # Must discard an active builder and stale prefetch.
        state = sampler_state_from_dict(
            dict(step=0, tsmixup=asdict(snapshot), kernel_synth=asdict(snapshot))
        )
        restored.load_state(state.tsmixup)
        rest = [next(restored) for _ in range(28 - saved_after)]
        for actual, wanted in zip(first + rest, expected, strict=True):
            assert_batches_equal(actual, wanted)
        assert restored.state() == full.state()
        assert snapshot == frozen
        assert [batch.size for batch in expected[:3]] == [4, 4, 2]


def test_weighted_horizon_streams_restore_scheduler_and_visit_rng(tmp_path):
    ref = _ref(tmp_path, [_row(0, target=np.arange(700) ** 2)])
    sources = {kind: [replace(ref, kind=kind)] for kind in ("tsmixup", "chronos_real")}
    kwargs = dict(
        sources=sources,
        weights={"tsmixup": 2, "chronos_real": 1},
        recipe="v3",
        period_policy="frequency",
        grid_policy="source",
        horizon_policy="fev_set",
        shard_visit_quota={"tsmixup": 10, "chronos_real": 7},
        horizon=288,
        max_context=32,
        min_context=8,
        batch_size=4,
        seed=11,
    )
    with closing(CorpusSampler(**kwargs)) as full, closing(CorpusSampler(**kwargs)) as restored:
        batches = full.batches()
        for _ in range(7):
            next(batches)
        restored.load_state(sampler_state_from_dict(asdict(full.state())))
        continuation = restored.batches()
        for _ in range(25):
            assert_batches_equal(next(batches), next(continuation))
            assert full.last_kind == restored.last_kind
        assert full.state() == restored.state()


def test_missing_labels_are_masked_and_do_not_define_geometric_feasibility(tmp_path):
    values = np.arange(13, dtype=float)
    values[10] = np.nan
    ref = _ref(tmp_path, [_row(0, target=values)])
    with closing(_stream([ref])) as stream:
        batch = next(stream)
        assert batch.label_mask.tolist() == [[True, True, False, True, True]]
        assert batch.loss_eligible.all()
    # Short-horizon eligibility must not see labels beyond the selected future.
    values[8:13] = np.nan
    ref2 = _ref(tmp_path, [_row(0, target=np.r_[values, np.arange(20)])], 1)
    with closing(_stream([ref2])) as stream:
        batch = stream._read(ref2.path)
        from demand_forecast.pretrain.corpus import shard_panels
        from demand_forecast.pretrain.sampler import V3BatchBuilder

        builder = V3BatchBuilder(shard_panels(batch.records), 288, 32, stream.encoder)
        short = builder.build([(0, 8)], horizon=5)
        assert not short.loss_eligible.any() and not short.label_mask.any()


def test_288_batch_runs_v3_model_forward_backward(tmp_path, tiny_pretrain_cfg):
    ref = _ref(tmp_path, [_row(0, target=np.arange(900, dtype=float) ** 2)])
    with closing(_stream([ref], max_context=512)) as stream:
        for _ in range(200):
            batch = next(stream)
            if batch.labels.shape[1] == 288:
                break
        else:
            pytest.fail("fixture never drew H=288")
        cfg = PretrainConfig.model_validate(tiny_pretrain_cfg.model_dump() | {"recipe": "v3"})
        spec = ModelSpec.from_panel(shard_panel([]), stream.encoder, cfg.forecast, 16, 4)
        model = build_model("v2", spec, cfg.model.backbone_kwargs())
        output = model(batch)
        assert output.shape == (batch.size, 288, len(cfg.forecast.quantiles))
        assert torch.isfinite(output).all()
        output.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_v3_train_resume_weights_optimizer_telemetry_and_artifacts(tmp_path, tiny_pretrain_cfg):
    from demand_forecast.core.training import InterruptedRun
    from demand_forecast.pretrain.train import pretrain

    for kind in ("tsmixup", "kernel_synth"):
        for index in (0, 1):
            write_corpus_fixture(
                tiny_pretrain_cfg.corpus.root,
                kind=kind,
                index=index,
                n_series=3,
                lengths=(340, 350),
                seed=index,
            )
    write_manifest(
        corpus_manifest(tiny_pretrain_cfg.corpus.root),
        tiny_pretrain_cfg.corpus.root / "manifest.json",
    )
    raw = tiny_pretrain_cfg.model_dump() | {
        "recipe": "v3",
        "horizon_policy": "fev_set",
        "period_policy": "frequency",
        "grid_policy": "source",
        "shard_visit_quota": {"tsmixup": 10, "kernel_synth": 7},
    }
    raw["forecast"]["horizon"] = 288
    raw["training"].update(max_steps=12, val_batch_size=4)
    cfg = PretrainConfig.model_validate(raw)
    full = pretrain(cfg, tmp_path / "full")
    with pytest.raises(InterruptedRun):
        pretrain(cfg, tmp_path / "part", stop_after=4)
    partial_progress = json.loads((tmp_path / "part/checkpoints/progress.json").read_text())
    assert sum(partial_progress["timings"]["horizon_histogram"].values()) == 4
    resumed = pretrain(cfg, tmp_path / "part", resume=True)
    assert full.history == resumed.history
    checkpoints = [
        torch.load(tmp_path / d / "checkpoints/latest.pt", weights_only=False)
        for d in ("full", "part")
    ]

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, list):
            for x, y in zip(a, b, strict=True):
                equal(x, y)
        else:
            assert a == b

    for key in (
        "model",
        "best_state",
        "optimizer",
        "scheduler",
        "sampler_state",
        "best_step",
        "best_score",
    ):
        equal(checkpoints[0][key], checkpoints[1][key])
    for key in ("sampling_exposure", "horizon_histogram"):
        expected = full.timings[key]
        assert expected == resumed.timings[key]
        for directory in ("full", "part"):
            progress = json.loads((tmp_path / directory / "checkpoints/progress.json").read_text())
            manifest = json.loads((tmp_path / directory / "bundle/manifest.json").read_text())
            assert progress["timings"][key] == manifest[key] == expected
    counted = Counter((row["kind"], row["group"], str(row["horizon"])) for row in full.history)
    realized = {
        (kind, group, horizon): cell["batches"]
        for kind, groups in full.timings["sampling_exposure"].items()
        for group, horizons in groups.items()
        for horizon, cell in horizons.items()
    }
    assert counted == realized
    cells = [
        cell
        for groups in full.timings["sampling_exposure"].values()
        for horizons in groups.values()
        for cell in horizons.values()
    ]
    assert sum(c["batches"] for c in cells) == 12
    assert sum(full.timings["horizon_histogram"].values()) == 12
    assert all(c["eligible_labels"] == c["label_cells"] > 0 for c in cells)


@pytest.mark.parametrize(
    "options",
    [
        {"horizon_policy": "fev_set"},
        {"shard_visit_quota": {"tsmixup": 7}},
    ],
)
def test_new_controls_hash_roundtrip_and_v2_rejection(tiny_pretrain_cfg, tmp_path, options):
    raw = tiny_pretrain_cfg.model_dump()
    with pytest.raises(ValidationError, match="recipe v3"):
        PretrainConfig.model_validate(raw | options)
    raw["forecast"]["horizon"] = 288
    base = PretrainConfig.model_validate(raw | {"recipe": "v3"})
    changed = PretrainConfig.model_validate(raw | {"recipe": "v3"} | options)
    assert pretrain_identity(base, "2.0.0", {}) != pretrain_identity(changed, "2.0.0", {})
    path = tmp_path / "config.yaml"
    changed.to_yaml(path)
    assert PretrainConfig.from_yaml(path) == changed


@pytest.mark.parametrize(
    "options",
    [
        {"horizon_policy": "typo"},
        {"horizon_policy": "fev_set"},
        {"shard_visit_quota": {"tsmixup": 0}},
        {"shard_visit_quota": {"tsmixup": True}},
        {"shard_visit_quota": {"unknown": 3}},
    ],
)
def test_invalid_sampling_controls(tiny_pretrain_cfg, options):
    with pytest.raises(ValidationError):
        PretrainConfig.model_validate(tiny_pretrain_cfg.model_dump() | {"recipe": "v3"} | options)


def test_explicit_new_defaults_preserve_v2_config_identity_and_state(tiny_pretrain_cfg):
    cfg = tiny_pretrain_cfg
    raw = cfg.model_dump(mode="json")
    explicit = PretrainConfig.model_validate(
        raw | {"horizon_policy": "fixed", "shard_visit_quota": {}}
    )
    assert explicit.model_dump(mode="json") == raw
    assert not {"horizon_policy", "shard_visit_quota"} & raw.keys()
    assert pretrain_identity(cfg, "2.0.0", {}) == pretrain_identity(explicit, "2.0.0", {})
    refs = corpus_manifest(cfg.corpus.root)
    with closing(
        CorpusSampler(
            sources={"tsmixup": [r for r in refs if r.kind == "tsmixup"]},
            weights={"tsmixup": 1},
            horizon=4,
            max_context=16,
            min_context=8,
            batch_size=4,
            seed=0,
        )
    ) as sampler:
        next(sampler.batches())
        state = asdict(sampler.state())
        assert set(state["streams"]["tsmixup"]) == {
            "pass_index",
            "shard_cursor",
            "batch_cursor",
            "rng_state",
        }
        assert asdict(sampler_state_from_dict(state)) == state


@pytest.mark.parametrize(
    "change",
    [
        {"visit_cursor": 11},
        {"visit_quota": 11},
        {"batch_cursor": 99},
        {"pass_has_batches": False},
    ],
)
def test_v3_rejects_inconsistent_visit_state(tmp_path, change):
    ref = _ref(tmp_path, [_row(0, target=np.arange(400))])
    with closing(_stream([ref], shard_visit_quota=10)) as stream:
        next(stream)
        state = replace(stream.state(), **change)
        with pytest.raises(ValueError, match="visit cursor/quota"):
            stream.load_state(state)
