"""Legacy golden identities/batches and weighted training/checkpoint integration."""

from contextlib import ExitStack, closing
from dataclasses import asdict, fields, replace
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from demand_forecast.pretrain.config import CorpusConfig, PretrainConfig, pretrain_identity


def test_legacy_identity_golden():
    # Captured from main @ cadb6af using PYTHONPATH outside this checkout.
    cfg = PretrainConfig.from_yaml(
        Path(__file__).resolve().parents[2] / "configs/pretrain-v2-first-try.yaml"
    )
    shards = {
        "training_corpus/tsmixup_10m/train-00000-of-00200.parquet": "a" * 64,
        "training_corpus/kernel_synth_1m/train-00000-of-00020.parquet": "b" * 64,
    }
    assert (
        pretrain_identity(cfg, "2.0.0", shards)
        == "a3065581ee5aacbcfb4781e9119972948b17b39cd33f12c98f82a808c193b69d"
    )
    assert "sources" not in cfg.model_dump(mode="json")["corpus"]


@pytest.mark.parametrize(
    "every,expected",
    [
        (0, "6ac2ca00f3e07fef406ae1ce08cd6b6f7f0ba407139784c8d56a26f9a49a007c"),
        (1, "34048d0782259cc5fcf6f169bb1eaad4890788d310feec9e1e4644ebfb5f0e48"),
        (10, "c7992710c4c0e4e6fd0600957bc0b83efbac9019145719680c08b95a08d7f790"),
    ],
)
def test_legacy_fifty_batches_golden(tmp_path, every, expected):
    pytest.importorskip("pyarrow")
    from demand_forecast.pretrain.corpus import corpus_manifest
    from demand_forecast.pretrain.sampler import CorpusSampler
    from tests.helpers.corpus_fixture import write_corpus_fixture

    for kind in ("tsmixup", "kernel_synth"):
        for index in (0, 1):
            write_corpus_fixture(tmp_path, kind=kind, index=index, seed=index)
    refs = corpus_manifest(tmp_path)
    digest = sha256()
    with closing(
        CorpusSampler(
            **{k: [r for r in refs if r.kind == k] for k in ("tsmixup", "kernel_synth")},
            horizon=4,
            max_context=16,
            min_context=8,
            batch_size=4,
            seed=0,
            kernel_synth_every=every,
        )
    ) as sampler:
        batches = sampler.batches()
        for _ in range(50):
            batch = next(batches)
            digest.update(sampler.last_kind.encode())
            for field in fields(batch):
                value = getattr(batch, field.name)
                if field.name == "loss_weight":
                    # The legacy golden predates task-only loss weights.
                    assert torch.equal(value, torch.ones_like(value))
                    continue
                for key, tensor in (
                    sorted(value.items()) if isinstance(value, dict) else [(field.name, value)]
                ):
                    digest.update(key.encode())
                    digest.update(str(tuple(tensor.shape)).encode())
                    digest.update(tensor.numpy().tobytes())
    assert digest.hexdigest() == expected


def test_sources_validation_and_yaml_roundtrip(tmp_path):
    sources = dict(
        tsmixup=dict(shards=[0], weight=8),
        kernel_synth=dict(shards=[0], weight=1),
        chronos_real=dict(shards=[0, 1], holdout=[2], weight=2),
    )
    cfg = CorpusConfig(root=tmp_path, sources=sources)
    assert cfg.selections()["chronos_real"] == ([0, 1], [2])
    assert cfg.tsmixup_shards == cfg.kernel_synth_shards == [0]
    assert CorpusConfig.model_validate(cfg.model_dump(mode="json")) == cfg
    for kwargs in [
        dict(tsmixup_shards=[1]),
        dict(holdout_kernel_synth=[19]),
        dict(kernel_synth_every=10),
    ]:
        with pytest.raises(ValidationError, match="conflicts"):
            CorpusConfig(root=tmp_path, sources=sources, **kwargs)
    for bad in [
        dict(unknown=dict(shards=[0], weight=1)),
        dict(chronos_real=sources["chronos_real"]),
        dict(tsmixup=dict(shards=[0], holdout=[0], weight=1)),
        dict(tsmixup=dict(shards=[0], weight=0)),
        dict(tsmixup=dict(shards=[0], weight=float("inf"))),
    ]:
        with pytest.raises(ValidationError):
            CorpusConfig(root=tmp_path, sources=bad)
    assert CorpusConfig(
        root=tmp_path,
        sources=dict(tsmixup=dict(shards=[0], weight=9), kernel_synth=dict(shards=[0], weight=1)),
        kernel_synth_every=10,
    )
    assert CorpusConfig(
        root=tmp_path, sources=dict(tsmixup=dict(shards=[0], weight=1)), kernel_synth_every=0
    )
    assert CorpusConfig(
        root=tmp_path,
        sources=dict(
            tsmixup=dict(shards=[0], weight=0.3), kernel_synth=dict(shards=[0], weight=0.1)
        ),
        kernel_synth_every=4,
    )


def test_weighted_quota_order_and_resume(tiny_pretrain_cfg):
    from demand_forecast.pretrain.corpus import corpus_manifest
    from demand_forecast.pretrain.sampler import CorpusSampler, sampler_state_from_dict
    from tests.helpers.reference_batch import assert_batches_equal

    refs = corpus_manifest(tiny_pretrain_cfg.corpus.root)
    primary = [r for r in refs if r.kind == "tsmixup"]
    kinds = ["tsmixup", "kernel_synth", "chronos_real", "gifteval_pretrain"]
    sources = {kind: [replace(ref, kind=kind) for ref in primary] for kind in kinds}
    weights = dict(zip(kinds, [0.4, 0.1, 0.2, 0.3], strict=True))
    kwargs = dict(
        sources=sources,
        weights=weights,
        horizon=4,
        max_context=16,
        min_context=8,
        batch_size=4,
        seed=0,
    )
    with ExitStack() as stack:
        reference, restored = [
            stack.enter_context(closing(CorpusSampler(**kwargs))) for _ in range(2)
        ]
        batches = reference.batches()
        sequence = []
        for i in range(50):
            batch = next(batches)
            sequence.append(reference.last_kind)
            if i == 16:
                restored.load_state(sampler_state_from_dict(asdict(reference.state())))
                replay = restored.batches()
            if i > 16:
                assert_batches_equal(batch, next(replay))
                assert restored.last_kind == reference.last_kind
        assert {k: sequence.count(k) for k in kinds} == dict(
            zip(kinds, [20, 5, 10, 15], strict=True)
        )
        assert sequence[:10] == [
            "tsmixup",
            "gifteval_pretrain",
            "chronos_real",
            "tsmixup",
            "kernel_synth",
            "gifteval_pretrain",
            "tsmixup",
            "chronos_real",
            "gifteval_pretrain",
            "tsmixup",
        ]
        assert restored.state() == reference.state()
        reversed_sampler = stack.enter_context(
            closing(
                CorpusSampler(
                    **(
                        kwargs
                        | dict(
                            sources=dict(reversed(list(sources.items()))),
                            weights=dict(reversed(list(weights.items()))),
                        )
                    )
                )
            )
        )
        it = reversed_sampler.batches()
        for kind in sequence:
            next(it)
            assert reversed_sampler.last_kind == kind


def test_weighted_training_resume_and_diagnostic_selection(tiny_pretrain_cfg, tmp_path):
    from demand_forecast.core.training import InterruptedRun
    from demand_forecast.pretrain.corpus import corpus_manifest, write_manifest
    from demand_forecast.pretrain.train import _PretrainLoop, pretrain
    from tests.helpers.corpus_fixture import write_corpus_fixture

    cfg = tiny_pretrain_cfg
    root = cfg.corpus.root
    for index in (0, 1):
        source = write_corpus_fixture(tmp_path / "extra", index=index, seed=42 + index)
        dest = root / "training_corpus/chronos_real" / f"train-{index:05d}-of-00002.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    write_manifest(corpus_manifest(root), root / "manifest.json")
    raw = cfg.model_dump(mode="json")
    raw["corpus"] = dict(
        root=str(root),
        external_validation=None,
        val_pairs_per_kind=8,
        sources={
            k: dict(shards=[0], holdout=[1], weight=w)
            for k, w in [("tsmixup", 2), ("kernel_synth", 1), ("chronos_real", 1)]
        },
    )
    cfg = PretrainConfig.model_validate(raw)
    loop = _PretrainLoop(cfg)
    try:
        scores = loop.validation.evaluate(loop.model, device="cpu")
        assert scores["val_sql_chronos_real"] is not None
        assert scores["val_sql"] == pytest.approx(
            (scores["val_sql_tsmixup"] + scores["val_sql_kernel_synth"]) / 2
        )
        assert all(r.index == 0 for stream in loop.sampler.streams.values() for r in stream.refs)
    finally:
        loop.sampler.close()
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    pretrain(cfg, continuous)
    with pytest.raises(InterruptedRun):
        pretrain(cfg, resumed, stop_after=5)
    pretrain(cfg, resumed, resume=True)
    left = torch.load(continuous / "checkpoints/latest.pt", weights_only=False)
    right = torch.load(resumed / "checkpoints/latest.pt", weights_only=False)
    assert left["sampler_state"] == right["sampler_state"]
    assert left["history"] == right["history"]
    assert all(torch.equal(value, right["model"][key]) for key, value in left["model"].items())
