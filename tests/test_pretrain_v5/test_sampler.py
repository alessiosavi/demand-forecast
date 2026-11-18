"""The batch index, rather than worker scheduling, addresses the stream."""

import importlib
from copy import deepcopy
from dataclasses import fields, replace
from itertools import islice

import numpy as np
import pytest
import torch

from demand_forecast.data.episode import EpisodeBuilder, EpisodeRequest, Item, Panel
from demand_forecast.data.episode.panel_stats import PanelStats
from demand_forecast.data.episode.schema import MAX_FEATURE_ROWS, N_SUMMARY, SUMMARY_FEATURES
from demand_forecast.data.episode.screening import screen
from demand_forecast.pretrain.v5.config import SourceConfig
from demand_forecast.pretrain.v5.corpus import CorpusValueError, ShortRow
from demand_forecast.pretrain.v5.sources import SourceMix, TypedSource
from demand_forecast.synthetic.typed import PanelRecipe, generate_typed_panel
from tests.helpers.batch_fingerprint import assert_fingerprint
from tests.helpers.corpus_v4_fixture import write_rows
from tests.helpers.v5_sampler import FixedSource, config, equal_batches, fixed_mix, panel


def api():
    return importlib.import_module("demand_forecast.pretrain.v5.sampler")


def test_batch_is_pure():
    c = config()
    a, ra = api().make_batch(c, 3)
    b, rb = api().make_batch(c, 3)
    equal_batches(a, b)
    assert ra == rb and ra["episodes"] == 4 and ra["supervised"] > 0


# Captured from the default batch after verifying the original byte pin.
_BASE_BATCH_FINGERPRINT = {
    "x": {
        "shape": [4, 5, 4, 16, 39],
        "nan": 0,
        "sums": [7603.942322080315, 10569.192989855248, 2981.167578761939],
    },
    "patch_valid": {"sha256": "40d4295b1d046b32fb1512afa1a246abd391d3c228c1744d1f0f9233312117c3"},
    "row_kind": {"sha256": "c50f83c4be15cc4026d790456c4a7dc76dc33b9397674dccc9fcc8ddac364d65"},
    "row_role": {"sha256": "f3b7e1440151dd1d9bab4f205e17532c648f7c20475c1246cba93cdb71b45cad"},
    "row_names": {"sha256": "f48341a6d92d0bd4066b0bc9eb5f6ad0b1e79e69da8c28ce34645a0e1c3bd171"},
    "row_summary": {
        "shape": [4, 5, 28],
        "nan": 0,
        "sums": [24.95725216716528, 66.92101139947772, 7.296810248700371],
    },
    "row_summary_flags": {
        "sha256": "0496f87a661e46b03776acc528e75b7fe583fa3486dc97924c765324ea372667"
    },
    "row_score": {
        "shape": [4, 5],
        "nan": 0,
        "sums": [9.71929798531346, 9.71929798531346, 4.004607424407731],
    },
    "row_cand": {"sha256": "cfba54ed55e48887711a3df2102e3455733cabcb0e03b7879b8ae4ce2dc7cd4d"},
    "forecast_rows": {"sha256": "d9b405550f8ba937554a2debb8cedcfc60819e43bf46ce546fe3de5ea85480f9"},
    "keep": {"sha256": "c33098e12386fa12ccf1c095fb5c2970b6fc77d81c77049e31446fda97c48118"},
    "labels": {
        "shape": [4, 2, 1, 16],
        "nan": 0,
        "sums": [-36.73539087176323, 70.0549784898758, -21.345336115686223],
    },
    "label_mask": {"sha256": "867726bc2b4795f845b0fa02f78ceb93ec3981d917fe3bf6fd62fa1c2f0873c4"},
    "loc": {
        "shape": [4, 1],
        "nan": 0,
        "sums": [665.2580041191961, 665.2580041191961, 565.2572249599358],
    },
    "scale": {
        "shape": [4, 1],
        "nan": 0,
        "sums": [487.8132051421306, 487.8132051421306, 448.84613740597064],
    },
    "fallback": {"sha256": "db7efe99401db002249bb62c50516312e56b7603670e2ca82c65b4b70b2a7b14"},
    "known_rows": {"sha256": "32019a29d4fabbce5d372e432300c2821923c06bb3b8066be1eff088033f5583"},
    "xh": {
        "shape": [4, 2, 1, 1, 16, 39],
        "nan": 0,
        "sums": [75.53048648685217, 82.93639671802521, 35.76268566896817],
    },
    "cal_phases": {
        "shape": [4, 4, 16, 16],
        "nan": 0,
        "sums": [-967.6367303989828, 1622.4486486576498, -482.09265684501406],
    },
    "cal_ids": {"sha256": "ffa9852dda0b1fc88af13e492109a4f17a081a62d8d7b9b9c6a3b6a80f2a1117"},
    "calh_phases": {
        "shape": [4, 2, 1, 16, 16],
        "nan": 0,
        "sums": [-126.77930046990514, 443.5837578885257, -57.88633335596023],
    },
    "calh_ids": {"sha256": "a8150d68fe149b463367dd739c2148da0ab4cce13f11fe5aa093616377e372b7"},
    "cand_names": {"sha256": "8588bf453a5cf737f6517f53971c2e3fe5ae8fd438e6d53936dd2f30399355cb"},
    "cand_kind": {"sha256": "0176a3baacf9e4bee69f8e17fe9efb7c5ff0d2a2fb441fa0f5df5989949c92f4"},
    "cand_summary": {
        "shape": [4, 2, 28],
        "nan": 0,
        "sums": [17.524523993663024, 57.61646507837577, 6.026995040776196],
    },
    "cand_flags": {"sha256": "5165afe560216db57c679ea14b1c3c907809ed942e530ae8d2d49ecc11deaa81"},
    "cand_score": {
        "shape": [4, 2],
        "nan": 0,
        "sums": [5.909398555755615, 5.909398555755615, 3.2973566353321075],
    },
    "attr_stats": {
        "shape": [4, 2, 57],
        "nan": 0,
        "sums": [28.157332506971784, 68.24927359168453, 7.738828159630274],
    },
    "attr_flags": {"sha256": "372fed7dfd6774f17a890fa9260d16935780be8c889bbf9e7c7a16730ed21306"},
    "attr_cand": {"sha256": "f2b2afad083e2d21c3dde2c55099624041b97310a87d6b293e791169bf55c7ed"},
    "attr_names": {"sha256": "05269ac5c67c028eb8a73b1cfbd65eeb9ac4f0ac390d70d62761d61d376da9c6"},
    "static_num": {
        "shape": [4, 2],
        "nan": 0,
        "sums": [-0.09850254654884338, 7.898325473070145, -1.2052506729960442],
    },
    "static_num_flags": {
        "sha256": "33731f74ef5f96de17cdcd3ad9302827adfbe62f8c6f037d434467470296a709"
    },
    "static_num_cand": {
        "sha256": "bc907162f3924eb228c4f88004159039bdac4bf41dbd7f9f01300bd9002d04d0"
    },
    "freq_class": {"sha256": "4f6d44463ea6c07f86e34e7367ebaf4834bca9843f886f0a60fb4aef93a6e10f"},
    "horizon": {"sha256": "12ce5f5f1732e2a6e3571b22c5fa4c728242f10f9a9a5c683a3795352a00a374"},
    "origin_patches": {
        "sha256": "490e558305cc4cbaef890f0692e7e5112e7b041a02df2e469ac6a32d5d7701eb"
    },
    "context_target": {
        "sha256": "60ecf51984c70257e47f21dcf6a885d7812adddfadd74c117730392618baaeb3"
    },
    "row_mask": {"sha256": "5622fa40b9b470762a613cae820435b282b1ce533a502db465791f63d56be3dc"},
    "forecast_mask": {"sha256": "4a2bf737d8509d9b4f7a0351c09a53e7b3cbfea3ea6d893012b246bfa6f02de1"},
    "known_mask": {"sha256": "6cc3a4279a316b4b2e867b340db06d1bfd36293fb34d92b037edeb82883c396d"},
    "cand_mask": {"sha256": "3f0c88411489f5672257c039c0cfe718ada438254cb10b472ab4189bc00ede60"},
    "attr_mask": {"sha256": "35a862cd7dcd9fd0ddfda9c54a9aa7d81dde7a7c54992e9be3a38a7952539a3d"},
    "static_num_mask": {
        "sha256": "64301f73f7cc57f8325cc83b23ef58ab4efbf8dc9cb4d1fd27b22cc81571c749"
    },
    "origin_mask": {"sha256": "dd14965ba984fd76d9d65a48e090f399d9c254a0755c404dde0e000767d67bcd"},
    "horizon_mask": {"sha256": "8a39239b3f6cea9702fa8ec16b8b9d918adb0e6ce07bfa140259cf994028a67e"},
}


def test_default_level_reproduces_the_base_batch():
    c = replace(config(drop=()), seed=12)
    source = replace(
        c.sources[0],
        recipes=(("static_number", "known_driver"),),
        max_items=20,
        length=(128, 160),
        n_nuisance=1,
        nuisance_from=("static_number",),
    )
    batch, _ = api().make_batch(replace(c, sources=(source,)), 3)
    assert any(batch.cand_names)
    assert_fingerprint(batch, _BASE_BATCH_FINGERPRINT)


@pytest.mark.parametrize("designate_target", [False, True])
def test_level_zeroes_scores_and_keeps_every_candidate_within_budget(designate_target):
    generated = generate_typed_panel(
        PanelRecipe(
            ("static_category",),
            n_items=20,
            length=(128, 160),
            n_nuisance=1,
            nuisance_from=("static_category",),
            complete_inputs=True,
            seed=2,
        )
    )
    nuisance = generated.inputs_of("nuisance")[0].name
    base = config(drop=(), designate_target=designate_target, cutoff_at_origin=1)
    sources = fixed_mix(base, FixedSource(generated.panel))
    level = 0.75
    cfg = replace(base, sampler=replace(base.sampler, screen_no_evidence=level))
    reference, _ = api().make_batch(base, 3, sources=sources)
    screened, _ = api().make_batch(cfg, 3, sources=sources)

    candidate_index = reference.cand_names[0].index(nuisance)
    row = next(name for name in reference.row_names[0] if name.startswith(f"{nuisance}="))
    assert 0 < reference.cand_score[0, candidate_index] < level
    assert 0 < reference.row_score[0, reference.row_names[0].index(row)] < level
    assert screened.cand_names == reference.cand_names
    assert screened.cand_score[0, candidate_index] == 0
    assert screened.row_score[0, screened.row_names[0].index(row)] == 0


def test_level_puts_candidates_without_evidence_last_when_the_budget_binds(monkeypatch):
    rng = np.random.default_rng(0)
    length = 160
    target = np.sin(np.arange(length) / 5)
    known = {f"k{i:02d}": rng.normal(size=length) for i in range(20)}
    known["k19"] = target + rng.normal(scale=0.03, size=length)
    data = Panel(
        "feature-budget",
        (Item("i", 0, {"y": target}, known, {}),),
        dict.fromkeys(known, "known"),
        "D",
        None,
        True,
    )
    base = config(
        drop=(), cutoff_at_origin=1, episodes_per_batch=1, episodes_per_panel=1, origins=1
    )
    sources = fixed_mix(base, FixedSource(data))
    requests = []
    build = EpisodeBuilder.build

    def observe(builder, request):
        requests.append(request)
        return build(builder, request)

    monkeypatch.setattr(EpisodeBuilder, "build", observe)
    level = 0.75
    cfg = replace(base, sampler=replace(base.sampler, screen_no_evidence=level))
    reference, _ = api().make_batch(base, 3, sources=sources)
    screened, _ = api().make_batch(cfg, 3, sources=sources)
    request, screened_request = requests
    assert screened_request == request
    reference_names = set(reference.row_names[0]) & known.keys()
    assert reference_names == set(
        "k01 k02 k03 k04 k05 k06 k08 k10 k11 k12 k14 k15 k16 k17 k18 k19".split()
    )

    stats = PanelStats(data, "y", request.cutoff)
    scores = {
        candidate.name: float(np.float32(candidate.score)) if candidate.score >= level else 0.0
        for candidate in screen(data, stats, request.item_index, ("y",), request.cutoff)
        if candidate.removed is None
    }
    names = screened.row_names[0]
    admitted = {
        name: score.item()
        for name, score in zip(names, screened.row_score[0, : len(names)], strict=True)
        if name in known
    }
    cut = scores.keys() - admitted.keys()
    assert len(scores) == len(known) > MAX_FEATURE_ROWS == len(admitted)
    assert admitted.keys() != reference_names
    assert admitted == {name: scores[name] for name in admitted}
    assert {scores[name] for name in cut} == {0.0}
    assert min(admitted.values()) >= max(scores[name] for name in cut)
    assert max(admitted.values()) >= level


@pytest.mark.parametrize("mechanism", ["static_category", "known_driver"])
def test_path_share_counts_kept_episodes_over_twenty_batches(mechanism):
    c = config(path_share=0.5, designate_target=True, episodes_per_panel=8)
    c = replace(
        c,
        sources=(replace(c.sources[0], recipes=((mechanism,),), max_items=24, length=(128, 160)),),
    )
    paths = kept = 0
    for k in range(20):
        _, report = api().make_batch(c, k)
        assert sum(report["paths"].values()) <= 2
        paths += sum(report["paths"].values())
        kept += report["episodes"]
    assert paths / kept >= 0.45
    if mechanism == "known_driver":
        assert paths / kept <= 0.55


@pytest.mark.parametrize("share", [0.0, 0.2, 0.5, 0.9, 1.0])
def test_path_quota_is_never_exceeded(share):
    c = config(path_share=share, designate_target=True, episodes_per_batch=7, episodes_per_panel=8)
    c = replace(c, sources=(replace(c.sources[0], recipes=(("known_driver",),)),))
    for k in range(3):
        _, report = api().make_batch(c, k)
        assert report["episodes"] == 7
        assert sum(report["paths"].values()) == round(share * 7)


def test_typed_corpus_mix_fills_any_open_quota(tmp_path):
    pytest.importorskip("pyarrow")
    write_rows(tmp_path / "univariate/a/train-000.parquet", [[np.arange(64, dtype=np.float64)]])
    c = config(path_share=1.0, designate_target=True)
    c = replace(
        c,
        sources=(
            replace(c.sources[0], recipes=(("known_driver",),)),
            SourceConfig("corpus", 1, root=str(tmp_path)),
        ),
    )
    seen_corpus = False
    for k in range(10):
        _, report = api().make_batch(c, k)
        assert report["episodes"] == 4
        assert sum(report["paths"].values()) <= 4
        seen_corpus |= report["sources"].get("corpus", 0) > 0
    assert seen_corpus


@pytest.mark.parametrize("mechanisms", [(), ("known_driver",)])
def test_plan_panel_respects_open_quotas(mechanisms):
    c = config(path_share=1.0, designate_target=True, episodes_per_panel=8)
    source = replace(c.sources[0], recipes=(mechanisms,))
    p, record = TypedSource(source).panel(np.random.default_rng(2))
    _, plans, _ = api().plan_panel(
        p, record, c.sampler, source, np.random.default_rng(3), path_quota=2, ordinary_quota=0
    )
    assert len(plans) == 2
    assert all(bool(q.mechanism) == bool(mechanisms) for q in plans)
    _, ordinary, _ = api().plan_panel(
        p, record, c.sampler, source, np.random.default_rng(3), path_quota=0, ordinary_quota=1
    )
    assert len(ordinary) == 1 and ordinary[0].mechanism is None


def test_failed_path_search_cannot_fill_closed_ordinary_quota():
    g = generate_typed_panel(PanelRecipe(("static_category",), n_items=24, seed=2))
    for item in g.panel.items:
        for name in item.static:
            item.static[name] = None
    c = config(path_share=1.0, designate_target=True)
    _, plans, skips = api().plan_panel(
        g.panel,
        {"source": "typed", "generated": g},
        c.sampler,
        c.sources[0],
        np.random.default_rng(3),
        path_quota=4,
        ordinary_quota=0,
    )
    assert not plans and skips["no_support"] == 1


@pytest.mark.parametrize("reason", ["short_row", "unsupervised", "oversized"])
def test_path_quota_replenishes_after_build_skip(monkeypatch, reason):
    c = config(path_share=0.5, designate_target=True, max_cells=100_000)
    c = replace(c, sources=(replace(c.sources[0], recipes=(("known_driver",),)),))
    original = api().EpisodeBuilder.build
    calls = 0

    def skip_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            if reason == "short_row":
                raise ShortRow()
            if reason == "oversized":
                return shaped_episode(40, 160)
            episode = original(*args, **kwargs)
            episode.label_mask[:] = False
            return episode
        return original(*args, **kwargs)

    monkeypatch.setattr(api().EpisodeBuilder, "build", skip_once)
    _, report = api().make_batch(c, 0)
    assert report["episodes"] == 4 and report["paths"] == {"known_driver": 2}
    assert report["skips"][reason] == 1


@pytest.mark.parametrize("invalid", ["bare", "mapping", "object", "sequence"])
def test_make_batch_rejects_sources_other_than_source_mix(invalid):
    c = config()
    sources = {
        "bare": TypedSource(c.sources[0]),
        "mapping": {},
        "object": object(),
        "sequence": c.sources,
    }[invalid]
    with pytest.raises(TypeError, match="SourceMix"):
        api().make_batch(c, 0, sources=sources)


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_loader_resume_workers_and_default_torch_generator(workers, monkeypatch):
    c = config(episodes_per_batch=1)
    initial = torch.get_rng_state().clone()
    reference = list(islice(api().loader(c, 0), 9))
    real_loader = api().DataLoader

    def bounded_loader(*args, **kwargs):
        # A denied worker IPC operation must fail promptly, rather than hang CI.
        return real_loader(*args, timeout=10 if workers else 0, **kwargs)

    monkeypatch.setattr(api(), "DataLoader", bounded_loader)
    actual = list(islice(api().loader(replace(c, workers=workers), 3), 6))
    for (k, a, ra), (j, b, rb) in zip(reference[3:], actual, strict=True):
        assert k == j and ra == rb
        equal_batches(a, b)
    assert torch.equal(initial, torch.get_rng_state())


def test_requests_share_patch_anchor_and_fit_context():
    p = panel()
    c = config().sampler
    for seed in range(10):
        _, plans, _ = api().plan_panel(p, {}, c, config().sources[0], np.random.default_rng(seed))
        for plan in plans:
            r = plan.request
            assert r.origins == tuple(sorted(r.origins)) and len(r.origins) <= 2
            assert all((r.origins[-1] - o) % 16 == 0 for o in r.origins)
            assert r.origins[0] >= r.origins[-1] - r.context + 16
            assert r.cutoff <= r.origins[0]
            EpisodeBuilder(p).build(r)


def test_short_items_use_own_cutoff_long_items_share_panel_cutoff():
    p = panel(512, count=3)
    short = replace(p.items[2], start=220, targets={"y0": np.arange(40, dtype=np.float64)})
    p = replace(p, items=(*p.items[:2], short))
    c = config(contexts=(256,), origins=4, cutoff_at_origin=0, designate_target=True).sampler
    seen = False
    for seed in range(100):
        _, plans, _ = api().plan_panel(p, {}, c, config().sources[0], np.random.default_rng(seed))
        long_cutoffs = [q.request.cutoff for q in plans if q.item_id != short.item_id]
        assert len(set(long_cutoffs)) <= 1
        for q in plans:
            if q.item_id == short.item_id:
                seen = True
                assert q.request.cutoff == q.request.origins[0]
                assert not EpisodeBuilder(p).build(q.request).fallback.any()
    assert seen


@pytest.mark.parametrize(
    "mechanisms",
    [(), ("calendar", "intermittent", "level_shift"), ("planned_event",), ("static_category",)],
)
def test_empty_structure_and_path_panels(mechanisms):
    c = config(path_share=1, designate_target=True)
    c = replace(
        c, sources=(replace(c.sources[0], recipes=(mechanisms,), max_items=24, length=(128, 160)),)
    )
    built, report = api().make_batch(c, 0)
    assert report["episodes"] and built.label_mask.any()
    if mechanisms in (("planned_event",), ("static_category",)):
        assert report["paths"].get(mechanisms[0], 0) > 0
    p, record = TypedSource(c.sources[0]).panel(np.random.default_rng(2))
    _, plans, _ = api().plan_panel(p, record, c.sampler, c.sources[0], np.random.default_rng(3))
    for q in plans:
        if q.mechanism == "planned_event":
            item = p.items[q.request.item_index]
            col = record["generated"].inputs_of(q.mechanism)[0].name
            left = q.request.origins[-1] - item.start
            assert (item.dynamic[col][left : left + q.request.horizon] == 1).any()
        if q.mechanism == "static_category":
            assert q.item_id in record["generated"].short_items


def test_protection_consumes_identical_drop_draws(monkeypatch):
    c = config(
        path_share=1,
        designate_target=True,
        drop=(("features", 1.0), ("zeros", 0.7), ("value_stats", 1.0)),
    )
    source = replace(c.sources[0], recipes=(("known_driver",),), length=(128, 160))
    p, record = TypedSource(source).panel(np.random.default_rng(1))
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    _, protected, _ = api().plan_panel(p, record, c.sampler, source, a)
    monkeypatch.setitem(api().PROTECT, "known_driver", ())
    _, unprotected, _ = api().plan_panel(p, record, c.sampler, source, b)
    assert a.bit_generator.state == b.bit_generator.state
    assert any(q.mechanism for q in protected)
    assert all("features" not in q.request.drop for q in protected if q.mechanism)
    assert all("features" in q.request.drop for q in unprotected)


def test_no_support_and_tiny_necessary_items_are_safe():
    g = generate_typed_panel(PanelRecipe(("static_category",), n_items=48, seed=1))
    assert any(len(i.targets["target"]) < 17 for i in g.panel.items if i.item_id in g.short_items)
    c = config(path_share=1, designate_target=True)
    record = {"source": "typed", "generated": g, "recipe": g.recipe}
    _, plans, _ = api().plan_panel(
        g.panel, record, c.sampler, c.sources[0], np.random.default_rng(0)
    )
    assert plans
    hidden = deepcopy(g)
    for item in hidden.panel.items:
        for name in item.static:
            item.static[name] = None
    _, plans, skips = api().plan_panel(
        hidden.panel,
        {**record, "generated": hidden},
        c.sampler,
        c.sources[0],
        np.random.default_rng(0),
        path_quota=2,
        ordinary_quota=2,
    )
    assert skips["no_support"] == 1 and len(plans) == 2
    assert all(q.mechanism is None for q in plans)


def test_no_labels_skipped_before_build():
    p = panel(length=17, count=1)
    p.items[0].targets["y0"][16:] = np.nan
    _, plans, skips = api().plan_panel(
        p, {}, config().sampler, config().sources[0], np.random.default_rng(1)
    )
    assert not plans and skips["no_labels"] == 1


def test_fallback_checked_on_uncontrolled_panel():
    p = panel(length=17, count=1)
    p.items[0].targets["y0"][:16] = 1
    p.items[0].targets["y0"][0] = np.nan
    source = replace(config().sources[0], control="missing_as_zero")
    _, plans, skips = api().plan_panel(
        p, {}, config(origins=1).sampler, source, np.random.default_rng(1)
    )
    assert not plans and skips["fallback"] == 1


def test_missing_as_zero_admission_uses_uncontrolled_record(monkeypatch):
    p = panel(length=17, count=4)
    complete = deepcopy(p)
    p.items[0].targets["y0"][:16] = np.nan
    p.items[0].targets["y0"][0] = 7.0
    filled = deepcopy(p)
    filled.items[0].targets["y0"][1:16] = 0.0
    # At the only possible cutoff (16), one observation becomes {7, 0}.
    assert np.isfinite(p.items[0].targets["y0"][:16]).sum() == 1
    assert np.unique(filled.items[0].targets["y0"][:16]).tolist() == [0.0, 7.0]
    generated = replace(
        generate_typed_panel(PanelRecipe(("calendar",), n_items=4, length=(17, 17))),
        panel=p,
        complete=complete,
    )
    record = {"source": "typed", "generated": generated}
    c = config(
        designate_target=True,
        origins=1,
        cutoff_at_origin=1,
        path_share=0.5,
        drop=(),
        drop_always=("panel_row",),
    )
    built = []
    real_build = api().EpisodeBuilder.build

    def track_build(builder, request):
        episode = real_build(builder, request)
        built.append((builder.panel.items[request.item_index].item_id, request))
        return episode

    monkeypatch.setattr(api().EpisodeBuilder, "build", track_build)
    results = []
    for control, exposed in ((None, p), ("missing_as_zero", filled)):
        arm = replace(c, sources=(replace(c.sources[0], control=control),))
        # The exposed control differs from the original in record["generated"].
        _, plans, skips = api().plan_panel(
            exposed, record, arm.sampler, arm.sources[0], np.random.default_rng(1)
        )
        assert skips == {"fallback": 1}
        assert {plan.item_id for plan in plans} == {"1", "2", "3"}
        built.clear()
        batch, report = api().make_batch(
            arm, 0, sources=fixed_mix(arm, FixedSource(exposed, record))
        )
        assert report["episodes"] == len(built) == 4
        assert report["skips"]["fallback"] >= 1
        assert report["paths"] == {"calendar": 2}
        assert {item_id for item_id, _ in built[:3]} == {"1", "2", "3"}
        assert all(item_id != "0" for item_id, _ in built)
        results.append((plans, skips, tuple(built), batch, report))
    true, null = results
    assert true[:3] == null[:3]  # Kept items and requests include later panel draws.
    assert true[4] == null[4]  # Includes skips and paths after filling the batch.
    equal_batches(true[3], null[3])


def test_sampled_recipes_never_keep_unsupervised():
    # Real sampled recipes, with a small item cap; batches remain tiny.
    c = replace(config(episodes_per_batch=1), sources=(SourceConfig("typed", 1, max_items=2),))
    for k in range(50):
        _, report = api().make_batch(c, k)
        assert report["skips"].get("unsupervised", 0) == 0


def test_designated_target_siblings_drop_keeps_label_cells():
    c = config(designate_target=True, drop=())
    c = replace(c, sources=(replace(c.sources[0], recipes=(("multi_target",),)),))
    a, ra = api().make_batch(c, 0)
    b, rb = api().make_batch(replace(c, sampler=replace(c.sampler, drop_always=("siblings",))), 0)
    assert a.context_target == b.context_target
    assert torch.equal(a.origin_patches, b.origin_patches)
    for batch in (a, b):
        assert not batch.label_mask.masked_select(~batch.keep[:, None, :, None]).any()
    assert torch.equal(a.labels[:, :, :1], b.labels[:, :, :1])
    assert torch.equal(a.label_mask[:, :, :1], b.label_mask[:, :, :1])
    assert ra["supervised"] == rb["supervised"]


def test_shuffle_admission_requests_labels_and_masks_equal():
    c = config(designate_target=True, path_share=1, drop=())
    c = replace(
        c,
        sources=(
            replace(c.sources[0], recipes=(("event_response",),), max_items=24, length=(128, 160)),
        ),
    )
    null = replace(c, sources=(replace(c.sources[0], control="shuffle"),))
    a, ra = api().make_batch(c, 3)
    b, rb = api().make_batch(null, 3)
    for name in ("labels", "label_mask", "origin_patches", "keep", "loc", "scale"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert a.context_target == b.context_target
    assert ra["skips"] == rb["skips"] and ra["paths"] == rb["paths"]
    p, record = TypedSource(c.sources[0]).panel(np.random.default_rng(2))
    pn, rn = TypedSource(null.sources[0]).panel(np.random.default_rng(2))
    assert (
        api().plan_panel(p, record, c.sampler, c.sources[0], np.random.default_rng(5))[1:]
        == (api().plan_panel(pn, rn, c.sampler, null.sources[0], np.random.default_rng(5))[1:])
    )


@pytest.mark.parametrize(
    "control,mechanism",
    [
        ("shuffle", "event_response"),
        ("no_dates", "calendar"),
        ("missing_as_zero", "missing_vs_zero"),
        ("drop_always", "seasonal_lag"),
    ],
)
def test_control_kept_fallback_and_supervised_cells_equal_over_ten_batches(control, mechanism):
    c = config(path_share=0.5, designate_target=True, origins=1, drop=())
    c = replace(
        c,
        sources=(replace(c.sources[0], recipes=((mechanism,),), max_items=24, length=(128, 160)),),
    )
    if control == "drop_always":
        null = replace(
            c,
            sampler=replace(
                c.sampler,
                drop_always=("rolling", "lags", "zeros"),
                summary_off=("zero_share", "missing_share", "log_length"),
            ),
        )
    else:
        null = replace(c, sources=(replace(c.sources[0], control=control),))
    for k in range(10):
        true, true_report = api().make_batch(c, k)
        other, other_report = api().make_batch(null, k)
        assert true.context_target == other.context_target
        assert torch.equal(true.origin_patches, other.origin_patches)
        assert torch.equal(true.keep, other.keep)
        kept = true.keep & true.forecast_mask
        assert kept.any()
        assert torch.equal(true.fallback[kept], other.fallback[kept])
        counts = []
        for batch in (true, other):
            steps = torch.arange(batch.labels.shape[-1])[None, :] < batch.horizon[:, None]
            valid = batch.keep & batch.forecast_mask & ~batch.fallback
            mask = batch.label_mask & valid[:, None, :, None] & steps[:, None, None, :]
            counts.append(mask.sum(dim=(1, 2, 3)))
        assert torch.equal(*counts)
        assert int(counts[0].sum()) == true_report["supervised"] == other_report["supervised"]
        assert true_report["paths"] == other_report["paths"]
        assert true_report["skips"] == other_report["skips"]


def test_missing_as_zero_equals_prefilled_panel_and_preserves_true_units():
    p = panel(length=17, count=2)
    for item in p.items:
        item.targets["y0"][3:5] = np.nan
    c = config(designate_target=True, origins=1, drop=(), cutoff_at_origin=1)
    true, _ = api().make_batch(c, 0, sources=fixed_mix(c, FixedSource(p)))
    null_config = replace(c, sources=(replace(c.sources[0], control="missing_as_zero"),))
    null, _ = api().make_batch(null_config, 0, sources=fixed_mix(null_config, FixedSource(p)))
    filled = deepcopy(p)
    for item in filled.items:
        item.targets["y0"][3:5] = 0
    expected, _ = api().make_batch(c, 0, sources=fixed_mix(c, FixedSource(filled)))
    equal_batches(null, expected)
    assert torch.equal(null.label_mask, true.label_mask)
    original = true.labels * true.scale[:, None, :, None] + true.loc[:, None, :, None]
    restored = null.labels * null.scale[:, None, :, None] + null.loc[:, None, :, None]
    torch.testing.assert_close(restored[true.label_mask], original[true.label_mask])
    assert np.isnan(p.items[0].targets["y0"][3])


def test_missing_as_zero_fills_corpus_history(tmp_path):
    """The control belongs to the sampler, so it reaches a corpus panel too."""
    pytest.importorskip("pyarrow")
    values = np.arange(64, dtype=np.float64)
    values[3:5] = np.nan
    write_rows(tmp_path / "univariate/a/train-000.parquet", [[values]])
    source = SourceConfig("corpus", 1, root=str(tmp_path), control="missing_as_zero")
    base = config(designate_target=True, origins=1, drop=(), cutoff_at_origin=1)
    c = replace(base, sources=(source,))
    p, record = SourceMix(c.sources).panel(np.random.default_rng(0))
    target = record["target"]
    assert np.isnan(p.items[0].targets[target][3:5]).all()
    planned = api().plan_panel(p, record, c.sampler, source, np.random.default_rng(0))
    exposed, plans = planned[0], planned[1]
    assert plans
    assert (exposed.items[0].targets[target][3:5] == 0.0).all()
    assert np.isnan(p.items[0].targets[target][3:5]).all()  # The source's row is untouched.
    true = replace(c, sources=(replace(source, control=None),))
    uncontrolled = api().plan_panel(
        p, record, true.sampler, true.sources[0], np.random.default_rng(0)
    )
    assert np.isnan(uncontrolled[0].items[0].targets[target][3:5]).all()
    assert [plan.request for plan in plans] == [plan.request for plan in uncontrolled[1]]
    _, report = api().make_batch(c, 0)
    assert set(report["sources"]) == {"corpus"}


def test_summary_off_changes_only_named_entries():
    c = config(designate_target=True, drop=())
    c = replace(c, sources=(replace(c.sources[0], recipes=(("static_category",),), max_items=12),))
    true, _ = api().make_batch(c, 0)
    null, _ = api().make_batch(
        replace(c, sampler=replace(c.sampler, summary_off=("zero_share",))), 0
    )
    j = SUMMARY_FEATURES.index("zero_share")
    for name in ("row_summary", "row_summary_flags", "cand_summary", "cand_flags", "attr_stats"):
        expected = getattr(true, name).clone()
        expected[..., j] = 0
        if name == "attr_stats":
            expected[..., j + N_SUMMARY] = 0
        assert torch.equal(expected, getattr(null, name)), name
    for f in fields(true):
        if f.name not in (
            "row_summary",
            "row_summary_flags",
            "cand_summary",
            "cand_flags",
            "attr_stats",
        ):
            a, b = getattr(true, f.name), getattr(null, f.name)
            assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b


def test_corpus_designated_target_siblings_and_bad_values(tmp_path):
    pytest.importorskip("pyarrow")
    targets = [np.arange(64, dtype=np.float64) + j for j in range(3)]
    bad = np.arange(64, dtype=np.float64)
    bad[3] = np.inf
    write_rows(tmp_path / "univariate/a/train-000.parquet", [[bad], targets])
    c = replace(config(drop=()), sources=(SourceConfig("corpus", 1, root=str(tmp_path)),))
    seen_bad = False
    for k in range(5):
        b, report = api().make_batch(c, k)
        seen_bad |= report["skips"].get("bad_values", 0) > 0
        assert report["sources"] == {"corpus": 4}
        assert all(t in names for t, names in zip(b.context_target, b.row_names, strict=True))
        assert b.label_mask[:, :, 1:].any()
        p, record = SourceMix(c.sources).panel(np.random.default_rng(0))
        _, plans, _ = api().plan_panel(p, record, c.sampler, c.sources[0], np.random.default_rng(0))
        assert len(plans) == 1 and plans[0].request.target == record["target"]
    assert seen_bad


def shaped_episode(rows, patches, known=0):
    e = EpisodeBuilder(panel(length=64, count=1)).build(EpisodeRequest(0, (32,), 16, 16))
    e.x = np.zeros((rows, patches, 16, 39), np.float32)
    e.patch_valid = np.ones((rows, patches), bool)
    e.known_rows = np.zeros(known, np.int16)
    e.xh = np.zeros((1, known, 1, 16, 39), np.float32)
    return e


def test_padded_budget_includes_cross_shape_and_known_decoder_rows(monkeypatch):
    c = config(episodes_per_batch=3, episodes_per_panel=3, max_cells=90_000)
    assert api()._cells([shaped_episode(1, 1, known=16)]) == 16 + 17 * 16
    episodes = iter([shaped_episode(40, 16), shaped_episode(1, 128)])
    monkeypatch.setattr(api().EpisodeBuilder, "build", lambda self, r: next(episodes))
    _, report = api().make_batch(c, 0, sources=fixed_mix(c, FixedSource(panel())))
    assert report["episodes"] == 1 and report["skips"]["budget"] == 1
    assert report["cells"] == 40 * 16 * 16 + 16


def test_oversized_skipped_then_stream_continues(monkeypatch):
    c = config(episodes_per_batch=1, max_cells=1000)
    episodes = iter([shaped_episode(40, 16), shaped_episode(1, 1)])
    monkeypatch.setattr(api().EpisodeBuilder, "build", lambda self, r: next(episodes))
    _, report = api().make_batch(c, 0, sources=fixed_mix(c, FixedSource(panel())))
    assert report["skips"]["oversized"] == 1 and report["episodes"] == 1


@pytest.mark.parametrize("stage", ["draw", "plan", "build"])
@pytest.mark.parametrize("error", [KeyError("injected"), ValueError("injected")])
def test_unexpected_exceptions_name_batch(monkeypatch, stage, error):
    fixed = FixedSource(panel())

    def fail(*args, **kwargs):
        raise error

    if stage == "draw":
        monkeypatch.setattr(fixed, "panel", fail)
    elif stage == "plan":
        monkeypatch.setattr(api(), "_plan_panel", fail)
    else:
        monkeypatch.setattr(api().EpisodeBuilder, "build", fail)
    with pytest.raises(RuntimeError, match="batch 17.*injected"):
        api().make_batch(config(), 17, sources=fixed_mix(config(), fixed))


def test_all_short_panels_fail_with_index_and_count():
    with pytest.raises(RuntimeError, match="batch 19.*no_room.*16"):
        api().make_batch(config(), 19, sources=fixed_mix(config(), FixedSource(panel(16))))


@pytest.mark.parametrize(
    "error,reason", [(ShortRow(), "short_row"), (CorpusValueError(), "bad_values")]
)
def test_only_sampler_skip_is_counted(monkeypatch, error, reason):
    fixed = FixedSource(panel())
    original = fixed.panel
    calls = 0

    def draw(rng):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        return original(rng)

    monkeypatch.setattr(fixed, "panel", draw)
    _, report = api().make_batch(config(), 0, sources=fixed_mix(config(), fixed))
    assert report["skips"][reason] == 1


def test_unlabelled_episode_is_never_built(monkeypatch):
    p = panel(length=17, count=1)
    p.items[0].targets["y0"][16] = np.nan

    def forbidden(*args):
        pytest.fail("an unlabelled request was built")

    monkeypatch.setattr(api().EpisodeBuilder, "build", forbidden)
    with pytest.raises(RuntimeError, match="no_labels.*16"):
        api().make_batch(config(), 0, sources=fixed_mix(config(), FixedSource(p)))


@pytest.mark.parametrize("stage", ["plan", "build"])
def test_sampler_skip_at_planning_and_building_is_counted(monkeypatch, stage):
    name = "_plan_panel" if stage == "plan" else "build"
    owner = api() if stage == "plan" else api().EpisodeBuilder
    original = getattr(owner, name)
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ShortRow()
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, fail_once)
    _, report = api().make_batch(config(), 0, sources=fixed_mix(config(), FixedSource(panel())))
    assert report["skips"]["short_row"] == 1 and report["episodes"] == 4
