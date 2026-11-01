import copy
import json
import time
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest
import torch

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.core.validation import plan_inner_validation
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import calendar_feature_names, calendar_features, grid_range
from demand_forecast.data.panel import Panel, Roles, build_series
from demand_forecast.data.sampler import Batch, WindowSampler, build_origins, make_batch
from demand_forecast.models.base import ModelSpec
from demand_forecast.models.losses import pinball_loss
from tests.helpers.models import build_v2
from tests.helpers.panels import synthetic_panel, weekly_panel
from tests.helpers.reference_sampler import assert_batch_matches_reference


def test_ragged_histories_never_produce_labels_past_their_own_end():
    lengths = (40, 38, 40, 28, 38, 40)
    timestamps = pd.date_range("2023-01-01", periods=41, freq="W-SUN")
    roles = Roles("target", (), (), (), ())
    panel = Panel(
        [
            build_series(
                str(i),
                pd.DataFrame(
                    {"target": np.arange(length, dtype=float) + 1}, index=timestamps[:length]
                ),
                freq="W-SUN",
                roles=roles,
                missing_target="nan",
                statics={},
            )
            for i, length in enumerate(lengths)
        ],
        roles,
        "W-SUN",
        1,
        "nonnegative",
    )
    plan = plan_inner_validation(panel, timestamps[40], 4, min_train_history=12)
    assert plan.val_items.tolist() == [0, 1, 2, 4, 5]
    assert plan.train_upto == timestamps[35]
    encoder = StaticEncoder().fit(panel, plan.fit_ids)
    origins = build_origins(panel, 4, 1, upto=plan.train_upto)
    assert set(origins[:, 0]) == set(range(6))
    for series_index, origin_pos in origins:
        assert origin_pos <= panel.series[series_index].history_length
    batch = make_batch(
        panel, [tuple(x) for x in origins], 4, 32, encoder, label_upto=plan.train_upto
    )
    for row, (series_index, origin_pos) in enumerate(origins):
        series = panel.series[series_index]
        grid = grid_range(series.start, int(origin_pos) + 4, panel.freq)
        expected = (np.arange(origin_pos, origin_pos + 4) < series.history_length) & (
            grid[origin_pos:] <= plan.train_upto
        )
        np.testing.assert_array_equal(batch.label_mask[row].numpy(), expected)
    early_last = np.flatnonzero((origins[:, 0] == 3) & (origins[:, 1] == 27)).item()
    assert batch.label_mask[early_last].tolist() == [True, False, False, False]


@pytest.mark.parametrize("horizon", [8, 13, 39, 52])
def test_batch_shapes_with_short_contexts(panel29, encoder29, horizon):
    origins = build_origins(panel29, horizon, 1, require_label=False)
    batch = make_batch(panel29, origins[-2:], horizon, 512, encoder29)
    assert batch.size == 2 and batch.target.shape == (2, 29)
    assert batch.dec_cov.shape == (2, horizon, 5)
    assert batch.enc_cov.shape == (2, 29, 6)
    assert batch.labels.shape == batch.label_mask.shape == (2, horizon)
    for name in ("target", "enc_cov", "dec_cov", "num", "labels", "loc", "scale"):
        assert getattr(batch, name).dtype == torch.float32
        assert torch.isfinite(getattr(batch, name)).all()
    for name in (
        "observed",
        "padding",
        "enc_cov_mask",
        "dec_cov_mask",
        "num_mask",
        "label_mask",
        "eligible",
        "has_context",
        "loss_eligible",
    ):
        assert getattr(batch, name).dtype == torch.bool
    assert batch.series_index.dtype == batch.origin_pos.dtype == torch.int64
    assert all(value.dtype == torch.int64 for value in batch.cat.values())


def test_known_future_nan_is_masked_and_finite(panel_with_future_nan, encoder_future_nan):
    batch = make_batch(panel_with_future_nan, [(0, 20)], 6, 512, encoder_future_nan)
    assert torch.isfinite(batch.dec_cov).all()
    assert batch.dec_cov[0, :, 4].tolist() == [1, 0, 0, 1, 0, 0]
    assert batch.dec_cov_mask[0, :, 4].tolist() == [True, True, False, True, False, False]
    assert batch.dec_cov_mask[0, :, :4].all()
    assert batch.eligible[0] and not batch.loss_eligible[0]
    assert not batch.label_mask.any()


def test_batch_float_dtype_is_independent_of_torch_default(panel8, encoder8):
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        batch = make_batch(panel8, [(0, 4)], 4, 8, encoder8)
    finally:
        torch.set_default_dtype(original)
    for name in ("target", "enc_cov", "dec_cov", "num", "labels", "loc", "scale"):
        assert getattr(batch, name).dtype == torch.float32


def test_origins_allow_partial_horizons_up_to_upto_and_labels_are_masked(panel8, encoder8):
    upto = panel8.timestamps(0)[7]
    origins = build_origins(panel8, 8, 1, upto=upto)
    assert len(origins) > 0 and origins[:, 1].max() == 7
    batch = make_batch(panel8, [(0, 7)], 8, 512, encoder8, label_upto=upto)
    assert batch.label_mask[0].tolist() == [True] + [False] * 7


def test_eligibility_flags_split(panel_constant, encoder_constant, panel29, encoder29):
    batch = make_batch(panel_constant, [(0, 10)], 4, 512, encoder_constant)
    assert batch.has_context[0] and not batch.eligible[0] and not batch.loss_eligible[0]
    assert batch.scale[0] == 1 and batch.loc[0] == 6
    panel29.series[0].target[4:8] = np.nan
    batch = make_batch(panel29, [(0, 8), (1, 1), (2, 29)], 4, 4, encoder29)
    assert batch.has_context.tolist() == [False, True, True]
    assert batch.eligible.tolist() == [True, False, True]
    assert batch.loss_eligible.tolist() == [True, False, False]
    assert torch.isfinite(batch.target).all()


def test_target_frame_uses_cropped_median_and_whole_past_scale(panel8, encoder8):
    panel8.series[0].target[:] = [1, 21, np.nan, 7, 9, np.nan, 15, 17]
    batch = make_batch(panel8, [(0, 6), (1, 1)], 4, 3, encoder8)
    assert batch.loc[0] == 8 and batch.scale[0] == 11
    torch.testing.assert_close(batch.target[0], torch.tensor([-1 / 11, 1 / 11, 0]))
    torch.testing.assert_close(batch.labels[0], torch.tensor([7 / 11, 9 / 11, 0, 0]))
    assert batch.padding.tolist() == [[True, True, True], [False, False, True]]
    assert batch.observed.tolist() == [[True, True, False], [False, False, True]]
    assert batch.label_mask[0].tolist() == [True, True, False, False]
    assert not batch.enc_cov_mask[1, :2].any() and not batch.enc_cov[1, :2].any()
    assert not batch.target[1, :2].any()


def test_seasonal_scale_uses_panel_period(panel8, encoder8):
    panel8.seasonal_period = 2
    panel8.series[0].target[:] = [1, 20, 3, 24, 7, 30, 9, 36]
    batch = make_batch(panel8, [(0, 6)], 2, 2, encoder8)
    assert batch.loc.item() == 18.5 and batch.scale.item() == 4


def test_label_cutoff_and_historical_span_hide_even_supplied_targets(panel8, encoder8):
    panel8.series[0].target = np.arange(12, dtype=float)
    batch = make_batch(panel8, [(0, 6), (0, 8)], 5, 4, encoder8, label_upto=panel8.timestamps(0)[6])
    assert batch.label_mask.tolist() == [[True, False, False, False, False], [False] * 5]
    assert not batch.labels[~batch.label_mask].any()
    uncut = make_batch(panel8, [(0, 6)], 5, 4, encoder8)
    assert uncut.label_mask[0].tolist() == [True, True, False, False, False]


def test_indicator_columns_untouched_and_others_standardised(panel29, encoder29):
    batch = make_batch(panel29, [(0, 20)], 4, 512, encoder29)
    assert set(batch.enc_cov[0, :, 4].tolist()) <= {0.0, 1.0}
    assert abs(batch.enc_cov[0, batch.padding[0], 5].mean().item()) < 1e-5


def test_dynamic_statistics_ignore_future_and_missing_values_and_use_whole_past(panel8, encoder8):
    series = panel8.series[0]
    series.dynamic["promo"] = np.array([2, np.nan, 4, 6, 8, 10, 1000, np.nan])
    series.dynamic["oil"] = np.array([5, 5, np.nan, 5, 5, 5, -999, -999])
    batch = make_batch(panel8, [(0, 6)], 3, 2, encoder8)
    torch.testing.assert_close(batch.enc_cov[0, :, 4], torch.tensor([2, 4]) / np.sqrt(8))
    torch.testing.assert_close(
        batch.dec_cov[0, :, 4], torch.tensor([994 / np.sqrt(8), 0, 0], dtype=torch.float32)
    )
    assert batch.dec_cov_mask[0, :, 4].tolist() == [True, False, False]
    assert not batch.enc_cov[0, :, 5].any()
    series.target[6:] = 999999
    series.dynamic["oil"][6:] = 999999
    changed = make_batch(panel8, [(0, 6)], 3, 2, encoder8)
    for name in ("target", "enc_cov", "dec_cov", "loc", "scale"):
        torch.testing.assert_close(getattr(batch, name), getattr(changed, name))


@pytest.mark.parametrize(
    "past,expected", [([0, 1, np.nan, 0], 2), ([7, 7, np.nan, 7], -5), ([np.nan] * 4, 0)]
)
def test_binary_constant_and_unobserved_dynamic_history(panel8, encoder8, past, expected):
    panel8.series[0].dynamic["promo"] = np.array([*past, 2, np.nan, 0, 0])
    batch = make_batch(panel8, [(0, 4)], 2, 4, encoder8)
    assert batch.dec_cov[0, 0, 4].item() == expected
    assert batch.dec_cov_mask[0, 0, 4].item() == bool(np.isfinite(past).any())
    assert not batch.enc_cov_mask[0, 2, 4] and batch.enc_cov[0, 2, 4] == 0


def test_absent_covariates_and_static_encoder_outputs(panel29, encoder29):
    panel29.series[0].dynamic = {}
    panel29.series[0].statics = {}
    batch = make_batch(panel29, [(0, 5), (1, 5)], 4, 3, encoder29)
    assert not batch.enc_cov_mask[0, :, 4:].any() and not batch.dec_cov_mask[0, :, 4:].any()
    for row, series in enumerate(panel29.series[:2]):
        codes, num, mask = encoder29.transform(series)
        assert {name: values[row].item() for name, values in batch.cat.items()} == codes
        np.testing.assert_array_equal(batch.num[row], num)
        np.testing.assert_array_equal(batch.num_mask[row], mask)


@pytest.mark.parametrize(
    "freq,start",
    [("W-WED", "2023-01-04"), ("MS", "2023-01-01"), ("D", "2023-03-23"), ("h", "2023-03-26")],
)
def test_calendar_features_follow_series_grid_across_origins(freq, start):
    panel = synthetic_panel(1, 8, freq, seed=0)
    panel.series[0].start = pd.Timestamp(start, tz="Europe/Rome")
    encoder = StaticEncoder().fit(panel, ["0"])
    batch = make_batch(panel, [(0, 8)], 4, 3, encoder)
    expected = calendar_features(panel.timestamps(0, length=12), freq)
    np.testing.assert_allclose(batch.enc_cov[0], expected[5:8])
    np.testing.assert_allclose(batch.dec_cov[0], expected[8:12])
    assert batch.enc_cov_mask.all() and batch.dec_cov_mask.all()
    assert batch.num.shape == batch.num_mask.shape == (1, 0)


def test_empty_context_empty_batch_and_clone_to(panel29, encoder29):
    batch = make_batch(panel29, [(0, 0)], 4, 16, encoder29)
    assert batch.target.shape == (1, 0) and not batch.has_context.any()
    assert not batch.eligible.any() and batch.label_mask.all()
    empty = make_batch(panel29, [], 4, 16, encoder29)
    assert empty.size == 0 and empty.target.shape == (0, 0)
    assert empty.dec_cov.shape == (0, 4, 5)
    batch = make_batch(panel29, [(0, 5)], 4, 16, encoder29)
    cloned = batch.clone()
    for field in fields(Batch):
        original, duplicate = getattr(batch, field.name), getattr(cloned, field.name)
        if isinstance(original, dict):
            for key in original:
                torch.testing.assert_close(original[key], duplicate[key])
                assert original[key].data_ptr() != duplicate[key].data_ptr()
        else:
            torch.testing.assert_close(original, duplicate)
            assert original.data_ptr() != duplicate.data_ptr()
    moved = batch.to("meta")
    for field in fields(Batch):
        value = getattr(moved, field.name)
        tensors = value.values() if isinstance(value, dict) else [value]
        assert all(tensor.device.type == "meta" for tensor in tensors)
    assert batch.target.device.type == "cpu"


@pytest.mark.parametrize("require_label", [False, True])
@pytest.mark.parametrize("cutoff", [None, -1, 0, 5, 14])
def test_origins_match_brute_force_on_ragged_missing_panel(require_label, cutoff):
    panel = weekly_panel([0, 1, 7, 13])
    rng = np.random.default_rng(42)
    for i, series in enumerate(panel.series):
        series.start += pd.Timedelta(weeks=i)
        series.target[rng.random(series.length) < 0.4] = np.nan
    upto = None if cutoff is None else pd.Timestamp("2023-01-01") + pd.Timedelta(weeks=cutoff)
    for horizon in (1, 4, 20):
        for min_context in (1, 3):
            expected = []
            for i, series in enumerate(panel.series):
                dates = panel.timestamps(i, length=series.length + 1)
                for origin in range(series.length + 1):
                    if upto is not None and dates[origin] > upto:
                        continue
                    if np.count_nonzero(~np.isnan(series.target[:origin])) < min_context:
                        continue
                    stop = min(origin + horizon, series.length)
                    valid = ~np.isnan(series.target[origin:stop])
                    if upto is not None:
                        valid &= dates[origin:stop] <= upto
                    if not require_label or valid.any():
                        expected.append((i, origin))
            actual = build_origins(panel, horizon, min_context, upto, require_label)
            np.testing.assert_array_equal(
                actual, np.asarray(expected, dtype=np.int32).reshape(-1, 2)
            )
            assert actual.dtype == np.int32


def test_origins_respect_history_length_and_cutoff_between_grid_steps(panel8):
    panel8.series[0].target = np.arange(20, dtype=float)
    upto = panel8.timestamps(0)[5] + pd.Timedelta(days=2)
    origins = build_origins(panel8, 4, 1, upto)
    assert origins[:, 1].max() == 5
    assert build_origins(panel8, 4, 1)[:, 1].max() == 7
    assert build_origins(panel8, 4, 1, require_label=False)[:, 1].max() == 8


def test_sampler_is_uniform_over_series_and_resumes_mid_epoch(panel_long, encoder_long):
    origins = build_origins(panel_long, 4, 1)
    sampler = WindowSampler(panel_long, origins, 4, 32, encoder_long, seed=0)
    assert len(sampler) == 20
    assert sorted(i for i, _ in sampler) == list(range(20))
    assert list(sampler) == []
    sampler.set_epoch(3)
    iterator = iter(sampler)
    head = [next(iterator) for _ in range(7)]
    state = json.loads(json.dumps(sampler.state()))
    tail = list(iterator)
    restored = WindowSampler(panel_long, origins, 4, 32, encoder_long, seed=0)
    restored.load_state(state)
    assert list(restored) == tail and len(head) + len(tail) == 20
    assert state["epoch"] == 3 and state["position"] == 7
    restored.load_state(sampler.state())
    assert list(restored) == []
    sampler.set_epoch(3)
    assert list(sampler) == head + tail
    rng = np.random.default_rng([0, 3])
    assert [i for i, _ in head + tail] == rng.permutation(20).tolist()


def test_sampler_groups_unsorted_origins_and_batches_remaining_draws(panel_long, encoder_long):
    origins = np.array([(5, 1), (2, 7), (19, 6), (5, 4), (19, 2)], dtype=np.int32)
    upto = panel_long.timestamps(0)[7]
    sampler = WindowSampler(panel_long, origins, 4, 3, encoder_long, seed=9, label_upto=upto)
    expected = list(sampler)
    assert sorted(i for i, _ in expected) == [2, 5, 19]
    assert all(item in origins.tolist() for item in map(list, expected))
    sampler.set_epoch(0)
    batches = list(sampler.batches(2))
    assert [b.size for b in batches] == [2, 1]
    assert [
        (i, o) for b in batches for i, o in zip(b.series_index.tolist(), b.origin_pos.tolist())
    ] == expected
    for batch in batches:
        for i, origin in enumerate(batch.origin_pos.tolist()):
            assert batch.label_mask[i].tolist() == [origin + h <= 7 for h in range(4)]
    sampler.set_epoch(0)
    assert next(iter(sampler)) == expected[0]
    assert sum(batch.size for batch in sampler.batches(2)) == 2


def test_origins_are_uniform_within_each_series_and_rng_is_local(panel8, encoder8):
    origins = np.array([(0, 1), (0, 2), (0, 3), (1, 5)], dtype=np.int32)
    sampler = WindowSampler(panel8, origins, 2, 4, encoder8, seed=71)
    counts = np.zeros(3, dtype=int)
    numpy_state = copy.deepcopy(np.random.get_state())
    torch_state = torch.random.get_rng_state().clone()
    for epoch in range(600):
        sampler.set_epoch(epoch)
        drawn = dict(sampler)
        assert drawn[1] == 5
        counts[drawn[0] - 1] += 1
    assert np.all((counts > 150) & (counts < 250)), counts
    actual_state = np.random.get_state()
    for before, after in zip(numpy_state, actual_state):
        np.testing.assert_equal(before, after)
    assert torch.equal(torch_state, torch.random.get_rng_state())


def test_sampler_with_no_origins(panel8, encoder8):
    sampler = WindowSampler(panel8, np.empty((0, 2), dtype=np.int32), 4, 8, encoder8)
    assert len(sampler) == 0 and list(sampler) == [] and list(sampler.batches(2)) == []
    sampler.load_state(sampler.state())
    assert list(sampler) == []


def test_invalid_sizes_items_and_sampler_state_fail_clearly(panel8, encoder8):
    for horizon, min_context in [(0, 1), (1, 0), (1, -1)]:
        with pytest.raises(ValueError):
            build_origins(panel8, horizon, min_context)
    for items in [[(-1, 2)], [(2, 1)], [(0, -1)], [(0, 9)], [(0, 1.5)]]:
        with pytest.raises(ValueError):
            make_batch(panel8, items, 4, 8, encoder8)
    for horizon, max_context in [(0, 8), (4, 0)]:
        with pytest.raises(ValueError):
            make_batch(panel8, [(0, 2)], horizon, max_context, encoder8)
    sampler = WindowSampler(panel8, build_origins(panel8, 4, 1), 4, 8, encoder8)
    for state in [{"epoch": -1, "position": 0}, {"epoch": 0, "position": 3}]:
        with pytest.raises(ValueError):
            sampler.load_state(state)
    with pytest.raises(ValueError):
        list(sampler.batches(0))


def test_m5_sized_origin_index_is_small_and_fast(record_property):
    panel = synthetic_panel(n_series=30_490, length=250, freq="W-SUN", seed=0)
    started = time.perf_counter()
    origins = build_origins(panel, 13, 1)
    elapsed = time.perf_counter() - started
    record_property("origin_seconds", elapsed)
    record_property("origin_bytes", origins.nbytes)
    print(f"origin index: {elapsed:.3f}s, {origins.nbytes:,} bytes, {len(origins):,} rows")
    assert origins.shape == (30_490 * 249, 2) and origins.dtype == np.int32
    assert origins.nbytes < 120_000_000
    assert elapsed < 20, f"origin index took {elapsed:.3f}s"


@pytest.fixture
def panel_with_cold_and_tiny_std():
    panel = weekly_panel([40, 38])
    panel.roles = replace(panel.roles, known_dynamic=("promo", "tiny", "price"))
    for series in panel.series:
        series.dynamic["promo"] = np.r_[np.full(30, np.nan), np.full(10, 1e5)]
        series.dynamic["tiny"] = np.r_[1e-9, np.zeros(27), np.full(12, 0.5)]
        series.dynamic["price"] = np.r_[100.0 + 1e-6, np.full(27, 100.0), np.full(12, 110.0)]
    return panel


def test_cold_known_covariate_is_masked_off_in_context_and_future(panel_with_cold_and_tiny_std):
    panel = panel_with_cold_and_tiny_std
    encoder = StaticEncoder().fit(panel, ["a", "b"])
    col = len(calendar_feature_names(panel.freq))
    batch = make_batch(panel, [(0, 28), (1, 25)], 4, 32, encoder)
    assert not batch.enc_cov_mask[0, :, col].any()
    assert not batch.dec_cov_mask[0, :, col].any()
    assert not batch.enc_cov[0, :, col].any() and not batch.dec_cov[0, :, col].any()
    assert batch.cold_covariates.tolist() == [1, 0]
    assert (
        batch.cold_covariates.dtype
        == batch.clipped_covariates.dtype
        == batch.floored_covariates.dtype
        == torch.int64
    )
    observed = make_batch(panel, [(0, 34)], 4, 32, encoder)
    assert observed.cold_covariates.tolist() == [0]
    assert observed.dec_cov_mask[0, :, col].all()
    # Observations outside the cropped context still prevent the cold rule.
    panel.series[0].dynamic["promo"][0] = 2.0
    cropped = make_batch(panel, [(0, 28)], 4, 2, encoder)
    assert cropped.cold_covariates.tolist() == [0]
    assert cropped.dec_cov_mask[0, :, col].tolist() == [False, False, True, True]


def test_standardized_covariate_is_bounded_and_counted(panel_with_cold_and_tiny_std):
    from demand_forecast.data.sampler import COVARIATE_BOUND

    panel = panel_with_cold_and_tiny_std
    encoder = StaticEncoder().fit(panel, ["a", "b"])
    col = len(calendar_feature_names(panel.freq)) + 1
    batch = make_batch(panel, [(0, 28)], 4, 32, encoder)
    assert batch.dec_cov[0, :, col].tolist() == [COVARIATE_BOUND] * 4
    assert batch.clipped_covariates.tolist() == [4]
    assert batch.enc_cov.abs().max() <= COVARIATE_BOUND
    assert batch.dec_cov_mask[0, :, col].all()


@pytest.mark.parametrize(
    "source,origin,stop,cold",
    [
        (None, 4, 8, False),
        ([np.nan, np.inf, -np.inf, np.nan], 4, 8, False),
        ([np.nan, np.inf, -np.inf, np.nan, 1e5], 4, 8, True),
        ([np.nan, np.inf, -np.inf, np.nan, 1e5], 4, 4, False),
        ([1e5], 0, 1, True),
    ],
)
def test_cold_detection_uses_finite_past_and_only_the_requested_slice(source, origin, stop, cold):
    from demand_forecast.data.sampler import _dynamic_values

    values, mask, diagnostics = _dynamic_values(
        None if source is None else np.array(source), 0, origin, stop
    )
    assert not values.any() and not mask.any()
    assert diagnostics.cold is cold and diagnostics.clipped == 0 and not diagnostics.floored


@pytest.mark.parametrize("past,expected", [([0, 1], [0, 1]), ([2, 4], [-1, 1])])
def test_inf_in_a_covariate_is_missing(panel8, encoder8, past, expected):
    panel8.series[0].dynamic["promo"] = np.array([*past, np.inf, -np.inf, np.inf, -np.inf, *past])
    batch = make_batch(panel8, [(0, 4)], 4, 4, encoder8)
    assert batch.enc_cov[0, :, 4].tolist() == [*expected, 0, 0]
    assert batch.enc_cov_mask[0, :, 4].tolist() == [True, True, False, False]
    assert batch.dec_cov[0, :, 4].tolist() == [0, 0, *expected]
    assert batch.dec_cov_mask[0, :, 4].tolist() == [False, False, True, True]
    assert batch.cold_covariates.tolist() == batch.clipped_covariates.tolist() == [0]


def test_indicator_future_is_bounded_and_ordinary_zscore_is_untouched(panel8, encoder8):
    from demand_forecast.data.sampler import COVARIATE_BOUND

    panel8.series[0].dynamic["promo"] = np.array([0.0, 1.0, 1e5, -1e5, 1e3, -1e3])
    batch = make_batch(panel8, [(0, 2)], 4, 2, encoder8)
    assert batch.enc_cov[0, :, 4].tolist() == [0, 1]
    assert batch.dec_cov[0, :, 4].tolist() == [COVARIATE_BOUND, -COVARIATE_BOUND] * 2
    assert batch.clipped_covariates.tolist() == [2]
    panel8.series[0].dynamic["promo"] = np.array([2.0, 4.0, 53.0, -47.0])
    batch = make_batch(panel8, [(0, 2)], 2, 2, encoder8)
    assert batch.dec_cov[0, :, 4].tolist() == [50, -50]
    assert batch.clipped_covariates.tolist() == [0]


def test_ordinary_covariates_are_byte_identical_to_the_previous_helper(monkeypatch):
    panel = weekly_panel([40, 40])
    panel.roles = replace(panel.roles, known_dynamic=("promo", "constant"))
    for series in panel.series:
        # Oil has an ordinary std; this new column has std == 0 before both origins.
        series.dynamic["constant"] = np.r_[np.full(32, 100.0), np.full(8, 110.0)]
    encoder = StaticEncoder().fit(panel, ["a", "b"])
    assert_batch_matches_reference(monkeypatch, panel, [(0, 28), (1, 32)], 4, 16, encoder)


def test_fp16_cast_inputs_and_model_pass_stay_finite_after_the_guards(
    panel_with_cold_and_tiny_std, record_property
):
    panel = panel_with_cold_and_tiny_std
    encoder = StaticEncoder().fit(panel, ["a", "b"])
    cfg = ForecastConfig(horizon=4, max_context=32)
    batch = make_batch(panel, [(0, 28), (1, 28)], cfg.horizon, cfg.max_context, encoder)
    for name in ("target", "enc_cov", "dec_cov", "labels"):
        assert torch.isfinite(getattr(batch, name).half()).all(), name
    model = build_v2(ModelSpec.from_panel(panel, encoder, cfg, 16, 4))
    outputs = model(batch)
    assert torch.isfinite(outputs).all()
    loss, count = pinball_loss(
        outputs, batch.labels, batch.label_mask, batch.loss_eligible, torch.tensor(cfg.quantiles)
    )
    assert count == 8 and torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    if torch.amp.autocast_mode.is_autocast_available("cpu"):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            outputs = model(batch)
        assert torch.isfinite(outputs).all()
        record_property("cpu_bfloat16_forward", "finite")
    else:
        record_property("cpu_bfloat16_forward", "unsupported")


def test_tiny_std_past_is_floored_and_counted():
    panel = weekly_panel([40])
    panel.series[0].dynamic["promo"] = np.r_[100.0 + 1e-6, np.full(27, 100.0), np.full(12, 110.0)]
    encoder = StaticEncoder().fit(panel, ["a"])
    col = len(calendar_feature_names(panel.freq))
    batch = make_batch(panel, [(0, 28)], 4, 32, encoder)
    torch.testing.assert_close(batch.dec_cov[0, :, col], torch.full((4,), 100.0), rtol=0, atol=1e-3)
    assert batch.floored_covariates.tolist() == [1]
    assert batch.clipped_covariates.tolist() == batch.cold_covariates.tolist() == [0]


def test_std_floor_uses_only_finite_whole_past_and_cannot_change_with_future():
    from demand_forecast.data.sampler import _dynamic_values

    source = np.array([-100.0 - 1e-6, -100.0, np.inf, -np.inf, np.nan, -110.0])
    values, mask, diagnostics = _dynamic_values(source, 1, 5, 6)
    assert values[-1] == pytest.approx(-100.0, abs=1e-3)
    assert mask.tolist() == [True, False, False, False, True]
    assert values[1:4].tolist() == [0.0] * 3
    assert diagnostics.floored and diagnostics.clipped == 0 and not diagnostics.cold
    source[-1] = 1e10
    changed, changed_mask, diagnostics = _dynamic_values(source, 1, 5, 6)
    assert changed[:-1].tobytes() == values[:-1].tobytes()
    np.testing.assert_array_equal(changed_mask, mask)
    assert diagnostics.floored and diagnostics.clipped == 1


def test_add_row_diagnostics_accumulates_each_column_in_only_its_row(panel8, encoder8):
    from demand_forecast.data.sampler import DynamicDiagnostics, _add_row_diagnostics

    batch = make_batch(panel8, [(0, 4), (1, 4)], 4, 4, encoder8)
    _add_row_diagnostics(batch, 1, DynamicDiagnostics(cold=True, clipped=3, floored=True))
    _add_row_diagnostics(batch, 1, DynamicDiagnostics(cold=False, clipped=4, floored=True))
    assert batch.cold_covariates.tolist() == [0, 1]
    assert batch.clipped_covariates.tolist() == [0, 7]
    assert batch.floored_covariates.tolist() == [0, 2]
