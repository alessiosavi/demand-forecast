import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.encoders import StaticEncoder
from tests.helpers.panels import series_with


def test_codes_reserve_unknown_and_missing(panel29):
    enc = StaticEncoder().fit(panel29, ["a", "b"])
    assert enc.transform(panel29.series[panel29.index_of("a")])[0] == {"cat": 2, "__series__": 2}
    assert enc.vocab_sizes == {"cat": 4, "__series__": 4}
    assert enc.transform(series_with(panel29, cat=None))[0]["cat"] == 1
    assert enc.transform(series_with(panel29, cat="Z"))[0]["cat"] == 0
    assert enc.transform(series_with(panel29, series_id="new"))[0]["__series__"] == 0


def test_numeric_standardised_with_mask_and_json_round_trip(panel29):
    enc = StaticEncoder().fit(panel29, ["a", "b"])
    _, num, mask = enc.transform(series_with(panel29, size=np.nan))
    assert num.tolist() == [0.0] and mask.tolist() == [False]
    assert num.dtype == np.float32 and mask.dtype == np.bool_
    assert num.shape == mask.shape == (1,) and enc.n_numeric == 1
    assert StaticEncoder.from_json(enc.to_json()).vocab_sizes == enc.vocab_sizes
    for series_id, expected in [("a", -1.0), ("b", 1.0), ("c", 3.0)]:
        _, num, mask = enc.transform(panel29.series[panel29.index_of(series_id)])
        np.testing.assert_allclose(num, [expected])
        assert mask.tolist() == [True]


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_missing_values_are_excluded_from_fit_and_masked(panel29, missing):
    panel = replace(
        panel29,
        series=[
            series_with(panel29, series_id="a", cat="B", size=2.0),
            series_with(panel29, series_id="b", cat="A", size=6.0),
            series_with(panel29, series_id="c", cat=missing, size=missing),
        ],
    )
    enc = StaticEncoder().fit(panel, ["a", "b", "c"])
    assert enc.vocab_sizes == {"cat": 4, "__series__": 5}
    codes, num, mask = enc.transform(panel.series[2])
    assert codes == {"cat": 1, "__series__": 4}
    np.testing.assert_array_equal(num, [0.0])
    np.testing.assert_array_equal(mask, [False])
    np.testing.assert_allclose(enc.transform(panel.series[0])[1], [-1.0])
    np.testing.assert_allclose(enc.transform(panel.series[1])[1], [1.0])


def test_fit_uses_only_training_ids_and_accepts_a_generator(panel29):
    panel = replace(
        panel29,
        series=[*panel29.series[:2], series_with(panel29, series_id="c", cat="Z", size=1e6)],
    )
    enc = StaticEncoder().fit(panel, (series_id for series_id in ["b", "a", "a"]))
    assert enc.to_json() == StaticEncoder().fit(panel, ["a", "b"]).to_json()
    codes, num, mask = enc.transform(panel.series[2])
    assert codes == {"cat": 0, "__series__": 0}
    assert enc.vocab_sizes == {"cat": 4, "__series__": 4}
    np.testing.assert_allclose(num, [(1e6 - 15.0) / 5.0])
    assert mask.tolist() == [True]


def test_string_vocabularies_sort_lexically_and_keep_missing_distinct(panel29):
    panel = replace(
        panel29,
        series=[
            series_with(panel29, series_id="z", cat=2),
            series_with(panel29, series_id="x", cat=10),
            series_with(panel29, series_id="y", cat="nan"),
        ],
    )
    enc = StaticEncoder().fit(panel, ["z", "y", "x"])
    assert enc.transform(panel.series[0])[0] == {"cat": 3, "__series__": 4}
    assert enc.transform(panel.series[1])[0] == {"cat": 2, "__series__": 2}
    assert enc.transform(panel.series[2])[0] == {"cat": 4, "__series__": 3}
    assert enc.transform(series_with(panel, cat="2"))[0]["cat"] == 3
    assert enc.transform(series_with(panel, cat=np.nan))[0]["cat"] == 1


@pytest.mark.parametrize("values", [[7.0, 7.0, None], [7.0, None, np.nan]])
def test_constant_or_single_observed_numeric_uses_unit_std(panel29, values):
    panel = replace(
        panel29,
        series=[
            replace(series, statics={"size": value})
            for series, value in zip(panel29.series, values)
        ],
    )
    enc = StaticEncoder().fit(panel, ["a", "b", "c"])
    np.testing.assert_allclose(enc.transform(series_with(panel, size=12.0))[1], [5.0])
    np.testing.assert_allclose(enc.transform(panel.series[0])[1], [0.0])
    assert enc.transform(panel.series[0])[2].tolist() == [True]


def test_all_missing_statics_use_neutral_stats_and_reserved_vocab_slots(panel29):
    panel = replace(panel29, series=[replace(series, statics={}) for series in panel29.series])
    enc = StaticEncoder().fit(panel, ["a", "b"])
    assert enc.vocab_sizes == {"cat": 2, "__series__": 4}
    codes, num, mask = enc.transform(panel.series[0])
    assert codes == {"cat": 1, "__series__": 2}
    assert num.tolist() == [0.0] and mask.tolist() == [False]
    codes, num, mask = enc.transform(series_with(panel, cat="new", size=7.0))
    assert codes["cat"] == 0 and num.tolist() == [7.0] and mask.tolist() == [True]
    json.dumps(enc.to_json(), allow_nan=False)


def test_empty_training_ids_do_not_learn_from_panel(panel29):
    enc = StaticEncoder().fit(panel29, [])
    assert enc.vocab_sizes == {"cat": 2, "__series__": 2}
    codes, num, mask = enc.transform(panel29.series[0])
    assert codes == {"cat": 0, "__series__": 0}
    assert num.tolist() == [10.0] and mask.tolist() == [True]
    assert StaticEncoder.from_json(enc.to_json()).to_json() == enc.to_json()


def test_no_static_columns_still_encodes_series_ids(panel29):
    panel = replace(panel29, roles=replace(panel29.roles, static_categorical=(), static_numeric=()))
    enc = StaticEncoder().fit(panel, ["b", "a"])
    codes, num, mask = enc.transform(panel.series[0])
    assert codes == {"__series__": 2} and enc.vocab_sizes == {"__series__": 4}
    assert enc.n_numeric == 0 and num.shape == mask.shape == (0,)
    assert num.dtype == np.float32 and mask.dtype == np.bool_
    assert StaticEncoder.from_json(enc.to_json()).to_json() == enc.to_json()


def test_json_round_trip_preserves_values_stats_and_column_order(panel29):
    panel = replace(
        panel29,
        roles=replace(
            panel29.roles,
            static_categorical=("z_cat", "a_cat"),
            static_numeric=("z_num", "a_num"),
        ),
        series=[
            series_with(panel29, series_id="a", z_cat="B", a_cat=2, z_num=2.0, a_num=-5.0),
            series_with(panel29, series_id="b", z_cat="A", a_cat=10, z_num=6.0, a_num=15.0),
        ],
    )
    enc = StaticEncoder().fit(panel, ["a", "b"])
    # Object key order can change on disk; role order must survive it.
    restored = StaticEncoder.from_json(
        json.loads(json.dumps(enc.to_json(), sort_keys=True, allow_nan=False))
    )
    assert restored.to_json() == enc.to_json()
    assert list(restored.vocab_sizes) == ["z_cat", "a_cat", "__series__"]
    assert restored.vocab_sizes == enc.vocab_sizes and restored.n_numeric == 2
    probe = series_with(panel, series_id="new", z_cat="Z", a_cat=None, z_num=8.0, a_num=35.0)
    codes, num, mask = restored.transform(probe)
    assert list(codes) == ["z_cat", "a_cat", "__series__"]
    assert codes == {"z_cat": 0, "a_cat": 1, "__series__": 0}
    np.testing.assert_allclose(num, [2.0, 3.0])
    assert mask.tolist() == [True, True]
    for series in [*panel.series, probe, series_with(panel, z_num=None, a_num=np.nan)]:
        expected = enc.transform(series)
        actual = restored.transform(series)
        assert actual[0] == expected[0]
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])


def test_refit_replaces_previous_columns_and_training_state(panel29, encoder29):
    panel = replace(panel29, roles=replace(panel29.roles, static_categorical=(), static_numeric=()))
    assert encoder29.fit(panel, ["b"]) is encoder29
    assert encoder29.to_json() == StaticEncoder().fit(panel, ["b"]).to_json()
    assert encoder29.transform(panel.series[0])[0] == {"__series__": 0}


def test_unknown_training_id_fails_loudly(panel29):
    with pytest.raises(KeyError, match="absent"):
        StaticEncoder().fit(panel29, ["a", "absent"])


def test_reserved_series_column_cannot_overwrite_id_vocabulary(panel29):
    panel = replace(panel29, roles=replace(panel29.roles, static_categorical=("__series__",)))
    with pytest.raises(ValueError, match="__series__"):
        StaticEncoder().fit(panel, ["a"])


def test_unfitted_encoder_fails_loudly(panel29):
    enc = StaticEncoder()
    with pytest.raises(RuntimeError, match="fit"):
        enc.transform(panel29.series[0])
    with pytest.raises(RuntimeError, match="fit"):
        enc.to_json()
