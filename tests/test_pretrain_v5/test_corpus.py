"""Native shard identity, decoded memory bounds and row validation."""

import gc
import hashlib
import importlib
import json
from dataclasses import replace

import numpy as np
import pytest

from tests.helpers.corpus_v4_fixture import write_corpus, write_malformed_row, write_rows

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def api():
    return importlib.import_module("demand_forecast.pretrain.v5.corpus")


def test_index_order_counts_and_footer_identity(tmp_path):
    paths = write_corpus(tmp_path)
    index = api().CorpusIndex.build(tmp_path)
    assert index.groups == (
        ("covariates", "covariates/drivers/train-000.parquet", 0, 3),
        ("univariate", "univariate/alpha/train-000.parquet", 0, 2),
        ("univariate", "univariate/alpha/train-000.parquet", 1, 2),
        ("univariate", "univariate/beta/validation-000.parquet", 0, 3),
    )
    assert index.rows == 10 and index.excluded == ()
    assert index.digest == api().CorpusIndex.build(tmp_path).digest
    assert isinstance(hash(index), int)
    assert len(paths) == 3


@pytest.fixture
def folder_index(tmp_path):
    paths = write_corpus(tmp_path)
    for name, targets in (
        ("univariate/alpha/train-001.parquet", [[np.arange(40)], [np.arange(1000)]]),
        ("univariate/excluded/train-000.parquet", [[np.arange(1000)]]),
    ):
        paths += (write_rows(tmp_path / name, targets, row_group_size=1),)
    objects = manifest(tmp_path, paths)
    return api().CorpusIndex.build(tmp_path, max_group_bytes=23_000), objects


def test_folders_cover_admitted_rows_and_values_without_affecting_digest(folder_index):
    index, objects = folder_index
    assert [(f.name, f.first_row, f.rows, f.values) for f in index.folders] == [
        ("covariates/drivers", 0, 3, 360),
        ("univariate/alpha", 3, 5, 200),
        ("univariate/beta", 8, 3, 880),
    ]
    assert sum(f.rows for f in index.folders) == index.rows == 11
    assert [group[:2] for group in index.excluded] == [
        ("univariate/alpha/train-001.parquet", 1),
        ("univariate/excluded/train-000.parquet", 0),
    ]
    assert index.digest == api().listing_digest(objects)
    assert index.digest == api().CorpusIndex.build(index.root).digest


def test_pick_none_matches_200_base_revision_row_draws(folder_index):
    index, _ = folder_index
    rng = np.random.default_rng(7)
    # The base draws were int(rng.integers(index.rows)), with 11 admitted rows.
    rows = """
    10 6 7 9 6 8 9 2 0 3 3 9 10 0 5 9 1 8 1 5 8 3 3 3 7 2 10 4 5 5 6 6 5 10 8 8 7 6 3 10
    5 2 9 1 9 6 1 0 4 0 1 5 10 5 8 10 9 6 4 5 2 5 4 2 10 0 1 2 10 7 9 2 7 4 5 0 6 9 7 1
    5 2 10 9 2 5 10 9 7 7 0 8 5 1 2 5 7 5 6 9 7 3 7 6 1 0 7 4 6 3 2 1 4 8 4 4 6 10 4 6
    4 6 4 7 6 7 10 1 6 4 3 2 0 4 9 1 4 10 10 2 0 7 0 3 5 9 1 7 5 1 9 9 5 10 6 9 10 6 3 1
    6 2 8 10 2 6 0 1 4 9 10 7 6 6 0 4 10 4 2 2 7 0 10 9 6 5 8 6 5 3 4 8 0 0 7 4 8 0 9 1
    """
    assert [index.pick(rng, None) for _ in range(200)] == list(map(int, rows.split()))


@pytest.mark.parametrize("power,weights", [(0, [1, 1, 1]), (1, [360, 200, 880])])
def test_pick_proposes_a_folder_then_a_uniform_row(folder_index, power, weights):
    index, _ = folder_index
    expected = np.array(weights) / sum(weights)
    calls = []

    class FixedDraws:
        def choice(self, n, *, p):
            calls.append("choice")
            assert n == 3
            np.testing.assert_allclose(p, expected, rtol=0, atol=1e-12)
            return 1

        def integers(self, low, high):
            calls.append("integers")
            assert (low, high) == (3, 8)
            return 7

    assert index.pick(FixedDraws(), power) == 7
    assert calls == ["choice", "integers"]

    rng = np.random.default_rng(42)
    counts = np.bincount([index.pick(rng, power) for _ in range(10_000)], minlength=index.rows)
    sizes = [3, 5, 3]
    folder_counts = np.add.reduceat(counts, [0, 3, 8])
    predicted = expected * 10_000
    assert np.sum((folder_counts - predicted) ** 2 / predicted) < 16
    by_row = np.repeat(predicted / sizes, sizes)
    assert np.sum((counts - by_row) ** 2 / by_row) < 32
    if power == 1:
        swapped = predicted[[2, 1, 0]]
        assert np.sum((folder_counts - swapped) ** 2 / swapped) > 16


@pytest.mark.parametrize("power", [0, 0.5, 1])
def test_pick_with_one_folder_and_one_row(tmp_path, power):
    write_rows(tmp_path / "univariate/only/train-000.parquet", [[np.arange(32)]])
    index = api().CorpusIndex.build(tmp_path)
    assert index.pick(np.random.default_rng(0), power) == 0


def test_pick_refuses_zero_total_folder_weight(folder_index):
    index, _ = folder_index
    index = replace(index, folders=tuple(replace(folder, values=0) for folder in index.folders))
    with pytest.raises(ValueError, match="^corpus folders hold no values to weight$"):
        index.pick(np.random.default_rng(0), power=1)


def test_equal_sized_changed_footer_changes_metadata_identity(tmp_path):
    path = tmp_path / "univariate/a/train-000.parquet"
    write_rows(path, [[np.zeros(32)]], compression=None)
    before = api().CorpusIndex.build(tmp_path).digest
    size = path.stat().st_size
    write_rows(path, [[np.ones(32)]], compression=None)
    assert path.stat().st_size == size
    assert api().CorpusIndex.build(tmp_path).digest != before


def manifest(root, paths):
    objects = [(p.relative_to(root).as_posix(), p.stat().st_size, "bucket-crc") for p in paths]
    (root / "manifest.json").write_text(json.dumps({"objects": objects}))
    return objects


def test_manifest_digest_is_sorted_listing_digest(tmp_path):
    paths = write_corpus(tmp_path)
    objects = manifest(tmp_path, paths)
    expected = hashlib.sha256(
        "".join(f"{p}\t{n}\t{c}\n" for p, n, c in sorted(objects)).encode()
    ).hexdigest()
    assert api().listing_digest(reversed(objects)) == expected
    assert api().CorpusIndex.build(tmp_path).digest == expected


@pytest.mark.parametrize("damage", ["missing", "size", "unlisted"])
def test_manifest_refuses_incomplete_or_extra_files(tmp_path, damage):
    paths = write_corpus(tmp_path)
    manifest(tmp_path, paths)
    if damage == "missing":
        paths[0].unlink()
    elif damage == "size":
        with paths[0].open("ab") as stream:
            stream.write(b"extra")
    else:
        write_rows(tmp_path / "univariate/extra/train-000.parquet", [[np.arange(32)]])
    with pytest.raises(ValueError, match="parquet"):
        api().CorpusIndex.build(tmp_path)


def test_groups_are_admitted_at_the_numeric_estimate_boundary(tmp_path):
    path = write_rows(tmp_path / "univariate/a/train-000.parquet", [[np.arange(32)]])
    with pq.ParquetFile(path) as reader:
        group = reader.metadata.row_group(0)
        values = sum(
            group.column(i).num_values
            for i in range(group.num_columns)
            if group.column(i).path_in_schema.split(".")[0] == "target"
        )
    estimate = 24 * values + 64
    assert api().CorpusIndex.build(tmp_path, max_group_bytes=estimate).rows == 1
    excluded = api().CorpusIndex.build(tmp_path, max_group_bytes=estimate - 1)
    assert excluded.rows == 0 and len(excluded.excluded) == 1
    assert excluded.excluded[0][:2] == ("univariate/a/train-000.parquet", 0)


def test_compressed_million_zeros_excluded_before_numeric_decode(tmp_path, monkeypatch):
    path = write_rows(tmp_path / "univariate/a/train-000.parquet", [[np.zeros(1_000_000)]])
    assert path.stat().st_size < 100_000
    original = pq.ParquetFile.read_row_group

    def read(reader, i, columns=None, **kwargs):
        assert columns == ["freq"], "oversized numeric group must never decode"
        return original(reader, i, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", read)
    index = api().CorpusIndex.build(tmp_path, max_group_bytes=1 << 20)
    assert index.rows == 0 and len(index.excluded) == 1


@pytest.mark.parametrize("length,variates", [(1, 1), (32, 1), (32, 1000)])
def test_numeric_decoded_table_is_below_metadata_estimate(tmp_path, length, variates):
    path = write_rows(
        tmp_path / "univariate/a/train-000.parquet",
        [[np.arange(length)] * variates],
        value_type=pa.int32(),
    )
    with pq.ParquetFile(path) as reader:
        group = reader.metadata.row_group(0)
        values = sum(
            group.column(i).num_values
            for i in range(group.num_columns)
            if group.column(i).path_in_schema.split(".")[0] == "target"
        )
        table = reader.read_row_group(0, columns=["target"])
    estimate = 24 * values + 64
    assert table.nbytes <= estimate
    assert api().CorpusIndex.build(tmp_path, max_group_bytes=estimate).rows == 1


def test_long_ids_never_decoded_or_charged_to_numeric_budget(tmp_path, monkeypatch):
    write_rows(
        tmp_path / "univariate/a/train-000.parquet",
        [[np.arange(32)]],
        item_id_size=4096,
    )
    original = pq.ParquetFile.read_row_group
    requested = []

    def read(reader, i, columns=None, **kwargs):
        assert columns and set(columns) <= {"target", "covariates", "freq"}
        requested.append(tuple(columns))
        return original(reader, i, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", read)
    index = api().CorpusIndex.build(tmp_path, max_group_bytes=832)
    assert index.rows == 1
    np.testing.assert_array_equal(
        api().read_values(index, 0, api().RowCache())["targets"], [np.arange(32)]
    )
    assert ("target",) in requested


def test_excluded_group_with_unparsed_frequency_is_skipped(tmp_path, monkeypatch):
    write_rows(tmp_path / "univariate/a/train-bad.parquet", [[np.zeros(32)]], freqs=["nonsense"])
    module = api()
    admitted = module.CorpusIndex.build(tmp_path)

    def scan(parquet, group, unparsed_frequencies):
        pytest.fail(f"scanned frequencies for excluded group {group}")

    monkeypatch.setattr(module, "_frequencies", scan)
    index = module.CorpusIndex.build(tmp_path, max_group_bytes=1)
    assert index.rows == 0
    assert [group[:2] for group in index.excluded] == [("univariate/a/train-bad.parquet", 0)]
    assert index.digest == admitted.digest
    assert index.unparsed_frequencies == {}


def test_mixed_frequencies_keep_row_order_and_count_labels_across_groups(tmp_path):
    freqs = ["1m", "D", "1m", "D", "irregular", "1m"]
    write_rows(
        tmp_path / "univariate/a/train-000.parquet",
        [[np.arange(32) + i] for i in range(len(freqs))],
        freqs=freqs,
        row_group_size=3,
    )
    module = api()
    index, cache = module.CorpusIndex.build(tmp_path), module.RowCache()
    assert index.rows == len(freqs)
    assert index.unparsed_frequencies == {"1m": 3, "irregular": 1}
    for i, expected in enumerate([None, "D", None, "D", None, None]):
        row = module.read_values(index, i, cache)
        assert row["freq"] == expected and row["row"] == i
        np.testing.assert_array_equal(row["targets"], [np.arange(32) + i])


@pytest.mark.parametrize("freqs", [[None, None], ["D", None, "h", None]])
def test_null_frequencies_are_counted_and_read_as_none(tmp_path, freqs):
    write_rows(
        tmp_path / "univariate/a/train-000.parquet",
        [[np.arange(32)]] * len(freqs),
        freqs=freqs,
        row_group_size=len(freqs),
    )
    module = api()
    index, cache = module.CorpusIndex.build(tmp_path), module.RowCache()
    assert index.unparsed_frequencies == {"null": 2}
    assert [module.read_values(index, i, cache)["freq"] for i in range(index.rows)] == freqs


def test_unparsed_label_and_nulls_in_one_group_are_counted_and_read_as_none(tmp_path):
    write_rows(
        tmp_path / "univariate/a/train-000.parquet",
        [[np.arange(32)]] * 4,
        freqs=["D", None, "nonsense", None],
        row_group_size=4,
    )
    module = api()
    index, cache = module.CorpusIndex.build(tmp_path), module.RowCache()
    assert index.unparsed_frequencies == {"nonsense": 1, "null": 2}
    assert [module.read_values(index, i, cache)["freq"] for i in range(index.rows)] == [
        "D",
        None,
        None,
        None,
    ]


@pytest.mark.parametrize("label", ["", "0D", "-2h", "999999999999999999999s"])
def test_empty_nonpositive_and_overflowing_frequencies_are_unparsed(tmp_path, label):
    write_rows(tmp_path / "univariate/a/train-000.parquet", [[np.arange(32)]], freqs=[label])
    module = api()
    index = module.CorpusIndex.build(tmp_path)
    assert index.unparsed_frequencies == {label: 1}
    assert module.read_values(index, 0, module.RowCache())["freq"] is None


@pytest.mark.parametrize("distinct", [("D",), ("h", "D", "ME")])
def test_large_frequency_group_only_materialises_distinct_strings(tmp_path, monkeypatch, distinct):
    count = 20_000
    freqs = [distinct[i % len(distinct)] for i in range(count)]
    write_rows(
        tmp_path / "univariate/a/train-000.parquet",
        [[np.zeros(1)]] * count,
        freqs=freqs,
        row_group_size=count,
    )
    original = pq.ParquetFile.read_row_group

    class FrequencyColumn:
        def __init__(self, column):
            self.column = column

        def to_pylist(self):
            assert len(self.column) <= 3, "materialised per-row Python frequency strings"
            return self.column.to_pylist()

        def dictionary_encode(self):
            return FrequencyColumn(self.column.dictionary_encode())

        def combine_chunks(self):
            return FrequencyColumn(self.column.combine_chunks())

        @property
        def dictionary(self):
            return FrequencyColumn(self.column.dictionary)

        def __getattr__(self, name):
            return getattr(self.column, name)

    def read(reader, i, columns=None, **kwargs):
        table = original(reader, i, columns=columns, **kwargs)
        assert columns == ["freq"]
        return {"freq": FrequencyColumn(table["freq"])}

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", read)
    index = api().CorpusIndex.build(tmp_path)
    assert index.rows == count
    frequency = index._frequencies[0]
    assert frequency.values == tuple(sorted(distinct))
    assert [frequency.at(i) for i in range(count)] == freqs
    if len(distinct) == 1:
        assert frequency.codes == b"" and frequency.width == 0
    else:
        assert len(frequency.codes) == count and frequency.width == 1


@pytest.mark.parametrize("location", ["quarantine", "univariate/quarantine-source"])
def test_quarantined_paths_refused(tmp_path_factory, location):
    tmp_path = tmp_path_factory.mktemp("placement")
    root = tmp_path / location if location == "quarantine" else tmp_path
    path = (
        root / "univariate/a/train-000.parquet"
        if location == "quarantine"
        else tmp_path / location / "train-000.parquet"
    )
    write_rows(path, [[np.arange(32)]])
    with pytest.raises(ValueError, match="quarantine"):
        api().CorpusIndex.build(root)


def test_all_rows_shapes_frequencies_nan_and_bounded_cache(tmp_path, monkeypatch):
    write_corpus(tmp_path)
    index = api().CorpusIndex.build(tmp_path)
    original = pq.ParquetFile.read_row_group
    reads = []

    def read(reader, i, columns=None, **kwargs):
        assert set(columns) <= {"target", "covariates"}
        reads.append(i)
        return original(reader, i, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", read)
    cache = api().RowCache(max_bytes=3000)
    expected_freqs = ["h", "D", "ME", "D", "h", "ME", "W-MON", "D", "D", "D"]
    for row in range(index.rows):
        cold = api().read_values(index, row, cache)
        warm = api().read_values(index, row, cache)
        assert cold["freq"] == warm["freq"] == expected_freqs[row]
        assert cold["targets"].shape == (20 if row == 7 else 1, 40)
        assert cold["covariates"].shape == (2 if row < 3 else 0, 40)
        for name in ("targets", "covariates"):
            assert cold[name].dtype == np.float64
            assert np.all(np.isfinite(cold[name]) | np.isnan(cold[name]))
            np.testing.assert_array_equal(cold[name], warm[name])
        assert 0 <= cache.nbytes <= 3000
    assert len(reads) == 9  # first three groups cached; 20-variate group cannot fit
    assert api().read_values(index, 6, cache)["row"] == 3


def test_cache_evicts_least_recently_used_group(tmp_path, monkeypatch):
    path = tmp_path / "univariate/a/train-000.parquet"
    write_rows(path, [[np.arange(32) + i] for i in range(3)], row_group_size=1)
    index = api().CorpusIndex.build(tmp_path)
    original = pq.ParquetFile.read_row_group
    reads = []

    def read(reader, i, **kwargs):
        reads.append(i)
        return original(reader, i, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", read)
    cache = api().RowCache(max_bytes=600)
    for row in (0, 1, 0, 2, 0, 1):
        api().read_values(index, row, cache)
        assert cache.nbytes <= 600
    assert reads == [0, 1, 2, 1]


@pytest.mark.parametrize("problem", ["infinity", "short"])
def test_bad_rows_raise_only_the_named_sampler_skips(tmp_path, problem):
    target = np.arange(20 if problem == "short" else 32, dtype=np.float64)
    if problem == "infinity":
        target[4] = np.inf
    write_rows(tmp_path / "univariate/a/train-bad.parquet", [[target]])
    module = api()
    expected = module.ShortRow if problem == "short" else module.CorpusValueError
    assert issubclass(expected, module.SamplerSkip)
    with pytest.raises(expected, match="train-bad.parquet.*0"):
        module.read_values(module.CorpusIndex.build(tmp_path), 0, module.RowCache())


def test_bad_values_in_covariates_are_checked(tmp_path):
    write_rows(
        tmp_path / "covariates/a/train-bad.parquet",
        [np.arange(32)],
        kind="covariates",
        covariates=[[np.full(32, -np.inf)]],
    )
    module = api()
    with pytest.raises(module.CorpusValueError, match="train-bad.parquet"):
        module.read_values(module.CorpusIndex.build(tmp_path), 0, module.RowCache())


@pytest.mark.parametrize(
    "problem", ["null_variate", "null_channel", "ragged_channels", "zero_variates", "bad_shape"]
)
def test_malformed_rows_are_counted_skips_with_file_row_context(tmp_path, problem):
    write_malformed_row(tmp_path, problem)
    module = api()
    index, cache = module.CorpusIndex.build(tmp_path), module.RowCache()
    for _ in range(2):  # cold and cached reads preserve the skip contract
        with pytest.raises(module.CorpusValueError, match=r"train-bad.parquet: row 1"):
            module.read_values(index, 1, cache)


def test_row_bounds_are_programming_errors(tmp_path):
    write_corpus(tmp_path)
    module = api()
    index = module.CorpusIndex.build(tmp_path)
    for row in (-1, index.rows):
        with pytest.raises(IndexError):
            module.read_values(index, row, module.RowCache())


def test_uncached_row_does_not_pin_whole_arrow_group(tmp_path):
    write_rows(
        tmp_path / "univariate/a/train-000.parquet",
        [[np.arange(4096, dtype=np.float64)] for _ in range(4)],
        row_group_size=4,
    )
    module = api()
    index = module.CorpusIndex.build(tmp_path)
    gc.collect()
    before = pa.total_allocated_bytes()
    values = module.read_values(index, 0, module.RowCache(max_bytes=0))
    gc.collect()
    assert pa.total_allocated_bytes() == before
    np.testing.assert_array_equal(values["targets"], [np.arange(4096)])


def test_excluded_groups_preserve_original_file_row_and_frequency(tmp_path):
    write_rows(
        tmp_path / "univariate/a/train-000.parquet",
        [[np.arange(100)], [np.arange(32)]],
        freqs=["h", "ME"],
        row_group_size=1,
    )
    module = api()
    index = module.CorpusIndex.build(tmp_path, max_group_bytes=832)
    assert index.rows == 1 and len(index.excluded) == 1
    row = module.read_values(index, 0, module.RowCache())
    assert row["row"] == 1 and row["freq"] == "ME"
    np.testing.assert_array_equal(row["targets"], [np.arange(32)])


def test_cache_limit_counts_full_arrow_buffers(tmp_path):
    path = write_rows(tmp_path / "univariate/a/train-000.parquet", [[np.arange(32)]])
    module = api()
    index = module.CorpusIndex.build(tmp_path)
    with pq.ParquetFile(path) as reader:
        table = reader.read_row_group(0, columns=["target"])
    assert table.get_total_buffer_size() > table.nbytes
    cache = module.RowCache(max_bytes=table.nbytes)
    module.read_values(index, 0, cache)
    assert cache.nbytes == 0  # logical nbytes misses trailing list offsets
