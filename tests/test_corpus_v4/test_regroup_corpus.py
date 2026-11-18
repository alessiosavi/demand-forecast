"""Fixture-only checks of row identity, Parquet bounds and reader admission."""

import hashlib
import importlib
import json
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from demand_forecast.pretrain.v5.config import SourceConfig
from demand_forecast.pretrain.v5.corpus import CorpusIndex, RowCache, read_values
from demand_forecast.pretrain.v5.sources import CorpusSource
from tests.helpers.corpus_v4_fixture import write_rows

UNI = "univariate/a/train-000.parquet"
COV = "covariates/b/validation-000.parquet"
LICENSE = "univariate/a/LICENSE.md"
ARTICLES = "covariates/b/articles.csv"
PARAMETERS = dict(threshold_bytes=800, group_bytes=420, group_rows=3)


@pytest.fixture
def regroup():
    return importlib.import_module("scripts.corpus_v4.regroup_corpus")


def _tree(root):
    write_rows(
        root / UNI,
        [[[float(i)] * length] for i, length in enumerate([2, 4, 7, 6, 8, 9])],
        domains=["other"] * 6,
        row_group_size=6,
    )
    write_rows(
        root / COV,
        [[1.0, 2.0]],
        kind="covariates",
        covariates=[[[3.0, 4.0]]],
    )
    (root / LICENSE).write_bytes(b"Fixture license\r\n")
    (root / ARTICLES).write_bytes(b"item,attribution\n0,fixture\n")
    for name in ("receipt.json", "stats.json", "irrelevant.txt"):
        (root / UNI).with_name(name).write_text("{}", encoding="utf-8")
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    return root


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_cli(parent, out, *args):
    return subprocess.run(
        [sys.executable, "-m", "scripts.corpus_v4.regroup_corpus", str(parent), str(out), *args],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("kind", ["univariate", "covariates"])
def test_estimate_equals_the_readers_decision(regroup, tmp_path, kind):
    cases = dict(targets=[None, [], [[2.0, None], [], None], [[3.0]]] * 2)
    relative = UNI
    if kind == "covariates":
        relative = COV
        cases = dict(
            targets=[None, [], [2.0, None], [3.0]] * 2,
            covariates=[[[1.0]], [], None, [[2.0, 3.0], []]] * 2,
            covariate_names=[["x"], [], None, ["x", "y"]] * 2,
        )
    source = write_rows(tmp_path / relative, kind=kind, row_group_size=4, **cases)
    metadata = pq.read_metadata(source)
    for group in range(metadata.num_row_groups):
        estimate = regroup.group_estimate(metadata, group)
        admitted = CorpusIndex.build(tmp_path, max_group_bytes=estimate)
        excluded = CorpusIndex.build(tmp_path, max_group_bytes=estimate - 1)
        assert (kind, relative, group, 4) in admitted.groups
        assert (
            relative,
            group,
            f"numeric estimate {estimate} exceeds max_group_bytes={estimate - 1}",
        ) in excluded.excluded
        assert not any(entry[2] == group for entry in excluded.groups)
        assert not regroup.needs_regroup(source, threshold_bytes=estimate)
        assert regroup.needs_regroup(source, threshold_bytes=estimate - 1)


def test_estimate_is_never_below_the_readers(regroup, tmp_path):
    source = tmp_path / UNI
    source.parent.mkdir(parents=True)
    table = pa.table({"freq": ["D"], "target": [[[1.0]]], "covariates": [[[2.0]]]})
    pq.write_table(table, source)

    reader_threshold = 88
    admitted = CorpusIndex.build(tmp_path, max_group_bytes=reader_threshold)
    excluded = CorpusIndex.build(tmp_path, max_group_bytes=reader_threshold - 1)
    assert admitted.groups == (("univariate", UNI, 0, 1),)
    assert not excluded.groups
    assert excluded.excluded == ((UNI, 0, "numeric estimate 88 exceeds max_group_bytes=87"),)
    assert regroup.group_estimate(pq.read_metadata(source), 0) >= reader_threshold


def _float_bits(array):
    width = array.type.bit_width // 8
    raw = memoryview(array.buffers()[1])
    return bytes(raw[array.offset * width : (array.offset + len(array)) * width])


def test_rows_are_identical_and_groups_are_bounded(regroup, tmp_path):
    prefix = np.array([0x7FF8000000000042, 0x8000000000000000], dtype=np.uint64).view(np.float64)
    targets = [[np.r_[prefix, np.arange(i % 8 + 1) + i]] for i in range(50)]
    targets[0] = None
    targets[1] = [[], None]
    source = write_rows(tmp_path / "before.parquet", targets, row_group_size=50)
    target = tmp_path / "after.parquet"
    record = regroup.regroup_shard(source, target, group_bytes=900, group_rows=50)
    metadata = pq.read_metadata(target)
    estimates = [regroup.group_estimate(metadata, i) for i in range(metadata.num_row_groups)]
    assert len(estimates) > 1
    assert max(estimates) <= 900
    before, after = pq.read_table(source), pq.read_table(target)
    assert before.schema.equals(after.schema, check_metadata=True)
    assert before.num_rows == after.num_rows == 50
    for original, rewritten in zip(before.to_pylist(), after.to_pylist(), strict=True):
        np.testing.assert_equal(original, rewritten)
    for i in range(2, 50):
        original = before["target"][i].values[0].values
        rewritten = after["target"][i].values[0].values
        assert _float_bits(original)[:16] == prefix.tobytes()
        assert _float_bits(original) == _float_bits(rewritten)

    with pq.ParquetFile(source) as parquet:
        expected_digest = hashlib.sha256()
        for batch in parquet.iter_batches(batch_size=1):
            expected_digest.update(regroup.row_digest(batch))
    assert record == dict(
        rows=50,
        source_groups=1,
        output_groups=len(estimates),
        largest_output_estimate=max(estimates),
        oversized_rows=0,
        rows_sha256=expected_digest.hexdigest(),
        source_sha256=_sha256(source),
        source_bytes=source.stat().st_size,
        output_sha256=_sha256(target),
        output_bytes=target.stat().st_size,
    )


@pytest.mark.parametrize("kind", ["univariate", "covariates"])
def test_univariate_and_covariate_shards(regroup, tmp_path, kind):
    parent, out = tmp_path / "parent", tmp_path / "out"
    kwargs = {"targets": [[[i, i * 2]] for i in range(5)]}
    relative = UNI
    if kind == "covariates":
        relative = COV
        kwargs = dict(
            targets=[[i, i * 2] for i in range(5)],
            covariates=[[[-i, -2 * i]] for i in range(5)],
            covariate_names=[["temp"], [None], ["wet"], [""], ["\N{SNOWMAN}"]],
        )
    source = write_rows(
        parent / relative,
        kind=kind,
        row_group_size=5,
        domains=["one", None, "", "two", "one"],
        subdomains=["a", "b", None, "d", ""],
        **kwargs,
    )
    record = regroup.regroup_tree(parent, out, threshold_bytes=10, group_bytes=480, group_rows=2)
    assert set(record["shards"]) == {relative}
    assert record["shards"][relative]["rows"] == 5
    assert pq.read_table(source).equals(pq.read_table(out / relative), check_metadata=True)
    regroup.verify_tree(parent, out, record)


def test_group_rows_bound(regroup, tmp_path):
    source = write_rows(tmp_path / "before.parquet", [[[]]] * 7, row_group_size=7)
    target = tmp_path / "after.parquet"
    regroup.regroup_shard(source, target, group_bytes=40_000, group_rows=3)
    metadata = pq.read_metadata(target)
    assert [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)] == [3, 3, 1]


def test_oversized_row_is_a_group_of_one(regroup, tmp_path):
    source = write_rows(
        tmp_path / "before.parquet", [[[1.0]], [[2.0] * 40], [[3.0]], [[4.0]]], row_group_size=4
    )
    target = tmp_path / "after.parquet"
    record = regroup.regroup_shard(source, target, group_bytes=200)
    metadata = pq.read_metadata(target)
    assert [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)] == [1, 1, 2]
    assert record["oversized_rows"] == 1
    assert record["largest_output_estimate"] == 24 * 40 + 64


@pytest.mark.parametrize("copied", [False, True], ids=["rewritten", "copied"])
def test_group_above_the_admission_limit_fails(regroup, tmp_path, monkeypatch, copied):
    source = tmp_path / "parent"
    shard = write_rows(source / UNI, [[[1.0] * 40]])
    # Exercise the footer guard with valid parameters and a small fixture.
    monkeypatch.setattr(regroup, "group_estimate", lambda *_: regroup.ADMISSION_BYTES + 1)
    with pytest.raises(ValueError, match="group 0 exceeds the admission limit"):
        if copied:
            regroup._copy_shard(shard, tmp_path / "copied.parquet")
        else:
            regroup.regroup_tree(
                source, tmp_path / "out", threshold_bytes=100, group_bytes=200, group_rows=10
            )


@pytest.mark.parametrize("parameter", ["group_bytes", "threshold_bytes"])
def test_tree_rejects_parameters_above_admission_limit(regroup, tmp_path, parameter):
    parent, out = _tree(tmp_path / "parent"), tmp_path / "out"
    limit = regroup.ADMISSION_BYTES
    parameters = {**PARAMETERS, parameter: limit + 1}

    with pytest.raises(ValueError, match=rf"{parameter}.*{limit}"):
        regroup.regroup_tree(parent, out, **parameters)
    assert not out.exists()


@pytest.mark.parametrize("parameter", ["group_bytes", "threshold_bytes"])
def test_cli_rejects_parameters_above_admission_limit(regroup, tmp_path, parameter):
    parent, out = _tree(tmp_path / "parent"), tmp_path / "out"
    limit = regroup.ADMISSION_BYTES
    result = _run_cli(parent, out, "--" + parameter.replace("_", "-"), str(limit + 1))

    assert result.returncode != 0
    assert parameter in result.stderr and str(limit) in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("column", ["target", "covariates"])
def test_fixed_size_list_is_rejected_before_output(regroup, tmp_path, column):
    parent, out = _tree(tmp_path / "parent"), tmp_path / "out"
    fixed_size = pa.list_(pa.float64(), 2)
    if column == "target":
        relative = "covariates/z/fixed-target.parquet"
        values = pa.array([[1.0, 2.0], None], type=fixed_size)
        table = pa.table({"target": values})
    else:
        relative = "covariates/z/fixed-covariates.parquet"
        values = pa.array([[[1.0, 2.0]], [None]], type=pa.list_(fixed_size))
        table = pa.table({"target": [[1.0, 2.0], [3.0, 4.0]], "covariates": values})
    source = parent / relative
    source.parent.mkdir(parents=True)
    pq.write_table(table, source)

    with pytest.raises(ValueError, match=rf"{relative}.*{column}.*fixed_size_list"):
        regroup.regroup_tree(parent, out, **PARAMETERS)
    assert not out.exists()


@pytest.mark.parametrize(
    ("field", "dtype", "left", "right"),
    [
        ("target", pa.list_(pa.float64()), [1.0], [2.0]),
        ("target", pa.large_list(pa.list_(pa.float64())), None, []),
        ("target", pa.list_(pa.float64()), [None], [0.0]),
        ("target", pa.list_(pa.float64()), [0.0], [-0.0]),
        (
            "target",
            pa.list_(pa.float64()),
            [np.uint64(0x7FF8000000000001).view(np.float64)],
            [np.uint64(0x7FF8000000000002).view(np.float64)],
        ),
        ("target", pa.list_(pa.list_(pa.float64())), [[1.0], [2.0]], [[1.0, 2.0]]),
        ("domain", pa.large_string(), "climate", "traffic"),
        ("freq", pa.string(), "D", "h"),
        ("freq", pa.string(), None, ""),
    ],
    ids=[
        "value",
        "null-list",
        "null-leaf",
        "signed-zero",
        "nan-payload",
        "shape",
        "optional-column",
        "frequency",
        "null-label",
    ],
)
def test_row_digest_sees(regroup, field, dtype, left, right):
    values = pa.array([left, right, left], type=dtype)
    batch = pa.record_batch([values], names=[field])
    assert regroup.row_digest(batch.slice(0, 1)) != regroup.row_digest(batch.slice(1, 1))
    assert regroup.row_digest(batch.slice(0, 1)) == regroup.row_digest(batch.slice(2, 1))


@pytest.mark.parametrize("width", [32, 64])
def test_row_digest_sees_nan_payload(regroup, width):
    unsigned = f"uint{width}"
    float_type = pa.float32() if width == 32 else pa.float64()
    quiet_nan = 0x7FC00000 if width == 32 else 0x7FF8000000000000
    values = pa.array(
        np.array([quiet_nan | 1, quiet_nan | 2], dtype=unsigned).view(f"float{width}")
    )
    assert values.type == float_type
    values = pa.ListArray.from_arrays([0, 1, 2], values)
    batch = pa.record_batch([values], names=["target"])
    assert regroup.row_digest(batch.slice(0, 1)) != regroup.row_digest(batch.slice(1, 1))


def test_reads_one_row_at_a_time(regroup, tmp_path, monkeypatch):
    parent, out = _tree(tmp_path / "parent"), tmp_path / "out"
    original = pq.ParquetFile
    reads = []

    class Guard:
        def __init__(self, path, *args, **kwargs):
            self.path = path
            self.inner = original(path, *args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

        def __getattr__(self, name):
            if name in {"read", "read_row_group", "read_row_groups"}:
                pytest.fail(f"unbounded read: {name}")
            return getattr(self.inner, name)

        def iter_batches(self, **kwargs):
            assert kwargs["batch_size"] == 1 and len(kwargs["row_groups"]) == 1
            reads.append((self.path, kwargs["row_groups"][0]))
            yield from self.inner.iter_batches(**kwargs)

    monkeypatch.setattr(pq, "ParquetFile", Guard)
    record = regroup.regroup_tree(parent, out, **PARAMETERS)
    assert (parent / UNI, 0) in reads
    assert (parent / COV, 0) in reads
    reads.clear()
    regroup.verify_tree(parent, out, record)
    assert (parent / UNI, 0) in reads
    assert {group for path, group in reads if path == out / UNI} == set(
        range(record["shards"][UNI]["output_groups"])
    )


@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.parametrize("compliant", [False, True])
def test_schema_metadata_and_both_nested_encodings_are_kept(regroup, tmp_path, stored, compliant):
    source, target = tmp_path / "before.parquet", tmp_path / "after.parquet"
    inner = pa.field("observation", pa.float64(), metadata={b"unit": b"test"})
    outer = pa.field("channel", pa.list_(inner))
    schema = pa.schema(
        [("target", pa.large_list(outer))], metadata={b"schema-key": b"schema-value"}
    )
    table = pa.Table.from_pylist(
        [{"target": [[1.0, None], [], None]}, {"target": None}, {"target": []}], schema=schema
    )
    with pq.ParquetWriter(
        source, schema, use_compliant_nested_type=compliant, store_schema=stored
    ) as writer:
        writer.add_key_value_metadata({b"footer-key": b"footer-value"})
        writer.write_table(table)

    regroup.regroup_shard(source, target, group_bytes=250, group_rows=1)
    with pq.ParquetFile(source) as before, pq.ParquetFile(target) as after:
        assert before.schema.column(0).path == (
            "target.list.element.list.element"
            if compliant
            else "target.list.channel.list.observation"
        )
        assert before.schema.equals(after.schema)
        assert before.schema_arrow.equals(after.schema_arrow, check_metadata=True)
        before_metadata = before.metadata.metadata
        after_metadata = after.metadata.metadata
        assert (b"ARROW:schema" in before_metadata) is stored
        assert (b"ARROW:schema" in after_metadata) is stored
        before_metadata.pop(b"ARROW:schema", None)
        after_metadata.pop(b"ARROW:schema", None)
        assert after_metadata == before_metadata
        assert after_metadata[b"footer-key"] == b"footer-value"
        assert before.read().equals(after.read(), check_metadata=True)


def test_shard_without_footer_metadata_keeps_it_absent(regroup, tmp_path):
    source, target = tmp_path / "before.parquet", tmp_path / "after.parquet"
    pq.write_table(pa.table({"target": [[1.0], [2.0]]}), source, store_schema=False)

    regroup.regroup_shard(source, target, group_rows=1)

    assert pq.read_metadata(source).metadata is None
    assert pq.read_metadata(target).metadata is None


def test_two_runs_are_byte_identical(regroup, tmp_path):
    parent = _tree(tmp_path / "parent")
    first, second = tmp_path / "first", tmp_path / "second"
    assert regroup.regroup_tree(parent, first, **PARAMETERS) == regroup.regroup_tree(
        parent, second, **PARAMETERS
    )
    for relative in (UNI, COV):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


@pytest.mark.parametrize(
    "bad_path",
    [
        None,
        pytest.param("quarantine", id="blocked-directory"),
        "symlink",
        "dangling",
        "root-link",
        "output-link",
        "existing",
        "child",
        "ancestor",
    ],
)
def test_tree_inventory(regroup, tmp_path, bad_path):
    parent, out = _tree(tmp_path / "parent"), tmp_path / "out"
    if bad_path == "quarantine":
        (parent / "univariate/quarantine-example").mkdir()
    elif bad_path == "symlink":
        (parent / "univariate/link").symlink_to(parent / "covariates")
    elif bad_path == "dangling":
        (parent / "univariate/link").symlink_to(tmp_path / "missing")
    elif bad_path == "root-link":
        alias = tmp_path / "alias"
        alias.symlink_to(parent)
        parent = alias
    elif bad_path == "output-link":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path)
        out = alias / "out"
    elif bad_path == "existing":
        out.mkdir()
    elif bad_path == "child":
        out = parent / "out"
    elif bad_path == "ancestor":
        out = tmp_path
    if bad_path:
        label = {
            "child": "overlap",
            "ancestor": "overlap",
            "existing": "exists",
            "quarantine": "quarantine",
        }.get(bad_path, "symlink")
        with pytest.raises(ValueError, match=label):
            regroup.regroup_tree(parent, out, **PARAMETERS)
        assert not (out / "regroup.json").exists()
        return

    record = regroup.regroup_tree(parent, out, **PARAMETERS)
    assert {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} == {
        UNI,
        COV,
        LICENSE,
        ARTICLES,
    }
    assert set(record) == {"parameters", "shards", "other"}
    assert record["parameters"] == PARAMETERS
    assert record["other"] == {name: _sha256(parent / name) for name in (LICENSE, ARTICLES)}
    assert set(record["shards"]) == {UNI, COV}
    assert set(record["shards"][COV]) == {
        "copied",
        "sha256",
        "bytes",
        "groups",
        "largest_estimate",
        "rows_sha256",
    }
    assert record["shards"][COV]["copied"] is True
    assert record["shards"][COV]["sha256"] == _sha256(parent / COV)
    for relative in (LICENSE, ARTICLES, COV):
        assert (out / relative).read_bytes() == (parent / relative).read_bytes()
    regroup.verify_tree(parent, out, record)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("value", "rows_sha256"),
        ("swap", "rows_sha256"),
        ("missing", "row count"),
        ("optional", "rows_sha256"),
        ("group", "group"),
        ("footer", "metadata"),
        ("license", "LICENSE.md"),
        ("copied", "validation-000.parquet"),
        ("extra", "extra.txt"),
    ],
)
def test_verify_detects(regroup, tmp_path, damage, message):
    parent, out = _tree(tmp_path / "parent"), tmp_path / "out"
    record = regroup.regroup_tree(parent, out, **PARAMETERS)
    if damage in {"license", "copied", "extra"}:
        path = {"license": LICENSE, "copied": COV, "extra": "univariate/a/extra.txt"}[damage]
        with (out / path).open("ab") as stream:
            stream.write(b"new bytes")
    else:
        path = out / UNI
        table = pq.read_table(path)
        rows = table.to_pylist()
        if damage == "value":
            rows[0]["target"][0][0] = -1.0
        elif damage == "swap":
            rows[0], rows[1] = rows[1], rows[0]
        elif damage == "missing":
            rows.pop()
        elif damage == "optional":
            rows[0]["domain"] = "changed"
        table = pa.Table.from_pylist(rows, schema=table.schema)
        with pq.ParquetWriter(path, table.schema) as writer:
            if damage == "footer":
                writer.add_key_value_metadata({b"unexpected": b"metadata"})
            writer.write_table(table, row_group_size=len(rows) if damage == "group" else 1)

    with pytest.raises(ValueError, match=message) as caught:
        regroup.verify_tree(parent, out, record)
    if damage not in {"license", "copied", "extra"}:
        assert UNI in str(caught.value)


def test_verify_accepts_the_cli_ledger_and_rejects_changes(regroup, tmp_path):
    parent, out = _tree(tmp_path / "parent"), tmp_path / "out"
    result = _run_cli(parent, out)
    assert result.returncode == 0, result.stderr
    ledger = out / "regroup.json"
    with ledger.open(encoding="utf-8") as stream:
        record = json.load(stream)

    regroup.verify_tree(parent, out, record)

    parameters = {**record["parameters"], "group_rows": record["parameters"]["group_rows"] + 1}
    ledger.write_text(json.dumps({**record, "parameters": parameters}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"regroup\.json"):
        regroup.verify_tree(parent, out, record)


def test_verify_rejects_extra_empty_directory(regroup, tmp_path):
    parent, out = _tree(tmp_path / "parent"), tmp_path / "out"
    directory = "univariate/a/empty"
    (parent / directory).mkdir()
    record = regroup.regroup_tree(parent, out, **PARAMETERS)
    (out / directory).mkdir()

    with pytest.raises(ValueError, match=directory):
        regroup.verify_tree(parent, out, record)


@pytest.mark.parametrize(
    ("section", "key", "add"),
    [
        ("parameters", "group_bytes", False),
        ("parameters", "unexpected", True),
        (None, "parameters", False),
        (None, "other", False),
        (COV, "sha256", False),
        (COV, "rows_sha256", False),
        (UNI, "output_sha256", False),
    ],
    ids=[
        "missing-parameter",
        "unknown-parameter",
        "missing-parameters-section",
        "missing-other-section",
        "missing-copied-sha256",
        "missing-copied-rows-sha256",
        "missing-rewritten-output-sha256",
    ],
)
def test_verify_rejects_malformed_record(regroup, tmp_path, section, key, add):
    parent, out = _tree(tmp_path / "parent"), tmp_path / "out"
    record = regroup.regroup_tree(parent, out, **PARAMETERS)
    fields = record
    if section in (UNI, COV):
        fields = record["shards"][section]
    elif section is not None:
        fields = record[section]
    if add:
        fields[key] = 1
    else:
        del fields[key]

    with pytest.raises(ValueError, match=key):
        regroup.verify_tree(parent, out, record)


def test_reader_admits_everything_afterwards_and_items_keep_their_names(regroup, tmp_path):
    parent, out = tmp_path / "parent", tmp_path / "out"
    a = np.arange(40.0)
    write_rows(parent / UNI, [[a + i, a - i] for i in range(4)], row_group_size=4)
    write_rows(
        parent / COV,
        [a * (i + 1) for i in range(4)],
        kind="covariates",
        covariates=[[a + i] for i in range(4)],
        freqs=["D", "h", "ME", "W-MON"],
        row_group_size=4,
    )
    bound = 4300
    before = CorpusIndex.build(parent, max_group_bytes=bound)
    assert before.rows == 0
    assert {(file, group) for file, group, _ in before.excluded} == {(UNI, 0), (COV, 0)}
    record = regroup.regroup_tree(
        parent, out, threshold_bytes=bound, group_bytes=bound, group_rows=3
    )
    regroup.verify_tree(parent, out, record)
    after = CorpusIndex.build(out, max_group_bytes=bound)
    assert not after.excluded and after.rows == 8

    class DrawRow:
        def __init__(self, row):
            self.draws = iter([row, 0])

        def integers(self, high):
            value = next(self.draws)
            assert 0 <= value < high
            return value

        def choice(self, *args, **kwargs):
            return np.random.default_rng(0).choice(*args, **kwargs)

    source = CorpusSource(SourceConfig("corpus", 1.0), root=out)
    expected = [
        (relative, file_row, row)
        for relative in sorted([UNI, COV])
        for file_row, row in enumerate(pq.read_table(parent / relative).to_pylist())
    ]
    cache = RowCache()
    for index, (relative, file_row, row) in enumerate(expected):
        values = read_values(after, index, cache)
        assert (values["file"], values["row"], values["freq"]) == (relative, file_row, row["freq"])
        targets = [row["target"]] if relative == COV else row["target"]
        np.testing.assert_array_equal(values["targets"], targets)
        covariates = row.get("covariates", np.empty((0, len(targets[0]))))
        np.testing.assert_array_equal(values["covariates"], covariates)
        panel, evidence = source.panel(DrawRow(index))
        assert panel.items[0].item_id == f"{relative}:{file_row}"
        assert (evidence["file"], evidence["row"]) == (relative, file_row)
