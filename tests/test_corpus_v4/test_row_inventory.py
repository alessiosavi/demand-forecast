"""Row evidence and rule provenance from tiny released-shard fixtures."""

import csv
import hashlib
import json
import struct

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from scripts.corpus_v4 import row_inventory as inventory  # noqa: E402

FOLDER = "covariates/source"


def shard(path, *, names=True, nested=False, dictionary=False, row_group_size=1):
    data = {
        "item_id": ["ERCOT", "PJM", "other"],
        "source": ["load", "load", "independent"],
        "freq": ["h"] * 3,
        "target": [[1.0, None, 3.0], [4.0, float("inf"), 6.0], [7.0, 8.0, 9.0]],
        "domain": ["energy"] * 3,
        "station": [17, 18, 19],
        "nested_metadata": [["ignored"]] * 3,
    }
    if nested:
        data["target"] = [[t, t] for t in data["target"]]
    else:
        data["covariates"] = [[[10.0, 11.0, 12.0]]] * 3
    if names:
        data["covariate_names"] = [["temperature"]] * 3
    table = pa.table(data)
    if dictionary:
        for name in ("item_id", "source", "freq", "domain"):
            table = table.set_column(
                table.schema.get_field_index(name), name, table[name].dictionary_encode()
            )
    pq.write_table(table, path, row_group_size=row_group_size)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_inventory_values_metadata_hashes_and_determinism(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    shard(folder / "validation-0.parquet", names=False)
    shard(folder / "train-0.parquet")
    output = tmp_path / "out"
    inventory.build_inventory(folder, output, folder_name="quarantine/covariates/source")
    rows = read_csv(output / "inventory.csv")
    assert len(rows) == 6
    assert [(r["shard"], r["row_index"]) for r in rows] == [
        (s, str(i)) for s in ("train-0.parquet", "validation-0.parquet") for i in range(3)
    ]
    row = rows[0]
    assert row["folder"] == "quarantine/covariates/source"
    assert (row["source"], row["item_id"], row["freq"]) == ("load", "ERCOT", "h")
    assert json.loads(row["target_shape"]) == [3]
    assert row["observed_target_values"] == "2"
    assert rows[1]["observed_target_values"] == "2"
    assert row["covariate_channels"] == "1"
    assert json.loads(row["covariate_names"]) == ["temperature"]
    assert rows[3]["covariate_names"] == ""
    assert (row["station"], row["domain"]) == ("17", "energy")
    assert "nested_metadata" not in row
    assert row["target_dtype"] == "double"
    assert row["target_sha256"] == expected_target_hash(
        "double", [3], b"\x01\x00\x01", struct.pack("<ddd", 1.0, 0.0, 3.0)
    )
    summary = json.loads((output / "inventory.json").read_text())
    assert summary["rows"] == 6
    assert summary["observed_target_values"] == 14
    assert summary["per_source_rows"] == {"load": 4, "independent": 2}
    assert "nested_metadata" in summary["schema_columns"]
    assert summary["scalar_columns"] == ["domain", "station"]
    assert summary["target_encoding_version"] == "arrow-target-v1"
    assert (
        summary["csv_sha256"] == hashlib.sha256((output / "inventory.csv").read_bytes()).hexdigest()
    )
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    inventory.build_inventory(folder, output, folder_name="quarantine/covariates/source")
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize("row_group_size", [1, 3])
def test_without_names_and_multivariate_target_streams_row_groups(
    tmp_path, monkeypatch, row_group_size
):
    folder = tmp_path / "source"
    folder.mkdir()
    shard(folder / "train-0.parquet", names=False, nested=True, row_group_size=row_group_size)
    original = pq.ParquetFile
    groups = []

    class Guard:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

        def __getattr__(self, name):
            if name in {"read", "read_row_groups", "read_row_group"}:
                pytest.fail("whole-file/group reads bypass bounded row iteration")
            return getattr(self.inner, name)

        def iter_batches(self, **kwargs):
            assert len(kwargs["row_groups"]) == 1
            assert kwargs["batch_size"] == 1
            groups.extend(kwargs["row_groups"])
            return self.inner.iter_batches(**kwargs)

    monkeypatch.setattr(pq, "ParquetFile", Guard)
    inventory.build_inventory(folder, tmp_path / "out", folder_name=FOLDER)
    rows = read_csv(tmp_path / "out/inventory.csv")
    assert "covariate_names" not in rows[0]
    assert json.loads(rows[0]["target_shape"]) == [2, 3]
    assert rows[0]["observed_target_values"] == "4"
    assert rows[0]["covariate_channels"] == "0"
    assert groups == ([0, 1, 2] if row_group_size == 1 else [0])
    assert [row["row_index"] for row in rows] == ["0", "1", "2"]


def rules():
    return {
        "folder": FOLDER,
        "default": "keep",
        "rules": [
            {"match": {"item_id": "^ERCOT$"}, "action": "exclude", "reason": "Texas"},
            {"match": {"source": "load"}, "action": "keep", "reason": "fallback"},
        ],
    }


def record(**changes):
    row = dict.fromkeys(inventory.BASE_FIELDS, "present")
    row.update(
        folder=FOLDER, shard="train-0.parquet", row_index="0", item_id="ERCOT", source="load"
    )
    row.update(changes)
    return row


def decide(rows, document, *, fields=inventory.BASE_FIELDS):
    return list(inventory.apply_rules(rows, document, fields=fields, folder_name=FOLDER))


def test_rules_first_match_default_and_all_fields():
    rows = [
        record(item_id=item, source=source)
        for item, source in [("ERCOT", "load"), ("PJM", "load"), ("other", "independent")]
    ]
    result = decide(iter(rows), rules())
    assert [(r["action"], r["rule"]) for r in result] == [
        ("exclude", "0"),
        ("keep", "1"),
        ("keep", "default"),
    ]
    assert result[0]["reason"] == "Texas"
    assert "action" not in rows[0]
    rule = {
        "folder": FOLDER,
        "default": "keep",
        "rules": [
            {
                "match": {"item_id": "ERCOT", "source": "independent"},
                "action": "exclude",
                "reason": "both required",
            }
        ],
    }
    assert decide(rows, rule)[0]["action"] == "keep"


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"rules": []},
        {"default": "drop"},
        {"default": "keep", "rules": {}},
        {
            "default": "keep",
            "rules": [{"match": {"unknown": ".*"}, "action": "keep", "reason": "x"}],
        },
        {
            "default": "keep",
            "rules": [{"match": {"item_id": "["}, "action": "keep", "reason": "x"}],
        },
        {"default": "keep", "rules": [{"match": {}, "action": "keep", "reason": "x"}]},
        {
            "default": "keep",
            "rules": [{"match": {"item_id": ".*"}, "action": "drop", "reason": "x"}],
        },
        {"default": "keep", "rules": [{"match": {"item_id": ".*"}, "action": "keep"}]},
        {"default": "keep", "rules": [{"match": {"item_id": 2}, "action": "keep", "reason": "x"}]},
    ],
)
def test_invalid_rule_documents_refused(document):
    document = {"folder": FOLDER, **document}
    with pytest.raises(ValueError):
        decide([record()], document)


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_cli_manifest_provenance_and_determinism(tmp_path, extension):
    folder = tmp_path / "source"
    folder.mkdir()
    shard(folder / "train-0.parquet")
    rule_path = tmp_path / ("rules." + extension)
    if extension == "yaml":
        yaml = pytest.importorskip("yaml")
        rule_path.write_text(yaml.safe_dump(rules()))
    else:
        rule_path.write_text(json.dumps(rules()))
    out = tmp_path / "out"
    args = [
        "--folder",
        str(folder),
        "--folder-name",
        FOLDER,
        "--output",
        str(out),
        "--rules",
        str(rule_path),
    ]
    assert inventory.main(args) == 0
    rows = read_csv(out / "row-manifest.csv")
    assert len(rows) == 3
    summary = json.loads((out / "row-manifest.json").read_text())
    assert summary["folder"] == FOLDER
    assert summary["counts_by_action"] == {"exclude": 1, "keep": 2}
    assert summary["counts_by_rule"] == {"0": 1, "1": 1, "default": 1}
    for field, path in [
        ("inventory_sha256", out / "inventory.csv"),
        ("rules_sha256", rule_path),
        ("csv_sha256", out / "row-manifest.csv"),
    ]:
        assert summary[field] == hashlib.sha256(path.read_bytes()).hexdigest()
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert inventory.main(args) == 0
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


@pytest.mark.parametrize(
    "dtype,values,fmt,label",
    [
        (pa.float32(), [1.0, None, 3.0], "<fff", "float"),
        (pa.int64(), [2**60 + 1, 2**60 + 2], "<qq", "int64"),
    ],
)
def test_hash_preserves_numeric_storage_and_precision(tmp_path, dtype, values, fmt, label):
    folder = tmp_path / "source"
    folder.mkdir()
    table = pa.table(
        {
            "item_id": ["a"],
            "source": ["load"],
            "freq": ["h"],
            "target": pa.array([values], type=pa.list_(dtype)),
        }
    )
    pq.write_table(table, folder / "train-0.parquet")
    inventory.build_inventory(folder, tmp_path / "out", folder_name=FOLDER)
    row = read_csv(tmp_path / "out/inventory.csv")[0]
    expected = [0 if v is None else v for v in values]
    assert row["target_sha256"] == expected_target_hash(
        label, [len(values)], bytes(v is not None for v in values), struct.pack(fmt, *expected)
    )
    assert row["target_dtype"] == label


def test_empty_inventory_still_validates_rules_against_header(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    schema = pa.schema(
        [
            ("item_id", pa.string()),
            ("source", pa.string()),
            ("freq", pa.string()),
            ("target", pa.list_(pa.float32())),
        ]
    )
    pq.write_table(pa.Table.from_batches([], schema=schema), folder / "train-0.parquet")
    output = tmp_path / "out"
    summary = inventory.build_inventory(folder, output, folder_name=FOLDER)
    assert summary["rows"] == 0
    rule_path = tmp_path / "rules.json"
    rule_path.write_text(json.dumps(rules()))
    manifest = inventory.write_manifest(output / "inventory.csv", rule_path, output)
    assert manifest["rows"] == 0
    rule_path.write_text(
        json.dumps(
            {
                "folder": FOLDER,
                "default": "keep",
                "rules": [{"match": {"missing": ".*"}, "action": "exclude", "reason": "invalid"}],
            }
        )
    )
    with pytest.raises(ValueError, match="unknown"):
        inventory.write_manifest(output / "inventory.csv", rule_path, output)


def test_scalar_metadata_collisions_are_preserved(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    pq.write_table(
        pa.table(
            {
                "item_id": ["one"],
                "source": ["load"],
                "freq": ["h"],
                "target": [[1.0]],
                "folder": ["original"],
                "action": ["original-action"],
                "scalar.folder": ["another"],
            }
        ),
        folder / "train-0.parquet",
    )
    summary = inventory.build_inventory(folder, tmp_path / "out", folder_name=FOLDER)
    row = read_csv(tmp_path / "out/inventory.csv")[0]
    assert row["folder"] == FOLDER
    assert row[summary["scalar_fields"]["folder"]] == "original"
    assert row[summary["scalar_fields"]["scalar.folder"]] == "another"
    assert row[summary["scalar_fields"]["action"]] == "original-action"


def expected_target_hash(dtype, shape, validity, raw):
    header = (
        json.dumps(
            {"encoding": "arrow-target-v1", "dtype": dtype, "shape": shape},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    return hashlib.sha256(header + validity + raw).hexdigest()


@pytest.mark.parametrize(
    "left_type,left,right_type,right",
    [
        (pa.list_(pa.float64()), [1, 2], pa.list_(pa.list_(pa.float64())), [[1, 2]]),
        (pa.list_(pa.int64()), [0], pa.list_(pa.float64()), [0]),
        (pa.list_(pa.int64()), [2**60 + 1, None], pa.list_(pa.int64()), [2**60 + 2, None]),
        (pa.list_(pa.float64()), [None, 1], pa.list_(pa.float64()), [float("nan"), 1]),
        (pa.list_(pa.int64()), [None, 0], pa.list_(pa.int64()), [0, None]),
    ],
    ids=["shape", "dtype", "nullable-integer-precision", "null-vs-nan", "validity-position"],
)
def test_target_fingerprint_collision_regressions(left_type, left, right_type, right):
    lhs = inventory.target_evidence(pa.array([left], type=left_type)[0])
    rhs = inventory.target_evidence(pa.array([right], type=right_type)[0])
    assert lhs[3] != rhs[3]
    if left_type == pa.list_(pa.int64()):
        assert lhs[2] == "int64"


def test_target_hash_ignores_slice_offsets_and_null_slot_bytes():
    values = pa.Array.from_buffers(
        pa.int64(), 3, [pa.py_buffer(b"\x05"), pa.py_buffer(struct.pack("<qqq", 7, 99, 2**60 + 1))]
    )
    sliced = pa.ListArray.from_arrays(pa.array([1, 3], type=pa.int32()), values)[0]
    plain = pa.array([[None, 2**60 + 1]], type=pa.list_(pa.int64()))[0]
    assert inventory.target_evidence(sliced) == inventory.target_evidence(plain)


def test_null_typed_target_has_stored_dtype_and_validity():
    shape, observed, dtype, sha = inventory.target_evidence(pa.array([[None, None]])[0])
    assert (shape, observed, dtype) == ([2], 0, "null")
    assert sha == expected_target_hash("null", [2], b"\x00\x00", b"")


def test_dictionary_encoded_strings(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    shard(folder / "train-0.parquet", dictionary=True, row_group_size=3)
    assert pa.types.is_dictionary(pq.read_schema(folder / "train-0.parquet").field("item_id").type)
    summary = inventory.build_inventory(folder, tmp_path / "out", folder_name=FOLDER)
    rows = read_csv(tmp_path / "out/inventory.csv")
    assert summary["rows"] == 3
    assert [(r["item_id"], r["domain"], r["row_index"]) for r in rows] == [
        ("ERCOT", "energy", "0"),
        ("PJM", "energy", "1"),
        ("other", "energy", "2"),
    ]


@pytest.mark.parametrize("field", ["item_id", "source", "freq"])
@pytest.mark.parametrize("value", ["absent", None, ""])
@pytest.mark.parametrize("policy", [None, "refuse", "exclude"])
def test_missing_values_never_fall_through_to_keep(field, value, policy):
    good, bad = record(item_id="other", source="independent"), record(row_index="1")
    if value == "absent":
        bad.pop(field)
    else:
        bad[field] = value
    document = rules()
    # Missing fields in later rules must be checked even when the first rule keeps.
    document["rules"].insert(
        0, {"match": {"folder": ".*"}, "action": "keep", "reason": "early keep"}
    )
    document["rules"].append({"match": {"freq": "h"}, "action": "keep", "reason": "frequency"})
    if policy is not None:
        document["missing"] = policy
    if policy == "exclude":
        result = decide([good, bad], document)
        assert result[1]["action"] == "exclude"
        assert result[1]["rule"] == "missing"
        assert field in result[1]["reason"]
    else:
        with pytest.raises(ValueError) as error:
            decide([good, bad], document)
        assert field in str(error.value)
        assert all(part in str(error.value) for part in (FOLDER, "train-0.parquet", "1"))


@pytest.mark.parametrize("policy", ["keep", "default", None, {}, True])
def test_invalid_missing_policy_refused(policy):
    document = {**rules(), "missing": policy}
    with pytest.raises(ValueError, match="missing"):
        decide([record()], document)


@pytest.mark.parametrize("policy", [None, "exclude"])
def test_missing_unreferenced_column_validates_entire_record(policy):
    row = record()
    row.pop("target_dtype")
    document = rules()
    if policy:
        document["missing"] = policy
        assert decide([row], document)[0]["action"] == "exclude"
    else:
        with pytest.raises(ValueError, match="target_dtype"):
            decide([row], document)


@pytest.mark.parametrize("mutation", ["short", "null", "absent-column", "extra"])
@pytest.mark.parametrize("policy", [None, "exclude"])
def test_manifest_refuses_malformed_or_missing_values(tmp_path, mutation, policy):
    folder = tmp_path / "source"
    folder.mkdir()
    shard(folder / "train-0.parquet")
    output = tmp_path / "inventory"
    inventory.build_inventory(folder, output, folder_name=FOLDER)
    path = output / "inventory.csv"
    rows = read_csv(path)
    fields = list(rows[0])
    # Put item_id last so a short record specifically loses the match value.
    fields.remove("item_id")
    fields.append("item_id")
    header = json.loads((output / "inventory.json").read_text())
    header["csv_fields"] = fields.copy()
    (output / "inventory.json").write_text(json.dumps(header))
    if mutation == "absent-column":
        fields.remove("item_id")
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        for index, row in enumerate(rows):
            values = [row[f] for f in fields]
            if index == 1:
                if mutation == "short":
                    values.pop()
                elif mutation == "null":
                    values[-1] = ""
                elif mutation == "extra":
                    values.append("surplus")
            writer.writerow(values)
    rule_path = tmp_path / "rules.json"
    document = rules()
    if policy:
        document["missing"] = policy
    rule_path.write_text(json.dumps(document))
    destination = tmp_path / "manifest"
    if policy == "exclude" and mutation in ("short", "null"):
        inventory.write_manifest(path, rule_path, destination)
        assert read_csv(destination / "row-manifest.csv")[1]["action"] == "exclude"
    else:
        with pytest.raises(ValueError) as error:
            inventory.write_manifest(path, rule_path, destination)
        if mutation != "absent-column":
            assert all(part in str(error.value) for part in (FOLDER, "train-0.parquet", "1"))
        assert not (destination / "row-manifest.json").exists()
        assert not (destination / "row-manifest.csv").exists()


@pytest.mark.parametrize(
    "name", [None, "source", "/covariates/source", "covariates/../source", "covariates/source/"]
)
def test_canonical_folder_identity_is_required(tmp_path, name):
    with pytest.raises(ValueError, match="folder"):
        inventory.build_inventory(tmp_path, tmp_path / "out", folder_name=name)


def test_module_and_cli_require_folder_identity(tmp_path):
    with pytest.raises(TypeError, match="folder_name"):
        inventory.build_inventory(tmp_path, tmp_path / "out")
    with pytest.raises(SystemExit) as error:
        inventory.main(["--folder", str(tmp_path), "--output", str(tmp_path / "out")])
    assert error.value.code == 2


def test_equal_basenames_have_distinct_row_identities(tmp_path):
    identities = []
    for name in ("covariates/same", "quarantine/covariates/same"):
        folder = tmp_path / name
        folder.mkdir(parents=True)
        shard(folder / "train-0.parquet")
        output = tmp_path / "out" / name
        inventory.build_inventory(folder, output, folder_name=name)
        row = read_csv(output / "inventory.csv")[0]
        identities.append((row["folder"], row["shard"], row["row_index"]))
    assert identities[0] != identities[1]
    assert identities[0][0] == "covariates/same"
    assert identities[1][0] == "quarantine/covariates/same"


@pytest.mark.parametrize("folder", [None, "covariates/other"])
def test_rules_require_matching_folder_binding(folder):
    document = rules()
    if folder is None:
        document.pop("folder")
    else:
        document["folder"] = folder
    with pytest.raises(ValueError, match="folder"):
        decide([record()], document)


def test_manifest_refuses_wrong_folder_even_when_inventory_is_empty(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    shard(folder / "train-0.parquet")
    output = tmp_path / "out"
    inventory.build_inventory(folder, output, folder_name=FOLDER)
    rule_path = tmp_path / "rules.json"
    rule_path.write_text(json.dumps({**rules(), "folder": "covariates/other"}))
    path = output / "inventory.csv"
    for empty in (False, True):
        if empty:
            path.write_text(path.read_text().splitlines()[0] + "\n")
        with pytest.raises(ValueError, match="folder"):
            inventory.write_manifest(path, rule_path, output)
        assert not (output / "row-manifest.json").exists()


def test_mixed_folder_rows_are_refused():
    with pytest.raises(ValueError, match="folder"):
        decide([record(), record(folder="covariates/other", row_index="1")], rules())


def test_rule_patterns_use_regex_substring_search():
    row = record(item_id="prefix-ERCOT-suffix")
    document = rules()
    assert decide([row], document)[0]["action"] == "keep"
    document["rules"][0]["match"]["item_id"] = "ERCOT"
    assert decide([row], document)[0]["action"] == "exclude"


@pytest.mark.parametrize("mutation", ["truncated", "reordered"])
def test_manifest_hash_binds_exact_inventory_order_and_length(tmp_path, mutation):
    folder = tmp_path / "source"
    folder.mkdir()
    shard(folder / "train-0.parquet")
    output = tmp_path / "out"
    inventory.build_inventory(folder, output, folder_name=FOLDER)
    path = output / "inventory.csv"
    rule_path = tmp_path / "rules.json"
    rule_path.write_text(json.dumps(rules()))
    original = inventory.write_manifest(path, rule_path, output)
    rows = read_csv(path)
    fields = list(rows[0])
    rows = rows[:-1] if mutation == "truncated" else list(reversed(rows))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    modified = inventory.write_manifest(path, rule_path, output)
    assert modified["inventory_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert modified["inventory_sha256"] != original["inventory_sha256"]
