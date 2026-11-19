"""Native v4.1 numeric rows, metadata identity and a byte-bounded group cache.

Without a bucket manifest the digest identifies Parquet metadata, not all data
pages. Only ``freq`` is decoded at index construction; payload reads decode
numeric columns alone. IDs and other descriptive columns are never requested.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections import Counter, OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np

from demand_forecast.data.frequency import to_offset


def listing_digest(objects: Iterable[tuple[str, int, str]]) -> str:
    """Hash the bucket's sorted relative-path, byte-size and checksum listing."""
    digest = hashlib.sha256()
    for path, size, checksum in sorted(objects, key=lambda entry: entry[0]):
        digest.update(f"{path}\t{size}\t{checksum}\n".encode())
    return digest.hexdigest()


def _refuse_quarantine(path: Path) -> None:
    if "quarantine" in str(path).lower() or "quarantine" in str(path.resolve()).lower():
        raise ValueError(f"quarantine is not training data: {path}")


def _footer_digest(path: Path) -> str:
    with path.open("rb") as stream:
        stream.seek(-8, 2)
        trailer = stream.read(8)
        if trailer[4:] != b"PAR1":
            raise ValueError(f"invalid parquet footer: {path}")
        length = int.from_bytes(trailer[:4], "little")
        if length > path.stat().st_size - 12:
            raise ValueError(f"invalid parquet footer length: {path}")
        stream.seek(-8 - length, 2)
        return hashlib.sha256(stream.read(length)).hexdigest()


@dataclass(frozen=True)
class _Frequencies:
    values: tuple[str | None, ...]
    codes: bytes = b""
    width: int = 0

    def at(self, row: int) -> str | None:
        code = int.from_bytes(self.codes[row * self.width : (row + 1) * self.width], "little")
        return self.values[code]


def _frequencies(parquet, group: int, unparsed_frequencies: Counter[str] | None) -> _Frequencies:
    column = parquet.read_row_group(group, columns=["freq"], use_threads=False)["freq"]
    encoded = column.dictionary_encode().combine_chunks()
    # Only the dictionary crosses into Python; per-row aliases remain Arrow codes.
    aliases = encoded.dictionary.to_pylist()
    parsed_aliases = []
    invalid = {}
    for code, label in enumerate(aliases):
        freq = None
        try:
            if isinstance(label, str) and label and to_offset(label).n > 0:
                freq = label
        except (TypeError, ValueError, AttributeError, OverflowError):
            pass
        parsed_aliases.append(freq)
        if freq is None:
            invalid[code] = "null" if label is None else str(label)

    indices = encoded.indices
    null_count = encoded.null_count
    if unparsed_frequencies is not None:
        if invalid:
            # Reduce the row codes in Arrow before materialising the distinct counts.
            for entry in indices.value_counts().to_pylist():
                label = invalid.get(entry["values"])
                if label is not None:
                    unparsed_frequencies[label] += entry["counts"]
        if null_count:
            unparsed_frequencies["null"] += null_count

    values = tuple(sorted(freq for freq in parsed_aliases if freq is not None))
    if invalid or null_count:
        values += (None,)
    if len(values) <= 1:
        return _Frequencies(values)
    lookup = {value: i for i, value in enumerate(values)}
    width = 1 if len(values) <= 256 else 2 if len(values) <= 65536 else 4
    if null_count:
        indices = indices.fill_null(len(parsed_aliases))
        parsed_aliases.append(None)
    dictionary_codes = np.fromiter((lookup[freq] for freq in parsed_aliases), dtype=f"<u{width}")
    codes = dictionary_codes[indices.to_numpy(zero_copy_only=False)]
    return _Frequencies(values, codes.tobytes(), width)


def _check_numeric_schema(parquet, kind: str, relative: str) -> tuple[str, ...]:
    import pyarrow as pa

    columns = ("target", "covariates") if kind == "covariates" else ("target",)
    schema = parquet.schema_arrow
    required = {"freq", *columns}
    if not required <= set(schema.names):
        raise ValueError(
            f"missing native columns in {relative}: {sorted(required - set(schema.names))}"
        )
    for name in columns:
        dtype = schema.field(name).type
        depth = 1 if kind == "covariates" and name == "target" else 2
        for _ in range(depth):
            if not (
                pa.types.is_list(dtype)
                or pa.types.is_large_list(dtype)
                or pa.types.is_fixed_size_list(dtype)
            ):
                raise ValueError(f"{relative}: {name} must be nested numeric lists")
            dtype = dtype.value_type
        if not (
            pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_null(dtype)
        ):
            raise ValueError(f"{relative}: {name} must contain numeric values")
    return columns


@dataclass(frozen=True)
class CorpusFolder:
    name: str
    first_row: int
    rows: int
    values: int


@dataclass(frozen=True)
class CorpusIndex:
    """Index of admitted corpus rows."""

    root: Path
    groups: tuple[tuple[str, str, int, int], ...]
    excluded: tuple[tuple[str, int, str], ...]
    digest: str
    _frequencies: tuple[_Frequencies, ...] = field(default=(), repr=False)
    _ends: tuple[int, ...] = field(default=(), repr=False)
    _file_starts: tuple[int, ...] = field(default=(), repr=False)
    unparsed_frequencies: dict[str, int] = field(default_factory=dict, hash=False)
    """Unparsed label counts for admitted rows, with ``rows`` as the denominator."""
    folders: tuple[CorpusFolder, ...] = ()

    @classmethod
    def build(cls, root: Path, *, max_group_bytes: int = 256 << 20) -> CorpusIndex:
        import pyarrow.parquet as pq

        root = Path(root)
        _refuse_quarantine(root)
        if max_group_bytes <= 0:
            raise ValueError("max_group_bytes must be positive")
        if not root.is_dir():
            raise ValueError(f"corpus root does not exist: {root}")
        paths = sorted(
            path
            for kind in ("univariate", "covariates")
            for path in (root / kind).glob("*/*.parquet")
        )
        for path in paths:
            _refuse_quarantine(path)

        manifest = root / "manifest.json"
        if manifest.exists():
            objects = json.loads(manifest.read_text(encoding="utf-8"))["objects"]
            listed = set()
            for relative, size, _ in objects:
                name = PurePosixPath(relative)
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError(f"invalid manifest path: {relative}")
                path = root / relative
                _refuse_quarantine(path)
                if name.suffix != ".parquet":
                    continue
                if relative in listed:
                    raise ValueError(f"duplicate manifest parquet: {relative}")
                listed.add(relative)
                if not path.is_file():
                    raise ValueError(f"missing manifest parquet: {relative}")
                if path.stat().st_size != size:
                    raise ValueError(f"wrong size for manifest parquet: {relative}")
            actual = {path.relative_to(root).as_posix() for path in paths}
            # Check every parquet beneath the two trees, even at an unsupported depth.
            discovered = {
                path.relative_to(root).as_posix()
                for kind in ("univariate", "covariates")
                for path in (root / kind).rglob("*.parquet")
            }
            if discovered - listed:
                raise ValueError(f"unlisted parquet: {sorted(discovered - listed)}")
            if actual - listed:
                raise ValueError(f"unlisted parquet: {sorted(actual - listed)}")
            digest = listing_digest(objects)
        else:
            digest = listing_digest(
                (path.relative_to(root).as_posix(), path.stat().st_size, _footer_digest(path))
                for path in paths
            )

        groups, excluded, frequencies, ends, starts = [], [], [], [], []
        unparsed_frequencies: Counter[str] = Counter()
        folders: dict[str, CorpusFolder] = {}
        rows = 0
        for path in paths:
            relative_path = path.relative_to(root)
            relative = relative_path.as_posix()
            kind = relative_path.parts[0]
            folder = relative_path.parent.as_posix()
            if not path.name.startswith(("train-", "validation-")):
                raise ValueError(f"unsupported parquet shard: {relative}")
            first_row, values = rows, 0
            with pq.ParquetFile(path, pre_buffer=False, buffer_size=64 << 10) as parquet:
                columns = _check_numeric_schema(parquet, kind, relative)
                file_start = 0
                for i in range(parquet.num_row_groups):
                    group = parquet.metadata.row_group(i)
                    num_values = sum(
                        group.column(j).num_values
                        for j in range(group.num_columns)
                        if group.column(j).path_in_schema.split(".")[0] in columns
                    )
                    estimate = 24 * num_values + 64 * group.num_rows
                    if estimate > max_group_bytes:
                        excluded.append(
                            (
                                relative,
                                i,
                                f"numeric estimate {estimate} exceeds max_group_bytes={max_group_bytes}",
                            )
                        )
                    else:
                        groups.append((kind, relative, i, group.num_rows))
                        frequencies.append(_frequencies(parquet, i, unparsed_frequencies))
                        starts.append(file_start)
                        rows += group.num_rows
                        values += num_values
                        ends.append(rows)
                    file_start += group.num_rows
            if rows > first_row:
                previous = folders.get(folder)
                if previous is not None:
                    first_row = previous.first_row
                    values += previous.values
                folders[folder] = CorpusFolder(folder, first_row, rows - first_row, values)
        return cls(
            root,
            tuple(groups),
            tuple(excluded),
            digest,
            tuple(frequencies),
            tuple(ends),
            tuple(starts),
            unparsed_frequencies=dict(unparsed_frequencies),
            folders=tuple(folders.values()),
        )

    @property
    def rows(self) -> int:
        return self._ends[-1] if self._ends else 0

    def pick(self, rng: np.random.Generator, power: float | None) -> int:
        """Propose an admitted row uniformly, or draw its folder first."""
        if power is None:
            return int(rng.integers(self.rows))
        weights = np.array([folder.values for folder in self.folders], dtype=np.float64) ** power
        total_weight = weights.sum()
        if total_weight <= 0:
            raise ValueError("corpus folders hold no values to weight")
        folder = self.folders[int(rng.choice(len(self.folders), p=weights / total_weight))]
        return int(rng.integers(folder.first_row, folder.first_row + folder.rows))


class SamplerSkip(Exception):  # noqa: N818 - public sampler contract
    """A known data limitation that a sampler can count and skip."""


class ShortRow(SamplerSkip):
    """A corpus row has fewer than 32 steps."""


class CorpusValueError(SamplerSkip):
    """A corpus row contains malformed numeric data or infinity."""


class RowCache:
    """LRU of numeric Arrow tables, charged by decoded table bytes."""

    def __init__(self, max_bytes: int = 512 << 20) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.max_bytes = max_bytes
        self.nbytes = 0
        self._groups = OrderedDict()

    def _get(self, key):
        table = self._groups.get(key)
        if table is not None:
            self._groups.move_to_end(key)
        return table

    def _put(self, key, table) -> None:
        size = table.get_total_buffer_size()
        if size > self.max_bytes:
            return
        while self._groups and self.nbytes + size > self.max_bytes:
            _, evicted = self._groups.popitem(last=False)
            self.nbytes -= evicted.get_total_buffer_size()
        self._groups[key] = table
        self.nbytes += size


def _matrix(scalar, *, nested: bool) -> np.ndarray:
    import pyarrow as pa

    if not scalar.is_valid:
        raise ValueError("null numeric row")
    array = scalar.values
    if nested:
        if array.null_count:
            raise ValueError("null numeric channel")
        lengths = np.fromiter((len(channel.values) for channel in array), dtype=np.int64)
        if not len(lengths):
            return np.empty((0, 0), dtype=np.float64)
        if not np.all(lengths == lengths[0]):
            raise ValueError("ragged numeric channels")
        if pa.types.is_fixed_size_list(array.type):
            leaf = array.values.slice(
                array.offset * array.type.list_size, len(array) * array.type.list_size
            )
        else:
            leaf = array.flatten()
    else:
        leaf = array
    values = (
        np.full(len(leaf), np.nan)
        if pa.types.is_null(leaf.type)
        # A row must not keep an evicted group's Arrow buffers alive.
        else leaf.to_numpy(zero_copy_only=False).astype(np.float64, copy=True)
    )
    return values.reshape(len(lengths), int(lengths[0])) if nested else values.reshape(1, -1)


def read_values(index: CorpusIndex, row: int, cache: RowCache) -> dict:
    """Read every variate and aligned covariate, with file-absolute row identity."""
    import pyarrow.parquet as pq

    if not 0 <= row < index.rows:
        raise IndexError(f"corpus row out of range: {row}")
    position = bisect_right(index._ends, row)
    kind, relative, group, _ = index.groups[position]
    local_row = row - (index._ends[position - 1] if position else 0)
    file_row = index._file_starts[position] + local_row
    key = (str(index.root.resolve()), index.digest, relative, group)
    table = cache._get(key)
    if table is None:
        columns = ["target", "covariates"] if kind == "covariates" else ["target"]
        with pq.ParquetFile(
            index.root / relative, pre_buffer=False, buffer_size=64 << 10
        ) as parquet:
            table = parquet.read_row_group(group, columns=columns, use_threads=False)
        cache._put(key, table)
    context = f"{relative}: row {file_row}"
    try:
        targets = _matrix(table["target"][local_row], nested=kind != "covariates")
        covariates = (
            _matrix(table["covariates"][local_row], nested=True)
            if kind == "covariates"
            else np.empty((0, targets.shape[1]), dtype=np.float64)
        )
    except ValueError as error:
        raise CorpusValueError(f"{context}: {error}") from error
    if covariates.shape == (0, 0):
        covariates = np.empty((0, targets.shape[1]), dtype=np.float64)
    if not targets.shape[0] or covariates.shape[1] != targets.shape[1]:
        raise CorpusValueError(f"invalid numeric shapes in {context}")
    if np.isinf(targets).any() or np.isinf(covariates).any():
        raise CorpusValueError(f"infinite value in {context}")
    if targets.shape[1] < 32:
        raise ShortRow(f"fewer than 32 steps in {context}")
    return {
        "targets": targets,
        "covariates": covariates,
        "freq": index._frequencies[position].at(local_row),
        "file": relative,
        "row": file_row,
    }
