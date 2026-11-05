"""Rebuild the score-C regression sample from the offline matcher-v2 work dir.

Usage: python tests/fixtures/audit_info/build_fixture.py /path/to/matcher-v2/work
No corpus or Hub access. Expected bits are copied from metrics.json, never
computed by the port under test. Record IDs identify the memo's census rows.
The sorted union contains all boomlet ramps/near-ramps, 48 open candidates,
32 longest and 32 highest scores, and up to 8 rows per (task, descriptive class).
Evenly spaced census records fill the sample to 360. Output is compact JSONL.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def build(source: Path, destination: Path) -> None:
    metrics = json.loads((source / "metrics.json").read_text())
    by_id = {row["record_id"]: row for row in metrics}
    selected = {row["record_id"] for row in metrics if row["task"] == "boomlet_1975"}
    candidates = [
        json.loads(line)["record_id"]
        for line in (source / "candidates.jsonl").read_text().splitlines()
    ]
    selected.update(sorted(candidates)[:48])
    for field in ("n", "bits_c"):
        selected.update(
            row["record_id"]
            for row in sorted(metrics, key=lambda r: (-r[field], r["record_id"]))[:32]
        )
    strata = defaultdict(list)
    for row in metrics:
        strata[row["task"], row["cls"]].append(row["record_id"])
    for ids in strata.values():
        selected.update(sorted(ids)[:8])
    ordered = sorted(by_id)
    # Coprime stride gives a deterministic sample over the full census.
    for i in range(len(ordered)):
        if len(selected) >= 360:
            break
        selected.add(ordered[(i * 113) % len(ordered)])
    values = {}
    for path in sorted((source / "values").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["record_id"] in selected:
                values[row["record_id"]] = row["values"]
    rows = [
        {
            **{
                key: by_id[i][key]
                for key in ("record_id", "task", "cls", "bits_c", "ramp", "n", "k", "runs")
            },
            "candidate": i in candidates,
            "values": values[i],
        }
        for i in sorted(selected)
    ]
    destination.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    assert destination.stat().st_size <= 300_000
    assert len(rows) >= 300
    assert {(r["task"], r["cls"]) for r in rows} == set(strata)
    print(
        f"{len(rows)} records, {destination.stat().st_size} bytes, {len(strata)} task/class strata"
    )


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(__file__).with_name("segments.jsonl"))
