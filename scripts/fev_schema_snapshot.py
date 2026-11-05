"""Regenerate pinned fev schema metadata without downloading dataset rows.

One anonymous README request contains all configs. Retry HTTP 429 sequentially
with exponential backoff; no Dataset Viewer endpoint (which follows main).
"""

import argparse
import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from demand_forecast.bench.specs import FEV_DATASETS_REVISION, fev_bench_tasks

URL = (
    f"https://huggingface.co/datasets/autogluon/fev_datasets/raw/{FEV_DATASETS_REVISION}/README.md"
)


def snapshot(card: str) -> dict:
    metadata = yaml.safe_load(card.split("---", 2)[1])
    configs = {task.dataset_config for task in fev_bench_tasks()}
    sizes = {}
    for line in card.splitlines():
        cells = [cell.strip().strip("`") for cell in line.split("|")]
        if len(cells) > 5 and cells[1] in configs:
            sizes[cells[1]] = {
                "frequency": cells[2],
                "items": int(cells[3].replace(",", "")),
                "median_length": int(cells[4].replace(",", "")),
                "observations_all_dynamic_columns": int(cells[5].replace(",", "")),
            }
    return {
        "revision": FEV_DATASETS_REVISION,
        "source": URL,
        "source_sha256": hashlib.sha256(card.encode()).hexdigest(),
        "configs": {
            info["config_name"]: {
                "features": {
                    f["name"]: {key: value for key, value in f.items() if key != "name"}
                    for f in info["features"]
                },
                **sizes.get(info["config_name"], {"frequency": None}),
                "dataset_bytes": info["dataset_size"],
            }
            for info in metadata["dataset_info"]
            if info["config_name"] in configs
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, help="Use an already downloaded pinned README")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("demand_forecast/bench/tasks/schema_fev100.yaml"),
    )
    args = parser.parse_args()
    if args.card:
        card = args.card.read_text()
    else:
        for attempt in range(7):
            try:
                with urllib.request.urlopen(URL, timeout=60) as response:
                    card = response.read().decode()
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == 6:
                    raise
                time.sleep(max(float(exc.headers.get("Retry-After", 0)), 2**attempt))
    args.out.write_text(yaml.safe_dump(snapshot(card), sort_keys=False))


if __name__ == "__main__":
    main()
