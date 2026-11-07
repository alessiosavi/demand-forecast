import json
from pathlib import Path

from demand_forecast.kaggle.dataset import dataset_argv, write_dataset_metadata


def test_metadata_round_trip(tmp_path: Path):
    path = write_dataset_metadata(
        tmp_path, dataset_id="alessiosavi/kb-store-sales", title="kb store sales"
    )
    assert path == tmp_path / "dataset-metadata.json"
    assert json.loads(path.read_text()) == {
        "title": "kb store sales",
        "id": "alessiosavi/kb-store-sales",
        "licenses": [{"name": "other"}],
    }


def test_create_and_version_argv(tmp_path: Path):
    assert dataset_argv("create", tmp_path) == [
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(tmp_path),
        "--dir-mode",
        "zip",
    ]
    assert dataset_argv("version", tmp_path, message="v2") == [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(tmp_path),
        "-m",
        "v2",
    ]
