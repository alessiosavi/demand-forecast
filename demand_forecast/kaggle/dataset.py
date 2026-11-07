"""Private Kaggle dataset packaging helpers.

Writes the ``dataset-metadata.json`` descriptor next to the prepared files
and builds the ``kaggle datasets`` argv for the create and version commands.
Commands are only composed here; the runner executes them.
"""

import json
from pathlib import Path
from typing import Literal

__all__ = ["dataset_argv", "write_dataset_metadata"]


def write_dataset_metadata(directory: Path, *, dataset_id: str, title: str) -> Path:
    """Write ``dataset-metadata.json`` into ``directory`` and return its path."""
    payload = {"title": title, "id": dataset_id, "licenses": [{"name": "other"}]}
    path = Path(directory) / "dataset-metadata.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return path


def dataset_argv(
    kind: Literal["create", "version"], directory: Path, message: str | None = None
) -> list[str]:
    """Compose the Kaggle CLI argv for a dataset ``create`` or ``version`` call.

    Creation always zips the directory; a message is appended when given, so
    version uploads carry their changelog text.
    """
    argv = ["kaggle", "datasets", kind, "-p", str(directory)]
    if kind == "create":
        argv += ["--dir-mode", "zip"]
    if message is not None:
        argv += ["-m", message]
    return argv
