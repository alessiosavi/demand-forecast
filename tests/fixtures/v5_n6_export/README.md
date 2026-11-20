# N6 naive mix export fixture

The export writer at revision `072ce26` produced this fixture with
`derive: [naive_anchor]` and `naive_mix: true` on a seeded, untrained tiny model.
The parity bundle includes a dated category case, seven single-target cases with
short contexts, and the stock hourly case where all six paths are available.

The command below ran with Torch 2.8.0. Run it from a source archive of
`072ce26` and set the output argument to an empty destination.

```sh
cd /private/tmp/v5-n7/base
PYTHONPATH=. /private/tmp/v5-b3a/test-venv/bin/python - /private/tmp/v5-n7/fix1-n6-export <<'PY'
import importlib
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from demand_forecast.data.episode import EpisodeRequest, Item, Panel

export = importlib.import_module("demand_forecast.models.v5.export")
train_config = export.V5TrainConfig
parity_cases = export._parity_cases


def with_mix(model, *args, **kwargs):
    return train_config(replace(model, naive_mix=True), *args, **kwargs)


def small_cases(screening, derive):
    cases = list(parity_cases(screening, derive))
    panel, requests = cases[5]
    yield panel, [replace(requests[0], context=16, horizon=2)]
    steps = np.arange(34, dtype=np.float64)
    for i in range(7):
        values = np.sin(steps / (i + 1)) + steps / 10
        item = Item(f"item-{i}", 0, {"y": values}, {}, {})
        panel = Panel(f"short-{i}", (item,), {}, None, None, False)
        yield panel, [EpisodeRequest(0, (32,), 2, context=16, derive=derive)]
    yield cases[-1]


torch.set_num_threads(1)
with patch.object(export, "V5TrainConfig", with_mix), patch.object(
    export, "_parity_cases", small_cases
):
    export._selftest(Path(sys.argv[1]), ("naive_anchor",))
PY
```

Keep `config.json`, `model.safetensors`, and `parity.npz` as written by that
revision so the replay test continues to exercise an N6 artifact.
