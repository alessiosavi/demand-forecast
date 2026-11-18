"""Adapt the prepared upstream generators without importing their trainer."""

import importlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from demand_forecast.pretrain.v5.config import SamplerConfig, SourceConfig
from tests.helpers.v5_loop import config
from tests.helpers.v5_sampler import equal_batches
from tests.test_pretrain_v5.test_sources import equal_panels


def loaded_simons():
    return [name for name in sys.modules if name == "simons" or name.startswith("simons.")]


@pytest.fixture
def upstream(monkeypatch):
    location = os.environ.get("SIMONS_UPSTREAM")
    if not location or not (Path(location) / "src/simons/synth").is_dir():
        pytest.skip("set SIMONS_UPSTREAM to a Simons clone at third_party/simons/PINNED")
    # Other suites import a patched copy of the package into this process. Take every
    # module of it out for the test, so the import below is this checkout's own and
    # "no trainer loaded" is a statement about this source, whatever ran before.
    for name in loaded_simons():
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(Path(location) / "src"))
    module = importlib.import_module("simons.synth")
    assert Path(module.__file__).is_relative_to(Path(location).resolve())
    yield importlib.import_module("demand_forecast.pretrain.v5.sources")
    # Drop what this test imported; monkeypatch then puts back what was loaded before.
    for name in loaded_simons():
        del sys.modules[name]


def test_absent_package_is_lazy_and_explains_setup():
    program = """
import sys
class NoSimons:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'simons':
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, NoSimons())
import numpy as np
from demand_forecast.pretrain.v5.config import SourceConfig
from demand_forecast.pretrain.v5.sources import SourceMix, TypedSource
TypedSource(SourceConfig('typed', 1, recipes=((),), max_items=1, length=(32, 32))).panel(np.random.default_rng(0))
mix = SourceMix((SourceConfig('upstream', 1),))
try:
    mix.panel(np.random.default_rng(0))
except ImportError as error:
    message = str(error)
    assert 'Simons' in message and 'PINNED' in message and 'PYTHONPATH' in message
    assert '--upstream' in message
else:
    raise AssertionError('missing upstream package must fail when requested')
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_unrelated_missing_dependency_keeps_its_original_error(tmp_path):
    package = tmp_path / "simons"
    package.mkdir()
    (package / "__init__.py").write_text("import definitely_missing_simons_dependency\n")
    program = """
import numpy as np
from demand_forecast.pretrain.v5.config import SourceConfig
from demand_forecast.pretrain.v5.sources import UpstreamSource
try:
    UpstreamSource(SourceConfig('upstream', 1)).panel(np.random.default_rng(0))
except ModuleNotFoundError as error:
    assert error.name == 'definitely_missing_simons_dependency'
else:
    raise AssertionError('the missing dependency must surface unchanged')
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_undated_panel_has_numeric_covariates_and_no_trainer(upstream):
    panel, record = upstream.UpstreamSource(SourceConfig("upstream", 1)).panel(
        np.random.default_rng(12)
    )
    panel.validate()
    assert panel.freq is None and panel.epoch is None and not panel.aligned
    assert record["source"] == "upstream"
    assert not any(name.startswith("simons.training") for name in sys.modules)
    assert len(panel.items) == 1
    item = panel.items[0]
    assert item.start == 0 and not item.static
    assert set(item.targets) == {record["target"]}
    assert np.isfinite(item.targets[record["target"]][:2048]).any()
    assert item.dynamic and item.dynamic.keys() == panel.roles.keys()
    for values in (*item.targets.values(), *item.dynamic.values()):
        assert values.dtype == np.float64 and values.ndim == 1
        assert len(values) > 2048 and np.isfinite(values).any()
        assert not np.isinf(values).any()
    assert set(panel.roles.values()) <= {"known", "past"}


def test_replay_uses_one_caller_draw_and_preserves_global_rngs(upstream):
    source = upstream.UpstreamSource(SourceConfig("upstream", 1))
    rng, duplicate = np.random.default_rng(9), np.random.default_rng(9)
    seed = int(duplicate.integers(0, 2**63))
    numpy_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
    first, record = source.panel(rng)
    assert record["seed"] == seed
    assert rng.bit_generator.state == duplicate.bit_generator.state
    second, replay = source.panel(np.random.default_rng(9))
    equal_panels(first, second)
    assert record == replay
    actual_numpy = np.random.get_state()
    assert numpy_state[0] == actual_numpy[0]
    np.testing.assert_array_equal(numpy_state[1], actual_numpy[1])
    assert numpy_state[2:] == actual_numpy[2:]
    assert torch.equal(torch_state, torch.get_rng_state())
    third, _ = source.panel(rng)
    assert not np.array_equal(
        first.items[0].targets[record["target"]],
        third.items[0].targets[record["target"]],
        equal_nan=True,
    )


def test_missing_as_zero_leaves_source_panel_and_rng_unchanged(upstream):
    left, right = np.random.default_rng(4), np.random.default_rng(4)
    source = SourceConfig("upstream", 1)
    true, _ = upstream.UpstreamSource(source).panel(left)
    controlled, _ = upstream.UpstreamSource(replace(source, control="missing_as_zero")).panel(right)
    equal_panels(true, controlled)
    assert left.bit_generator.state == right.bit_generator.state


def test_different_seeds_give_distinct_item_identifiers(upstream):
    source = upstream.UpstreamSource(SourceConfig("upstream", 1))
    first, first_record = source.panel(np.random.default_rng(12))
    second, second_record = source.panel(np.random.default_rng(13))
    assert first_record["seed"] != second_record["seed"]
    assert first.items[0].item_id != second.items[0].item_id
    assert first.items[0].item_id == f"upstream:{first_record['seed']}"
    assert second.items[0].item_id == f"upstream:{second_record['seed']}"


def test_source_mix_produces_repeatable_sampler_batches(upstream):
    from demand_forecast.pretrain.v5.sampler import make_batch

    settings = replace(config(), sources=(SourceConfig("upstream", 1),), sampler=SamplerConfig())
    sources = upstream.SourceMix(settings.sources)
    assert sources.corpus_digest is None
    first, counts = make_batch(settings, 0, sources=sources)
    second, repeated = make_batch(settings, 0, sources=sources)
    equal_batches(first, second)
    assert counts == repeated
    assert counts["episodes"] == first.x.shape[0] == 32
    assert counts["sources"] == {"upstream": 32}
