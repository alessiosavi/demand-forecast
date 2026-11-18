"""File descriptor headroom for v5 batch transfer."""

import subprocess
import sys
from dataclasses import replace
from textwrap import dedent
from unittest.mock import Mock, call

import pytest

from demand_forecast.pretrain.v5 import sampler
from tests.helpers.v5_sampler import config


@pytest.fixture
def file_limits(monkeypatch):
    resource = pytest.importorskip("resource")
    get_limit = Mock(return_value=(128, resource.RLIM_INFINITY))

    def set_limit(kind, limits):
        assert kind == resource.RLIMIT_NOFILE
        assert limits[1] == get_limit.return_value[1]
        get_limit.return_value = limits

    monkeypatch.setattr(resource, "getrlimit", get_limit)
    monkeypatch.setattr(resource, "setrlimit", Mock(side_effect=set_limit))
    return resource


@pytest.fixture
def data_loader(monkeypatch):
    constructor = Mock(return_value=())
    monkeypatch.setattr(sampler, "DataLoader", constructor)
    return constructor


def test_raises_soft_limit_before_building_loader(file_limits, data_loader, capsys):
    file_limits.getrlimit.return_value = (128, 4096)

    def build(*args, **kwargs):
        assert file_limits.getrlimit.return_value == (4096, 4096)
        return ()

    data_loader.side_effect = build
    sampler.loader(replace(config(), workers=2), 0)

    file_limits.setrlimit.assert_called_once_with(file_limits.RLIMIT_NOFILE, (4096, 4096))
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert "RLIMIT_NOFILE" in lines[0] and "128" in lines[0] and lines[0].count("4096") == 2


@pytest.mark.parametrize(
    "attempts",
    [
        (1_048_576,),
        (1_048_576, 65_536),
        (1_048_576, 65_536, 16_384),
        (1_048_576, 65_536, 16_384, 624),
    ],
    ids=["target", "first-fallback", "second-fallback", "minimum"],
)
def test_unlimited_hard_limit_falls_back(file_limits, data_loader, attempts):
    accept = file_limits.setrlimit.side_effect

    def set_limit(kind, limits):
        if limits[0] > attempts[-1]:
            if limits[0] == 1_048_576:
                raise ValueError("refused")
            raise OSError("refused")
        accept(kind, limits)

    file_limits.setrlimit.side_effect = set_limit
    sampler.loader(replace(config(), workers=2), 0)

    assert file_limits.setrlimit.call_args_list == [
        call(file_limits.RLIMIT_NOFILE, (limit, file_limits.RLIM_INFINITY)) for limit in attempts
    ]


@pytest.mark.parametrize("soft", [20_000, 1_200_000, None], ids=["fallback", "above", "unlimited"])
def test_never_lowers_soft_limit(file_limits, data_loader, capsys, soft):
    hard = file_limits.RLIM_INFINITY
    file_limits.getrlimit.return_value = (hard if soft is None else soft, hard)
    file_limits.setrlimit.side_effect = OSError("refused")

    sampler.loader(replace(config(), workers=2), 0)

    attempts = (1_048_576, 65_536) if soft == 20_000 else ()
    assert file_limits.setrlimit.call_args_list == [
        call(file_limits.RLIMIT_NOFILE, (limit, hard)) for limit in attempts
    ]
    assert capsys.readouterr().err == ""


def test_rejects_insufficient_soft_limit_before_building_loader(file_limits, data_loader):
    file_limits.getrlimit.return_value = (128, 650)

    with pytest.raises(RuntimeError, match=r"RLIMIT_NOFILE.*650.*808.*3.*ulimit -n.*fewer workers"):
        sampler.loader(replace(config(), workers=3), 0)

    file_limits.setrlimit.assert_called_once_with(file_limits.RLIMIT_NOFILE, (650, 650))
    data_loader.assert_not_called()


def test_serial_loader_does_not_touch_file_limit(file_limits, data_loader):
    sampler.loader(config(), 0)

    file_limits.getrlimit.assert_not_called()
    file_limits.setrlimit.assert_not_called()


def test_loader_without_resource_module(monkeypatch, data_loader):
    monkeypatch.setitem(sys.modules, "resource", None)

    sampler.loader(replace(config(), workers=2), 0)

    data_loader.assert_called_once()


def test_worker_processes_load_twelve_batches_with_low_initial_file_limit():
    pytest.importorskip("resource")
    child = dedent("""\
        import resource
        from dataclasses import replace
        from itertools import islice

        from demand_forecast.pretrain.v5.sampler import loader
        from tests.helpers.v5_sampler import config

        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, hard))
        batches = list(islice(loader(replace(config(episodes_per_batch=1), workers=2), 0), 12))
        assert [index for index, _, _ in batches] == list(range(12))
        soft, actual_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        assert soft == resource.RLIM_INFINITY or soft >= 624, soft
        assert actual_hard == hard
        """)
    result = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
