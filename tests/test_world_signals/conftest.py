"""All producer tests are offline, including an explicit guard against HTTP."""

import io
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must inject an opener; network is forbidden")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)


@pytest.fixture
def fixture_opener():
    def factory(filename):
        def opener(request, **kwargs):
            return io.BytesIO((FIXTURES / filename).read_bytes())

        return opener

    return factory


@pytest.fixture
def json_opener():
    def factory(payload):
        def opener(request, **kwargs):
            return io.BytesIO(json.dumps(payload).encode())

        return opener

    return factory
