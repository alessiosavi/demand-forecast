from scripts.world_signals.sources import SOURCES


def test_sources_registry_is_complete():
    names = [source.name for source in SOURCES]
    assert len(set(names)) == len(names)
    assert set(names) == {"yahoo", "cryptopanic", "fng", "gpr", "gdelt", "fomc", "reference_crypto"}
    for source in SOURCES:
        assert all((source.name, source.url, source.license, source.attribution, source.kind))
        assert source.url.startswith("https://")
        assert source.kind in {"prices", "news", "sentiment", "risk", "events", "reference"}


def test_request_pacing_uses_elapsed_time(monkeypatch):
    from scripts.world_signals import sources

    now, sleeps = [100.0], []

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr(sources.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(sources.time, "sleep", sleep)
    monkeypatch.setattr(sources, "_LAST_REQUEST", {})
    sources._pace("yahoo", 1)
    now[0] += 0.25
    sources._pace("yahoo", 1)
    sources._pace("gdelt", 5)
    now[0] += 1
    sources._pace("gdelt", 5)
    assert sleeps == [0.75, 4.0]


def test_cache_integrity_rejects_modified_raw_bytes(tmp_path):
    import io

    import pytest

    from scripts.world_signals.sources import cached_download

    url = "https://example.invalid/synthetic.json"
    cached_download(url, tmp_path, opener=lambda *a, **k: io.BytesIO(b'{"value":1}'))
    raw = next(path for path in tmp_path.glob("*.json") if not path.name.endswith(".meta.json"))
    raw.write_bytes(b'{"value":2}')
    with pytest.raises(ValueError, match="integrity"):
        cached_download(url, tmp_path)
