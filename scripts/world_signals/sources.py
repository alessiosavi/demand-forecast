"""Source identity and attribution for the probe's eventual bundle manifest."""

import hashlib
import json
import os
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    license: str
    attribution: str
    kind: str


SOURCES = (
    Source(
        "yahoo",
        "https://query1.finance.yahoo.com/v8/finance/chart/",
        "No redistribution license established; private bundle only",
        "Yahoo Finance",
        "prices",
    ),
    Source(
        "cryptopanic",
        "https://github.com/soheilrahsaz/cryptoNewsDataset",
        "CC0-1.0",
        "CryptoPanic archive by soheilrahsaz",
        "news",
    ),
    Source(
        "fng",
        "https://api.alternative.me/fng/?limit=0&format=json",
        "Free with attribution",
        "Alternative.me Crypto Fear & Greed Index",
        "sentiment",
    ),
    Source(
        "gpr",
        "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.dta",
        "CC-BY-4.0",
        "Caldara and Iacoviello (2022), Measuring Geopolitical Risk, AER",
        "risk",
    ),
    Source(
        "gdelt",
        "https://api.gdeltproject.org/api/v2/doc/doc",
        "Open data; cite and link GDELT",
        "The GDELT Project",
        "news",
    ),
    Source(
        "fomc",
        "https://www.kaggle.com/datasets/pushbuildteam/fomc-meeting-calendar",
        "CC0-1.0",
        "pushbuildteam FOMC Meeting Calendar",
        "events",
    ),
    Source(
        "reference_crypto",
        "https://www.kaggle.com/datasets/fleongg/multi-coin-crypto-daily-market-dataset",
        "CC-BY-4.0",
        "fleongg Multi-Coin Crypto Daily Market Dataset (reference only)",
        "reference",
    ),
)

# Shared transport keeps cache provenance and pacing consistent across producers.
# Each process serializes requests; callers should not run parallel producer processes.

DEFAULT_RAW_DIR = Path("data/world-signals/raw")
USER_AGENT = "Mozilla/5.0 (compatible; WorldSignalsResearch/1.0)"
_LAST_REQUEST = {}
_REQUEST_LOCK = threading.Lock()


def _pace(host, interval):
    previous = _LAST_REQUEST.get(host)
    if previous is not None:
        delay = interval - (time.monotonic() - previous)
        if delay > 0:
            time.sleep(delay)
    _LAST_REQUEST[host] = time.monotonic()


def _atomic_write(path, body):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(body)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cached_download(
    url, raw_dir=None, *, suffix=".json", refresh=False, opener=None, interval=0, validate=None
):
    """Fetch bytes with a URL-addressed cache and adjacent provenance JSON.

    ``opener`` has urllib's ``urlopen(Request, timeout=...)`` contract. A validator
    raises on invalid content before it enters the cache. Retries are limited to
    three total attempts, only on HTTP 429; all other failures propagate.
    """
    directory = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    path = directory / f"{key}{suffix}"
    metadata_path = directory / f"{key}{suffix}.meta.json"
    if path.exists() and metadata_path.exists() and not refresh:
        body = path.read_bytes()
        metadata = json.loads(metadata_path.read_text())
        if metadata["url"] != url or metadata["sha256"] != hashlib.sha256(body).hexdigest():
            raise ValueError(f"Cache integrity mismatch: {path}")
        if validate is not None:
            validate(body)
        return body

    open_url = opener if opener is not None else urllib.request.urlopen
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with _REQUEST_LOCK:
                _pace(urlsplit(url).netloc, interval)
                with open_url(request, timeout=60) as response:
                    body = response.read()
                    status = getattr(response, "status", 200)
            break
        except HTTPError as error:
            if error.code != 429 or attempt == 2:
                raise
            time.sleep(max(interval, 2**attempt))
    if validate is not None:
        validate(body)
    metadata = {
        "url": url,
        "http_status": status,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
    }
    _atomic_write(path, body)
    _atomic_write(metadata_path, (json.dumps(metadata, indent=2) + "\n").encode())
    return body
