"""EDGAR company resolution — turn a name or ticker into a CIK.

The SEC publishes ``company_tickers.json`` (ticker, name, CIK for every
listed company). It is fetched once with the declared user agent and
fair-access pacing, then cached under ``data/cache/`` (gitignored);
resolution itself is a pure local lookup. If neither cache nor network
is available, a small seed map (the bundled companies + Tesla) keeps
the demo's core flows working offline.
"""

from __future__ import annotations

import json
import re

from src.ingest import CACHE_DIR, DATA_DIR, _edgar_get

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
TICKERS_CACHE = CACHE_DIR / "company_tickers.json"

# Offline last resort — only companies the demo can actually serve
# without the full map (bundled fixtures / precached Tesla).
SEED_TICKERS = [
    {"ticker": "AAPL", "name": "Apple Inc.", "cik": "0000320193"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "cik": "0000789019"},
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "cik": "0001045810"},
    {"ticker": "TSLA", "name": "Tesla, Inc.", "cik": "0001318605"},
]

_map_cache: list[dict] | None = None


def _normalize(raw: dict) -> list[dict]:
    out = []
    for row in raw.values():
        out.append({
            "ticker": str(row["ticker"]).upper(),
            "name": str(row["title"]),
            "cik": str(row["cik_str"]).zfill(10),
        })
    return out


def load_ticker_map(allow_network: bool = True) -> list[dict]:
    """The full SEC ticker map: cache -> network -> seed fallback."""
    global _map_cache
    if _map_cache is not None:
        return _map_cache
    if TICKERS_CACHE.exists():
        try:
            _map_cache = _normalize(json.loads(TICKERS_CACHE.read_text(encoding="utf-8")))
            return _map_cache
        except (ValueError, KeyError):
            pass  # corrupt cache: fall through to network/seed
    if allow_network:
        try:
            payload = json.loads(_edgar_get(TICKERS_URL))
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            TICKERS_CACHE.write_text(json.dumps(payload), encoding="utf-8")
            _map_cache = _normalize(payload)
            return _map_cache
        except Exception:
            pass  # offline: seed map below
    _map_cache = list(SEED_TICKERS)
    return _map_cache


def resolve(query: str, ticker_map: list[dict] | None = None,
            limit: int = 5) -> list[dict]:
    """Resolve a ticker or company name to candidate companies.

    Exact ticker match wins; then name prefix, name substring, and
    all-query-words-present matches, shortest names first.
    """
    q = query.strip()
    if not q:
        return []
    rows = ticker_map if ticker_map is not None else load_ticker_map()
    q_ticker = q.upper()
    q_name = q.lower()
    q_words = [w for w in re.split(r"[^a-z0-9]+", q_name) if w]

    exact = [r for r in rows if r["ticker"] == q_ticker]
    scored: list[tuple[int, int, str, dict]] = []
    for r in rows:
        if r in exact:
            continue
        name = r["name"].lower()
        if name.startswith(q_name):
            rank = 0
        elif q_name in name:
            rank = 1
        elif q_words and all(w in name for w in q_words):
            rank = 2
        elif r["ticker"].startswith(q_ticker) and len(q_ticker) >= 2:
            rank = 3
        else:
            continue
        scored.append((rank, len(r["name"]), r["name"], r))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return (exact + [t[3] for t in scored])[:limit]


def company_for_ticker(ticker: str,
                       ticker_map: list[dict] | None = None) -> dict | None:
    """The map row for one exact ticker, or None."""
    rows = ticker_map if ticker_map is not None else load_ticker_map()
    for r in rows:
        if r["ticker"] == ticker.upper():
            return r
    return None
