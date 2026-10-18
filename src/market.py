"""Live market context — delayed quotes, price history, trending.

A presentation layer over the RAG app: real market data from free,
keyless sources so the UI can sit the filings next to the tape. This
module NEVER feeds chat answers or the eval — API surface only.

* Primary source: Yahoo Finance via ``yfinance``.
* Fallback: Stooq's free CSV endpoints (quote + daily history).
* Trending: Yahoo's public trending endpoint (no yfinance API).

Every fetch is timeout-bounded and failure-isolated: any source error
degrades the payload to ``available: False`` (or falls back) instead
of raising, so a market outage can never break the app. Results are
cached in-memory per process (quotes 60s, history 15min, trending
5min), and every payload carries ``source`` + ``as_of`` +
``delayed: True`` — the UI labels the data as delayed, because it is.

Fetcher functions are module-level names (``_yahoo_quote_fetcher``
etc.) so tests can inject fakes without network.
"""

from __future__ import annotations

import csv
import io
import os
import re
import threading
import time
from datetime import datetime, timezone

import httpx

from src import registry

TIMEOUT = 8.0
QUOTE_TTL = 60.0
HISTORY_TTL = 15 * 60.0
TRENDING_TTL = 5 * 60.0

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

INDICES = (
    ("^GSPC", "S&P 500"),
    ("^IXIC", "Nasdaq Composite"),
    ("^DJI", "Dow Jones"),
)

RANGES = ("1mo", "6mo", "1y")
_RANGE_ROWS = {"1mo": 22, "6mo": 126, "1y": 260}
SPARK_POINTS = 22

_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}([.\-][A-Za-z]{1,2})?$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Yahoo via yfinance
# ---------------------------------------------------------------------------

_yf_session = None
_yf_session_lock = threading.Lock()
# curl_cffi sessions are not safe for concurrent requests (responses
# can cross between threads) — serialize Yahoo fetches. They are all
# cached, so the lock is rarely contended outside warm-up.
_yf_fetch_lock = threading.Lock()


def _get_yf_session():
    """One shared curl_cffi session for yfinance. Proxies are read from
    the environment when present (this sandbox egresses via a proxy;
    Railway does not set one and connects directly)."""
    global _yf_session
    if _yf_session is None:
        with _yf_session_lock:
            if _yf_session is None:
                from curl_cffi import requests as creq

                proxy = (
                    os.environ.get("https_proxy")
                    or os.environ.get("HTTPS_PROXY")
                    or os.environ.get("http_proxy")
                    or os.environ.get("HTTP_PROXY")
                )
                proxies = {"http": proxy, "https": proxy} if proxy else None
                _yf_session = creq.Session(
                    proxies=proxies, impersonate="chrome", timeout=TIMEOUT)
    return _yf_session


def _num(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if out == out else None  # NaN guard
    except (TypeError, ValueError):
        return None


def _quote_payload(ticker: str, source: str, *, name=None, price=None,
                   prev_close=None, day_high=None, day_low=None, volume=None,
                   market_cap=None, pe_ratio=None, high_52w=None,
                   low_52w=None, currency="USD") -> dict | None:
    price = _num(price)
    if price is None:
        return None
    prev_close = _num(prev_close)
    change = change_pct = None
    if prev_close:
        change = round(price - prev_close, 4)
        change_pct = round((price - prev_close) / prev_close * 100.0, 4)
    return {
        "ticker": ticker,
        "name": name or ticker,
        "price": price,
        "change": change,
        "change_pct": change_pct,
        "prev_close": prev_close,
        "day_high": _num(day_high),
        "day_low": _num(day_low),
        "volume": int(volume) if _num(volume) is not None else None,
        "market_cap": int(market_cap) if _num(market_cap) is not None else None,
        "pe_ratio": _num(pe_ratio),
        "high_52w": _num(high_52w),
        "low_52w": _num(low_52w),
        "currency": currency or "USD",
        "source": source,
    }


def _fetch_yahoo_quote(ticker: str) -> dict | None:
    with _yf_fetch_lock:
        return _fetch_yahoo_quote_locked(ticker)


def _fetch_yahoo_quote_locked(ticker: str) -> dict | None:
    import yfinance as yf

    t = yf.Ticker(ticker, session=_get_yf_session())
    info: dict = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    fast = None
    try:
        fast = t.fast_info
    except Exception:
        fast = None

    def fi(attr):
        try:
            return getattr(fast, attr) if fast is not None else None
        except Exception:
            return None

    return _quote_payload(
        ticker, "yahoo",
        name=info.get("shortName") or info.get("longName"),
        price=info.get("regularMarketPrice") or fi("last_price"),
        prev_close=(info.get("previousClose")
                    or info.get("regularMarketPreviousClose")
                    or fi("previous_close")),
        day_high=info.get("dayHigh") or fi("day_high"),
        day_low=info.get("dayLow") or fi("day_low"),
        volume=(info.get("volume") or info.get("regularMarketVolume")
                or fi("last_volume")),
        market_cap=info.get("marketCap") or fi("market_cap"),
        pe_ratio=info.get("trailingPE"),
        high_52w=info.get("fiftyTwoWeekHigh") or fi("year_high"),
        low_52w=info.get("fiftyTwoWeekLow") or fi("year_low"),
        currency=info.get("currency") or fi("currency") or "USD",
    )


def _fetch_yahoo_history(ticker: str, range_: str) -> list[dict] | None:
    with _yf_fetch_lock:
        return _fetch_yahoo_history_locked(ticker, range_)


def _fetch_yahoo_history_locked(ticker: str, range_: str) -> list[dict] | None:
    import yfinance as yf

    t = yf.Ticker(ticker, session=_get_yf_session())
    hist = t.history(period=range_, interval="1d", timeout=TIMEOUT,
                     auto_adjust=False)
    if hist is None or len(hist) == 0:
        return None
    closes = hist["Close"].dropna()
    points = [
        {"date": idx.date().isoformat(), "close": round(float(c), 4)}
        for idx, c in closes.items()
    ]
    return _downsample(points, 260)


def _downsample(points: list[dict], cap: int) -> list[dict]:
    if len(points) <= cap:
        return points
    step = (len(points) + cap - 1) // cap
    out = points[::step]
    if out[-1] is not points[-1]:
        out.append(points[-1])
    return out[:cap]


# ---------------------------------------------------------------------------
# Stooq fallback (free CSV, no key)
# ---------------------------------------------------------------------------

def _stooq_symbol(ticker: str) -> str | None:
    if ticker.startswith("^") or "-" in ticker:
        return None  # Stooq's .us CSV covers plain US tickers
    return f"{ticker.lower()}.us"


def _httpx_client() -> httpx.Client:
    """httpx with the env proxy passed explicitly and trust_env off:
    some environments carry bracketed-IPv6 NO_PROXY entries that crash
    httpx's env parsing outright (this sandbox is one)."""
    proxy = (
        os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("HTTP_PROXY")
    )
    # Prefer the OS CA bundle when present: TLS-intercepting egress
    # proxies (this sandbox) chain a CA the OS store trusts but
    # certifi's bundle does not.
    verify: str | bool = True
    if os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
        verify = "/etc/ssl/certs/ca-certificates.crt"
    return httpx.Client(
        timeout=TIMEOUT, follow_redirects=True,
        headers={"User-Agent": BROWSER_UA},
        proxy=proxy, trust_env=False, verify=verify)


def _stooq_get(url: str) -> str:
    with _httpx_client() as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def _stooq_history_rows(ticker: str) -> list[dict]:
    sym = _stooq_symbol(ticker)
    if sym is None:
        return []
    text = _stooq_get(f"https://stooq.com/q/d/l/?s={sym}&i=d")
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        close = _num(row.get("Close"))
        if close is not None and row.get("Date"):
            rows.append({"date": row["Date"], "close": close,
                         "volume": _num(row.get("Volume"))})
    return rows


def _fetch_stooq_history(ticker: str, range_: str) -> list[dict] | None:
    rows = _stooq_history_rows(ticker)
    if not rows:
        return None
    keep = _RANGE_ROWS.get(range_, 126)
    points = [{"date": r["date"], "close": r["close"]} for r in rows[-keep:]]
    return _downsample(points, 260)


def _fetch_stooq_quote(ticker: str) -> dict | None:
    sym = _stooq_symbol(ticker)
    if sym is None:
        return None
    text = _stooq_get(
        f"https://stooq.com/q/l/?s={sym}&f=sd2t2ohlcv&h&e=csv")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return None
    row = rows[0]
    price = _num(row.get("Close"))
    if price is None:
        return None
    prev_close = None
    try:  # change needs the prior close — daily history's second-to-last
        hist = _stooq_history_rows(ticker)
        if len(hist) >= 2:
            prev_close = hist[-2]["close"]
    except Exception:
        prev_close = None
    return _quote_payload(
        ticker, "stooq",
        price=price, prev_close=prev_close,
        day_high=row.get("High"), day_low=row.get("Low"),
        volume=row.get("Volume"),
    )


# ---------------------------------------------------------------------------
# Trending (Yahoo public endpoint)
# ---------------------------------------------------------------------------

def _fetch_trending_symbols() -> list[str]:
    url = "https://query1.finance.yahoo.com/v1/finance/trending/US?count=10"
    with _httpx_client() as client:
        resp = client.get(url)
        resp.raise_for_status()
        payload = resp.json()
    result = (payload.get("finance") or {}).get("result") or []
    if not result:
        return []
    return [q.get("symbol", "") for q in result[0].get("quotes", [])]


# Fetcher indirection — tests monkeypatch these names with fakes.
_yahoo_quote_fetcher = _fetch_yahoo_quote
_stooq_quote_fetcher = _fetch_stooq_quote
_yahoo_history_fetcher = _fetch_yahoo_history
_stooq_history_fetcher = _fetch_stooq_history
_trending_fetcher = _fetch_trending_symbols


# ---------------------------------------------------------------------------
# Cache + public API
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _cached(key: str, ttl: float, producer) -> dict:
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
    payload = producer()
    with _cache_lock:
        _cache[key] = (time.monotonic() + ttl, payload)
    return payload


def _unavailable(ticker: str | None = None, **extra) -> dict:
    out = {"available": False, "delayed": True, "source": None,
           "as_of": _now_iso()}
    if ticker is not None:
        out["ticker"] = ticker
    out.update(extra)
    return out


def get_quote(ticker: str) -> dict:
    ticker = ticker.strip().upper()

    def produce() -> dict:
        for fetcher in (_yahoo_quote_fetcher, _stooq_quote_fetcher):
            try:
                quote = fetcher(ticker)
            except Exception:
                quote = None
            if quote:
                return {**quote, "delayed": True, "available": True,
                        "as_of": _now_iso()}
        return _unavailable(ticker)

    return _cached(f"quote:{ticker}", QUOTE_TTL, produce)


def get_history(ticker: str, range_: str = "6mo") -> dict:
    ticker = ticker.strip().upper()
    if range_ not in RANGES:
        range_ = "6mo"

    def produce() -> dict:
        for fetcher, source in ((_yahoo_history_fetcher, "yahoo"),
                                (_stooq_history_fetcher, "stooq")):
            try:
                points = fetcher(ticker, range_)
            except Exception:
                points = None
            if points:
                return {"ticker": ticker, "range": range_, "points": points,
                        "source": source, "delayed": True, "available": True,
                        "as_of": _now_iso()}
        return _unavailable(ticker, range=range_, points=[])

    return _cached(f"history:{ticker}:{range_}", HISTORY_TTL, produce)


def _with_spark(quote: dict, ticker: str, name: str | None = None) -> dict:
    entry = dict(quote)
    if name:
        entry["name"] = name
    hist = get_history(ticker, "1mo")
    entry["spark"] = (
        [p["close"] for p in hist["points"][-SPARK_POINTS:]]
        if hist.get("available") else []
    )
    return entry


def get_overview() -> dict:
    def produce() -> dict:
        indices = [
            _with_spark(get_quote(sym), sym, display)
            for sym, display in INDICES
        ]
        companies = [
            _with_spark(get_quote(t), t, registry.company_name(t))
            for t in sorted(registry.known_tickers())
        ]
        live = [q for q in indices + companies if q.get("available")]
        source = live[0]["source"] if live else None
        return {
            "indices": indices,
            "companies": companies,
            "source": source,
            "delayed": True,
            "available": bool(live),
            "as_of": _now_iso(),
        }

    # The overview is a composition of individually-cached quotes, so
    # it is not cached as a blob — a session ingest shows up at once.
    return produce()


def get_trending() -> dict:
    def produce() -> dict:
        try:
            symbols = _trending_fetcher()
        except Exception:
            symbols = []
        out = []
        for sym in symbols:
            sym = (sym or "").strip().upper()
            if not _TICKER_RE.match(sym):
                continue  # crypto / futures symbols can't be ingested
            quote = get_quote(sym)
            if not quote.get("available"):
                continue
            out.append({
                "ticker": sym,
                "name": quote.get("name") or sym,
                "price": quote.get("price"),
                "change_pct": quote.get("change_pct"),
            })
            if len(out) >= 10:
                break
        return {
            "trending": out,
            "source": "yahoo" if out else None,
            "delayed": True,
            "available": bool(out),
            "as_of": _now_iso(),
        }

    return _cached("trending", TRENDING_TTL, produce)


def get_filing_figures(ticker: str) -> dict:
    """Revenue + net income by fiscal year from the corpus figures
    table — the same rows the chat cites (bundled AAPL + every session
    company), regrouped as [{fiscal_year, revenue, net_income}]."""
    ticker = ticker.strip().upper()
    by_year: dict[int, dict] = {}
    for row in registry.all_figures():
        if row["ticker"] != ticker:
            continue
        entry = by_year.setdefault(
            row["fiscal_year"],
            {"fiscal_year": row["fiscal_year"], "revenue": None,
             "net_income": None})
        if row["concept"] == "Total revenue":
            entry["revenue"] = row["value_usd"]
        elif row["concept"] == "Net income":
            entry["net_income"] = row["value_usd"]
    rows = [by_year[y] for y in sorted(by_year, reverse=True)]
    return {
        "ticker": ticker,
        "rows": rows,
        "available": bool(rows),
        "source": "SEC company facts — corpus figures table",
        "as_of": _now_iso(),
    }


def prewarm() -> None:
    """Startup warm-up: index/strip quotes + 1-month sparks so the
    first page load is not the cold one. Best-effort by design."""
    for sym, _display in INDICES:
        get_quote(sym)
        get_history(sym, "1mo")
    for ticker in ("AAPL", "MSFT", "NVDA"):
        get_quote(ticker)
        get_history(ticker, "1mo")
