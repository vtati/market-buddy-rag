"""Feature 8 — live market context: quotes with source fallback,
caching, overview composition, and the corpus filing-figures view.

Offline by design: every source fetcher is replaced with an injected
fake, so the fallback order, the cache, and the degrade-to-unavailable
contract are tested without touching Yahoo or Stooq.
"""

import pytest

from src import market
from src.session_ingest import ingest_company


@pytest.fixture(autouse=True)
def _clean_market_cache():
    market.clear_cache()
    yield
    market.clear_cache()


def _quote(ticker, source="yahoo", price=250.0):
    return {
        "ticker": ticker, "name": f"{ticker} Inc.", "price": price,
        "change": 2.5, "change_pct": 1.01, "prev_close": price - 2.5,
        "day_high": price + 1.0, "day_low": price - 1.0,
        "volume": 1_000_000, "market_cap": 1_000_000_000,
        "pe_ratio": 25.0, "high_52w": 300.0, "low_52w": 150.0,
        "currency": "USD", "source": source,
    }


def _points(ticker, range_):
    return [{"date": "2026-10-01", "close": 100.0},
            {"date": "2026-10-02", "close": 101.5}]


def test_quote_returns_yahoo_shaped_data(monkeypatch):
    monkeypatch.setattr(market, "_yahoo_quote_fetcher",
                        lambda t: _quote(t))
    monkeypatch.setattr(market, "_stooq_quote_fetcher",
                        lambda t: None)
    out = market.get_quote("aapl")
    assert out["available"] is True
    assert out["source"] == "yahoo"
    assert out["ticker"] == "AAPL"
    assert out["price"] == 250.0
    assert out["delayed"] is True
    assert out["as_of"]


def test_quote_falls_back_to_stooq_when_yahoo_raises(monkeypatch):
    def boom(ticker):
        raise RuntimeError("yahoo down")

    monkeypatch.setattr(market, "_yahoo_quote_fetcher", boom)
    monkeypatch.setattr(market, "_stooq_quote_fetcher",
                        lambda t: _quote(t, source="stooq"))
    out = market.get_quote("MSFT")
    assert out["available"] is True
    assert out["source"] == "stooq"


def test_quote_both_sources_failing_degrades_without_raising(monkeypatch):
    def boom(ticker):
        raise RuntimeError("down")

    monkeypatch.setattr(market, "_yahoo_quote_fetcher", boom)
    monkeypatch.setattr(market, "_stooq_quote_fetcher", boom)
    out = market.get_quote("NVDA")
    assert out["available"] is False
    assert out["ticker"] == "NVDA"


def test_quote_is_cached_within_ttl(monkeypatch):
    calls = {"n": 0}

    def counting(ticker):
        calls["n"] += 1
        return _quote(ticker)

    monkeypatch.setattr(market, "_yahoo_quote_fetcher", counting)
    monkeypatch.setattr(market, "_stooq_quote_fetcher", lambda t: None)
    first = market.get_quote("AAPL")
    second = market.get_quote("AAPL")
    assert first == second
    assert calls["n"] == 1, "second call within the TTL must not refetch"


def test_history_falls_back_and_degrades(monkeypatch):
    def boom(ticker, range_):
        raise RuntimeError("down")

    monkeypatch.setattr(market, "_yahoo_history_fetcher", boom)
    monkeypatch.setattr(market, "_stooq_history_fetcher", _points)
    out = market.get_history("AAPL", "6mo")
    assert out["available"] is True
    assert out["source"] == "stooq"
    assert out["points"][-1]["close"] == 101.5

    monkeypatch.setattr(market, "_stooq_history_fetcher", boom)
    market.clear_cache()
    out = market.get_history("AAPL", "6mo")
    assert out["available"] is False
    assert out["points"] == []


def test_overview_has_indices_and_bundled_companies(monkeypatch):
    monkeypatch.setattr(market, "_yahoo_quote_fetcher",
                        lambda t: _quote(t))
    monkeypatch.setattr(market, "_stooq_quote_fetcher", lambda t: None)
    monkeypatch.setattr(market, "_yahoo_history_fetcher", _points)
    monkeypatch.setattr(market, "_stooq_history_fetcher",
                        lambda t, r: None)
    out = market.get_overview()
    assert [q["ticker"] for q in out["indices"]] == ["^GSPC", "^IXIC", "^DJI"]
    assert out["indices"][0]["name"] == "S&P 500"
    company_tickers = {q["ticker"] for q in out["companies"]}
    assert {"AAPL", "MSFT", "NVDA"} <= company_tickers
    assert all(q["spark"] for q in out["companies"])
    assert out["available"] is True


def test_trending_filters_to_ingestible_tickers(monkeypatch):
    monkeypatch.setattr(
        market, "_trending_fetcher",
        lambda: ["ZZTOP", "BTC-USD", "ES=F", "AAPL"])
    monkeypatch.setattr(market, "_yahoo_quote_fetcher",
                        lambda t: _quote(t))
    monkeypatch.setattr(market, "_stooq_quote_fetcher", lambda t: None)
    out = market.get_trending()
    tickers = [row["ticker"] for row in out["trending"]]
    assert tickers == ["ZZTOP", "AAPL"]  # crypto/futures filtered out
    assert out["available"] is True


def test_filing_figures_aapl_fy2024():
    out = market.get_filing_figures("AAPL")
    assert out["available"] is True
    by_year = {row["fiscal_year"]: row for row in out["rows"]}
    assert by_year[2024]["revenue"] == 391_035_000_000
    assert by_year[2024]["net_income"] == 93_736_000_000
    years = [row["fiscal_year"] for row in out["rows"]]
    assert years == sorted(years, reverse=True)


def test_filing_figures_tsla_after_precache_registration():
    ingest_company("TSLA", allow_network=False)
    out = market.get_filing_figures("TSLA")
    assert out["available"] is True
    by_year = {row["fiscal_year"]: row for row in out["rows"]}
    assert by_year[2024]["revenue"] == 97_690_000_000


def test_filing_figures_unknown_company_unavailable():
    out = market.get_filing_figures("MSFT")  # bundled, but no facts table
    assert out["available"] is False
    assert out["rows"] == []
