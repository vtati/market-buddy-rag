"""Feature 7 — "Add a Company": resolution, precache ingest, and the
dynamic-scope arc (refuse -> ingest -> cited answer).

Offline by design: resolution is tested against an injected ticker
map, and ingestion runs through the committed Tesla precache
(data/precached/TSLA) with network explicitly disabled — if any code
path tried EDGAR, these tests would fail.
"""

from src import edgar, registry
from src.graph import run
from src.session_ingest import ingest_company

_MAP = [
    {"ticker": "TSLA", "name": "Tesla, Inc.", "cik": "0001318605"},
    {"ticker": "KO", "name": "Coca-Cola Co", "cik": "0000021344"},
    {"ticker": "AAPL", "name": "Apple Inc.", "cik": "0000320193"},
]

QUESTION = "What was Tesla's revenue in fiscal 2024?"


def test_resolve_tesla_by_ticker_and_name():
    by_ticker = edgar.resolve("tsla", ticker_map=_MAP)
    assert by_ticker[0]["ticker"] == "TSLA"
    assert by_ticker[0]["cik"] == "0001318605"
    assert "Tesla" in by_ticker[0]["name"]
    by_name = edgar.resolve("Tesla", ticker_map=_MAP)
    assert by_name and by_name[0]["ticker"] == "TSLA"


def test_resolve_gibberish_returns_no_candidates():
    assert edgar.resolve("zzqqx-not-a-company", ticker_map=_MAP) == []


def test_ingest_tsla_from_precache_registers_company():
    meta = ingest_company("TSLA", allow_network=False)
    assert meta["ticker"] == "TSLA"
    assert meta["source"] == "precached"
    assert meta["chunks"]["fixed"] > 0
    assert meta["chunks"]["section"] > 0
    assert meta["figures_rows"] >= 1
    # Registered in the (isolated) session registry + retrievable docs.
    assert "TSLA" in registry.session_companies()
    docs = registry.session_documents()
    assert len(docs) == 1 and docs[0].metadata["ticker"] == "TSLA"
    figs = [f for f in registry.session_figures() if f["ticker"] == "TSLA"]
    assert any(f["fiscal_year"] == 2024 and f["value_usd"] == 97_690_000_000
               for f in figs)


def test_tesla_question_refuses_before_ingest_answers_after():
    before = run(QUESTION)
    assert before["refused"] is True
    assert before["citations"] == []

    ingest_company("TSLA", allow_network=False)

    after = run(QUESTION)
    assert after["refused"] is False
    assert after["citations"], "expected a cited answer after ingest"
    assert "97,690,000,000" in after["answer"]
    # And with the corpus pinned to bundled, it must refuse again —
    # session companies never leak into the measured snapshot.
    bundled_only = run(QUESTION, corpus="bundled")
    assert bundled_only["refused"] is True
    assert bundled_only["citations"] == []
