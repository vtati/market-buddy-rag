"""Add a Company: bundled tickers run the REAL pipeline.

Before this fix, adding a company whose ticker is in the bundled
corpus (AAPL/MSFT/NVDA) short-circuited: every stage was marked done
with "part of the bundled corpus snapshot" and nothing was fetched.
The owner expects adding Apple to fetch its LATEST 10-K. The bundled
snapshot is a curated composite (AAPL = FY2024 Items 7/7A/8 + FY2025
Items 1/1A) pinned for the Lab/Eval measurements; a live add now
registers a SESSION copy of the complete latest filing alongside it
(e.g. FY2025 Item 7 MD&A, which the snapshot lacks). The bundled
snapshot itself never changes, and chat corpus=active searches both.
"""

import json

import pytest

from src import registry, session_ingest
from src.ingest import DATA_DIR, load_bundled_documents, load_figures
from src.session_ingest import create_job, ingest_company

AAPL_LATEST = "0000320193-25-000079"  # AAPL FY2025 10-K

_FILING_TEXT = """Item 1. Business
Apple Inc. designs, manufactures and markets smartphones, personal
computers, tablets, wearables and accessories, and sells a variety of
related services, including AppleCare and the App Store.

Item 1A. Risk Factors
The Company's business requires a highly skilled workforce and depends
on global supply chains subject to disruption and concentration risk.

Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Services net sales were $96,169 million during 2025, an increase of
15% compared to 2024, driven by growth across the App Store, cloud
services and AppleCare.

Item 8. Financial Statements and Supplementary Data
The consolidated statements of operations show total net sales of
$416,161 million for the year ended September 27, 2025.
"""


def _meta(accession: str = AAPL_LATEST, filed: str = "2025-10-31") -> dict:
    return {"ticker": "AAPL", "accession": accession,
            "primary_document": "aapl-10k.htm", "filed": filed,
            "report_date": "2025-09-27"}


def _fake_facts(cik: str, ticker: str = "") -> dict:
    """The real bundled AAPL company-facts payload — a live session
    fetch for AAPL returns these same facts from SEC."""
    return json.loads((DATA_DIR / "companyfacts_aapl.json").read_text())


@pytest.fixture()
def _faked_edgar(monkeypatch):
    monkeypatch.setattr(
        session_ingest, "fetch_filing_text_for",
        lambda cik, ticker, use_cache=True: (_FILING_TEXT, _meta()))
    monkeypatch.setattr(session_ingest, "fetch_company_facts", _fake_facts)


def _stage(job: dict, name: str) -> dict:
    return next(s for s in job["stages"] if s["name"] == name)


def test_bundled_ticker_add_runs_real_pipeline(_faked_edgar):
    bundled_before = [(d.metadata["doc_label"], len(d.page_content))
                      for d in load_bundled_documents()]

    job = create_job("AAPL")
    result = ingest_company(
        "AAPL", job=job,
        latest_meta_fn=lambda cik, ticker: _meta())

    # A real fetch of the latest 10-K, registered as a session company.
    assert result["source"] == "live"
    assert result["accession"] == AAPL_LATEST
    assert result["fiscal_year"] == 2025
    assert "AAPL" in registry.session_companies()
    assert registry.session_companies()["AAPL"]["accession"] == AAPL_LATEST

    # No stage claims the bundled snapshot did the work.
    for s in job["stages"]:
        assert "bundled corpus snapshot" not in s["detail"]
    assert AAPL_LATEST in _stage(job, "fetch")["detail"]
    assert job["stage"] == "ready"

    # The session doc is the complete latest filing — Item 7 MD&A,
    # which the bundled composite does not carry for FY2025.
    docs = [d for d in registry.session_documents()
            if d.metadata["ticker"] == "AAPL"]
    assert len(docs) == 1
    assert docs[0].metadata["doc_label"] == "AAPL 10-K FY2025"
    assert docs[0].metadata["source_file"] == "session:AAPL"
    assert "Services net sales were $96,169 million" in docs[0].page_content

    # The frozen bundled snapshot is untouched.
    bundled_after = [(d.metadata["doc_label"], len(d.page_content))
                     for d in load_bundled_documents()]
    assert bundled_after == bundled_before


def test_figures_dedupe_after_bundled_add(_faked_edgar):
    bundled_rows = load_figures()
    ingest_company("AAPL", latest_meta_fn=lambda cik, ticker: _meta())

    rows = [f for f in registry.all_figures() if f["ticker"] == "AAPL"]
    keys = [(f["ticker"], f["concept"], f["fiscal_year"]) for f in rows]
    assert len(keys) == len(set(keys)), "duplicate figures rows per year"
    assert len(rows) == len(bundled_rows)

    # The market snapshot regrouping sees exactly one row per year too.
    from src import market
    out = market.get_filing_figures("AAPL")
    years = [r["fiscal_year"] for r in out["rows"]]
    assert len(years) == len(set(years))
    fy2024 = next(r for r in out["rows"] if r["fiscal_year"] == 2024)
    assert fy2024["revenue"] == 391_035_000_000


def test_second_add_reports_already_current_against_session_copy(
        _faked_edgar, monkeypatch):
    ingest_company("AAPL", latest_meta_fn=lambda cik, ticker: _meta())

    def _no_fetch(*a, **k):
        raise AssertionError("fetch must not run when the accession is current")
    monkeypatch.setattr(session_ingest, "fetch_filing_text_for", _no_fetch)

    job = create_job("AAPL")
    result = ingest_company(
        "AAPL", job=job,
        latest_meta_fn=lambda cik, ticker: _meta())
    assert result["accession"] == AAPL_LATEST
    ready = _stage(job, "ready")
    assert "already current" in ready["detail"]
    assert AAPL_LATEST in ready["detail"]
    assert job["stage"] == "ready"
