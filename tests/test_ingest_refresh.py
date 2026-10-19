"""Feature 7 follow-up — re-adding a session company refreshes it.

Before this fix, ``ingest_company`` short-circuited any company
already in the session registry with "already in the corpus — nothing
to fetch", so a newer 10-K on EDGAR was never picked up. Now a re-add
checks EDGAR (via an injectable meta function — no live SEC calls in
tests): same accession -> "already current"; new accession -> full
re-ingest replacing the registry entry. Adding a BUNDLED ticker live
(frozen snapshot untouched, session copy registered alongside it) is
covered in test_ingest_bundled_add.py.
"""

import pytest

from src import registry, session_ingest
from src.session_ingest import create_job, ingest_company

OLD_ACCESSION = "0001628280-26-003952"   # the committed TSLA precache
NEW_ACCESSION = "0001628280-27-000777"

_NEW_FILING_TEXT = """Item 1. Business
Tesla, Inc. designs, develops, manufactures, leases and sells high-performance
fully electric vehicles, energy generation and storage systems. This refreshed
filing text stands in for the newer 10-K in tests.

Item 1A. Risk Factors
Our financial condition and operating results are subject to a number of risks,
including competition in the electric vehicle market and supply constraints.

Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Total revenues were $97,690 million for the year ended December 31, 2024,
an increase compared to the prior year, driven by vehicle deliveries.

Item 8. Financial Statements and Supplementary Data
The consolidated statements of operations are incorporated by reference
in this refreshed test filing.
"""


def _meta(accession: str, filed: str = "2027-01-28") -> dict:
    return {"ticker": "TSLA", "accession": accession,
            "primary_document": "tsla-10k.htm", "filed": filed,
            "report_date": "2026-12-31"}


@pytest.fixture()
def _ingested_tsla():
    ingest_company("TSLA", allow_network=False)
    assert registry.session_companies()["TSLA"]["accession"] == OLD_ACCESSION


def _stage(job: dict, name: str) -> dict:
    return next(s for s in job["stages"] if s["name"] == name)


def test_readd_with_same_accession_reports_already_current(_ingested_tsla,
                                                           monkeypatch):
    def _no_fetch(*a, **k):  # a refresh that decides current must not download
        raise AssertionError("fetch must not run when the accession is current")
    monkeypatch.setattr(session_ingest, "fetch_filing_text_for", _no_fetch)

    job = create_job("TSLA")
    result = ingest_company("TSLA", job=job,
                            latest_meta_fn=lambda cik, ticker: _meta(OLD_ACCESSION, "2026-01-29"))
    assert result["accession"] == OLD_ACCESSION
    fetch = _stage(job, "fetch")
    assert fetch["status"] == "done"
    assert "already current" in fetch["detail"]
    assert OLD_ACCESSION in fetch["detail"]
    assert job["stage"] == "ready"


def test_readd_with_newer_accession_reingests_and_replaces(_ingested_tsla,
                                                            monkeypatch):
    monkeypatch.setattr(
        session_ingest, "fetch_filing_text_for",
        lambda cik, ticker, use_cache=True: (_NEW_FILING_TEXT, _meta(NEW_ACCESSION)))
    monkeypatch.setattr(
        session_ingest, "fetch_company_facts",
        lambda cik, ticker="": {"facts": {}})

    job = create_job("TSLA")
    result = ingest_company(
        "TSLA", job=job,
        latest_meta_fn=lambda cik, ticker: _meta(NEW_ACCESSION))
    assert result["accession"] == NEW_ACCESSION
    assert registry.session_companies()["TSLA"]["accession"] == NEW_ACCESSION
    fetch = _stage(job, "fetch")
    assert NEW_ACCESSION in fetch["detail"]
    assert OLD_ACCESSION in fetch["detail"]
    assert "replac" in fetch["detail"].lower()
    # The replacement text is what chat now retrieves.
    docs = registry.session_documents()
    assert len(docs) == 1 and "refreshed" in docs[0].page_content
