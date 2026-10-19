"""Chat must answer from SESSION companies — including a bundled
ticker's live session copy (regression suite for the release-blocking
defect found live after 33c6c33).

Two distinct mechanisms refused questions the corpus could answer:

1. **The LLM grade veto (deployed, LLM mode).** For a company-named
   question the calibrated gate is company match at retrieval rank —
   the windowed cross-encoder scores are recorded, not gated (see
   grade_node's calibration note). But when an LLM is configured, its
   per-chunk verdicts could still overrule that gate — and it judges
   only the first 1,200 chars of each score-eligible chunk. For
   "What did Apple say about Services in its fiscal 2025 MD&A?" the
   eligible chunks were the session Item 7 preamble (no Services
   content) and the FY2024 Item 7 — the judge said "no" to both, the
   graph rewrote, re-judged, and honestly refused a question whose
   evidence was sitting in the corpus.

2. **The clarify route ignored a single-company corpus pin.** With
   corpus="AAPL" the allowed set is exactly {AAPL}, so a company-less
   question has an unambiguous referent — yet grade/routing keyed
   only on companies named in the question TEXT, so "How did Services
   net sales change in fiscal 2025?" was met with "Which company do
   you mean?".
"""

import json

import pytest

from src import graph, registry, session_ingest
from src.graph import grade_node, run
from src.ingest import DATA_DIR
from src.session_ingest import ingest_company

_FILING_TEXT = """Item 1. Business
Apple Inc. designs, manufactures and markets smartphones, personal
computers, tablets, wearables and accessories, and sells a variety of
related services, including AppleCare and the App Store.

Item 1A. Risk Factors
The Company's business requires a highly skilled workforce and depends
on global supply chains subject to disruption and concentration risk.

Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
The Company's fiscal year 2025 spanned 52 weeks. The following
discussion examines the results of operations for the Services
segment in fiscal 2025 compared with fiscal 2024.
Services net sales were $96,169 million during 2025, an increase of
15% compared to 2024, driven by growth across the App Store, cloud
services and AppleCare. Services net sales rose in every geographic
segment during 2025, led by the Americas and Europe. The increase in
Services net sales reflects higher net sales from advertising, the
App Store and cloud services, as the installed base of active
devices reached an all-time high. Services gross margin percentage
increased during 2025 on a more favourable mix of higher-margin
services offerings. The Company expects Services net sales to keep
growing with the installed base, while Services costs rise with
content and infrastructure investment.

Item 8. Financial Statements and Supplementary Data
The consolidated statements of operations show total net sales of
$416,161 million for the year ended September 27, 2025.
"""


def _meta() -> dict:
    return {"ticker": "AAPL", "accession": "0000320193-25-000079",
            "primary_document": "aapl-10k.htm", "filed": "2025-10-31",
            "report_date": "2025-09-27"}


@pytest.fixture()
def _session_aapl(monkeypatch):
    """AAPL ingested as a live session company (faked EDGAR)."""
    monkeypatch.setattr(
        session_ingest, "fetch_filing_text_for",
        lambda cik, ticker, use_cache=True: (_FILING_TEXT, _meta()))
    monkeypatch.setattr(
        session_ingest, "fetch_company_facts",
        lambda cik, ticker="": json.loads(
            (DATA_DIR / "companyfacts_aapl.json").read_text()))
    ingest_company("AAPL", latest_meta_fn=lambda cik, ticker: _meta())
    assert "AAPL" in registry.session_companies()


def test_session_company_answers_in_chat(_session_aapl):
    """The session copy's Item 7 is citable evidence in chat."""
    res = run("What did Apple say about Services in its fiscal 2025 MD&A?",
              corpus="active")
    assert res["refused"] is False
    assert res["clarification"] is False
    assert "96,169" in res["answer"]
    labels = {(c["doc"], c["section"]) for c in res["citations"]}
    assert ("AAPL 10-K FY2025", "Item 7") in labels


def test_pinned_single_company_corpus_answers_without_clarifying(
        _session_aapl):
    """corpus='AAPL' pins the referent: a company-less question about
    the pinned company must be answered, not clarified."""
    res = run("How did Services net sales change in fiscal 2025?",
              corpus="AAPL")
    assert res["clarification"] is False
    assert res["refused"] is False
    assert "96,169" in res["answer"]


class _StubReranker:
    """Every window scores well — company match, not the score, is
    what these tests exercise."""

    def rerank(self, query, texts):
        return [2.0 for _ in texts]


class _NoJudge:
    """An LLM judge that finds no excerpt helpful (the deployed
    behaviour when the judged prefixes don't show the topic)."""

    def invoke(self, prompt):
        class _Resp:
            content = "no"
        return _Resp()


def _aapl_results() -> list[dict]:
    return [{
        "text": "Item 7. Management's Discussion and Analysis. The "
                "Company's fiscal years 2025 and 2024 spanned 52 weeks "
                "each, whereas fiscal year 2023 spanned 53 weeks.",
        "score": 2.0, "signals": {}, "doc": "AAPL 10-K FY2025",
        "ticker": "AAPL", "section": "Item 7",
        "char_start": 0, "char_end": 400, "accession": "",
    }]


def test_llm_grade_cannot_veto_company_matched_evidence(monkeypatch):
    """Company-named question + the named company in the top-3 is a
    PASSING grade (the calibrated gate) — an LLM judge's per-prefix
    'no' must not overrule it."""
    monkeypatch.setattr(graph, "get_reranker", lambda: _StubReranker())
    monkeypatch.setattr(graph, "get_llm", lambda: _NoJudge())
    out = grade_node({
        "question": "What did Apple say about Services in its fiscal "
                    "2025 MD&A?",
        "corpus": "active",
        "results": _aapl_results(),
    })
    assert out["relevant"] is True


def test_llm_grade_still_tightens_company_less_questions(monkeypatch):
    """The veto's legitimate home is the company-less path, where the
    windowed score is the only gate: there the LLM check still
    tightens."""
    monkeypatch.setattr(graph, "get_reranker", lambda: _StubReranker())
    monkeypatch.setattr(graph, "get_llm", lambda: _NoJudge())
    out = grade_node({
        "question": "What are the risks?",
        "corpus": "active",
        "results": _aapl_results(),
    })
    assert out["relevant"] is False
