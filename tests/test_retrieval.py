"""Retrieval: determinism on the fixture corpus + a gold-hit smoke
check. First search in the process builds the FAISS/BM25 indexes and
loads the local models — subsequent ones are fast."""

from src.retrieval import search

QUERY = "What was Apple's total net sales in fiscal 2024?"


def _signature(out: dict) -> list[tuple]:
    return [
        (r["chunk"].metadata["ticker"], r["chunk"].metadata["section"],
         r["chunk"].metadata["char_start"])
        for r in out["results"]
    ]


def test_retrieval_is_deterministic():
    first = search(QUERY, strategy="section", mode="hybrid+rerank", k=4)
    second = search(QUERY, strategy="section", mode="hybrid+rerank", k=4)
    assert _signature(first) == _signature(second)


def test_gold_section_in_top_4():
    out = search(QUERY, strategy="section", mode="hybrid+rerank", k=4)
    hits = [
        r for r in out["results"]
        if r["chunk"].metadata["ticker"] == "AAPL"
        and r["chunk"].metadata["section"] in ("Item 7", "Item 8")
    ]
    assert hits, "expected an AAPL Item 7/8 chunk in the top-4"
    top_text = out["results"][0]["chunk"].page_content
    assert "391,035" in top_text  # the FY2024 total net sales line


def test_ticker_filter_restricts_pool():
    out = search("revenue growth and segment performance",
                 strategy="section", mode="hybrid+rerank", k=4,
                 tickers={"NVDA"})
    assert out["results"]
    assert all(r["chunk"].metadata["ticker"] == "NVDA" for r in out["results"])
