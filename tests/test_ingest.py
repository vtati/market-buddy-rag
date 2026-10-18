"""Ingestion: bundled corpus shape + the SEC figures table values."""

from src.ingest import load_bundled_documents, load_figures


def test_bundled_corpus_has_four_documents():
    docs = load_bundled_documents()
    assert len(docs) == 4
    labels = {d.metadata["doc_label"] for d in docs}
    assert labels == {
        "AAPL 10-K FY2024", "AAPL 10-K FY2025",
        "MSFT 10-K FY2026", "NVDA 10-K FY2026",
    }
    for d in docs:
        assert len(d.page_content) > 20_000  # real filing text, not stubs


def test_figures_table_aapl_values():
    """Guard the period-end grouping fix: the company-facts `fy` field
    is the FILING's year, so figures must be grouped by period end —
    FY2024 revenue is $391,035M (period ended 2024-09-28), not the
    value from whatever filing happened to report it."""
    figures = load_figures()

    def value(year: int, concept: str) -> int:
        rows = [f for f in figures
                if f["fiscal_year"] == year and f["concept"] == concept]
        assert rows, f"missing {concept} for FY{year}"
        return rows[0]["value_usd"]

    assert value(2024, "Total revenue") == 391_035_000_000
    assert value(2024, "Net income") == 93_736_000_000
    assert value(2025, "Total revenue") == 416_161_000_000
    assert value(2025, "Net income") == 112_010_000_000
