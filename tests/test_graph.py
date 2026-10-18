"""The graph end to end (extractive mode — conftest scrubs all LLM
keys): a cited answer for an answerable question, honest refusals for
an out-of-corpus company and a future prediction, and clarification
for a company-less question."""

from src.graph import run


def test_answerable_question_returns_cited_grounded_answer():
    result = run("What was Apple's total net sales in fiscal 2024?")
    assert result["mode"] == "extractive"
    assert result["refused"] is False
    assert "391,035" in result["answer"]
    assert result["citations"], "an answer without citations is a failure here"
    cited = {(c["doc"], c["section"]) for c in result["citations"]}
    assert any(doc.startswith("AAPL") for doc, _ in cited)
    nodes = [t["node"] for t in result["trace"]]
    assert nodes[0] == "scope" and "retrieve" in nodes and "verify" in nodes


def test_out_of_corpus_company_is_refused():
    result = run("What was Tesla's revenue in fiscal 2024?")
    assert result["refused"] is True
    assert result["citations"] == []


def test_future_prediction_is_refused():
    result = run("What will Apple's stock price be at the end of 2027?")
    assert result["refused"] is True
    assert result["citations"] == []


def test_company_less_question_clarifies_or_refuses():
    result = run("What are the risks?")
    # Never silently answer from one company's filing.
    assert result["clarification"] is True or result["refused"] is True
    assert result["citations"] == []
