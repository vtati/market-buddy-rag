"""Conversation context + free-form performance questions.

Defect 1 (owner live-test): the chat was stateless — "What are the
sales for Apple in the last year?" followed by "Are there any risks
with it?" lost Apple entirely. The graph now accepts the previous
turn's ``resolved`` as ``context`` and deterministically rewrites
follow-ups into standalone questions before scope.

Defect 2: "How did Apple perform last year?" returned fiscal-year
definition boilerplate from Item 7 even though the figures table
holds the answer. Performance-intent questions now lead with a
composed figures headline, and the ranking demotes definitional
boilerplate.
"""

from src.graph import _contextualize, run, verify_answer

CTX = {"ticker": "AAPL", "fiscal_year": 2025}


def test_contextualize_risk_followup_matches_spec_example():
    out = _contextualize("Are there any risks with it?", CTX)
    assert out == "What are Apple's principal risk factors?"


def test_contextualize_needs_context():
    assert _contextualize("Are there any risks with it?", None) is None
    assert _contextualize("Are there any risks with it?", {}) is None
    assert _contextualize("Are there any risks with it?",
                          {"ticker": None, "fiscal_year": None}) is None


def test_contextualize_leaves_company_named_questions_alone():
    # A question that names its own company is not a follow-up, even
    # when it differs from the context company.
    assert _contextualize("What are Microsoft's risk factors?", CTX) is None
    assert _contextualize("What was Apple's net income in fiscal 2024?", CTX) is None


def test_contextualize_substitutes_anaphora():
    out = _contextualize("How much revenue did it report?", CTX)
    assert out is not None and "Apple" in out and " it " not in f" {out} "


def test_two_turn_conversation_keeps_the_company():
    first = run("What are the sales for Apple in the last year?")
    assert first["refused"] is False
    assert first["resolved"]["ticker"] == "AAPL"

    second = run("Are there any risks with it?", context=first["resolved"])
    assert second["refused"] is False
    assert second["clarification"] is False
    assert second["understood_as"] and "Apple" in second["understood_as"]
    assert any(c["doc"].startswith("AAPL") and c["section"] == "Item 1A"
               for c in second["citations"]), "follow-up must land in Apple's Item 1A"
    assert [t["node"] for t in second["trace"]][0] == "contextualize"


def test_followup_without_context_still_clarifies_never_guesses():
    # The clarify path specifically — not an answer, not a refusal.
    result = run("Are there any risks with it?")
    assert result["clarification"] is True
    assert result["refused"] is False
    assert result["citations"] == []
    assert result["understood_as"] is None


def test_performance_question_leads_with_figures_headline():
    result = run("How did Apple perform last year?")
    assert result["refused"] is False
    answer = result["answer"]
    # FY2025 headline vs FY2024, from the figures table.
    assert "416,161,000,000" in answer
    assert "391,035,000,000" in answer
    assert "112,010,000,000" in answer
    assert "93,736,000,000" in answer
    assert "6.43%" in answer
    assert any(c["doc"].startswith("AAPL") and c["section"] == "Figures table"
               for c in result["citations"])
    # The screenshot failure mode: definitional boilerplate as the answer.
    assert "52- or 53-week" not in answer
    assert "This Item generally discusses" not in answer


def test_verify_accepts_yoy_percentage_derived_from_figures_table():
    ok, detail = verify_answer(
        "AAPL total revenue was $416,161,000,000 for fiscal year 2025, "
        "up 6.43% from $391,035,000,000 in fiscal year 2024.", [])
    assert ok, detail
