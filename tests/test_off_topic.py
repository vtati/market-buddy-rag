"""Item B — off-topic guardrail tightening.

The chat answers ONLY filing questions. Before this fix, a request
with nothing to do with any filing ("Write a Python program…",
"What is the capital of France?") fell through retrieval into the
company-clarification ("Which company do you mean?"), which reads as
evasive. A deterministic scope-stage check now refuses those with
the honest-scope copy — BEFORE the clarification route — while
company-less questions that ARE filing-shaped ("What are the
risks?") still clarify (eval E12/E13/E18 behaviour).
"""

import pytest

from src.graph import _is_off_topic, refuse_node, scope_node


@pytest.mark.parametrize("question", [
    "Write a Python program that sorts a list",
    "Write a short poem about the moon",
    "Compose a haiku about autumn leaves",
    "What is the capital of France?",
    "Tell me a joke",
    "Who won the World Cup in 2018?",
    "How do I bake sourdough bread?",
    "What's the weather going to be tomorrow?",
])
def test_off_topic_requests_detected(question):
    assert _is_off_topic(question, "bundled") is True


@pytest.mark.parametrize("question", [
    "What are the risks?",                    # E13: filing-shaped -> clarify
    "How did the company perform last year?",  # E12: filing-shaped -> clarify
    "Are there any risks with it?",            # E18: filing-shaped -> clarify
    "What was Apple's total net sales in fiscal 2024?",
    "What was Tesla's revenue in fiscal 2024?",  # other-company refusal, not off-topic
    "Should I buy Apple stock? Give me investment advice",
    "Write a summary of Apple's principal risk factors",
    "How did Microsoft's Intelligent Cloud segment perform?",
    "Compare Apple and NVIDIA on revenue",
    "What does the MD&A say about services growth?",
])
def test_filing_shaped_questions_not_off_topic(question):
    assert _is_off_topic(question, "bundled") is False


def test_scope_node_refuses_off_topic_before_clarify_path():
    out = scope_node({"question": "Write a Python program that sorts a list",
                      "corpus": "bundled"})
    assert out["refusal_kind"] == "off_topic"


def test_scope_node_passes_filing_shaped_company_less_question():
    out = scope_node({"question": "What are the risks?", "corpus": "bundled"})
    assert "refusal_kind" not in out


def test_refuse_node_off_topic_copy_names_the_scope():
    out = refuse_node({"question": "What is the capital of France?",
                       "corpus": "bundled", "refusal_kind": "off_topic"})
    assert out["refused"] is True
    assert out["citations"] == []
    assert "only answer questions about" in out["answer"]
    assert "10-K" in out["answer"]
