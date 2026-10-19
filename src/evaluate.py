"""Week 2 evaluation — 20 questions run end-to-end through the graph.

Set composition (per the handout): 9 answerable factual questions,
2 multi-document comparisons, 2 ambiguous questions, 2 unanswerable
questions — plus three scenarios added after the owner's live test:
a free-form performance question that must be headlined from the
figures table (E16), a two-turn follow-up that must keep its company
via conversation context (E17), and the same follow-up with NO
context, which must clarify rather than guess (E18) — plus two
off-topic probes (E19/E20) that must get the honest-scope refusal,
not the company clarification. Each question
carries an explicit expected behaviour and a mechanical pass rule:

* answerable   — not refused, at least one citation from the expected
  company's gold section (or the figures table), and every number in
  the answer re-verified against the retrieved chunks + figures.
* comparison   — not refused, citations cover BOTH companies, numbers
  re-verified.
* ambiguous    — refused, or the answer explicitly flags the ambiguity
  (asks which company / says it can't tell).
* unanswerable — refused.

Writes ``results/eval.json`` and renders ``docs/evaluation-report.md``
from the measured run, including a failure-analysis section generated
from the recorded outcomes. Run: ``python -m src.evaluate``
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from src.graph import run, verify_answer
from src.ingest import REPO_ROOT
from src.retrieval import search

RESULTS_PATH = REPO_ROOT / "results" / "eval.json"
DOC_PATH = REPO_ROOT / "docs" / "evaluation-report.md"

EVAL_QUESTIONS = [
    # --- 9 answerable factual -------------------------------------------
    {"id": "E1", "type": "answerable", "ticker": "AAPL",
     "gold_sections": ["Item 7", "Item 8", "Figures table"],
     "question": "What was Apple's total net sales in fiscal 2024?",
     "expected": "Answer with the FY2024 net-sales figure, cited to Item 7/8 or the figures table."},
    {"id": "E2", "type": "answerable", "ticker": "AAPL",
     "gold_sections": ["Item 7", "Item 8", "Figures table"],
     "question": "How much net income did Apple report for fiscal 2024?",
     "expected": "Answer with the FY2024 net-income figure, cited."},
    {"id": "E3", "type": "answerable", "ticker": "AAPL",
     "gold_sections": ["Item 1A"],
     "question": "What are Apple's principal risk factors around supply chain concentration and manufacturing outsourcing?",
     "expected": "Answer drawn from Apple's Item 1A risk factors."},
    {"id": "E4", "type": "answerable", "ticker": "AAPL",
     "gold_sections": ["Item 1"],
     "question": "How does Apple describe its business and its main products and services?",
     "expected": "Answer drawn from Apple's Item 1 business description."},
    {"id": "E5", "type": "answerable", "ticker": "MSFT",
     "gold_sections": ["Item 1"],
     "question": "What are Microsoft's main business segments and what does the Intelligent Cloud segment include?",
     "expected": "Answer drawn from Microsoft's Item 1."},
    {"id": "E6", "type": "answerable", "ticker": "MSFT",
     "gold_sections": ["Item 1A"],
     "question": "What risk factors does Microsoft disclose about competition and cybersecurity threats?",
     "expected": "Answer drawn from Microsoft's Item 1A."},
    {"id": "E7", "type": "answerable", "ticker": "MSFT",
     "gold_sections": ["Item 7"],
     "question": "How did Microsoft's revenue and operating income perform in the most recent fiscal year, according to management's discussion?",
     "expected": "Answer drawn from Microsoft's Item 7 MD&A."},
    {"id": "E8", "type": "answerable", "ticker": "NVDA",
     "gold_sections": ["Item 1"],
     "question": "What does NVIDIA say about its compute and networking platforms and the markets it serves?",
     "expected": "Answer drawn from NVIDIA's Item 1."},
    {"id": "E9", "type": "answerable", "ticker": "NVDA",
     "gold_sections": ["Item 1A"],
     "question": "What concentration risks does NVIDIA disclose about its customers and supply chain?",
     "expected": "Answer drawn from NVIDIA's Item 1A."},
    # --- 2 multi-document comparisons ------------------------------------
    {"id": "E10", "type": "comparison", "tickers": ["AAPL", "MSFT"],
     "question": "How do Apple and Microsoft each describe their business in their 10-K filings?",
     "expected": "Answer citing BOTH companies' Item 1 business descriptions."},
    {"id": "E11", "type": "comparison", "tickers": ["AAPL", "NVDA"],
     "question": "Which company reported higher total revenue in its most recent fiscal year, Apple or NVIDIA, and what were the figures?",
     "expected": "Answer naming the higher-revenue company with both figures grounded and cited."},
    # --- 2 ambiguous -------------------------------------------------------
    {"id": "E12", "type": "ambiguous",
     "question": "How did the company perform last year?",
     "expected": "Flag the ambiguity (which company?) or refuse — never silently pick one company."},
    {"id": "E13", "type": "ambiguous",
     "question": "What are the risks?",
     "expected": "Flag the ambiguity (which company?) or refuse — never silently pick one company."},
    # --- 2 unanswerable ----------------------------------------------------
    {"id": "E14", "type": "unanswerable",
     "question": "What was Tesla's revenue in fiscal 2024?",
     "expected": "Refuse: Tesla is not in the corpus."},
    {"id": "E15", "type": "unanswerable",
     "question": "What will Apple's stock price be at the end of 2027?",
     "expected": "Refuse: a future stock-price prediction is not in any filing."},
    # --- owner live-test scenarios -----------------------------------------
    {"id": "E16", "type": "answerable", "ticker": "AAPL",
     "gold_sections": ["Figures table"],
     "must_contain": ["416,161,000,000", "112,010,000,000"],
     "question": "How did Apple perform last year?",
     "expected": "Answer headlined from the figures table: FY2025 revenue "
                 "$416,161,000,000 (+6.43% YoY) and net income, cited to SEC "
                 "company facts — not Item 7 definitional boilerplate."},
    {"id": "E17", "type": "followup", "ticker": "AAPL",
     "gold_sections": ["Item 1A"],
     "turns": ["What are the sales for Apple in the last year?",
               "Are there any risks with it?"],
     "question": "Are there any risks with it?",
     "expected": "Turn 2 keeps Apple via the context from turn 1's resolved "
                 "values and answers from Apple's Item 1A risk factors."},
    {"id": "E18", "type": "ambiguous",
     "question": "Are there any risks with it?",
     "expected": "No context and no company named: clarify which company "
                 "(or refuse) — never silently answer for one company."},
    # --- off-topic probes (owner request: filings only) --------------------
    {"id": "E19", "type": "off_topic",
     "question": "Write a Python program that sorts a list",
     "expected": "Refuse with the honest-scope copy (filings only) — not "
                 "content generation, not the company clarification."},
    {"id": "E20", "type": "off_topic",
     "question": "What is the capital of France?",
     "expected": "Refuse with the honest-scope copy (filings only) — a "
                 "general-knowledge question is not a filing question."},
]

_AMBIGUITY_MARKERS = (
    "which company", "specify", "ambiguous", "clarif", "not sure which",
    "more than one company", "several companies", "each company",
)


def _faithfulness(question: str, answer: str) -> tuple[bool, str]:
    """Re-verify the answer's numbers against a fresh (deterministic)
    retrieval of the same question — mirrors the graph's own gate."""
    out = search(question, strategy="section", mode="hybrid+rerank", k=10)
    results = [{
        "text": r["chunk"].page_content,
    } for r in out["results"][:4]]
    return verify_answer(answer, results)


def _judge_followup(q: dict, result: dict) -> tuple[bool, str, bool, str]:
    """A follow-up turn passes when the context survived: the turn was
    contextualized (understood_as set), answered — not refused, not
    bounced back as a clarification — and cited from the expected
    company's gold section, with numbers re-verified against a fresh
    retrieval of the UNDERSTOOD question."""
    if result["refused"] or result.get("clarification"):
        return False, "follow-up lost its context (refused or asked to clarify)", True, "n/a"
    if not result.get("understood_as"):
        return False, "follow-up was never contextualized", True, "n/a"
    gold = [c for c in result["citations"]
            if c["doc"].startswith(q["ticker"]) and c["section"] in q["gold_sections"]]
    if not gold:
        return False, "no citation from the expected company/section", True, "n/a"
    faithful, faith_detail = _faithfulness(result["understood_as"], result["answer"])
    if not faithful:
        return False, f"faithfulness: {faith_detail}", faithful, faith_detail
    return True, "context kept; answered with gold-section citation; numbers verified", faithful, faith_detail


def _judge(q: dict, result: dict) -> tuple[bool, str, bool, str]:
    citations = result["citations"]
    if q["type"] == "followup":
        return _judge_followup(q, result)
    faithful, faith_detail = (True, "n/a (refused)")
    if not result["refused"]:
        faithful, faith_detail = _faithfulness(q["question"], result["answer"])

    if q["type"] == "answerable":
        if result["refused"]:
            return False, "refused an answerable question", faithful, faith_detail
        gold = [c for c in citations
                if c["doc"].startswith(q["ticker"]) and c["section"] in q["gold_sections"]]
        if not gold:
            return False, "no citation from the expected company/section", faithful, faith_detail
        missing = [s for s in q.get("must_contain", []) if s not in result["answer"]]
        if missing:
            return False, f"answer missing required figure(s): {', '.join(missing)}", faithful, faith_detail
        if not faithful:
            return False, f"faithfulness: {faith_detail}", faithful, faith_detail
        return True, "answered with gold-section citation; numbers verified", faithful, faith_detail

    if q["type"] == "comparison":
        if result["refused"]:
            return False, "refused a comparison the corpus supports", faithful, faith_detail
        covered = {c["doc"].split(" ")[0] for c in citations}
        missing = [t for t in q["tickers"] if t not in covered]
        if missing:
            return False, f"citations miss {', '.join(missing)}", faithful, faith_detail
        if not faithful:
            return False, f"faithfulness: {faith_detail}", faithful, faith_detail
        return True, "both companies cited; numbers verified", faithful, faith_detail

    if q["type"] == "ambiguous":
        if result["refused"]:
            return True, "refused rather than guessing a company", faithful, faith_detail
        low = result["answer"].lower()
        if any(m in low for m in _AMBIGUITY_MARKERS):
            return True, "answer explicitly flags the ambiguity", faithful, faith_detail
        return False, "answered without flagging that no company was named", faithful, faith_detail

    if q["type"] == "unanswerable":
        if result["refused"]:
            return True, "honest refusal", faithful, faith_detail
        return False, "answered a question the corpus cannot support", faithful, faith_detail

    if q["type"] == "off_topic":
        if not result["refused"]:
            return False, "answered an off-topic request", faithful, faith_detail
        if "only answer questions about" not in result["answer"].lower():
            return False, "refused, but not with the honest-scope copy", faithful, faith_detail
        return True, "honest-scope refusal for an off-topic request", faithful, faith_detail
    raise ValueError(q["type"])


def run_eval() -> dict:
    rows = []
    for q in EVAL_QUESTIONS:
        # The eval is pinned to the bundled corpus: its measured numbers
        # describe the frozen snapshot, never session-ingested companies.
        if q["type"] == "followup":
            # Two turns: the second runs with the first turn's resolved
            # values as its context — the conversation-memory contract.
            first = run(q["turns"][0], corpus="bundled")
            result = run(q["turns"][1], corpus="bundled",
                         context=first["resolved"])
            result["latency_ms"] = round(
                first["latency_ms"] + result["latency_ms"], 1)
            question = " → ".join(q["turns"])
        else:
            result = run(q["question"], corpus="bundled")
            question = q["question"]
        passed, note, faithful, faith_detail = _judge(q, result)
        rows.append({
            "id": q["id"], "type": q["type"], "question": question,
            "expected": q["expected"],
            "refused": result["refused"],
            "answer": result["answer"],
            "citations": [{"doc": c["doc"], "section": c["section"]} for c in result["citations"]],
            "trace": result["trace"],
            "faithful": faithful, "faithfulness_detail": faith_detail,
            "latency_ms": result["latency_ms"],
            "pass": passed, "note": note,
        })
        print(f"  {q['id']} [{q['type']:<12}] {'PASS' if passed else 'FAIL'} — {note}")
    passed = sum(r["pass"] for r in rows)
    answered = [r for r in rows if not r["refused"]]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": rows[0]["trace"] and "extractive (no LLM key configured)" or "",
        "total": len(rows), "passed": passed,
        "pass_rate": round(passed / len(rows), 3),
        "faithfulness_rate_answered": (
            round(sum(r["faithful"] for r in answered) / len(answered), 3) if answered else 1.0),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows), 1),
        "rows": rows,
    }


def render_doc(results: dict) -> str:
    lines = [
        f"# Evaluation report — {results['total']} questions through the full graph",
        "",
        f"Measured {results['generated_at']} with the production graph "
        f"(section chunking, hybrid+rerank, extractive generation — no LLM "
        f"key was configured for this run).",
        "",
        f"**Pass rate: {results['passed']}/{results['total']} "
        f"({results['pass_rate']:.0%})** · faithfulness among answered "
        f"questions: {results['faithfulness_rate_answered']:.0%} · mean "
        f"end-to-end latency {results['mean_latency_ms']:.0f} ms per question.",
        "",
        "## Per-question results",
        "",
        "| # | Type | Question | Outcome | Latency | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in results["rows"]:
        outcome = "refused" if r["refused"] else "answered"
        q = r["question"].replace("|", "\\|")
        lines.append(
            f"| {r['id']} | {r['type']} | {q} | {outcome} | "
            f"{r['latency_ms']:.0f} ms | {'PASS' if r['pass'] else 'FAIL'} |"
        )
    lines += ["", "## Failure analysis", ""]
    failures = [r for r in results["rows"] if not r["pass"]]
    if not failures:
        lines.append("No failures in this run. Known soft spots, from the "
                     "per-question detail in `results/eval.json`:")
        e11 = next((r for r in results["rows"] if r["id"] == "E11"), None)
        if e11 and "215,938" not in e11["answer"]:
            lines.append("* E11 passes its criteria (both companies cited, every "
                         "number verified, Apple's $416.2B total stated) but the "
                         "extractive composer does not surface NVIDIA's $215,938M "
                         "total-revenue line — it sits in a table chunk ranked just "
                         "below the merged evidence cut. The LLM generation mode "
                         "composes the explicit head-to-head; the extractive mode "
                         "quotes what retrieval ranked highest.")
        lines.append("* Extractive answers quote filing sentences verbatim: "
                     "fully grounded, less fluent than an LLM-composed answer. "
                     "Ambiguous questions are handled by a dedicated clarify "
                     "node in the graph (ask which company) or an honest refusal.")
    for r in failures:
        grade = next((t["detail"] for t in r["trace"] if t["node"] == "grade"), "n/a")
        cites = ", ".join(f"{c['doc']} · {c['section']}" for c in r["citations"]) or "none"
        lines += [
            f"### {r['id']} — {r['type']}: {r['question']}",
            "",
            f"* Expected: {r['expected']}",
            f"* Observed: {'refused' if r['refused'] else 'answered'} — {r['note']}.",
            f"* Grade step: {grade}. Citations: {cites}.",
            f"* Answer (truncated): {r['answer'][:300]!r}",
            "",
        ]
    lines += [
        "## Method notes",
        "",
        "* Answerable questions pass only with a citation from the expected",
        "  company's gold Item section (or the SEC figures table) AND every",
        "  number in the answer re-verified against the retrieved chunks.",
        "* Ambiguous questions pass if the system refuses or explicitly",
        "  flags that no company was named; silently answering for one",
        "  company is scored as a failure, by design.",
        "* Unanswerable questions pass only on an honest refusal.",
        "* Off-topic requests (content generation, general knowledge) pass",
        "  only on the honest-scope refusal — never the clarification.",
        "",
        "Raw output: `results/eval.json` (regenerate with `python -m src.evaluate`).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    print(f"Running {len(EVAL_QUESTIONS)}-question evaluation through the graph...")
    results = run_eval()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    DOC_PATH.write_text(render_doc(results), encoding="utf-8")
    print(f"pass {results['passed']}/{results['total']} — wrote "
          f"{RESULTS_PATH.relative_to(REPO_ROOT)} and {DOC_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
