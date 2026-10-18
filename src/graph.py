"""LangGraph application — the question-answering graph.

Flow::

    retrieve ──► grade ──┬─ relevant ───────────────► generate ──► verify ─┬─ ok ──► END
                         │                                               └─ fail ─► generate (once) ─► verify ─► refuse/END
                         ├─ weak, no rewrite yet ──► rewrite ──► retrieve
                         └─ weak, rewrite spent ───► refuse ──► END

* **retrieve** — hybrid retrieval (mode selectable) over the chosen
  chunking strategy.
* **grade** — an independent relevance check: the cross-encoder scores
  the question against the top chunks (threshold 0.0 on the ms-marco
  logit); when an LLM is configured it must also judge at least one
  chunk relevant. Weak evidence never reaches generation.
* **rewrite** — one query rewrite (LLM if configured, else a
  deterministic keyword expansion), then re-retrieve.
* **refuse** — the honest path: "I couldn't find this in the filings",
  with the corpus scope stated. A RAG app that hallucinates when
  retrieval fails is worse than one that refuses.
* **generate** — provider-pluggable: ChatOpenAI (OpenAI, or Nebius via
  its OpenAI-compatible endpoint) when a key is configured, otherwise
  an extractive composer that assembles the answer from the retrieved
  chunks and the figures table, verbatim, with citations.
* **verify** — every number in the draft answer must be grounded in a
  cited chunk or the figures table (exact, or a millions/billions
  scaling of a grounded value). One regeneration is allowed; a second
  failure refuses rather than presenting unverified figures.

Run CLI: ``python -m src.graph "What was Apple's net income in fiscal 2024?"``
"""

from __future__ import annotations

import argparse
import json
import operator
import os
import re
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from src.ingest import figures_as_text, load_figures
from src.retrieval import get_reranker, search

load_dotenv()

GRADE_THRESHOLD = 0.0  # ms-marco cross-encoder logit; >0 reads "relevant"
REFUSAL_ANSWER = (
    "I couldn't find this in the filings I have access to. My corpus "
    "covers excerpts of the most recent 10-K filings for Apple, "
    "Microsoft, and NVIDIA (business, risk factors, MD&A, and financial "
    "statements) plus Apple's SEC company-facts figures — the answer "
    "isn't in that material, so I won't guess at it."
)
VERIFY_REFUSAL_ANSWER = (
    "I found related material in the filings, but I couldn't verify "
    "every figure in the draft answer against the cited sources, so "
    "I won't present it. Please try narrowing the question."
)

SYSTEM_PROMPT = """You are Market Buddy, a financial-documents assistant.
Answer the question using ONLY the context provided below, which comes
from real SEC 10-K filings and SEC company-facts data.

Rules:
- State figures exactly as they appear in the context (filing tables are
  in millions unless the context says otherwise).
- Cite every claim inline like [AAPL 10-K FY2024 · Item 7].
- If the context does not contain the answer, say so plainly instead of
  guessing. Never use outside knowledge."""

REWRITE_PROMPT = """Rewrite this question into a search query for SEC 10-K
filings. Keep the company, the figures or topics asked about, and add
filing vocabulary (e.g. "total net sales", "risk factors", "Item 7").
Return only the rewritten query.

Question: {question}"""

GRADE_PROMPT = """Does this filing excerpt help answer the question?
Reply with exactly "yes" or "no".

Question: {question}

Excerpt:
{excerpt}"""


# ---------------------------------------------------------------------------
# LLM (provider-pluggable)
# ---------------------------------------------------------------------------

def get_llm():
    """ChatOpenAI for Nebius or OpenAI when a key is set, else None."""
    from langchain_openai import ChatOpenAI

    if os.environ.get("NEBIUS_API_KEY"):
        return ChatOpenAI(
            api_key=os.environ["NEBIUS_API_KEY"],
            base_url="https://api.studio.nebius.com/v1/",
            model=os.environ.get("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            temperature=0,
        )
    if os.environ.get("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    return None


def llm_mode() -> str:
    return "llm" if (os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY")) else "extractive"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    question: str
    strategy: str
    retrieval_mode: str
    query: str
    results: list[dict]
    grade_score: float
    relevant: bool
    rewrites: int
    generations: int
    answer: str
    citations: list[dict]
    refused: bool
    verify_ok: bool
    verify_detail: str
    trace: Annotated[list[dict], operator.add]


def _trace(node: str, started: float, detail: str) -> list[dict]:
    return [{"node": node, "latency_ms": round((time.perf_counter() - started) * 1000, 1),
             "detail": detail}]


def _serialize(results: list[dict]) -> list[dict]:
    out = []
    for r in results:
        m = r["chunk"].metadata
        out.append({
            "text": r["chunk"].page_content,
            "score": r["score"],
            "signals": r.get("signals", {}),
            "doc": m.get("doc_label", ""),
            "ticker": m.get("ticker", ""),
            "section": m.get("section", ""),
            "char_start": m.get("char_start", 0),
            "char_end": m.get("char_end", 0),
            "accession": m.get("accession", ""),
        })
    return out


def _citations_from(results: list[dict], limit: int = 4) -> list[dict]:
    citations = []
    for r in results[:limit]:
        citations.append({
            "doc": r["doc"],
            "section": r["section"],
            "span": [r["char_start"], r["char_end"]],
            "excerpt": r["text"][:320].replace("\n", " "),
            "score": round(float(r["score"]), 3),
        })
    return citations


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve_node(state: GraphState) -> dict:
    started = time.perf_counter()
    out = search(state["query"], strategy=state.get("strategy", "section"),
                 mode=state.get("retrieval_mode", "hybrid+rerank"), k=10)
    results = _serialize(out["results"])
    top = results[0]["score"] if results else float("nan")
    return {
        "results": results,
        "trace": _trace("retrieve", started,
                        f"{len(results)} candidates via {state.get('retrieval_mode', 'hybrid+rerank')} "
                        f"on {state.get('strategy', 'section')} chunks; top score {top:.3f}"),
    }


def grade_node(state: GraphState) -> dict:
    started = time.perf_counter()
    results = state.get("results", [])
    if not results:
        return {"relevant": False, "grade_score": float("-inf"),
                "trace": _trace("grade", started, "no candidates to grade")}
    reranker = get_reranker()
    top_texts = [r["text"] for r in results[:3]]
    scores = [float(s) for s in reranker.rerank(state["question"], top_texts)]
    best = max(scores) if scores else float("-inf")
    relevant = best >= GRADE_THRESHOLD
    detail = f"cross-encoder top score {best:.2f} (threshold {GRADE_THRESHOLD})"
    llm = get_llm()
    if relevant and llm is not None:
        verdicts = []
        for r, s in zip(results[:3], scores):
            if s < GRADE_THRESHOLD:
                continue
            resp = llm.invoke(GRADE_PROMPT.format(
                question=state["question"], excerpt=r["text"][:1200]))
            verdicts.append(str(resp.content).strip().lower().startswith("y"))
        if verdicts and not any(verdicts):
            relevant = False
            detail += "; LLM grade overruled: no chunk judged relevant"
        else:
            detail += "; LLM grade concurred"
    return {"relevant": relevant, "grade_score": best,
            "trace": _trace("grade", started, detail)}


_STOPWORDS = {"what", "was", "were", "is", "are", "did", "does", "how", "much",
              "many", "the", "a", "an", "in", "of", "for", "to", "and", "or",
              "about", "say", "says", "describe", "company", "please"}


def _heuristic_rewrite(question: str) -> str:
    words = re.findall(r"[A-Za-z0-9$.,%'-]+", question)
    kept = [w for w in words if w.lower().strip("?") not in _STOPWORDS]
    tickers = [t for t in ("Apple", "Microsoft", "NVIDIA", "AAPL", "MSFT", "NVDA")
               if t.lower() in question.lower()]
    return " ".join(kept + tickers + ["Form 10-K annual report"]).strip()


def rewrite_node(state: GraphState) -> dict:
    started = time.perf_counter()
    llm = get_llm()
    if llm is not None:
        resp = llm.invoke(REWRITE_PROMPT.format(question=state["question"]))
        new_query = str(resp.content).strip()
        how = "LLM rewrite"
    else:
        new_query = _heuristic_rewrite(state["question"])
        how = "keyword rewrite"
    return {"query": new_query, "rewrites": state.get("rewrites", 0) + 1,
            "trace": _trace("rewrite", started, f"{how}: {new_query!r}")}


def refuse_node(state: GraphState) -> dict:
    started = time.perf_counter()
    answer = VERIFY_REFUSAL_ANSWER if state.get("verify_detail") else REFUSAL_ANSWER
    return {"refused": True, "answer": answer, "citations": [],
            "trace": _trace("refuse", started, "evidence insufficient or unverifiable — honest refusal")}


_FIGURE_TRIGGER = re.compile(r"(revenue|net sales|net income|income)", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _figure_sentence(state: GraphState) -> tuple[str, dict] | None:
    """A grounded figures-table sentence when the question asks for one."""
    q = state["question"]
    if "AAPL" not in q and "Apple" not in q:
        return None
    if not _FIGURE_TRIGGER.search(q):
        return None
    years = _YEAR_RE.findall(q)
    wants_income = "income" in q.lower()
    candidates = [
        f for f in load_figures()
        if wants_income == (f["concept"] == "Net income")
        and (not years or str(f["fiscal_year"]) in years)
    ]
    if not candidates:
        return None
    # An explicit year in the question wins; otherwise the most recent
    # fiscal year (never the oldest row in the table).
    f = max(candidates, key=lambda c: c["fiscal_year"])
    line = (f"{f['ticker']} reported {f['concept'].lower()} of "
            f"${f['value_usd']:,} for fiscal year {f['fiscal_year']} "
            f"(period ended {f['period_end']}).")
    citation = {
        "doc": "AAPL · SEC company facts",
        "section": "Figures table",
        "span": [0, 0],
        "excerpt": figures_as_text([f]),
        "score": 0.0,
    }
    return line, citation


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_COMPANY_ALIASES = {"apple": "AAPL", "aapl": "AAPL", "microsoft": "MSFT",
                    "msft": "MSFT", "nvidia": "NVDA", "nvda": "NVDA"}


def _mentioned_tickers(question: str) -> set[str]:
    q = question.lower()
    return {t for alias, t in _COMPANY_ALIASES.items() if alias in q}


def _extractive_answer(state: GraphState) -> tuple[str, list[dict]]:
    results = state.get("results", [])[:4]
    mentioned = _mentioned_tickers(state["question"])
    if mentioned and any(r["ticker"] in mentioned for r in results):
        # A question about one company must not be answered with another
        # company's sentences (comparison questions name both, and keep both).
        results = [r for r in results if r["ticker"] in mentioned] or results
    q_words = {w.lower() for w in re.findall(r"[a-z0-9]+", state["question"].lower())
               if w not in _STOPWORDS and len(w) > 2}
    asks_figure = bool(_FIGURE_TRIGGER.search(state["question"]))

    scored: list[tuple[float, str, dict]] = []
    for r in results:
        for sent in _SENT_SPLIT.split(r["text"]):
            sent = " ".join(sent.split())
            if len(sent) < 30 or len(sent) > 420:
                continue
            words = {w.lower() for w in re.findall(r"[a-z0-9]+", sent.lower())}
            overlap = len(q_words & words)
            has_number = bool(re.search(r"\d", sent))
            score = overlap + (2.0 if (asks_figure and has_number) else 0.0)
            if overlap >= 2 or (asks_figure and has_number and overlap >= 1):
                scored.append((score, sent, r))
    scored.sort(key=lambda t: -t[0])

    picked: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for _score, sent, r in scored:
        key = sent[:60]
        if key in seen:
            continue
        seen.add(key)
        picked.append((sent, r))
        if len(picked) >= 5:
            break

    parts: list[str] = []
    used: list[dict] = []
    fig = _figure_sentence(state)
    if fig:
        parts.append(fig[0] + " [AAPL · SEC company facts]")
        used.append(fig[1])
    for sent, r in picked:
        parts.append(f"{sent} [{r['doc']} · {r['section']}]")
        citation = {
            "doc": r["doc"], "section": r["section"],
            "span": [r["char_start"], r["char_end"]],
            "excerpt": r["text"][:320].replace("\n", " "),
            "score": round(float(r["score"]), 3),
        }
        if citation not in used:
            used.append(citation)
    if not parts:  # fallback: lead with the top chunk's opening
        r = results[0]
        parts.append(r["text"][:400].replace("\n", " ") + f" [{r['doc']} · {r['section']}]")
        used.append(_citations_from([r], 1)[0])
    return "\n\n".join(parts), used[:5]


def _llm_answer(state: GraphState, strict_note: str = "") -> tuple[str, list[dict]]:
    llm = get_llm()
    results = state.get("results", [])[:4]
    context_parts = []
    for r in results:
        context_parts.append(f"[{r['doc']} · {r['section']}]\n{r['text'][:2600]}")
    figures = load_figures()
    context_parts.append("[AAPL · SEC company facts]\n" + figures_as_text(figures[-8:]))
    prompt = (
        f"{SYSTEM_PROMPT}\n\nContext:\n" + "\n\n".join(context_parts)
        + f"\n\nQuestion: {state['question']}{strict_note}"
    )
    resp = llm.invoke(prompt)
    return str(resp.content).strip(), _citations_from(results, 4)


def generate_node(state: GraphState) -> dict:
    started = time.perf_counter()
    generations = state.get("generations", 0)
    strict = ""
    if generations:
        strict = ("\n\nImportant: a previous draft contained figures that could "
                  "not be verified. Use only figures copied exactly from the context.")
    llm = get_llm()
    if llm is not None:
        answer, citations = _llm_answer(state, strict)
        mode = "llm"
    else:
        answer, citations = _extractive_answer(state)
        mode = "extractive"
    return {
        "answer": answer, "citations": citations, "refused": False,
        "generations": generations + 1,
        "trace": _trace("generate", started, f"{mode} generation"
                        + (" (regeneration after failed verification)" if generations else "")),
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(
    r"\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?"          # 391,035 / $1,234.50
    r"|\$?\s*\d+\.\d+\s*(?:billion|million|%)?"     # 391.035 billion / 46.9%
    r"|\d+(?:\.\d+)?\s?%"                            # 13 %
    r"|\b(?:19|20)\d{2}\b"                           # years
)


def _token_value(token: str) -> float | None:
    t = token.replace("$", "").replace(",", "").strip()
    mult = 1.0
    if t.lower().endswith("billion"):
        mult, t = 1e9, t[: -len("billion")].strip()
    elif t.lower().endswith("million"):
        mult, t = 1e6, t[: -len("million")].strip()
    t = t.rstrip("%").strip()
    try:
        return float(t) * mult
    except ValueError:
        return None


def verify_answer(answer: str, results: list[dict]) -> tuple[bool, str]:
    """Every number in the answer must trace to a cited chunk or figure."""
    grounded_text = "\n".join(r["text"] for r in results)
    figures = load_figures()
    grounded_text += "\n" + figures_as_text(figures)
    grounded_values: set[float] = set()
    for tok in _NUM_RE.findall(grounded_text):
        v = _token_value(tok)
        if v is not None:
            grounded_values.add(v)
    for f in figures:
        grounded_values.add(float(f["value_usd"]))
        grounded_values.add(float(f["fiscal_year"]))

    ungrounded: list[str] = []
    for tok in _NUM_RE.findall(answer):
        v = _token_value(tok)
        if v is None:
            continue
        ok = False
        for g in grounded_values:
            if g == v:
                ok = True
                break
            # scaled statements: filing tables are in millions, so a chunk
            # number like 391,035 grounds an answer value of 391,035,000,000
            # only via the figures table (exact) — also allow millions /
            # billions rounding of a grounded raw value within 0.6%.
            for scale in (1e6, 1e9):
                if g and abs(v / scale - g) / max(g, 1.0) < 0.006 and v >= scale:
                    ok = True
                    break
            if ok:
                break
        if not ok:
            ungrounded.append(tok)
    if ungrounded:
        return False, f"ungrounded numbers: {', '.join(sorted(set(ungrounded))[:6])}"
    return True, "all numbers in the answer trace to cited chunks or the figures table"


def verify_node(state: GraphState) -> dict:
    started = time.perf_counter()
    ok, detail = verify_answer(state.get("answer", ""), state.get("results", [])[:4])
    return {"verify_ok": ok, "verify_detail": "" if ok else detail,
            "trace": _trace("verify", started, detail)}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _route_after_grade(state: GraphState) -> str:
    if state.get("relevant"):
        return "generate"
    if state.get("rewrites", 0) < 1:
        return "rewrite"
    return "refuse"


def _route_after_verify(state: GraphState) -> str:
    if state.get("verify_ok"):
        return END
    if state.get("generations", 0) < 2:  # one regeneration allowed
        return "generate"
    return "refuse"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify_node)
    g.add_node("refuse", refuse_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route_after_grade,
                            {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("verify", _route_after_verify,
                            {"generate": "generate", "refuse": "refuse", END: END})
    g.add_edge("generate", "verify")
    g.add_edge("refuse", END)
    return g.compile()


_APP = None


def get_app():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def run(question: str, strategy: str = "section",
        retrieval_mode: str = "hybrid+rerank") -> dict:
    """Ask one question; returns the JSON-able result for API/UI/tests."""
    started = time.perf_counter()
    final = get_app().invoke({
        "question": question,
        "query": question,
        "strategy": strategy,
        "retrieval_mode": retrieval_mode,
        "rewrites": 0,
        "generations": 0,
        "trace": [],
    })
    return {
        "question": question,
        "answer": final.get("answer", ""),
        "refused": bool(final.get("refused", False)),
        "citations": final.get("citations", []),
        "trace": final.get("trace", []),
        "mode": llm_mode(),
        "strategy": strategy,
        "retrieval_mode": retrieval_mode,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Market Buddy RAG graph")
    parser.add_argument("question")
    parser.add_argument("--strategy", default="section", choices=["fixed", "section"])
    parser.add_argument("--mode", default="hybrid+rerank",
                        choices=["vector", "hybrid", "hybrid+rerank"])
    args = parser.parse_args()
    result = run(args.question, strategy=args.strategy, retrieval_mode=args.mode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
