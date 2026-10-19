"""LangGraph application — the question-answering graph.

Flow::

    contextualize (follow-ups rewritten against the previous turn's
    resolved context — rules only, no model call)
            └─► scope ──┬─ prediction question ──► refuse
                        └─ off-topic request ────► refuse (filings only)
            └─► retrieve ──► grade ──┬─ relevant, company named ──► generate ──► verify ─┬─ ok ──► END
                                     │                                                     └─ fail ─► generate (once) ─► verify ─► refuse/END
                                     ├─ relevant, NO company named ──► clarify (which company?)
                                     ├─ weak, no rewrite yet ──► rewrite ──► retrieve
                                     └─ weak, rewrite spent ───► refuse ──► END

Grading uses *windowed* cross-encoder scores (best 700-char window of
each top chunk): absolute ms-marco logits over a whole 4,000-char chunk
systematically underrate relevant prose — the first eval run refused
correct Item 1A retrievals at −2 — while passage-level grading matches
how the model was trained.

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

from src import registry
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
PREDICTION_REFUSAL_ANSWER = (
    "I can't answer that from the filings: they report what has already "
    "happened, and no filing can support a future stock price or a "
    "prediction. Ask me about a reported figure, a business segment, or "
    "a disclosed risk instead."
)
OUT_OF_CORPUS_ANSWER = (
    "That company isn't in the filings I can search. My corpus covers "
    "excerpts of the most recent 10-K filings for Apple, Microsoft, and "
    "NVIDIA plus Apple's SEC company-facts figures — I won't answer "
    "about a company I have no source documents for."
)

_KNOWN_OTHER_COMPANIES = {
    "tesla", "amazon", "alphabet", "google", "meta", "netflix", "amd",
    "intel", "walmart", "disney", "boeing", "ford", "oracle", "adobe",
    "salesforce", "berkshire", "jpmorgan", "exxon", "chevron", "nike",
    "pfizer", "broadcom", "qualcomm", "paypal", "uber", "airbnb",
    "spotify", "shopify", "snowflake", "palantir",
}
_POSSESSIVE_RE = re.compile(r"\b([A-Z][a-z]{2,})'s\b")
_GENERIC_POSSESSIVES = {"Company", "Board", "Year", "Quarter", "Market",
                        "Group", "Fund", "Index", "Bank"}


def _allowed_tickers(corpus: str) -> set[str]:
    """Companies the chat may answer about under a corpus selector:
    "bundled" -> the frozen snapshot; a ticker -> that company only;
    anything else ("active") -> bundled + session-ingested."""
    if corpus == "bundled":
        return set(registry.BUNDLED_TICKERS)
    if corpus in registry.known_tickers():
        return {corpus}
    return registry.known_tickers()


def _other_company_mentioned(question: str, corpus: str = "active") -> bool:
    """True when the question names a company outside the allowed set
    (a well-known company name, or a capitalised possessive like
    "Tesla's" that isn't an allowed company or a generic noun). A
    session-ingested company stops being 'other' the moment it
    registers — the registry, not this list, decides."""
    allowed_aliases = registry.company_aliases(_allowed_tickers(corpus))
    low = question.lower()
    for name in _KNOWN_OTHER_COMPANIES:
        if name in low and name not in allowed_aliases:
            return True
    for m in _POSSESSIVE_RE.finditer(question):
        name = m.group(1)
        if name in _GENERIC_POSSESSIVES:
            continue
        if name.lower() in allowed_aliases:
            continue
        return True
    return False


def _corpus_answers(corpus: str) -> tuple[str, str, str]:
    """(refusal, out-of-corpus, clarify) copy naming the ACTIVE corpus.

    With no session companies involved the original bundled-corpus
    strings are returned verbatim; otherwise the coverage sentence is
    composed from the registry so refusals name what is actually
    searchable right now.
    """
    allowed = _allowed_tickers(corpus)
    if allowed == set(registry.BUNDLED_TICKERS):
        return REFUSAL_ANSWER, OUT_OF_CORPUS_ANSWER, CLARIFY_ANSWER
    listing, fig_clause = registry.coverage_sentence(allowed)
    refusal = (
        "I couldn't find this in the filings I have access to. My corpus "
        f"covers excerpts of the most recent 10-K filings for {listing} "
        "(business, risk factors, MD&A, and financial statements)"
        f"{fig_clause} — the answer isn't in that material, so I won't "
        "guess at it."
    )
    out_of_corpus = (
        "That company isn't in the filings I can search. My corpus covers "
        f"excerpts of the most recent 10-K filings for {listing}"
        f"{fig_clause} — I won't answer about a company I have no source "
        "documents for."
    )
    clarify = (
        f"Which company do you mean? I can search the 10-K filings for "
        f"{listing} — name one, or ask me to compare them, and I'll pull "
        "the relevant section."
    )
    return refusal, out_of_corpus, clarify
CLARIFY_ANSWER = (
    "Which company do you mean? I can search the 10-K filings for "
    "Apple, Microsoft, and NVIDIA — name one, or ask me to compare "
    "them, and I'll pull the relevant section."
)

_PREDICTION_PATTERNS = (
    re.compile(r"\bwill\b[^?]*\b(stock price|share price|price|be worth)\b", re.IGNORECASE),
    re.compile(r"\b(predict|prediction|forecast)\w*\b[^?]*\b(price|stock|revenue)\b", re.IGNORECASE),
    re.compile(r"\bprice target\b", re.IGNORECASE),
    re.compile(r"\bstock price\b[^?]*\b(20\d{2}|future)\b", re.IGNORECASE),
)


def _is_prediction_question(question: str) -> bool:
    return any(p.search(question) for p in _PREDICTION_PATTERNS)


# ---------------------------------------------------------------------------
# Off-topic guardrail — this assistant answers filing questions only
# ---------------------------------------------------------------------------
# Two deterministic shapes, checked in scope BEFORE retrieval (and so
# before the clarify route — answering "Which company do you mean?" to
# "Write a poem about the moon" reads as evasive):
#
# * imperative content generation: a write/compose/generate verb with
#   an artifact noun (program, poem, essay…) and no filing referent;
# * general knowledge / everyday tasks: no company named and no
#   filing-domain vocabulary at all.
#
# The guards matter as much as the triggers: a question naming a
# corpus company is never off-topic here (other scope rules judge
# it), and company-less questions carrying filing vocabulary ("What
# are the risks?", "How did the company perform?") are filing-SHAPED —
# they fall through to the clarify path (eval E12/E13/E18).
_GEN_REQUEST_RE = re.compile(
    r"\b(write|compose|generate|draft|create|produce|make|code|program|"
    r"build|design|draw|paint|sing)\b[^.?!]{0,60}?\b(program|code|script|"
    r"function|poem|story|essay|song|joke|haiku|limerick|letter|e-?mail|"
    r"resume|cover letter|rap|novel|recipe|image|picture|drawing|"
    r"painting|app|website|game|python|javascript|typescript|sql)\b",
    re.IGNORECASE)

_FILING_TERM_STEMS = (
    "risk", "revenue", "sale", "income", "profit", "earn", "perform",
    "grow", "fiscal", "filing", "10-k", "10k", "segment", "business",
    "financial", "margin", "debt", "dividend", "operation", "result",
    "expense", "asset", "liabilit", "customer", "supplier",
    "manufactur", "compet", "product", "service", "md&a", "cash flow",
    "balance sheet", "annual report", "net sales", "supply chain",
)


def _is_off_topic(question: str, corpus: str = "active") -> bool:
    """True when the request has nothing to do with the filings in
    the corpus: a content-generation task or a general-knowledge
    question with no company named and no filing vocabulary."""
    if _mentioned_tickers(question, corpus):
        return False
    if _GEN_REQUEST_RE.search(question):
        return True
    low = question.lower()
    if any(stem in low for stem in _FILING_TERM_STEMS):
        return False
    return True  # no company, no filing vocabulary: not a filing question


def _off_topic_answer(corpus: str) -> str:
    """Honest-scope refusal for off-topic requests: says what this
    assistant DOES answer, naming the active corpus."""
    allowed = _allowed_tickers(corpus)
    if allowed == set(registry.BUNDLED_TICKERS):
        listing = "Apple, Microsoft, and NVIDIA"
    else:
        listing, _fig = registry.coverage_sentence(allowed)
    return (
        "That's outside what I do — I only answer questions about the "
        f"10-K filings in my corpus ({listing}: their business, risk "
        "factors, MD&A and financial statements). Ask me about one of "
        "those filings and I'll pull the relevant section, with "
        "citations."
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
    corpus: str
    context: dict
    understood_as: str
    query: str
    results: list[dict]
    grade_score: float
    relevant: bool
    rewrites: int
    generations: int
    answer: str
    citations: list[dict]
    refused: bool
    refusal_kind: str
    clarification: bool
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
# Conversation context — deterministic follow-up rewriting
# ---------------------------------------------------------------------------

_ANAPHORA_WORDS = {"it", "its", "they", "them", "their", "this", "that"}


def _contextualize(question: str, context: dict | None,
                   corpus: str = "active") -> str | None:
    """Rewrite a follow-up question into a standalone one using the
    previous turn's resolved context — rules only, no model call.

    Returns the rewritten question, or None when no rewrite applies:
    no context ticker, the context ticker isn't in the allowed corpus,
    or the question names a company of its own (then it isn't a
    follow-up at all). A company-less question WITH context is a
    follow-up by definition here — and with NO context it falls
    through to the existing clarify path, never a guess.
    """
    if not context or not context.get("ticker"):
        return None
    ticker = str(context["ticker"]).upper()
    if ticker not in _allowed_tickers(corpus):
        return None
    if _mentioned_tickers(question, corpus):
        return None
    name = registry.short_name(ticker)
    possessive = name if name.endswith("s") else f"{name}'s"
    low = question.lower()

    if "risk" in low:
        return f"What are {possessive} principal risk factors?"

    def _sub(pattern: str, repl: str, text: str) -> str:
        def f(m: re.Match) -> str:
            return repl.capitalize() if m.group(0)[:1].isupper() else repl
        return re.sub(pattern, f, text)

    out = question
    out = _sub(r"\bthe company\b", name, out)
    out = _sub(r"\bits\b", possessive, out)
    out = _sub(r"\btheir\b", possessive, out)
    out = _sub(r"\b(it|they|them|this|that)\b", name, out)
    if out == question:
        # Elliptical with no anaphora to substitute ("And in 2023?"):
        # anchor it to the context company explicitly.
        out = f"Regarding {name}: {question}"
    fy = context.get("fiscal_year")
    if (fy and not _YEAR_RE.search(out) and _FIGURE_TRIGGER.search(out)):
        out = re.sub(r"\?\s*$", "", out).rstrip() + f" in fiscal {fy}?"
    return out


def contextualize_node(state: GraphState) -> dict:
    """Entry step: resolve follow-ups against the conversation context
    BEFORE scope runs, so every downstream node sees a standalone
    question. Silent (no trace entry) when nothing was rewritten."""
    started = time.perf_counter()
    rewritten = _contextualize(state["question"], state.get("context"),
                               state.get("corpus", "active"))
    if not rewritten or rewritten == state["question"]:
        return {}
    return {"question": rewritten, "query": rewritten,
            "understood_as": rewritten,
            "trace": _trace("contextualize", started,
                            f"Understood as: {rewritten}")}


def _resolved_for(question: str, corpus: str) -> dict:
    """What this turn resolved to — sent back to the client and
    accepted as the next turn's context."""
    mentioned = _mentioned_tickers(question, corpus)
    ticker = next(iter(mentioned)) if len(mentioned) == 1 else None
    fiscal_year = None
    years = _YEAR_RE.findall(question)
    if years:
        fiscal_year = int(years[-1])
    elif ticker and re.search(r"\blast year\b|\bmost recent\b|\blatest\b",
                              question, re.IGNORECASE):
        rows = [f for f in registry.all_figures() if f["ticker"] == ticker]
        if rows:
            fiscal_year = max(f["fiscal_year"] for f in rows)
    return {"ticker": ticker, "fiscal_year": fiscal_year}


# ---------------------------------------------------------------------------
# Performance intent — figures headline + ranking adjustment
# ---------------------------------------------------------------------------

_PERF_TRIGGERS = ("perform", "how did", "results", "earnings", "growth",
                  "sales", "revenue", "last year", "most recent")
_BOILERPLATE_MARKERS = ("52- or 53-week", "this item generally discusses",
                        "forward-looking statements")
_DEFINITION_RE = re.compile(r"\b(define|definition|meaning of|what does .{0,40} mean)\b",
                            re.IGNORECASE)
_FIGUREY_RE = re.compile(r"\$\s?\d|\d(?:\.\d+)?\s?%")


def _performance_intent(question: str, corpus: str = "active") -> tuple[str, int] | None:
    """(ticker, fiscal_year) when the question asks how ONE company
    performed and the figures table can headline it: the company needs
    revenue rows for the target year and the year before it. An
    explicit year in the question wins; otherwise the latest fiscal
    year in the table ("last year" / "most recent")."""
    mentioned = _mentioned_tickers(question, corpus)
    if len(mentioned) != 1:
        return None
    low = question.lower()
    if not any(t in low for t in _PERF_TRIGGERS):
        return None
    ticker = next(iter(mentioned))
    rev_years = sorted({f["fiscal_year"] for f in registry.all_figures()
                        if f["ticker"] == ticker
                        and f["concept"] == "Total revenue"})
    if len(rev_years) < 2:
        return None
    years = _YEAR_RE.findall(question)
    target = int(years[-1]) if years and int(years[-1]) in rev_years \
        else rev_years[-1]
    if target == rev_years[0]:
        return None  # no prior year to compare against
    return ticker, target


def _intent_adjust(question: str, results: list[dict],
                   corpus: str = "active") -> tuple[list[dict], bool]:
    """Re-rank retrieved chunks for figure/performance questions:
    figure-bearing chunks up, definitional boilerplate down (stable —
    relative order otherwise preserved). The owner's screenshot
    failure was MD&A boilerplate (the fiscal-year definition, "This
    Item generally discusses…") outranking the results passages.
    Skipped for definition questions, which WANT the boilerplate."""
    if _DEFINITION_RE.search(question):
        return results, False
    if (_performance_intent(question, corpus) is None
            and not _FIGURE_TRIGGER.search(question)):
        return results, False

    def key(item: tuple[int, dict]) -> tuple[int, int, int]:
        idx, r = item
        head = r["text"][:900].lower()
        boiler = any(m in head for m in _BOILERPLATE_MARKERS)
        return (1 if boiler else 0,
                0 if _FIGUREY_RE.search(r["text"]) else 1,
                idx)

    adjusted = [r for _i, r in sorted(enumerate(results), key=key)]
    return adjusted, adjusted != results


def _prepend_headline(answer: str, headline: str) -> str:
    """Prepend the deterministic figures headline unless the model already
    leads with it (the steered LLM often opens with the headline verbatim —
    duplicating it reads as a bug)."""
    first_sentence = headline.split(". ")[0].strip()
    if first_sentence and first_sentence in answer[: len(headline) + 200]:
        return answer
    return f"{headline}\n\n{answer}"


def _performance_headline(state: GraphState) -> tuple[str, dict] | None:
    """A composed headline from the figures table for performance
    questions: revenue + net income for the target fiscal year against
    the year before, with YoY % — every number a figures-table row,
    cited to SEC company facts. Leads the answer; retrieved passages
    then support it."""
    found = _performance_intent(state["question"], state.get("corpus", "active"))
    if not found:
        return None
    ticker, year = found
    figs = registry.all_figures()

    def row(concept: str, fy: int) -> dict | None:
        return next((f for f in figs if f["ticker"] == ticker
                     and f["concept"] == concept
                     and f["fiscal_year"] == fy), None)

    name = registry.short_name(ticker)
    sentences: list[str] = []
    used_rows: list[dict] = []
    for concept, label in (("Total revenue", "total revenue"),
                           ("Net income", "net income")):
        cur, prev = row(concept, year), row(concept, year - 1)
        if cur is None:
            continue
        used_rows.append(cur)
        if prev is None:
            sentences.append(
                f"{name} reported {label} of ${cur['value_usd']:,} for "
                f"fiscal year {year} (period ended {cur['period_end']}).")
            continue
        used_rows.append(prev)
        pct = (cur["value_usd"] - prev["value_usd"]) / prev["value_usd"] * 100
        direction = "up" if pct >= 0 else "down"
        verb = "reported" if concept == "Total revenue" else "posted"
        sentences.append(
            f"{name} {verb} {label} of ${cur['value_usd']:,} for fiscal "
            f"year {year}, {direction} {abs(pct):.2f}% from "
            f"${prev['value_usd']:,} in fiscal year {year - 1}.")
    if not sentences:
        return None
    citation = {
        "doc": f"{ticker} · SEC company facts",
        "section": "Figures table",
        "span": [0, 0],
        "excerpt": figures_as_text(used_rows),
        "score": 0.0,
    }
    return " ".join(sentences), citation


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _index_corpus(state: GraphState) -> str:
    """Which retrieval corpus a run searches: the bundled snapshot, or
    the active corpus (bundled + session) for every other selector."""
    return "bundled" if state.get("corpus", "active") == "bundled" else "active"


def retrieve_node(state: GraphState) -> dict:
    started = time.perf_counter()
    strategy = state.get("strategy", "section")
    mode = state.get("retrieval_mode", "hybrid+rerank")
    corpus = state.get("corpus", "active")
    index_corpus = _index_corpus(state)
    mentioned = _mentioned_tickers(state["question"], corpus)
    if len(mentioned) >= 2:
        # Comparison questions: one fused ranking crowds the second
        # company out of the pool entirely (measured: MSFT Item 1 never
        # entered the top-10 for "Apple and Microsoft business"). Retrieve
        # per company and interleave — the standard multi-aspect fix.
        per_company = []
        for ticker in sorted(mentioned):
            out = search(state["query"], strategy=strategy, mode=mode, k=4,
                         tickers={ticker}, corpus=index_corpus)
            per_company.append(_serialize(out["results"]))
        merged: list[dict] = []
        for i in range(4):
            for lst in per_company:
                if i < len(lst):
                    merged.append(lst[i])
        for rank, r in enumerate(merged, start=1):
            r["rank"] = rank
        return {
            "results": merged,
            "trace": _trace("retrieve", started,
                            f"per-company retrieval for {sorted(mentioned)}: "
                            f"{len(merged)} merged candidates via {mode} on {strategy} chunks"),
        }
    out = search(state["query"], strategy=strategy, mode=mode, k=10,
                 corpus=index_corpus)
    results = _serialize(out["results"])
    results, adjusted = _intent_adjust(state["question"], results, corpus)
    top = results[0]["score"] if results else float("nan")
    return {
        "results": results,
        "trace": _trace("retrieve", started,
                        f"{len(results)} candidates via {mode} "
                        f"on {strategy} chunks; top score {top:.3f}"
                        + ("; intent-adjusted ranking (figure-bearing "
                           "chunks up, definitional boilerplate down)"
                           if adjusted else "")),
    }


def _windowed_score(reranker, question: str, text: str,
                    window: int = 700, stride: int = 350) -> float:
    """Best cross-encoder score over sliding windows of a chunk.

    Cross-encoders are trained on short passages; scoring a 4,000-char
    chunk as one string systematically underrates relevant prose (the
    first eval run refused correct Item 1A retrievals at -2). Grading
    the best-matching window is the standard passage-level fix.
    """
    if len(text) <= window:
        windows = [text]
    else:
        windows = [text[i:i + window] for i in range(0, len(text) - window + 1, stride)]
        windows.append(text[-window:])
    return max(float(s) for s in reranker.rerank(question, windows))


def scope_node(state: GraphState) -> dict:
    """Entry checks for questions no retrieval can ever satisfy:
    future predictions (filings report the past) and companies that
    are not in the corpus (no source documents exist for them)."""
    started = time.perf_counter()
    corpus = state.get("corpus", "active")
    if _is_prediction_question(state["question"]):
        return {"refusal_kind": "prediction",
                "trace": _trace("scope", started, "future-prediction question — not answerable from filings")}
    if _is_off_topic(state["question"], corpus):
        return {"refusal_kind": "off_topic",
                "trace": _trace("scope", started, "off-topic request — this assistant answers filing questions only")}
    if not _mentioned_tickers(state["question"], corpus) and _other_company_mentioned(state["question"], corpus):
        return {"refusal_kind": "other_company",
                "trace": _trace("scope", started, "named company is not in the corpus")}
    return {"trace": _trace("scope", started, "question is in corpus scope")}


def grade_node(state: GraphState) -> dict:
    started = time.perf_counter()
    results = state.get("results", [])
    if not results:
        return {"relevant": False, "grade_score": float("-inf"),
                "trace": _trace("grade", started, "no candidates to grade")}
    reranker = get_reranker()
    scores = [_windowed_score(reranker, state["question"], r["text"])
              for r in results[:3]]
    best = max(scores) if scores else float("-inf")
    # Calibrated grading (see the probe in docs/project-documentation.md):
    # absolute ms-marco logits do NOT separate gold evidence from noise
    # on legal prose (gold Item 1A chunk scored -4.8; a cross-company
    # chunk scored +2.2), so for company-named questions the gate is
    # company match at retrieval rank — the signal that held up: the
    # gold chunk was rank 1 for 11 of 12 probe questions. The windowed
    # score is still computed and traced, and it gates company-less
    # questions, where it is well-behaved.
    mentioned = _effective_tickers(state["question"], state.get("corpus", "active"))
    if len(mentioned) == 1:
        ticker = next(iter(mentioned))
        relevant = any(r["ticker"] == ticker for r in results[:3])
        detail = (f"company-match grade: {ticker} in top-3 = {relevant} "
                  f"(windowed cross-encoder best {best:.2f}, recorded not gated)")
    elif len(mentioned) > 1:
        missing = [t for t in mentioned
                   if not any(r["ticker"] == t for r in results[:8])]
        relevant = not missing
        detail = (f"company-match grade (comparison): covered "
                  f"{sorted(mentioned - set(missing))}, missing {missing} in merged top-8 "
                  f"(windowed cross-encoder best {best:.2f}, recorded not gated)")
    else:
        relevant = best >= GRADE_THRESHOLD
        detail = f"windowed cross-encoder top score {best:.2f} (threshold {GRADE_THRESHOLD})"
    llm = get_llm()
    # The LLM check tightens ONLY the company-less path, where
    # relevance came from the windowed score. For company-named (or
    # corpus-pinned) questions the company-match gate above is the
    # calibrated signal — and the judge sees only 1,200-char chunk
    # prefixes, so letting it veto company-matched evidence refused
    # questions whose on-topic passage sat deeper in the chunk (the
    # session MD&A case: the eligible prefixes were an Item 7 preamble
    # and the prior year's filing, neither showing the asked topic).
    if relevant and llm is not None and not mentioned:
        verdicts = []
        for r, s in zip(results[:3], scores):
            if s < GRADE_THRESHOLD:
                continue
            resp = llm.invoke(GRADE_PROMPT.format(
                question=state["question"], excerpt=r["text"][:1200]))
            verdicts.append(str(resp.content).strip().lower().startswith("y"))
        # (LLM grade can only tighten the windowed-score verdict.)
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
    names = ["Apple", "Microsoft", "NVIDIA", "AAPL", "MSFT", "NVDA"]
    for ticker in registry.session_companies():
        names += [registry.short_name(ticker), ticker]
    tickers = [t for t in names if t.lower() in question.lower()]
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
    refusal, out_of_corpus, _clarify = _corpus_answers(state.get("corpus", "active"))
    if state.get("refusal_kind") == "off_topic":
        answer = _off_topic_answer(state.get("corpus", "active"))
        why = "off-topic request — filings only"
    elif state.get("refusal_kind") == "prediction":
        answer, why = PREDICTION_REFUSAL_ANSWER, "prediction question — filings report the past"
    elif state.get("refusal_kind") == "other_company":
        answer, why = out_of_corpus, "named company is outside the corpus"
    elif state.get("verify_detail"):
        answer, why = VERIFY_REFUSAL_ANSWER, "draft failed numeric verification twice"
    else:
        answer, why = refusal, "evidence insufficient after retrieval + rewrite"
    return {"refused": True, "answer": answer, "citations": [],
            "trace": _trace("refuse", started, f"honest refusal — {why}")}


def clarify_node(state: GraphState) -> dict:
    """Relevant evidence exists, but the question names no company and
    the corpus spans several — answering would silently pick one."""
    started = time.perf_counter()
    _r, _o, clarify = _corpus_answers(state.get("corpus", "active"))
    return {"clarification": True, "refused": False, "answer": clarify,
            "citations": [],
            "trace": _trace("clarify", started,
                            "no company named in a multi-company corpus — asking which one")}


_FIGURE_TRIGGER = re.compile(r"(revenue|net sales|net income|income)", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _figure_sentence(state: GraphState) -> tuple[str, dict] | None:
    """A grounded figures-table sentence when the question asks for one.

    Works for any company with figures rows — bundled AAPL plus every
    session-ingested company (the registry's combined table)."""
    q = state["question"]
    mentioned = _mentioned_tickers(q, state.get("corpus", "active"))
    if not mentioned or not _FIGURE_TRIGGER.search(q):
        return None
    years = _YEAR_RE.findall(q)
    wants_income = "income" in q.lower()
    all_figures = registry.all_figures()
    for ticker in sorted(mentioned):
        candidates = [
            f for f in all_figures
            if f["ticker"] == ticker
            and wants_income == (f["concept"] == "Net income")
            and (not years or str(f["fiscal_year"]) in years)
        ]
        if not candidates:
            continue
        # An explicit year in the question wins; otherwise the most recent
        # fiscal year (never the oldest row in the table).
        f = max(candidates, key=lambda c: c["fiscal_year"])
        line = (f"{f['ticker']} reported {f['concept'].lower()} of "
                f"${f['value_usd']:,} for fiscal year {f['fiscal_year']} "
                f"(period ended {f['period_end']}).")
        citation = {
            "doc": f"{ticker} · SEC company facts",
            "section": "Figures table",
            "span": [0, 0],
            "excerpt": figures_as_text([f]),
            "score": 0.0,
        }
        return line, citation
    return None


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|;\s+|\n+")


def _word_match(a: str, b: str) -> bool:
    """Crude morphological match: exact, plural-stripped, or a shared
    5-char prefix (competition / compete / competitive). Exact-token
    overlap alone starved the extractive composer — filings write
    "risks" where questions write "risk"."""
    if a == b:
        return True
    if len(a) > 3 and len(b) > 3 and a.rstrip("s") == b.rstrip("s"):
        return True
    return len(a) >= 5 and len(b) >= 5 and (a.startswith(b[:5]) or b.startswith(a[:5]))


def _stem_overlap(q_words: set[str], sent_words: set[str]) -> int:
    return sum(1 for qw in q_words if any(_word_match(qw, sw) for sw in sent_words))
_COMPANY_ALIASES = {"apple": "AAPL", "aapl": "AAPL", "microsoft": "MSFT",
                    "msft": "MSFT", "nvidia": "NVDA", "nvda": "NVDA"}


def _mentioned_tickers(question: str, corpus: str = "active") -> set[str]:
    """Tickers the question names, resolved through the registry's
    alias map for the companies allowed under ``corpus`` — bundled
    aliases plus every session-ingested company's name/ticker."""
    aliases = registry.company_aliases(_allowed_tickers(corpus))
    q = question.lower()
    return {t for alias, t in aliases.items() if alias in q}


def _effective_tickers(question: str, corpus: str = "active") -> set[str]:
    """The companies a question is effectively about: the ones it
    names, or — when it names none but the corpus selector pins
    exactly ONE company (corpus='AAPL') — that company. A pinned
    corpus makes a company-less question's referent unambiguous, so
    grading and routing must treat it as company-named; keying only
    on the question text sent pinned questions to the clarify route
    ("Which company do you mean?" about the one company in scope)."""
    mentioned = _mentioned_tickers(question, corpus)
    if mentioned:
        return mentioned
    allowed = _allowed_tickers(corpus)
    if len(allowed) == 1:
        return set(allowed)
    return set()


def _evidence(state: GraphState) -> list[dict]:
    """The chunk set generation may cite and verification checks
    against: top-4 normally, the full merged top-8 for comparisons."""
    n = 8 if len(_mentioned_tickers(state["question"],
                                     state.get("corpus", "active"))) >= 2 else 4
    return state.get("results", [])[:n]


def _extractive_answer(state: GraphState) -> tuple[str, list[dict]]:
    results = _evidence(state)
    mentioned = _mentioned_tickers(state["question"], state.get("corpus", "active"))
    if mentioned and any(r["ticker"] in mentioned for r in results):
        # A question about one company must not be answered with another
        # company's sentences (comparison questions name both, and keep both).
        results = [r for r in results if r["ticker"] in mentioned] or results
    q_words = {w.lower() for w in re.findall(r"[a-z0-9]+", state["question"].lower())
               if w not in _STOPWORDS and len(w) > 2}
    headline = _performance_headline(state)
    asks_figure = bool(_FIGURE_TRIGGER.search(state["question"])) or headline is not None

    scored: list[tuple[float, str, dict]] = []
    for rank_idx, r in enumerate(results):
        for sent in _SENT_SPLIT.split(r["text"]):
            sent = " ".join(sent.split())
            if len(sent) < 30 or len(sent) > 800:
                continue
            words = {w.lower() for w in re.findall(r"[a-z0-9]+", sent.lower())}
            overlap = _stem_overlap(q_words, words)
            has_number = bool(re.search(r"\d", sent))
            # Retrieval rank is a real relevance signal: prefer sentences
            # from higher-ranked chunks when overlap is comparable.
            score = 1.5 * overlap + (2.0 if (asks_figure and has_number) else 0.0) \
                + max(0, 4 - rank_idx) * 0.4
            if (overlap >= 2 or (overlap >= 1 and rank_idx == 0)
                    or (asks_figure and has_number and overlap >= 1)):
                scored.append((score, sent, r))
    scored.sort(key=lambda t: -t[0])

    picked: list[tuple[str, dict]] = []
    seen: set[str] = set()

    def _take(cands: list[tuple[float, str, dict]], limit: int) -> None:
        for _s, sent, r in cands:
            if len(picked) >= limit:
                return
            key = sent[:60]
            if key in seen:
                continue
            seen.add(key)
            picked.append((sent, r))

    if len(mentioned) >= 2:
        # Comparison composition: each named company contributes its
        # best figure sentence first (when figures are asked), then its
        # best content sentence — one company must not flood the top-5
        # on raw overlap while the other goes unquoted.
        for ticker in sorted(mentioned):
            comp = [c for c in scored if c[2]["ticker"] == ticker]
            if asks_figure:
                fig_c = [c for c in comp if re.search(r"\d", c[1]) and
                         any(w in c[1].lower() for w in ("revenue", "net sales", "income"))]
                _take(fig_c[:1], len(picked) + 1)
            _take(comp, len(picked) + 2)
    else:
        _take(scored, 5)

    parts: list[str] = []
    used: list[dict] = []
    if headline is not None:
        # Performance questions lead with the composed figures
        # headline INSTEAD of the single-row figure sentence.
        parts.append(headline[0] + f" [{headline[1]['doc']}]")
        used.append(headline[1])
    else:
        fig = _figure_sentence(state)
        if fig:
            parts.append(fig[0] + f" [{fig[1]['doc']}]")
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
    results = _evidence(state)
    context_parts = []
    for r in results:
        context_parts.append(f"[{r['doc']} · {r['section']}]\n{r['text'][:2600]}")
    figures = load_figures()
    context_parts.append("[AAPL · SEC company facts]\n" + figures_as_text(figures[-8:]))
    # Session rows that duplicate a bundled row (a live add of AAPL
    # fetches the same company-facts) are already in the block above.
    bundled_keys = {(f["ticker"], f["concept"], f["fiscal_year"]) for f in figures}
    session_figs = [f for f in registry.session_figures()
                    if (f["ticker"], f["concept"], f["fiscal_year"])
                    not in bundled_keys]
    if session_figs:
        context_parts.append("[Session companies · SEC company facts]\n"
                             + figures_as_text(session_figs))
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
        # Performance questions carry a deterministic figures headline.
        # In LLM mode the model still writes the supporting prose, but
        # it is steered to the headline's fiscal year and the headline
        # is prepended verbatim — the owner's live test answered FY2024
        # figures for "last year" when FY2025 was the latest year,
        # because the bundled MD&A context is the FY2024 filing. The
        # headline makes the year deterministic; verification still
        # checks every number in the combined answer.
        headline = _performance_headline(state)
        intent = _performance_intent(state["question"],
                                     state.get("corpus", "active"))
        if headline is not None and intent is not None:
            _ticker, year = intent
            strict += (
                f"\n\nThis question asks how the company performed in fiscal "
                f"year {year}, its latest reported fiscal year. A verified "
                f"headline computed from SEC company facts will be placed "
                f"before your answer: \"{headline[0]}\" Discuss fiscal year "
                f"{year} (comparing with fiscal year {year - 1} where the "
                f"context supports it). Do not present another year's "
                f"figures as the result for fiscal year {year}.")
        answer, citations = _llm_answer(state, strict)
        if headline is not None:
            answer = _prepend_headline(answer, headline[0])
            citations = [headline[1]] + citations[:4]
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
    figures = registry.all_figures()
    grounded_text += "\n" + figures_as_text(figures)
    grounded_values: set[float] = set()
    for tok in _NUM_RE.findall(grounded_text):
        v = _token_value(tok)
        if v is not None:
            grounded_values.add(v)
    for f in figures:
        grounded_values.add(float(f["value_usd"]))
        grounded_values.add(float(f["fiscal_year"]))

    # Year-over-year percentages are arithmetic on two grounded figure
    # rows (same ticker + concept, adjacent fiscal years) — the
    # performance headline states them, so verification must recognise
    # them as grounded rather than invented. Answers print them at 2
    # decimals, hence the rounding tolerance.
    by_concept: dict[tuple[str, str], dict[int, float]] = {}
    for f in figures:
        by_concept.setdefault((f["ticker"], f["concept"]), {})[
            f["fiscal_year"]] = float(f["value_usd"])
    derived_pcts: set[float] = set()
    for years in by_concept.values():
        for fy, value in years.items():
            prev = years.get(fy - 1)
            if prev:
                pct = (value - prev) / prev * 100
                derived_pcts.update((pct, abs(pct)))

    ungrounded: list[str] = []
    for tok in _NUM_RE.findall(answer):
        v = _token_value(tok)
        if v is None:
            continue
        if "%" in tok and any(abs(v - p) <= 0.006 for p in derived_pcts):
            continue  # a YoY % derived from the figures table (2dp rounding)
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
    ok, detail = verify_answer(state.get("answer", ""), _evidence(state))
    return {"verify_ok": ok, "verify_detail": "" if ok else detail,
            "trace": _trace("verify", started, detail)}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _route_after_scope(state: GraphState) -> str:
    return "refuse" if state.get("refusal_kind") else "retrieve"


def _route_after_grade(state: GraphState) -> str:
    mentioned = _effective_tickers(state["question"],
                                   state.get("corpus", "active"))
    if state.get("relevant"):
        # Relevant evidence but no company named: don't silently pick one.
        if not mentioned:
            return "clarify"
        return "generate"
    if not mentioned:
        # Company-less questions can never be answered from a
        # multi-company corpus, whatever the rewrite — ask which
        # company instead of refusing (or guessing). A follow-up that
        # arrives with NO context lands here by design.
        return "clarify"
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
    g.add_node("contextualize", contextualize_node)
    g.add_node("scope", scope_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify_node)
    g.add_node("refuse", refuse_node)
    g.add_node("clarify", clarify_node)
    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "scope")
    g.add_conditional_edges("scope", _route_after_scope,
                            {"retrieve": "retrieve", "refuse": "refuse"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route_after_grade,
                            {"generate": "generate", "rewrite": "rewrite",
                             "refuse": "refuse", "clarify": "clarify"})
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("verify", _route_after_verify,
                            {"generate": "generate", "refuse": "refuse", END: END})
    g.add_edge("generate", "verify")
    g.add_edge("refuse", END)
    g.add_edge("clarify", END)
    return g.compile()


_APP = None


def get_app():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def run(question: str, strategy: str = "section",
        retrieval_mode: str = "hybrid+rerank", corpus: str = "active",
        context: dict | None = None) -> dict:
    """Ask one question; returns the JSON-able result for API/UI/tests.

    ``corpus``: "active" (bundled + session, the default), "bundled"
    (the frozen snapshot the measurements are pinned to), or a ticker
    restricting scope + retrieval to that one company.

    ``context``: the previous turn's ``resolved`` ({ticker,
    fiscal_year}) — follow-up questions are rewritten against it by
    the contextualize step. The result carries this turn's
    ``resolved`` plus ``understood_as`` (the standalone rewrite, when
    one happened).
    """
    started = time.perf_counter()
    final = get_app().invoke({
        "question": question,
        "query": question,
        "strategy": strategy,
        "retrieval_mode": retrieval_mode,
        "corpus": corpus,
        "context": context or {},
        "rewrites": 0,
        "generations": 0,
        "trace": [],
    })
    return {
        "question": question,
        "answer": final.get("answer", ""),
        "refused": bool(final.get("refused", False)),
        "clarification": bool(final.get("clarification", False)),
        "citations": final.get("citations", []),
        "trace": final.get("trace", []),
        "mode": llm_mode(),
        "strategy": strategy,
        "retrieval_mode": retrieval_mode,
        "corpus": corpus,
        "resolved": _resolved_for(final.get("question", question), corpus),
        "understood_as": final.get("understood_as"),
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
