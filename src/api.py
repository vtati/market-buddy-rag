"""FastAPI backend for the Week 2 demo UI.

Endpoints:
    POST /api/chat              {question, strategy?, mode?, corpus?}
                                -> {answer, refused, clarification,
                                    citations[], trace[], mode, latency_ms}
    POST /api/ingest/resolve    {query} -> {candidates[]} (SEC map)
    POST /api/ingest            {ticker} -> {job_id} (background job)
    GET  /api/ingest/status/id  -> {stage, stages[], result?, error?}
    GET  /api/results/comparison -> results/comparison.json (Feature 4)
    GET  /api/results/eval       -> results/eval.json (Feature 6)
    GET  /api/corpus             -> corpus + chunk statistics (bundled
                                    documents + session companies)
    GET  /api/health             -> liveness + generation mode

Run: ``uvicorn src.api:app --port 8000`` (the Next.js app in web/
expects it at NEXT_PUBLIC_API_BASE, default http://localhost:8000).
"""

from __future__ import annotations

import json
import os
import re
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import edgar, registry, session_ingest
from src.chunking import STRATEGIES, chunk_corpus
from src.graph import get_llm, run
from src.ingest import REPO_ROOT, find_item_sections, load_bundled_documents
from src.retrieval import EMBED_MODEL, MODES, RERANK_MODEL

app = FastAPI(title="Market Buddy RAG — Week 2 demo API", version="1.0.0")

# Local dev origins are always allowed; deployed front-ends are added
# via the ALLOWED_ORIGINS env var (comma-separated), e.g. on Railway:
# ALLOWED_ORIGINS=https://ragdemo.genalphalabs.com
_allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"] + [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _warm() -> None:
    """Build indexes + load models off-thread so the first chat call
    isn't the cold one (embedding/reranker downloads + FAISS build)."""
    try:
        from src.retrieval import get_index, get_reranker
        get_reranker()
        for strategy in STRATEGIES:
            get_index(strategy)
        edgar.load_ticker_map()  # cache the SEC company map for resolve
    except Exception:
        pass  # a warm-up failure surfaces properly on the first request


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_warm, daemon=True).start()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    strategy: str = "section"
    mode: str = "hybrid+rerank"
    # "active" (bundled + session companies, default), "bundled" (the
    # frozen snapshot), or a ticker restricting scope + retrieval.
    corpus: str = "active"


class ResolveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=100)


class IngestRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)


_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}([.\-][A-Za-z]{1,2})?$")


def _valid_corpus(corpus: str) -> str | None:
    """Normalize a chat corpus selector, or None when invalid."""
    if corpus in ("active", "bundled"):
        return corpus
    ticker = corpus.strip().upper()
    return ticker if ticker in registry.known_tickers() else None


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "generation_mode": "llm" if get_llm() is not None else "extractive"}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"strategy must be one of {sorted(STRATEGIES)}")
    if req.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {list(MODES)}")
    corpus = _valid_corpus(req.corpus)
    if corpus is None:
        raise HTTPException(
            400, f"unknown corpus {req.corpus!r} — use 'active', 'bundled', "
                 "or a bundled/session ticker")
    result = run(req.question, strategy=req.strategy,
                 retrieval_mode=req.mode, corpus=corpus)
    return {
        "answer": result["answer"],
        "refused": result["refused"],
        "clarification": result.get("clarification", False),
        "citations": [
            {"doc": c["doc"], "section": c["section"], "excerpt": c["excerpt"],
             "span": c["span"], "score": c["score"]}
            for c in result["citations"]
        ],
        "trace": result["trace"],
        "mode": result["mode"],
        "latency_ms": result["latency_ms"],
        "strategy": result["strategy"],
        "retrieval_mode": result["retrieval_mode"],
        "corpus": result["corpus"],
    }


@app.post("/api/ingest/resolve")
def ingest_resolve(req: ResolveRequest) -> dict:
    """Company name or ticker -> candidate companies (SEC map)."""
    try:
        return {"candidates": edgar.resolve(req.query)}
    except Exception as exc:
        raise HTTPException(500, f"resolve failed: {exc}") from exc


@app.post("/api/ingest")
def ingest_start(req: IngestRequest) -> dict:
    """Start a background ingest job for one ticker."""
    ticker = req.ticker.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(400, f"invalid ticker {req.ticker!r}")
    job = session_ingest.create_job(ticker)
    threading.Thread(target=session_ingest.run_job, args=(job,),
                     daemon=True).start()
    return {"job_id": job["job_id"], "ticker": ticker}


@app.get("/api/ingest/status/{job_id}")
def ingest_status(job_id: str) -> dict:
    job = session_ingest.get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    return {
        "job_id": job["job_id"],
        "ticker": job["ticker"],
        "stage": job["stage"],
        "stages": job["stages"],
        "result": job["result"],
        "error": job["error"],
    }


def _results_file(name: str) -> dict:
    path = REPO_ROOT / "results" / name
    if not path.exists():
        raise HTTPException(404, f"{name} not generated yet — run the experiment/eval first")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/results/comparison")
def comparison() -> dict:
    """Normalize the experiment file into the web contract: combos ->
    rows, enriched with per-strategy chunk counts and model names."""
    raw = _results_file("comparison.json")
    docs = load_bundled_documents()
    chunk_counts = {name: len(chunk_corpus(docs, name)) for name in STRATEGIES}
    per_q: dict[tuple[str, str], list[dict]] = {}
    for entry in raw.get("per_question", []):
        per_q.setdefault((entry["strategy"], entry["mode"]), []).append({
            "id": entry["question_id"],
            "hit_rank": entry.get("first_gold_rank"),
            "latency_ms": entry.get("latency_ms"),
        })
    return {
        "generated_at": raw["generated_at"],
        "question_count": raw["questions"],
        "embedding_model": EMBED_MODEL,
        "reranker_model": RERANK_MODEL,
        "corpus_documents": [d.metadata["doc_label"] for d in docs],
        "rows": [
            {"strategy": c["strategy"], "mode": c["mode"],
             "chunks": chunk_counts.get(c["strategy"]),
             "hit_at_4": c["hit_rate_at_4"], "mrr": c["mrr"],
             "mean_latency_ms": c["mean_latency_ms"],
             "per_question": per_q.get((c["strategy"], c["mode"]), [])}
            for c in raw["combos"]
        ],
    }


@app.get("/api/results/eval")
def evaluation() -> dict:
    return _results_file("eval.json")


@app.get("/api/corpus")
def corpus() -> dict:
    docs = load_bundled_documents()
    chunks = {name: len(chunk_corpus(docs, name)) for name in sorted(STRATEGIES)}
    documents = [
        {"doc_id": d.metadata["source_file"], "ticker": d.metadata["ticker"],
         "label": d.metadata["doc_label"], "chars": len(d.page_content),
         "sections": sorted(find_item_sections(d.page_content).keys()),
         "source": "bundled"}
        for d in docs
    ]
    session = registry.session_companies()
    for d in registry.session_documents():
        documents.append(
            {"doc_id": d.metadata["source_file"], "ticker": d.metadata["ticker"],
             "label": d.metadata["doc_label"], "chars": len(d.page_content),
             "sections": sorted(find_item_sections(d.page_content).keys()),
             "source": session[d.metadata["ticker"]]["source"]})
    return {
        "documents": documents,
        "total_chars": sum(d["chars"] for d in documents),
        "chunks": chunks,
        "generation_mode": "llm" if get_llm() is not None else "extractive",
        "session_companies": [session[t] for t in sorted(session)],
    }
