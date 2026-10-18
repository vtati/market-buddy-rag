"""FastAPI backend for the Week 2 demo UI.

Endpoints:
    POST /api/chat              {question, strategy?, mode?}
                                -> {answer, refused, clarification,
                                    citations[], trace[], mode, latency_ms}
    GET  /api/results/comparison -> results/comparison.json (Feature 4)
    GET  /api/results/eval       -> results/eval.json (Feature 6)
    GET  /api/corpus             -> corpus + chunk statistics
    GET  /api/health             -> liveness + generation mode

Run: ``uvicorn src.api:app --port 8000`` (the Next.js app in web/
expects it at NEXT_PUBLIC_API_BASE, default http://localhost:8000).
"""

from __future__ import annotations

import json
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chunking import STRATEGIES, chunk_corpus
from src.graph import get_llm, run
from src.ingest import REPO_ROOT, find_item_sections, load_bundled_documents
from src.retrieval import EMBED_MODEL, MODES, RERANK_MODEL

app = FastAPI(title="Market Buddy RAG — Week 2 demo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
    except Exception:
        pass  # a warm-up failure surfaces properly on the first request


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_warm, daemon=True).start()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    strategy: str = "section"
    mode: str = "hybrid+rerank"


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "generation_mode": "llm" if get_llm() is not None else "extractive"}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"strategy must be one of {sorted(STRATEGIES)}")
    if req.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {list(MODES)}")
    result = run(req.question, strategy=req.strategy, retrieval_mode=req.mode)
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
    return {
        "documents": [
            {"doc_id": d.metadata["source_file"], "ticker": d.metadata["ticker"],
             "label": d.metadata["doc_label"], "chars": len(d.page_content),
             "sections": sorted(find_item_sections(d.page_content).keys())}
            for d in docs
        ],
        "total_chars": sum(len(d.page_content) for d in docs),
        "chunks": chunks,
        "generation_mode": "llm" if get_llm() is not None else "extractive",
    }
