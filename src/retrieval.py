"""Retrieval — BM25 + vector hybrid with reciprocal-rank fusion and a
cross-encoder reranking stage. All models run locally via fastembed
(no API key): embeddings ``BAAI/bge-small-en-v1.5``, reranker
``Xenova/ms-marco-MiniLM-L-6-v2``. The vector store is FAISS.

Modes (``search(..., mode=...)``):

* ``vector`` — dense retrieval only (FAISS, top-k by similarity).
* ``hybrid`` — RRF fusion (k=60) of BM25 top-10 and vector top-10.
* ``hybrid+rerank`` — the hybrid top-10 pool re-scored by the
  cross-encoder, top-k returned.

Indexes are built per chunking strategy and cached in-process (FAISS is
also persisted under ``indexes/`` so repeat processes skip re-embedding).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.chunking import chunk_corpus
from src.ingest import REPO_ROOT, load_bundled_documents

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
POOL_SIZE = 10  # candidates each retriever contributes / the rerank pool
INDEX_DIR = REPO_ROOT / "indexes"

MODES = ("vector", "hybrid", "hybrid+rerank")


class FastEmbedEmbeddings(Embeddings):
    """LangChain Embeddings wrapper over fastembed (local ONNX models)."""

    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        query_embed = getattr(self._model, "query_embed", None)
        if query_embed is not None:
            return next(iter(query_embed(text))).tolist()
        return next(iter(self._model.embed([text]))).tolist()


_embeddings: FastEmbedEmbeddings | None = None
_reranker = None


def get_embeddings() -> FastEmbedEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings()
    return _embeddings


def get_reranker():
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _reranker = TextCrossEncoder(RERANK_MODEL)
    return _reranker


def _chunk_key(doc: Document) -> tuple:
    m = doc.metadata
    return (m.get("doc_label"), m.get("char_start"), m.get("chunk_index"))


class StrategyIndex:
    """BM25 + FAISS indexes over one strategy's chunks."""

    def __init__(self, strategy: str, chunks: list[Document]) -> None:
        self.strategy = strategy
        self.chunks = chunks
        self.bm25 = BM25Retriever.from_documents(chunks)
        self.bm25.k = POOL_SIZE
        index_path = INDEX_DIR / strategy
        embeddings = get_embeddings()
        if index_path.exists():
            self.faiss = FAISS.load_local(
                str(index_path), embeddings, allow_dangerous_deserialization=True
            )
        else:
            self.faiss = FAISS.from_documents(chunks, embeddings)
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            self.faiss.save_local(str(index_path))

    # -- candidate pools ----------------------------------------------------
    def vector_pool(self, query: str, n: int = POOL_SIZE) -> list[tuple[Document, float]]:
        hits = self.faiss.similarity_search_with_score(query, k=n)
        return [(doc, 1.0 / (1.0 + float(dist))) for doc, dist in hits]

    def bm25_pool(self, query: str, n: int = POOL_SIZE) -> list[tuple[Document, float]]:
        docs = self.bm25.invoke(query)[:n]
        # BM25Retriever does not surface scores; rank is what RRF consumes.
        return [(doc, float(POOL_SIZE - i)) for i, doc in enumerate(docs)]

    # -- modes ----------------------------------------------------------------
    def search(self, query: str, mode: str = "hybrid+rerank", k: int = 4) -> list[dict]:
        if mode == "vector":
            pool = self.vector_pool(query, n=max(k, POOL_SIZE))[:k]
            return [
                {"chunk": doc, "score": score, "rank": i + 1, "signals": {"vector": score}}
                for i, (doc, score) in enumerate(pool)
            ]

        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

        fused: dict[tuple, dict] = {}
        for source, pool in (("bm25", self.bm25_pool(query)), ("vector", self.vector_pool(query))):
            for rank, (doc, _score) in enumerate(pool, start=1):
                key = _chunk_key(doc)
                entry = fused.setdefault(key, {"chunk": doc, "rrf": 0.0, "signals": {}})
                entry["rrf"] += 1.0 / (RRF_K + rank)
                entry["signals"][source] = rank
        ranked = sorted(
            fused.values(),
            key=lambda e: (-e["rrf"], e["chunk"].metadata.get("char_start", 0)),
        )
        pool = ranked[:POOL_SIZE]

        if mode == "hybrid":
            return [
                {"chunk": e["chunk"], "score": e["rrf"], "rank": i + 1, "signals": e["signals"]}
                for i, e in enumerate(pool[:k])
            ]

        # hybrid+rerank: cross-encoder over the hybrid pool
        reranker = get_reranker()
        texts = [e["chunk"].page_content for e in pool]
        scores = list(reranker.rerank(query, texts))
        rescored = sorted(
            (
                {"chunk": e["chunk"], "score": float(s),
                 "signals": {**e["signals"], "rrf": e["rrf"]}}
                for e, s in zip(pool, scores)
            ),
            key=lambda e: -e["score"],
        )
        return [{**e, "rank": i + 1} for i, e in enumerate(rescored[:k])]


_indexes: dict[str, StrategyIndex] = {}


def get_index(strategy: str) -> StrategyIndex:
    if strategy not in _indexes:
        docs = load_bundled_documents()
        chunks = chunk_corpus(docs, strategy)
        _indexes[strategy] = StrategyIndex(strategy, chunks)
    return _indexes[strategy]


def search(query: str, strategy: str = "section", mode: str = "hybrid+rerank",
           k: int = 4) -> dict:
    """Search one strategy's index. Returns results + measured latency."""
    index = get_index(strategy)
    started = time.perf_counter()
    results = index.search(query, mode=mode, k=k)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {"results": results, "latency_ms": round(latency_ms, 1),
            "strategy": strategy, "mode": mode}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval smoke test")
    parser.add_argument("query", nargs="?", default="What was Apple's total net sales in fiscal 2024?")
    parser.add_argument("--strategy", default="section", choices=["fixed", "section"])
    parser.add_argument("--mode", default="hybrid+rerank", choices=list(MODES))
    args = parser.parse_args()
    out = search(args.query, strategy=args.strategy, mode=args.mode, k=4)
    print(f"query: {args.query}")
    print(f"strategy={out['strategy']} mode={out['mode']} latency={out['latency_ms']} ms")
    for r in out["results"]:
        m = r["chunk"].metadata
        print(f"  #{r['rank']} score={r['score']:.4f} [{m['doc_label']} · {m['section']}] "
              f"{r['chunk'].page_content[:110]!r}")


if __name__ == "__main__":
    main()
