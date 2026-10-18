"""Chunking x retrieval comparison experiment.

Runs the fixed gold question set (``src/questions.py``) across
{fixed, section} x {vector, hybrid, hybrid+rerank} and reports, per
combination: hit-rate@4 (a chunk from a gold section in the top-4),
MRR over the top-10, and mean query latency. Writes the measured
results to ``results/comparison.json`` and renders
``docs/comparison.md`` from them — numbers are always the measured
output of the run that produced the files.

Run: ``python -m src.experiments``
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.ingest import REPO_ROOT
from src.questions import EXPERIMENT_QUESTIONS
from src.retrieval import MODES, get_index, search

STRATEGIES = ("fixed", "section")
RESULTS_PATH = REPO_ROOT / "results" / "comparison.json"
DOC_PATH = REPO_ROOT / "docs" / "comparison.md"


def _is_gold(result: dict, q: dict) -> bool:
    m = result["chunk"].metadata
    return m.get("ticker") == q["ticker"] and m.get("section") in q["gold_sections"]


def run() -> dict:
    # Warm-up: build both indexes and load the reranker off the clock.
    for strategy in STRATEGIES:
        get_index(strategy)
    search("warm up", strategy="section", mode="hybrid+rerank", k=1)

    combos: list[dict] = []
    per_question: list[dict] = []
    for strategy in STRATEGIES:
        for mode in MODES:
            hits, rr_total, latencies = 0, 0.0, []
            for q in EXPERIMENT_QUESTIONS:
                out = search(q["question"], strategy=strategy, mode=mode, k=10)
                results = out["results"]
                first_gold_rank = next(
                    (r["rank"] for r in results if _is_gold(r, q)), None
                )
                hit4 = any(_is_gold(r, q) for r in results[:4])
                hits += int(hit4)
                rr_total += (1.0 / first_gold_rank) if first_gold_rank else 0.0
                latencies.append(out["latency_ms"])
                per_question.append({
                    "question_id": q["id"], "strategy": strategy, "mode": mode,
                    "first_gold_rank": first_gold_rank, "hit_at_4": hit4,
                    "latency_ms": out["latency_ms"],
                })
            n = len(EXPERIMENT_QUESTIONS)
            combos.append({
                "strategy": strategy, "mode": mode,
                "hit_rate_at_4": round(hits / n, 3),
                "mrr": round(rr_total / n, 3),
                "mean_latency_ms": round(sum(latencies) / n, 1),
            })
            print(f"  {strategy:<8} {mode:<15} hit@4={combos[-1]['hit_rate_at_4']:.3f} "
                  f"MRR={combos[-1]['mrr']:.3f} mean latency={combos[-1]['mean_latency_ms']} ms")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "questions": len(EXPERIMENT_QUESTIONS),
        "combos": combos,
        "per_question": per_question,
    }


def _fmt_table(results: dict) -> str:
    rows = {}
    for c in results["combos"]:
        rows.setdefault(c["strategy"], {})[c["mode"]] = c
    lines = [
        "| Chunking strategy | Retrieval mode | Hit-rate@4 | MRR | Mean latency |",
        "|---|---|---|---|---|",
    ]
    for strategy in STRATEGIES:
        for mode in MODES:
            c = rows[strategy][mode]
            lines.append(
                f"| {strategy} | {mode} | {c['hit_rate_at_4']:.2f} | "
                f"{c['mrr']:.2f} | {c['mean_latency_ms']:.0f} ms |"
            )
    return "\n".join(lines)


def render_doc(results: dict) -> str:
    combos = {(c["strategy"], c["mode"]): c for c in results["combos"]}
    best = max(results["combos"], key=lambda c: (c["hit_rate_at_4"], c["mrr"]))
    sec_rr = combos[("section", "hybrid+rerank")]
    fix_vec = combos[("fixed", "vector")]
    sec_hyb = combos[("section", "hybrid")]
    fix_hyb = combos[("fixed", "hybrid")]
    rerank_gain_sec = sec_rr["hit_rate_at_4"] - sec_hyb["hit_rate_at_4"]
    rerank_gain_fix = combos[("fixed", "hybrid+rerank")]["hit_rate_at_4"] - fix_hyb["hit_rate_at_4"]
    return f"""# Chunking × retrieval comparison

Measured {results['generated_at']} on the bundled corpus
(4 documents, AAPL/MSFT/NVDA 10-K excerpts) with a fixed set of
{results['questions']} questions, each labelled with the company and
Item section a correct retrieval must land in ("gold sections").

* **Hit-rate@4** — share of questions where a chunk from a gold section
  appears in the top-4 results.
* **MRR** — mean reciprocal rank of the first gold-section chunk in the
  top-10.
* **Latency** — mean wall-clock retrieval time per question (warm
  indexes, local models).

## Results

{_fmt_table(results)}

## What the numbers say

* Best hit-rate@4: **{best['strategy']} chunking +
  {best['mode']}** (hit-rate@4 {best['hit_rate_at_4']:.2f},
  MRR {best['mrr']:.2f}, {best['mean_latency_ms']:.0f} ms mean latency).
  Hit-rate alone doesn't pick the app default — see the note on rank
  quality below.
* The naive baseline — fixed-size chunks with dense-only retrieval —
  lands at hit-rate@4 **{fix_vec['hit_rate_at_4']:.2f}** /
  MRR **{fix_vec['mrr']:.2f}**. Every other combination is measured
  against it in the table above.
* Section-aware chunking keeps each chunk inside one Item, so a hit
  carries exactly the context the question needs (a full MD&A table or
  a complete risk-factor passage) instead of a 1000-char window that
  may straddle two sections.
* Reranking effect on section chunks: hybrid → hybrid+rerank moves
  hit-rate@4 by **{rerank_gain_sec:+.2f}** and on fixed chunks by
  **{rerank_gain_fix:+.2f}**; on section chunks its bigger effect is
  rank quality — MRR {sec_hyb['mrr']:.2f} → {sec_rr['mrr']:.2f} — which
  is what the answer generator actually reads (top-1 to top-4 order).
* **Why the app defaults to section + hybrid+rerank anyway:** it ties
  for the best MRR ({sec_rr['mrr']:.2f}) with a near-best hit-rate, and
  the same cross-encoder doubles as the graph's relevance grader — the
  score that decides answer vs. honest refusal. The cost is real:
  ~{sec_rr['mean_latency_ms']:.0f} ms per query locally, vs
  ~{combos[('section', 'vector')]['mean_latency_ms']:.0f} ms for dense-only. On a
  4-document corpus hybrid fusion adds noise (BM25 promotes
  keyword-matching cross-company chunks); we keep hybrid+rerank
  because its top-of-ranking precision is what generation and
  grading consume, and we report the trade-off rather than hide it.

Raw per-question output: `results/comparison.json` (regenerate with
`python -m src.experiments`).
"""


def main() -> None:
    import sys

    if "--render-only" in sys.argv:  # re-render the doc from stored results
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        DOC_PATH.write_text(render_doc(results), encoding="utf-8")
        print(f"Re-rendered {DOC_PATH.relative_to(REPO_ROOT)} from {RESULTS_PATH.relative_to(REPO_ROOT)}")
        return
    print("Running chunking x retrieval comparison (warm-up first)...")
    results = run()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(render_doc(results), encoding="utf-8")
    print(f"Wrote {RESULTS_PATH.relative_to(REPO_ROOT)} and {DOC_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
