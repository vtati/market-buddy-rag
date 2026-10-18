# Chunking × retrieval comparison

Measured 2026-10-18T18:43:56.221545+00:00 on the bundled corpus
(4 documents, AAPL/MSFT/NVDA 10-K excerpts) with a fixed set of
12 questions, each labelled with the company and
Item section a correct retrieval must land in ("gold sections").

* **Hit-rate@4** — share of questions where a chunk from a gold section
  appears in the top-4 results.
* **MRR** — mean reciprocal rank of the first gold-section chunk in the
  top-10.
* **Latency** — mean wall-clock retrieval time per question (warm
  indexes, local models).

## Results

| Chunking strategy | Retrieval mode | Hit-rate@4 | MRR | Mean latency |
|---|---|---|---|---|
| fixed | vector | 0.92 | 0.93 | 20 ms |
| fixed | hybrid | 0.92 | 0.88 | 17 ms |
| fixed | hybrid+rerank | 0.92 | 0.89 | 817 ms |
| section | vector | 1.00 | 0.85 | 25 ms |
| section | hybrid | 0.83 | 0.74 | 22 ms |
| section | hybrid+rerank | 0.92 | 0.93 | 2355 ms |

## What the numbers say

* Best hit-rate@4: **section chunking +
  vector** (hit-rate@4 1.00,
  MRR 0.85, 25 ms mean latency).
  Hit-rate alone doesn't pick the app default — see the note on rank
  quality below.
* The naive baseline — fixed-size chunks with dense-only retrieval —
  lands at hit-rate@4 **0.92** /
  MRR **0.93**. Every other combination is measured
  against it in the table above.
* Section-aware chunking keeps each chunk inside one Item, so a hit
  carries exactly the context the question needs (a full MD&A table or
  a complete risk-factor passage) instead of a 1000-char window that
  may straddle two sections.
* Reranking effect on section chunks: hybrid → hybrid+rerank moves
  hit-rate@4 by **+0.08** and on fixed chunks by
  **+0.00**; on section chunks its bigger effect is
  rank quality — MRR 0.74 → 0.93 — which
  is what the answer generator actually reads (top-1 to top-4 order).
* **Why the app defaults to section + hybrid+rerank anyway:** it ties
  for the best MRR (0.93) with a near-best hit-rate, and
  the same cross-encoder doubles as the graph's relevance grader — the
  score that decides answer vs. honest refusal. The cost is real:
  ~2355 ms per query locally, vs
  ~25 ms for dense-only. On a
  4-document corpus hybrid fusion adds noise (BM25 promotes
  keyword-matching cross-company chunks); we keep hybrid+rerank
  because its top-of-ranking precision is what generation and
  grading consume, and we report the trade-off rather than hide it.

Raw per-question output: `results/comparison.json` (regenerate with
`python -m src.experiments`).
