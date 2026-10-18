# Framework — Market Buddy RAG (Week 2, Use Case 2)

The handout's framework fields, filled with this build's actual design
and measured numbers.

**Use case (one line).** A retail investor asks questions about a
company's business, risks, and reported financials, and gets answers
retrieved from that company's real SEC 10-K filings — cited to the
filing section, with every figure verified, and an honest "I don't
know" when the filings don't contain the answer.

**Corpus.** Excerpts of the most recent 10-K filings for three
companies, from SEC EDGAR: Apple FY2024 (Items 7/7A/8) + FY2025
(Items 1/1A), Microsoft FY2026 (Items 1/1A/7/8), NVIDIA FY2026
(Items 1/1A/7/8) — 4 documents, 527,205 characters — plus Apple's SEC
company-facts data as a structured figures table (total revenue and
net income by fiscal year, 38 concept-year records). Full provenance
(accessions, fetch method, slicing rules) in `data/PROVENANCE-*.md`.

**Ingestion + cleaning.** Filings are fetched from EDGAR's submissions
API with a declared user agent and ≥0.11 s pacing (implemented at
0.15 s), cached under `data/cache/`. HTML is stripped with a stdlib
parser (block tags → line breaks, table cells → spaces), boilerplate
and comment headers removed, and each document is mapped to its Item
sections (1, 1A, 7, 7A, 8) by a heading detector that requires a title
keyword and scans in reverse document order — defeating both the
table-of-contents entries and the running page headers filings repeat
on every page. Items are sliced whole with per-item character caps so
the bundled corpus stays demo-sized while remaining real filing text.

**Freshness.** The corpus is a point-in-time snapshot: each company's
most recent 10-K as of the build date (Oct 2026), which for these
fiscal calendars means FY2024–FY2026 filings. The ingestion path is
live — `python -m src.ingest --live AAPL MSFT NVDA` re-fetches the
latest 10-K per ticker — but the demo ships the frozen snapshot so
every measured number in this repo is reproducible. A production
deployment would re-run ingestion on EDGAR's daily filings feed and
re-index on accession change.

**Chunking + embedding.** Two strategies, compared head-to-head in
`docs/comparison.md`: **fixed** — RecursiveCharacterTextSplitter,
1,000 chars / 200 overlap (856 chunks); **section-aware** — split on
Item headings first (a chunk never crosses a section boundary — an
invariant the test suite verifies independently), pack paragraphs up
to ~4,000 chars (224 chunks). Every chunk carries ticker, document,
section, and source character span. Embeddings: `BAAI/bge-small-en-v1.5`
via fastembed, computed locally — no API key anywhere in the retrieval
path.

**Retrieve / store / top-k.** Per strategy, a FAISS index (dense) and
a BM25 index (sparse) over the same chunks. Modes: **vector** (FAISS
similarity), **hybrid** (RRF fusion, k=60, over BM25 top-10 + vector
top-10), **hybrid+rerank** (cross-encoder `ms-marco-MiniLM-L-6-v2`
re-scores the fused top-10). Retrieval returns top-10 candidates; the
graph grades and generates from the top-4 (top-8 merged for
multi-company comparisons, retrieved per company). FAISS indexes
persist under `indexes/`; the corpus and indexes rebuild from
`data/` in seconds.

**Refusal design.** Refusal is a first-class path, designed before
generation quality: (1) a **scope** check refuses future-prediction
questions and companies outside the corpus before any retrieval;
(2) a **grade** step rejects weak evidence — company-named questions
require the named company's chunks at the top of the ranking (with one
query rewrite + re-retrieval before giving up); (3) a **verify** step
checks every number in the draft answer against the cited chunks and
the figures table — one regeneration, then refusal. Company-less
questions in a three-company corpus route to a **clarification**
("which company do you mean?") rather than a silently-guessed answer.
Each refusal names what the corpus does cover.

**Latency ceiling + measured numbers.** Budget: a chat answer inside
~5 s on a laptop CPU, all-local. Measured end-to-end through the graph
(15-question eval, extractive mode): **mean 3.0 s, worst case ~6 s
cold**. Retrieval alone: ~20 ms (vector/hybrid) vs ~2.4 s
(hybrid+rerank) per question — the cross-encoder on CPU is the entire
latency story, and the comparison experiment shows what it buys
(MRR 0.74 → 0.93 on section chunks). Quality gates measured on the
same runs: comparison hit-rate@4 up to 1.00 (section+vector), eval
15/15 with 100% faithfulness among answered questions.
