# Project documentation — Market Buddy RAG

Gen Academy "Mastering Agentic AI", Week 2 project: *Build Your RAG
Application*. Track: LangChain + LangGraph (code). Use case: #2,
Financial Document Intelligence, with a BYO framing — the application
is a demo distillation of Market Buddy, a conversational stock-market
research product, reduced to its document-intelligence core: ask
questions of real SEC filings and get cited, verified answers.

## (a) Project overview — what was built

A retrieval-augmented Q&A system over real 10-K filings, in four parts:

1. **A corpus pipeline** (`src/ingest.py`) that fetches filings live
   from SEC EDGAR, cleans the HTML to text, maps each document to its
   Item sections, and parses Apple's SEC company-facts JSON into a
   structured figures table.
2. **A retrieval lab** (`src/chunking.py`, `src/retrieval.py`,
   `src/experiments.py`) implementing two chunking strategies
   (fixed-size vs section-aware) and three retrieval modes (vector,
   hybrid BM25+vector with RRF fusion, hybrid + cross-encoder
   rerank), with a repeatable experiment that measures hit-rate@4,
   MRR, and latency for every combination.
3. **A LangGraph application** (`src/graph.py`): a state graph that
   scopes the question, retrieves, grades the evidence, rewrites the
   query once if the evidence is weak, generates a cited answer,
   verifies every figure in it against the cited sources, and refuses
   honestly when the filings can't support an answer. Generation is
   provider-pluggable — an LLM (OpenAI or Nebius) when a key is set,
   an extractive composer over the filings' own sentences when it is
   not — so the whole system runs with zero API keys.
4. **A web product** (`src/api.py` + `web/`): a FastAPI backend and a
   Next.js 15 UI in the Market Buddy design language (Obsidian &
   Champagne) with three pages — Chat (live step trail, expandable
   evidence cards, strategy/mode switches), Chunking Lab (the measured
   comparison), and Eval Dashboard (the 15-question evaluation,
   expandable per question).

Quality gates that ship with it: a 12-test pytest suite (chunking
invariants, figures-table values, retrieval determinism, graph
refusal/answer paths) and the 15-question evaluation described in
`docs/evaluation-report.md` — **15/15 passed, 100% faithfulness among
answered questions, mean latency ~3.0 s** in the final measured run.

## (b) Datasets used — corpus sources, sizes, provenance

All sources are public SEC EDGAR data, fetched 2026-10-18 with a
declared user agent and fair-access pacing; raw downloads are cached
in the repo's (git-ignored) `data/cache/`, and the sliced corpus is
bundled so the demo runs offline.

| Document | Source filing (accession) | Items | Cleaned chars |
|---|---|---|---|
| AAPL 10-K FY2024 | 0000320193-24-000123 | 7, 7A, 8 | 26,990 |
| AAPL 10-K FY2025 | 0000320193-25-000079 | 1, 1A | 84,084 |
| MSFT 10-K FY2026 | 0001193125-26-323660 | 1, 1A, 7, 8 | 244,152 |
| NVDA 10-K FY2026 | 0001045810-26-000021 | 1, 1A, 7, 8 | 171,979 |
| **Total** | | | **527,205** |

Notes that matter for interpreting results:

- The FY2024 Apple excerpt is the scaffold fixture this project began
  with (Items 7/7A/8 only, ~4,080 words) — Apple's Item 1/1A coverage
  comes from its FY2025 filing instead. The corpus manifest, slicing
  rules (whole Items, per-item character caps: 1→70k, 1A→90k, 7→130k,
  8→70k), and fetch method are recorded in `data/PROVENANCE-*.md`.
- **Apple company facts** (`data/companyfacts_aapl.json`, the SEC's
  structured company-facts feed) are parsed into a figures table:
  total revenue and net income by fiscal year, grouped by *period
  end* — the feed's `fy` field is the filing's year, not the period's,
  and grouping by it silently mislabels years (a bug caught and fixed
  during the build; a regression test guards it). Verified values:
  FY2024 revenue $391,035,000,000, net income $93,736,000,000;
  FY2025 revenue $416,161,000,000, net income $112,010,000,000.
- NVIDIA's Item 8 in its primary document is a 206-character pointer
  paragraph (its statements sit after Item 15 in HTML order and were
  not sliced); NVIDIA's financial statements content therefore
  arrives via Item 7 MD&A, which includes its revenue tables.

Derived datasets built for measurement (both committed in `src/`):

- **12 gold questions** (`src/questions.py`) with known relevant
  companies and Item sections, used by the chunking × retrieval
  comparison experiment.
- **15 evaluation questions** (`src/evaluate.py`): 9 answerable
  factual, 2 multi-document comparisons, 2 ambiguous, 2 unanswerable —
  each with an explicit expected behaviour and a mechanical pass rule.

## (c) Prompts and agent instructions used

The graph has two kinds of instruction: LLM prompts (used when a key
is configured) and deterministic node rules (which define behaviour in
every mode). Both are quoted/described exactly as implemented in
`src/graph.py`.

**Generation system prompt** (verbatim — the LLM receives this plus
the retrieved context and the question):

> You are Market Buddy, a financial-documents assistant.
> Answer the question using ONLY the context provided below, which comes
> from real SEC 10-K filings and SEC company-facts data.
>
> Rules:
> - State figures exactly as they appear in the context (filing tables are
>   in millions unless the context says otherwise).
> - Cite every claim inline like [AAPL 10-K FY2024 · Item 7].
> - If the context does not contain the answer, say so plainly instead of
>   guessing. Never use outside knowledge.

**Query-rewrite prompt** (verbatim; used once when grading finds the
first retrieval weak):

> Rewrite this question into a search query for SEC 10-K
> filings. Keep the company, the figures or topics asked about, and add
> filing vocabulary (e.g. "total net sales", "risk factors", "Item 7").
> Return only the rewritten query.
>
> Question: {question}

**LLM grade prompt** (verbatim; when an LLM is configured it may only
*tighten* the deterministic grade, never loosen it):

> Does this filing excerpt help answer the question?
> Reply with exactly "yes" or "no".
>
> Question: {question}
>
> Excerpt:
> {excerpt}

**Deterministic node instructions** (the graph's actual rules):

- **Scope** — refuse before retrieval if the question asks for a
  future prediction (pattern-matched: "will … stock price", forecast/
  price-target phrasing) or names a company outside the corpus
  (a known-companies list plus capitalised-possessive detection,
  e.g. "Tesla's"). Refusals say what the corpus does cover.
- **Grade** — for a question naming one corpus company: evidence is
  relevant iff at least one of the top-3 retrieved chunks is from that
  company; for a comparison: every named company must appear in the
  merged top-8. The windowed cross-encoder score is computed and
  traced but does not gate these (see iteration notes for why). For
  company-less questions the cross-encoder gate applies, and a pass
  routes to *clarify*, not generate.
- **Extractive composer** (the keyless generator) — lead with the
  figures-table sentence when the question asks for revenue/income
  figures a company reports; then pick filing sentences by
  morphological word overlap with the question (plural/prefix aware),
  boosted by retrieval rank and by containing a figure when figures
  are asked; only sentences from companies named in the question;
  comparisons are composed per company (figure sentence first) so
  one company can't flood the answer. Every sentence carries its
  [doc · section] citation inline.
- **Verify** — extract every number from the draft answer; each must
  appear in a cited chunk or the figures table (exact match, or
  millions/billions scaling within 0.6%). Failure → regenerate once
  with a strictness note → refuse with a verification-specific
  message rather than present unverified figures.

## (d) Iterations tried

**Retrieval/chunking variants** (all measured in one experiment run,
`docs/comparison.md`): {fixed, section} × {vector, hybrid,
hybrid+rerank} over the 12 gold questions.

| Chunking | Retrieval | Hit@4 | MRR | Latency |
|---|---|---|---|---|
| fixed | vector | 0.92 | 0.93 | 20 ms |
| fixed | hybrid | 0.92 | 0.88 | 17 ms |
| fixed | hybrid+rerank | 0.92 | 0.89 | 817 ms |
| section | vector | 1.00 | 0.85 | 25 ms |
| section | hybrid | 0.83 | 0.74 | 22 ms |
| section | hybrid+rerank | 0.92 | 0.93 | 2,355 ms |

Decisions from the data: the app defaults to **section +
hybrid+rerank** because generation and verification consume the top
of the ranking (MRR is the operative metric) — while documenting that
on a corpus this small, plain vector search is nearly as good and
100× cheaper.

**Application iterations** — the eval harness was run five times;
each failure class got a mechanism-level fix, never a question-level
patch:

1. **Run 1 — 8/15.** Absolute cross-encoder grade (threshold 0.0 on
   the top chunks' logits) refused correct Item 1A answers. A probe
   over the gold set showed why: gold chunks scored −4.8 (windowed)
   while a cross-company chunk scored +2.2 — ms-marco logits are not
   a calibrated relevance probability for legal prose. → Grading
   redesigned around **windowed passage scores** (best 700-char
   window per chunk) for company-less questions and **company-match
   at retrieval rank** for named companies, the signal the probe
   showed actually separates (gold chunk rank 1 on 11/12 questions).
2. **Run 2 — 10/15.** Windowing alone let a Tesla question through to
   a "which company?" clarification (wrong — Tesla was *named*, just
   out of corpus), and a stock-price prediction got answered from a
   dividend sentence. → **Scope node** added: predictions and
   out-of-corpus companies refuse up front, with reason-specific
   answers; company-less questions that pass grading route to a
   dedicated **clarify node**.
3. **Run 3 — 13/15.** A comparison question ("How do Apple and
   Microsoft each describe their business?") was refused: the single
   fused pool contained *no* Microsoft Item 1 chunk at all — one
   ranking crowds the second company out. → **Per-company
   retrieval** for multi-company questions (ticker-filtered pools,
   merged round-robin). E6 (Microsoft risks) answered but cited a
   cross-reference sentence from Item 1 instead of Item 1A content.
4. **Run 4 — 14/15.** Diagnosis of E6: exact-token sentence matching
   starved the composer — the filing writes "risks" and "competition"
   where the question writes "risk" and "cybersecurity", and most
   Item 1A sentences exceeded the composer's length cap. →
   **Morphological word matching** (plural/prefix), length cap
   420→800 chars, retrieval-rank bonus, and **per-company comparison
   composition**.
5. **Run 5 — 15/15**, faithfulness 100%, mean latency ~3.0 s.

**Interface iteration.** The initial plan was a Streamlit app; the
project owner replaced it with the house product stack (Next.js +
FastAPI, the Market Buddy Obsidian & Champagne design system) so the
demo presents at the same bar as the product it distills. The
Streamlit dependency was dropped; the API contract (chat / results /
corpus) was designed so the UI is a pure consumer of the same graph
the CLI and the eval harness use.

**Testing iteration.** The first full test run stood at 11/12: the
section-chunk span metadata claimed `char_end = start + len(text)`,
but packed chunks re-join paragraphs with a normalized separator, so
the claimed range didn't cover the content. Fixed in the chunker
(span = first-paragraph start → last-paragraph end) and the test now
asserts paragraph-level containment plus single-Item containment,
with boundaries recomputed independently from the source text.

## (e) Learnings and observations

- **Measure the grader, not just the retriever.** Retrieval ranked
  the gold chunk first on 11 of 12 probe questions from the start;
  nearly every eval failure was in grading, scoping, or answer
  composition. RAG quality is a pipeline property — the experiment
  that only measures hit-rate would have declared victory at run 1.
- **Cross-encoder logits are ranking scores, not probabilities.** On
  web-search passages a 0.0 threshold is sane; on 4,000-char legal
  chunks it refused correct evidence at −4.8 while admitting noise
  at +2.2. Any absolute threshold needs calibration data *from the
  target corpus* — or a different gate (company-match) whose
  behaviour you can measure directly.
- **Hybrid retrieval is not automatically better.** RRF fusion hurt
  section-aware chunks (hit@4 1.00 → 0.83): BM25 promoted
  cross-company keyword matches (every filing says "revenue" and
  "risk factors"). The handout's "hybrid is usually right" survived
  only once reranking re-imposed semantic order (MRR 0.74 → 0.93).
- **Multi-document questions break single-ranking retrieval.** A
  fused top-10 can contain *zero* chunks from one of the compared
  companies. Per-company retrieval is a small change with an
  outsized effect on comparison quality.
- **Refusal is a feature with its own taxonomy.** "Not in the
  corpus", "company not covered", "future prediction", and "which
  company did you mean?" are four different honest answers; users
  (and the eval) reward naming the right one. Designing the refusal
  paths first, as the handout advised, is what made the failures
  diagnosable.
- **A figures table beats prose for headline numbers.** The single
  most reliable answer in the system (Apple's FY2024 net sales) comes
  from structured SEC company-facts data, not from chunk retrieval —
  and the period-end grouping bug it hid (the feed's `fy` is the
  *filing* year) is exactly the class of silent data error a
  faithfulness checker can't catch, because the wrong number is
  perfectly "grounded" in a real record. Provenance and a regression
  test were the fix.
- **Extractive generation is an honest floor, not a compromise to
  hide.** With no API key the system still answers, cites, and
  verifies — it just quotes the filing instead of paraphrasing it.
  Labelling the mode in the UI (rather than silently degrading)
  turned a constraint into a demonstrable property: the same graph,
  the same gates, two generators.
- **Verbatim answers expose retrieval's table problem.** Filing
  tables flatten into lines like "Total $ 215,938 $ 130,497", which
  sentence-level composition handles poorly (the E11 soft spot in the
  evaluation report). Table-aware chunking — keeping a table's header
  row attached to its data rows — is the clearest next iteration.

## (f) Addendum — live ingestion ("Add a Company")

*Added after the measured build above; nothing in sections (a)–(e)
was rewritten. The comparison and evaluation numbers in this
document describe the bundled corpus only — this addendum is the
feature that keeps them honest while the corpus grows.*

**What it is.** A user can point the system at any SEC-listed
company (Chat sidebar → "Add a Company", or `POST /api/ingest`).
The company's latest 10-K is fetched from EDGAR and run through the
*same* pipeline that built the bundled corpus — same cleaning, same
whole-Item slicing caps (Items 1/1A/7/8 at 70k/90k/130k/70k chars),
both chunking strategies, bge-small embeddings + FAISS/BM25, and a
figures table built from its company-facts payload by the same
period-end grouping code as AAPL's (the AAPL path is now the
generalized `figures_from_facts`; AAPL's table is byte-identical and
its regression test still passes). The pipeline runs as a
background job with seven recorded stages — resolve → fetch →
clean → chunk → embed → figures → ready — and the UI polls them
live, because a pipeline you can watch fail honestly beats a spinner.

**The registry design.** The interesting problem was not fetching —
it was that "the corpus" was hardcoded in four places (scope
aliases, refusal copy, retrieval targeting, the figures table) and
each had silently assumed three companies forever. They now all
read one registry (`src/registry.py`) with two tiers:

- **Bundled** — the frozen AAPL/MSFT/NVDA snapshot. Immutable at
  runtime; its FAISS indexes persist on disk.
- **Session** — companies ingested at runtime, persisted under
  `data/session/<TICKER>/` (sliced filing + figures + meta), so a
  restart keeps them.

A company is in-corpus iff bundled *or* registered. Scope reads the
registry's alias map (bundled names plus every registered company's
name/ticker), refusal and clarification copy is composed from the
registry's coverage list, and retrieval targets a corpus selector —
`active` (bundled + session), `bundled`, or a single ticker. Two
engineering details mattered: the active index *extends* the
bundled snapshot's persisted FAISS vectors and embeds only the new
company's chunks (re-embedding 527k chars per ingestion is minutes
of CPU), and embedding is batched at 32 chunks — a one-shot
`embed_documents` over a whole filing spikes memory hard enough to
get the process OOM-killed on a small host (measured; the batched
peak is ~1.3 GB).

**Tesla is precached, deliberately.** `data/precached/TSLA/` is
the committed output of one real run of this pipeline (FY2025 10-K,
accession 0001628280-26-003952), so the demo's signature arc —
"Tesla's revenue?" refuses → ingest Tesla → the same question
returns $97,690,000,000 for FY2024 with citations — never depends
on SEC latency or uptime. It is a cache of the pipeline, not a
different pipeline: stages still run and report, and the result is
labelled `precached`. Any other ticker goes live to EDGAR.

**Honesty labels.** Session companies are visibly second-class
citizens in exactly the right way: tagged "session" in the corpus
panel, selectable but never silent in chat, and excluded from every
measurement. The Chunking Lab and Eval Dashboard both carry the
label verbatim: "Measured on the bundled corpus (AAPL · MSFT ·
NVDA snapshot, Oct 2026). Session-ingested companies are not part
of these measurements." The eval harness itself is pinned to
`corpus="bundled"` in code, and a test asserts the Tesla question
still refuses under that pin after ingestion — the measurements
can't drift even if the demo state does.

## (g) Addendum — live market context (Feature 8)

The final layer is presentational: real market data around the
filings, so the product reads like a market tool rather than a
chatbot bolted onto a corpus. Three surfaces, all fed by
`src/market.py`:

- **Market strip** above the chat — the S&P 500, Nasdaq Composite
  and Dow, plus every active-corpus company (bundled and session),
  each with price, day change, and a 1-month sparkline; auto-refresh
  every 60 seconds.
- **Company snapshot** in the chat rail. Focus follows the
  conversation (the ticker behind the most recent cited answer) or a
  corpus-panel click. It pairs the market's numbers (delayed quote,
  price chart at 1M/6M/1Y, market cap, trailing P/E, 52-week and
  day ranges, volume) with a separated, source-labelled block —
  "From the filing — our corpus" — carrying the latest fiscal
  year's revenue and net income (with a YoY revenue delta) from the
  same figures table the chat cites. The juxtaposition is the
  feature: two provenances, one company, neither masquerading as
  the other.
- **Trending rail** on Add a Company — Yahoo's trending tickers as
  chips; one tap prefills the ingest form and resolves the company,
  closing the loop from "what's moving" to "what does its 10-K say".

Sources are free and keyless: Yahoo Finance via `yfinance` primary,
Stooq's CSV endpoints as fallback, Yahoo's public trending endpoint
for the rail. Engineering choices, in the spirit of the rest of the
build: every fetch is timeout-bounded and failure-isolated (a
source outage degrades the payload to `available: false` and the UI
hides the surface — it can never break chat); results cache
in-memory (quotes 60s, history 15 min, trending 5 min) with startup
pre-warming; and every payload carries `source`, `as_of`, and
`delayed: true`, rendered in the UI as "Delayed quotes · Yahoo
Finance · as of …" — free market data is delayed, and the interface
says so. Two environment lessons were earned, not read: curl_cffi
sessions are not thread-safe (concurrent warm-up fetches crossed
responses until Yahoo calls were serialized), and httpx needs its
proxy passed explicitly with the OS CA bundle where an egress proxy
intercepts TLS. Charts are hand-rolled SVG in the design system's
tokens — no chart library was added for two path elements.

The boundary is absolute: market data never enters retrieval,
grading, generation, or the eval. Answers still come only from
filings and the SEC figures table; the tape is context for the
human, not evidence for the model.
