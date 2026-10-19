# Market Buddy RAG — Week 2 Demo

A demo-purpose RAG application over real SEC filings, built for The Gen
Academy "Mastering Agentic AI" Week 2 project ("Build Your RAG
Application", Use Case 2: Financial Document Intelligence, with a BYO
framing). It distills the research core of **Market Buddy** — a separate,
private, production product — into a small public codebase on the
LangChain + LangGraph code track. The production product is not touched
by this repo.

**One-liner:** chat with real 10-K filings — ask about a company's
business, risks, or reported financials and get a cited answer whose
every figure is verified against the source text before you see it, or
an honest "I couldn't find this in the filings" when the evidence isn't
there.

## What it does

- **Corpus:** excerpts of the most recent 10-K filings for Apple,
  Microsoft, and NVIDIA (Items 1, 1A, 7, 8 — business, risk factors,
  MD&A, financial statements), fetched live from SEC EDGAR, plus Apple's
  structured SEC company-facts figures (revenue / net income by fiscal
  year). 4 documents, ~527k characters.
- **Two chunking strategies, measured against each other:** fixed-size
  (1,000 chars / 200 overlap) vs section-aware (split on Item headings,
  pack paragraphs to ~4,000 chars, never cross a section boundary).
- **Three retrieval modes:** vector (FAISS + bge-small-en-v1.5), hybrid
  (BM25 + vector fused with RRF k=60), hybrid+rerank (cross-encoder
  `ms-marco-MiniLM-L-6-v2` over the fused top-10). All local — no API
  key needed for retrieval.
- **A LangGraph Q&A app** that scopes, retrieves, grades, rewrites once
  if the evidence is weak, generates with inline citations, and then
  *verifies every number in the answer* against the cited chunks before
  answering — regenerating once, then refusing, if verification fails.
- **Generation is provider-pluggable:** with `OPENAI_API_KEY` or
  `NEBIUS_API_KEY` set, an LLM writes the answer; with no key, an
  extractive composer assembles it from the filings' own sentences.
  The demo runs end-to-end with **zero keys**.
- **Add a Company (live ingestion):** any SEC-listed company can be
  ingested at runtime — resolve the name/ticker, fetch its latest 10-K
  from EDGAR, slice/chunk/embed it with the same pipeline, build its
  figures table — and the chat answers from it immediately, in a
  session corpus layered over the bundled snapshot (see below).
  Re-adding a session company checks EDGAR: if a newer 10-K has been
  filed, the company is re-ingested and replaced; if not, the job
  reports "already current" with the accession on file.
- **Conversation memory:** the chat is multi-turn. Each answer reports
  the company + fiscal year it resolved to; the next question sends
  them back, and a deterministic contextualize step (rules, no model
  call) rewrites follow-ups — "Are there any risks with it?" becomes
  "What are Apple's principal risk factors?" — shown in the step trail
  as "Understood as: …". With no context, a company-less follow-up
  asks which company rather than guessing.
- **Performance questions are figures-first:** "How did Apple perform
  last year?" is answered with a composed headline from the figures
  table (latest fiscal year vs the year before, with YoY %, cited to
  SEC company facts) instead of whatever MD&A passage ranks first —
  and the ranking demotes definitional boilerplate for figure-seeking
  questions. In LLM mode the same headline is prepended verbatim and
  the model is steered to its fiscal year.
- **A premium web UI** (Next.js 15 + FastAPI): chat with a live step
  trail and expandable evidence cards, a sticky bottom composer with
  a conversation-context chip and a New chat control (clears the
  conversation, never the corpus), an Add a Company page with a
  live pipeline visualizer, a Chunking Lab with the measured
  comparison, and an Eval Dashboard with the 20-question results.

## Architecture

```mermaid
flowchart TB
    subgraph ingest["Ingestion — src/ingest.py"]
        EDGAR["SEC EDGAR (live fetch, cached)"] --> TXT["Clean text + Item section map"]
        FACTS["SEC company facts JSON"] --> FIG["Figures table (AAPL revenue / net income by FY)"]
        EDGAR --> SES["Session ingest, any ticker — src/session_ingest.py + src/registry.py"]
    end
    subgraph index["Chunk + index — src/chunking.py · src/retrieval.py"]
        TXT --> CH1["fixed: 1,000/200 splitter — 856 chunks"]
        TXT --> CH2["section: Item-aware packer — 224 chunks"]
        CH1 --> IDX["Per-strategy index: FAISS (bge-small) + BM25"]
        CH2 --> IDX
    end
    subgraph graph["LangGraph app — src/graph.py"]
        Q["Question"] --> SCOPE["scope: prediction / out-of-corpus → refuse"]
        SCOPE --> RET["retrieve (per-company for comparisons)"]
        RET --> GRADE["grade: company-match at rank (+ cross-encoder)"]
        GRADE -- weak, once --> RW["rewrite query"] --> RET
        GRADE -- weak, spent --> REF["refuse honestly"]
        GRADE -- relevant, no company named --> CLAR["clarify: which company?"]
        GRADE -- relevant --> GEN["generate: LLM or extractive, cited"]
        GEN --> VER["verify: every number grounded?"]
        VER -- fail, once --> GEN
        VER -- fail, twice --> REF
        VER -- ok --> OUT["Answer + citations + step trace"]
    end
    SES --> CH1
    SES --> CH2
    SES --> FIG
    IDX --> RET
    FIG --> GEN
    FIG --> VER
    graph --> API["FastAPI — src/api.py"]
    API --> WEB["Next.js UI — web/ (Chat · Add a Company · Chunking Lab · Eval Dashboard)"]
```

## Add a Company — live ingestion

The bundled corpus is a frozen snapshot. **Add a Company** (the
`/ingest` page, or the API below) runs the *same* pipeline for any
SEC-listed company, live, in seven recorded stages:

`resolve → fetch → clean → chunk → embed → figures → ready`

1. **Resolve** — a name or ticker is matched against the SEC's
   `company_tickers.json` map (`src/edgar.py`; cached, declared UA,
   fair-access pacing) and confirmed by CIK.
2. **Fetch** — the company's latest Form 10-K via the EDGAR
   submissions API + Archives.
3. **Clean** — HTML → text, then whole-Item slicing (Items 1, 1A, 7, 8)
   with the *same per-item char caps* as the bundled corpus.
4. **Chunk** — both strategies; counts are reported per strategy.
5. **Embed** — bge-small embeddings; the active corpus's FAISS index
   *extends* the bundled snapshot's persisted vectors, embedding only
   the new company's chunks (batched — one-shot batch embedding of a
   whole filing is a memory spike, not a shortcut).
6. **Figures** — the company's SEC company-facts payload reduced to
   revenue + net income by fiscal year with the same period-end
   grouping as the AAPL table (`figures_from_facts` — the AAPL code
   path is the generalized one; AAPL output is unchanged).
7. **Ready** — the company registers in the corpus registry
   (`src/registry.py`), persisted under `data/session/<TICKER>/`, so a
   server restart keeps it.

**Session vs bundled.** The registry is the single source of truth
for what the chat may answer about: the graph's scope node, company
alias detection, refusal copy ("my corpus covers…"), retrieval
targeting, and the figures table all read it dynamically — a company
is in-corpus iff it is bundled *or* registered, and the moment Tesla
registers, "What was Tesla's revenue in fiscal 2024?" goes from an
honest refusal to a cited answer. Chat accepts a corpus selector —
`active` (bundled + session, the default), `bundled`, or one ticker.
The measured artifacts never move: the comparison experiment and the
20-question evaluation are pinned to the bundled corpus, and the Lab
and Eval pages say so ("Measured on the bundled corpus (AAPL · MSFT ·
NVDA snapshot, Oct 2026). Session-ingested companies are not part of
these measurements.").

**Tesla pre-cache.** `data/precached/TSLA/` holds the output of one
real run of this pipeline (FY2025 10-K, accession
0001628280-26-003952, sliced to the same caps; provenance note in the
folder), so `POST /api/ingest {"ticker": "TSLA"}` completes with zero
network — the refusal → ingest → answer demo arc never depends on
SEC latency. Every other ticker goes live to EDGAR. Rebuild with
`.venv/bin/python scripts/precache_company.py TSLA`.

**API** — `POST /api/ingest/resolve {query}` → candidates;
`POST /api/ingest {ticker}` → `{job_id}` (background job);
`GET /api/ingest/status/{job_id}` → stage-by-stage progress + result
(accession, items, chars, chunk counts, figures rows);
`GET /api/corpus` lists bundled documents plus session companies
with their source (`bundled` · `precached` · `live`);
`POST /api/chat` accepts `corpus: "active" | "bundled" | "<TICKER>"`.

## Live market context

The app sits inside real market data, so a filing answer never floats
in a vacuum. A **market strip** runs above the chat (S&P 500, Nasdaq,
Dow, plus every active-corpus company — price, day change, 1-month
sparkline, refreshed every 60s), a **company snapshot** in the chat's
side rail pairs the market's view of the company in focus (delayed
quote, 6-month price chart with 1M/6M/1Y ranges, market cap, P/E,
52-week range) with **the filing's view** — the latest revenue and
net income from the same corpus figures table the chat cites — and a
**trending rail** on the Add a Company page turns Yahoo's trending
tickers into one-tap 10-K ingestions.

Sources are free and keyless: **Yahoo Finance** (via `yfinance`) is
primary, **Stooq**'s CSV endpoints are the fallback, and every
payload carries its `source` and `as_of` with `delayed: true` — the
UI labels the data "Delayed quotes", because it is. Quotes cache for
60s, history for 15 minutes, trending for 5, and any source failure
degrades to `available: false` (the strip simply doesn't render)
rather than an error. Market data is a presentation layer only: it
**never** feeds chat answers, grading, or the eval. Endpoints:
`GET /api/market/quote`, `/history`, `/overview`, `/trending`,
`/filing-figures`.

## Measured results (real runs, committed under `results/`)

**Chunking × retrieval comparison** — 12 questions with known gold
sections (`python -m src.experiments`, full write-up in
`docs/comparison.md`):

| Chunking | Retrieval | Hit@4 | MRR | Mean latency |
|---|---|---|---|---|
| fixed | vector | 0.92 | 0.93 | 20 ms |
| fixed | hybrid | 0.92 | 0.88 | 17 ms |
| fixed | hybrid+rerank | 0.92 | 0.89 | 817 ms |
| section | vector | 1.00 | 0.85 | 25 ms |
| section | hybrid | 0.83 | 0.74 | 22 ms |
| section | hybrid+rerank | **0.92** | **0.93** | 2,355 ms |

The interesting findings: dense-only retrieval already saturates on a
corpus this small; RRF hybrid *hurt* section chunks (BM25 promotes
cross-company keyword matches); reranking's value is at the top of the
ranking (MRR 0.74 → 0.93), which is exactly what generation and the
verifier consume — so the app defaults to section + hybrid+rerank and
pays the ~2.4 s.

**Evaluation** — 18 questions through the full graph (9 answerable,
2 comparisons, 2 ambiguous, 2 unanswerable, plus a figures-headline
performance question, a two-turn follow-up that must keep its company
via context, and the same follow-up with no context, which must
clarify), `python -m src.evaluate`, report in
`docs/evaluation-report.md`: **20/20 passed, faithfulness 100% among
answered questions** in extractive mode on local CPU. The first
measured run of the original 15 passed 8/15; the failure analysis
and the mechanism-level fixes are documented in
`docs/project-documentation.md`.

**Tests** — 41 pytest tests (chunking invariants, figures-table
values, retrieval determinism, graph refusal/clarification/answer
paths, conversation contextualization, performance headlines and
YoY-percentage verification, ticker resolution, precache ingest,
session refresh-on-re-add with a faked EDGAR, Item-slicer heading
styles across filers (including Netflix's no-space iXBRL headings),
the dynamic refuse → ingest → answer scope arc, plus the market
layer's fallback order, cache behavior, and degrade-to-unavailable
contract with injected fake fetchers), all passing:
`python -m pytest tests/ -q`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.ingest          # corpus stats (bundled fixtures; no network needed)
```

Optional — regenerate every measured artifact:

```bash
.venv/bin/python -m src.experiments     # -> results/comparison.json + docs/comparison.md
.venv/bin/python -m src.evaluate        # -> results/eval.json + docs/evaluation-report.md
.venv/bin/python -m pytest tests/ -q
```

## Run

Two processes:

```bash
# 1) API (LangGraph service)
.venv/bin/python -m uvicorn src.api:app --port 8000

# 2) Web UI (production build)
npm --prefix web install
npm --prefix web run build
npm --prefix web run start              # http://localhost:3000
```

The web app reads `NEXT_PUBLIC_API_BASE` (default
`http://localhost:8000`). Or skip the UI and ask the graph directly:

```bash
.venv/bin/python -m src.graph "What was Apple's total net sales in fiscal 2024?"
```

### LLM mode (optional)

```bash
export OPENAI_API_KEY=...               # or NEBIUS_API_KEY=...
export LLM_MODEL=gpt-4o-mini            # default per provider
```

Nebius uses base URL `https://api.studio.nebius.com/v1/` with default
model `meta-llama/Llama-3.3-70B-Instruct`. With no key set, the app
runs in extractive grounded mode and labels itself as such in the UI.
Never commit a key — this repo's history is clean of secrets.

## Repo layout

```
src/ingest.py       EDGAR ingestion, HTML cleaning, figures table
src/edgar.py        SEC company-map ticker resolution
src/registry.py     corpus registry (bundled + session tiers)
src/session_ingest.py  live ingest pipeline + job records
src/chunking.py     fixed vs section-aware chunking
src/retrieval.py    FAISS + BM25, RRF fusion, cross-encoder rerank
src/experiments.py  chunking x retrieval comparison experiment
src/graph.py        LangGraph Q&A app (contextualize/scope/retrieve/grade/rewrite/generate/verify/refuse/clarify)
src/evaluate.py     20-question evaluation harness
src/questions.py    gold question sets
src/api.py          FastAPI backend for the web UI
web/                Next.js 15 UI (Chat, Add a Company, Chunking Lab, Eval Dashboard)
data/               bundled corpus fixtures + provenance notes
data/precached/     committed pipeline output for TSLA (zero-network ingest)
data/session/       runtime-ingested companies (persisted registrations)
results/            measured outputs (comparison.json, eval.json)
docs/               framework.md, comparison.md, evaluation-report.md, project-documentation.md
tests/              pytest suite
```

## Public demo limits

The deployed demo is an open site, and in LLM mode every question
burns the owner's API key — so the API rate-limits itself
(`src/rate_limit.py`, HTTP 429 with a friendly in-product message):

- **Chat** — 30 questions per visitor IP per hour
  (`CHAT_LIMIT_PER_HOUR`), a global circuit breaker of 600 questions
  per day across all visitors (`CHAT_DAILY_CAP`), and at most 4
  generations in flight at once (`CHAT_MAX_CONCURRENCY`).
- **Add a Company** — 6 ingests per visitor IP per hour
  (`INGEST_LIMIT_PER_HOUR`); the session corpus holds at most 40
  live-ingested companies (`SESSION_COMPANY_CAP`). Re-adding a
  company that is already registered (the refresh path) is never
  blocked by the cap.

Visitor identity is the first `X-Forwarded-For` hop (the site is
reached through the Vercel rewrite proxy). Counters are in-memory
and reset on redeploy — acceptable at demo scale.

## Honest limitations

- **Corpus scope is small and excerpted** — four documents, Items
  1/1A/7/8 only, tables flattened to text. NVIDIA's Item 8 in its filing
  is a pointer paragraph, so its statements arrive via Item 7. Numbers
  in dense tables can be hard for sentence-level extraction to surface
  (see the E11 note in the evaluation report).
- **Extractive mode quotes filing sentences verbatim.** Every figure is
  grounded and cited, but the prose reads like filings, not like a
  written summary. LLM mode composes fluently under the same citation
  and verification rules.
- **Grading is calibrated to this corpus.** Absolute cross-encoder
  logits proved uncalibrated on legal prose (a gold chunk scored −4.8
  while cross-company noise scored +2.2), so company-named questions
  grade on company-match at retrieval rank — measured to be the signal
  that holds here. It would need re-calibration on a larger corpus.
- **Verification checks numbers, not prose claims.** A sentence can be
  topically wrong while containing only grounded figures; the eval's
  gold-section citation requirement is the backstop, and an LLM judge
  joins the grade step when a key is configured.
- Local CPU inference sets the latency profile: ~20 ms retrieval
  without rerank, ~2.4 s with, ~3 s end-to-end. A hosted embedding /
  rerank endpoint would change the trade-off, not the design.

## Framing

Built for Gen Academy Week 2 (Use Case 2 — Financial Document
Intelligence). The full submission documentation — overview, datasets,
the actual prompt templates, iterations, and learnings — is in
`docs/project-documentation.md`; the handout framework fields are in
`docs/framework.md`.
