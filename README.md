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
- **A premium web UI** (Next.js 15 + FastAPI): chat with a live step
  trail and expandable evidence cards, a Chunking Lab with the measured
  comparison, and an Eval Dashboard with the 15-question results.

## Architecture

```mermaid
flowchart TB
    subgraph ingest["Ingestion — src/ingest.py"]
        EDGAR["SEC EDGAR (live fetch, cached)"] --> TXT["Clean text + Item section map"]
        FACTS["SEC company facts JSON"] --> FIG["Figures table (AAPL revenue / net income by FY)"]
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
    IDX --> RET
    FIG --> GEN
    FIG --> VER
    graph --> API["FastAPI — src/api.py"]
    API --> WEB["Next.js UI — web/ (Chat · Chunking Lab · Eval Dashboard)"]
```

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

**Evaluation** — 15 questions through the full graph (9 answerable,
2 comparisons, 2 ambiguous, 2 unanswerable), `python -m src.evaluate`,
report in `docs/evaluation-report.md`: **15/15 passed, faithfulness
100% among answered questions, mean end-to-end latency ~3.0 s** in
extractive mode on local CPU. The first measured run passed 8/15; the
failure analysis and the mechanism-level fixes are documented in
`docs/project-documentation.md`.

**Tests** — 12 pytest tests (chunking invariants, figures-table
values, retrieval determinism, graph refusal/clarification/answer
paths), all passing: `python -m pytest tests/ -q`.

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
src/chunking.py     fixed vs section-aware chunking
src/retrieval.py    FAISS + BM25, RRF fusion, cross-encoder rerank
src/experiments.py  chunking x retrieval comparison experiment
src/graph.py        LangGraph Q&A app (scope/retrieve/grade/rewrite/generate/verify/refuse/clarify)
src/evaluate.py     15-question evaluation harness
src/questions.py    gold question sets
src/api.py          FastAPI backend for the web UI
web/                Next.js 15 UI (Chat, Chunking Lab, Eval Dashboard)
data/               bundled corpus fixtures + provenance notes
results/            measured outputs (comparison.json, eval.json)
docs/               framework.md, comparison.md, evaluation-report.md, project-documentation.md
tests/              pytest suite
```

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
