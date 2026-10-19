"""Session ingest pipeline — "Add a Company".

Runs one ticker through the same pipeline that built the bundled
corpus, stage by stage, recording progress into a job record the API
polls:

    resolve -> fetch -> clean -> chunk -> embed -> figures -> ready

* **resolve** — ticker -> company name + CIK (SEC ticker map).
* **fetch** — the latest 10-K from EDGAR submissions + Archives
  (declared UA, fair-access pacing, cached). Tesla is served from
  ``data/precached/TSLA/`` with zero network so the demo never
  depends on SEC latency.
* **clean** — HTML -> text, then whole-Item slicing with the bundled
  corpus caps (Items 1, 1A, 7, 8).
* **chunk** — both strategies (fixed + section-aware), counts kept.
* **embed** — embeddings + indexes rebuilt for the active corpus
  (bundled + session), so chat can retrieve the new company at once.
* **figures** — the company's SEC company-facts payload reduced to
  revenue + net income rows by fiscal year (period-end grouping).
* **ready** — registered in the corpus registry (persisted under
  ``data/session/<TICKER>/``).
"""

from __future__ import annotations

import threading
import uuid

from src import edgar, registry
from src.chunking import chunk_document
from src.ingest import (COMPANY_NAMES, TICKER_CIK, fetch_company_facts,
                        fetch_filing_text_for, figures_from_facts,
                        latest_10k_meta_for, load_bundled_documents, slice_items)

STAGES = ["resolve", "fetch", "clean", "chunk", "embed", "figures", "ready"]

_STAGE_DETAILS = {
    "resolve": "ticker → company + CIK (SEC company map)",
    "fetch": "latest 10-K from SEC EDGAR",
    "clean": "HTML → text, whole-Item slicing (Items 1, 1A, 7, 8)",
    "chunk": "fixed-size + section-aware chunking",
    "embed": "bge-small embeddings + FAISS/BM25 index for the active corpus",
    "figures": "SEC company-facts → revenue + net income by fiscal year",
    "ready": "registered in the corpus",
}


class IngestError(Exception):
    """A pipeline failure with a user-honest reason."""


def _new_stages() -> list[dict]:
    return [{"name": n, "status": "pending", "detail": _STAGE_DETAILS[n]}
            for n in STAGES]


def _mark(job: dict | None, stage: str, status: str, detail: str = "") -> None:
    if job is None:
        return
    job["stage"] = stage
    for s in job["stages"]:
        if s["name"] == stage:
            s["status"] = status
            if detail:
                s["detail"] = detail


def _fail(job: dict | None, stage: str, reason: str) -> None:
    _mark(job, stage, "error", reason)
    if job is not None:
        job["stage"] = "error"
        job["error"] = reason


def ingest_company(ticker: str, job: dict | None = None,
                   allow_network: bool = True,
                   latest_meta_fn=None) -> dict:
    """Run the full pipeline for one ticker; returns the result meta.

    Raises IngestError with an honest reason on failure. ``job`` (from
    :func:`create_job`) is updated in place as stages complete.

    Re-adding a SESSION company refreshes it: EDGAR's latest 10-K
    accession is compared with the registered one — identical means
    "already current" (no download); a newer filing means a full
    re-ingest that replaces the registry entry. Bundled companies are
    the frozen snapshot and never refresh. ``latest_meta_fn`` injects
    the EDGAR latest-10-K lookup (tests fake it; default
    :func:`src.ingest.latest_10k_meta_for`).
    """
    ticker = ticker.strip().upper()
    try:
        # -- resolve ------------------------------------------------------
        _mark(job, "resolve", "active")
        if ticker in TICKER_CIK:
            name, cik = COMPANY_NAMES[ticker], TICKER_CIK[ticker]
        else:
            row = edgar.company_for_ticker(ticker)
            if row is None:
                raise IngestError(
                    f"unknown ticker {ticker!r} — the SEC company map has no such symbol")
            name, cik = row["name"], row["cik"]
        _mark(job, "resolve", "done", f"{name} · CIK {cik}")

        existing = registry.session_companies().get(ticker)
        refresh_from: dict | None = None
        if existing is not None:
            latest = None
            if allow_network:
                try:
                    latest = (latest_meta_fn or latest_10k_meta_for)(cik, ticker)
                except Exception:
                    latest = None  # EDGAR unreachable: keep what we have
            if latest is None or latest["accession"] == existing["accession"]:
                for stage in STAGES[1:]:
                    _mark(job, stage, "done",
                          "already in the corpus — nothing to fetch")
                if latest is not None:
                    _mark(job, "fetch", "done",
                          f"already current — EDGAR's latest 10-K is accession "
                          f"{latest['accession']}, filed {latest['filed']}, "
                          f"the one already in the corpus")
                    _mark(job, "ready", "done",
                          f"already current (accession {latest['accession']}, "
                          f"filed {latest['filed']})")
                else:
                    _mark(job, "ready", "done",
                          "already registered in this session corpus")
                if job is not None:
                    job["stage"] = "ready"
                    job["result"] = existing
                return existing
            # A newer 10-K exists: fall through and re-ingest, replacing
            # the registry entry. The precache must NOT serve the old
            # filing, so it is bypassed for this run.
            refresh_from = {
                "accession": existing["accession"],
                "new_accession": latest["accession"],
                "filed": latest["filed"],
            }
            _mark(job, "fetch", "active",
                  f"newer filing found: 10-K accession {latest['accession']} "
                  f"(filed {latest['filed']}) replaces {existing['accession']}")

        if ticker in TICKER_CIK:
            # Bundled company: it is already part of the frozen corpus.
            docs = [d for d in load_bundled_documents()
                    if d.metadata["ticker"] == ticker]
            meta = _bundled_meta(ticker, name, docs)
            for stage in STAGES[1:]:
                _mark(job, stage, "done", "part of the bundled corpus snapshot")
            _mark(job, "ready", "done", "already in the bundled corpus")
            if job is not None:
                job["stage"] = "ready"
                job["result"] = meta
            return meta

        # -- fetch + clean --------------------------------------------------
        precached = None if refresh_from else registry.precached_company(ticker)
        if precached is not None:
            meta_pre, text, figures = precached
            _mark(job, "fetch", "done",
                  f"precached excerpt of accession {meta_pre['accession']} — zero network")
            _mark(job, "clean", "done",
                  f"Items {', '.join(meta_pre['items'])} · {len(text):,} chars (pre-sliced)")
            accession = meta_pre["accession"]
            fiscal_year = meta_pre["fiscal_year"]
            filed, report_date = meta_pre["filed"], meta_pre["report_date"]
            items = meta_pre["items"]
            source = "precached"
        else:
            if not allow_network:
                raise IngestError(
                    f"{ticker} is not precached and network access is disabled")
            _mark(job, "fetch", "active")
            try:
                meta10k = (latest_meta_fn or latest_10k_meta_for)(cik, ticker)
                full_text, meta10k = fetch_filing_text_for(cik, ticker)
            except IngestError:
                raise
            except Exception as exc:  # EDGAR down, no 10-K, bad payload…
                raise IngestError(f"EDGAR fetch failed for {ticker}: {exc}") from exc
            if refresh_from:
                _mark(job, "fetch", "done",
                      f"newer filing ingested: 10-K accession "
                      f"{meta10k['accession']}, filed {meta10k['filed']} — "
                      f"replaces accession {refresh_from['accession']}")
            else:
                _mark(job, "fetch", "done",
                      f"10-K accession {meta10k['accession']}, filed {meta10k['filed']}")
            _mark(job, "clean", "active")
            text, items = slice_items(full_text)
            if not items:
                raise IngestError(
                    f"no slicable Items (1, 1A, 7, 8) found in {ticker}'s latest 10-K")
            _mark(job, "clean", "done",
                  f"Items {', '.join(items)} · {len(text):,} chars after caps")
            accession = meta10k["accession"]
            fiscal_year = int(meta10k["report_date"][:4])
            filed, report_date = meta10k["filed"], meta10k["report_date"]
            figures = None  # built at the figures stage below
            source = "live"

        from langchain_core.documents import Document
        doc = Document(
            page_content=text,
            metadata={
                "ticker": ticker, "company": name, "form": "10-K",
                "fiscal_year": fiscal_year,
                "doc_label": f"{ticker} 10-K FY{fiscal_year}",
                "accession": accession, "source_file": f"session:{ticker}",
            },
        )

        # -- chunk ----------------------------------------------------------
        _mark(job, "chunk", "active")
        chunk_counts = {}
        for strategy in ("fixed", "section"):
            chunk_counts[strategy] = len(chunk_document(doc, strategy))
        _mark(job, "chunk", "done",
              f"fixed {chunk_counts['fixed']} · section-aware {chunk_counts['section']} chunks")

        meta = {
            "ticker": ticker, "name": name, "cik": cik, "source": source,
            "accession": accession, "fiscal_year": fiscal_year,
            "filed": filed, "report_date": report_date,
            "doc_label": f"{ticker} 10-K FY{fiscal_year}",
            "items": items, "chars": len(text), "chunks": chunk_counts,
            "figures_rows": 0,
        }

        # -- figures --------------------------------------------------------
        _mark(job, "figures", "active")
        if figures is None:
            try:
                payload = fetch_company_facts(cik, ticker)
                figures = figures_from_facts(
                    payload.get("facts", {}), ticker,
                    f"SEC company facts (companyfacts_{ticker.lower()}.json)")
            except Exception as exc:
                raise IngestError(
                    f"company-facts fetch failed for {ticker}: {exc}") from exc
        meta["figures_rows"] = len(figures)
        _mark(job, "figures", "done",
              f"{len(figures)} rows (revenue + net income by fiscal year)")

        # -- register, then embed/index the enlarged active corpus ---------
        registry.register_company(meta, text, figures)
        _mark(job, "embed", "active")
        from src import retrieval
        retrieval.invalidate_indexes("active")
        retrieval.build_active_indexes()
        _mark(job, "embed", "done",
              "section-aware index extended with the new company's chunks "
              "(fixed-size index builds on first use)")

        _mark(job, "ready", "done", f"{name} is in the corpus — ask it in Chat")
        if job is not None:
            job["stage"] = "ready"
            job["result"] = meta
        return meta
    except IngestError as exc:
        _fail(job, job["stage"] if job else "resolve", str(exc))
        raise
    except Exception as exc:  # unexpected: still report honestly
        _fail(job, job["stage"] if job else "resolve", f"ingest failed: {exc}")
        raise IngestError(f"ingest failed for {ticker}: {exc}") from exc


def _bundled_meta(ticker: str, name: str, docs: list) -> dict:
    from src.ingest import load_figures
    chars = sum(len(d.page_content) for d in docs)
    counts = {"fixed": 0, "section": 0}
    for d in docs:
        for strategy in counts:
            counts[strategy] += len(chunk_document(d, strategy))
    figures_rows = len([f for f in load_figures() if f["ticker"] == ticker]) \
        if ticker == "AAPL" else 0
    return {
        "ticker": ticker, "name": name, "cik": TICKER_CIK[ticker],
        "source": "bundled",
        "accession": docs[0].metadata["accession"] if docs else "",
        "fiscal_year": docs[0].metadata["fiscal_year"] if docs else 0,
        "filed": "", "report_date": "",
        "doc_label": f"{ticker} (bundled corpus)",
        "items": [], "chars": chars, "chunks": counts,
        "figures_rows": figures_rows,
    }


# ---------------------------------------------------------------------------
# Job store (in-memory; the API polls these)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def create_job(ticker: str) -> dict:
    job = {
        "job_id": uuid.uuid4().hex[:12],
        "ticker": ticker.strip().upper(),
        "stage": "resolve",
        "stages": _new_stages(),
        "result": None,
        "error": None,
    }
    with _jobs_lock:
        _jobs[job["job_id"]] = job
    return job


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def run_job(job: dict) -> None:
    """Thread target: run the pipeline, recording into the job."""
    try:
        ingest_company(job["ticker"], job=job)
    except IngestError:
        pass  # reason already recorded on the job by _fail
