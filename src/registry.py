"""Corpus registry — what filings the chat may answer from.

Two tiers:

* **Bundled** — the frozen AAPL/MSFT/NVDA snapshot in ``data/`` that
  the comparison experiment and the 15-question evaluation measured.
  Immutable at runtime.
* **Session** — companies ingested live via "Add a Company"
  (:mod:`src.session_ingest`). Each registration persists under
  ``data/session/<TICKER>/`` (``filing.txt``, ``figures.json``,
  ``meta.json``) so a restart keeps the company, and this module is
  the single reader the graph, retrieval and API consult: scope
  decisions, alias detection, refusal coverage copy and the figures
  table all derive from the registry, never from hardcoded lists.

``data/precached/<TICKER>/`` mirrors the session layout for companies
(Tesla) whose sliced corpus is committed so the ingest demo runs
with zero network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.documents import Document

from src.ingest import (COMPANY_NAMES, DATA_DIR, TICKER_CIK, _COMMENT_HEADER_RE,
                        load_figures)

SESSION_DIR = DATA_DIR / "session"
PRECACHE_DIR = DATA_DIR / "precached"

BUNDLED_TICKERS = tuple(TICKER_CIK.keys())

_cache: dict[str, dict] | None = None


def reset_cache() -> None:
    global _cache
    _cache = None


def _read_company(base: Path) -> tuple[dict, str, list[dict]] | None:
    meta_path = base / "meta.json"
    text_path = base / "filing.txt"
    if not meta_path.exists() or not text_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = text_path.read_text(encoding="utf-8", errors="replace")
    text = _COMMENT_HEADER_RE.sub("", raw).strip()
    figures: list[dict] = []
    fig_path = base / "figures.json"
    if fig_path.exists():
        figures = json.loads(fig_path.read_text(encoding="utf-8"))
    return meta, text, figures


def session_companies() -> dict[str, dict]:
    """{ticker: meta} for every registered session company."""
    global _cache
    if _cache is None:
        _cache = {}
        if SESSION_DIR.exists():
            for base in sorted(SESSION_DIR.iterdir()):
                found = _read_company(base)
                if found:
                    meta, _text, _figs = found
                    _cache[meta["ticker"]] = meta
    return _cache


def register_company(meta: dict, text: str, figures: list[dict]) -> dict:
    """Persist a session company and make it visible to the registry."""
    ticker = meta["ticker"]
    base = SESSION_DIR / ticker
    base.mkdir(parents=True, exist_ok=True)
    header = (
        f"<!-- {meta['source']} excerpt of {ticker} Form 10-K, accession "
        f"{meta['accession']}, report period ended {meta['report_date']}, "
        f"filed {meta['filed']}. Sliced by Item with per-item char caps "
        f"by the live ingest pipeline. -->\n"
    )
    (base / "filing.txt").write_text(header + text, encoding="utf-8")
    (base / "figures.json").write_text(json.dumps(figures, indent=1), encoding="utf-8")
    (base / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    session_companies()[ticker] = meta
    return meta


def precached_company(ticker: str) -> tuple[dict, str, list[dict]] | None:
    """(meta, text, figures) from data/precached/<TICKER>/, if present."""
    return _read_company(PRECACHE_DIR / ticker.upper())


def session_documents() -> list[Document]:
    """Session companies as Documents, same metadata shape as bundled."""
    docs: list[Document] = []
    if SESSION_DIR.exists():
        for base in sorted(SESSION_DIR.iterdir()):
            found = _read_company(base)
            if not found:
                continue
            meta, text, _figs = found
            docs.append(Document(
                page_content=text,
                metadata={
                    "ticker": meta["ticker"],
                    "company": meta["name"],
                    "form": "10-K",
                    "fiscal_year": meta["fiscal_year"],
                    "doc_label": meta["doc_label"],
                    "accession": meta["accession"],
                    "source_file": f"session:{meta['ticker']}",
                },
            ))
    return docs


def session_figures() -> list[dict]:
    rows: list[dict] = []
    for ticker in session_companies():
        found = _read_company(SESSION_DIR / ticker)
        if found:
            rows.extend(found[2])
    return rows


def all_figures() -> list[dict]:
    """Bundled (AAPL) figures + every session company's figures."""
    return load_figures() + session_figures()


def known_tickers() -> set[str]:
    return set(BUNDLED_TICKERS) | set(session_companies())


def company_name(ticker: str) -> str:
    if ticker in COMPANY_NAMES:
        return COMPANY_NAMES[ticker]
    meta = session_companies().get(ticker)
    return meta["name"] if meta else ticker


_NAME_SUFFIXES = {"inc", "inc.", "corporation", "corp", "corp.", "company",
                  "co", "co.", "ltd", "ltd.", "llc", "group", "holdings"}


def _aliases_for(ticker: str, name: str) -> set[str]:
    aliases = {ticker.lower(), name.lower()}
    words = [w.strip(",.").lower() for w in name.split()]
    core = [w for w in words if w and w not in _NAME_SUFFIXES]
    if core:
        aliases.add(" ".join(core))
        if len(core) <= 2:
            aliases.add(core[0])
    return {a for a in aliases if len(a) > 1}


def company_aliases(tickers: set[str] | None = None) -> dict[str, str]:
    """{alias: ticker} for mention detection over the given tickers
    (default: every known company, bundled + session)."""
    tickers = tickers if tickers is not None else known_tickers()
    out: dict[str, str] = {}
    for ticker in sorted(tickers):
        for alias in _aliases_for(ticker, company_name(ticker)):
            out.setdefault(alias, ticker)
    return out


def short_name(ticker: str) -> str:
    """Display name without the legal suffix: 'Tesla, Inc.' -> 'Tesla'."""
    name = company_name(ticker)
    words = [w.strip(",") for w in name.replace(",", " ").split()]
    core = [w for w in words if w.lower().strip(".") not in _NAME_SUFFIXES]
    return " ".join(core) if core else name


def coverage_sentence(tickers: set[str]) -> tuple[str, str]:
    """('Apple, Microsoft, and NVIDIA', ' plus SEC ... figures for ...')
    — the two pieces refusal copy is composed from."""
    names = [short_name(t) for t in sorted(tickers)]
    if len(names) > 1:
        listing = ", ".join(names[:-1]) + f", and {names[-1]}"
    else:
        listing = names[0] if names else ""
    fig_tickers = [t for t in sorted(tickers)
                   if t == "AAPL" or t in session_companies()]
    fig_clause = ""
    if fig_tickers:
        figs = [short_name(t) for t in fig_tickers]
        fig_listing = ", ".join(figs[:-1]) + (f" and {figs[-1]}" if len(figs) > 1 else figs[0])
        fig_clause = f" plus SEC company-facts figures for {fig_listing}"
    return listing, fig_clause
