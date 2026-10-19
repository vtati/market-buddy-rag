"""Ingestion — SEC EDGAR 10-K filings -> cleaned text documents + a
structured figures table.

Two corpus paths:

* **Bundled fixtures** (offline, always available): real EDGAR slices in
  ``data/`` — the AAPL FY2024 10-K excerpt (Items 7/7A/8) that shipped with
  the scaffold, plus live-fetched excerpts for AAPL (Items 1/1A), MSFT and
  NVDA produced by ``scripts/build_corpus.py``. Provenance for every file
  is in ``data/PROVENANCE-*.md``.
* **Live EDGAR**: :func:`fetch_latest_10k` resolves a ticker's most recent
  10-K from the SEC submissions JSON and downloads the primary document,
  honouring SEC fair access (declared user agent, >= 0.11 s pacing).
  Downloads are cached under ``data/cache/`` (gitignored).

HTML is stripped with the standard-library HTML parser — inline-XBRL
filings are tag soup with heavy inline styles, and a forgiving parser
plus whitespace normalisation gets clean, line-structured text where
"Item N." headings land at line starts (the section chunker relies on
that).

The figures table parses ``data/companyfacts_aapl.json`` (the SEC
company-facts payload for Apple) into one row per (concept, fiscal year)
for annual 10-K figures — the graph cites it alongside text chunks.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from bisect import bisect_right
from html.parser import HTMLParser
from pathlib import Path

from langchain_core.documents import Document

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

EDGAR_UA = "GenAlphaLabs MarketBuddyRAG demo contact@example.com"
PACING_SECONDS = 0.15  # SEC fair access asks for <= 10 req/s; we stay far under

TICKER_CIK = {"AAPL": "0000320193", "MSFT": "0000789019", "NVDA": "0001045810"}
COMPANY_NAMES = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation"}

# Bundled corpus manifest: file -> document metadata. Accessions/report
# dates are copied from the filings themselves (see data/PROVENANCE-*).
BUNDLED_FILES = [
    {
        "file": "filing_aapl_10k_2024_excerpt.htm",
        "ticker": "AAPL",
        "doc_label": "AAPL 10-K FY2024",
        "fiscal_year": 2024,
        "accession": "0000320193-24-000123",
        "note": "Items 7, 7A, 8 excerpt (scaffold fixture)",
    },
    {
        "file": "filing_aapl_10k_items1_1a.txt",
        "ticker": "AAPL",
        "doc_label": "AAPL 10-K FY2025",
        "fiscal_year": 2025,
        "accession": "0000320193-25-000079",
        "note": "Items 1, 1A excerpt (live-fetched, sliced by scripts/build_corpus.py)",
    },
    {
        "file": "filing_msft_10k_excerpt.txt",
        "ticker": "MSFT",
        "doc_label": "MSFT 10-K FY2026",
        "fiscal_year": 2026,
        "accession": "0001193125-26-323660",
        "note": "Items 1, 1A, 7, 8 excerpt (live-fetched, sliced by scripts/build_corpus.py)",
    },
    {
        "file": "filing_nvda_10k_excerpt.txt",
        "ticker": "NVDA",
        "doc_label": "NVDA 10-K FY2026",
        "fiscal_year": 2026,
        "accession": "0001045810-26-000021",
        "note": "Items 1, 1A, 7, 8 excerpt (live-fetched, sliced by scripts/build_corpus.py)",
    },
]


# ---------------------------------------------------------------------------
# HTML -> text
# ---------------------------------------------------------------------------

class _TextStripper(HTMLParser):
    """Block tags become newlines, table cells become spaces."""

    NEWLINE_TAGS = {"div", "p", "tr", "h1", "h2", "h3", "h4", "li", "table", "br", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):  # noqa: D102
        if tag in self.NEWLINE_TAGS:
            self.parts.append("\n")
        elif tag == "td":
            self.parts.append(" ")

    def handle_data(self, data):  # noqa: D102
        self.parts.append(data)


def html_to_text(raw: str) -> str:
    """Strip filing HTML to clean, line-structured plain text."""
    stripper = _TextStripper()
    stripper.feed(raw)
    text = "".join(stripper.parts)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Item-section detection
# ---------------------------------------------------------------------------

# Canonical 10-K item order and a title keyword that body headings carry.
# Table-of-contents lines and running page headers ("Item 1" with no
# title) are filtered out by requiring a plausible title.
ITEM_ORDER = ["1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A",
              "8", "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16"]
ITEM_TITLE_KEYWORDS = {
    "1": ("business",),
    "1A": ("risk factors",),
    "1B": ("unresolved staff comments",),
    "1C": ("cybersecurity",),
    "2": ("properties",),
    "3": ("legal proceedings",),
    "4": ("mine safety",),
    "5": ("market for",),
    "6": ("reserved", "selected financial", "[reserved]"),
    "7": ("management",),
    "7A": ("quantitative and qualitative",),
    "8": ("financial statements",),
    "9": ("changes in and disagreements",),
    "9A": ("controls and procedures",),
    "9B": ("other information",),
    "9C": ("disclosure regarding foreign",),
    "10": ("directors",),
    "11": ("executive compensation",),
    "12": ("security ownership",),
    "13": ("certain relationships",),
    "14": ("principal accountant",),
    "15": ("exhibit",),
    "16": ("form 10-k summary",),
}
_HEADING_RE = re.compile(r"^Item\s+(\d{1,2}[A-Z]?)\.\s*(.{2,110})$", re.IGNORECASE)
_ITEM_ONLY_RE = re.compile(r"^Item\s+(\d{1,2}[A-Z]?)\.$", re.IGNORECASE)


def _norm_line(line: str) -> str:
    """Collapse every whitespace run (incl. NBSP) to one space, stripped."""
    return re.sub(r"\s+", " ", line.replace("\xa0", " ")).strip()


def find_item_sections(text: str) -> dict[str, tuple[int, int]]:
    """Return ``{item: (start, end)}`` char spans for each 10-K Item body.

    Candidates are heading lines whose title matches the item's canonical
    title. When an item has several candidates (table of contents + body),
    the selection walks the canonical item order in reverse and keeps the
    last candidate that still precedes the next item's chosen heading —
    which lands on the body headings in real filings.

    Heading shapes vary by filer (Netflix's iXBRL 10-K broke this):
    ``Item 1.Business`` with NO space after the period, and TOC entries
    with the label alone on one line and the title on the next. Lines
    are whitespace-normalized before matching, the post-period space is
    optional, and a bare ``Item N.`` line pairs with the next non-empty
    line as its title. TOC entries pair up the same way, but the
    reverse-walk still prefers the later body headings, which are also
    the occurrences followed by substantial body text.
    """
    lines = text.split("\n")
    norms = [_norm_line(line) for line in lines]
    candidates: dict[str, list[int]] = {}
    offset = 0
    for i, line in enumerate(lines):
        norm = norms[i]
        m = _HEADING_RE.match(norm)
        if m:
            item = m.group(1).upper()
            title = m.group(2).strip().lower()
            keywords = ITEM_TITLE_KEYWORDS.get(item, ())
            if keywords and any(k in title for k in keywords):
                candidates.setdefault(item, []).append(offset)
        else:
            bare = _ITEM_ONLY_RE.match(norm)
            if bare:
                item = bare.group(1).upper()
                keywords = ITEM_TITLE_KEYWORDS.get(item, ())
                nxt = next((n for n in norms[i + 1:] if n), "")
                if keywords and any(k in nxt.lower() for k in keywords):
                    candidates.setdefault(item, []).append(offset)
        offset += len(line) + 1

    # TOC disambiguation. Table-of-contents entries form a dense
    # cluster — the next heading candidate follows within a few
    # hundred chars — while a body heading is followed by substantial
    # text before the next heading. Candidates with a >= 1000-char gap
    # to the next candidate (of ANY item) are "body" candidates and
    # win outright; an item with no body candidate falls back to all
    # its candidates (short closing sections in excerpt files).
    all_offsets = sorted(o for offs in candidates.values() for o in offs)

    def _gap(off: int) -> int:
        i = bisect_right(all_offsets, off)
        nxt = all_offsets[i] if i < len(all_offsets) else len(text)
        return nxt - off

    pools: dict[str, list[int]] = {}
    for item, offs in candidates.items():
        body = [o for o in offs if _gap(o) >= 1000]
        pools[item] = body or offs

    # Choose in canonical order, enforcing strictly increasing starts:
    # each item takes its latest pooled candidate that still follows
    # the previously chosen item's start. (A reverse walk lets one
    # TOC-only late item — e.g. a hyphen-variant Item 16 — drag every
    # earlier choice back into the contents cluster.)
    chosen: dict[str, int] = {}
    floor = -1
    for item in ITEM_ORDER:
        for off in sorted(pools.get(item, []), reverse=True):
            if off > floor:
                chosen[item] = off
                floor = off
                break

    spans: dict[str, tuple[int, int]] = {}
    ordered = sorted(chosen.items(), key=lambda kv: kv[1])
    for idx, (item, start) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(text)
        spans[item] = (start, end)
    return spans


# ---------------------------------------------------------------------------
# Live EDGAR
# ---------------------------------------------------------------------------

def _edgar_get(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": EDGAR_UA,
            "Accept-Encoding": "gzip, deflate",
            "Host": url.split("/")[2],
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            import gzip

            data = gzip.decompress(data)
    time.sleep(PACING_SECONDS)
    return data


def latest_10k_meta_for(cik: str, ticker: str = "") -> dict:
    """Most recent 10-K entry from the SEC submissions feed for a CIK."""
    subs = json.loads(_edgar_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    recent = subs["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "ticker": ticker,
                "accession": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
                "filed": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
            }
    raise RuntimeError(f"no 10-K found for {ticker or cik}")


def latest_10k_meta(ticker: str) -> dict:
    """Most recent 10-K entry from the SEC submissions feed for a ticker."""
    return latest_10k_meta_for(TICKER_CIK[ticker], ticker)


def fetch_filing_text_for(cik: str, ticker: str,
                          use_cache: bool = True) -> tuple[str, dict]:
    """Download the latest 10-K for a CIK and return (clean text, metadata)."""
    meta = latest_10k_meta_for(cik, ticker)
    cik_int = int(cik)
    acc_nodash = meta["accession"].replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
        f"{acc_nodash}/{meta['primary_document']}"
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker.lower()}_10k_{meta['accession']}.htm"
    if use_cache and cache_path.exists():
        raw = cache_path.read_text(encoding="utf-8", errors="replace")
    else:
        raw = _edgar_get(url).decode("utf-8", errors="replace")
        cache_path.write_text(raw, encoding="utf-8")
    meta["url"] = url
    return html_to_text(raw), meta


def fetch_filing_text(ticker: str, use_cache: bool = True) -> tuple[str, dict]:
    """Download a bundled ticker's latest 10-K (clean text, metadata)."""
    return fetch_filing_text_for(TICKER_CIK[ticker], ticker, use_cache=use_cache)


def fetch_company_facts(cik: str, ticker: str = "",
                        use_cache: bool = True) -> dict:
    """SEC company-facts payload for a CIK (cached under data/cache/)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"companyfacts_{ticker.lower() or cik}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = json.loads(_edgar_get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"))
    if use_cache:
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


# Per-Item character caps used when slicing a full 10-K down to a
# demo-sized excerpt — the same caps scripts/build_corpus.py applied
# to the bundled fixtures (whole Items, truncated at a line boundary).
ITEM_CAPS = {"1": 70_000, "1A": 90_000, "7": 130_000, "8": 70_000}
DEFAULT_SLICE_ITEMS = ("1", "1A", "7", "8")


def slice_items(text: str, items: tuple[str, ...] = DEFAULT_SLICE_ITEMS,
                ) -> tuple[str, list[str]]:
    """Slice whole Items out of a cleaned 10-K, capped per Item.

    Returns (excerpt text, items actually found). The heading line of
    each Item starts the slice, so find_item_sections/chunking keep
    working on the excerpt exactly as on a full filing.
    """
    spans = find_item_sections(text)
    parts: list[str] = []
    found: list[str] = []
    for item in items:
        if item not in spans:
            continue
        start, end = spans[item]
        body = text[start:end].strip()
        cap = ITEM_CAPS.get(item, 80_000)
        if len(body) > cap:
            body = body[:cap].rsplit("\n", 1)[0]
        parts.append(body)
        found.append(item)
    return "\n\n".join(parts), found


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

_COMMENT_HEADER_RE = re.compile(r"\A\s*<!--.*?-->\s*", re.DOTALL)


def load_bundled_documents() -> list[Document]:
    """Load every bundled fixture file as one LangChain Document."""
    docs: list[Document] = []
    for entry in BUNDLED_FILES:
        path = DATA_DIR / entry["file"]
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        if entry["file"].endswith((".htm", ".html")):
            text = html_to_text(raw)
        else:
            text = _COMMENT_HEADER_RE.sub("", raw).strip()
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "ticker": entry["ticker"],
                    "company": COMPANY_NAMES[entry["ticker"]],
                    "form": "10-K",
                    "fiscal_year": entry["fiscal_year"],
                    "doc_label": entry["doc_label"],
                    "accession": entry["accession"],
                    "source_file": entry["file"],
                },
            )
        )
    return docs


def load_live_documents(tickers: list[str]) -> list[Document]:
    """Fetch the latest 10-K for each ticker from EDGAR (cached)."""
    docs: list[Document] = []
    for ticker in tickers:
        text, meta = fetch_filing_text(ticker)
        fy = int(meta["report_date"][:4])
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "ticker": ticker,
                    "company": COMPANY_NAMES[ticker],
                    "form": "10-K",
                    "fiscal_year": fy,
                    "doc_label": f"{ticker} 10-K FY{fy}",
                    "accession": meta["accession"],
                    "source_file": f"edgar:{meta['accession']}",
                },
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Figures table (SEC company facts, AAPL)
# ---------------------------------------------------------------------------

_REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]
_NET_INCOME_CONCEPT = "NetIncomeLoss"


def _annual_figures(facts: dict, concept: str) -> dict[int, dict]:
    """Annual 10-K figures for one us-gaap concept, keyed by fiscal year.

    Rows are grouped by *period end*, never by the payload's ``fy``
    field: a 10-K reports three comparative years, and ``fy`` names the
    filing's year, not the period's — grouping by it would label FY2022
    revenue as "fiscal 2024". Within one period, the latest-filed
    reporting wins (a restated comparative supersedes the original).
    Apple's fiscal year ends in late September, so the fiscal-year label
    is the calendar year of the period end.
    """
    from datetime import date

    rows = facts.get("us-gaap", {}).get(concept, {}).get("units", {}).get("USD", [])
    by_period: dict[str, dict] = {}
    for row in rows:
        if row.get("form") != "10-K" or row.get("fp") != "FY":
            continue
        start, end = row.get("start"), row.get("end")
        if not start or not end:
            continue
        span = (date.fromisoformat(end) - date.fromisoformat(start)).days
        if not (350 <= span <= 380):
            continue
        current = by_period.get(end)
        if current is None or row.get("filed", "") > current.get("filed", ""):
            by_period[end] = row
    return {date.fromisoformat(end).year: row for end, row in by_period.items()}


def figures_from_facts(facts: dict, ticker: str, source: str) -> list[dict]:
    """Revenue + net income per fiscal year from a company-facts payload.

    The fiscal-year label is the calendar year of the period end (see
    :func:`_annual_figures` for why ``fy`` can't be trusted) — right
    for September (AAPL) and December (TSLA) year-ends alike.
    """
    revenue: dict[int, dict] = {}
    for concept in _REVENUE_CONCEPTS:
        for fy, row in _annual_figures(facts, concept).items():
            if fy not in revenue:  # earlier concepts in the list win
                revenue[fy] = {**row, "concept": concept}
    net_income = _annual_figures(facts, _NET_INCOME_CONCEPT)

    figures: list[dict] = []
    for fy in sorted(set(revenue) | set(net_income)):
        if fy in revenue:
            r = revenue[fy]
            figures.append({
                "ticker": ticker,
                "concept": "Total revenue",
                "taxonomy_concept": r["concept"],
                "fiscal_year": fy,
                "period_end": r["end"],
                "value_usd": int(r["val"]),
                "accession": r["accn"],
                "source": source,
            })
        if fy in net_income:
            n = net_income[fy]
            figures.append({
                "ticker": ticker,
                "concept": "Net income",
                "taxonomy_concept": _NET_INCOME_CONCEPT,
                "fiscal_year": fy,
                "period_end": n["end"],
                "value_usd": int(n["val"]),
                "accession": n["accn"],
                "source": source,
            })
    return figures


def load_figures(path: Path | None = None, ticker: str = "AAPL") -> list[dict]:
    """Structured figures for a bundled company (AAPL by default)."""
    path = path or (DATA_DIR / "companyfacts_aapl.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return figures_from_facts(payload.get("facts", {}), ticker,
                              f"SEC company facts ({path.name})")


def figures_as_text(figures: list[dict]) -> str:
    """Render the figures table as citable text lines."""
    lines = []
    for f in figures:
        lines.append(
            f"{f['ticker']} {f['concept']} for fiscal year {f['fiscal_year']} "
            f"(period ended {f['period_end']}): ${f['value_usd']:,} "
            f"[SEC company facts, accession {f['accession']}]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def corpus_stats() -> dict:
    docs = load_bundled_documents()
    figures = load_figures()
    per_doc = []
    total_chars = 0
    for d in docs:
        sections = find_item_sections(d.page_content)
        total_chars += len(d.page_content)
        per_doc.append({
            "doc": d.metadata["doc_label"],
            "chars": len(d.page_content),
            "sections": sorted(sections.keys()),
        })
    return {"documents": per_doc, "total_chars": total_chars, "figures": figures}


def main() -> None:
    parser = argparse.ArgumentParser(description="Market Buddy RAG ingestion")
    parser.add_argument("--live", nargs="*", choices=list(TICKER_CIK),
                        help="also fetch these tickers live from EDGAR")
    args = parser.parse_args()

    stats = corpus_stats()
    print("=== Bundled corpus ===")
    for d in stats["documents"]:
        print(f"  {d['doc']:<18} {d['chars']:>8,} chars   Items: {', '.join(d['sections'])}")
    print(f"  total: {stats['total_chars']:,} chars across {len(stats['documents'])} documents")
    print(f"\n=== Figures table (AAPL, SEC company facts) ===")
    for line in figures_as_text(stats["figures"]).split("\n"):
        print(" ", line)
    if args.live:
        print("\n=== Live EDGAR ===")
        for doc in load_live_documents(args.live):
            secs = find_item_sections(doc.page_content)
            print(f"  {doc.metadata['doc_label']:<18} {len(doc.page_content):>8,} chars   "
                  f"Items: {', '.join(sorted(secs.keys()))}")


if __name__ == "__main__":
    main()
