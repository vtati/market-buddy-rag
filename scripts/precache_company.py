"""Build data/precached/<TICKER>/ by running the REAL EDGAR pipeline
once: resolve -> latest 10-K -> clean + whole-Item slicing with the
bundled corpus caps -> company-facts figures table.

The output (sliced filing text + figures + meta) is committed, so
``POST /api/ingest {"ticker": "TSLA"}`` completes with zero network —
the demo's refusal -> ingest -> answer arc never depends on SEC
latency or availability. Re-run to refresh the snapshot:

    .venv/bin/python scripts/precache_company.py TSLA
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402

from src import edgar  # noqa: E402
from src.chunking import chunk_document  # noqa: E402
from src.ingest import (fetch_company_facts, fetch_filing_text_for,  # noqa: E402
                        figures_from_facts, latest_10k_meta_for, slice_items)
from src.registry import PRECACHE_DIR  # noqa: E402


def build(ticker: str) -> dict:
    ticker = ticker.upper()
    row = edgar.company_for_ticker(ticker)
    if row is None:
        raise SystemExit(f"unknown ticker {ticker}")
    meta10k = latest_10k_meta_for(row["cik"], ticker)
    full_text, meta10k = fetch_filing_text_for(row["cik"], ticker)
    excerpt, items = slice_items(full_text)
    if not items:
        raise SystemExit(f"no slicable Items found in {ticker}'s latest 10-K")
    payload = fetch_company_facts(row["cik"], ticker)
    figures = figures_from_facts(
        payload.get("facts", {}), ticker,
        f"SEC company facts (companyfacts_{ticker.lower()}.json)")

    fiscal_year = int(meta10k["report_date"][:4])
    doc = Document(
        page_content=excerpt,
        metadata={"ticker": ticker, "company": row["name"], "form": "10-K",
                  "fiscal_year": fiscal_year,
                  "doc_label": f"{ticker} 10-K FY{fiscal_year}",
                  "accession": meta10k["accession"],
                  "source_file": f"precached:{ticker}"},
    )
    meta = {
        "ticker": ticker, "name": row["name"], "cik": row["cik"],
        "source": "precached",
        "accession": meta10k["accession"], "fiscal_year": fiscal_year,
        "filed": meta10k["filed"], "report_date": meta10k["report_date"],
        "doc_label": f"{ticker} 10-K FY{fiscal_year}",
        "items": items, "chars": len(excerpt),
        "chunks": {s: len(chunk_document(doc, s)) for s in ("fixed", "section")},
        "figures_rows": len(figures),
    }
    base = PRECACHE_DIR / ticker
    base.mkdir(parents=True, exist_ok=True)
    header = (
        f"<!-- Precached excerpt of {ticker} Form 10-K, accession "
        f"{meta['accession']}, report period ended {meta['report_date']}, "
        f"filed {meta['filed']}. Fetched live from SEC EDGAR by "
        f"scripts/precache_company.py and sliced by Item with the bundled "
        f"corpus per-item caps. See PROVENANCE.md in this folder. -->\n"
    )
    (base / "filing.txt").write_text(header + excerpt, encoding="utf-8")
    (base / "figures.json").write_text(json.dumps(figures, indent=1), encoding="utf-8")
    (base / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


if __name__ == "__main__":
    result = build(sys.argv[1] if len(sys.argv) > 1 else "TSLA")
    print(json.dumps(result, indent=1))
