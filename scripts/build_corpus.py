#!/usr/bin/env python3
"""Build the bundled MSFT/NVDA/AAPL-Items-1/1A excerpts from live EDGAR.

Run once (network) to (re)generate the sliced fixtures in ``data/``;
the running app only reads the committed fixtures, so it works offline.
Each excerpt keeps whole Items (heading to next heading), truncated to a
per-item character cap at a line boundary so the repo stays lean.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import DATA_DIR, fetch_filing_text, find_item_sections, latest_10k_meta

CAPS = {"1": 70_000, "1A": 90_000, "7": 130_000, "8": 70_000}
JOBS = [
    # (ticker, wanted items, output filename)
    ("AAPL", ["1", "1A"], "filing_aapl_10k_items1_1a.txt"),
    ("MSFT", ["1", "1A", "7", "8"], "filing_msft_10k_excerpt.txt"),
    ("NVDA", ["1", "1A", "7", "8"], "filing_nvda_10k_excerpt.txt"),
]


def slice_excerpt(text: str, wanted: list[str]) -> str:
    spans = find_item_sections(text)
    parts = []
    for item in wanted:
        if item not in spans:
            print(f"    ! Item {item} not found")
            continue
        start, end = spans[item]
        body = text[start:end].strip()
        cap = CAPS.get(item, 80_000)
        if len(body) > cap:
            body = body[:cap].rsplit("\n", 1)[0]
        print(f"    Item {item}: {len(body):,} chars")
        parts.append(body)
    return "\n\n".join(parts)


def main() -> None:
    for ticker, wanted, fname in JOBS:
        meta = latest_10k_meta(ticker)
        print(f"{ticker}: accession {meta['accession']} report {meta['report_date']}")
        text, meta = fetch_filing_text(ticker)
        body = slice_excerpt(text, wanted)
        header = (
            f"<!-- EXCERPT of {ticker} Form 10-K, accession {meta['accession']}, "
            f"report period ended {meta['report_date']}, filed {meta['filed']}. "
            f"Source: {meta['url']} (SEC EDGAR, fetched live). Whole Items sliced "
            f"by heading with per-item char caps. See data/PROVENANCE-live-filings.md. -->\n"
        )
        out = DATA_DIR / fname
        out.write_text(header + body, encoding="utf-8")
        print(f"    wrote {out.name} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
