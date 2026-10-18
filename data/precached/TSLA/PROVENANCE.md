# Precached session company — Tesla, Inc. (TSLA)

Built by running the real EDGAR pipeline once
(`scripts/precache_company.py TSLA`, 2026-10-18) so that
`POST /api/ingest {"ticker": "TSLA"}` completes with **zero network** —
the demo's refusal → ingest → answer arc never depends on SEC latency.

| | |
|---|---|
| Company | Tesla, Inc. — CIK 0001318605 |
| Filing | Form 10-K, fiscal year 2025 |
| Accession | 0001628280-26-003952 |
| Period ended | 2025-12-31 |
| Filed | 2026-01-29 |
| Source | SEC EDGAR submissions API + Archives (declared UA, fair-access pacing) |

Contents:

- `filing.txt` — the filing cleaned to text and sliced to whole Items
  1, 1A, 7, 8 with the same per-item char caps as the bundled corpus
  (70k / 90k / 130k / 70k): 254,492 chars → 438 fixed-size chunks,
  116 section-aware chunks.
- `figures.json` — 34 rows from Tesla's SEC company-facts payload
  (revenue + net income by fiscal year), built by the same period-end
  grouping code as the bundled AAPL figures table. E.g. FY2024 total
  revenue $97,690,000,000.
- `meta.json` — the registration record the ingest pipeline reports.

This snapshot is a **session-tier** artifact: it is not part of the
bundled corpus and not part of any measured result (comparison
experiment, 15-question evaluation). Rebuild with
`.venv/bin/python scripts/precache_company.py TSLA`.
