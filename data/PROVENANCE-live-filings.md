# Fixture provenance — live-fetched EDGAR excerpts

These three files were sliced by `scripts/build_corpus.py` from filings
downloaded live from SEC EDGAR on **2026-10-18** (America/New_York) with
the declared user agent `GenAlphaLabs MarketBuddyRAG demo
contact@example.com`, at fair-access pace. Nothing was hand-written.

| File | Filing | Accession | Report period | Items kept |
|---|---|---|---|---|
| `filing_aapl_10k_items1_1a.txt` | Apple Inc. Form 10-K | 0000320193-25-000079 | FY ended 2025-09-27 | 1, 1A |
| `filing_msft_10k_excerpt.txt` | Microsoft Corp. Form 10-K | 0001193125-26-323660 | FY ended 2026-06-30 | 1, 1A, 7, 8 |
| `filing_nvda_10k_excerpt.txt` | NVIDIA Corp. Form 10-K | 0001045810-26-000021 | FY ended 2026-01-25 | 1, 1A, 7, 8 |

Slicing: whole Items (heading line to next Item heading), each Item
truncated to a character cap (Item 1: 70k, 1A: 90k, 7: 130k, 8: 70k) at a
line boundary so the repository stays lean. The AAPL Items 7/7A/8
coverage comes from the scaffold fixture `filing_aapl_10k_2024_excerpt.htm`
(FY2024 10-K, accession 0000320193-24-000123 — see
`PROVENANCE-aapl-fixtures.md`), and the AAPL figures table comes from
`companyfacts_aapl.json` (SEC company-facts payload).

Known quirk: in NVIDIA's primary document, Item 8 is a one-paragraph
pointer ("set forth in our Consolidated Financial Statements … included
in this Annual Report") and the statements themselves appear after
Item 15 in the HTML ordering, so the NVDA Item 8 slice is that pointer
paragraph only. NVIDIA's financial discussion is carried by its Item 7
(MD&A) slice.
