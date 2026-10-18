# Fixture provenance — REAL EDGAR data

All three fixtures are slices of payloads fetched live from SEC EDGAR
on **2026-10-18** (America/New_York) with the declared user agent
`MarketBuddy/0.1 (contact: dev@genalphalabs.com)`, at fair-access pace.
Nothing in these files was hand-written, approximated, or recalled.

| Fixture | Source URL | Slicing applied |
|---|---|---|
| `submissions_aapl.json` | `https://data.sec.gov/submissions/CIK0000320193.json` | Company metadata complete; `filings.recent` parallel arrays trimmed from 1000 rows to the first 300 (newest first — reaches back to the FY2023 10-K, accession `0000320193-23-000106`). |
| `companyfacts_aapl.json` | `https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json` | Only four `us-gaap` concepts retained — `RevenueFromContractWithCustomerExcludingAssessedTax`, `Revenues`, `SalesRevenueNet`, `NetIncomeLoss` — each with its COMPLETE units/data arrays, unmodified. |
| `filing_aapl_10k_2024_excerpt.htm` | `https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm` | Byte slice of the full filing HTML (1,503,887 bytes) covering the body of Item 7 (MD&A) and Item 8 (Financial Statements) through the consolidated statements; an HTML comment at the top of the file marks it as an excerpt. |

Company: Apple Inc., CIK 0000320193, ticker AAPL.

## Known-filing verification values (read from `companyfacts_aapl.json`)

Fiscal year 2024, as reported in Apple's FY2024 Form 10-K, accession
`0000320193-24-000123`, filed 2024-11-01, period 2023-10-01 → 2024-09-28:

- Revenue — `facts["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"]`,
  entry with `accn="0000320193-24-000123"`, `fy=2024`, `fp="FY"`, `form="10-K"`,
  `start="2023-10-01"`, `end="2024-09-28"`: **`val` = 391035000000** (USD).
- Net income — `facts["us-gaap"]["NetIncomeLoss"]["units"]["USD"]`, same
  accession/period keys: **`val` = 93736000000** (USD).

Cross-check against the filing document itself: the excerpt's
Consolidated Statements of Operations (Item 8) reports total net sales
of **391,035** and net income of **93,736** — in millions of USD —
which agree with the XBRL values above (391035000000 / 93736000000).
