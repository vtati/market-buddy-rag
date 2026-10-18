# Evaluation report — 15 questions through the full graph

Measured 2026-10-18T19:04:12.952378+00:00 with the production graph (section chunking, hybrid+rerank, extractive generation — no LLM key was configured for this run).

**Pass rate: 15/15 (100%)** · faithfulness among answered questions: 100% · mean end-to-end latency 3040 ms per question.

## Per-question results

| # | Type | Question | Outcome | Latency | Verdict |
|---|---|---|---|---|---|
| E1 | answerable | What was Apple's total net sales in fiscal 2024? | answered | 6579 ms | PASS |
| E2 | answerable | How much net income did Apple report for fiscal 2024? | answered | 2957 ms | PASS |
| E3 | answerable | What are Apple's principal risk factors around supply chain concentration and manufacturing outsourcing? | answered | 1908 ms | PASS |
| E4 | answerable | How does Apple describe its business and its main products and services? | answered | 3658 ms | PASS |
| E5 | answerable | What are Microsoft's main business segments and what does the Intelligent Cloud segment include? | answered | 2283 ms | PASS |
| E6 | answerable | What risk factors does Microsoft disclose about competition and cybersecurity threats? | answered | 2360 ms | PASS |
| E7 | answerable | How did Microsoft's revenue and operating income perform in the most recent fiscal year, according to management's discussion? | answered | 2599 ms | PASS |
| E8 | answerable | What does NVIDIA say about its compute and networking platforms and the markets it serves? | answered | 3223 ms | PASS |
| E9 | answerable | What concentration risks does NVIDIA disclose about its customers and supply chain? | answered | 3799 ms | PASS |
| E10 | comparison | How do Apple and Microsoft each describe their business in their 10-K filings? | answered | 3484 ms | PASS |
| E11 | comparison | Which company reported higher total revenue in its most recent fiscal year, Apple or NVIDIA, and what were the figures? | answered | 4389 ms | PASS |
| E12 | ambiguous | How did the company perform last year? | refused | 5862 ms | PASS |
| E13 | ambiguous | What are the risks? | answered | 2497 ms | PASS |
| E14 | unanswerable | What was Tesla's revenue in fiscal 2024? | refused | 2 ms | PASS |
| E15 | unanswerable | What will Apple's stock price be at the end of 2027? | refused | 2 ms | PASS |

## Failure analysis

No failures in this run. Known soft spots, from the per-question detail in `results/eval.json`:
* E11 passes its criteria (both companies cited, every number verified, Apple's $416.2B total stated) but the extractive composer does not surface NVIDIA's $215,938M total-revenue line — it sits in a table chunk ranked just below the merged evidence cut. The LLM generation mode composes the explicit head-to-head; the extractive mode quotes what retrieval ranked highest.
* Extractive answers quote filing sentences verbatim: fully grounded, less fluent than an LLM-composed answer. Ambiguous questions are handled by a dedicated clarify node in the graph (ask which company) or an honest refusal.
## Method notes

* Answerable questions pass only with a citation from the expected
  company's gold Item section (or the SEC figures table) AND every
  number in the answer re-verified against the retrieved chunks.
* Ambiguous questions pass if the system refuses or explicitly
  flags that no company was named; silently answering for one
  company is scored as a failure, by design.
* Unanswerable questions pass only on an honest refusal.

Raw output: `results/eval.json` (regenerate with `python -m src.evaluate`).
