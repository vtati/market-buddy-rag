"""Shared question sets.

``EXPERIMENT_QUESTIONS`` drives the chunking/retrieval comparison: each
question names the company and the Item section(s) a correct retrieval
must land in (the "gold" sections, verified against the corpus when the
set was authored). The Week 2 evaluation set in ``src/evaluate.py``
reuses the answerable questions from here and adds the comparison,
ambiguous, and unanswerable cases.
"""

EXPERIMENT_QUESTIONS = [
    {"id": "X1", "question": "What was Apple's total net sales in fiscal 2024?",
     "ticker": "AAPL", "gold_sections": ["Item 7", "Item 8"]},
    {"id": "X2", "question": "How much net income did Apple report for fiscal 2024?",
     "ticker": "AAPL", "gold_sections": ["Item 7", "Item 8"]},
    {"id": "X3", "question": "What are Apple's principal risk factors around supply chain concentration and manufacturing outsourcing?",
     "ticker": "AAPL", "gold_sections": ["Item 1A"]},
    {"id": "X4", "question": "How does Apple describe its business and its main products and services?",
     "ticker": "AAPL", "gold_sections": ["Item 1"]},
    {"id": "X5", "question": "How did Apple's net sales by region (Americas, Europe, Greater China) change in fiscal 2024?",
     "ticker": "AAPL", "gold_sections": ["Item 7"]},
    {"id": "X6", "question": "What are Microsoft's main business segments and what does the Intelligent Cloud segment include?",
     "ticker": "MSFT", "gold_sections": ["Item 1"]},
    {"id": "X7", "question": "What risk factors does Microsoft disclose about competition and cybersecurity threats?",
     "ticker": "MSFT", "gold_sections": ["Item 1A"]},
    {"id": "X8", "question": "How did Microsoft's revenue and operating income perform in the most recent fiscal year, according to management's discussion?",
     "ticker": "MSFT", "gold_sections": ["Item 7"]},
    {"id": "X9", "question": "What do Microsoft's consolidated financial statements show for total assets and liabilities?",
     "ticker": "MSFT", "gold_sections": ["Item 8"]},
    {"id": "X10", "question": "What does NVIDIA say about its compute and networking platforms and the markets it serves?",
     "ticker": "NVDA", "gold_sections": ["Item 1"]},
    {"id": "X11", "question": "What concentration risks does NVIDIA disclose about its customers and supply chain?",
     "ticker": "NVDA", "gold_sections": ["Item 1A"]},
    {"id": "X12", "question": "How did NVIDIA's revenue and gross margin change in the most recent fiscal year?",
     "ticker": "NVDA", "gold_sections": ["Item 7"]},
]
