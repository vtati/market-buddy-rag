"""Item-section detection across filer heading styles.

Regression for the Netflix ingest failure (owner live test): NFLX's
iXBRL 10-K renders headings as ``Item 1.Business`` — no space after
the period — and its table of contents splits label and title across
two lines (``Item 1.`` / ``Business`` / page number). The slicer found
zero Items and "Add a Company" failed at the clean stage. The samples
below mirror that filing's real heading shapes; the disambiguation
must land on the BODY headings, not the contents cluster.
"""

from src.ingest import find_item_sections, slice_items

_BODY_1 = "\n".join([
    "Netflix, Inc. is one of the world's leading entertainment services "
    "with over 300 million paid memberships in over 190 countries.",
] * 12)
_BODY_1A = "\n".join([
    "If any of the following risks actually occur, our business, financial "
    "condition and results of operations could be materially adversely affected.",
] * 12)
_BODY_7 = "\n".join([
    "Revenues increased year over year driven by membership growth, price "
    "increases and advertising revenue across our service offerings.",
] * 12)
_BODY_8 = "\n".join([
    "The consolidated financial statements present fairly, in all material "
    "respects, the financial position of Netflix, Inc. and subsidiaries.",
] * 12)

NFLX_STYLE = f"""TABLE OF CONTENTS
Item 1.
Business
1
Item 1A.
Risk Factors
4
Item 7.
Management’s Discussion and Analysis of Financial Condition and Results of Operations
22
Item 8.
Financial Statements and Supplementary Data
30
Item 1.Business
{_BODY_1}
Item 1A.Risk Factors
{_BODY_1A}
Item 7.Management’s Discussion and Analysis of Financial Condition and Results of Operations
{_BODY_7}
Item 8.Financial Statements and Supplementary Data
{_BODY_8}
"""


def test_nflx_style_headings_are_found():
    spans = find_item_sections(NFLX_STYLE)
    assert {"1", "1A", "7", "8"} <= set(spans)


def test_disambiguation_prefers_body_headings_over_toc_cluster():
    spans = find_item_sections(NFLX_STYLE)
    body_start = NFLX_STYLE.index("Item 1.Business")
    assert spans["1"][0] == body_start
    sliced, items = slice_items(NFLX_STYLE)
    assert items == ["1", "1A", "7", "8"]
    # The slice carries body text, not the contents page.
    assert "paid memberships" in sliced
    assert "materially adversely affected" in sliced
    assert "TABLE OF CONTENTS" not in sliced


def test_canonical_spaced_headings_still_found():
    text = ("Item 1. Business\n" + _BODY_1 + "\n"
            "Item 1A. Risk Factors\n" + _BODY_1A + "\n"
            "Item 7. Management's Discussion and Analysis\n" + _BODY_7 + "\n"
            "Item 8. Financial Statements and Supplementary Data\n" + _BODY_8)
    spans = find_item_sections(text)
    assert {"1", "1A", "7", "8"} <= set(spans)
    sliced, items = slice_items(text)
    assert items == ["1", "1A", "7", "8"]
