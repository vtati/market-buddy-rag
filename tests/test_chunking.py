"""Chunking invariants — most importantly: section-aware chunks never
cross an Item boundary. Boundaries are recomputed here directly from
the source text (find_item_sections), independently of the chunker's
own bookkeeping, so this is a real check and not a tautology."""

import pytest

from src.chunking import chunk_corpus, chunk_document
from src.ingest import find_item_sections, load_bundled_documents


@pytest.fixture(scope="module")
def docs():
    return load_bundled_documents()


def test_section_chunks_never_cross_item_boundaries(docs):
    for doc in docs:
        sections = find_item_sections(doc.page_content)
        spans = sorted(sections.values())
        chunks = chunk_document(doc, "section")
        assert chunks, doc.metadata["doc_label"]
        for chunk in chunks:
            start, end = chunk.metadata["char_start"], chunk.metadata["char_end"]
            # The claimed span covers the chunk: every paragraph in the
            # chunk appears verbatim inside the source range (a packed
            # chunk re-joins paragraphs with "\n\n", so the range is a
            # covering span, not always a byte-exact slice).
            source_range = doc.page_content[start:end]
            for part in chunk.page_content.split("\n\n"):
                assert part in source_range, (
                    f"{doc.metadata['doc_label']} chunk "
                    f"{chunk.metadata['chunk_index']}: paragraph not inside "
                    f"its claimed span [{start}:{end}]"
                )
            # ...and that slice lies inside exactly one Item section.
            containing = [s for s in spans if s[0] <= start and end <= s[1]]
            assert len(containing) == 1, (
                f"{doc.metadata['doc_label']} chunk {chunk.metadata['chunk_index']} "
                f"[{start}:{end}] crosses an Item boundary"
            )
            # The section label agrees with the containing section.
            # find_item_sections keys spans by item number ("7") while
            # chunk metadata carries the display title ("Item 7") —
            # normalize both to the item key before comparing.
            def _key(title: str) -> str:
                return title.removeprefix("Item ").strip()
            titles = {_key(t) for t, span in sections.items() if span == containing[0]}
            assert _key(chunk.metadata["section"]) in titles


def test_chunk_metadata_is_complete(docs):
    for strategy in ("fixed", "section"):
        for chunk in chunk_corpus(docs, strategy):
            for key in ("ticker", "doc_label", "section", "char_start",
                        "char_end", "chunk_index", "strategy"):
                assert key in chunk.metadata
            assert chunk.metadata["strategy"] == strategy
            assert 0 <= chunk.metadata["char_start"] < chunk.metadata["char_end"]


def test_section_chunking_is_coarser_than_fixed(docs):
    fixed = chunk_corpus(docs, "fixed")
    section = chunk_corpus(docs, "section")
    assert 0 < len(section) < len(fixed)
