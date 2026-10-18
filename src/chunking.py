"""Chunking lab — two strategies over the same filing documents.

* **fixed** — ``RecursiveCharacterTextSplitter`` at 1000 chars / 200
  overlap over the whole document, the standard baseline. Each chunk is
  labelled with the Item section its start offset falls in (so it can be
  scored against gold sections), but nothing stops a chunk from
  straddling a section boundary — that is the baseline's known weakness.
* **section** — section-aware, like the production Market Buddy chunker:
  the document is split on Item headings first (a chunk *never* crosses
  a section boundary), then paragraphs inside a section are packed
  greedily up to ~4000 chars; an oversized paragraph is split with the
  recursive splitter *within* the paragraph only.

Every chunk carries metadata: ``ticker``, ``doc_label``, ``section``,
``char_start``/``char_end`` (offsets into the source document text),
``chunk_index`` and ``strategy``.
"""

from __future__ import annotations

import argparse
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingest import find_item_sections, load_bundled_documents

FIXED_CHUNK_SIZE = 1000
FIXED_OVERLAP = 200
SECTION_MAX_CHARS = 4000
SECTION_OVERLAP = 200

STRATEGIES = ("fixed", "section")


def _base_meta(doc: Document) -> dict:
    return {
        "ticker": doc.metadata["ticker"],
        "doc_label": doc.metadata["doc_label"],
        "accession": doc.metadata.get("accession", ""),
    }


def _section_at(sections: dict[str, tuple[int, int]], pos: int) -> str:
    for item, (start, end) in sections.items():
        if start <= pos < end:
            return f"Item {item}"
    return "Front matter"


def chunk_fixed(doc: Document) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FIXED_CHUNK_SIZE,
        chunk_overlap=FIXED_OVERLAP,
        add_start_index=True,
    )
    sections = find_item_sections(doc.page_content)
    chunks = []
    for i, piece in enumerate(splitter.create_documents([doc.page_content])):
        start = int(piece.metadata.get("start_index", 0))
        meta = {
            **_base_meta(doc),
            "section": _section_at(sections, start),
            "char_start": start,
            "char_end": start + len(piece.page_content),
            "chunk_index": i,
            "strategy": "fixed",
        }
        chunks.append(Document(page_content=piece.page_content, metadata=meta))
    return chunks


_PARA_SPLIT = re.compile(r"\n\s*\n")


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of blank-line-separated blocks, trimmed."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _PARA_SPLIT.finditer(text):
        seg = text[pos:m.start()]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            spans.append((pos + lead, pos + len(seg.rstrip())))
        pos = m.end()
    seg = text[pos:]
    if seg.strip():
        lead = len(seg) - len(seg.lstrip())
        spans.append((pos + lead, pos + len(seg.rstrip())))
    return spans


def chunk_section(doc: Document) -> list[Document]:
    text = doc.page_content
    sections = find_item_sections(text)
    if not sections:
        sections = {"—": (0, len(text))}
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SECTION_MAX_CHARS,
        chunk_overlap=SECTION_OVERLAP,
        add_start_index=True,
    )
    chunks: list[Document] = []

    def emit(content: str, section: str, start: int) -> None:
        meta = {
            **_base_meta(doc),
            "section": section,
            "char_start": start,
            "char_end": start + len(content),
            "chunk_index": len(chunks),
            "strategy": "section",
        }
        chunks.append(Document(page_content=content, metadata=meta))

    for item, (sec_start, sec_end) in sorted(sections.items(), key=lambda kv: kv[1][0]):
        section_label = f"Item {item}" if item != "—" else "Full text"
        section_text = text[sec_start:sec_end]
        packed: list[tuple[int, str]] = []  # (abs start, text)
        packed_len = 0

        def flush() -> None:
            nonlocal packed, packed_len
            if packed:
                emit("\n\n".join(t for _, t in packed), section_label, packed[0][0])
            packed, packed_len = [], 0

        for p_start, p_end in _paragraph_spans(section_text):
            para = section_text[p_start:p_end]
            abs_start = sec_start + p_start
            if len(para) > SECTION_MAX_CHARS:
                flush()
                for piece in splitter.create_documents([para]):
                    rel = int(piece.metadata.get("start_index", 0))
                    emit(piece.page_content, section_label, abs_start + rel)
                continue
            added = len(para) + (2 if packed else 0)
            if packed and packed_len + added > SECTION_MAX_CHARS:
                flush()
            packed.append((abs_start, para))
            packed_len += len(para) + (2 if packed_len else 0)
        flush()
    return chunks


CHUNKERS = {"fixed": chunk_fixed, "section": chunk_section}


def chunk_document(doc: Document, strategy: str) -> list[Document]:
    if strategy not in CHUNKERS:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    return CHUNKERS[strategy](doc)


def chunk_corpus(docs: list[Document], strategy: str) -> list[Document]:
    chunks: list[Document] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, strategy))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunking lab stats")
    parser.parse_args()
    docs = load_bundled_documents()
    for strategy in STRATEGIES:
        print(f"=== strategy: {strategy} ===")
        total = 0
        for doc in docs:
            chunks = chunk_document(doc, strategy)
            total += len(chunks)
            sizes = [len(c.page_content) for c in chunks]
            sections = sorted({c.metadata["section"] for c in chunks})
            print(f"  {doc.metadata['doc_label']:<18} {len(chunks):>4} chunks  "
                  f"avg {sum(sizes) / max(len(sizes), 1):>6.0f} chars  sections: {', '.join(sections)}")
        print(f"  total: {total} chunks")


if __name__ == "__main__":
    main()
