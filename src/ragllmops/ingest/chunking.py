"""Document chunking.

Chunks follow the document's own structure: each ``##`` section becomes at
least one chunk, and oversized sections are split into overlapping windows.
Heading-aware chunking beats fixed-size splitting for documentation, because
a chunk that ends mid-sentence loses the context that makes it retrievable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragllmops.ingest.loader import Document
from ragllmops.logging_utils import get_logger
from ragllmops.models import DocumentChunk

logger = get_logger("ingest.chunking")

DEFAULT_TARGET_CHARS = 1200
DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP_CHARS = 150

_HEADING_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Section:
    """A heading and the body underneath it."""

    heading: str
    text: str


def split_sections(body: str) -> list[Section]:
    """Split a document body into ``##`` sections.

    Text before the first heading becomes the document preamble.
    """
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        preamble = body.strip()
        return [Section(heading="", text=preamble)] if preamble else []

    sections: list[Section] = []
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(heading="", text=preamble))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            sections.append(Section(heading=match.group(1).strip(), text=text))
    return sections


def _split_long(text: str, target: int, maximum: int, overlap: int) -> list[str]:
    """Split a long section into overlapping windows on paragraph bounds."""
    if len(text) <= maximum:
        return [text]

    windows: list[str] = []
    paragraphs = text.split("\n\n")
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        windows.append("\n\n".join(current))
        # Carry the tail of the window so the next one keeps the context.
        tail: list[str] = []
        tail_len = 0
        for paragraph in reversed(current):
            if tail_len + len(paragraph) > overlap:
                break
            tail.insert(0, paragraph)
            tail_len += len(paragraph)
        current = list(tail)
        current_len = tail_len

    for paragraph in paragraphs:
        if len(paragraph) > maximum:
            flush()
            # A single oversized paragraph is windowed rather than truncated,
            # so no text is lost.
            windows.extend(_window_text(paragraph, maximum, overlap))
            continue
        if current_len + len(paragraph) > target and current:
            flush()
        current.append(paragraph)
        current_len += len(paragraph) + 2

    if current:
        windows.append("\n\n".join(current))
    return windows


def _window_text(text: str, size: int, overlap: int) -> list[str]:
    """Cut *text* into overlapping windows of at most *size* characters."""
    if size <= 0:
        return [text]
    step = max(1, size - max(0, overlap))
    windows: list[str] = []
    start = 0
    while start < len(text):
        window = text[start : start + size]
        if window.strip():
            windows.append(window)
        if start + size >= len(text):
            break
        start += step
    return windows


def chunk_document(
    document: Document,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[DocumentChunk]:
    """Turn a document into retrievable chunks."""
    chunks: list[DocumentChunk] = []
    index = 0

    for section in split_sections(document.body):
        windows = _split_long(section.text, target_chars, max_chars, overlap_chars)
        for window in windows:
            text = f"## {section.heading}\n{window}" if section.heading else window
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.doc_id}::{index}",
                    doc_id=document.doc_id,
                    title=document.title,
                    category=document.category,
                    text=text.strip(),
                    chunk_index=index,
                    metadata={
                        "source_file": document.path.name,
                        "section": section.heading,
                        "doc_words": document.word_count,
                    },
                )
            )
            index += 1

    logger.debug("chunked %s into %s chunks", document.doc_id, len(chunks))
    return chunks


def chunk_documents(
    documents: list[Document],
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[DocumentChunk]:
    """Chunk every document, keeping document order."""
    chunks: list[DocumentChunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(document, target_chars=target_chars, max_chars=max_chars,
                           overlap_chars=overlap_chars)
        )
    logger.info("produced %s chunks from %s documents", len(chunks), len(documents))
    return chunks
