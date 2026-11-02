"""Corpus loading.

Documents are markdown files with a small YAML front-matter block. The loader
is intentionally strict: a document without a ``doc_id`` is an error, because
silent skips in a corpus are how "why did retrieval miss it?" bugs start.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ragllmops.logging_utils import get_logger

logger = get_logger("ingest.loader")

FRONT_MATTER_FENCE = "---"


@dataclass(frozen=True, slots=True)
class Document:
    """A source document from the corpus."""

    doc_id: str
    title: str
    category: str
    body: str
    path: Path

    @property
    def word_count(self) -> int:
        return len(self.body.split())


def parse_document(path: Path) -> Document:
    """Parse a single markdown document."""
    raw = path.read_text(encoding="utf-8")

    if not raw.startswith(FRONT_MATTER_FENCE):
        raise ValueError(f"{path} is missing a front-matter block")

    parts = raw.split(FRONT_MATTER_FENCE, 2)
    if len(parts) < 3:
        raise ValueError(f"{path} has a malformed front-matter block")

    meta = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()

    doc_id = meta.get("doc_id")
    if not doc_id:
        raise ValueError(f"{path} has no doc_id in its front matter")

    return Document(
        doc_id=str(doc_id),
        title=str(meta.get("title") or path.stem),
        category=str(meta.get("category") or "general"),
        body=body,
        path=path,
    )


def load_documents(corpus_dir: Path, pattern: str = "*.md") -> list[Document]:
    """Load every document in *corpus_dir*, sorted by filename."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")

    documents: list[Document] = []
    for path in sorted(corpus_dir.glob(pattern)):
        if not path.is_file():
            continue
        document = parse_document(path)
        documents.append(document)
        logger.debug("loaded document %s (%s words)", document.doc_id, document.word_count)

    if not documents:
        raise ValueError(f"no documents found in {corpus_dir}")
    logger.info("loaded %s documents from %s", len(documents), corpus_dir)
    return documents
