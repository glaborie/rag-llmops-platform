"""Index building: corpus -> chunks -> embeddings -> vector store."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ragllmops.ingest.chunking import chunk_documents
from ragllmops.ingest.loader import load_documents
from ragllmops.logging_utils import get_logger
from ragllmops.models import DocumentChunk

logger = get_logger("ingest.pipeline")

EMBED_BATCH_SIZE = 64


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Summary of a completed indexing run."""

    documents: int
    chunks: int
    avg_chunk_chars: float
    duration_ms: float
    collection: str

    def to_dict(self) -> dict[str, object]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "avg_chunk_chars": self.avg_chunk_chars,
            "duration_ms": self.duration_ms,
            "collection": self.collection,
        }


def embed_chunks(embedder, chunks: list[DocumentChunk]) -> list[list[float]]:
    """Embed chunks in batches."""
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        vectors.extend(embedder.embed([chunk.text for chunk in batch]))
    return vectors


def build_index(
    *,
    embedder,
    vector_store,
    corpus_dir: Path,
    reset: bool = True,
) -> tuple[list[DocumentChunk], IndexResult]:
    """Build the retrieval index from the corpus.

    Returns the chunks and a summary of the run.
    """
    started = time.perf_counter()

    documents = load_documents(corpus_dir)
    chunks = chunk_documents(documents)

    if reset:
        try:
            vector_store.reset()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not reset vector store (%s)", exc)

    vectors = embed_chunks(embedder, chunks)
    vector_store.upsert(chunks, vectors)

    avg_chars = sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0.0
    result = IndexResult(
        documents=len(documents),
        chunks=len(chunks),
        avg_chunk_chars=round(avg_chars, 1),
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        collection=getattr(vector_store, "collection", "memory"),
    )
    logger.info(
        "indexed %s documents -> %s chunks in %sms", result.documents, result.chunks,
        result.duration_ms,
    )
    return chunks, result
