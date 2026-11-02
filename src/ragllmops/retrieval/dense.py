"""In-memory dense vector store (numpy).

Used by the tests and as a zero-dependency fallback when Qdrant is not
running. The implementation mirrors the Qdrant store's contract exactly:
cosine similarity over L2-normalised vectors.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ragllmops.models import DocumentChunk


class InMemoryVectorStore:
    """A small cosine-similarity vector store backed by a numpy matrix."""

    name = "memory"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._chunks: dict[str, DocumentChunk] = {}
        self._vectors: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    def upsert(self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]) -> int:
        """Insert or replace chunks. Returns the number of chunks stored."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.dim:
                raise ValueError(
                    f"vector for {chunk.chunk_id!r} has dimension {len(vector)}, expected {self.dim}"
                )
            if chunk.chunk_id not in self._chunks:
                self._ids.append(chunk.chunk_id)
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = _normalise(np.asarray(vector, dtype="float32"))

        return len(self._ids)

    def reset(self) -> None:
        """Drop everything."""
        self._ids.clear()
        self._chunks.clear()
        self._vectors.clear()

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------
    def search(
        self,
        vector: Sequence[float],
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return the ``top_k`` chunks nearest to *vector*."""
        if not self._ids:
            return []

        query = _normalise(np.asarray(vector, dtype="float32"))
        candidates: list[tuple[DocumentChunk, float]] = []
        for chunk_id in self._ids:
            chunk = self._chunks[chunk_id]
            if filters and not _matches(chunk, filters):
                continue
            score = float(np.dot(self._vectors[chunk_id], query))
            candidates.append((chunk, score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:top_k]

    def get(self, chunk_id: str) -> DocumentChunk | None:
        """Fetch a single chunk by id."""
        return self._chunks.get(chunk_id)

    def count(self) -> int:
        """Number of chunks stored."""
        return len(self._ids)

    def all_chunks(self) -> list[DocumentChunk]:
        """Every chunk, in insertion order."""
        return [self._chunks[chunk_id] for chunk_id in self._ids]


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _matches(chunk: DocumentChunk, filters: dict[str, str]) -> bool:
    """Check a chunk against simple equality filters on its fields."""
    for key, value in filters.items():
        actual = getattr(chunk, key, None) if hasattr(chunk, key) else chunk.metadata.get(key)
        if str(actual) != str(value):
            return False
    return True
