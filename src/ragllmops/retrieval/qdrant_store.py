"""Qdrant vector store backend (optional service).

The store keeps the full chunk payload alongside the vector so that a search
result can be reconstructed without a second lookup. Chunk ids are mapped to
deterministic UUIDs so that re-indexing is idempotent.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Sequence
from typing import Any

from ragllmops.logging_utils import get_logger
from ragllmops.models import DocumentChunk

logger = get_logger("retrieval.qdrant")

NAMESPACE = uuid.UUID("6f8d1c2e-0f4a-5b7c-9d3e-1a2b3c4d5e6f")


def _point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk id (Qdrant requires UUID or int ids)."""
    return str(uuid.uuid5(NAMESPACE, chunk_id))


class QdrantVectorStore:
    """Cosine-similarity vector store backed by Qdrant."""

    name = "qdrant"

    def __init__(self, url: str, collection: str, dim: int) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("qdrant-client is not installed") from exc

        self.url = url
        self.collection = collection
        self.dim = dim
        # The client probes the server version on construction; when the
        # server is down that probe raises a noisy warning that we handle
        # ourselves in `health()`.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Failed to obtain server version")
            self._client = QdrantClient(url=url)
        self._ensured = False

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------
    def ensure_collection(self) -> None:
        """Create the collection if it does not exist yet."""
        if self._ensured:
            return
        from qdrant_client import models  # type: ignore[import-not-found]

        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dim, distance=models.Distance.COSINE
                ),
            )
            logger.info("created Qdrant collection %s (dim=%s)", self.collection, self.dim)
        self._ensured = True

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    def upsert(self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]) -> int:
        """Insert or replace chunks."""
        from qdrant_client import models  # type: ignore[import-not-found]

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.ensure_collection()

        points = [
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=list(vector),
                payload={"chunk": chunk.to_dict()},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def reset(self) -> None:
        """Delete the collection entirely."""
        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)
        self._ensured = False

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------
    def search(
        self,
        vector: Sequence[float],
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return the ``top_k`` nearest chunks."""

        self.ensure_collection()
        query_filter = self._build_filter(filters)

        try:
            response = self._client.query_points(
                collection_name=self.collection,
                query=list(vector),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            hits = response.points
        except AttributeError:  # pragma: no cover - older qdrant-client
            hits = self._client.search(
                collection_name=self.collection,
                query_vector=list(vector),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

        results: list[tuple[DocumentChunk, float]] = []
        for hit in hits:
            payload: dict[str, Any] = (hit.payload or {}).get("chunk", {})
            chunk = DocumentChunk.from_dict(payload)
            results.append((chunk, float(hit.score)))
        return results

    def count(self) -> int:
        """Number of points in the collection."""
        self.ensure_collection()
        try:
            return int(self._client.count(collection_name=self.collection).count)
        except Exception:  # pragma: no cover - defensive
            return 0

    def health(self) -> bool:
        """Best-effort reachability probe used by ``cli doctor``."""
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    @staticmethod
    def _build_filter(filters: dict[str, str] | None):
        """Translate simple equality filters into a Qdrant filter."""
        if not filters:
            return None
        from qdrant_client import models  # type: ignore[import-not-found]

        conditions = [
            models.FieldCondition(key=f"chunk.{key}", match=models.MatchValue(value=value))
            for key, value in filters.items()
        ]
        return models.Filter(must=conditions)
