"""The retrieval service: dense + sparse candidates, fused, then reranked."""

from __future__ import annotations

from collections.abc import Sequence

from ragllmops.logging_utils import get_logger
from ragllmops.models import DocumentChunk, RetrievedChunk
from ragllmops.retrieval.bm25 import Bm25Index
from ragllmops.retrieval.dense import InMemoryVectorStore
from ragllmops.retrieval.hybrid import reciprocal_rank_fusion
from ragllmops.retrieval.rerank import Reranker

logger = get_logger("retrieval.service")


class RetrievalService:
    """Hybrid retrieval with reciprocal-rank fusion and optional reranking.

    The flow is:

    1. dense search proposes ``candidates`` chunks (semantic recall);
    2. BM25 proposes ``candidates`` chunks (exact-term recall);
    3. the two ranked lists are fused with RRF;
    4. a reranker re-scores the fused list against the query;
    5. the top ``top_k`` chunks are returned.
    """

    def __init__(
        self,
        embedder,
        dense_store: InMemoryVectorStore,
        bm25: Bm25Index,
        reranker: Reranker | None = None,
        top_k: int = 4,
        candidates: int = 20,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> None:
        self.embedder = embedder
        self.dense_store = dense_store
        self.bm25 = bm25
        self.reranker = reranker
        self.top_k = top_k
        self.candidates = candidates
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
        ranker=None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for *query*.

        ``ranker`` replaces the built-in reranker for this call. It is how a
        bundle that carries a learned ranker gets it applied: the champion
        and the challenger can be served side by side, each with its own
        re-scoring.
        """
        limit = top_k or self.top_k
        candidate_count = max(self.candidates, limit)

        dense_hits = self.dense_store.search(
            self.embedder.embed_one(query), candidate_count, filters=filters
        )
        sparse_hits = self.bm25.search(query, candidate_count)

        chunks: dict[str, DocumentChunk] = {}
        dense_rank: dict[str, int] = {}
        sparse_rank: dict[str, int] = {}

        for rank, (chunk, _score) in enumerate(dense_hits, start=1):
            chunks[chunk.chunk_id] = chunk
            dense_rank[chunk.chunk_id] = rank
        for rank, (chunk, _score) in enumerate(sparse_hits, start=1):
            chunks.setdefault(chunk.chunk_id, chunk)
            sparse_rank.setdefault(chunk.chunk_id, rank)

        if not chunks:
            logger.warning("no candidates retrieved for query=%r", query[:60])
            return []

        fused = reciprocal_rank_fusion(
            [
                {cid: float(rank) for cid, rank in dense_rank.items()},
                {cid: float(rank) for cid, rank in sparse_rank.items()},
            ],
            weights=[self.dense_weight, self.sparse_weight],
        )

        retrieved = [
            RetrievedChunk(
                chunk=chunks[chunk_id],
                score=round(score, 6),
                dense_rank=dense_rank.get(chunk_id),
                sparse_rank=sparse_rank.get(chunk_id),
            )
            for chunk_id, score in fused.items()
            if chunk_id in chunks
        ]

        reranker = ranker if ranker is not None else self.reranker
        if reranker is not None and retrieved:
            retrieved = self._apply_reranker(query, retrieved, reranker)

        return retrieved[:limit]

    # ------------------------------------------------------------------
    def _apply_reranker(
        self, query: str, retrieved: list[RetrievedChunk], reranker=None
    ) -> list[RetrievedChunk]:
        """Re-score and reorder the fused candidates."""
        active = reranker if reranker is not None else self.reranker
        chunks: Sequence[DocumentChunk] = [item.chunk for item in retrieved]
        try:
            scores = active.score(query, chunks)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("reranker failed (%s); keeping fusion order", exc)
            return retrieved

        reranked = [
            RetrievedChunk(
                chunk=item.chunk,
                score=item.score,
                dense_rank=item.dense_rank,
                sparse_rank=item.sparse_rank,
                rerank_score=round(float(score), 6),
            )
            for item, score in zip(retrieved, scores, strict=True)
        ]
        reranked.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
        return reranked

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, int | str | None]:
        """Introspection used by ``/health`` and ``cli doctor``."""
        return {
            "chunks_dense": self.dense_store.count(),
            "chunks_sparse": self.bm25.size,
            "reranker": getattr(self.reranker, "name", None),
            "embedder": getattr(self.embedder, "name", None),
        }
