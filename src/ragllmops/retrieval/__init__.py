"""Retrieval backends: vector stores, BM25, fusion, reranking."""

from __future__ import annotations

from ragllmops.config import Settings, get_settings
from ragllmops.logging_utils import get_logger
from ragllmops.retrieval.bm25 import Bm25Index
from ragllmops.retrieval.catalog import ChunkCatalog
from ragllmops.retrieval.dense import InMemoryVectorStore
from ragllmops.retrieval.hybrid import reciprocal_rank_fusion
from ragllmops.retrieval.rerank import (
    CrossEncoderReranker,
    Reranker,
    TermOverlapReranker,
    build_reranker,
)
from ragllmops.retrieval.service import RetrievalService

logger = get_logger("retrieval")


def build_vector_store(settings: Settings | None = None):
    """Build the configured vector store.

    If Qdrant is configured but unreachable, the platform falls back to the
    in-memory store with a warning — an unavailable optional service should
    degrade the system, not stop it.
    """
    s = settings or get_settings()
    if s.vector_store == "memory":
        return InMemoryVectorStore(dim=s.embedding_dim)
    if s.vector_store == "qdrant":
        try:
            from ragllmops.retrieval.qdrant_store import QdrantVectorStore

            store = QdrantVectorStore(
                url=s.qdrant_url, collection=s.qdrant_collection, dim=s.embedding_dim
            )
            if store.health():
                return store
            logger.warning(
                "Qdrant at %s did not answer; falling back to the in-memory store", s.qdrant_url
            )
        except Exception as exc:
            logger.warning(
                "Qdrant unavailable at %s (%s); falling back to the in-memory store",
                s.qdrant_url,
                exc,
            )
        return InMemoryVectorStore(dim=s.embedding_dim)
    raise ValueError(f"unknown vector store: {s.vector_store!r}")


__all__ = [
    "Bm25Index",
    "ChunkCatalog",
    "CrossEncoderReranker",
    "InMemoryVectorStore",
    "Reranker",
    "RetrievalService",
    "TermOverlapReranker",
    "build_reranker",
    "build_vector_store",
    "reciprocal_rank_fusion",
]
