"""Embedding backends."""

from __future__ import annotations

from ragllmops.config import Settings, get_settings
from ragllmops.embed.base import Embedder
from ragllmops.embed.hashing import HashingEmbedder


def build_embedder(settings: Settings | None = None) -> Embedder:
    """Instantiate the configured embedding backend.

    Falls back to the hashing embedder when the heavyweight backend is
    requested but not installed — a missing optional dependency should
    degrade the platform, not break it.
    """
    s = settings or get_settings()
    if s.embedder_backend == "hashing":
        return HashingEmbedder(dim=s.embedding_dim)
    if s.embedder_backend == "sentence-transformer":
        try:
            from ragllmops.embed.sentence_transformer import SentenceTransformerEmbedder

            return SentenceTransformerEmbedder(s.embedding_model, dim=s.embedding_dim)
        except ImportError:
            from ragllmops.logging_utils import get_logger

            get_logger("embed").warning(
                "sentence-transformers unavailable; falling back to the hashing embedder"
            )
            return HashingEmbedder(dim=s.embedding_dim)
    raise ValueError(f"unknown embedder backend: {s.embedder_backend!r}")


__all__ = ["Embedder", "HashingEmbedder", "build_embedder"]
