"""Rerankers.

Reranking is the cheapest big win in a RAG system: the retriever proposes
candidates, a cross-encoder re-scores them against the actual query, and only
the best few survive into the prompt. The cross-encoder needs
``sentence-transformers``; a deterministic term-overlap reranker is the
dependency-free fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ragllmops.artifacts import builtin_artifact_hash, model_artifact_hash
from ragllmops.logging_utils import get_logger
from ragllmops.models import DocumentChunk
from ragllmops.text import term_set

logger = get_logger("retrieval.rerank")


@runtime_checkable
class Reranker(Protocol):
    """Scores ``(query, chunk)`` pairs; higher is more relevant."""

    name: str

    def score(self, query: str, chunks: Sequence[DocumentChunk]) -> list[float]:
        """Return one score per chunk, in the same order."""
        ...


class TermOverlapReranker:
    """Deterministic reranker based on weighted query-term coverage.

    Scores are normalised to ``[0, 1]`` so that they can be blended with
    fusion scores without dominating the scale.
    """

    name = "term-overlap-v1"

    def __init__(self) -> None:
        # No files back this reranker; the algorithm itself is the identity.
        self.artifact_hash = builtin_artifact_hash(self.name)

    def score(self, query: str, chunks: Sequence[DocumentChunk]) -> list[float]:
        query_terms = term_set(query)
        if not query_terms:
            return [0.0 for _ in chunks]

        scores: list[float] = []
        for chunk in chunks:
            chunk_terms = term_set(f"{chunk.title} {chunk.text}")
            if not chunk_terms:
                scores.append(0.0)
                continue
            matched = query_terms & chunk_terms
            weighted = sum(len(term) for term in matched)
            total = sum(len(term) for term in query_terms)
            # Coverage weighted by term specificity, damped by chunk length so
            # that long chunks do not win purely by containing more words.
            coverage = weighted / total if total else 0.0
            density = weighted / (len(chunk_terms) ** 0.5) if chunk_terms else 0.0
            scores.append(round(min(1.0, 0.7 * coverage + 0.3 * min(1.0, density)), 6))
        return scores


class CrossEncoderReranker:
    """Cross-encoder reranking with sentence-transformers (optional)."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install -r requirements-ml.txt"
            ) from exc
        self.model_name = model_name
        # The name carries the model so two cross-encoders are never confused
        # in the registry.
        self.name = f"cross-encoder:{model_name}"
        self._model = CrossEncoder(model_name)
        self.artifact_hash = model_artifact_hash(model_name)
        if not self.artifact_hash:
            logger.warning(
                "could not locate the files of %s; bundles built on it cannot pin "
                "the exact weights",
                model_name,
            )

    def score(self, query: str, chunks: Sequence[DocumentChunk]) -> list[float]:
        if not chunks:
            return []
        pairs = [(query, f"{chunk.title}. {chunk.text}") for chunk in chunks]
        raw = self._model.predict(pairs)
        values = [float(v) for v in raw]
        low, high = min(values), max(values)
        if high - low < 1e-9:
            return [0.5 for _ in values]
        return [(v - low) / (high - low) for v in values]


def build_reranker(model_name: str, enabled: bool = True) -> Reranker | None:
    """Build the best available reranker, degrading gracefully."""
    if not enabled:
        return None
    try:
        return CrossEncoderReranker(model_name)
    except ImportError:
        logger.info("cross-encoder unavailable; using the term-overlap reranker")
        return TermOverlapReranker()
