"""Features for the learned ranker.

The ranker is trained on the signals the platform already computes — the
dense and sparse scores of the hybrid retriever, the term overlap of the
rule-based reranker, the per-document statistics of the feature store — plus
two cheap ones (position in the candidate list, chunk length).

Nothing here needs a new dependency, and every feature is computable at
serving time for the same query, which is what makes the learned ranker
usable in the canary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ragllmops.features.base import doc_id_from_chunk
from ragllmops.logging_utils import get_logger
from ragllmops.models import DocumentChunk
from ragllmops.retrieval.hybrid import reciprocal_rank_fusion
from ragllmops.retrieval.rerank import TermOverlapReranker

logger = get_logger("feedback.features")

#: Bumped whenever the meaning of a feature changes. The trained artifact
#: records it, so a model can never be applied to features it was not
#: trained on.
FEATURE_SCHEMA_VERSION = "ranker-features-v1"

#: The features, in the order the model expects them.
FEATURE_NAMES: tuple[str, ...] = (
    "bm25_score",
    "dense_score",
    "term_overlap",
    "doc_avg_relevance",
    "doc_retrieval_count",
    "candidate_position",
    "chunk_length",
)


class RankerFeatureExtractor:
    """Computes the feature vector of every candidate for a query.

    The extractor holds the same backends the retrieval service uses, so the
    features it produces at training time are the features the ranker will
    see at serving time.
    """

    def __init__(
        self,
        *,
        bm25=None,
        dense_store=None,
        embedder=None,
        feature_provider=None,
        candidates: int = 20,
        dense_weight: float = 0.35,
        sparse_weight: float = 0.65,
    ) -> None:
        self.bm25 = bm25
        self.dense_store = dense_store
        self.embedder = embedder
        self.feature_provider = feature_provider
        self.candidate_count = candidates
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self._reranker = TermOverlapReranker()

    # ------------------------------------------------------------------
    def candidates(self, query: str) -> list[DocumentChunk]:
        """The candidate list the ranker gets to reorder.

        It is the fused dense + sparse list — the same list the retrieval
        service hands to the reranker.
        """
        dense_hits: list[tuple[DocumentChunk, float]] = []
        if self.dense_store is not None and self.embedder is not None:
            try:
                dense_hits = self.dense_store.search(
                    self.embedder.embed_one(query), max(self.candidate_count, 50)
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("dense search failed (%s); using the sparse list only", exc)

        sparse_hits: list[tuple[DocumentChunk, float]] = []
        if self.bm25 is not None:
            sparse_hits = self.bm25.search(query, self.candidate_count)

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
            return []

        fused = reciprocal_rank_fusion(
            [
                {cid: float(rank) for cid, rank in dense_rank.items()},
                {cid: float(rank) for cid, rank in sparse_rank.items()},
            ],
            weights=[self.dense_weight, self.sparse_weight],
        )
        return [chunks[chunk_id] for chunk_id in fused if chunk_id in chunks]

    def extract(self, query: str, chunks: Sequence[DocumentChunk]) -> dict[str, dict[str, float]]:
        """Compute the features of every chunk in *chunks*."""
        if not chunks:
            return {}

        dense_scores: dict[str, float] = {}
        if self.dense_store is not None and self.embedder is not None:
            try:
                dense_scores = {
                    chunk.chunk_id: float(score)
                    for chunk, score in self.dense_store.search(
                        self.embedder.embed_one(query), max(self.candidate_count, 50)
                    )
                }
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("dense search failed (%s); dense_score stays 0", exc)

        overlaps = self._reranker.score(query, list(chunks))

        doc_features = self._doc_features(chunks)

        features: dict[str, dict[str, float]] = {}
        for position, (chunk, overlap) in enumerate(zip(chunks, overlaps, strict=True), start=1):
            # The feature store keys documents the way the request log does.
            doc = doc_features.get(doc_id_from_chunk(chunk.chunk_id), {})
            features[chunk.chunk_id] = {
                "bm25_score": float(
                    self.bm25.score(query, chunk.chunk_id) if self.bm25 is not None else 0.0
                ),
                "dense_score": float(dense_scores.get(chunk.chunk_id, 0.0)),
                "term_overlap": float(overlap),
                "doc_avg_relevance": float(doc.get("avg_relevance", 0.0) or 0.0),
                "doc_retrieval_count": float(doc.get("retrieval_count", 0.0) or 0.0),
                "candidate_position": float(position),
                "chunk_length": float(len(chunk.text)),
            }
        return features

    def feature_map(self, query: str) -> dict[str, dict[str, float]]:
        """The features of every candidate for *query*, keyed by chunk id."""
        return self.extract(query, self.candidates(query))

    def _doc_features(self, chunks: Sequence[DocumentChunk]) -> dict[str, Mapping[str, float]]:
        """The per-document statistics from the feature store."""
        if self.feature_provider is None:
            return {}
        doc_ids = {doc_id_from_chunk(chunk.chunk_id) for chunk in chunks}
        try:
            return self.feature_provider.get_online_features(doc_ids)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("feature lookup failed (%s); doc features stay 0", exc)
            return {}
