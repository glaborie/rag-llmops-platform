"""The learned ranker, as a reranker.

The platform's reranker interface is small — ``score(query, chunks)`` — so a
trained model can take the place of the rule-based reranker without the
serving path knowing anything about training.

This is what makes the canary work: the champion bundle keeps the rule-based
reranker, the challenger bundle carries a learned one, and the traffic split
compares them on the same queries.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragllmops.feedback.features import FEATURE_SCHEMA_VERSION, RankerFeatureExtractor
from ragllmops.feedback.model import RankerModel
from ragllmops.logging_utils import get_logger
from ragllmops.models import DocumentChunk

logger = get_logger("feedback.ranker")


class LearnedRanker:
    """Scores candidates with a trained model instead of a rule."""

    def __init__(
        self,
        model: RankerModel,
        extractor: RankerFeatureExtractor,
        *,
        artifact_id: str = "",
    ) -> None:
        if model.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"ranker was trained on feature schema "
                f"{model.feature_schema_version!r} but this platform serves "
                f"{FEATURE_SCHEMA_VERSION!r}"
            )
        self.model = model
        self.extractor = extractor
        self.artifact_id = artifact_id
        self.name = f"learned:{artifact_id or 'ranker'}"

    def score(self, query: str, chunks: Sequence[DocumentChunk]) -> list[float]:
        """Score every chunk; higher is more relevant."""
        if not chunks:
            return []
        features = self.extractor.extract(query, list(chunks))
        return [
            self.model.score(features.get(chunk.chunk_id, {})) for chunk in chunks
        ]
