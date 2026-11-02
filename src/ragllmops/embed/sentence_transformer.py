"""Sentence-Transformers embedding backend (optional dependency)."""

from __future__ import annotations

from collections.abc import Sequence

from ragllmops.artifacts import model_artifact_hash
from ragllmops.logging_utils import get_logger

logger = get_logger("embed.sentence_transformer")


class SentenceTransformerEmbedder:
    """Dense embeddings from a sentence-transformers checkpoint.

    Requires ``sentence-transformers`` (see ``requirements-ml.txt``). The
    import is deferred so the rest of the platform works without it.
    """

    def __init__(self, model_name: str, dim: int | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install -r requirements-ml.txt"
            ) from exc

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        detected = self._model.get_sentence_embedding_dimension()
        self.dim = int(dim or detected)
        if dim and detected != dim:
            logger.warning(
                "configured embedding_dim=%s but model %s produces %s; using the model value",
                dim,
                model_name,
                detected,
            )
        # Hash the weights themselves, not the name: the same repo name can
        # resolve to different files over time.
        self.artifact_hash = model_artifact_hash(model_name)
        if not self.artifact_hash:
            logger.warning(
                "could not locate the files of %s; bundles built on it cannot pin "
                "the exact weights",
                model_name,
            )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
