"""Feast-backed online feature store (optional).

Serving reads the same feature view that the offline job materialised, which
is what makes the offline/online parity check meaningful: the values coming
back from Redis must equal the values computed from the request log.
"""

from __future__ import annotations

from ragllmops.logging_utils import get_logger

logger = get_logger("features.feast")


class FeastFeatureProvider:
    """Reads online features from a Feast feature store."""

    def __init__(self, repo_path: str) -> None:
        try:
            from feast import FeatureStore  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "feast is not installed. Run: pip install -r requirements-ml.txt "
                "(or set RAG_FEATURE_BACKEND=local)"
            ) from exc
        self.repo_path = repo_path
        self._store = FeatureStore(repo_path=repo_path)

    def get_online_features(self, entity_keys: set[str]) -> dict[str, dict[str, float]]:
        """Fetch the ``doc_stats`` feature view for the given document ids."""
        if not entity_keys:
            return {}

        rows = self._store.get_online_features(
            features=["doc_stats"], entity_rows=[{"doc_id": key} for key in sorted(entity_keys)]
        ).to_dict()

        result: dict[str, dict[str, float]] = {}
        for doc_id, row in rows.items():
            values = row.get("doc_stats")
            if not isinstance(values, dict):
                continue
            result[doc_id] = {
                name: float(value)
                for name, value in values.items()
                if isinstance(value, (int, float))
            }
        return result

    def all_features(self) -> dict[str, dict[str, float]]:
        """Not supported by Feast — used by the parity report instead."""
        return {}

    def count(self) -> int:
        return 0
