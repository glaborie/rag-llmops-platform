"""Local (file-backed) online feature store.

The default backend. ``materialize`` writes the same file that serving
reads, so the online store is always consistent with the last offline run.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragllmops.logging_utils import get_logger

logger = get_logger("features.local")


class LocalFeatureProvider:
    """Serves features from a JSON document on disk."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._features: dict[str, dict[str, float]] = {}
        self.refresh()

    def refresh(self) -> None:
        """Reload the feature file from disk."""
        if not self.path.exists():
            self._features = {}
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("could not read features at %s (%s)", self.path, exc)
            self._features = {}
            return
        self._features = {
            str(key): {str(k): float(v) for k, v in value.items()}
            for key, value in payload.get("features", {}).items()
        }

    def get_online_features(self, entity_keys: set[str]) -> dict[str, dict[str, float]]:
        """Return the features for the requested keys that exist."""
        return {key: self._features[key] for key in entity_keys if key in self._features}

    def all_features(self) -> dict[str, dict[str, float]]:
        """Every known entity's features."""
        return dict(self._features)

    def count(self) -> int:
        return len(self._features)
