"""A file-backed bundle registry.

This is the default backend: it needs no service, it is human-readable, and it
makes the promotion workflow inspectable (``cat data/registry/registry.json``).
The MLflow backend implements the same contract for teams that want the
registry to live in a tracking server.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ragllmops.logging_utils import get_logger
from ragllmops.models import RAGConfigBundle

logger = get_logger("registry.local")


class LocalBundleRegistry:
    """Stores bundles as a single JSON document on disk."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._versions: dict[str, RAGConfigBundle] = {}
        self._aliases: dict[str, str] = {}
        self._order: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("could not read registry at %s (%s); starting empty", self.path, exc)
            return
        for version, bundle in payload.get("versions", {}).items():
            self._versions[version] = RAGConfigBundle.from_dict(bundle)
            self._order.append(version)
        self._aliases = dict(payload.get("aliases", {}))

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "versions": {v: self._versions[v].to_dict() for v in self._order},
            "aliases": self._aliases,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------------
    # registry contract
    # ------------------------------------------------------------------
    def register(self, bundle: RAGConfigBundle) -> str:
        """Store a bundle. Re-registering an existing version overwrites it."""
        with self._lock:
            if bundle.version not in self._versions:
                self._order.append(bundle.version)
            self._versions[bundle.version] = bundle
            self._flush()
        logger.info("registered bundle %s (prompt=%s)", bundle.version, bundle.prompt_id)
        return bundle.version

    def get(self, version: str) -> RAGConfigBundle | None:
        return self._versions.get(version)

    def list_versions(self) -> list[str]:
        return list(reversed(self._order))

    def set_alias(self, alias: str, version: str) -> None:
        if version not in self._versions:
            raise KeyError(f"cannot alias {alias!r} to unknown version {version!r}")
        with self._lock:
            self._aliases[alias] = version
            self._flush()
        logger.info("alias %s -> %s", alias, version)

    def get_by_alias(self, alias: str) -> RAGConfigBundle | None:
        version = self._aliases.get(alias)
        return self._versions.get(version) if version else None

    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def latest(self) -> RAGConfigBundle | None:
        versions = self.list_versions()
        return self._versions[versions[0]] if versions else None

    def metrics_for(self, version: str) -> dict[str, float]:
        """Recorded metrics for a version (used in promotion reports)."""
        bundle = self.get(version)
        return dict(bundle.metrics) if bundle else {}
