"""Model registry interface.

The registry is the heart of the promotion workflow: it stores configuration
bundles, points aliases (``champion`` / ``challenger``) at specific versions,
and lets the serving layer hot-reload when an alias moves.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragllmops.models import RAGConfigBundle


@runtime_checkable
class BundleRegistry(Protocol):
    """Stores versioned bundles and resolves aliases to them."""

    def register(self, bundle: RAGConfigBundle) -> str:
        """Store a bundle and return its version."""
        ...

    def get(self, version: str) -> RAGConfigBundle | None:
        """Fetch a bundle by version."""
        ...

    def list_versions(self) -> list[str]:
        """All known versions, newest first."""
        ...

    def set_alias(self, alias: str, version: str) -> None:
        """Point an alias at a version."""
        ...

    def get_by_alias(self, alias: str) -> RAGConfigBundle | None:
        """Resolve an alias to a bundle."""
        ...

    def aliases(self) -> dict[str, str]:
        """Current alias -> version mapping."""
        ...

    def latest(self) -> RAGConfigBundle | None:
        """The most recently registered bundle."""
        ...
