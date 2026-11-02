"""Canary routing between the champion and the challenger bundle.

Traffic splitting is deterministic on the request id: the same request always
lands on the same side. That matters for evaluation — a canary whose routing
changes per request cannot be compared fairly against the champion.
"""

from __future__ import annotations

import hashlib

from ragllmops.logging_utils import get_logger
from ragllmops.models import RAGConfigBundle

logger = get_logger("serve.canary")


class CanaryRouter:
    """Chooses the bundle that serves a given request."""

    def __init__(
        self,
        champion: RAGConfigBundle | None,
        challenger: RAGConfigBundle | None,
        traffic_pct: int = 0,
    ) -> None:
        self.champion = champion
        self.challenger = challenger
        self.traffic_pct = max(0, min(100, int(traffic_pct)))

    @property
    def active(self) -> bool:
        """True when a canary is actually serving traffic."""
        return bool(self.challenger) and self.traffic_pct > 0 and bool(self.champion)

    def bucket(self, request_id: str) -> int:
        """Map a request id to a bucket in ``[0, 100)``."""
        digest = hashlib.blake2b(request_id.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % 100

    def select(self, request_id: str) -> RAGConfigBundle | None:
        """Return the bundle that should serve this request."""
        if not self.champion:
            return self.challenger
        if not self.active:
            return self.champion
        if self.bucket(request_id) < self.traffic_pct:
            return self.challenger
        return self.champion

    def route_of(self, request_id: str) -> str:
        """Which side a request went to — used for metrics and logs."""
        selected = self.select(request_id)
        if selected is None:
            return "none"
        if self.challenger and selected.version == self.challenger.version:
            return "challenger"
        return "champion"
