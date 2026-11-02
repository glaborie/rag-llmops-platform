"""Request log interface.

The request log is the platform's "production data": drift monitoring reads
it, evaluation can replay it, and future training runs can consume it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from ragllmops.models import RequestRecord


@runtime_checkable
class LogStore(Protocol):
    """Append-only store of served requests."""

    def log(self, record: RequestRecord) -> None:
        """Persist a request record."""
        ...

    def fetch(self, limit: int = 1000, since: datetime | None = None) -> list[RequestRecord]:
        """Fetch records, newest last."""
        ...

    def count(self) -> int:
        """Total number of stored records."""
        ...

    def get(self, request_id: str) -> RequestRecord | None:
        """Fetch a single record by id."""
        ...

    def clear(self) -> None:
        """Remove every record (used by tests and by ``cli reset``)."""
        ...

    def set_feedback(self, request_id: str, code: int, chunk_id: str | None = None) -> bool:
        """Attach a verdict to a stored request.

        Returns whether a record was updated. The judgement is written back
        to the request that produced it rather than into a separate table,
        so the log stays the single source of truth about production traffic.
        """
        ...
