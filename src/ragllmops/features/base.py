"""Online feature store interface.

For a RAG system the useful "features" are per-document statistics computed
offline from the request log: how often the document is retrieved, how
relevant it turns out to be, how often it leads to a refusal.

Serving reads them online to order otherwise-equivalent candidates. The
offline/online parity check is the point of the exercise: it proves that the
value used at serving time is the value that was computed offline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FeatureProvider(Protocol):
    """Reads online features for a set of entity keys."""

    def get_online_features(self, entity_keys: set[str]) -> dict[str, dict[str, float]]:
        """Return ``{entity_key: {feature_name: value}}``."""
        ...


def doc_id_from_chunk(chunk_id: str) -> str:
    """Derive a document id from a chunk id (``doc_id::index``)."""
    return chunk_id.split("::", 1)[0]
