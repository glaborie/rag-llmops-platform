"""Feast feature view definitions for the RAG platform.

Only imported when the Feast backend is enabled — the rest of the platform
never touches this module.
"""

from __future__ import annotations

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

doc = Entity(name="doc", join_keys=["doc_id"])

doc_stats_source = FileSource(
    name="doc_stats_source",
    path="data/features/offline_features.parquet",
    timestamp_field="event_timestamp",
)

doc_stats = FeatureView(
    name="doc_stats",
    entities=[doc],
    schema=[
        Field(name="retrieval_count", dtype=Int64),
        Field(name="avg_relevance", dtype=Float32),
        Field(name="refusal_rate", dtype=Float32),
        Field(name="recency_hours", dtype=Float32),
    ],
    source=doc_stats_source,
)
