"""Offline feature materialisation.

Computes per-document statistics from the request log and writes them to

* ``data/features/offline_features.json`` — the computed truth (used by the
  parity check and by the Feast materialisation step);
* ``data/features/online_store.json`` — what the local online store serves;
* ``data/features/offline_features.parquet`` — the Feast feature-view source.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ragllmops.features.base import doc_id_from_chunk
from ragllmops.logging_utils import get_logger
from ragllmops.models import RequestRecord

logger = get_logger("features.materialize")


def compute_doc_features(
    records: list[RequestRecord],
    doc_ids: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate the request log into per-document features.

    Features per document:

    ``retrieval_count``  how often the document was retrieved
    ``avg_relevance``    mean top-score of requests that retrieved it
    ``refusal_rate``     share of those requests that ended in a refusal
    ``recency_hours``    hours since it was last retrieved (0 = just now)
    """
    universe = set(doc_ids or [])
    retrieved: dict[str, int] = defaultdict(int)
    relevance: dict[str, float] = defaultdict(float)
    refusals: dict[str, int] = defaultdict(int)
    last_seen: dict[str, datetime] = {}

    for record in records:
        seen_docs = {doc_id_from_chunk(chunk_id) for chunk_id in record.retrieved_ids}
        for doc_id in seen_docs:
            retrieved[doc_id] += 1
            relevance[doc_id] += record.top_score
            if record.refused:
                refusals[doc_id] += 1
            previous = last_seen.get(doc_id)
            if previous is None or record.ts > previous:
                last_seen[doc_id] = record.ts

    # Documents that were never retrieved still get a row, so the online store
    # can answer for them instead of silently missing.
    for doc_id in universe:
        retrieved.setdefault(doc_id, 0)

    newest = max((record.ts for record in records), default=None)

    features: dict[str, dict[str, float]] = {}
    for doc_id in sorted(set(universe) | set(retrieved)):
        count = retrieved.get(doc_id, 0)
        seen = last_seen.get(doc_id)
        recency_hours = 0.0
        if seen is not None and newest is not None:
            recency_hours = round((newest - seen).total_seconds() / 3600.0, 3)
        features[doc_id] = {
            "retrieval_count": float(count),
            "avg_relevance": round(relevance.get(doc_id, 0.0) / count, 6) if count else 0.0,
            "refusal_rate": round(refusals.get(doc_id, 0) / count, 6) if count else 0.0,
            "recency_hours": recency_hours,
        }
    return features


def write_features(
    features: dict[str, dict[str, float]],
    offline_path: Path,
    online_path: Path,
    parquet_path: Path | None = None,
) -> dict[str, Path]:
    """Write the materialised features to disk in all supported formats."""
    offline_path = Path(offline_path)
    online_path = Path(online_path)
    offline_path.parent.mkdir(parents=True, exist_ok=True)
    online_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"features": features}
    offline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    online_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    written = {"offline": offline_path, "online": online_path}

    if parquet_path is not None:
        parquet_path = Path(parquet_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pandas as pd

            frame = pd.DataFrame(
                [
                    {"doc_id": doc_id, **values, "event_timestamp": pd.Timestamp.now("UTC")}
                    for doc_id, values in features.items()
                ]
            )
            frame.to_parquet(parquet_path, index=False)
            written["parquet"] = parquet_path
        except Exception as exc:  # pragma: no cover - pandas optional
            logger.info(
                "parquet feature view skipped (%s); install pyarrow to enable the "
                "Feast materialisation step",
                type(exc).__name__,
            )

    return written


def materialize(
    log_store,
    doc_ids: list[str] | None = None,
    offline_path: Path | None = None,
    online_path: Path | None = None,
    parquet_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Read the request log, compute features and write them out."""
    records = log_store.fetch(limit=100000)
    features = compute_doc_features(records, doc_ids)
    if offline_path and online_path:
        write_features(features, offline_path, online_path, parquet_path)
    logger.info(
        "materialised features for %s documents from %s requests", len(features), len(records)
    )
    return features
