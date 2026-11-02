"""Online/offline feature store."""

from __future__ import annotations

from pathlib import Path

from ragllmops.config import Settings, get_settings
from ragllmops.features.base import FeatureProvider, doc_id_from_chunk
from ragllmops.features.local_provider import LocalFeatureProvider
from ragllmops.features.materialize import compute_doc_features, materialize, write_features
from ragllmops.logging_utils import get_logger

logger = get_logger("features")


def build_feature_provider(settings: Settings | None = None) -> FeatureProvider:
    """Instantiate the configured online feature provider."""
    s = settings or get_settings()
    if s.feature_backend == "local":
        return LocalFeatureProvider(s.features_online_file)
    if s.feature_backend == "feast":
        try:
            from ragllmops.features.feast_provider import FeastFeatureProvider

            return FeastFeatureProvider(repo_path=str(s.path(s.feast_repo)))
        except ImportError as exc:
            logger.warning("Feast unavailable (%s); using the local feature store", exc)
            return LocalFeatureProvider(s.features_online_file)
    raise ValueError(f"unknown feature backend: {s.feature_backend!r}")


def offline_features_path(settings: Settings | None = None) -> Path:
    """Where the materialised (offline) features live."""
    s = settings or get_settings()
    return s.features_online_file.with_name("offline_features.json")


def parquet_features_path(settings: Settings | None = None) -> Path:
    """Where the Feast feature-view source parquet lives."""
    s = settings or get_settings()
    return s.features_online_file.with_name("offline_features.parquet")


__all__ = [
    "FeatureProvider",
    "LocalFeatureProvider",
    "build_feature_provider",
    "compute_doc_features",
    "doc_id_from_chunk",
    "materialize",
    "offline_features_path",
    "parquet_features_path",
    "write_features",
]
