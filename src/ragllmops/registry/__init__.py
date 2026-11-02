"""Model registry backends."""

from __future__ import annotations

from ragllmops.config import Settings, get_settings
from ragllmops.logging_utils import get_logger
from ragllmops.registry.base import BundleRegistry
from ragllmops.registry.local_registry import LocalBundleRegistry

logger = get_logger("registry")


def build_registry(settings: Settings | None = None) -> BundleRegistry:
    """Instantiate the configured registry backend."""
    s = settings or get_settings()
    if s.registry_backend == "local":
        return LocalBundleRegistry(s.registry_file)
    if s.registry_backend == "mlflow":
        try:
            from ragllmops.registry.mlflow_registry import MlflowBundleRegistry

            return MlflowBundleRegistry(
                tracking_uri=s.mlflow_tracking_uri,
                registered_model_name=s.registry_name,
                artifact_root=s.mlflow_artifact_root,
            )
        except ImportError as exc:
            logger.warning("MLflow registry unavailable (%s); using the local registry", exc)
            return LocalBundleRegistry(s.registry_file)
    raise ValueError(f"unknown registry backend: {s.registry_backend!r}")


__all__ = ["BundleRegistry", "LocalBundleRegistry", "build_registry"]
