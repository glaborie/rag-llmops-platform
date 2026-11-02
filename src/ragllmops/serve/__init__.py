"""Serving layer."""

from ragllmops.serve.canary import CanaryRouter
from ragllmops.serve.watcher import AliasWatcher

__all__ = ["AliasWatcher", "CanaryRouter", "create_app"]


def create_app(platform=None):  # noqa: ANN001 - lazy import to keep deps light
    """Build the FastAPI app (imported lazily so the package stays light)."""
    from ragllmops.serve.app import create_app as _create_app

    return _create_app(platform)
