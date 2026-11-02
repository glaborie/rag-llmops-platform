"""Request log backends."""

from __future__ import annotations

from ragllmops.config import Settings, get_settings
from ragllmops.logging_utils import get_logger
from ragllmops.store.base import LogStore
from ragllmops.store.sqlite_log_store import SQLiteLogStore

logger = get_logger("store")


def build_log_store(settings: Settings | None = None) -> LogStore:
    """Instantiate the configured request log backend."""
    s = settings or get_settings()
    if s.log_store_backend == "sqlite":
        return SQLiteLogStore(s.sqlite_file)
    if s.log_store_backend == "postgres":
        try:
            from ragllmops.store.postgres_log_store import PostgresLogStore

            return PostgresLogStore(s.postgres_dsn)
        except ImportError as exc:
            logger.warning("psycopg unavailable (%s); using the SQLite request log", exc)
            return SQLiteLogStore(s.sqlite_file)
    raise ValueError(f"unknown log store backend: {s.log_store_backend!r}")


__all__ = ["LogStore", "SQLiteLogStore", "build_log_store"]
