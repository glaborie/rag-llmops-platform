"""SQLite-backed request log (default backend)."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from ragllmops.logging_utils import get_logger
from ragllmops.models import RequestRecord

logger = get_logger("store.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id        TEXT PRIMARY KEY,
    ts                TEXT NOT NULL,
    query             TEXT NOT NULL,
    answer            TEXT NOT NULL,
    bundle_version    TEXT NOT NULL,
    prompt_id         TEXT NOT NULL,
    llm_model         TEXT NOT NULL,
    llm_provider      TEXT NOT NULL,
    latency_ms        REAL NOT NULL,
    top_score         REAL NOT NULL,
    chunk_count       INTEGER NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    answer_chars      INTEGER NOT NULL,
    refused           INTEGER NOT NULL,
    retrieved_ids     TEXT NOT NULL,
    user_id           TEXT,
    tenant            TEXT,
    feedback          INTEGER,
    feedback_chunk_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests (ts);
"""

_COLUMNS = (
    "request_id, ts, query, answer, bundle_version, prompt_id, llm_model, llm_provider, "
    "latency_ms, top_score, chunk_count, prompt_tokens, completion_tokens, answer_chars, "
    "refused, retrieved_ids, user_id, tenant, feedback, feedback_chunk_id"
)


class SQLiteLogStore:
    """Durable request log stored in a local SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._has_chunk_column = self._migrate()
        self._connection.commit()

    def _migrate(self) -> bool:
        """Add columns that older databases on disk do not have yet.

        ``CREATE TABLE IF NOT EXISTS`` never touches an existing table, so a
        log written by an earlier version keeps its old shape.

        A log on read-only media cannot be migrated; the store then keeps
        working without the column instead of refusing to open.
        """
        existing = {
            row["name"] for row in self._connection.execute("PRAGMA table_info(requests)")
        }
        if "feedback_chunk_id" in existing:
            return True
        try:
            self._connection.execute("ALTER TABLE requests ADD COLUMN feedback_chunk_id TEXT")
        except sqlite3.Error as exc:
            logger.warning(
                "could not add the feedback_chunk_id column to %s (%s); "
                "the chunk that was at fault will not be recorded",
                self.path,
                exc,
            )
            return False
        return True

    def log(self, record: RequestRecord) -> None:
        """Persist a single request record."""
        with self._lock:
            self._connection.execute(
                f"INSERT OR REPLACE INTO requests ({_COLUMNS}) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.request_id,
                    record.ts.isoformat(),
                    record.query,
                    record.answer,
                    record.bundle_version,
                    record.prompt_id,
                    record.llm_model,
                    record.llm_provider,
                    record.latency_ms,
                    record.top_score,
                    record.chunk_count,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.answer_chars,
                    int(record.refused),
                    json.dumps(record.retrieved_ids),
                    record.user_id,
                    record.tenant,
                    record.feedback,
                    record.feedback_chunk_id,
                ),
            )
            self._connection.commit()

    def set_feedback(self, request_id: str, code: int, chunk_id: str | None = None) -> bool:
        """Attach a verdict to a stored request."""
        with self._lock:
            if self._has_chunk_column:
                cursor = self._connection.execute(
                    "UPDATE requests SET feedback = ?, feedback_chunk_id = ? "
                    "WHERE request_id = ?",
                    (code, chunk_id, request_id),
                )
            else:
                cursor = self._connection.execute(
                    "UPDATE requests SET feedback = ? WHERE request_id = ?",
                    (code, request_id),
                )
            self._connection.commit()
            return cursor.rowcount > 0

    def fetch(self, limit: int = 1000, since: datetime | None = None) -> list[RequestRecord]:
        """Fetch the most recent records, oldest first."""
        query = "SELECT * FROM requests"
        params: list[object] = []
        if since is not None:
            query += " WHERE ts >= ?"
            params.append(since.isoformat())
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._connection.execute(query, params).fetchall()

        records = [self._to_record(row) for row in reversed(rows)]
        return records

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) AS n FROM requests").fetchone()
        return int(row["n"]) if row else 0

    def get(self, request_id: str) -> RequestRecord | None:
        """Fetch a single record by id."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM requests WHERE request_id = ?", (request_id,)
            ).fetchone()
        return self._to_record(row) if row else None

    def clear(self) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM requests")
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> RequestRecord:
        columns = row.keys()
        return RequestRecord(
            request_id=row["request_id"],
            ts=datetime.fromisoformat(row["ts"]),
            query=row["query"],
            answer=row["answer"],
            bundle_version=row["bundle_version"],
            prompt_id=row["prompt_id"],
            llm_model=row["llm_model"],
            llm_provider=row["llm_provider"],
            latency_ms=float(row["latency_ms"]),
            top_score=float(row["top_score"]),
            chunk_count=int(row["chunk_count"]),
            prompt_tokens=int(row["prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            answer_chars=int(row["answer_chars"]),
            refused=bool(row["refused"]),
            retrieved_ids=json.loads(row["retrieved_ids"] or "[]"),
            user_id=row["user_id"],
            tenant=row["tenant"],
            feedback=row["feedback"],
            feedback_chunk_id=row["feedback_chunk_id"] if "feedback_chunk_id" in columns else None,
        )
