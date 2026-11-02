"""Postgres-backed request log (optional backend).

Used when you want the request log to outlive the container, or to share it
between the API and the monitoring jobs.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime

from ragllmops.logging_utils import get_logger
from ragllmops.models import RequestRecord

logger = get_logger("store.postgres")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id        TEXT PRIMARY KEY,
    ts                TIMESTAMPTZ NOT NULL,
    query             TEXT NOT NULL,
    answer            TEXT NOT NULL,
    bundle_version    TEXT NOT NULL,
    prompt_id         TEXT NOT NULL,
    llm_model         TEXT NOT NULL,
    llm_provider      TEXT NOT NULL,
    latency_ms        DOUBLE PRECISION NOT NULL,
    top_score         DOUBLE PRECISION NOT NULL,
    chunk_count       INTEGER NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    answer_chars      INTEGER NOT NULL,
    refused           BOOLEAN NOT NULL,
    retrieved_ids     JSONB NOT NULL,
    user_id           TEXT,
    tenant            TEXT,
    feedback          INTEGER,
    feedback_chunk_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests (ts);
"""

#: Columns added after the first release. ``CREATE TABLE IF NOT EXISTS``
#: leaves an existing table alone, so they are applied separately.
#:
#: The check matters: ``ALTER TABLE`` takes an ACCESS EXCLUSIVE lock even
#: when the column is already there, and it would then wait behind the API's
#: own open transaction — the CLI would hang until the API restarted.
_MIGRATIONS = (
    (
        "feedback_chunk_id",
        "ALTER TABLE requests ADD COLUMN IF NOT EXISTS feedback_chunk_id TEXT",
    ),
)


class PostgresLogStore:
    """Request log stored in Postgres."""

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "psycopg is not installed. Run: pip install -r requirements-ml.txt"
            ) from exc

        self.dsn = dsn
        self._psycopg = psycopg
        self._lock = threading.Lock()
        # Autocommit: the store must not hold a transaction open between
        # statements. An idle transaction keeps a snapshot and a lock on the
        # table, and the next process that wants to alter it would wait
        # behind this one for as long as the API stays up.
        self._connection = psycopg.connect(dsn, autocommit=True)
        self._connection.execute(_SCHEMA)
        self._has_chunk_column = False
        self._migrate()
        self._connection.commit()

    def _migrate(self) -> None:
        """Add the columns an older table does not have yet."""
        for column, statement in _MIGRATIONS:
            if self._column_exists("requests", column):
                self._has_chunk_column = True
                continue
            try:
                # Do not wait forever for a lock another process is holding.
                self._connection.execute("SET lock_timeout = '10s'")
                self._connection.execute(statement)
                self._has_chunk_column = True
                logger.info("added the %s column to the requests table", column)
            except Exception as exc:  # pragma: no cover - depends on the server
                logger.warning(
                    "could not add the %s column to the requests table (%s); "
                    "the chunk that was at fault will not be recorded",
                    column,
                    exc,
                )
            finally:
                self._connection.execute("SET lock_timeout = DEFAULT")

    def _column_exists(self, table: str, column: str) -> bool:
        """Whether a column is already there — a read, not a lock."""
        with self._connection.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = %s AND column_name = %s",
                (table, column),
            )
            return cur.fetchone() is not None

    def log(self, record: RequestRecord) -> None:
        """Persist a single request record."""
        with self._lock:
            with self._connection.cursor() as cur:
                if self._has_chunk_column:
                    cur.execute(
                        """
                        INSERT INTO requests (
                            request_id, ts, query, answer, bundle_version, prompt_id,
                            llm_model, llm_provider, latency_ms, top_score, chunk_count,
                            prompt_tokens, completion_tokens, answer_chars, refused,
                            retrieved_ids, user_id, tenant, feedback, feedback_chunk_id
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s::jsonb, %s, %s, %s, %s
                        )
                        ON CONFLICT (request_id) DO UPDATE SET
                            ts = EXCLUDED.ts,
                            answer = EXCLUDED.answer,
                            latency_ms = EXCLUDED.latency_ms,
                            feedback = EXCLUDED.feedback,
                            feedback_chunk_id = EXCLUDED.feedback_chunk_id
                        """,
                        (
                            record.request_id,
                            record.ts,
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
                            record.refused,
                            json.dumps(record.retrieved_ids),
                            record.user_id,
                            record.tenant,
                            record.feedback,
                            record.feedback_chunk_id,
                        ),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO requests (
                            request_id, ts, query, answer, bundle_version, prompt_id,
                            llm_model, llm_provider, latency_ms, top_score, chunk_count,
                            prompt_tokens, completion_tokens, answer_chars, refused,
                            retrieved_ids, user_id, tenant, feedback
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s::jsonb, %s, %s, %s
                        )
                        ON CONFLICT (request_id) DO UPDATE SET
                            ts = EXCLUDED.ts,
                            answer = EXCLUDED.answer,
                            latency_ms = EXCLUDED.latency_ms,
                            feedback = EXCLUDED.feedback
                        """,
                        (
                            record.request_id,
                            record.ts,
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
                            record.refused,
                            json.dumps(record.retrieved_ids),
                            record.user_id,
                            record.tenant,
                            record.feedback,
                        ),
                    )
            self._connection.commit()

    def set_feedback(self, request_id: str, code: int, chunk_id: str | None = None) -> bool:
        """Attach a verdict to a stored request."""
        with self._lock:
            with self._connection.cursor() as cur:
                if self._has_chunk_column:
                    cur.execute(
                        """
                        UPDATE requests
                        SET feedback = %s, feedback_chunk_id = %s
                        WHERE request_id = %s
                        """,
                        (code, chunk_id, request_id),
                    )
                else:
                    cur.execute(
                        "UPDATE requests SET feedback = %s WHERE request_id = %s",
                        (code, request_id),
                    )
                updated = cur.rowcount > 0
            self._connection.commit()
            return updated

    def fetch(self, limit: int = 1000, since: datetime | None = None) -> list[RequestRecord]:
        """Fetch the most recent records, oldest first."""
        query = "SELECT * FROM requests"
        params: list[object] = []
        if since is not None:
            query += " WHERE ts >= %s"
            params.append(since)
        query += " ORDER BY ts DESC LIMIT %s"
        params.append(limit)

        with self._lock:
            with self._connection.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]

        records = []
        for row in reversed(rows):
            payload = dict(zip(columns, row, strict=True))
            payload["ts"] = payload["ts"].isoformat()
            if isinstance(payload.get("retrieved_ids"), str):
                payload["retrieved_ids"] = json.loads(payload["retrieved_ids"] or "[]")
            records.append(RequestRecord.from_dict(payload))
        return records

    def count(self) -> int:
        with self._lock:
            with self._connection.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM requests")
                row = cur.fetchone()
        return int(row[0]) if row else 0

    def get(self, request_id: str) -> RequestRecord | None:
        """Fetch a single record by id."""
        with self._lock:
            with self._connection.cursor() as cur:
                cur.execute("SELECT * FROM requests WHERE request_id = %s", (request_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                columns = [desc[0] for desc in cur.description]
        payload = dict(zip(columns, row, strict=True))
        if isinstance(payload.get("retrieved_ids"), str):
            payload["retrieved_ids"] = json.loads(payload["retrieved_ids"] or "[]")
        return RequestRecord.from_dict(payload)

    def clear(self) -> None:
        with self._lock:
            with self._connection.cursor() as cur:
                cur.execute("DELETE FROM requests")
            self._connection.commit()
