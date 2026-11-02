"""Chunk catalogue — the in-process map of chunk id to chunk.

The catalogue is the platform's view of "what is in the index". It is written
to disk at ingest time so that a restarted API can serve without re-reading
the corpus.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ragllmops.models import DocumentChunk


class ChunkCatalog:
    """An ordered, persistable collection of chunks."""

    def __init__(self, chunks: Iterable[DocumentChunk] | None = None) -> None:
        self._chunks: dict[str, DocumentChunk] = {}
        self._order: list[str] = []
        for chunk in chunks or []:
            self.upsert(chunk)

    def upsert(self, chunk: DocumentChunk) -> None:
        """Add or replace a chunk, preserving first-seen ordering."""
        if chunk.chunk_id not in self._chunks:
            self._order.append(chunk.chunk_id)
        self._chunks[chunk.chunk_id] = chunk

    def get(self, chunk_id: str) -> DocumentChunk | None:
        return self._chunks.get(chunk_id)

    def all(self) -> list[DocumentChunk]:
        return [self._chunks[chunk_id] for chunk_id in self._order]

    def doc_ids(self) -> list[str]:
        seen: list[str] = []
        for chunk in self.all():
            if chunk.doc_id not in seen:
                seen.append(chunk.doc_id)
        return seen

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._chunks

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        """Write the catalogue to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": [chunk.to_dict() for chunk in self.all()]}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> ChunkCatalog:
        """Read a catalogue from JSON, returning an empty one if absent."""
        path = Path(path)
        if not path.exists():
            return cls()
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(DocumentChunk.from_dict(item) for item in payload.get("chunks", []))
