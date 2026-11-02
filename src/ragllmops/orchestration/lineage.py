"""Lineage and run history.

Every asset execution is recorded, together with the edges between assets in
a run. That is what lets the platform answer "which model version came from
which data?" and "what broke when this started failing?".
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AssetRun:
    """The record of one asset execution."""

    asset: str
    status: str
    run_id: str
    started_at: str
    finished_at: str = ""
    duration_ms: float = 0.0
    fingerprint: str = ""
    error: str | None = None
    summary: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LineageStore:
    """Append-only run and edge history, persisted as JSONL."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.runs_path = self.directory / "runs.jsonl"
        self.edges_path = self.directory / "edges.jsonl"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def record_run(self, run: AssetRun) -> None:
        """Append a run record."""
        with self._lock:
            with self.runs_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(run.to_dict(), default=str) + "\n")

    def record_edge(self, run_id: str, source: str, target: str) -> None:
        """Record that *target* consumed *source* within a run."""
        payload = {
            "run_id": run_id,
            "source": source,
            "target": target,
            "ts": datetime.now(UTC).isoformat(),
        }
        with self._lock:
            with self.edges_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

    # ------------------------------------------------------------------
    def runs_for(self, asset: str, limit: int = 50) -> list[dict[str, Any]]:
        """The most recent runs of an asset."""
        entries = self._read(self.runs_path)
        matching = [entry for entry in entries if entry.get("asset") == asset]
        return matching[-limit:]

    def last_run(self, asset: str) -> dict[str, Any] | None:
        """The most recent run of an asset."""
        runs = self.runs_for(asset, limit=1)
        return runs[-1] if runs else None

    def upstream_of(self, asset: str) -> set[str]:
        """Assets that fed into *asset* in the recorded history."""
        sources: set[str] = set()
        for edge in self._read(self.edges_path):
            if edge.get("target") == asset:
                sources.add(str(edge.get("source")))
        return sources

    def downstream_of(self, asset: str) -> set[str]:
        """Assets that consumed *asset* in the recorded history."""
        targets: set[str] = set()
        for edge in self._read(self.edges_path):
            if edge.get("source") == asset:
                targets.add(str(edge.get("target")))
        return targets

    def summary(self) -> dict[str, Any]:
        """An overview of the recorded history."""
        runs = self._read(self.runs_path)
        by_status: dict[str, int] = {}
        for run in runs:
            status = str(run.get("status", "unknown"))
            by_status[status] = by_status.get(status, 0) + 1
        return {
            "runs": len(runs),
            "edges": len(self._read(self.edges_path)),
            "by_status": by_status,
        }

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
