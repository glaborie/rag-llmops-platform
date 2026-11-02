"""The DAG execution engine.

Guarantees:

* **order** — an asset never runs before its dependencies;
* **idempotence** — a cached asset with an unchanged fingerprint is not re-run;
* **retries** — a failing asset is retried with exponential backoff;
* **containment** — a failed dependency skips its dependents instead of
  running them on stale inputs;
* **auditability** — every run is recorded with its inputs and outcome.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragllmops.logging_utils import get_logger
from ragllmops.orchestration.dag import DAG, Asset, AssetContext
from ragllmops.orchestration.lineage import AssetRun, LineageStore

logger = get_logger("orchestration.engine")

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_CACHED = "cached"


@dataclass(slots=True)
class RunSummary:
    """The outcome of one DAG run."""

    run_id: str
    dag: str
    started_at: str
    finished_at: str = ""
    duration_ms: float = 0.0
    runs: list[AssetRun] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dag": self.dag,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "runs": [run.to_dict() for run in self.runs],
        }

    def summary(self) -> str:
        lines = [f"run {self.run_id} ({self.dag})"]
        for run in self.runs:
            marker = {
                STATUS_SUCCESS: "ok",
                STATUS_CACHED: "cached",
                STATUS_FAILED: "FAIL",
                STATUS_SKIPPED: "skip",
            }.get(run.status, run.status)
            lines.append(
                f"  [{marker:>6}] {run.asset:<22} {run.duration_ms:>8.1f}ms {run.summary}"
            )
        lines.append(f"  overall: {'SUCCESS' if self.success else 'FAILURE'}")
        return "\n".join(lines)


class Engine:
    """Executes a DAG."""

    def __init__(
        self,
        dag: DAG,
        platform: Any = None,
        cache_dir: Path | None = None,
        lineage: LineageStore | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.dag = dag
        self.platform = platform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lineage = lineage
        self.params = params or {}

    # ------------------------------------------------------------------
    def fingerprint(self, asset: Asset, upstream: dict[str, str]) -> str:
        """Compute the cache key of an asset given its upstream fingerprints."""
        payload = {
            "asset": asset.name,
            "code": asset.code_fingerprint(),
            "deps": {dep: upstream.get(dep, "") for dep in asset.deps},
            "params": self.params.get(asset.name, {}),
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    def run(
        self,
        only: list[str] | None = None,
        force: bool = False,
        skip_cached: bool = False,
    ) -> RunSummary:
        """Run the DAG."""
        started = time.perf_counter()
        run_id = uuid.uuid4().hex[:12]
        summary = RunSummary(
            run_id=run_id,
            dag=self.dag.name,
            started_at=datetime.now(UTC).isoformat(),
        )

        context = AssetContext(platform=self.platform, run_id=run_id, params=self.params)
        fingerprints: dict[str, str] = {}
        failed: set[str] = set()

        selected = set(only) if only else None
        for asset in self.dag.topological_order():
            if selected is not None and asset.name not in selected:
                if asset.name not in self.dag.upstream_of_all(selected):
                    continue

            blocked = [dep for dep in asset.deps if dep in failed]
            if blocked:
                failed.add(asset.name)
                run = self._record(
                    asset, run_id, STATUS_SKIPPED, 0.0, fingerprints.get(asset.name, ""),
                    summary=f"skipped: upstream {', '.join(blocked)} failed",
                )
                summary.runs.append(run)
                continue

            fingerprint = self.fingerprint(asset, fingerprints)
            fingerprints[asset.name] = fingerprint

            cached = None
            if asset.cache and not force and not skip_cached:
                cached = self._read_cache(asset.name, fingerprint)

            if cached is not None:
                context.results[asset.name] = cached
                run = self._record(
                    asset, run_id, STATUS_CACHED, 0.0, fingerprint, summary="cache hit"
                )
                summary.runs.append(run)
                continue

            run, result = self._run_asset(asset, context, run_id, fingerprint)
            summary.runs.append(run)
            if run.status == STATUS_SUCCESS:
                context.results[asset.name] = result
                self._write_cache(asset.name, fingerprint, result)
            else:
                failed.add(asset.name)

            for dep in asset.deps:
                if self.lineage is not None:
                    self.lineage.record_edge(run_id, dep, asset.name)

        summary.results = dict(context.results)
        summary.finished_at = datetime.now(UTC).isoformat()
        summary.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        summary.success = not failed
        return summary

    # ------------------------------------------------------------------
    def _run_asset(
        self, asset: Asset, context: AssetContext, run_id: str, fingerprint: str
    ) -> tuple[AssetRun, Any]:
        """Run one asset, with retries."""
        started = time.perf_counter()
        attempt = 0
        last_error = ""

        while attempt <= asset.retries:
            try:
                result = asset.fn(context)
                duration = round((time.perf_counter() - started) * 1000, 2)
                return (
                    self._record(
                        asset, run_id, STATUS_SUCCESS, duration, fingerprint,
                        summary=_summarise(result),
                    ),
                    result,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempt += 1
                if attempt > asset.retries:
                    break
                delay = asset.retry_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "asset %s failed (%s); retrying in %.1fs", asset.name, last_error, delay
                )
                time.sleep(delay)

        duration = round((time.perf_counter() - started) * 1000, 2)
        logger.error("asset %s failed: %s", asset.name, last_error)
        logger.debug("%s", traceback.format_exc())
        return (
            self._record(
                asset, run_id, STATUS_FAILED, duration, fingerprint,
                summary=last_error, error=last_error,
            ),
            None,
        )

    # ------------------------------------------------------------------
    def _record(
        self,
        asset: Asset,
        run_id: str,
        status: str,
        duration_ms: float,
        fingerprint: str,
        summary: str = "",
        error: str | None = None,
    ) -> AssetRun:
        """Create and persist an asset run record."""
        run = AssetRun(
            asset=asset.name,
            status=status,
            run_id=run_id,
            started_at=datetime.now(UTC).isoformat(),
            finished_at=datetime.now(UTC).isoformat(),
            duration_ms=duration_ms,
            fingerprint=fingerprint,
            error=error,
            summary=summary,
            tags=list(asset.tags),
        )
        if self.lineage is not None:
            self.lineage.record_run(run)
        return run

    # ------------------------------------------------------------------
    def _cache_path(self, asset_name: str, fingerprint: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / asset_name / f"{fingerprint}.json"

    def _read_cache(self, asset_name: str, fingerprint: str) -> Any | None:
        """Read a cached result, or ``None`` when there is no usable cache."""
        if self.cache_dir is None:
            return None
        path = self._cache_path(asset_name, fingerprint)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if payload.get("fingerprint") != fingerprint:
            return None
        return payload.get("result")

    def _write_cache(self, asset_name: str, fingerprint: str, result: Any) -> None:
        """Cache a result when it is JSON-serialisable."""
        if self.cache_dir is None:
            return
        try:
            payload = {"fingerprint": fingerprint, "result": result}
            blob = json.dumps(payload, default=str)
        except (TypeError, ValueError):
            return
        path = self._cache_path(asset_name, fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(blob, encoding="utf-8")


def _summarise(result: Any) -> str:
    """A short human-readable description of an asset result."""
    if result is None:
        return ""
    if isinstance(result, dict):
        for key in ("summary", "message", "status", "passed", "version"):
            if key in result:
                return f"{key}={result[key]}"
        return f"{len(result)} keys"
    if isinstance(result, (list, tuple, set)):
        return f"{len(result)} items"
    if isinstance(result, str):
        return result[:80]
    return str(result)[:80]
