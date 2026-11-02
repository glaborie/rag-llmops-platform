"""A minimal scheduler.

Assets can declare a schedule (``"@every 15m"``). The scheduler ticks, works
out what is due, and runs it. It is intentionally not a replacement for
Airflow — it is the smallest thing that makes "the platform does this on a
schedule" true on a laptop.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ragllmops.logging_utils import get_logger
from ragllmops.orchestration.dag import DAG
from ragllmops.orchestration.engine import Engine, RunSummary

logger = get_logger("orchestration.scheduler")

_EVERY_RE = re.compile(r"^@every\s+(\d+)\s*([smhd])$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_schedule(spec: str | None) -> int | None:
    """Turn a schedule spec into an interval in seconds.

    Supported: ``@every 30s``, ``@every 15m``, ``@every 2h``, ``@every 1d``.
    """
    if not spec:
        return None
    match = _EVERY_RE.match(spec.strip())
    if not match:
        logger.warning("unsupported schedule %r; asset will not be scheduled", spec)
        return None
    amount, unit = int(match.group(1)), match.group(2).lower()
    return amount * _UNIT_SECONDS[unit]


@dataclass(slots=True)
class Scheduler:
    """Runs due assets on a fixed tick."""

    dag: DAG
    engine: Engine
    platform: Any = None
    last_run: dict[str, float] = field(default_factory=dict)

    def due_assets(self, now: float | None = None) -> list[str]:
        """Assets whose schedule has elapsed."""
        now = now if now is not None else time.time()
        due: list[str] = []
        for asset in self.dag.assets.values():
            interval = parse_schedule(asset.schedule)
            if interval is None:
                continue
            last = self.last_run.get(asset.name)
            if last is None or (now - last) >= interval:
                due.append(asset.name)
        return due

    def tick(self) -> RunSummary | None:
        """Run whatever is due. Returns ``None`` when nothing was due."""
        due = self.due_assets()
        if not due:
            return None

        now = time.time()
        logger.info("scheduler running due assets: %s", ", ".join(due))
        summary = self.engine.run(only=due)
        for asset in due:
            self.last_run[asset] = now
        return summary

    def run_forever(self, interval_seconds: int = 30) -> None:  # pragma: no cover - blocking
        """Tick until interrupted."""
        logger.info("scheduler started (tick every %ss)", interval_seconds)
        try:
            while True:
                try:
                    self.tick()
                except Exception as exc:
                    logger.error("scheduler tick failed: %s", exc)
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("scheduler stopped at %s", datetime.now(UTC).isoformat())
