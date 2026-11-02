"""Alias watcher.

The serving layer never restarts to pick up a promotion: a background task
polls the registry, notices that ``champion`` moved, and refreshes the
router. This is the difference between a deployment and a promotion.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from ragllmops.logging_utils import get_logger
from ragllmops.monitor.alerts import SEVERITY_INFO, Notifier

logger = get_logger("serve.watcher")


class AliasWatcher:
    """Polls the registry for alias changes."""

    def __init__(
        self,
        platform,
        notifier: Notifier | None = None,
        interval_seconds: int = 15,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.platform = platform
        self.notifier = notifier
        self.interval_seconds = interval_seconds
        #: Called after an alias moved, so the serving layer can pick the new
        #: bundle up without a restart.
        self.on_change = on_change
        self._last_aliases: dict[str, str] = {}
        self._primed = False
        self._changes: list[dict] = []

    # ------------------------------------------------------------------
    def poll(self) -> bool:
        """Check the registry once. Returns True when something changed."""
        try:
            aliases = self.platform.registry.aliases()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("alias poll failed: %s", exc)
            return False

        if not self._primed:
            # First poll: record the current state without reporting a change.
            # The registry can legitimately be empty at this point, so this
            # cannot be told apart from "nothing changed yet" by looking at
            # the aliases alone.
            self._primed = True
            self._last_aliases = dict(aliases)
            return False

        changed = False
        for alias, version in aliases.items():
            previous = self._last_aliases.get(alias)
            if previous != version:
                changed = True
                event = {
                    "alias": alias,
                    "previous": previous,
                    "version": version,
                    "at": _now(),
                }
                self._changes.append(event)
                logger.info("alias %s moved %s -> %s", alias, previous, version)
                if self.notifier is not None:
                    self.notifier.send(
                        title=f"Alias {alias} moved",
                        message=f"{alias}: {previous} -> {version}",
                        severity=SEVERITY_INFO,
                    )

        if changed:
            self._last_aliases = dict(aliases)
            self._notify_change()
        return changed

    def _notify_change(self) -> None:
        """Let the serving layer know that the router has to be rebuilt."""
        if self.on_change is None:
            return
        try:
            self.on_change()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not refresh the router after an alias move: %s", exc)

    @property
    def changes(self) -> list[dict]:
        """Alias changes observed so far."""
        return list(self._changes)

    async def run(self) -> None:
        """Poll forever (used as a FastAPI background task)."""
        logger.info("alias watcher started (every %ss)", self.interval_seconds)
        try:
            while True:
                await asyncio.sleep(self.interval_seconds)
                try:
                    self.poll()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("alias watcher iteration failed: %s", exc)
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            logger.info("alias watcher stopped")
            raise


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
