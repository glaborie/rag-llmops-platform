"""Alerting.

Alerts are delivered through pluggable notifiers. The default is a console
notifier plus a file notifier, so the platform never depends on an external
service to tell you something broke; a Slack-compatible webhook can be added
with ``RAG_ALERT_WEBHOOK_URL``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import requests

from ragllmops.config import Settings, get_settings
from ragllmops.logging_utils import get_logger

logger = get_logger("monitor.alerts")

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"


@runtime_checkable
class Notifier(Protocol):
    """Delivers a message somewhere a human will see it."""

    def send(self, title: str, message: str, severity: str = SEVERITY_INFO) -> bool:
        """Deliver an alert. Returns ``True`` when it was delivered."""
        ...


@dataclass(slots=True)
class ConsoleNotifier:
    """Prints alerts to the log (always available)."""

    def send(self, title: str, message: str, severity: str = SEVERITY_INFO) -> bool:
        level = {
            SEVERITY_INFO: logger.info,
            SEVERITY_WARNING: logger.warning,
            SEVERITY_CRITICAL: logger.error,
        }.get(severity, logger.info)
        level("ALERT [%s] %s\n%s", severity.upper(), title, message)
        return True


@dataclass(slots=True)
class FileNotifier:
    """Appends alerts to a JSONL file."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, title: str, message: str, severity: str = SEVERITY_INFO) -> bool:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "severity": severity,
            "title": title,
            "message": message,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        return True


@dataclass(slots=True)
class WebhookNotifier:
    """Posts to a Slack-compatible incoming webhook."""

    url: str
    timeout: float = 10.0

    def send(self, title: str, message: str, severity: str = SEVERITY_INFO) -> bool:
        icon = {
            SEVERITY_INFO: ":information_source:",
            SEVERITY_WARNING: ":warning:",
            SEVERITY_CRITICAL: ":rotating_light:",
        }.get(severity, ":bell:")
        payload = {"text": f"{icon} *{title}*\n```{message}```"}
        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout)
            return response.status_code < 400
        except requests.RequestException as exc:
            logger.warning("webhook delivery failed: %s", exc)
            return False


@dataclass(slots=True)
class MultiNotifier:
    """Fans a message out to several notifiers."""

    notifiers: list[Notifier]

    def send(self, title: str, message: str, severity: str = SEVERITY_INFO) -> bool:
        delivered = False
        for notifier in self.notifiers:
            try:
                delivered = notifier.send(title, message, severity) or delivered
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("notifier %s failed: %s", type(notifier).__name__, exc)
        return delivered


def build_notifier(settings: Settings | None = None) -> Notifier:
    """Build the configured notifier."""
    s = settings or get_settings()
    notifiers: list[Notifier] = [ConsoleNotifier()]

    alert_file = s.data_path / "logs" / "alerts.jsonl"
    notifiers.append(FileNotifier(alert_file))

    if s.alert_webhook_url:
        notifiers.append(WebhookNotifier(s.alert_webhook_url))

    return MultiNotifier(notifiers)
