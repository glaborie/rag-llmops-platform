"""Logging helpers.

One format for the whole platform so that console output, the request log and
the alert channel stay readable in the same style.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"


def configure_logging(level: int | str = logging.INFO, log_file: Path | None = None) -> None:
    """Install the platform's root handler exactly once."""
    global _CONFIGURED
    root = logging.getLogger("ragllmops")
    if _CONFIGURED:
        root.setLevel(level)
        return

    root.setLevel(level)
    root.propagate = False

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``ragllmops`` root."""
    return logging.getLogger(f"ragllmops.{name}")
