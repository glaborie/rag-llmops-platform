"""Embedding backend interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a fixed-width vector."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        ...
