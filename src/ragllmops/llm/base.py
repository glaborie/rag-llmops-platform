"""LLM provider interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class LLMResult:
    """The outcome of a single completion."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMProvider(Protocol):
    """Generates text from a system + user prompt.

    ``context`` is the numbered list of retrieved snippets. Providers that do
    their own grounding work (the local extractive one) use it directly;
    providers that simply forward a rendered prompt ignore it.
    """

    provider: str
    model: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        context: list[str] | None = None,
    ) -> LLMResult:
        """Generate a completion."""
        ...
