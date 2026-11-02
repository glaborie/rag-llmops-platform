"""A deterministic, grounded, local LLM provider.

This provider does not call a model. It selects the sentences from the
retrieved context that best answer the question and stitches them together
with citations.

Why it exists:

* the platform must boot, answer and self-evaluate with no API key and no
  network — this provider is what makes the golden-set gate meaningful in CI;
* it gives a deterministic baseline to compare real models against, so you can
  prove a model change actually helped.

It is intentionally simple (term overlap scoring), and it is honest about it:
it only ever returns sentences that appear in the retrieved context.
"""

from __future__ import annotations

import time

from ragllmops.llm.base import LLMResult
from ragllmops.text import sentence_split, term_set

REFUSAL_TEXT = "I don't have enough information in the retrieved context to answer this."


class ExtractiveLLM:
    """Answers strictly from the provided context snippets."""

    provider = "extractive"

    def __init__(
        self,
        model: str = "local-extractive-v1",
        max_sentences: int = 4,
        max_sentence_chars: int = 400,
    ) -> None:
        self.model = model
        self.max_sentences = max_sentences
        self.max_sentence_chars = max_sentence_chars

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        context: list[str] | None = None,
    ) -> LLMResult:
        started = time.perf_counter()
        snippets = list(context or [])
        if not snippets:
            return LLMResult(
                text=REFUSAL_TEXT,
                prompt_tokens=_estimate_tokens(user),
                completion_tokens=_estimate_tokens(REFUSAL_TEXT),
                latency_ms=(time.perf_counter() - started) * 1000,
                model=self.model,
                provider=self.provider,
            )

        question = _extract_question(user)
        query_terms = term_set(question)

        candidates: list[tuple[float, int, int, str]] = []
        for snippet_index, snippet in enumerate(snippets, start=1):
            for sentence_index, sentence in enumerate(sentence_split(snippet)):
                overlap = query_terms & term_set(sentence)
                if not overlap:
                    continue
                # Longer matched terms are more specific, so they count more.
                score = sum(len(term) for term in overlap) / max(len(query_terms), 1)
                candidates.append((score, snippet_index, sentence_index, sentence))

        if not candidates:
            return LLMResult(
                text=REFUSAL_TEXT,
                prompt_tokens=_estimate_tokens(user),
                completion_tokens=_estimate_tokens(REFUSAL_TEXT),
                latency_ms=(time.perf_counter() - started) * 1000,
                model=self.model,
                provider=self.provider,
            )

        # Keep the best sentences, then restore document order so the answer
        # reads top-to-bottom like the source material.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected = candidates[: self.max_sentences]
        selected.sort(key=lambda item: (item[1], item[2]))

        parts: list[str] = []
        for _, snippet_index, _, sentence in selected:
            trimmed = sentence
            if len(trimmed) > self.max_sentence_chars:
                trimmed = trimmed[: self.max_sentence_chars].rsplit(" ", 1)[0] + "..."
            parts.append(f"{trimmed} [S{snippet_index}]")

        answer = " ".join(parts)
        return LLMResult(
            text=answer,
            prompt_tokens=_estimate_tokens(user),
            completion_tokens=_estimate_tokens(answer),
            latency_ms=(time.perf_counter() - started) * 1000,
            model=self.model,
            provider=self.provider,
        )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token)."""
    return max(1, len(text) // 4)


def _extract_question(user_prompt: str) -> str:
    """Pull the question out of a rendered prompt.

    The user templates always contain a ``Question:`` line; we take
    everything up to the following blank line.
    """
    marker = "Question:"
    index = user_prompt.rfind(marker)
    if index == -1:
        return user_prompt
    tail = user_prompt[index + len(marker) :].strip()
    return tail.split("\n\n")[0].strip()
