"""Versioned prompt templates.

Prompts are treated as *code artifacts*: every template carries an id and a
version, and the id is part of the registrable configuration bundle. That is
what makes "champion vs challenger" a meaningful comparison — you can promote a
prompt change without touching the model.

Two templates ship by default:

``v1``
    Plain grounded answering. Good enough to work, not enough to be safe.
``v2``
    Adds the citation contract, an explicit refusal rule and an answer-shape
    instruction. It is the default and is what the golden set expects.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_V1 = (
    "You are a support assistant for the Northwind Cloud Platform. "
    "Answer using only the provided context. If the context does not contain "
    "the answer, say you do not know."
)

SYSTEM_V2 = (
    "You are a support assistant for the Northwind Cloud Platform.\n"
    "Rules:\n"
    "1. Answer ONLY from the numbered context snippets. Never use outside knowledge.\n"
    "2. Every factual claim must be traceable to a snippet, and you must cite it "
    "inline as [S1], [S2], ...\n"
    "3. If the context does not contain the answer, reply exactly: "
    "I don't have enough information in the retrieved context to answer this.\n"
    "4. Be concise and factual. Do not speculate, and do not mention that you were "
    "given a context or that you are an AI model.\n"
    "5. Prefer the most specific operational fact (a limit, a default, a command) "
    "over a vague summary."
)

USER_V1 = """Context:
{context}

Question: {question}

Answer:"""

USER_V2 = """Retrieved context:
{context}

Question: {question}

Write a grounded answer using inline citations like [S1]. If the context does
not support an answer, use the exact refusal sentence from your instructions."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A named, versioned pair of system/user templates."""

    prompt_id: str
    system: str
    user: str
    description: str = ""

    def render(self, question: str, context: str) -> tuple[str, str]:
        """Render the (system, user) pair for a question and a context block."""
        return self.system, self.user.format(context=context, question=question)


PROMPT_REGISTRY: dict[str, PromptTemplate] = {
    "v1": PromptTemplate(
        prompt_id="v1",
        system=SYSTEM_V1,
        user=USER_V1,
        description="Baseline grounded answering without a citation contract.",
    ),
    "v2": PromptTemplate(
        prompt_id="v2",
        system=SYSTEM_V2,
        user=USER_V2,
        description="Adds citation contract, refusal rule and answer-shape guidance.",
    ),
}

DEFAULT_PROMPT_ID = "v2"

#: Marker used to detect refusals without depending on an LLM's phrasing.
REFUSAL_MARKER = "don't have enough information"


def get_prompt(prompt_id: str) -> PromptTemplate:
    """Look up a prompt template, failing loudly on typos."""
    try:
        return PROMPT_REGISTRY[prompt_id]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"unknown prompt_id {prompt_id!r}; available: {sorted(PROMPT_REGISTRY)}"
        ) from exc


def list_prompts() -> list[str]:
    return sorted(PROMPT_REGISTRY)


def is_refusal(answer: str) -> bool:
    """Detect the platform's canonical refusal sentence."""
    return REFUSAL_MARKER in answer.lower()
