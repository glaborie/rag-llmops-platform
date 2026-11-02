"""Optional LLM-as-judge scoring.

The deterministic metrics are the gate; the judge adds nuance and is useful
when a real model is in the loop. Because a judge is itself a model, its
verdict is recorded alongside the deterministic metrics — never instead of
them.
"""

from __future__ import annotations

from ragllmops.logging_utils import get_logger
from ragllmops.text import sentence_split

logger = get_logger("evaluate.judge")

JUDGE_INSTRUCTION = """You are grading whether an answer is grounded in the given context.

Context:
{context}

Question: {question}

Answer: {answer}

Reply with exactly one word: GROUNDED if every claim in the answer is supported
by the context, otherwise UNGROUNDED."""


def judge_faithfulness(llm, question: str, answer: str, context: str) -> float:
    """Score answer faithfulness with an LLM judge.

    Returns ``1.0`` for grounded, ``0.0`` for ungrounded, and ``-1.0`` when the
    judge could not be reached (so callers can exclude it from averages).
    """
    if not answer.strip():
        return 0.0

    prompt = JUDGE_INSTRUCTION.format(context=context[:4000], question=question, answer=answer)
    try:
        result = llm.complete(
            system="You are a strict grading assistant. Answer with one word.",
            user=prompt,
            temperature=0.0,
            max_tokens=5,
        )
    except Exception as exc:
        logger.warning("judge failed (%s)", exc)
        return -1.0

    verdict = result.text.strip().upper()
    if "UNGROUNDED" in verdict:
        return 0.0
    if "GROUNDED" in verdict:
        return 1.0
    return -1.0


def judge_claim_support(llm, answer: str, context: str) -> list[float]:
    """Score each answer sentence individually."""
    scores: list[float] = []
    for sentence in sentence_split(answer):
        scores.append(judge_faithfulness(llm, sentence, sentence, context))
    return scores
