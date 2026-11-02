"""RAG quality metrics.

The metrics follow the RAGAS vocabulary (context precision, context recall,
faithfulness) but are computed deterministically, so the gate behaves the same
on a laptop, in CI and in production:

* **context precision** — of the documents we retrieved, how many were the
  ones the question was actually about.
* **context recall** — of the documents that should have been retrieved, how
  many did we get.
* **faithfulness** — of the claims in the answer, how many are supported by
  the retrieved context.
* **key-term coverage** — of the facts the answer must contain, how many are
  present. This is the metric that catches "fluent but wrong".

A deterministic metric is not as good as an LLM judge for nuance, but it is
*reproducible* — which is what a CI gate needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragllmops.text import sentence_split, term_set

_CITATION_RE = re.compile(r"\[S(\d+)\]")


@dataclass(slots=True)
class CaseResult:
    """The evaluation result for a single golden case."""

    case_id: str
    question: str
    answer: str
    expected_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    key_terms: list[str]
    missing_key_terms: list[str]
    faithfulness: float
    context_precision: float
    context_recall: float
    key_term_coverage: float
    answer_relevancy: float
    citation_hit_rate: float
    latency_ms: float
    refused: bool
    failures: list[str] = field(default_factory=list)
    #: The LLM judge's verdict: 1.0 grounded, 0.0 ungrounded, ``None`` when
    #: no judge ran (or the judge could not be reached).
    judge_score: float | None = None

    @property
    def passed(self) -> bool:
        return not self.failures


def context_precision(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """Fraction of retrieved documents that are relevant."""
    unique = set(retrieved_doc_ids)
    if not unique:
        return 0.0
    relevant = unique & set(expected_doc_ids)
    return len(relevant) / len(unique)


def context_recall(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """Fraction of the expected documents that were retrieved."""
    expected = set(expected_doc_ids)
    if not expected:
        return 1.0
    return len(set(retrieved_doc_ids) & expected) / len(expected)


def key_term_coverage(answer: str, key_terms: list[str]) -> tuple[float, list[str]]:
    """Fraction of the required key terms present in the answer."""
    if not key_terms:
        return 1.0, []
    lowered = answer.lower()
    missing = [term for term in key_terms if term.lower() not in lowered]
    return (len(key_terms) - len(missing)) / len(key_terms), missing


def faithfulness(answer: str, context: str, threshold: float = 0.6) -> float:
    """Fraction of answer sentences supported by the retrieved context.

    A sentence counts as supported when most of its content terms appear in
    the context, or when it carries a citation marker (the extractive
    provider copies context verbatim, so citations are strong evidence).
    """
    sentences = sentence_split(answer)
    if not sentences:
        return 0.0

    context_terms = term_set(context)
    supported = 0
    for sentence in sentences:
        if _CITATION_RE.search(sentence):
            supported += 1
            continue
        sentence_terms = term_set(sentence)
        if not sentence_terms:
            supported += 1
            continue
        overlap = len(sentence_terms & context_terms) / len(sentence_terms)
        if overlap >= threshold:
            supported += 1

    return supported / len(sentences)


def answer_relevancy(question: str, answer: str) -> float:
    """How much of the question's content appears in the answer."""
    question_terms = term_set(question)
    if not question_terms:
        return 0.0
    answer_terms = term_set(answer)
    return len(question_terms & answer_terms) / len(question_terms)


def citation_hit_rate(citations: list[dict], expected_doc_ids: list[str]) -> float:
    """Fraction of citations pointing at an expected document."""
    if not citations:
        return 0.0
    expected = set(expected_doc_ids)
    hits = sum(1 for citation in citations if citation.get("doc_id") in expected)
    return hits / len(citations)


def evaluate_case(
    case,
    answer: str,
    retrieved_doc_ids: list[str],
    citations: list[dict],
    context: str,
    latency_ms: float,
    refused: bool = False,
) -> CaseResult:
    """Score a single answered case."""
    coverage, missing = key_term_coverage(answer, case.key_terms)

    failures: list[str] = []
    if refused:
        failures.append("refused")
    if context_recall(retrieved_doc_ids, case.expected_doc_ids) < 0.5:
        failures.append("context_recall")
    if coverage < 0.5:
        failures.append("key_term_coverage")
    if not answer.strip():
        failures.append("empty_answer")

    return CaseResult(
        case_id=case.id,
        question=case.question,
        answer=answer,
        expected_doc_ids=list(case.expected_doc_ids),
        retrieved_doc_ids=list(retrieved_doc_ids),
        key_terms=list(case.key_terms),
        missing_key_terms=missing,
        faithfulness=round(faithfulness(answer, context), 4),
        context_precision=round(context_precision(retrieved_doc_ids, case.expected_doc_ids), 4),
        context_recall=round(context_recall(retrieved_doc_ids, case.expected_doc_ids), 4),
        key_term_coverage=round(coverage, 4),
        answer_relevancy=round(answer_relevancy(case.question, answer), 4),
        citation_hit_rate=round(citation_hit_rate(citations, case.expected_doc_ids), 4),
        latency_ms=round(latency_ms, 3),
        refused=refused,
        failures=failures,
    )
