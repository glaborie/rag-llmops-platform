"""The evaluation runner and the quality gate."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from ragllmops.evaluate.dataset import GoldenCase
from ragllmops.evaluate.metrics import CaseResult, evaluate_case
from ragllmops.logging_utils import get_logger

logger = get_logger("evaluate.runner")


@dataclass(frozen=True, slots=True)
class GateThresholds:
    """Minimum acceptable aggregate metrics."""

    min_faithfulness: float = 0.5
    min_context_recall: float = 0.6
    min_key_term_coverage: float = 0.5
    min_context_precision: float = 0.5
    max_latency_ms: float = 5000.0
    #: How many answers the LLM judge has to call grounded. The check only
    #: applies when a judge actually returned verdicts — an evaluation
    #: without a judge is not failed for having no judge score.
    min_judge_grounded_rate: float = 0.5


@dataclass(slots=True)
class EvalReport:
    """The outcome of evaluating a bundle against the golden set."""

    bundle_version: str
    prompt_id: str
    llm_model: str
    cases: list[CaseResult] = field(default_factory=list)
    aggregates: dict[str, float] = field(default_factory=dict)
    thresholds: GateThresholds = field(default_factory=GateThresholds)
    failures: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "bundle_version": self.bundle_version,
            "prompt_id": self.prompt_id,
            "llm_model": self.llm_model,
            "passed": self.passed,
            "failures": self.failures,
            "aggregates": self.aggregates,
            "thresholds": asdict(self.thresholds),
            "duration_ms": self.duration_ms,
            "cases": [
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "expected_doc_ids": case.expected_doc_ids,
                    "retrieved_doc_ids": case.retrieved_doc_ids,
                    "faithfulness": case.faithfulness,
                    "context_precision": case.context_precision,
                    "context_recall": case.context_recall,
                    "key_term_coverage": case.key_term_coverage,
                    "answer_relevancy": case.answer_relevancy,
                    "citation_hit_rate": case.citation_hit_rate,
                    "latency_ms": case.latency_ms,
                    "refused": case.refused,
                    "judge_score": case.judge_score,
                    "failures": case.failures,
                    "missing_key_terms": case.missing_key_terms,
                }
                for case in self.cases
            ],
        }

    def summary(self) -> str:
        """A human-readable summary for the CLI."""
        lines = [
            f"bundle          : {self.bundle_version}",
            f"prompt          : {self.prompt_id}",
            f"llm             : {self.llm_model}",
            f"cases           : {len(self.cases)}",
            f"faithfulness    : {self.aggregates.get('faithfulness', 0):.3f}"
            f" (min {self.thresholds.min_faithfulness})",
            f"context recall  : {self.aggregates.get('context_recall', 0):.3f}"
            f" (min {self.thresholds.min_context_recall})",
            f"context precision: {self.aggregates.get('context_precision', 0):.3f}"
            f" (min {self.thresholds.min_context_precision})",
            f"key term cover  : {self.aggregates.get('key_term_coverage', 0):.3f}"
            f" (min {self.thresholds.min_key_term_coverage})",
            f"mean latency    : {self.aggregates.get('latency_ms', 0):.1f} ms",
            f"gate            : {'PASS' if self.passed else 'FAIL'}",
        ]
        if "judge_grounded_rate" in self.aggregates:
            lines.append(
                f"judge grounded  : {self.aggregates['judge_grounded_rate']:.3f} "
                f"({int(self.aggregates.get('judge_cases', 0))} cases judged)"
            )
        if self.failures:
            lines.append("failed checks   : " + ", ".join(self.failures))
        return "\n".join(lines)


def aggregate(cases: Sequence[CaseResult]) -> dict[str, float]:
    """Average the per-case metrics."""
    if not cases:
        return {}

    def mean(attr: str) -> float:
        return sum(getattr(case, attr) for case in cases) / len(cases)

    aggregates = {
        "faithfulness": round(mean("faithfulness"), 4),
        "context_precision": round(mean("context_precision"), 4),
        "context_recall": round(mean("context_recall"), 4),
        "key_term_coverage": round(mean("key_term_coverage"), 4),
        "answer_relevancy": round(mean("answer_relevancy"), 4),
        "citation_hit_rate": round(mean("citation_hit_rate"), 4),
        "latency_ms": round(mean("latency_ms"), 3),
        "refusal_rate": round(sum(1 for case in cases if case.refused) / len(cases), 4),
        "cases": float(len(cases)),
    }

    # The judge rate only exists when the judge answered. A judge that could
    # not be reached must not silently become a perfect score — and must not
    # silently become a failing one either.
    judged = [case.judge_score for case in cases if case.judge_score is not None]
    if judged:
        aggregates["judge_grounded_rate"] = round(
            sum(1 for score in judged if score >= 0.5) / len(judged), 4
        )
        aggregates["judge_cases"] = float(len(judged))
    return aggregates


def check_gate(aggregates: dict[str, float], thresholds: GateThresholds) -> list[str]:
    """Return the list of failed checks (empty means the gate passed)."""
    failures: list[str] = []

    checks = (
        ("faithfulness", thresholds.min_faithfulness, "min"),
        ("context_recall", thresholds.min_context_recall, "min"),
        ("context_precision", thresholds.min_context_precision, "min"),
        ("key_term_coverage", thresholds.min_key_term_coverage, "min"),
        ("latency_ms", thresholds.max_latency_ms, "max"),
        ("judge_grounded_rate", thresholds.min_judge_grounded_rate, "min"),
    )
    for metric, threshold, direction in checks:
        value = aggregates.get(metric)
        if value is None:
            continue
        if direction == "min" and value < threshold:
            failures.append(f"{metric} {value:.3f} < {threshold:.3f}")
        if direction == "max" and value > threshold:
            failures.append(f"{metric} {value:.1f} > {threshold:.1f}")

    return failures


def run_evaluation(
    chain,
    cases: list[GoldenCase],
    thresholds: GateThresholds | None = None,
    judge=None,
) -> EvalReport:
    """Run the golden set against a chain and evaluate the gate."""
    started = time.perf_counter()
    thresholds = thresholds or GateThresholds()

    results: list[CaseResult] = []
    for case in cases:
        chain_result = chain.answer(case.question)
        response = chain_result.response
        retrieved_doc_ids = [citation.doc_id for citation in response.citations]
        citations = [citation.to_dict() for citation in response.citations]

        result = evaluate_case(
            case=case,
            answer=response.answer,
            retrieved_doc_ids=retrieved_doc_ids,
            citations=citations,
            context=chain_result.context,
            latency_ms=response.latency_ms,
            refused=response.refused,
        )
        results.append(result)

        if judge is not None:
            verdict = judge(
                question=case.question,
                answer=response.answer,
                context=chain_result.context,
            )
            if verdict >= 0:
                result.judge_score = verdict
                result.failures.extend([] if verdict else ["judge_ungrounded"])

    aggregates = aggregate(results)
    failures = check_gate(aggregates, thresholds)

    report = EvalReport(
        bundle_version=chain.bundle.version,
        prompt_id=chain.bundle.prompt_id,
        llm_model=chain.bundle.llm_model,
        cases=results,
        aggregates=aggregates,
        thresholds=thresholds,
        failures=failures,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        passed=not failures,
    )
    logger.info(
        "evaluation %s for bundle %s (%s)",
        "passed" if report.passed else "failed",
        report.bundle_version,
        ", ".join(failures) or "all checks ok",
    )
    return report
