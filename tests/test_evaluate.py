"""Evaluation metrics, the gate and the golden-set loader."""

from ragllmops.evaluate.dataset import load_golden_set
from ragllmops.evaluate.metrics import (
    answer_relevancy,
    citation_hit_rate,
    context_precision,
    context_recall,
    evaluate_case,
    faithfulness,
    key_term_coverage,
)
from ragllmops.evaluate.runner import GateThresholds, aggregate, check_gate


def test_context_precision_counts_only_relevant_documents():
    assert context_precision(["compute", "observability"], ["compute"]) == 0.5
    assert context_precision(["compute", "compute"], ["compute"]) == 1.0
    assert context_precision([], ["compute"]) == 0.0


def test_context_recall_counts_missing_documents():
    assert context_recall(["compute"], ["compute", "networking"]) == 0.5
    assert context_recall(["compute", "networking"], ["compute", "networking"]) == 1.0
    assert context_recall(["compute"], []) == 1.0


def test_key_term_coverage_reports_missing_terms():
    coverage, missing = key_term_coverage("The limit is 600 requests", ["600 requests", "rto"])
    assert coverage == 0.5
    assert missing == ["rto"]


def test_faithfulness_is_high_for_cited_grounded_answers():
    context = "The autoscaling cooldown is 120 seconds. Retention is 90 days."
    answer = "The autoscaling cooldown is 120 seconds [S1]."
    assert faithfulness(answer, context) == 1.0


def test_faithfulness_is_low_for_invented_answers():
    context = "The autoscaling cooldown is 120 seconds."
    answer = "Bananas are shipped from Ecuador by air freight."
    assert faithfulness(answer, context) < 0.5


def test_answer_relevancy_overlaps_question_terms():
    assert answer_relevancy("what is the cooldown", "the cooldown is 120 seconds") > 0.5


def test_citation_hit_rate():
    citations = [{"doc_id": "compute"}, {"doc_id": "billing"}]
    assert citation_hit_rate(citations, ["compute"]) == 0.5
    assert citation_hit_rate([], ["compute"]) == 0.0


def test_evaluate_case_flags_failures(golden_set_path):
    case = load_golden_set(golden_set_path)[0]
    result = evaluate_case(
        case=case,
        answer="totally unrelated",
        retrieved_doc_ids=["billing"],
        citations=[{"doc_id": "billing"}],
        context="irrelevant",
        latency_ms=1.0,
    )
    assert "context_recall" in result.failures
    assert "key_term_coverage" in result.failures
    assert not result.passed


def test_gate_passes_when_metrics_are_above_thresholds():
    thresholds = GateThresholds(
        min_faithfulness=0.5, min_context_recall=0.6, min_key_term_coverage=0.5
    )
    aggregates = {"faithfulness": 0.9, "context_recall": 0.8, "key_term_coverage": 0.7}
    assert check_gate(aggregates, thresholds) == []


def test_gate_fails_when_a_metric_is_below_the_threshold():
    thresholds = GateThresholds(min_faithfulness=0.95)
    failures = check_gate({"faithfulness": 0.5}, thresholds)
    assert failures and "faithfulness" in failures[0]


def test_aggregate_of_no_cases_is_empty():
    assert aggregate([]) == {}


# ----------------------------------------------------------------------
# the LLM judge
# ----------------------------------------------------------------------
def _case_result(judge_score: float | None = None):
    from ragllmops.evaluate.metrics import CaseResult

    return CaseResult(
        case_id="q1",
        question="q",
        answer="a",
        expected_doc_ids=["compute"],
        retrieved_doc_ids=["compute"],
        key_terms=[],
        missing_key_terms=[],
        faithfulness=1.0,
        context_precision=1.0,
        context_recall=1.0,
        key_term_coverage=1.0,
        answer_relevancy=1.0,
        citation_hit_rate=1.0,
        latency_ms=1.0,
        refused=False,
        judge_score=judge_score,
    )


def test_an_evaluation_without_a_judge_has_no_judge_rate():
    aggregates = aggregate([_case_result(), _case_result()])
    assert "judge_grounded_rate" not in aggregates
    assert check_gate(aggregates, GateThresholds()) == []


def test_the_judge_rate_is_recorded_when_the_judge_answers():
    aggregates = aggregate([_case_result(1.0), _case_result(0.0), _case_result(1.0)])
    assert aggregates["judge_grounded_rate"] == round(2 / 3, 4)
    assert aggregates["judge_cases"] == 3.0


def test_the_gate_fails_when_the_judge_finds_ungrounded_answers():
    aggregates = aggregate([_case_result(0.0), _case_result(0.0)])
    failures = check_gate(aggregates, GateThresholds(min_judge_grounded_rate=0.5))
    assert failures and "judge_grounded_rate" in failures[0]


def test_a_judge_that_never_answers_does_not_fail_the_gate():
    """A judge that could not be reached is not a failing judge."""
    aggregates = aggregate([_case_result(None), _case_result(None)])
    assert "judge_grounded_rate" not in aggregates
    assert check_gate(aggregates, GateThresholds()) == []


def test_the_judge_records_its_verdict_on_the_case(platform):
    """The judge is wired into the runner, not just sitting in a module."""
    report = platform.evaluate(use_judge=True)
    # The extractive provider cannot grade, so there is no verdict — but the
    # run must not break, and the report must not claim a judge score.
    assert "judge_grounded_rate" not in report.aggregates
    assert report.aggregates["faithfulness"] >= 0


def test_a_judge_that_answers_reaches_the_gate(platform):
    """With a working judge the verdict is recorded and gated on."""
    from ragllmops.evaluate.dataset import load_golden_set
    from ragllmops.evaluate.runner import run_evaluation

    bundle = platform.current_bundle()
    cases = load_golden_set(platform.settings.eval_file)
    report = run_evaluation(
        chain=platform.chain_for(bundle),
        cases=cases,
        thresholds=GateThresholds(min_judge_grounded_rate=1.0),
        judge=lambda question, answer, context: 1.0,
    )
    assert report.aggregates["judge_grounded_rate"] == 1.0
    assert report.aggregates["judge_cases"] == len(cases)
    assert report.passed
    assert all(case.judge_score == 1.0 for case in report.cases)


def test_an_ungrounded_verdict_fails_the_gate(platform):
    from ragllmops.evaluate.dataset import load_golden_set
    from ragllmops.evaluate.runner import run_evaluation

    bundle = platform.current_bundle()
    cases = load_golden_set(platform.settings.eval_file)
    report = run_evaluation(
        chain=platform.chain_for(bundle),
        cases=cases,
        thresholds=GateThresholds(min_judge_grounded_rate=0.5),
        judge=lambda question, answer, context: 0.0,
    )
    assert report.aggregates["judge_grounded_rate"] == 0.0
    assert not report.passed
    assert any("judge_grounded_rate" in failure for failure in report.failures)
