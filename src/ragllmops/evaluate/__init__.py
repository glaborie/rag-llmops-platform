"""Offline evaluation and quality gates."""

from ragllmops.evaluate.dataset import GoldenCase, load_golden_set
from ragllmops.evaluate.metrics import CaseResult, evaluate_case
from ragllmops.evaluate.runner import (
    EvalReport,
    GateThresholds,
    aggregate,
    check_gate,
    run_evaluation,
)

__all__ = [
    "CaseResult",
    "EvalReport",
    "GateThresholds",
    "GoldenCase",
    "aggregate",
    "check_gate",
    "evaluate_case",
    "load_golden_set",
    "run_evaluation",
]
