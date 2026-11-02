"""The closed loop.

This is the piece the tutorial leaves manual. It:

1. serves a baseline of in-domain traffic (building a reference window);
2. serves a shifted mix of traffic (the "users changed their behaviour"
   scenario);
3. detects the drift in the request log;
4. builds a candidate configuration, evaluates it against the golden set,
   and promotes it only if it clears the quality gate and beats the
   incumbent;
5. writes an incident report describing what happened.

Run it with ``python -m ragllmops.cli loop``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragllmops.logging_utils import get_logger
from ragllmops.models import RAGConfigBundle
from ragllmops.monitor.drift import compute_drift

logger = get_logger("orchestration.closed_loop")

#: Questions the knowledge base does not cover. They stand in for a change in
#: user behaviour: same service, different topic.
OFF_TOPIC_QUERIES = (
    "How do I bake sourdough bread with a starter?",
    "What is the capital city of Peru?",
    "How do I train for a marathon in eight weeks?",
    "What is the best way to learn the violin?",
    "How does photosynthesis work?",
    "Who wrote the novel Dune?",
    "What should I look for when buying a used car?",
    "How do I set up a home network with a router?",
    "What is the difference between weather and climate?",
    "How do I plan a two week trip to Japan?",
    "What are good stretches for lower back pain?",
    "How do I make cold brew coffee at home?",
    "What is a good beginner weightlifting routine?",
    "How do I keep basil plants alive indoors?",
    "What are the rules of chess castling?",
)


@dataclass(slots=True)
class LoopResult:
    """What happened during the closed loop."""

    steps: list[dict[str, Any]] = field(default_factory=list)
    drift_detected: bool = False
    drift_features: list[str] = field(default_factory=list)
    promoted: bool = False
    champion_before: str | None = None
    champion_after: str | None = None
    candidate_version: str | None = None
    gate: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    report_path: Path | None = None

    def summary(self) -> str:
        lines = [f"closed loop ({len(self.steps)} steps)"]
        for step in self.steps:
            lines.append(f"  {step['at']}  {step['name']}: {step['detail']}")
        lines.append(f"drift detected : {self.drift_detected} {self.drift_features}")
        lines.append(f"promoted       : {self.promoted}")
        if self.report_path:
            lines.append(f"report         : {self.report_path}")
        return "\n".join(lines)


def _now() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")


def _serve_traffic(platform, queries: list[str], label: str) -> int:
    """Answer every query, logging the requests. Returns the count."""
    served = 0
    for query in queries:
        try:
            platform.answer(query)
            served += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not answer %r: %s", query[:40], exc)
    logger.info("served %s %s requests", served, label)
    return served


def run_closed_loop(
    platform,
    baseline_queries: list[str] | None = None,
    drift_queries: list[str] | None = None,
    candidate: RAGConfigBundle | None = None,
    report_dir: Path | None = None,
) -> LoopResult:
    """Run the full detect → evaluate → gate → promote loop."""
    result = LoopResult()
    settings = platform.settings

    # ------------------------------------------------------------------
    # 0. make sure there is something to serve
    # ------------------------------------------------------------------
    if not len(platform.catalog):
        platform.reindex()
        result.steps.append(
            {"at": _now(), "name": "ingest", "detail": "built the retrieval index"}
        )

    # ------------------------------------------------------------------
    # 1. establish the incumbent
    # ------------------------------------------------------------------
    champion = platform.champion_bundle()
    if champion is None:
        # The starting champion is deliberately narrow: it retrieves a single
        # chunk, which is precise but misses multi-part questions. The
        # retraining candidate later widens the context.
        champion = platform.build_bundle(
            retrieval_top_k=1, notes="initial champion (narrow context)"
        )
        report = platform.evaluate(bundle=champion)
        champion = champion.with_metrics(report.aggregates)
        platform.register_bundle(champion)
        platform.promote(settings.champion_alias, champion.version)
        result.steps.append(
            {
                "at": _now(),
                "name": "bootstrap",
                "detail": (
                    f"registered {champion.version} as champion "
                    f"(recall={report.aggregates.get('context_recall', 0):.3f}, "
                    f"faithfulness={report.aggregates.get('faithfulness', 0):.3f})"
                ),
            }
        )
    result.champion_before = champion.version

    # ------------------------------------------------------------------
    # 2. baseline traffic -> reference window
    # ------------------------------------------------------------------
    if baseline_queries is None:
        baseline_queries = _baseline_queries(platform)
    _serve_traffic(platform, baseline_queries, "baseline")
    result.steps.append(
        {"at": _now(), "name": "baseline", "detail": f"served {len(baseline_queries)} requests"}
    )

    # ------------------------------------------------------------------
    # 3. shifted traffic -> current window
    # ------------------------------------------------------------------
    drift_queries = list(drift_queries or OFF_TOPIC_QUERIES)
    _serve_traffic(platform, drift_queries, "shifted")
    result.steps.append(
        {"at": _now(), "name": "traffic-shift", "detail": f"served {len(drift_queries)} requests"}
    )

    # ------------------------------------------------------------------
    # 4. detect drift
    # ------------------------------------------------------------------
    # Only look at the traffic this run served: the request log may already
    # contain history from earlier runs.
    served = len(baseline_queries) + len(drift_queries)
    all_records = platform.log_store.fetch(limit=100000)
    records = all_records[-served:] if served else []
    reference = records[: len(baseline_queries)]
    current = records[len(baseline_queries) :]

    drift_report = compute_drift(
        reference, current, threshold=settings.drift_psi_threshold
    )
    result.drift_detected = drift_report.is_drifted
    result.drift_features = drift_report.drifted_features
    result.steps.append(
        {
            "at": _now(),
            "name": "monitor",
            "detail": (
                f"drift={'detected' if drift_report.is_drifted else 'none'} "
                f"features={drift_report.drifted_features or 'none'}"
            ),
        }
    )

    if not drift_report.is_drifted:
        result.report_path = _write_report(platform, result, drift_report)
        return result

    # ------------------------------------------------------------------
    # 5. build a candidate and evaluate it
    # ------------------------------------------------------------------
    if candidate is None:
        candidate = platform.build_bundle(
            prompt_id="v2",
            retrieval_top_k=3,
            notes="auto-retrain candidate after drift (wider context)",
        )

    # ------------------------------------------------------------------
    # 5b. learn from the traffic that was just served
    # ------------------------------------------------------------------
    candidate = _attach_learned_ranker(platform, candidate, result)

    result.candidate_version = candidate.version
    platform.register_bundle(candidate)
    platform.promote(settings.challenger_alias, candidate.version)
    result.steps.append(
        {
            "at": _now(),
            "name": "candidate",
            "detail": f"registered {candidate.version} as challenger",
        }
    )

    evaluation = platform.evaluate(bundle=candidate)
    result.evaluation = evaluation.to_dict()
    result.steps.append(
        {
            "at": _now(),
            "name": "evaluate",
            "detail": (
                f"gate={'PASS' if evaluation.passed else 'FAIL'} "
                f"faithfulness={evaluation.aggregates.get('faithfulness', 0):.3f} "
                f"recall={evaluation.aggregates.get('context_recall', 0):.3f}"
            ),
        }
    )

    # ------------------------------------------------------------------
    # 6. gate and promote
    # ------------------------------------------------------------------
    gate = _gate_decision(platform, candidate, evaluation)
    result.gate = gate
    result.steps.append(
        {
            "at": _now(),
            "name": "gate",
            "detail": f"promote={gate['promote']} {gate.get('reason', '')}".strip(),
        }
    )

    if gate["promote"]:
        platform.promote(settings.champion_alias, candidate.version)
        candidate = candidate.with_metrics(evaluation.aggregates)
        platform.register_bundle(candidate)
        platform.promote(settings.challenger_alias, champion.version)
        result.promoted = True
        result.champion_after = candidate.version
        result.steps.append(
            {
                "at": _now(),
                "name": "promote",
                "detail": f"champion {champion.version} -> {candidate.version}",
            }
        )
    else:
        result.steps.append(
            {"at": _now(), "name": "promote", "detail": "champion left unchanged"}
        )

    platform.materialize_features()
    result.report_path = _write_report(platform, result, drift_report)
    return result


# ----------------------------------------------------------------------
def _attach_learned_ranker(
    platform,
    candidate: RAGConfigBundle,
    result: LoopResult,
) -> RAGConfigBundle:
    """Train a ranker on the feedback and put it into the candidate.

    The ranker only goes in when it beat the rule it replaces on the
    held-out slice. A learned ranker that does not beat the rule-based one
    is not an improvement, whatever its training metrics look like.
    """
    from dataclasses import replace

    from ragllmops.rag.bundle import with_version

    report = platform.train_ranker_from_feedback()
    if not report.get("trained"):
        result.steps.append(
            {
                "at": _now(),
                "name": "train-reranker",
                "detail": f"no ranker trained ({report.get('reason', 'unknown')})",
            }
        )
        return candidate

    validation = report.get("validation_metrics", {})
    baseline = report.get("baseline_metrics", {})
    learned_score = float(validation.get("pairwise_accuracy", 0.0))
    baseline_score = float(baseline.get("pairwise_accuracy", 0.0))

    if learned_score < baseline_score:
        result.steps.append(
            {
                "at": _now(),
                "name": "train-reranker",
                "detail": (
                    f"ranker {report['artifact_id']} rejected: held-out pairwise "
                    f"accuracy {learned_score:.3f} < rule-based {baseline_score:.3f}"
                ),
            }
        )
        return candidate

    ranker_metrics = {
        "pairwise_accuracy": learned_score,
        "accuracy": float(validation.get("accuracy", 0.0)),
        "auc": float(validation.get("auc", 0.0)),
        "baseline_pairwise_accuracy": baseline_score,
        "validation_examples": float(validation.get("examples", 0.0)),
    }
    candidate = with_version(
        replace(
            candidate,
            ranker_artifact_id=report["artifact_id"],
            ranker_metrics=ranker_metrics,
            notes=f"learned ranker {report['artifact_id']}",
        )
    )
    result.steps.append(
        {
            "at": _now(),
            "name": "train-reranker",
            "detail": (
                f"trained ranker {report['artifact_id']} on {report['examples']} examples "
                f"(held-out pairwise accuracy {learned_score:.3f} vs {baseline_score:.3f})"
            ),
        }
    )
    return candidate


# ----------------------------------------------------------------------
def _baseline_queries(platform) -> list[str]:
    """Take the questions from the golden set as in-domain traffic."""
    from ragllmops.evaluate.dataset import load_golden_set

    try:
        cases = load_golden_set(platform.settings.eval_file)
    except FileNotFoundError:
        return []
    queries = [case.question for case in cases]
    # Repeat so the reference window is large enough for a stable PSI.
    return queries * 3


def _gate_decision(platform, candidate: RAGConfigBundle, evaluation) -> dict[str, Any]:
    """Promote only if the candidate passes the gate and beats the incumbent."""
    if not evaluation.passed:
        return {
            "promote": False,
            "reason": f"quality gate failed: {', '.join(evaluation.failures)}",
            "comparisons": {},
        }

    incumbent = platform.champion_bundle()
    comparisons: dict[str, dict[str, float]] = {}
    improved = True
    for metric in ("faithfulness", "context_recall", "key_term_coverage"):
        candidate_value = evaluation.aggregates.get(metric)
        incumbent_value = (incumbent.metrics if incumbent else {}).get(metric)
        if candidate_value is None:
            continue
        if incumbent_value is None:
            comparisons[metric] = {"candidate": candidate_value}
            continue
        comparisons[metric] = {
            "candidate": candidate_value,
            "champion": incumbent_value,
            "delta": round(candidate_value - incumbent_value, 4),
        }
        if candidate_value < incumbent_value:
            improved = False

    # A learned ranker is judged on the held-out feedback slice, never on
    # the golden set the retriever was tuned on.
    if candidate.ranker_metrics:
        learned = candidate.ranker_metrics.get("pairwise_accuracy")
        baseline = candidate.ranker_metrics.get("baseline_pairwise_accuracy")
        champion_score = (
            (incumbent.ranker_metrics if incumbent else None) or {}
        ).get("pairwise_accuracy")
        if learned is not None:
            comparison = {"candidate": learned}
            if baseline is not None:
                comparison["baseline"] = baseline
                comparison["delta_vs_baseline"] = round(learned - baseline, 4)
                if learned < baseline:
                    improved = False
            if champion_score is not None:
                comparison["champion"] = champion_score
                comparison["delta"] = round(learned - champion_score, 4)
                if learned < champion_score:
                    improved = False
            comparisons["ranker_pairwise_accuracy"] = comparison

    return {
        "promote": improved,
        "reason": "" if improved else "candidate is not better than the champion",
        "comparisons": comparisons,
    }


def _write_report(platform, result: LoopResult, drift_report, report_dir: Path | None = None) -> Path:
    """Write the incident report."""
    directory = Path(report_dir or platform.settings.reports_path)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"incident-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.md"

    lines = [
        "# RAG platform incident report",
        "",
        f"**Generated:** {datetime.now(UTC).isoformat()}",
        f"**Champion before:** `{result.champion_before}`",
        f"**Champion after:** `{result.champion_after or 'unchanged'}`",
        f"**Drift detected:** {result.drift_detected} "
        f"({', '.join(result.drift_features) or 'none'})",
        f"**Promoted:** {result.promoted}",
        "",
        "## Timeline",
        "",
        "| time | step | detail |",
        "| --- | --- | --- |",
    ]
    for step in result.steps:
        detail = str(step["detail"]).replace("|", "\\|")
        lines.append(f"| {step['at']} | {step['name']} | {detail} |")

    lines += ["", "## Drift measurements", "", "| feature | psi | drifted |", "| --- | --- | --- |"]
    for feature in drift_report.features:
        lines.append(
            f"| {feature.feature} | {feature.psi:.4f} | {'yes' if feature.drifted else 'no'} |"
        )

    if result.evaluation:
        aggregates = result.evaluation.get("aggregates", {})
        lines += ["", "## Candidate evaluation", ""]
        lines += ["| metric | value |", "| --- | --- |"]
        for metric, value in aggregates.items():
            lines.append(f"| {metric} | {value} |")
        lines += [
            "",
            f"**Gate:** {'PASS' if result.evaluation.get('passed') else 'FAIL'} "
            f"({', '.join(result.evaluation.get('failures', [])) or 'no failures'})",
        ]

    if result.gate.get("comparisons"):
        lines += ["", "## Gate comparison", "", "| metric | candidate | champion | delta |",
                  "| --- | --- | --- | --- |"]
        for metric, values in result.gate["comparisons"].items():
            lines.append(
                f"| {metric} | {values.get('candidate')} | {values.get('champion')} "
                f"| {values.get('delta')} |"
            )

    lines += [
        "",
        "## What happened",
        "",
        "Traffic shifted away from the topics the knowledge base covers. The "
        "retrieval scores of the served requests dropped, which the drift "
        "monitor flagged above the PSI threshold. A candidate configuration "
        "was registered and evaluated against the golden set; the quality "
        "gate decided whether it was allowed to become the champion.",
        "",
        "## What you would do next",
        "",
        "1. Read the drift report — which feature moved, and by how much.",
        "2. Check whether the cause is data (a new document) or behaviour "
        "(a new question type).",
        "3. If it is data, add the missing document and re-run the pipeline.",
        "4. If it is behaviour, decide whether to expand the knowledge base "
        "or to route those questions elsewhere.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("incident report written to %s", path)
    return path


__all__ = ["LoopResult", "OFF_TOPIC_QUERIES", "run_closed_loop"]
