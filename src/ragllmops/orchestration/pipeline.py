"""The platform's own pipeline.

This is the DAG that implements the closed loop:

    ingest -> materialize_features -> evaluate -> gate -> promote
                                     monitor -> alert

``evaluate`` scores the *candidate* bundle, ``gate`` compares it with the
incumbent, and ``promote`` only moves the champion alias when the candidate
is both above the quality floor and better than the incumbent.
"""

from __future__ import annotations

from typing import Any

from ragllmops.logging_utils import get_logger
from ragllmops.monitor.alerts import SEVERITY_CRITICAL, SEVERITY_WARNING
from ragllmops.monitor.drift import compute_drift
from ragllmops.orchestration.dag import DAG, Asset, AssetContext

logger = get_logger("orchestration.pipeline")


# ----------------------------------------------------------------------
# assets
# ----------------------------------------------------------------------
def _ingest(context: AssetContext) -> dict[str, Any]:
    """Rebuild the retrieval index from the corpus."""
    platform = context.platform
    result = platform.reindex()
    return result.to_dict()


def _materialize_features(context: AssetContext) -> dict[str, Any]:
    """Recompute the online features from the request log."""
    platform = context.platform
    features = platform.materialize_features()
    return {"entities": len(features)}


def _evaluate(context: AssetContext) -> dict[str, Any]:
    """Run the golden set against the candidate bundle."""
    platform = context.platform
    candidate = platform.challenger_bundle() or platform.current_bundle()
    report = platform.evaluate(bundle=candidate)
    return report.to_dict()


def _train_reranker(context: AssetContext) -> dict[str, Any]:
    """Train a ranker on the production feedback and register it.

    The trained model is compared with the rule-based baseline on the
    held-out slice before it is put into a bundle: a learned ranker that
    does not beat the rule it replaces is not worth shipping.
    """
    from dataclasses import replace

    from ragllmops.rag.bundle import with_version

    platform = context.platform
    report = platform.train_ranker_from_feedback()
    if not report.get("trained"):
        logger.info("no ranker trained: %s", report.get("reason"))
        return report

    validation = report.get("validation_metrics", {})
    baseline = report.get("baseline_metrics", {})
    ranker_metrics = {
        "pairwise_accuracy": float(validation.get("pairwise_accuracy", 0.0)),
        "accuracy": float(validation.get("accuracy", 0.0)),
        "auc": float(validation.get("auc", 0.0)),
        "baseline_pairwise_accuracy": float(baseline.get("pairwise_accuracy", 0.0)),
        "validation_examples": float(validation.get("examples", 0.0)),
    }

    base = platform.challenger_bundle() or platform.champion_bundle() or platform.current_bundle()
    candidate = with_version(
        replace(
            base,
            ranker_artifact_id=report["artifact_id"],
            ranker_metrics=ranker_metrics,
            notes=f"learned ranker {report['artifact_id']}",
        )
    )
    platform.register_bundle(candidate)
    platform.promote(platform.settings.challenger_alias, candidate.version)

    report["candidate_version"] = candidate.version
    report["ranker_metrics"] = ranker_metrics
    logger.info(
        "trained ranker %s (held-out pairwise accuracy %.3f vs baseline %.3f) -> %s",
        report["artifact_id"],
        ranker_metrics["pairwise_accuracy"],
        ranker_metrics["baseline_pairwise_accuracy"],
        candidate.version,
    )
    return report


def _gate(context: AssetContext) -> dict[str, Any]:
    """Decide whether the candidate should become the champion."""
    platform = context.platform
    evaluation = context.result("evaluate") or {}
    candidate = platform.challenger_bundle()
    champion = platform.champion_bundle()

    candidate_metrics = evaluation.get("aggregates", {})
    champion_metrics = (champion.metrics if champion else {}) or {}

    passed = bool(evaluation.get("passed"))
    failures = list(evaluation.get("failures", []))

    improved = True
    comparisons: dict[str, dict[str, float]] = {}
    for metric in ("faithfulness", "context_recall", "key_term_coverage"):
        candidate_value = candidate_metrics.get(metric)
        champion_value = champion_metrics.get(metric)
        if candidate_value is None:
            continue
        if champion_value is None:
            comparisons[metric] = {"candidate": candidate_value}
            continue
        comparisons[metric] = {
            "candidate": candidate_value,
            "champion": champion_value,
            "delta": round(candidate_value - champion_value, 4),
        }
        if candidate_value < champion_value:
            improved = False

    # A learned ranker has to beat the rule it replaces, and the champion it
    # would replace, on the held-out slice — never on the golden set.
    ranker_comparison = _compare_rankers(candidate, champion)
    if ranker_comparison:
        comparisons["ranker_pairwise_accuracy"] = ranker_comparison
        if ranker_comparison.get("regression"):
            improved = False

    if candidate is None:
        reason = "no candidate registered as challenger"
    elif not passed:
        reason = f"quality gate failed: {', '.join(failures) or 'unknown'}"
    elif not improved:
        reason = "candidate is not better than the champion"
    else:
        reason = "candidate passed the gate and improved on the champion"

    decision = {
        "promote": bool(passed and improved and candidate is not None),
        "candidate_version": candidate.version if candidate else None,
        "champion_version": champion.version if champion else None,
        "gate_passed": passed,
        "gate_failures": failures,
        "improved": improved,
        "reason": reason,
        "comparisons": comparisons,
    }
    logger.info("gate decision: %s", decision)
    return decision


def _compare_rankers(
    candidate,
    champion,
) -> dict[str, float] | None:
    """Compare the candidate's learned ranker with the incumbent's.

    Returns None when the candidate does not use a learned ranker — there is
    nothing to compare then.
    """
    if candidate is None or not candidate.ranker_metrics:
        return None

    candidate_score = candidate.ranker_metrics.get("pairwise_accuracy")
    if candidate_score is None:
        return None

    comparison: dict[str, float] = {"candidate": candidate_score}
    baseline = candidate.ranker_metrics.get("baseline_pairwise_accuracy")
    if baseline is not None:
        comparison["baseline"] = baseline
        comparison["delta_vs_baseline"] = round(candidate_score - baseline, 4)
        if candidate_score < baseline:
            comparison["regression"] = 1.0

    champion_score = ((champion.ranker_metrics if champion else None) or {}).get(
        "pairwise_accuracy"
    )
    if champion_score is not None:
        comparison["champion"] = champion_score
        comparison["delta"] = round(candidate_score - champion_score, 4)
        if candidate_score < champion_score:
            comparison["regression"] = 1.0
    return comparison


def _promote(context: AssetContext) -> dict[str, Any]:
    """Move the champion alias to the candidate, if the gate allowed it."""
    platform = context.platform
    decision = context.result("gate") or {}

    if not decision.get("promote"):
        reason = decision.get("reason") or "gate did not pass"
        if not decision.get("candidate_version"):
            reason = "no candidate is registered as challenger"
        return {"promoted": False, "reason": reason}

    version = decision.get("candidate_version")
    if not version:
        return {"promoted": False, "reason": "no candidate version"}

    previous = decision.get("champion_version")
    platform.promote(platform.settings.champion_alias, version)

    # Keep the previous champion reachable for a rollback.
    if previous:
        platform.promote(platform.settings.challenger_alias, previous)

    logger.info("promoted %s -> %s", platform.settings.champion_alias, version)
    return {"promoted": True, "version": version, "previous": previous}


def _monitor(context: AssetContext) -> dict[str, Any]:
    """Compute drift over the request log."""
    platform = context.platform
    records = platform.log_store.fetch(limit=5000)
    if len(records) < 20:
        return {"is_drifted": False, "reason": "not enough samples", "samples": len(records)}

    split = max(1, int(len(records) * 0.7))
    report = compute_drift(
        records[:split],
        records[split:],
        threshold=platform.settings.drift_psi_threshold,
    )
    return report.to_dict()


def _alert(context: AssetContext) -> dict[str, Any]:
    """Send an alert when drift was detected."""
    platform = context.platform
    monitoring = context.result("monitor") or {}
    if not monitoring.get("is_drifted"):
        return {"sent": False, "reason": "no drift"}

    from ragllmops.monitor.alerts import build_notifier

    notifier = build_notifier(platform.settings)
    drifted = ", ".join(monitoring.get("drifted_features", []))
    delivered = notifier.send(
        title="RAG drift detected",
        message=(
            f"drifted features: {drifted}\n"
            f"reference={monitoring.get('reference_size')} "
            f"current={monitoring.get('current_size')} requests"
        ),
        severity=SEVERITY_CRITICAL if len(drifted) > 2 else SEVERITY_WARNING,
    )
    return {"sent": delivered, "drifted_features": monitoring.get("drifted_features", [])}


# ----------------------------------------------------------------------
def build_pipeline_dag() -> DAG:
    """Build the platform's pipeline DAG."""
    return DAG(
        name="ragllmops-pipeline",
        assets=[
            Asset(
                name="ingest",
                fn=_ingest,
                deps=(),
                retries=1,
                tags=("data",),
                description="Rebuild the retrieval index from the corpus.",
            ),
            Asset(
                name="materialize_features",
                fn=_materialize_features,
                deps=("ingest",),
                tags=("features",),
                description="Recompute online features from the request log.",
            ),
            Asset(
                name="evaluate",
                fn=_evaluate,
                deps=("materialize_features",),
                retries=1,
                tags=("quality",),
                description="Run the golden set against the candidate bundle.",
            ),
            Asset(
                name="train_reranker",
                fn=_train_reranker,
                deps=("evaluate",),
                tags=("learning",),
                description=(
                    "Train a ranker on the production feedback and register it "
                    "as the challenger."
                ),
            ),
            Asset(
                name="gate",
                fn=_gate,
                deps=("train_reranker",),
                tags=("quality",),
                description="Compare the candidate with the incumbent.",
            ),
            Asset(
                name="promote",
                fn=_promote,
                deps=("gate",),
                tags=("release",),
                description="Promote the candidate to champion when the gate passes.",
            ),
            Asset(
                name="monitor",
                fn=_monitor,
                deps=(),
                schedule="@every 5m",
                tags=("monitoring",),
                description="Detect drift in the request log.",
            ),
            Asset(
                name="alert",
                fn=_alert,
                deps=("monitor",),
                tags=("monitoring",),
                description="Alert when drift is detected.",
            ),
        ],
    )


__all__ = [
    "build_pipeline_dag",
    "_alert",
    "_evaluate",
    "_gate",
    "_ingest",
    "_materialize_features",
    "_monitor",
    "_promote",
    "_train_reranker",
]
