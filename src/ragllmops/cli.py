"""Command line interface.

    python -m ragllmops.cli doctor          # what is wired up, what is missing
    python -m ragllmops.cli ingest          # build the retrieval index
    python -m ragllmops.cli evaluate        # run the golden set + quality gate
    python -m ragllmops.cli serve           # run the API
    python -m ragllmops.cli monitor         # drift report over the request log
    python -m ragllmops.cli loop            # the full closed loop
    python -m ragllmops.cli pipeline run    # run the DAG once
    python -m ragllmops.cli registry list   # versions and aliases
"""

from __future__ import annotations

import argparse
import json
import sys

from ragllmops.config import get_settings
from ragllmops.feedback.dataset import summarise
from ragllmops.logging_utils import configure_logging, get_logger
from ragllmops.monitor.drift import compute_drift, split_windows
from ragllmops.platform import Platform

logger = get_logger("cli")


# ----------------------------------------------------------------------
def _platform() -> Platform:
    return Platform()


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what is available and what is missing."""
    settings = get_settings()
    platform = _platform()
    health = platform.health()

    print("== ragllmops doctor ==")
    print(f"project root    : {settings.project_root}")
    print(f"llm             : {settings.llm_provider} / {settings.llm_model}")
    print(f"embedder        : {settings.embedder_backend}")
    print(f"vector store    : {settings.vector_store}")
    print(f"registry        : {settings.registry_backend}")
    print(f"log store       : {settings.log_store_backend}")
    print(f"feature store   : {settings.feature_backend}")
    print()

    try:
        platform.load_index()
        print(f"index           : {len(platform.catalog)} chunks loaded")
    except Exception as exc:
        print(f"index           : FAILED ({exc})")

    golden = settings.eval_file
    print(f"golden set      : {golden if golden.exists() else 'MISSING'}")
    print(f"corpus          : {settings.corpus_path}")
    print()

    print("== health ==")
    print(json.dumps(health, indent=2, default=str))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Build the retrieval index."""
    platform = _platform()
    result = platform.reindex(reset=not args.keep)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run the golden set and the quality gate."""
    platform = _platform()
    platform.load_index()
    report = platform.evaluate(use_judge=args.judge)
    print(report.summary())

    if args.judge and "judge_grounded_rate" not in report.aggregates:
        print(
            "\nnote: the judge returned no verdict. The extractive provider "
            "cannot grade — set RAG_LLM_PROVIDER=openai or ollama to use it.",
            file=sys.stderr,
        )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        failures = [case for case in report.cases if not case.passed]
        if failures:
            print("\nfailed cases:")
            for case in failures:
                print(
                    f"  {case.case_id}: {case.question}\n"
                    f"    expected {case.expected_doc_ids}, got {case.retrieved_doc_ids}\n"
                    f"    missing terms: {case.missing_key_terms or 'none'}"
                )

    return 0 if report.passed else 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the API."""
    import uvicorn

    print(f"starting the API on http://{args.host}:{args.port}")
    uvicorn.run(
        "ragllmops.serve.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    """Drift report over the request log."""
    platform = _platform()
    records = platform.log_store.fetch(limit=args.limit)
    reference, current = split_windows(records, current_fraction=args.window)
    report = compute_drift(
        reference, current, threshold=platform.settings.drift_psi_threshold
    )
    print(report.summary())
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    return 1 if report.is_drifted else 0


def cmd_loop(args: argparse.Namespace) -> int:
    """Run the closed loop."""
    from ragllmops.orchestration.closed_loop import run_closed_loop

    platform = _platform()
    platform.load_index()
    result = run_closed_loop(platform)
    print(result.summary())
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Run the platform pipeline."""
    from ragllmops.orchestration.engine import Engine
    from ragllmops.orchestration.lineage import LineageStore
    from ragllmops.orchestration.pipeline import build_pipeline_dag

    settings = get_settings()
    platform = _platform()
    dag = build_pipeline_dag()
    engine = Engine(
        dag=dag,
        platform=platform,
        cache_dir=settings.data_path / "cache",
        lineage=LineageStore(settings.data_path / "runs"),
    )
    summary = engine.run(only=args.assets or None, force=args.force)
    print(summary.summary())
    return 0 if summary.success else 1


def cmd_registry(args: argparse.Namespace) -> int:
    """Inspect or move registry aliases."""
    platform = _platform()
    registry = platform.registry

    if args.action == "list":
        print("aliases:")
        for alias, version in registry.aliases().items():
            print(f"  {alias:<12} -> {version}")
        print("versions:")
        for version in registry.list_versions():
            bundle = registry.get(version)
            if bundle is None:
                continue
            print(
                f"  {version:<14} prompt={bundle.prompt_id} "
                f"embedder={bundle.embedder_id or 'unknown'} "
                f"reranker={bundle.reranker_id or 'unknown'}"
            )
        return 0

    if args.action == "show":
        bundle = registry.get(args.version) or registry.get_by_alias(args.version)
        if bundle is None:
            print(f"no such version or alias: {args.version}", file=sys.stderr)
            return 1
        print(json.dumps(bundle.to_dict(), indent=2, default=str))
        return 0

    if args.action == "promote":
        try:
            platform.promote(args.version, args.target)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"alias {args.version} -> {args.target}")
        return 0

    if args.action == "register":
        bundle = platform.build_bundle(
            prompt_id=args.prompt or "v2", notes=args.notes or "registered via CLI"
        )
        platform.register_bundle(bundle)
        print(f"registered {bundle.version}")
        print(f"  embedder={bundle.embedder_id} dim={bundle.embedding_dim}")
        print(f"  reranker={bundle.reranker_id}")
        return 0

    return 1


def cmd_reset(args: argparse.Namespace) -> int:
    """Clear the local state: registry, request log, index and reports."""
    settings = get_settings()
    targets = [
        settings.registry_file,
        settings.sqlite_file,
        settings.features_online_file,
        settings.data_path / "index" / "catalog.json",
    ]
    removed: list[str] = []
    for target in targets:
        if target.exists():
            target.unlink()
            removed.append(str(target.relative_to(settings.project_root)))

    reports = settings.reports_path
    if reports.exists():
        for report in reports.glob("incident-*.md"):
            report.unlink()
            removed.append(str(report.relative_to(settings.project_root)))

    print(f"removed {len(removed)} files")
    for item in removed:
        print(f"  {item}")
    return 0


# ----------------------------------------------------------------------
def cmd_feedback(args: argparse.Namespace) -> int:
    """Show the judgement set the request log has produced."""
    platform = _platform()
    judgements = platform.build_judgements()
    summary = summarise(judgements)

    print("== judgement set ==")
    print(f"examples           : {summary['examples']}")
    print(f"requests           : {summary['requests']}")
    print(f"positives/negatives: {summary['positives']}/{summary['negatives']}")
    print(
        f"explicit/sampled/inferred: {summary['explicit']}/"
        f"{summary['sampled']}/{summary['implicit']}"
    )
    print("by source:")
    for source, count in summary["by_source"].items():
        print(f"  {source:<28} {count}")
    print("by split:")
    for split, count in summary["by_split"].items():
        print(f"  {split:<28} {count}")

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_train_ranker(args: argparse.Namespace) -> int:
    """Train a ranker on the feedback and store it as an artifact."""
    platform = _platform()
    report = platform.train_ranker_from_feedback(min_examples=args.min_examples)

    if not report.get("trained"):
        print(f"no ranker trained: {report.get('reason')}")
        return 1

    validation = report["validation_metrics"]
    baseline = report["baseline_metrics"]
    print("== trained ranker ==")
    print(f"artifact      : {report['artifact_id']}")
    print(
        f"examples      : {report['examples']} "
        f"(train {report['train_examples']}, validation {report['validation_examples']})"
    )
    print(f"held-out pairwise accuracy : {validation['pairwise_accuracy']:.3f}")
    print(f"rule-based baseline        : {baseline['pairwise_accuracy']:.3f}")
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_canary(args: argparse.Namespace) -> int:
    """Compare the champion and the challenger on the traffic they served."""
    from ragllmops.monitor.experiment import canary_report

    platform = _platform()
    records = platform.log_store.fetch(limit=args.limit)
    champion = platform.champion_bundle()
    challenger = platform.challenger_bundle()
    report = canary_report(
        records,
        champion.version if champion else None,
        challenger.version if challenger else None,
        min_samples=args.min_samples,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """Materialise the online features and check offline/online parity."""
    platform = _platform()
    platform.load_index()
    features = platform.materialize_features()

    from ragllmops.features import offline_features_path

    offline = json.loads(offline_features_path().read_text(encoding="utf-8")).get("features", {})
    online = platform.feature_provider.get_online_features(set(offline))

    mismatches = {
        key: {"offline": offline.get(key), "online": online.get(key)}
        for key in offline
        if online.get(key) != offline.get(key)
    }

    print(f"materialised {len(features)} entities")
    if mismatches:
        print(f"PARITY FAILURE: {len(mismatches)} mismatches")
        print(json.dumps(mismatches, indent=2, default=str))
        return 1
    print("offline/online parity: OK")
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="ragllmops", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check the wiring and services")
    doctor.set_defaults(func=cmd_doctor)

    ingest = sub.add_parser("ingest", help="build the retrieval index")
    ingest.add_argument("--keep", action="store_true", help="do not clear the store first")
    ingest.set_defaults(func=cmd_ingest)

    evaluate = sub.add_parser("evaluate", help="run the golden set and the quality gate")
    evaluate.add_argument("--json", action="store_true")
    evaluate.add_argument("--judge", action="store_true", help="also run the LLM judge")
    evaluate.set_defaults(func=cmd_evaluate)

    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8001)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    monitor = sub.add_parser("monitor", help="drift report")
    monitor.add_argument("--limit", type=int, default=5000)
    monitor.add_argument("--window", type=float, default=0.3, help="current window fraction")
    monitor.add_argument("--json", action="store_true")
    monitor.set_defaults(func=cmd_monitor)

    loop = sub.add_parser("loop", help="run the closed loop (detect -> gate -> promote)")
    loop.set_defaults(func=cmd_loop)

    pipeline = sub.add_parser("pipeline", help="run the platform pipeline")
    pipeline.add_argument("action", choices=["run", "describe"], default="run", nargs="?")
    pipeline.add_argument("--assets", nargs="*", help="only run these assets")
    pipeline.add_argument("--force", action="store_true", help="ignore the cache")
    pipeline.set_defaults(func=cmd_pipeline)

    registry = sub.add_parser("registry", help="inspect the model registry")
    registry.add_argument("action", choices=["list", "show", "promote", "register"])
    registry.add_argument("version", nargs="?", help="version or alias")
    registry.add_argument("--target", help="target version (promote)")
    registry.add_argument("--prompt", help="prompt id (register)")
    registry.add_argument("--notes", help="notes (register)")
    registry.set_defaults(func=cmd_registry)

    features = sub.add_parser("features", help="materialise features and check parity")
    features.set_defaults(func=cmd_features)

    feedback = sub.add_parser("feedback", help="show the judgement set from the request log")
    feedback.add_argument("--json", action="store_true")
    feedback.set_defaults(func=cmd_feedback)

    ranker = sub.add_parser("train-ranker", help="train a ranker on the feedback")
    ranker.add_argument("--min-examples", type=int, default=20)
    ranker.add_argument("--json", action="store_true")
    ranker.set_defaults(func=cmd_train_ranker)

    canary = sub.add_parser("canary", help="compare the champion and the challenger")
    canary.add_argument("--limit", type=int, default=50000)
    canary.add_argument("--min-samples", type=int, default=30)
    canary.set_defaults(func=cmd_canary)

    reset = sub.add_parser("reset", help="clear the local state (registry, logs, index)")
    reset.set_defaults(func=cmd_reset)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if getattr(args, "action", None) == "describe" and args.command == "pipeline":
        from ragllmops.orchestration.pipeline import build_pipeline_dag

        print(json.dumps(build_pipeline_dag().describe(), indent=2))
        return 0

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.exception("command failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
