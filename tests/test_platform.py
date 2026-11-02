"""The platform façade and the CLI."""

from ragllmops.cli import build_parser
from ragllmops.platform import Platform


def test_platform_health_snapshot(platform):
    health = platform.health()
    assert health["status"] == "ok"
    assert health["index"]["chunks"] > 0
    assert health["index"]["documents"] == 2
    assert health["log_store"]["backend"] == "sqlite"


def test_platform_reindex_is_idempotent(platform):
    first = platform.reindex()
    second = platform.reindex()
    assert first.chunks == second.chunks
    assert len(platform.catalog) == first.chunks


def test_platform_load_index_restores_from_disk(platform, settings):
    platform.reindex()
    fresh = Platform(
        settings=settings,
        embedder=platform.embedder,
        vector_store=platform.vector_store,
        llm=platform.llm,
        registry=platform.registry,
        log_store=platform.log_store,
        feature_provider=platform.feature_provider,
    )
    assert fresh.load_index() == len(platform.catalog)


def test_platform_registers_and_resolves_aliases(platform):
    bundle = platform.current_bundle()
    platform.register_bundle(bundle)
    platform.promote("champion", bundle.version)
    assert platform.champion_bundle().version == bundle.version
    assert platform.current_bundle().version == bundle.version


def test_platform_evaluate_runs_the_gate(platform):
    report = platform.evaluate()
    assert report.aggregates["faithfulness"] > 0
    assert report.passed


def test_platform_answer_logs(platform):
    platform.answer("What is the default object retention period?")
    records = platform.log_store.fetch()
    assert records
    assert records[-1].retrieved_ids


def test_cli_parser_exposes_the_commands():
    parser = build_parser()
    for command in ("doctor", "ingest", "evaluate", "serve", "monitor", "loop", "registry"):
        args = parser.parse_args([command] if command != "registry" else ["registry", "list"])
        assert args.command == command
