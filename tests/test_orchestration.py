"""The DAG engine, lineage and the scheduler."""

import pytest

from ragllmops.orchestration.dag import DAG, Asset, AssetContext
from ragllmops.orchestration.engine import Engine
from ragllmops.orchestration.lineage import LineageStore
from ragllmops.orchestration.scheduler import Scheduler, parse_schedule


def _asset(name: str, deps: tuple[str, ...] = (), **kwargs) -> Asset:
    return Asset(name=name, fn=kwargs.pop("fn", lambda ctx: name), deps=deps, **kwargs)


def test_assets_run_in_dependency_order():
    order: list[str] = []
    dag = DAG(
        "test",
        [
            _asset("c", deps=("b",), fn=lambda ctx: order.append("c")),
            _asset("a", fn=lambda ctx: order.append("a")),
            _asset("b", deps=("a",), fn=lambda ctx: order.append("b")),
        ],
    )
    Engine(dag).run()
    assert order == ["a", "b", "c"]


def test_dag_rejects_unknown_dependencies():
    with pytest.raises(ValueError, match="unknown asset"):
        DAG("test", [_asset("a", deps=("missing",))])


def test_dag_detects_cycles():
    with pytest.raises(ValueError, match="cycle detected"):
        DAG(
            "test",
            [
                _asset("a", deps=("b",)),
                _asset("b", deps=("a",)),
            ],
        )


def test_dag_upstream_and_downstream():
    dag = DAG("test", [_asset("a"), _asset("b", deps=("a",)), _asset("c", deps=("b",))])
    assert dag.upstream_of("c") == {"a", "b"}
    assert dag.downstream_of("a") == {"b", "c"}


def test_engine_caches_unchanged_assets(tmp_path):
    calls = {"count": 0}

    def expensive(ctx):  # noqa: ANN001, ANN202
        calls["count"] += 1
        return {"value": 42}

    dag = DAG("test", [_asset("a", fn=expensive)])
    engine = Engine(dag, cache_dir=tmp_path / "cache")

    first = engine.run()
    second = engine.run()

    assert calls["count"] == 1
    assert first.runs[0].status == "success"
    assert second.runs[0].status == "cached"


def test_engine_force_ignores_the_cache(tmp_path):
    calls = {"count": 0}

    def expensive(ctx):  # noqa: ANN001, ANN202
        calls["count"] += 1
        return {"value": 42}

    dag = DAG("test", [_asset("a", fn=expensive)])
    engine = Engine(dag, cache_dir=tmp_path / "cache")
    engine.run()
    engine.run(force=True)
    assert calls["count"] == 2


def test_engine_retries_a_failing_asset():
    attempts = {"count": 0}

    def flaky(ctx):  # noqa: ANN001, ANN202
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")
        return "ok"

    dag = DAG("test", [_asset("a", fn=flaky, retries=3, retry_delay_seconds=0.01)])
    summary = Engine(dag).run()
    assert summary.success
    assert attempts["count"] == 3


def test_engine_skips_dependents_of_a_failed_asset():
    ran = []

    def boom(ctx):  # noqa: ANN001, ANN202
        raise RuntimeError("nope")

    dag = DAG(
        "test",
        [
            _asset("a", fn=boom),
            _asset("b", deps=("a",), fn=lambda ctx: ran.append("b")),
        ],
    )
    summary = Engine(dag).run()
    assert not summary.success
    assert ran == []
    statuses = {run.asset: run.status for run in summary.runs}
    assert statuses["a"] == "failed"
    assert statuses["b"] == "skipped"


def test_engine_records_lineage(tmp_path):
    lineage = LineageStore(tmp_path / "runs")
    dag = DAG("test", [_asset("a"), _asset("b", deps=("a",))])
    Engine(dag, lineage=lineage).run()

    assert lineage.last_run("a") is not None
    assert lineage.upstream_of("b") == {"a"}
    assert lineage.downstream_of("a") == {"b"}
    assert lineage.summary()["runs"] == 2


def test_parse_schedule():
    assert parse_schedule("@every 30s") == 30
    assert parse_schedule("@every 15m") == 900
    assert parse_schedule("@every 2h") == 7200
    assert parse_schedule(None) is None
    assert parse_schedule("every day") is None


def test_scheduler_runs_only_due_assets():
    ran: list[str] = []
    dag = DAG(
        "test",
        [
            _asset("a", fn=lambda ctx: ran.append("a"), schedule="@every 1m"),
            _asset("b", fn=lambda ctx: ran.append("b")),
        ],
    )
    engine = Engine(dag)
    scheduler = Scheduler(dag=dag, engine=engine)

    summary = scheduler.tick()
    assert summary is not None
    assert ran == ["a"]

    # Nothing is due immediately afterwards.
    assert scheduler.tick() is None


def test_asset_context_exposes_upstream_results():
    seen = {}

    def capture(ctx: AssetContext):  # noqa: ANN202
        seen["value"] = ctx.result("a")
        return None

    dag = DAG("test", [_asset("a", fn=lambda ctx: "value"), _asset("b", deps=("a",), fn=capture)])
    Engine(dag).run()
    assert seen["value"] == "value"
