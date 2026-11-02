"""The HTTP API."""

import pytest
from fastapi.testclient import TestClient

from ragllmops.serve.app import create_app
from ragllmops.serve.canary import CanaryRouter
from ragllmops.serve.watcher import AliasWatcher


@pytest.fixture
def client(platform):
    """A TestClient over a platform with a registered champion."""
    bundle = platform.current_bundle()
    platform.register_bundle(bundle)
    platform.promote(platform.settings.champion_alias, bundle.version)
    with TestClient(create_app(platform)) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["index"]["chunks"] > 0
    assert payload["registry"]["aliases"]


def test_ask_returns_an_answer_with_citations(client):
    response = client.post("/ask", json={"query": "What CPU target does the autoscaler use?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["citations"]
    assert payload["citations"][0]["doc_id"] == "compute"
    assert payload["route"] == "champion"
    assert payload["latency_ms"] >= 0


def test_ask_rejects_an_empty_query(client):
    assert client.post("/ask", json={"query": ""}).status_code == 422


def test_ask_writes_to_the_request_log(client, platform):
    client.post("/ask", json={"query": "What CPU target does the autoscaler use?"})
    assert platform.log_store.count() == 1


def test_metrics_are_exposed(client):
    client.post("/ask", json={"query": "What CPU target does the autoscaler use?"})
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "ragllmops_requests_total" in body
    assert "ragllmops_request_duration_seconds" in body


def test_registry_admin_endpoint(client):
    response = client.get("/admin/registry")
    assert response.status_code == 200
    payload = response.json()
    assert payload["aliases"]
    assert payload["versions"]


def test_promote_unknown_version_is_a_404(client):
    response = client.post("/admin/promote", json={"alias": "champion", "version": "nope"})
    assert response.status_code == 404


def test_drift_endpoint(client):
    response = client.get("/admin/drift")
    assert response.status_code == 200
    assert "is_drifted" in response.json()


def test_evaluate_endpoint(client):
    response = client.post("/admin/evaluate")
    assert response.status_code == 200
    assert "aggregates" in response.json()


def test_canary_endpoint_reports_the_comparison(client, platform):
    """The canary report is an experiment, not just a traffic split."""
    bundle = platform.current_bundle()
    platform.register_bundle(bundle)
    platform.promote("challenger", bundle.version)

    response = client.get("/admin/canary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["champion"]
    assert payload["challenger"]
    assert "comparisons" in payload
    # Not enough traffic to conclude anything: the report must say so.
    assert payload["promotable"] is False


# ----------------------------------------------------------------------
# canary routing
# ----------------------------------------------------------------------
def test_canary_router_sends_everything_to_the_champion_without_traffic():
    champion = _bundle("v1")
    router = CanaryRouter(champion, _bundle("v2"), traffic_pct=0)
    assert all(router.select(f"req-{i}").version == "v1" for i in range(50))
    assert not router.active


def test_canary_router_splits_traffic_deterministically():
    champion = _bundle("v1")
    challenger = _bundle("v2")
    router = CanaryRouter(champion, challenger, traffic_pct=50)

    first = router.select("stable-request-id")
    second = router.select("stable-request-id")
    assert first.version == second.version

    routed = [router.route_of(f"req-{i}") for i in range(200)]
    assert "challenger" in routed
    assert "champion" in routed
    share = routed.count("challenger") / len(routed)
    assert 0.3 < share < 0.7


def test_alias_watcher_detects_a_promotion(platform):
    bundle = platform.current_bundle()
    platform.register_bundle(bundle)
    platform.promote("champion", bundle.version)

    watcher = AliasWatcher(platform)
    watcher.poll()  # first poll records the current state
    assert watcher.changes == []

    platform.promote("challenger", bundle.version)
    assert watcher.poll()
    assert watcher.changes


def test_alias_watcher_sees_the_first_alias_of_an_empty_registry(platform):
    """An empty registry must not swallow the first promotion."""
    bundle = platform.current_bundle()
    platform.register_bundle(bundle)

    watcher = AliasWatcher(platform)
    assert watcher.poll() is False  # first poll: the registry is still empty

    platform.promote("champion", bundle.version)
    assert watcher.poll()
    assert watcher.changes[0]["alias"] == "champion"


def test_alias_watcher_notifies_the_serving_layer(platform):
    """A promotion made outside the API has to reach the traffic."""
    bundle = platform.current_bundle()
    platform.register_bundle(bundle)
    platform.promote("champion", bundle.version)

    refreshed: list[int] = []
    watcher = AliasWatcher(platform, on_change=lambda: refreshed.append(1))
    watcher.poll()

    platform.promote("challenger", bundle.version)
    assert watcher.poll()
    assert refreshed == [1]


def test_a_fresh_stack_boots_with_a_champion(platform):
    """An empty registry must not leave the API unable to answer."""
    with TestClient(create_app(platform)) as client:
        response = client.post(
            "/ask", json={"query": "What CPU target does the autoscaler use?"}
        )
        assert response.status_code == 200
        assert response.json()["route"] == "champion"


def _bundle(version: str):
    from dataclasses import replace

    from ragllmops.rag.bundle import BundleBuilder

    return replace(BundleBuilder().build(), version=version)
