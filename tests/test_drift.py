"""Drift detection."""

from datetime import UTC, datetime, timedelta

from ragllmops.models import RequestRecord
from ragllmops.monitor.drift import (
    compute_drift,
    population_stability_index,
    split_windows,
    total_variation_distance,
)


def _record(index: int, latency: float, score: float, ts: datetime) -> RequestRecord:
    return RequestRecord(
        request_id=f"r{index}",
        ts=ts,
        query="q",
        answer="a",
        bundle_version="v1",
        prompt_id="v2",
        llm_model="m",
        llm_provider="extractive",
        latency_ms=latency,
        top_score=score,
        chunk_count=2,
        prompt_tokens=10,
        completion_tokens=20,
        answer_chars=30,
        refused=False,
        retrieved_ids=["compute::0"],
    )


def _window(count: int, latency: float, score: float, start: int = 0) -> list[RequestRecord]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        _record(start + i, latency, score, base + timedelta(seconds=start + i))
        for i in range(count)
    ]


def test_psi_is_zero_for_identical_distributions():
    values = [float(i % 10) for i in range(200)]
    assert population_stability_index(values, values) == 0.0


def test_psi_grows_with_a_distribution_shift():
    reference = [float(i % 10) for i in range(200)]
    current = [float(i % 10) + 50 for i in range(200)]
    assert population_stability_index(reference, current) > 0.2


def test_psi_handles_empty_input():
    assert population_stability_index([], [1.0]) == 0.0
    assert population_stability_index([1.0], []) == 0.0


def test_total_variation_distance_detects_a_categorical_shift():
    reference = ["a"] * 100
    current = ["b"] * 100
    distance, distribution = total_variation_distance(reference, current)
    assert distance == 1.0
    assert distribution["b"] == 1.0


def test_compute_drift_flags_a_shift():
    reference = _window(100, latency=10.0, score=0.8)
    current = _window(60, latency=90.0, score=0.2, start=100)
    report = compute_drift(reference, current, threshold=0.2)
    assert report.is_drifted
    assert "latency_ms" in report.drifted_features
    assert report.reference_size == 100
    assert report.current_size == 60


def test_compute_drift_is_quiet_when_nothing_changed():
    reference = _window(100, latency=10.0, score=0.8)
    current = _window(60, latency=10.0, score=0.8, start=100)
    report = compute_drift(reference, current, threshold=0.2)
    assert not report.is_drifted
    assert report.drifted_features == []


def test_compute_drift_without_samples_is_not_drifted():
    report = compute_drift([], [], threshold=0.2)
    assert not report.is_drifted


def test_split_windows_keeps_the_newest_tail():
    records = _window(100, latency=1.0, score=1.0)
    reference, current = split_windows(records, current_fraction=0.3)
    assert len(reference) == 70
    assert len(current) == 30
    assert current[-1].request_id == records[-1].request_id


def test_drift_report_serialises():
    reference = _window(50, latency=10.0, score=0.8)
    current = _window(50, latency=90.0, score=0.2, start=50)
    payload = compute_drift(reference, current).to_dict()
    assert payload["is_drifted"] is True
    assert payload["features"]
    assert "summary" not in payload
