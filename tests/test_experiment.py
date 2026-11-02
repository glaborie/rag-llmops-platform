"""Canary statistics.

A traffic split says nothing about which side is better. These tests check
that the comparison says the right thing — and, just as importantly, that it
refuses to say anything when there is not enough traffic.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ragllmops.models import RequestRecord
from ragllmops.monitor.experiment import (
    bootstrap_p_value,
    canary_report,
    compare_refusal_rates,
    compare_scores,
    two_proportion_p_value,
)


def _record(
    request_id: str,
    *,
    bundle_version: str,
    refused: bool = False,
    top_score: float = 0.5,
) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        ts=datetime.now(UTC),
        query="q",
        answer="a",
        bundle_version=bundle_version,
        prompt_id="v2",
        llm_model="local-extractive-v1",
        llm_provider="extractive",
        latency_ms=1.0,
        top_score=top_score,
        chunk_count=1,
        prompt_tokens=1,
        completion_tokens=1,
        answer_chars=1,
        refused=refused,
        retrieved_ids=["c::0"],
    )


def test_two_identical_proportions_are_not_significant():
    p_value = two_proportion_p_value(10, 100, 10, 100)
    assert p_value > 0.5


def test_a_large_difference_is_significant():
    p_value = two_proportion_p_value(50, 100, 10, 100)
    assert p_value < 0.01


def test_a_small_difference_is_not_significant():
    p_value = two_proportion_p_value(50, 100, 48, 100)
    assert p_value > 0.05


def test_the_bootstrap_finds_a_real_difference():
    champion = [0.1] * 40
    challenger = [0.9] * 40
    assert bootstrap_p_value(champion, challenger) < 0.05


def test_the_bootstrap_finds_nothing_in_noise():
    champion = [0.5, 0.51, 0.49, 0.5] * 10
    challenger = [0.5, 0.49, 0.51, 0.5] * 10
    assert bootstrap_p_value(champion, challenger) > 0.05


def test_a_comparison_without_enough_traffic_says_nothing():
    champion = [_record(f"c{i}", bundle_version="v1", refused=True) for i in range(5)]
    challenger = [_record(f"n{i}", bundle_version="v2") for i in range(5)]
    comparison = compare_refusal_rates(champion, challenger)

    assert not comparison.significant
    assert "not enough traffic" in comparison.verdict


def test_a_challenger_that_refuses_less_is_better():
    champion = [_record(f"c{i}", bundle_version="v1", refused=True) for i in range(60)]
    challenger = [_record(f"n{i}", bundle_version="v2") for i in range(60)]
    comparison = compare_refusal_rates(champion, challenger)

    assert comparison.significant
    assert comparison.delta < 0
    assert "significantly better" in comparison.verdict


def test_a_challenger_that_refuses_more_is_worse():
    champion = [_record(f"c{i}", bundle_version="v1") for i in range(60)]
    challenger = [_record(f"n{i}", bundle_version="v2", refused=True) for i in range(60)]
    comparison = compare_refusal_rates(champion, challenger)

    assert comparison.significant
    assert comparison.delta > 0
    assert "significantly worse" in comparison.verdict


def test_the_score_comparison_uses_the_traffic_of_both_sides():
    champion = [_record(f"c{i}", bundle_version="v1", top_score=0.2) for i in range(60)]
    challenger = [_record(f"n{i}", bundle_version="v2", top_score=0.8) for i in range(60)]
    comparison = compare_scores(champion, challenger)

    assert comparison.significant
    assert comparison.delta > 0


def test_the_report_needs_a_champion_and_a_challenger():
    report = canary_report([], None, None)
    assert report["ready"] is False


def test_the_report_compares_the_two_sides():
    records = [
        *[_record(f"c{i}", bundle_version="v1", refused=True) for i in range(60)],
        *[_record(f"n{i}", bundle_version="v2") for i in range(60)],
    ]
    report = canary_report(records, "v1", "v2")

    assert report["ready"] is True
    assert report["champion_requests"] == 60
    assert report["challenger_requests"] == 60
    assert report["significant"] is True
    assert report["promotable"] is True


def test_a_report_without_traffic_is_not_promotable():
    records = [_record("c1", bundle_version="v1")]
    report = canary_report(records, "v1", "v2")
    assert report["promotable"] is False
