"""The learning loop: capture judgements, build a dataset, train a ranker.

These tests are the plumbing test. They do not claim the learned ranker is
good — that needs real traffic — they claim the machinery is real: a verdict
reaches the request log, a verdict becomes a labelled example with a known
provenance, the examples are split without leaking, and a model trained on
them is stored as an auditable artifact.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from ragllmops.feedback.dataset import TRAIN, VALIDATION, build_judgements, summarise
from ragllmops.feedback.labels import Verdict, decode_verdict, encode_verdict
from ragllmops.feedback.model import (
    TrainingExample,
    evaluate_model,
    load_ranker,
    save_ranker,
    train_ranker,
    training_data_hash,
)
from ragllmops.models import RequestRecord
from ragllmops.serve.app import create_app


def _record(
    request_id: str = "r1",
    *,
    retrieved: list[str] | None = None,
    feedback: int | None = None,
    chunk_id: str | None = None,
    refused: bool = False,
    query: str = "What CPU target does the autoscaler use?",
) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        ts=datetime.now(UTC),
        query=query,
        answer="65% CPU",
        bundle_version="v1",
        prompt_id="v2",
        llm_model="local-extractive-v1",
        llm_provider="extractive",
        latency_ms=1.0,
        top_score=0.8,
        chunk_count=len(retrieved or []),
        prompt_tokens=10,
        completion_tokens=5,
        answer_chars=6,
        refused=refused,
        retrieved_ids=list(retrieved or ["compute::0", "storage::0"]),
        feedback=feedback,
        feedback_chunk_id=chunk_id,
    )


# ----------------------------------------------------------------------
# the verdict vocabulary
# ----------------------------------------------------------------------
def test_verdicts_round_trip_through_the_log_encoding():
    for verdict in Verdict:
        assert decode_verdict(encode_verdict(verdict.value)) is verdict


def test_an_unknown_verdict_is_rejected():
    with pytest.raises(ValueError):
        encode_verdict("looks_fine_to_me")


# ----------------------------------------------------------------------
# M1 — capture
# ----------------------------------------------------------------------
def test_feedback_endpoint_writes_to_the_request_log(client, platform):
    asked = client.post("/ask", json={"query": "What CPU target does the autoscaler use?"})
    request_id = asked.json()["request_id"]

    response = client.post("/feedback", json={"request_id": request_id, "verdict": "helpful"})
    assert response.status_code == 200
    assert response.json()["recorded"] is True

    record = platform.log_store.get(request_id)
    assert record is not None
    assert record.feedback == encode_verdict("helpful")


def test_feedback_endpoint_rejects_an_unknown_request(client):
    response = client.post("/feedback", json={"request_id": "nope", "verdict": "helpful"})
    assert response.status_code == 404


def test_feedback_endpoint_rejects_a_chunk_that_was_not_served(client, platform):
    asked = client.post("/ask", json={"query": "What CPU target does the autoscaler use?"})
    request_id = asked.json()["request_id"]

    response = client.post(
        "/feedback",
        json={"request_id": request_id, "verdict": "wrong_citation", "chunk_id": "nope::9"},
    )
    assert response.status_code == 400


def test_feedback_endpoint_needs_a_chunk_for_a_wrong_citation(client):
    asked = client.post("/ask", json={"query": "What CPU target does the autoscaler use?"})
    request_id = asked.json()["request_id"]

    response = client.post(
        "/feedback", json={"request_id": request_id, "verdict": "wrong_citation"}
    )
    assert response.status_code == 400


def test_feedback_endpoint_rejects_an_unknown_verdict(client):
    asked = client.post("/ask", json={"query": "What CPU target does the autoscaler use?"})
    response = client.post(
        "/feedback", json={"request_id": asked.json()["request_id"], "verdict": "meh"}
    )
    assert response.status_code == 422


def test_the_platform_records_the_verdict(platform):
    platform.log_response(_fake_result("r1"))
    platform.record_feedback("r1", "unhelpful")
    assert platform.log_store.get("r1").feedback == encode_verdict("unhelpful")


def test_a_log_written_by_an_earlier_version_is_migrated(tmp_path):
    """A log on disk without the column must not stop the platform booting."""
    import sqlite3

    path = tmp_path / "requests.db"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE requests (
            request_id TEXT PRIMARY KEY, ts TEXT NOT NULL, query TEXT NOT NULL,
            answer TEXT NOT NULL, bundle_version TEXT NOT NULL, prompt_id TEXT NOT NULL,
            llm_model TEXT NOT NULL, llm_provider TEXT NOT NULL, latency_ms REAL NOT NULL,
            top_score REAL NOT NULL, chunk_count INTEGER NOT NULL, prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL, answer_chars INTEGER NOT NULL,
            refused INTEGER NOT NULL, retrieved_ids TEXT NOT NULL, user_id TEXT,
            tenant TEXT, feedback INTEGER
        )
        """
    )
    connection.commit()
    connection.close()

    from ragllmops.store.sqlite_log_store import SQLiteLogStore

    store = SQLiteLogStore(path)
    store.log(_record("r1"))
    assert store.set_feedback("r1", encode_verdict("helpful"), "compute::0")

    record = store.get("r1")
    assert record is not None
    assert record.feedback == encode_verdict("helpful")
    assert record.feedback_chunk_id == "compute::0"


def _fake_result(request_id: str):
    from ragllmops.models import Citation, RAGResponse

    response = RAGResponse(
        request_id=request_id,
        answer="65% CPU",
        citations=[
            Citation(index=1, chunk_id="compute::0", doc_id="compute", title="c", score=0.8)
        ],
        bundle_version="v1",
        prompt_id="v2",
        llm_model="local-extractive-v1",
        llm_provider="extractive",
        latency_ms=1.0,
        prompt_tokens=1,
        completion_tokens=1,
        retrieved_count=1,
        refused=False,
        query="What CPU target does the autoscaler use?",
    )

    class _Result:
        pass

    result = _Result()
    result.response = response
    return result


# ----------------------------------------------------------------------
# M2 — the judgement set
# ----------------------------------------------------------------------
def test_a_helpful_verdict_labels_the_leading_chunk():
    judgements = build_judgements([_record(feedback=encode_verdict("helpful"))])
    assert len(judgements) == 2
    leading = [j for j in judgements if j.label == 1]
    assert len(leading) == 1
    assert leading[0].chunk_id == "compute::0"
    assert leading[0].label_source == "explicit:helpful"


def test_a_helpful_verdict_ranks_the_rest_of_the_context_below():
    """A candidate list with only positives teaches nothing about the order."""
    judgements = build_judgements([_record(feedback=encode_verdict("helpful"))])
    rest = [j for j in judgements if j.label == 0]
    assert len(rest) == 1
    assert rest[0].chunk_id == "storage::0"
    assert rest[0].label_source == "explicit:context"


def test_the_cited_chunk_is_the_positive_one_not_the_first_one():
    """The answer records which chunk it used — that is the positive."""
    record = _record(feedback=encode_verdict("helpful"))
    record.answer = "Objects are retained for 90 days. [S2]"
    judgements = build_judgements([record])

    positive = [j for j in judgements if j.label == 1]
    assert [j.chunk_id for j in positive] == ["storage::0"]
    negative = [j for j in judgements if j.label == 0]
    assert [j.chunk_id for j in negative] == ["compute::0"]


def test_an_unhelpful_verdict_labels_every_served_chunk_negative():
    judgements = build_judgements([_record(feedback=encode_verdict("unhelpful"))])
    assert {j.chunk_id for j in judgements} == {"compute::0", "storage::0"}
    assert all(j.label == 0 for j in judgements)
    assert all(j.label_source == "explicit:unhelpful" for j in judgements)


def test_a_wrong_citation_labels_only_the_blamed_chunk():
    record = _record(feedback=encode_verdict("wrong_citation"), chunk_id="storage::0")
    judgements = build_judgements([record])
    assert len(judgements) == 1
    assert judgements[0].chunk_id == "storage::0"
    assert judgements[0].label == 0
    assert judgements[0].label_source == "explicit:wrong_citation"


def test_candidates_that_were_not_served_become_weak_negatives():
    """Answer-level feedback alone leaves a request with no ranking signal."""
    candidates = ["compute::0", "storage::0", "billing::0", "identity::0", "network::0"]
    judgements = build_judgements(
        [_record(feedback=encode_verdict("helpful"))],
        candidate_fn=lambda query: candidates,
    )
    sampled = [j for j in judgements if j.label_source == "sampled:candidate"]
    assert len(sampled) == 3  # capped
    assert all(j.label == 0 for j in sampled)
    assert {j.chunk_id for j in sampled} <= set(candidates[2:])


def test_the_weak_negatives_are_deterministic():
    """The same log must always produce the same training set."""
    candidates = [f"c{index}::0" for index in range(10)]
    first = build_judgements(
        [_record(feedback=encode_verdict("helpful"))],
        candidate_fn=lambda query: candidates,
    )
    second = build_judgements(
        [_record(feedback=encode_verdict("helpful"))],
        candidate_fn=lambda query: candidates,
    )
    assert [j.chunk_id for j in first] == [j.chunk_id for j in second]


def test_a_refusal_is_an_implicit_negative():
    judgements = build_judgements([_record(refused=True)])
    assert judgements
    assert all(j.label == 0 for j in judgements)
    assert all(j.label_source == "implicit:refusal" for j in judgements)


def test_an_answer_without_a_signal_produces_nothing():
    assert build_judgements([_record()]) == []


def test_explicit_feedback_wins_over_the_implicit_signal():
    record = _record(feedback=encode_verdict("helpful"), refused=True)
    judgements = build_judgements([record])
    assert len(judgements) == 2
    assert all(j.label_source.startswith("explicit:") for j in judgements)


def test_the_split_keeps_a_request_together():
    records = [
        _record(f"r{index}", feedback=encode_verdict("unhelpful")) for index in range(10)
    ]
    judgements = build_judgements(records, validation_fraction=0.3)
    by_request: dict[str, set[str]] = {}
    for judgement in judgements:
        by_request.setdefault(judgement.request_id, set()).add(judgement.split)
    assert all(len(splits) == 1 for splits in by_request.values())
    assert {j.split for j in judgements} == {TRAIN, VALIDATION}


def test_the_split_is_deterministic():
    records = [_record(f"r{index}", feedback=encode_verdict("unhelpful")) for index in range(10)]
    first = build_judgements(records, seed=7)
    second = build_judgements(records, seed=7)
    assert [(j.request_id, j.split) for j in first] == [(j.request_id, j.split) for j in second]


def test_the_summary_separates_explicit_from_implicit():
    records = [
        _record("r1", feedback=encode_verdict("helpful")),
        _record("r2", refused=True),
    ]
    summary = summarise(build_judgements(records))
    assert summary["examples"] == 4
    assert summary["explicit"] == 2
    assert summary["implicit"] == 2
    assert summary["sampled"] == 0


# ----------------------------------------------------------------------
# M3 — train a ranker
# ----------------------------------------------------------------------
def _examples(count: int = 40) -> list[TrainingExample]:
    """Examples where the term overlap separates the classes."""
    examples: list[TrainingExample] = []
    for index in range(count):
        label = index % 2
        examples.append(
            TrainingExample(
                features={
                    "bm25_score": 10.0 if label else 1.0,
                    "dense_score": 0.9 if label else 0.1,
                    "term_overlap": 0.8 if label else 0.05,
                    "doc_avg_relevance": 0.5,
                    "doc_retrieval_count": 3.0,
                    "candidate_position": 1.0 if label else 5.0,
                    "chunk_length": 200.0,
                },
                label=label,
                request_id=f"r{index // 2}",
                chunk_id=f"chunk-{index}",
            )
        )
    return examples


def test_a_trained_ranker_separates_the_classes():
    model = train_ranker(_examples())
    metrics = evaluate_model(model, _examples())
    assert metrics["accuracy"] > 0.9
    assert metrics["pairwise_accuracy"] > 0.9


def test_the_artifact_is_content_addressed(tmp_path):
    model = train_ranker(_examples())
    first = save_ranker(model, tmp_path)
    second = save_ranker(model, tmp_path)
    assert first == second
    assert (tmp_path / f"{first}.json").exists()


def test_a_stored_ranker_can_be_loaded(tmp_path):
    model = train_ranker(_examples())
    artifact_id = save_ranker(model, tmp_path, metadata={"training_data_hash": "abc"})
    loaded = load_ranker(tmp_path / f"{artifact_id}.json")
    assert loaded.feature_names == model.feature_names
    assert loaded.weights == pytest.approx(model.weights, abs=1e-6)


def test_the_training_data_hash_changes_with_the_data():
    examples = _examples()
    assert training_data_hash(examples) != training_data_hash(examples[:-1])


# ----------------------------------------------------------------------
# M4 — the platform wires it together
# ----------------------------------------------------------------------
def test_the_platform_trains_a_ranker_from_the_feedback(platform):
    # Enough records for the held-out slice to hold a comparable pair: the
    # platform refuses to ship a model it cannot judge.
    for index in range(20):
        platform.log_response(_fake_result(f"r{index}"))
        platform.record_feedback(f"r{index}", "helpful" if index % 2 else "unhelpful")

    report = platform.train_ranker_from_feedback(min_examples=2)
    assert report["trained"] is True
    assert report["artifact_id"]
    assert report["validation_metrics"]["pairs"] >= 1
    assert (platform.settings.ranker_dir / f"{report['artifact_id']}.json").exists()


def test_a_ranker_is_not_shipped_without_evidence(platform):
    """A model nobody can judge is not an improvement."""
    for index in range(2):
        platform.log_response(_fake_result(f"r{index}"))
        platform.record_feedback(f"r{index}", "unhelpful")

    report = platform.train_ranker_from_feedback(min_examples=2)
    assert report["trained"] is False
    assert "judge" in report["reason"]


def test_the_platform_refuses_to_train_without_enough_examples(platform):
    """A model fitted on three labels is noise, and the platform says so."""
    platform.log_response(_fake_result("r1"))
    platform.record_feedback("r1", "helpful")

    report = platform.train_ranker_from_feedback(min_examples=20)
    assert report["trained"] is False
    assert "at least 20" in report["reason"]


def test_the_platform_refuses_to_train_on_one_kind_of_label(platform):
    """A single class cannot teach a ranker which order is better."""
    for index in range(6):
        platform.log_response(_fake_result(f"r{index}"))
        platform.record_feedback(f"r{index}", "unhelpful")

    report = platform.train_ranker_from_feedback(min_examples=2)
    assert report["trained"] is False
    assert "one kind of label" in report["reason"]


def test_a_bundle_can_carry_the_trained_ranker(platform):
    for index in range(20):
        platform.log_response(_fake_result(f"r{index}"))
        platform.record_feedback(f"r{index}", "helpful" if index % 2 else "unhelpful")

    report = platform.train_ranker_from_feedback(min_examples=2)
    bundle = platform.build_bundle()
    from dataclasses import replace

    from ragllmops.rag.bundle import with_version

    bundle = with_version(
        replace(bundle, ranker_artifact_id=report["artifact_id"])
    )
    assert bundle.ranker_artifact_id == report["artifact_id"]

    platform.register_bundle(bundle)
    ranker = platform.ranker_for(bundle)
    assert ranker is not None
    assert ranker.name.startswith("learned:")


def test_the_gate_rejects_a_ranker_that_is_worse_than_the_rule(platform):
    from ragllmops.orchestration.pipeline import _compare_rankers

    candidate = platform.build_bundle()
    from dataclasses import replace

    candidate = replace(
        candidate,
        ranker_metrics={"pairwise_accuracy": 0.4, "baseline_pairwise_accuracy": 0.8},
    )
    comparison = _compare_rankers(candidate, None)
    assert comparison is not None
    assert comparison["regression"] == 1.0


def test_the_gate_accepts_a_ranker_that_beats_the_rule(platform):
    from ragllmops.orchestration.pipeline import _compare_rankers

    candidate = platform.build_bundle()
    from dataclasses import replace

    candidate = replace(
        candidate,
        ranker_metrics={"pairwise_accuracy": 0.9, "baseline_pairwise_accuracy": 0.6},
    )
    comparison = _compare_rankers(candidate, None)
    assert comparison is not None
    assert "regression" not in comparison


# ----------------------------------------------------------------------
# the API
# ----------------------------------------------------------------------
@pytest.fixture
def client(platform):
    bundle = platform.current_bundle()
    platform.register_bundle(bundle)
    platform.promote(platform.settings.champion_alias, bundle.version)
    with TestClient(create_app(platform)) as test_client:
        yield test_client
