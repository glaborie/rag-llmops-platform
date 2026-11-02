"""The feature store: materialisation and offline/online parity."""

from datetime import UTC, datetime

from ragllmops.features.materialize import compute_doc_features, write_features
from ragllmops.models import RequestRecord


def _record(index: int, chunk_ids, score: float, refused: bool = False) -> RequestRecord:
    return RequestRecord(
        request_id=f"r{index}",
        ts=datetime(2026, 1, 1, tzinfo=UTC),
        query="q",
        answer="a",
        bundle_version="v1",
        prompt_id="v2",
        llm_model="m",
        llm_provider="extractive",
        latency_ms=10.0,
        top_score=score,
        chunk_count=len(chunk_ids),
        prompt_tokens=10,
        completion_tokens=20,
        answer_chars=30,
        refused=refused,
        retrieved_ids=list(chunk_ids),
    )


def test_features_are_computed_per_document():
    records = [
        _record(0, ["compute::0"], 0.9),
        _record(1, ["compute::1"], 0.7),
        _record(2, ["storage::0"], 0.5),
    ]
    features = compute_doc_features(records, ["compute", "storage", "billing"])

    assert features["compute"]["retrieval_count"] == 2.0
    assert abs(features["compute"]["avg_relevance"] - 0.8) < 1e-6
    assert features["storage"]["retrieval_count"] == 1.0
    assert features["billing"]["retrieval_count"] == 0.0


def test_refusal_rate_is_tracked():
    records = [
        _record(0, ["compute::0"], 0.9, refused=True),
        _record(1, ["compute::1"], 0.9, refused=False),
    ]
    features = compute_doc_features(records)
    assert features["compute"]["refusal_rate"] == 0.5


def test_doc_id_is_derived_from_the_chunk_id():
    from ragllmops.features.base import doc_id_from_chunk

    assert doc_id_from_chunk("compute::3") == "compute"


def test_offline_and_online_stores_agree(tmp_path):
    from ragllmops.features.local_provider import LocalFeatureProvider

    records = [_record(0, ["compute::0"], 0.9), _record(1, ["compute::1"], 0.5)]
    features = compute_doc_features(records)

    offline_path = tmp_path / "offline.json"
    online_path = tmp_path / "online.json"
    write_features(features, offline_path, online_path)

    provider = LocalFeatureProvider(online_path)
    online = provider.get_online_features({"compute"})

    assert online["compute"] == features["compute"]


def test_local_provider_reloads_from_disk(tmp_path):
    from ragllmops.features.local_provider import LocalFeatureProvider

    path = tmp_path / "online.json"
    write_features({"compute": {"retrieval_count": 3.0}}, tmp_path / "off.json", path)

    provider = LocalFeatureProvider(path)
    assert provider.get_online_features({"compute"})["compute"]["retrieval_count"] == 3.0
    assert provider.count() == 1


def test_platform_feature_materialisation_and_parity(platform):
    platform.answer("What CPU target does the autoscaler use?")
    features = platform.materialize_features()

    from ragllmops.features import offline_features_path

    offline = offline_features_path(platform.settings)
    assert offline.exists()

    provider_features = platform.feature_provider.get_online_features(set(features))
    assert provider_features == features
