"""The registry contract."""

import pytest

from ragllmops.models import RAGConfigBundle
from ragllmops.registry.local_registry import LocalBundleRegistry


def _bundle(version: str = "v1", prompt_id: str = "v2") -> RAGConfigBundle:
    return RAGConfigBundle(
        version=version,
        prompt_id=prompt_id,
        prompt_template="Question: {question}\nContext: {context}",
        embedder_id="hashing-v1",
        embedding_dim=128,
        embedder_requested="sentence-transformers/all-MiniLM-L6-v2",
        reranker_id="term-overlap-v1",
        reranker_requested="cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_provider="extractive",
        llm_model="local-extractive-v1",
    )


def test_register_and_get(tmp_path):
    registry = LocalBundleRegistry(tmp_path / "registry.json")
    registry.register(_bundle())
    assert registry.get("v1") is not None
    assert registry.get("v1").prompt_id == "v2"
    assert registry.list_versions() == ["v1"]


def test_persistence_across_instances(tmp_path):
    path = tmp_path / "registry.json"
    LocalBundleRegistry(path).register(_bundle())
    reloaded = LocalBundleRegistry(path)
    assert reloaded.get("v1") is not None
    assert reloaded.list_versions() == ["v1"]


def test_aliases_resolve(tmp_path):
    registry = LocalBundleRegistry(tmp_path / "registry.json")
    registry.register(_bundle())
    registry.set_alias("champion", "v1")
    assert registry.get_by_alias("champion").version == "v1"
    assert registry.aliases() == {"champion": "v1"}


def test_alias_to_unknown_version_is_rejected(tmp_path):
    registry = LocalBundleRegistry(tmp_path / "registry.json")
    with pytest.raises(KeyError):
        registry.set_alias("champion", "nope")


def test_unknown_alias_returns_none(tmp_path):
    registry = LocalBundleRegistry(tmp_path / "registry.json")
    assert registry.get_by_alias("champion") is None


def test_latest_returns_the_newest(tmp_path):
    registry = LocalBundleRegistry(tmp_path / "registry.json")
    registry.register(_bundle("v1"))
    registry.register(_bundle("v2"))
    assert registry.latest().version == "v2"
    assert registry.list_versions() == ["v2", "v1"]


def test_bundle_fingerprint_is_content_addressed():
    assert _bundle().fingerprint() == _bundle().fingerprint()
    assert _bundle().fingerprint() != _bundle(prompt_id="v1").fingerprint()


def test_bundle_round_trips_through_dict():
    bundle = _bundle()
    assert RAGConfigBundle.from_dict(bundle.to_dict()) == bundle
