"""Provenance: a bundle records what actually ran, not what was requested.

The registry used to store the *configured* model names, so a bundle built
with the hashing embedder claimed to have been evaluated with
``all-MiniLM-L6-v2``. A promotion with the wrong provenance cannot be
reproduced, which defeats the point of versioning.
"""

from ragllmops.artifacts import builtin_artifact_hash, combined_artifact_hash
from ragllmops.embed.hashing import HashingEmbedder
from ragllmops.models import RAGConfigBundle
from ragllmops.rag.bundle import BundleBuilder
from ragllmops.retrieval.rerank import TermOverlapReranker


def test_bundle_records_the_resolved_embedder_not_the_requested_one(settings):
    """The hashing embedder must not be recorded as all-MiniLM-L6-v2."""
    builder = BundleBuilder(settings)
    bundle = builder.build(embedder=HashingEmbedder(dim=settings.embedding_dim))

    assert bundle.embedder_id == "hashing-v1"
    assert "all-MiniLM-L6-v2" not in bundle.embedder_id
    assert bundle.embedder_requested == settings.embedding_model


def test_bundle_records_the_resolved_reranker(settings):
    bundle = BundleBuilder(settings).build(reranker=TermOverlapReranker())

    assert bundle.reranker_id == "term-overlap-v1"
    assert "ms-marco" not in bundle.reranker_id
    assert bundle.reranker_requested == settings.rerank_model


def test_bundle_records_the_embedding_dimension(settings):
    bundle = BundleBuilder(settings).build(embedder=HashingEmbedder(dim=256))
    assert bundle.embedding_dim == 256


def test_bundle_pins_the_model_artifact(settings):
    bundle = BundleBuilder(settings).build(
        embedder=HashingEmbedder(dim=settings.embedding_dim),
        reranker=TermOverlapReranker(),
    )
    assert bundle.model_artifact_hash
    assert bundle.model_artifact_hash == combined_artifact_hash(
        builtin_artifact_hash("hashing-v1", dim=settings.embedding_dim),
        builtin_artifact_hash("term-overlap-v1"),
    )


def test_artifact_hash_changes_with_the_parameters():
    assert builtin_artifact_hash("hashing-v1", dim=128) != builtin_artifact_hash(
        "hashing-v1", dim=384
    )


def test_platform_bundle_matches_the_running_backends(platform):
    bundle = platform.build_bundle()
    assert bundle.embedder_id == platform.embedder.name
    assert bundle.embedding_dim == platform.embedder.dim
    assert bundle.reranker_id == platform.reranker.name


def test_health_reports_resolved_and_requested_backends(platform, settings):
    backends = platform.health()["backends"]
    assert backends["embedder"]["id"] == "hashing-v1"
    assert backends["embedder"]["requested"] == settings.embedding_model
    assert backends["embedder"]["requested_backend"] == settings.embedder_backend
    assert backends["reranker"]["id"] == "term-overlap-v1"


def test_a_different_embedder_produces_a_different_fingerprint(settings):
    """The fingerprint must separate two bundles that behave differently."""
    hashing = BundleBuilder(settings).build(embedder=HashingEmbedder(dim=128))
    other = BundleBuilder(settings).build(embedder=HashingEmbedder(dim=384))
    assert hashing.fingerprint() != other.fingerprint()


def test_legacy_bundles_keep_their_requested_names():
    """Bundles registered before the fix only had the configured names."""
    legacy = RAGConfigBundle.from_dict(
        {
            "version": "vlegacy",
            "prompt_id": "v2",
            "prompt_template": "p",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "llm_provider": "extractive",
            "llm_model": "local-extractive-v1",
        }
    )
    assert legacy.embedder_requested == "sentence-transformers/all-MiniLM-L6-v2"
    assert legacy.reranker_requested == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert legacy.embedder_id == ""
